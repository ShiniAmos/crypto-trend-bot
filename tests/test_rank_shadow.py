"""The shadow rank scorer must compute exactly what the model was trained on, and touch nothing.

The scorer recomputes the model's 57 inputs live, on a 1 GB server, from a 260-bar tail of candles
instead of nine years. If those inputs differ from the training pipeline's in any column, the
forward record measures a different model than the one that was validated, silently - the classic
train/serve skew, and the same shape as every serious defect this project has had (one rule, two
implementations). So the central tests here run BOTH pipelines on the same real Binance candles and
require the same numbers.

Tests that need the local survivorship-free cache (data/pit/klines) or the exported model
(models/rank_model.json) skip where those are absent, e.g. on the server.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import data, rankfeatures as rf, rankmodel, rankuniverse as ru  # noqa: E402

PIT = ROOT / "data" / "pit" / "klines"
MODEL = ROOT / "models" / "rank_model.json"
needs_pit = pytest.mark.skipif(not (PIT / "BTCUSDT_4h.parquet").exists(),
                               reason="needs the local survivorship-free kline cache")
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="needs models/rank_model.json")


# ------------------------------------------------------------------------------- the universe
def _daily(n_days=120, symbols=("AAAUSDT", "BBBUSDT"), seed=0, start="2026-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_days, freq="D", tz="UTC")
    vol = pd.DataFrame({s: rng.uniform(1e6, 2e6, n_days) for s in symbols}, index=idx)
    close = pd.DataFrame({s: 10 * np.exp(np.cumsum(rng.normal(0, 0.03, n_days))) for s in symbols},
                         index=idx)
    return vol, close


def test_a_stock_is_a_b_name_listed_after_the_launch():
    listed = {"NVDABUSDT": pd.Timestamp("2026-06-11", tz="UTC"),
              "BNBUSDT": pd.Timestamp("2017-11-06", tz="UTC"),       # a coin, ends in B
              "ARBUSDT": pd.Timestamp("2023-03-23", tz="UTC"),       # a coin, ends in B
              "NEWBUSDT": None,                                      # no date: never a stock
              "ETHUSDT": pd.Timestamp("2026-07-01", tz="UTC")}       # late listing, not a B name
    assert ru.tokenized_stocks(listed, listed) == {"NVDABUSDT"}


def test_stocks_and_pegged_tokens_are_removed_before_ranking_not_after():
    """The universe must stay N wide: an excluded heavy-volume name is replaced, not left empty."""
    syms = ["AAAUSDT", "BBBUSDT", "CCCUSDT", "STKBUSDT", "PEGUSDT"]
    vol, close = _daily(n_days=90, symbols=syms)
    vol["STKBUSDT"] *= 50                    # the biggest volume of all
    vol["PEGUSDT"] *= 40
    close["PEGUSDT"] = 1.0 + np.random.default_rng(1).normal(0, 1e-4, len(close))
    listed = {s: vol.index[0] for s in syms}
    listed["STKBUSDT"] = pd.Timestamp("2026-06-11", tz="UTC")
    vol.index = pd.date_range("2026-06-12", periods=len(vol), freq="D", tz="UTC")
    close.index = vol.index
    m = ru.membership(vol, close, 3, listed)
    last = m.iloc[-1]
    assert not last["STKBUSDT"] and not last["PEGUSDT"]
    assert int(last.sum()) == 3, "an excluded name left a hole instead of being replaced"


def test_a_coin_is_only_excluded_while_it_is_pegged():
    """UST was a stablecoin until May 2022 and a collapsing coin after. Judge each day."""
    vol, close = _daily(n_days=120, symbols=["USTUSDT", "AAAUSDT", "BBBUSDT"])
    close.iloc[:60, 0] = 1.0                 # flat for 60 days, then it moves
    peg = ru.pegged(close)
    assert peg["USTUSDT"].iloc[40:60].all(), "a dollar-flat token was not recognised as pegged"
    assert not peg["USTUSDT"].iloc[95:].any(), "it stayed excluded after it stopped being pegged"


def test_today_has_a_membership_row_and_history_does_not_change():
    vol, close = _daily(n_days=80, symbols=["AAAUSDT", "BBBUSDT", "CCCUSDT"])
    listed = {s: vol.index[0] for s in vol.columns}
    base = ru.membership(vol, close, 2, listed)
    today = vol.index[-1] + pd.Timedelta(days=1)
    ext = ru.membership(vol, close, 2, listed, extend_to=today)
    assert today in ext.index, "today's 4h bars would find no membership"
    pd.testing.assert_frame_equal(ext.loc[base.index], base)


# ----------------------------------------------------------------------------- the model file
def test_the_exported_model_reproduces_scikit_learn_on_unseen_rows(tmp_path):
    from sklearn.ensemble import HistGradientBoostingRegressor
    rng = np.random.default_rng(2)
    X = rng.normal(size=(6000, 12))
    X[rng.random(X.shape) < 0.2] = np.nan
    y = np.nan_to_num(X[:, 0]) * 0.02 - np.nan_to_num(X[:, 1]) * 0.01 + rng.normal(0, 0.02, 6000)
    m = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.06, max_iter=60,
                                      early_stopping=False, random_state=0).fit(X, y)
    path = tmp_path / "m.json"
    rankmodel.export(m, [f"f{i}" for i in range(12)], X[:2000], path)
    spec = rankmodel.load(path)
    Xn = rng.normal(size=(1500, 12))
    Xn[rng.random(Xn.shape) < 0.4] = np.nan
    assert np.max(np.abs(rankmodel.predict(spec, Xn) - m.predict(Xn))) <= 1e-12


def test_a_column_order_mismatch_is_refused():
    p = pd.DataFrame({"bar_time": [pd.Timestamp("2026-01-01", tz="UTC")]})
    with pytest.raises(Exception):
        rf.design(p, ["not", "the", "model's", "columns"])


# ------------------------------------------------------------------------- tail = full history
def _write_tail(src, dst, symbols, T):
    """What the live store would hold at bar T: closed bars only, the last TAIL rows of each."""
    T_ms = int(T.value // 10**6)
    for s in symbols:
        b4 = data.load(src, s, "4h")
        b4 = b4[b4["open_time"] + data.TF_MS["4h"] <= T_ms].tail(rf.TAIL_4H)
        b1 = data.load(src, s, "1d")
        b1 = b1[b1["open_time"] + data.TF_MS["1d"] <= T_ms].tail(rf.TAIL_1D)
        data.save(dst, s, "4h", b4)
        data.save(dst, s, "1d", b1)


def _rows_at(full, symbols, T):
    member = pd.DataFrame(True, columns=sorted(set(symbols) | {"BTCUSDT"}),
                          index=pd.date_range(pd.Timestamp("2017-01-01", tz="UTC"),
                                              T.normalize(), freq="D"))
    p = rf.attach_members(full, member)
    p = rf.add_market_state(p[p["bar_time"] == T].copy())
    return p.sort_values("symbol").reset_index(drop=True)


COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "ADAUSDT", "LINKUSDT",
         "AVAXUSDT", "LTCUSDT", "NEARUSDT"]


@needs_pit
def test_features_from_a_tail_equal_features_from_nine_years(tmp_path):
    """THE train/serve test. Every one of the model's inputs, both ways, on real candles."""
    syms = [s for s in COINS if (PIT / f"{s}_4h.parquet").exists()]
    btc = data.load(PIT, "BTCUSDT", "4h")
    T = pd.Timestamp(int(btc["open_time"].iloc[-7]) + data.TF_MS["4h"], unit="ms", tz="UTC")

    a = _rows_at(rf.build(PIT, syms, 20), syms, T)
    _write_tail(PIT, tmp_path, syms, T)
    b = _rows_at(rf.build(tmp_path, syms, 20), syms, T)

    assert list(a["symbol"]) == list(b["symbol"]) and len(a) == len(syms)
    names = rf.feature_names(a)
    Xa, Xb = rf.design(a, names), rf.design(b, names)
    bad = [names[j] for j in range(len(names))
           if not np.allclose(Xa[:, j], Xb[:, j], rtol=0, atol=1e-9, equal_nan=True)]
    assert not bad, f"these inputs differ between the tail and the full history: {bad}"


@needs_pit
def test_a_fresh_relisting_is_excluded_from_a_tail_too(tmp_path):
    """LUNAUSDT resumed on 31 May 2022 after an 18-day stop. Ten days later it must be skipped
    by both pipelines - the tail has to see the gap the full history sees."""
    syms = ["BTCUSDT", "ETHUSDT", "LUNAUSDT"]
    if not (PIT / "LUNAUSDT_4h.parquet").exists():
        pytest.skip("LUNA not in the cache")
    T = pd.Timestamp("2022-06-10 12:00", tz="UTC")
    a = _rows_at(rf.build(PIT, syms, 20), syms, T)
    _write_tail(PIT, tmp_path, syms, T)
    b = _rows_at(rf.build(tmp_path, syms, 20), syms, T)
    assert "LUNAUSDT" not in set(a["symbol"]), "the full history scored a fresh relisting"
    assert "LUNAUSDT" not in set(b["symbol"]), "the tail missed the gap and scored a fresh relisting"


# --------------------------------------------------------------------------------- isolation
def test_the_trading_code_does_not_import_the_scorer():
    """The shadow scorer runs beside the bot, never inside it."""
    for f in ("app/main.py", "app/fleet.py", "app/broker.py", "app/portfolio.py", "app/signals.py",
              "app/risk.py", "app/backtest.py", "app/reconcile.py"):
        src = (ROOT / f).read_text(encoding="utf-8")
        for mod in ("rankscore", "rankmodel", "rankuniverse", "rankfeatures"):
            assert mod not in src, f"{f} refers to {mod}: the shadow scorer has reached trading code"


def test_the_scorer_writes_only_under_its_own_directory():
    src = (ROOT / "app" / "rankscore.py").read_text(encoding="utf-8")
    assert "state_" not in src and "db_path" not in src, "the scorer refers to a trading book"
    assert '/ "rank"' in src


# ------------------------------------------------------------------------------- end to end
@needs_pit
@needs_model
def test_a_live_run_scores_what_the_training_pipeline_would(tmp_path, monkeypatch):
    """score_once against frozen candles, with every network call served from the cache, must
    record the same members, predictions and percentiles as the training pipeline computes from
    full history for the same bars."""
    from app import config, pit, rankscore, universe

    btc = data.load(PIT, "BTCUSDT", "4h")
    T = pd.Timestamp(int(btc["open_time"].iloc[-25]) + data.TF_MS["4h"], unit="ms", tz="UTC")
    now = T + pd.Timedelta(minutes=7)
    now_ms = int(now.value // 10**6)

    pool = [s for s in sorted({f.name.rsplit("_", 1)[0] for f in PIT.glob("*_1d.parquet")})
            if pit.keep(s, True)]
    # a manageable but realistic universe: the 60 largest by volume over the month before T
    vols = {}
    for s in pool:
        d = data.load(PIT, s, "1d")
        d = d[(d["open_time"] + data.TF_MS["1d"] <= now_ms)].tail(30)
        if len(d) >= 30:
            vols[s] = float(d["quote_volume"].sum())
    symbols = sorted(sorted(vols, key=vols.get, reverse=True)[:60] + ["BTCUSDT"])
    symbols = sorted(set(symbols))

    def fake_fetch(symbol, tf, start_ms=0, end_ms=None, limit=1000):
        d = data.load(PIT, symbol, tf)
        d = d[(d["open_time"] >= start_ms) & (d["close_time"] < now_ms)]
        return d.head(limit).reset_index(drop=True)

    monkeypatch.setattr(data, "fetch_klines", fake_fetch)
    monkeypatch.setattr(data, "first_candle_ms",
                        lambda s, tf="1d": int(data.load(PIT, s, "1d")["open_time"].iloc[0]))
    monkeypatch.setattr(universe, "binance_usdt_pairs",
                        lambda: {s[:-4]: {"symbol": s} for s in symbols})
    s = config.load()
    monkeypatch.setattr(s, "data_dir", tmp_path)
    out = rankscore.score_once(s, log=lambda *a: None, now=now.timestamp())
    assert out["status"] == "ok", out
    con = rankscore.connect(tmp_path / "rank" / "shadow.sqlite")
    live = pd.read_sql("SELECT bar_time, symbol, pred, pct FROM scores", con)
    live["bar_time"] = pd.to_datetime(live["bar_time"], utc=True)

    # the training pipeline, full history, same symbols, same bars
    spec = rankmodel.load(MODEL)
    vol, close, listed = ru.daily_frames(PIT, symbols, data.load)
    member = ru.membership(vol, close, 30, listed, extend_to=T)
    full = rf.build(PIT, sorted(set(member.columns[member.any()]) | {"BTCUSDT"}), s.donchian_bars)
    p = rf.attach_members(full, member)
    p = rf.add_market_state(p[p["bar_time"].isin(live["bar_time"].unique())].copy())
    p["pred"] = rankmodel.predict(spec, rf.design(p, spec["features"]))
    p["pct"] = p.groupby("bar_time")["pred"].rank(pct=True)
    want = p[["bar_time", "symbol", "pred", "pct"]]

    j = want.merge(live, on=["bar_time", "symbol"], how="outer", suffixes=("_train", "_live"),
                   indicator=True)
    assert (j["_merge"] == "both").all(), (
        "the live run scored a different set of coins than training would:\n"
        + j[j["_merge"] != "both"][["bar_time", "symbol", "_merge"]].to_string())
    assert np.allclose(j["pred_train"], j["pred_live"], rtol=0, atol=1e-10)
    assert np.allclose(j["pct_train"], j["pct_live"], rtol=0, atol=1e-12)
    assert out["bars"] == rankscore.MAX_BACKFILL, "a first run should score the last two days"


@needs_pit
def test_btc_self_beta_and_correlation_are_exactly_one_however_much_history(tmp_path):
    """The defect the end-to-end test found: float noise in BTC's correlation with itself depended
    on history length, and the trained model had split thresholds inside that noise."""
    syms = ["BTCUSDT", "ETHUSDT"]
    btc = data.load(PIT, "BTCUSDT", "4h")
    T = pd.Timestamp(int(btc["open_time"].iloc[-3]) + data.TF_MS["4h"], unit="ms", tz="UTC")
    full = rf.build(PIT, syms, 20)
    _write_tail(PIT, tmp_path, syms, T)
    tail = rf.build(tmp_path, syms, 20)
    for f in (full, tail):
        b = f[(f["symbol"] == "BTCUSDT") & (f["bar_time"] == T)]
        assert float(b["beta_30d"].iloc[0]) == 1.0 and float(b["corr_30d"].iloc[0]) == 1.0


# ------------------------------------------------------------------------------------ report
@needs_pit
def test_the_report_measures_outcomes_with_the_models_own_label(tmp_path, monkeypatch):
    """The forward record is only evidence about this model if the outcome the report scores is
    the label the model was trained on: the 24h return entered at the next bar's open."""
    from app import rankreport
    syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    btc = data.load(PIT, "BTCUSDT", "4h")
    last = pd.Timestamp(int(btc["open_time"].iloc[-10]) + data.TF_MS["4h"], unit="ms", tz="UTC")
    first = last - pd.Timedelta(days=3)
    monkeypatch.setattr(data, "fetch_klines", lambda sym, tf, start_ms=0, end_ms=None, limit=1000:
                        data.load(PIT, sym, tf).pipe(lambda d: d[d["open_time"] >= start_ms])
                        .head(limit).reset_index(drop=True))
    got = rankreport._outcomes(tmp_path, syms, first, last).set_index(["bar_time", "symbol"])["fwd"]
    full = rf.build(PIT, syms, 20)
    full["fwd"] = rf.forward_returns(full, rf.HORIZON)
    want = full.set_index(["bar_time", "symbol"])["fwd"].reindex(got.index)
    assert len(got) > 30
    assert np.allclose(got.to_numpy(), want.to_numpy(), rtol=0, atol=1e-12, equal_nan=True), (
        "the report scores a different outcome than the model was trained to predict")


def test_the_report_opens_trading_books_read_only():
    src = (ROOT / "app" / "rankreport.py").read_text(encoding="utf-8")
    for line in src.splitlines():
        if "sqlite3.connect" in line:
            assert "mode=ro" in line, f"a database is opened writable: {line.strip()}"


def test_days_to_verdict_is_the_standard_error_arithmetic():
    """The report's 'a verdict needs about N days' line. At N days the expected mean sits two
    standard errors from zero, and one day fewer it does not."""
    from app import rankreport
    assert rankreport.days_to_verdict(0.0759, 0.2615) == 48          # rank correlation
    assert rankreport.days_to_verdict(-0.00383, 0.02478) == 168      # bottom third minus rest
    n = rankreport.days_to_verdict(0.01, 0.05)
    assert 0.01 / (0.05 / n ** 0.5) >= 2.0 > 0.01 / (0.05 / (n - 1) ** 0.5)


def test_the_progress_line_refuses_to_judge_two_days():
    from app import rankreport
    line = rankreport.progress_line("x", pd.Series([0.1, 0.2]), 0.0759, 0.2615)
    assert "2 independent days" in line and "z" not in line.split("so far")[0]
