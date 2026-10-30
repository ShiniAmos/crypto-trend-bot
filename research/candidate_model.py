"""Track C: a model trained on the bot's own trades.

The earlier model predicted next-day excess return for every coin and was rejected. This one asks a
narrower and more useful question:

    given a candidate the shipped rules already like, will this trade make money?

Label: the net return of the trade the bot would actually have taken - entered at the next bar's
open, exited when the live exit rule fires (regime off, 20-bar low, or rank worse than 10), with
7 bps slippage and 0.10% fee on both legs. That is the real thing being predicted, not a proxy.

Walk-forward: 730 days train, 91 days test, rolling. The purge gap is 30 days, comfortably longer
than the average holding period (~7 days), so no training label can overlap its test window.

Used two ways, both measured by the ship rule:
    veto  - skip a candidate whose win probability is below a threshold
    rank  - among candidates, prefer the highest probability instead of the highest score

    python research/candidate_model.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import backtest, config, features, signals  # noqa: E402
from app.flags import Flags  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SHIPPED = Flags(abs_momentum=True, volume_min=0.8, taker_min=0.5)
TRAIN_DAYS, TEST_DAYS, PURGE_DAYS = 730, 91, 30
FEATS = ["rank", "score", "rs_7d", "rs_30d", "ret_7d", "ret_30d", "dist_sma50", "dist_sma200",
         "vol_30d", "vol_7d", "atr_pct", "volume_ratio", "taker_ratio", "don_pos",
         "btc_dist_sma200", "btc_ret_7d", "btc_ret_30d", "n_eligible"]
DERIVED = ["breadth", "rs_accel", "stop_distance_pct", "vol_vs_btc"]


def label_trades(panel, s):
    """For every candidate bar, the net return of the trade it would have become."""
    fee, slip = s.fee_rate, s.slippage
    rows = []
    for sym, g in panel.groupby("symbol", sort=False):
        g = g.sort_values("bar_time").reset_index(drop=True)
        o, c = g["open"].to_numpy(), g["close"].to_numpy()
        exit_cond = ((~g["regime_on"].fillna(False).to_numpy())
                     | g["breakdown"].fillna(False).to_numpy()
                     | (g["rank"].fillna(999).to_numpy() > 10))
        exits = np.flatnonzero(exit_cond)
        cand = np.flatnonzero(g["_cand"].to_numpy())
        for i in cand:
            if i + 1 >= len(g):
                continue
            entry = o[i + 1] * (1 + slip)
            nxt = exits[np.searchsorted(exits, i + 1, side="left"):]
            j = nxt[0] if len(nxt) else len(g) - 2
            if j + 1 >= len(g):
                continue
            exit_px = o[j + 1] * (1 - slip)
            net = (exit_px * (1 - fee)) / (entry * (1 + fee)) - 1
            rows.append({"bar_time": g["bar_time"].iloc[i], "symbol": sym, "net": net,
                         "bars_held": j - i, "win": net > 0})
    return pd.DataFrame(rows)


def add_derived(panel):
    el = panel["eligible"] & (panel["symbol"] != "BTCUSDT")
    br = (panel[el].assign(p=lambda d: d["rs_30d"] > 0)
          .groupby("bar_time")["p"].mean().rename("breadth"))
    panel = panel.join(br, on="bar_time")
    panel["rs_accel"] = panel["rs_7d"] - panel["rs_30d"]
    btcvol = (panel[panel.symbol == "BTCUSDT"].set_index("bar_time")["vol_30d"].rename("btcvol"))
    panel = panel.join(btcvol, on="bar_time")
    panel["vol_vs_btc"] = panel["vol_30d"] / panel["btcvol"]
    panel["stop_distance_pct"] = 1 - panel["stop_level"] / panel["close"]
    return panel


def main():
    s = config.load()
    u = json.loads((ROOT / "reports" / "_parity_universe.json").read_text())
    panel = add_derived(features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars))

    # The candidate population: what the shipped rules propose, before slot limits.
    rank = panel["rank"].to_numpy()
    btc = (panel["symbol"] == "BTCUSDT").to_numpy()
    cand = ((panel["eligible"] & panel["regime_on"] & panel["breakout"]).to_numpy()
            & ((panel["rs_7d"] > 0).to_numpy() | btc) & (rank <= s.candidates)
            & signals.extra_filters(panel, SHIPPED).to_numpy())
    panel["_cand"] = cand
    print(f"candidate bars: {int(cand.sum())}")

    lab = label_trades(panel, s)
    print(f"labelled trades: {len(lab)}  win rate {lab['win'].mean():.1%}  "
          f"mean net {lab['net'].mean():+.2%}  median {lab['net'].median():+.2%}")

    d = panel[panel["_cand"]].merge(lab, on=["bar_time", "symbol"], how="inner")
    X_all = d[FEATS + DERIVED].replace([np.inf, -np.inf], np.nan)
    y_all = d["win"].astype(int)
    t = d["bar_time"]

    d["p_win"] = np.nan
    d["e_ret"] = np.nan
    # Win probability is the wrong objective for a trend system - the trades most likely to win are
    # the small ones. So a second model predicts the trade's RETURN, clipped so a single squeeze
    # cannot dominate the fit.
    y_ret = d["net"].clip(-0.5, 1.0)
    first = t.min() + pd.Timedelta(days=TRAIN_DAYS)
    test_start, folds = first, 0
    while test_start < t.max():
        test_end = test_start + pd.Timedelta(days=TEST_DAYS)
        tr = (t >= test_start - pd.Timedelta(days=TRAIN_DAYS)) & \
             (t < test_start - pd.Timedelta(days=PURGE_DAYS))
        te = (t >= test_start) & (t < test_end)
        if tr.sum() >= 150 and te.any() and y_all[tr].nunique() > 1:
            model = HistGradientBoostingClassifier(
                max_iter=250, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=25,
                l2_regularization=1.0, random_state=0)
            model.fit(X_all[tr], y_all[tr])
            d.loc[te, "p_win"] = model.predict_proba(X_all[te])[:, 1]
            reg = HistGradientBoostingRegressor(
                max_iter=250, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=25,
                l2_regularization=1.0, random_state=0)
            reg.fit(X_all[tr], y_ret[tr])
            d.loc[te, "e_ret"] = reg.predict(X_all[te])
            folds += 1
        test_start = test_end
    oos = d["p_win"].notna()
    print(f"folds trained: {folds}; scored candidates: {int(oos.sum())}")

    o = d[oos]
    print("\nDoes the probability separate winners from losers, out of sample?")
    q = pd.qcut(o["p_win"], 4, labels=["lowest", "low", "high", "highest"], duplicates="drop")
    print(o.groupby(q, observed=True).agg(trades=("net", "size"), win_rate=("win", "mean"),
                                          mean_net=("net", "mean")).to_string(
        float_format=lambda x: f"{x:.4f}"))
    ic = o["p_win"].corr(o["net"], method="spearman")
    print(f"rank correlation, win probability vs realised return: {ic:+.4f}")

    print("\nSecond model - predicted RETURN rather than win probability:")
    q2 = pd.qcut(o["e_ret"], 4, labels=["lowest", "low", "high", "highest"], duplicates="drop")
    print(o.groupby(q2, observed=True).agg(trades=("net", "size"), win_rate=("win", "mean"),
                                           mean_net=("net", "mean")).to_string(
        float_format=lambda x: f"{x:.4f}"))
    ic2 = o["e_ret"].corr(o["net"], method="spearman")
    print(f"rank correlation, predicted return vs realised return: {ic2:+.4f}")

    merged = panel.merge(o[["bar_time", "symbol", "p_win", "e_ret"]],
                         on=["bar_time", "symbol"], how="left")
    start = o["bar_time"].min()
    results = []
    base = backtest.run(merged, s, flags=SHIPPED, start=start)
    mb = backtest.metrics(base.equity, base.trades, base.exposure)
    results.append(("baseline", mb))
    for thr in (0.40, 0.45, 0.50):
        m2 = merged.copy()
        # veto: a candidate below the threshold simply is not a candidate
        m2["ml_score"] = m2["p_win"] - thr
        r = backtest.run(m2, s, flags=Flags(abs_momentum=True, volume_min=0.8, taker_min=0.5,
                                            ml_veto=True), start=start)
        results.append((f"veto p>{thr:g}", backtest.metrics(r.equity, r.trades, r.exposure)))
    for thr in (0.0, 0.02):
        m4 = merged.copy()
        m4["ml_score"] = m4["e_ret"] - thr
        r = backtest.run(m4, s, flags=Flags(abs_momentum=True, volume_min=0.8, taker_min=0.5,
                                            ml_veto=True), start=start)
        results.append((f"veto e_ret>{thr:g}", backtest.metrics(r.equity, r.trades, r.exposure)))
    m5 = merged.copy()
    m5["e_rank_score"] = m5["e_ret"].fillna(-9) * 100
    r = backtest.run(m5, s, flags=SHIPPED, score_col="e_rank_score", start=start)
    results.append(("rank by e_ret", backtest.metrics(r.equity, r.trades, r.exposure)))

    # rank by probability instead of score, among the same candidates
    m3 = merged.copy()
    m3["p_rank_score"] = m3["p_win"].fillna(-1) * 100
    r = backtest.run(m3, s, flags=SHIPPED, score_col="p_rank_score", start=start)
    results.append(("rank by p_win", backtest.metrics(r.equity, r.trades, r.exposure)))

    # The one formulation that deletes no trades: keep every candidate and let the model decide
    # how large each one is. A trend system lives on a few huge winners, so any filter risks
    # removing them; sizing cannot.
    m6 = merged.copy()
    pct = m6.groupby("bar_time")["e_ret"].rank(pct=True)
    m6["ml_size"] = (0.5 + pct).clip(0.5, 1.5).fillna(1.0)
    r = backtest.run(m6, s, flags=SHIPPED, start=start, size_scale_col="ml_size")
    results.append(("size by e_ret", backtest.metrics(r.equity, r.trades, r.exposure)))

    print(f"\nBacktests from {start.date()} (shipped filters on every row):")
    out = pd.DataFrame([{"run": k, "cagr": v["cagr"], "sharpe": v["sharpe"],
                         "maxdd": v["max_drawdown"], "trades": v.get("trades", 0),
                         "win": v.get("win_rate")} for k, v in results])
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    (ROOT / "reports" / "candidate_model.json").write_text(
        json.dumps({"ic_pwin": float(ic), "ic_eret": float(ic2), "folds": folds,
                    "runs": out.to_dict("records")},
                   indent=2, default=str))


if __name__ == "__main__":
    main()
