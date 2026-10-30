"""SEQUENCE MODEL ON RAW BARS: can a window of recent candles predict the TREND STATE ahead?

WHY THIS EXISTS. The owner does not want a reactive breakout bot. The shipped bot waits for a close
above its 20-bar high and then reacts. This file asks whether a model can be given the raw recent
history and PREDICT the trend instead, so the entry comes from a forward-looking output rather than
from a trigger that has already fired.

It is built on a distinction nothing in this repo has tested. Everything that failed here asked
about RETURNS: next-7d direction (AUC 0.4688-0.5219 against a 0.50 coin flip), a right-tail
classifier (0.4914), an ML ranker, an ML veto, win-probability and return-regression models (each
cut CAGR by a third). Returns are a martingale here; that is settled. But TREND STATE is not a
return - it is a slow, autocorrelated state variable, and the one state question anybody did ask
("will BTC still be above its 200-day average in 7 days") scored AUC 0.9779. So the question here
is not "is the trend state predictable" - it obviously is - but:

    does a 60-bar raw-candle window predict the trend state BETTER THAN SIMPLY ASSUMING IT PERSISTS,
    and if it does, is that margin worth any money?

THE HONEST TRAP THIS FILE IS DESIGNED AROUND. A 7-day-ahead trend label is ~90% the same as today's
trend label. Any model fed trend-ish inputs will post a huge AUC. Reporting that number alone would
be a lie by omission. PERSISTENCE - predict "still in an uptrend" iff it is in one now, zero
parameters - is therefore the only baseline that matters, and the deliverable is the MARGIN over it.

TARGET
    uptrend(t) = close(t) > sma50_d(t)  AND  sma50_d(t) > sma50_d(t - 10 days)
    y(t)       = uptrend(t + 7 days)
  i.e. "will this coin be above a RISING 50-day average a week from now". The rising test is what
  makes it a trend label rather than a price-above-a-line label.

INPUT (the sequence)
    the last 60 four-hour bars (10 days) of 4 channels, for the coin AND for BTC, flattened:
        log return, bar volume / its own trailing 20-day mean, taker-buy share, (high-low)/close
    = 60 x 4 x 2 = 480 columns, every one of them from a bar closed at or before t.

MODELS, in the order the brief demands
    naive_persistence  today's trend state, carried forward.                  zero parameters
    naive_dist_sma50   an unfitted single-column sort.                        zero parameters
    naive_trend_over_vol  dist_sma50/vol_30d, unfitted. The coin-level shape of the one arithmetic
                       reading that already beat a 14-feature model in this repo (forecast_regime:
                       dist_sma200/vol_30d, AUC 0.9779, +0.034 over the fitted model). Included so
                       that "the fitted models add nothing over arithmetic" is a measured claim.
    gbm_hand           HistGradientBoosting on the 16 hand-made FEATURE_COLS. the thing to beat
    gbm_seq            HistGradientBoosting on the 480 raw-window columns.    the hypothesis
    gbm_seq_d          the same 480 columns, sampled DAILY so the window spans 60 days.  the SPAN
                       diagnostic: a 10-day window of returns and volumes cannot see where price
                       sits against a 50-day average, which is most of what the label IS. If the
                       raw sequence fails, this separates "the architecture is blind" from "the
                       window is too short", and a negative result is only worth reporting once
                       that distinction has been ruled out.
    gbm_both/_d        hand-made + raw window together.                       does raw ADD anything
    mlp_seq            MLPClassifier on the window - RUN ONLY IF gbm_seq clears naive.
    shuffled null      train labels permuted inside each fold, scored against the real test labels.
                       Traces what a provably signal-free model still scores in this exact harness.

ANTI-LOOKAHEAD, the whole reason to trust any of it
  * every channel at row t is built from bars closed at or before t; only the label looks forward.
  * the forward label is found by WALL-CLOCK (bar_time + 7d, nearest bar within 8h), not by counting
    42 rows, so the 2017-2019 outages cannot smear a target across a gap.
  * walk-forward only: train 730 days, test the next 91, roll. Training STOPS 10 days before the
    test window opens, and 10 > the 7-day label span, so no training row's label can reach into its
    own test window.
  * MLP standardisation and NaN imputation use training-window statistics only.
  * cross-sectional ranks are computed inside a single bar_time, never across bars.
  * the entry threshold for the strategy is a quantile of the TRAIN fold's own predictions. The
    test fold never sees its own distribution.
  * the strategy trades only on bars that have an out-of-sample prediction. Before the first fold
    closes it holds cash; it cannot borrow a model from its own future.

HOW IT TRADES WITHOUT TOUCHING THE ENGINE. app/backtest.run reads columns off the panel, so the
strategy is expressed by overwriting them on a COPY:
    breakout  <- the MODEL's probability clears its fold's threshold. Never don_high. This is the
                 line that removes the reactive trigger.
    score/rank<- the model probability, ranked inside each bar_time.
    breakdown <- shipped 20-bar low (variant "shipped exit") or the model turning negative
                 (variant "model exit").
    stop_level, eligible, costs, fills at the next open, risk limits, 3 slots: untouched.
Two gates are NOT flags - backtest.run hardcodes `regime_on` and `rs_7d > 0` - so a third variant
("model-only gate") neutralises both, which is the only way to see the prediction completely
unassisted. That also disables the regime-flip exit, so it is a different strategy, not a filter
change, and is reported as a diagnostic rather than as a candidate.

MEASUREMENT, reported separately, because skill and money are different questions
    A. skill  - AUC and accuracy per fold, mean +- SE across folds, against persistence.
    A2. the PLACEBO, on the money side. The best variant is re-run with its probabilities permuted
        INSIDE each bar_time, five draws. That leaves the set of probabilities on every bar
        untouched, so the firing rate, the trade count and the exposure profile are identical by
        construction, and the only thing destroyed is which coin got which probability. A real
        economic edge has to land outside that band.
    B. money  - the ship rule against the shipped bot (abs_momentum + volume_min 0.8 + taker_min
                0.5): CAGR >= and Sharpe >= and maxDD not worse, on BOTH panels.
                standard (2017-08..2026-09): CAGR 0.2073  Sharpe 1.4295  maxDD -0.1324  366 trades
                held-out (from 2022-01-01):  CAGR 0.1548  Sharpe 1.2229  maxDD -0.1324  210 trades
                A third window is also reported - "matched", starting at the first out-of-sample
                prediction, with the SHIPPED BOT RE-MEASURED over the same window. The standard
                panel hands the model two years of forced cash, which flatters nothing and distorts
                CAGR; the matched window is the apples-to-apples read and is labelled as such.

RUN
    cd crypto-bot && PYTHONIOENCODING=utf-8 python research/trend_sequence.py
    optional: --no-null (skip the shuffled null)   --panel-cache PATH   --quick (fewer folds)
              --pred-cache DIR (store stage A's output so a re-run of stage B skips the refit)

Writes reports/trend_sequence.json. Touches nothing else.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import backtest, config, features, funding, universe  # noqa: E402
from app.flags import Flags  # noqa: E402

REPORTS = ROOT / "reports"
PINNED_UNIVERSE = REPORTS / "_parity_universe.json"
OUT_JSON = REPORTS / "trend_sequence.json"

BARS_PER_DAY = 6
HORIZON_DAYS = 7            # how far ahead the trend state is asked about
SMA_RISE_DAYS = 10          # over how long the 50-day average must be rising
WINDOW_BARS = 60            # 10 days of 4h candles
TRAIN_DAYS, TEST_DAYS, PURGE_DAYS = 730, 91, 10
MIN_TRAIN_ROWS, MIN_TEST_ROWS = 400, 50

SHIPPED = Flags(abs_momentum=True, volume_min=0.8, taker_min=0.5)
HELD_OUT = "2022-01-01"
EPS_CAGR, EPS_DD, EPS_SHARPE = 0.005, 0.01, 0.0     # same epsilons research/compare.py uses

CHANNELS = ["lr", "vol", "taker", "range"]
HAND = list(features.FEATURE_COLS)                   # 16 columns

GBM_HP = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=25,
              l2_regularization=1.0, random_state=0)
MLP_HP = dict(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=120, early_stopping=True,
              n_iter_no_change=8, validation_fraction=0.15, random_state=0)
THRESH_QS = [0.50, 0.70, 0.80, 0.90]                 # train-fold quantiles for the entry gate
N_NULL_SEEDS = 2


# ============================================================ panel, label, sequence

def load_panel(cache: Path = None) -> tuple:
    """The shipped panel plus funding, from cache when one is supplied."""
    s = config.load()
    u = (json.loads(PINNED_UNIVERSE.read_text()) if PINNED_UNIVERSE.exists()
         else universe.load(s))
    if cache and cache.exists():
        return pd.read_parquet(cache), s, u
    panel = features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars)
    panel = funding.attach(panel, s)
    if cache:
        panel.to_parquet(cache, index=False)
    return panel, s, u


def _asof_back(g: pd.DataFrame, col: str, days: int, newname: str) -> pd.Series:
    """`col` as it stood `days` ago, matched by wall clock (nearest bar within 8h)."""
    want = pd.DataFrame({"k": g["bar_time"] - pd.Timedelta(days=days),
                         "_i": np.arange(len(g))})
    src = pd.DataFrame({"k": g["bar_time"], newname: g[col].to_numpy()})
    m = pd.merge_asof(want.sort_values("k"), src.sort_values("k"), on="k",
                      direction="nearest", tolerance=pd.Timedelta(hours=8))
    return m.sort_values("_i")[newname].to_numpy()


def _asof_fwd(g: pd.DataFrame, col: str, days: int, newname: str) -> np.ndarray:
    """`col` as it WILL stand `days` ahead. A target; never a feature."""
    want = pd.DataFrame({"k": g["bar_time"] + pd.Timedelta(days=days),
                         "_i": np.arange(len(g))})
    src = pd.DataFrame({"k": g["bar_time"], newname: g[col].to_numpy()})
    m = pd.merge_asof(want.sort_values("k"), src.sort_values("k"), on="k",
                      direction="nearest", tolerance=pd.Timedelta(hours=8))
    return m.sort_values("_i")[newname].to_numpy()


def add_trend_label(panel: pd.DataFrame) -> pd.DataFrame:
    """uptrend now, uptrend in 7 days, and the forward 7-day return for economic diagnostics."""
    out = []
    for sym, g in panel.groupby("symbol", sort=False):
        g = g.sort_values("bar_time").reset_index(drop=True)
        sma_then = _asof_back(g, "sma50_d", SMA_RISE_DAYS, "sma_then")
        rising = g["sma50_d"].to_numpy() > sma_then
        above = g["close"].to_numpy() > g["sma50_d"].to_numpy()
        g["uptrend"] = np.where(np.isnan(sma_then) | g["sma50_d"].isna().to_numpy(),
                                np.nan, (above & rising).astype(float))
        g["y_uptrend_fwd"] = _asof_fwd(g, "uptrend", HORIZON_DAYS, "y")
        g["fwd_close"] = _asof_fwd(g, "close", HORIZON_DAYS, "fc")
        g["fwd_ret"] = g["fwd_close"] / g["close"] - 1
        out.append(g)
    panel = pd.concat(out, ignore_index=True)
    # The coin-level analogue of the one arithmetic reading that already beat a 14-feature model in
    # this repo: reports/forecast_regime.json has dist_sma200/vol_30d scoring AUC 0.9779 for "will
    # BTC still be above its 200-day average in 7 days", 0.0344 AUC ABOVE the fitted model. If the
    # same shape wins again here on 17 coins and a 50-day average, that is independent
    # corroboration rather than a coincidence, so it is measured as a named naive, not an aside.
    panel["trend_over_vol"] = panel["dist_sma50"] / panel["vol_30d"].replace(0, np.nan)
    return panel


def channels_for(g: pd.DataFrame) -> np.ndarray:
    """The 4 raw-bar channels for one symbol, (n_bars, 4). Row t uses bar t and earlier only."""
    c = g["close"].to_numpy(float)
    lr = np.r_[np.nan, np.diff(np.log(np.where(c > 0, c, np.nan)))]
    qv = pd.Series(g["quote_volume"].to_numpy(float))
    # per-bar turnover against its own trailing 20-day mean, shifted so t's own bar is excluded
    # from its own benchmark (a self-referencing denominator would leak nothing but would be odd)
    base = qv.rolling(20 * BARS_PER_DAY, min_periods=30).mean().shift(1).to_numpy()
    vol = np.where((base > 0) & np.isfinite(base), qv.to_numpy() / base, np.nan)
    v = g["volume"].to_numpy(float)
    taker = np.where(v > 0, g["taker_buy_volume"].to_numpy(float) / v, np.nan)
    rng = np.where(c > 0, (g["high"].to_numpy(float) - g["low"].to_numpy(float)) / c, np.nan)
    return np.column_stack([lr, vol, taker, rng]).astype(np.float32)


def window_matrix(ch: np.ndarray, rows: np.ndarray, stride=1) -> np.ndarray:
    """Flatten WINDOW_BARS samples of `ch`, ending at each index in `rows`, taken every `stride`.

    stride=1 is the brief's window: the last 60 four-hour bars, i.e. 10 days.
    stride=6 samples one bar per day for 60 days, so the window SPANS the same 50-day horizon the
    label's moving average is built from. That second read exists because of what the first one
    shows - see the SPAN diagnostic in the module docstring.

    Column order is oldest-first. Samples before the start of the series come back NaN, which
    HistGradientBoosting handles natively.
    """
    lags = np.arange(WINDOW_BARS - 1, -1, -1) * stride      # (59..0) * stride
    idx = rows[:, None] - lags[None, :]                     # (m, 60), last column = t itself
    valid = idx >= 0
    take = ch[np.clip(idx, 0, None)]                        # (m, 60, 4)
    take[~valid] = np.nan
    return take.reshape(len(rows), WINDOW_BARS * ch.shape[1])


def seq_names(prefix: str, stride=1) -> list:
    return [f"{prefix}_{c}_lag{lag * stride}"
            for lag in range(WINDOW_BARS - 1, -1, -1) for c in CHANNELS]


def build_rows(panel: pd.DataFrame) -> tuple:
    """Daily decision grid (the 00:00 UTC bar), with the coin's and BTC's raw windows attached.

    Returns (rows DataFrame, {name: float32 array}, {name: column names}). Row orders all match.
      "seq"   the brief's window: 60 consecutive 4h bars  = 10 days
      "seq_d" the SPAN diagnostic: 60 daily-spaced samples = 60 days
    """
    btc = panel[panel["symbol"] == "BTCUSDT"].sort_values("bar_time").reset_index(drop=True)
    btc_ch = channels_for(btc)
    btc_pos = pd.Series(np.arange(len(btc)), index=btc["bar_time"].to_numpy())

    frames, blocks, blocks_d = [], [], []
    for sym, g in panel.groupby("symbol", sort=False):
        g = g.sort_values("bar_time").reset_index(drop=True)
        ch = channels_for(g)
        keep = (g["eligible"].fillna(False).to_numpy()
                & (g["bar_time"].dt.hour == 0).to_numpy()
                & np.isfinite(g["y_uptrend_fwd"].to_numpy(float))
                & np.isfinite(g["uptrend"].to_numpy(float)))
        rows = np.flatnonzero(keep)
        if not len(rows):
            continue
        bpos = btc_pos.reindex(g["bar_time"].to_numpy()[rows]).to_numpy(float)
        ok = np.isfinite(bpos)
        rows, bpos = rows[ok], bpos[ok].astype(int)
        if not len(rows):
            continue
        blocks.append(np.hstack([window_matrix(ch, rows), window_matrix(btc_ch, bpos)]))
        blocks_d.append(np.hstack([window_matrix(ch, rows, BARS_PER_DAY),
                                   window_matrix(btc_ch, bpos, BARS_PER_DAY)]))
        frames.append(g.iloc[rows])

    rows_df = pd.concat(frames, ignore_index=True)
    order = np.lexsort((rows_df["symbol"].to_numpy(), rows_df["bar_time"].to_numpy()))
    rows_df = rows_df.iloc[order].reset_index(drop=True)
    X = {"seq": np.vstack(blocks)[order], "seq_d": np.vstack(blocks_d)[order]}
    cols = {"seq": seq_names("coin") + seq_names("btc"),
            "seq_d": seq_names("coin_d", BARS_PER_DAY) + seq_names("btc_d", BARS_PER_DAY)}
    return rows_df, X, cols


# ============================================================ A. skill, walk-forward

def folds_of(t: pd.Series, quick=False):
    """(test_start, train mask, test mask). Training ends PURGE_DAYS before the test opens."""
    start = t.min() + pd.Timedelta(days=TRAIN_DAYS)
    step = TEST_DAYS * (4 if quick else 1)
    while start < t.max():
        end = start + pd.Timedelta(days=TEST_DAYS)
        tr = ((t >= start - pd.Timedelta(days=TRAIN_DAYS))
              & (t < start - pd.Timedelta(days=PURGE_DAYS)))
        te = (t >= start) & (t < end)
        yield start, tr.to_numpy(), te.to_numpy()
        start = start + pd.Timedelta(days=step)


def _impute_scale(A_tr, A_te):
    """Train-window median imputation and standardisation. Test rows never contribute a statistic."""
    med = np.nanmedian(A_tr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    a = np.where(np.isfinite(A_tr), A_tr, med)
    b = np.where(np.isfinite(A_te), A_te, med)
    mu, sd = a.mean(axis=0), a.std(axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    return (a - mu) / sd, (b - mu) / sd


def fit_predict(A_tr, y_tr, A_te, kind):
    """One fold. Returns (test probabilities, train in-sample probabilities for the threshold)."""
    if kind == "gbm":
        m = HistGradientBoostingClassifier(**GBM_HP).fit(A_tr, y_tr)
        return m.predict_proba(A_te)[:, 1], m.predict_proba(A_tr)[:, 1]
    a, b = _impute_scale(A_tr, A_te)
    m = MLPClassifier(**MLP_HP).fit(a, y_tr)
    return m.predict_proba(b)[:, 1], m.predict_proba(a)[:, 1]


def fold_scores(p, y, fwd, t) -> dict:
    base = float(np.mean(y))
    r = {"n": int(len(y)), "base_rate": base, "majority_acc": float(max(base, 1 - base)),
         "acc": float(np.mean((p > 0.5).astype(int) == y)),
         "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
         "brier": float(np.mean((p - y) ** 2))}
    # economic read-through: what the model's top decile actually earned over the next 7 days
    if len(np.unique(p)) >= 10:
        q = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False).to_numpy()
        r["top_decile_fwd_ret"] = float(np.nanmean(fwd[q == 9]))
        r["bot_decile_fwd_ret"] = float(np.nanmean(fwd[q == 0]))
        r["decile_spread_fwd_ret"] = r["top_decile_fwd_ret"] - r["bot_decile_fwd_ret"]
    else:
        r["top_decile_fwd_ret"] = r["bot_decile_fwd_ret"] = r["decile_spread_fwd_ret"] = float("nan")
    return r


AGG_KEYS = ["auc", "acc", "majority_acc", "base_rate", "brier", "decile_spread_fwd_ret",
            "top_decile_fwd_ret", "bot_decile_fwd_ret"]


def agg(per_fold) -> dict:
    """Mean +- SE ACROSS FOLDS. Folds are disjoint in time; rows inside a fold are not."""
    out = {"folds": len(per_fold), "rows": int(sum(f["n"] for f in per_fold))}
    for k in AGG_KEYS:
        v = np.array([f.get(k, np.nan) for f in per_fold], float)
        v = v[np.isfinite(v)]
        if not len(v):
            out[k] = {"mean": float("nan"), "se": float("nan")}
            continue
        se = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
        out[k] = {"mean": float(v.mean()), "se": se, "n": int(len(v))}
    return out


def paired_margin(cand_folds, base_folds, key="auc") -> dict:
    """PAIRED difference fold by fold. The only number that answers 'is the model worth it'."""
    by = {f["test_start"]: f for f in base_folds}
    d = [c[key] - by[c["test_start"]][key] for c in cand_folds
         if c["test_start"] in by and np.isfinite(c[key]) and np.isfinite(by[c["test_start"]][key])]
    d = np.array(d, float)
    if not len(d):
        return {"mean": float("nan"), "se": float("nan"), "t": float("nan"), "folds": 0}
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    return {"mean": float(d.mean()), "se": se,
            "t": float(d.mean() / se) if se and se > 0 else float("nan"),
            "folds": int(len(d)), "folds_positive": int((d > 0).sum())}


def feature_matrix(rows, X, feat):
    """Assemble one model's design matrix. `X` is the dict from build_rows."""
    hand = rows[HAND].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
    return {"hand": hand,
            "seq": X["seq"],
            "seq_d": X["seq_d"],
            "both": np.hstack([hand, X["seq"]]),
            "both_d": np.hstack([hand, X["seq_d"]])}[feat]


def walk_forward(rows, X, kind="gbm", feat="seq", shuffle_seed=None,
                 naive=None, quick=False, collect=False, log=print):
    """One model, walked forward. `naive` names a column used unfitted as the score.

    With collect=True also returns the out-of-sample probability per test row and that row's
    TRAIN-fold threshold for each quantile in THRESH_QS - which is how the strategy gets a gate
    that never saw its own test distribution.
    """
    t = rows["bar_time"]
    y = rows["y_uptrend_fwd"].to_numpy(float).astype(int)
    fwd = rows["fwd_ret"].to_numpy(float)
    A = None if naive is not None else feature_matrix(rows, X, feat)
    rng = np.random.default_rng(0 if shuffle_seed is None else shuffle_seed)
    per_fold, oos = [], []
    for start, tr, te in folds_of(t, quick):
        if tr.sum() < MIN_TRAIN_ROWS or te.sum() < MIN_TEST_ROWS:
            continue
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        if naive is not None:
            p = rows[naive].to_numpy(float)[te]
            ok = np.isfinite(p)
            if ok.sum() < MIN_TEST_ROWS:
                continue
            m = fold_scores(p[ok], y[te][ok], fwd[te][ok], t.to_numpy()[te][ok])
            p_tr = rows[naive].to_numpy(float)[tr]
            p_tr = p_tr[np.isfinite(p_tr)]
            idx_te = np.flatnonzero(te)[ok]
        else:
            y_tr = y[tr]
            if shuffle_seed is not None:
                y_tr = rng.permutation(y_tr)      # train on noise, score against the REAL labels
                if len(np.unique(y_tr)) < 2:
                    continue
            p, p_tr = fit_predict(A[tr], y_tr, A[te], kind)
            m = fold_scores(p, y[te], fwd[te], t.to_numpy()[te])
            idx_te = np.flatnonzero(te)
        m["test_start"] = str(pd.Timestamp(start).date())
        m["train_rows"] = int(tr.sum())
        per_fold.append(m)
        if collect:
            thr = {f"thr_{q:g}": float(np.quantile(p_tr, q)) for q in THRESH_QS}
            oos.append(pd.DataFrame({"_row": idx_te, "p": p, **thr}))
        log(f"    fold {m['test_start']}  n={m['n']:<5d} AUC {m['auc']:.4f}  "
            f"acc {m['acc']:.4f} (majority {m['majority_acc']:.4f})")
    if collect:
        return per_fold, (pd.concat(oos, ignore_index=True) if oos else pd.DataFrame())
    return per_fold


# ============================================================ B. money

def attach_predictions(panel: pd.DataFrame, rows: pd.DataFrame, oos: pd.DataFrame,
                       thr_col: str) -> pd.DataFrame:
    """Carry each daily prediction onto the 4h panel, forward only, 5 bars (the rest of that day).

    A probability stamped 00:00 is used at 04:00..20:00 unchanged. That is stale, not clairvoyant,
    and it is what lets a slot freed mid-day be filled by a decision already on the record.
    """
    sig = rows.iloc[oos["_row"].to_numpy()][["bar_time", "symbol"]].copy()
    sig["p"] = oos["p"].to_numpy()
    sig["thr"] = oos[thr_col].to_numpy()
    p = panel.merge(sig, on=["bar_time", "symbol"], how="left")
    p = p.sort_values(["symbol", "bar_time"]).reset_index(drop=True)
    g = p.groupby("symbol", sort=False)
    p["p_f"] = g["p"].ffill(limit=BARS_PER_DAY - 1)
    p["thr_f"] = g["thr"].ffill(limit=BARS_PER_DAY - 1)
    return p.sort_values(["bar_time", "symbol"]).reset_index(drop=True)


def make_strategy_panel(p: pd.DataFrame, fresh=False, model_exit=False, exit_q=0.5,
                        model_only_gate=False) -> pd.DataFrame:
    """Overwrite the engine's decision columns with the MODEL's output. No breakout anywhere."""
    p = p.copy()
    live = (p["p_f"].notna() & p["thr_f"].notna()).to_numpy()
    say_up = pd.Series(live & (p["p_f"] >= p["thr_f"]).to_numpy(), index=p.index)

    if fresh:
        # only the bar the prediction FLIPS positive, so a coin predicted up for 60 days straight
        # is not a standing invitation. Still a prediction, just a de-duplicated one.
        prev = say_up.groupby(p["symbol"], sort=False).shift(1, fill_value=False).to_numpy(bool)
        enter = say_up.to_numpy() & ~prev
    else:
        enter = say_up.to_numpy()

    p["breakout"] = enter
    if model_exit:
        # the model turning negative is the exit. Rows with no live prediction do not force one.
        p["breakdown"] = live & (p["p_f"] < p["thr_f"] * exit_q).to_numpy()
    if model_only_gate:
        # The engine also demands regime_on and rs_7d > 0 before any entry, and those are not
        # flags - they are wired into backtest.run. Neutralising them is the only way to see the
        # model's signal completely unassisted. It also disables the "BTC regime flipped off"
        # exit, so this is a DIFFERENT strategy, not a filter change, and is labelled as such.
        p["regime_on"] = True
        p["rs_7d"] = p["rs_7d"].fillna(0.0).clip(lower=1e-9)

    # ranking: the model's probability, inside each bar_time. Eligible rows with no prediction get
    # a floor score so they still carry a valid rank - a NaN rank makes the engine's "fell out of
    # the top N" rule eject a position for a reason that has nothing to do with the strategy.
    el = p["eligible"].fillna(False)
    p["score"] = np.where(el & p["p_f"].notna(), 100.0 * p["p_f"].fillna(0.0), -1.0)
    p["score"] = np.where(el, p["score"], np.nan)
    p["rank"] = np.nan
    p.loc[el, "rank"] = p[el].groupby("bar_time")["score"].rank(ascending=False, method="first")
    return p.sort_values(["bar_time", "rank"]).reset_index(drop=True)


def placebo_predictions(oos: pd.DataFrame, rows: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Permute the probability WITHIN each bar_time. The sharpest control there is.

    The set of probabilities on any given bar is untouched, so the number of coins clearing the
    threshold on every single bar - and therefore the firing rate, the trade count and the
    exposure profile - is IDENTICAL. The only thing destroyed is WHICH coin got which probability.
    If the real strategy's money result sits inside the band this produces, the model's ranking
    contributed nothing economically, whatever its AUC said.
    """
    out = oos.copy()
    bt = rows.iloc[oos["_row"].to_numpy()]["bar_time"].to_numpy()
    rng = np.random.default_rng(seed)
    p = out["p"].to_numpy().copy()
    order = np.argsort(bt, kind="stable")
    starts = np.flatnonzero(np.r_[True, bt[order][1:] != bt[order][:-1], True])
    for a, b in zip(starts[:-1], starts[1:]):
        block = order[a:b]
        p[block] = p[rng.permutation(block)]
    out["p"] = p
    return out


def measure(panel, s, flags, start=None, p_random=True):
    res = backtest.run(panel, s, flags=flags, start=start)
    m = backtest.metrics(res.equity, res.trades, res.exposure)
    if p_random and len(res.trades):
        rb = backtest.random_baselines(panel, res.trades, s) or {}
        m["p_coin"] = rb.get("p_value_vs_random_coin", float("nan"))
        m["p_time"] = rb.get("p_value_vs_random_time", float("nan"))
    return res, m


def verdict(base, cand, held_out=False):
    """research/compare.py's rule verbatim: drawdown is negative, so 'not worse' means >=."""
    raw = {"CAGR": (cand["cagr"], base["cagr"], EPS_CAGR),
           "Sharpe": (cand["sharpe"], base["sharpe"], EPS_SHARPE),
           "maxDD": (cand["max_drawdown"], base["max_drawdown"], EPS_DD)}
    checks, used = {}, 0
    for name, (c, b, eps) in raw.items():
        if c >= b:
            checks[name] = True
        elif held_out and c >= b - eps:
            checks[name], used = True, used + 1
        else:
            checks[name] = False
    if held_out and used > 1:
        checks["one-eps-only"] = False
    return all(checks.values()), checks


def line(tag, m):
    return (f"  {tag:<34} CAGR {m['cagr']:+.4f}  Sharpe {m['sharpe']:7.4f}  "
            f"maxDD {m['max_drawdown']:.4f}  trades {int(m.get('trades', 0)):>4}  "
            f"win {m.get('win_rate', float('nan')):.3f}")


# ============================================================ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-cache", default="")
    ap.add_argument("--pred-cache", default="",
                    help="directory to cache the walk-forward output in, so re-running the money "
                         "stage does not refit 28 folds. Written after stage A, reused if present.")
    ap.add_argument("--no-null", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cache = Path(args.pred_cache) if args.pred_cache else None

    t0 = time.time()
    report = {"spec": {"target": "close > sma50_d AND sma50_d rising over 10d, 7 days ahead",
                       "window_bars": WINDOW_BARS, "channels": CHANNELS,
                       "walk_forward": f"{TRAIN_DAYS}/{TEST_DAYS} purge {PURGE_DAYS}",
                       "quick": args.quick}}

    panel, s, u = load_panel(Path(args.panel_cache) if args.panel_cache else None)
    panel = add_trend_label(panel)
    rows, X, seq_cols = build_rows(panel)
    X_seq = X["seq"]
    print(f"panel {panel.shape}  symbols {panel.symbol.nunique()}  "
          f"rows {rows.shape}  seq features {X_seq.shape[1]}  ({time.time() - t0:.0f}s)")
    print(f"grid {rows.bar_time.min().date()} .. {rows.bar_time.max().date()}  "
          f"base rate (uptrend in 7d) {rows.y_uptrend_fwd.mean():.4f}  "
          f"uptrend now {rows.uptrend.mean():.4f}")
    report["data"] = {"panel_rows": int(len(panel)), "symbols": int(panel.symbol.nunique()),
                      "grid_rows": int(len(rows)), "seq_features": int(X_seq.shape[1]),
                      "base_rate": float(rows.y_uptrend_fwd.mean()),
                      "uptrend_now": float(rows.uptrend.mean())}

    # ---------------- A. does it predict?
    print("\n=== A. SKILL (walk-forward, never in-sample) ===")
    skill, folds = {}, {}

    ORDER = ["naive_persistence", "naive_dist_sma50", "naive_trend_over_vol", "gbm_hand",
             "gbm_seq", "gbm_seq_d", "gbm_both", "gbm_both_d"]

    NAIVE = {"naive_persistence": ("uptrend", "today's trend state, zero parameters"),
             "naive_dist_sma50": ("dist_sma50", "unfitted single-column sort"),
             "naive_trend_over_vol": ("trend_over_vol",
                                      "dist_sma50/vol_30d, the arithmetic that beat a "
                                      "14-feature model in forecast_regime.json")}
    FITTED = {"gbm_hand": ("hand", "16 hand-made features  <- the thing to beat"),
              "gbm_seq": ("seq", "480 raw 4h-bar columns, 10-day span  <- the hypothesis"),
              "gbm_seq_d": ("seq_d", "480 daily-spaced columns, 60-day span  <- SPAN diagnostic"),
              "gbm_both": ("both", "hand-made + 10-day window  <- does raw ADD anything"),
              "gbm_both_d": ("both_d", "hand-made + 60-day window")}

    # The cache is split so that adding a naive baseline (seconds) does not force a refit of the
    # four gradient-boosted models (many minutes). Delete one file to recompute only that half.
    ctag = "quick" if args.quick else "full"
    f_naive = (cache / f"folds_naive_{ctag}.json") if cache else None
    f_fit = (cache / f"folds_fitted_{ctag}.json") if cache else None
    oos = {}

    if f_naive and f_naive.exists():
        print(f"  reusing cached naive baselines from {cache}")
        folds.update(json.loads(f_naive.read_text()))
    else:
        for k, (col, note) in NAIVE.items():
            print(f"\n  {k:<20} ({note})")
            folds[k] = walk_forward(rows, X, naive=col, quick=args.quick)
        if cache:
            cache.mkdir(parents=True, exist_ok=True)
            f_naive.write_text(json.dumps({k: folds[k] for k in NAIVE}, default=str))

    if f_fit and f_fit.exists() and all((cache / f"oos_{k}_{ctag}.parquet").exists()
                                        for k in FITTED):
        print(f"  reusing cached fitted models from {cache} (no refit)")
        folds.update(json.loads(f_fit.read_text()))
        for k in FITTED:
            oos[k] = pd.read_parquet(cache / f"oos_{k}_{ctag}.parquet")
    else:
        for k, (feat, note) in FITTED.items():
            print(f"\n  {k:<20} ({note})")
            folds[k], oos[k] = walk_forward(rows, X, feat=feat, quick=args.quick, collect=True)
        if cache:
            cache.mkdir(parents=True, exist_ok=True)
            f_fit.write_text(json.dumps({k: folds[k] for k in FITTED}, default=str))
            for k, v in oos.items():
                v.to_parquet(cache / f"oos_{k}_{ctag}.parquet", index=False)
            print(f"  cached walk-forward output to {cache}")
    oos_seq, oos_seq_d, oos_both = oos["gbm_seq"], oos["gbm_seq_d"], oos["gbm_both"]

    for k, v in folds.items():
        skill[k] = agg(v)

    base = folds["naive_persistence"]
    margins = {k: {"auc": paired_margin(v, base, "auc"), "acc": paired_margin(v, base, "acc")}
               for k, v in folds.items() if k != "naive_persistence"}

    print("\n  --- skill summary (mean +- SE across folds) ---")
    for k in [k for k in ORDER if k in skill]:
        a = skill[k]
        print(f"  {k:<20} folds {a['folds']:>3}  AUC {a['auc']['mean']:.4f} +- "
              f"{a['auc']['se']:.4f}   acc {a['acc']['mean']:.4f}  "
              f"(majority {a['majority_acc']['mean']:.4f})  "
              f"brier {a['brier']['mean']:.4f}")
    print("\n  --- PAIRED margin over naive_persistence (the number that matters) ---")
    for k in [k for k in ORDER[1:] if k in margins]:
        mg = margins[k]
        print(f"  {k:<20} dAUC {mg['auc']['mean']:+.4f} +- {mg['auc']['se']:.4f} "
              f"(t={mg['auc']['t']:+.2f}, {mg['auc']['folds_positive']}/{mg['auc']['folds']} folds up)"
              f"   dACC {mg['acc']['mean']:+.4f} (t={mg['acc']['t']:+.2f})")

    # the three comparisons that decide what this file concludes
    vs_hand = {k: paired_margin(folds[k], folds["gbm_hand"], "auc")
               for k in ("gbm_seq", "gbm_seq_d", "gbm_both", "gbm_both_d")}
    print("\n  --- PAIRED margin over gbm_hand (does the raw sequence ADD anything) ---")
    for k, mg in vs_hand.items():
        print(f"  {k:<20} dAUC {mg['mean']:+.4f} +- {mg['se']:.4f} (t={mg['t']:+.2f}, "
              f"{mg['folds_positive']}/{mg['folds']} folds up)")

    seq_beats_naive = (margins["gbm_seq"]["auc"]["mean"] > 0
                       and np.isfinite(margins["gbm_seq"]["auc"]["t"])
                       and margins["gbm_seq"]["auc"]["t"] > 2.0)
    report["skill"] = skill
    report["margin_vs_persistence"] = margins
    report["margin_vs_gbm_hand"] = vs_hand

    # MLP only after the simple model has earned it, exactly as the brief requires
    if seq_beats_naive:
        print("\n  gbm_seq cleared naive -> fitting mlp_seq")
        folds["mlp_seq"] = walk_forward(rows, X, kind="mlp", feat="seq", quick=args.quick)
        skill["mlp_seq"] = agg(folds["mlp_seq"])
        margins["mlp_seq"] = {"auc": paired_margin(folds["mlp_seq"], base, "auc"),
                              "acc": paired_margin(folds["mlp_seq"], base, "acc")}
        a = skill["mlp_seq"]
        print(f"  mlp_seq  AUC {a['auc']['mean']:.4f} +- {a['auc']['se']:.4f}  "
              f"dAUC vs persistence {margins['mlp_seq']['auc']['mean']:+.4f}")
        report["skill"]["mlp_seq"] = a
        report["margin_vs_persistence"]["mlp_seq"] = margins["mlp_seq"]
    else:
        print("\n  gbm_seq did NOT clear naive_persistence by a margin worth chasing "
              "-> mlp_seq deliberately NOT fitted (the brief's rule)")
        report["mlp_skipped"] = "gbm_seq did not beat naive_persistence; no deep model attempted"

    # shuffled null: what a provably signal-free model scores in this exact harness
    if not args.no_null:
        print("\n  shuffled null (train labels permuted, scored on real test labels)")
        nulls = []
        for seed in range(1, N_NULL_SEEDS + 1):
            nf = walk_forward(rows, X, feat="seq", shuffle_seed=seed,
                              quick=args.quick, log=lambda *a: None)
            nulls.append(agg(nf))
            print(f"    seed {seed}: AUC {nulls[-1]['auc']['mean']:.4f}  "
                  f"acc {nulls[-1]['acc']['mean']:.4f}")
        report["shuffled_null"] = nulls

    # ---------------- B. does it make money?
    print(f"\n=== B. MONEY (ship rule vs the shipped bot) ===   ({time.time() - t0:.0f}s)")
    first_oos = rows.iloc[oos_seq["_row"].to_numpy()]["bar_time"].min()
    print(f"  first out-of-sample prediction: {first_oos.date()}  "
          f"(before this the strategy is forced into cash)")
    report["first_oos"] = str(first_oos.date())

    windows = [("standard", None), ("held-out", HELD_OUT), ("matched", str(first_oos.date()))]
    # Strip every forward-looking column before anything is traded. Nothing in backtest.run reads
    # them, but a panel that still carries fwd_ret is a panel a reviewer cannot audit at a glance.
    trade_panel = panel.drop(columns=[c for c in ("uptrend", "y_uptrend_fwd", "fwd_close",
                                                  "fwd_ret") if c in panel.columns])
    base_m = {}
    for name, start in windows:
        _, m = measure(trade_panel, s, SHIPPED, start)
        base_m[name] = m
        print(line(f"SHIPPED [{name}]", m))
    report["shipped"] = base_m

    pred_sets = [("gbm_seq", oos_seq), ("gbm_seq_d", oos_seq_d), ("gbm_both", oos_both)]
    results = {}
    for model_name, oos in pred_sets:
        for q in THRESH_QS:
            p_att = attach_predictions(trade_panel, rows, oos, f"thr_{q:g}")
            for fresh in (False, True):
                for flag_name, fl in (("shipped-filters", SHIPPED), ("no-filters", Flags())):
                    tag = f"{model_name} q{q:g} {'fresh' if fresh else 'standing'} {flag_name}"
                    p2 = make_strategy_panel(p_att, fresh=fresh)
                    row = {}
                    for name, start in windows:
                        try:
                            _, m = measure(p2, s, fl, start, p_random=False)
                        except Exception as e:                     # noqa: BLE001
                            row[name] = {"error": str(e)}
                            continue
                        ok, checks = verdict(base_m[name], m,
                                             held_out=(name == "held-out"))
                        row[name] = {**{k: float(v) if isinstance(v, (int, float)) else str(v)
                                        for k, v in m.items()},
                                     "pass": bool(ok), "checks": checks}
                    results[tag] = row
                    st, ho, mt = row["standard"], row["held-out"], row["matched"]
                    print(line(tag, st) + f"  | matched CAGR {mt['cagr']:+.4f} "
                          f"Sh {mt['sharpe']:.3f} DD {mt['max_drawdown']:.4f} "
                          f"| heldout CAGR {ho['cagr']:+.4f} Sh {ho['sharpe']:.3f}"
                          f" | PASS std/ho/mt "
                          f"{int(st['pass'])}/{int(ho['pass'])}/{int(mt['pass'])}")

    report["strategies"] = results

    # Two diagnostics on the strongest prediction set, so nobody can say the result hangs on the
    # shipped exit ladder or on the shipped regime gate doing the model's work for it.
    print("\n  diagnostics (gbm_both q0.8 standing):")
    p_att = attach_predictions(trade_panel, rows, oos_both, "thr_0.8")
    for tag, kw in (("model-exit", dict(model_exit=True)),
                    ("model-only-gate (regime + RS removed)", dict(model_only_gate=True))):
        p3 = make_strategy_panel(p_att, fresh=False, **kw)
        row = {}
        for name, start in windows:
            _, m = measure(p3, s, SHIPPED, start, p_random=False)
            ok, _c = verdict(base_m[name], m, held_out=(name == "held-out"))
            row[name] = {**{k: float(v) if isinstance(v, (int, float)) else str(v)
                            for k, v in m.items()}, "pass": bool(ok)}
        print(line(f"gbm_both q0.8 {tag}", row["standard"])
              + f"  | matched CAGR {row['matched']['cagr']:+.4f} "
                f"Sh {row['matched']['sharpe']:.3f}")
        report["strategies"][f"gbm_both q0.8 standing {tag} shipped-filters"] = row

    # ---------------- the placebo: same firing rate, same bars, WRONG coins
    # Whichever variant did best on the matched window is re-run with its probabilities permuted
    # inside each bar_time. Trade count and exposure are preserved by construction, so a real
    # result has to sit OUTSIDE this band. If it sits inside, the AUC bought nothing.
    standing = {k: v for k, v in results.items()
                if "standing" in k and "shipped-filters" in k
                and np.isfinite(v["matched"].get("sharpe", np.nan))}
    best = max(standing, key=lambda k: standing[k]["matched"]["sharpe"])
    model_of = {"gbm_seq_d": oos_seq_d, "gbm_both": oos_both, "gbm_seq": oos_seq}
    mkey = next(k for k in ("gbm_seq_d", "gbm_both", "gbm_seq") if best.startswith(k))
    qkey = best.split()[1][1:]
    print(f"\n  placebo for the best standing variant: {best}")
    band = []
    for seed in range(1, 6):
        pl = placebo_predictions(model_of[mkey], rows, seed)
        p4 = make_strategy_panel(attach_predictions(trade_panel, rows, pl, f"thr_{qkey}"))
        _, m = measure(p4, s, SHIPPED, str(first_oos.date()), p_random=False)
        band.append({"seed": seed, "cagr": float(m["cagr"]), "sharpe": float(m["sharpe"]),
                     "max_drawdown": float(m["max_drawdown"]), "trades": int(m.get("trades", 0))})
        print(f"    placebo seed {seed}: matched CAGR {m['cagr']:+.4f}  "
              f"Sharpe {m['sharpe']:7.4f}  maxDD {m['max_drawdown']:.4f}  "
              f"trades {int(m.get('trades', 0))}")
    real = results[best]["matched"]
    sh = [b["sharpe"] for b in band]
    beat = sum(1 for v in sh if v >= real["sharpe"])
    print(f"    real {best}: matched Sharpe {real['sharpe']:.4f}  vs placebo band "
          f"[{min(sh):.4f}, {max(sh):.4f}] mean {np.mean(sh):.4f}  "
          f"-> {beat}/5 placebos matched or beat it")
    report["placebo"] = {"variant": best, "real_matched": real, "draws": band,
                         "placebos_beating_real": beat}

    # ---------------- verdict
    shippable = [k for k, v in results.items()
                 if v["standard"].get("pass") and v["held-out"].get("pass")]
    print(f"\nSHIPPABLE (standard AND held-out, vs the quoted shipped numbers): "
          f"{shippable if shippable else 'NONE'}")
    matched_pass = [k for k, v in results.items()
                    if v["matched"].get("pass") and v["held-out"].get("pass")]
    print(f"PASSES on the matched window AND held-out: "
          f"{matched_pass if matched_pass else 'NONE'}")
    report["shippable"] = shippable
    report["matched_pass"] = matched_pass
    report["runtime_s"] = round(time.time() - t0, 1)

    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {OUT_JSON}   ({report['runtime_s']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
