"""FORWARD TREND-STRENGTH REGRESSION: predict the trend ahead instead of reacting to a breakout.

WHY THIS EXISTS. The shipped bot is reactive: it waits for a close above the prior 20-bar high and
then buys. The owner asked for a bot that PREDICTS the trend and enters on that prediction. This
project has already closed the question "can the next 7-day RETURN DIRECTION be predicted" - it
cannot (AUC 0.4688-0.5219 on an honest harness that scored 0.7905 on a predictable target), and
six ML formulations of return all cut CAGR. But returns and TRENDS are not the same question.
Trends are autocorrelated; a coin grinding upward tends to keep grinding for a while. So the label
here is not a return, it is a TREND-QUALITY measure:

    target_t = OLS slope of log(close) over bars t+1 .. t+84   (= the next 14 days on the 4h grid)
               converted to per-day, divided by the coin's TRAILING daily volatility

Dividing by trailing vol (vol_30d / sqrt(365), a number known at t) makes the label comparable
across a calm large cap and a wild small cap without putting one scrap of forward information into
the denominator. The label is, in effect, the forward trend's daily t-statistic: a steady climb
scores high, a violent chop that ends in the same place scores near zero, and a round trip up and
back scores ~0 even though its max return was large.

FEATURES (all strictly trailing, every value at t uses only bars closed at or before t):
  slope_7 / slope_14 / slope_30 / slope_60     the same vol-scaled slope measured BACKWARD
  r2_7 / r2_14 / r2_30 / r2_60                 R-squared of those trailing fits = trend cleanliness
  vol_ratio_7_30, log_vol_30d, atr_pct         volatility state
  volume_ratio, taker_ratio                    turnover and buying pressure
  funding_pct                                  crowding (the one non-price signal that works here)
  rs_7d, rs_30d                                relative strength against BTC
  btc_slope_14, btc_slope_30, btc_dist_sma200  the market's own trend

NAIVE BASELINE - the margin over this is the whole result, not the raw R2:
  naive        prediction = slope_14, the trailing 14-day slope carried forward unchanged
  naive_cal    OLS target ~ a + b*slope_14 fitted on the TRAIN window only. This is the honest
               carry-forward: it strips the level/shrinkage bias, so any margin over naive_cal is
               forecasting skill and not the model noticing that forward slopes are flatter than
               trailing ones (mean reversion in the LEVEL, which the raw naive gets wrong for free).
  shuffled     the same GBM on a permuted label - the harness control. It must score <= 0.

WALK-FORWARD, and nothing else:
    |<-------- 730d train -------->|  15d purge  |<--- 91d test --->|   then roll forward 91d
  The purge is longer than the 14-day label span, so no training label resolves inside a test
  window. Standardisation and imputation use TRAIN-window statistics only. Cross-sectional ranks
  (funding_pct, rs_*) are computed within a bar_time by the shipped feature code.

THEN TRADE IT. backtest.run reads `breakout`, `breakdown`, `stop_level`, `eligible` and a score
column off the panel, so the strategy is expressed by overwriting those on a COPY - the engine,
the costs, the fills at the next open, the risk limits and the slots are untouched:
    breakout  <- predicted slope > 0        (there is NO breakout trigger left anywhere)
    breakdown <- predicted slope < 0        (exit when the predicted trend turns down)
    score_col <- the predicted slope        (rank candidates by forecast trend strength)
    stop_level, eligible                     unchanged, so sizing is identical to the shipped bot

    python research/trend_slope.py                 # everything
    python research/trend_slope.py --stage skill   # part A only
    python research/trend_slope.py --stage trade   # part B only (reuses the cached predictions)
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import backtest, config, features, funding            # noqa: E402
from app.flags import Flags                                    # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PINNED_UNIVERSE = ROOT / "reports" / "_parity_universe.json"

BPD = 6                      # 4h bars per day
TF_MS = 4 * 3_600_000
ANNUAL = np.sqrt(365.0)      # vol_30d is annualised; /sqrt(365) makes it a daily sigma
HORIZON_D = 14               # the label looks 14 days forward
HORIZON = HORIZON_D * BPD    # ... which is 84 bars
TRAIL_D = (7, 14, 30, 60)

TRAIN_D, TEST_D, PURGE_D = 730, 91, 15
HELD_OUT_START = "2022-01-01"

# The shipped bot. This flag set reproduces the frozen baseline to four decimals on both panels
# (standard 0.2073 / 1.4295 / -0.1324 / 366, held-out 0.1548 / 1.2229 / -0.1324 / 210), which is
# how we know the reference is the real one and not a convenient one.
SHIPPED = Flags(abs_momentum=True, volume_min=0.8, taker_min=0.5)
SHIP_RULE = {"standard": {"cagr": 0.2073, "sharpe": 1.4295, "maxdd": -0.1324, "trades": 366},
             "held-out": {"cagr": 0.1548, "sharpe": 1.2229, "maxdd": -0.1324, "trades": 210}}

SLOPE_COLS = [f"slope_{d}" for d in TRAIL_D]
R2_COLS = [f"r2_{d}" for d in TRAIL_D]
FEATS = SLOPE_COLS + R2_COLS + [
    "vol_ratio_7_30", "log_vol_30d", "atr_pct", "volume_ratio", "taker_ratio",
    "funding_pct", "rs_7d", "rs_30d", "btc_slope_14", "btc_slope_30", "btc_dist_sma200",
]
SLIM = SLOPE_COLS + R2_COLS + ["vol_ratio_7_30"]      # the vol study's lesson: fewer columns win
# every predictor that gets scored, level models and the cross-sectional ranker together
ALL_PRED = ["model", "gbm_slim", "ridge", "gbm_xs", "naive", "naive_cal", "shuffled",
            "shipped_score"]
MODELS = ["model", "gbm_slim", "ridge", "gbm_xs"]


# ---------------------------------------------------------------- rolling OLS on a fixed grid

def _roll_sum(a, w):
    """Trailing sum of width w, NaN for the first w-1 positions."""
    c = np.cumsum(np.insert(a, 0, 0.0))
    out = np.full(a.shape, np.nan)
    out[w - 1:] = c[w:] - c[:-w]
    return out


def slope_r2(y, w, open_ms):
    """Trailing OLS slope per bar, and its R2, over a window of w bars ending at each index.

    x is 0..w-1 so the design is fixed and the slope is a single dot product. A window whose bars
    are not a contiguous 4h grid (an exchange outage) is dropped rather than measured, because the
    missing bars would be silently treated as having no time pass.
    """
    n = len(y)
    x = np.arange(w) - (w - 1) / 2.0
    sxx = float((x * x).sum())
    out_s = np.full(n, np.nan)
    if n >= w:
        # convolve with the reversed weights = trailing dot product, aligned to the window's end
        out_s[w - 1:] = np.convolve(y, (x / sxx)[::-1], mode="valid")
    s1, s2 = _roll_sum(y, w), _roll_sum(y * y, w)
    sst = s2 - s1 * s1 / w
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = np.where(sst > 0, out_s ** 2 * sxx / sst, np.nan)
    # contiguity of the trailing window
    ok = np.full(n, False)
    if n >= w:
        ok[w - 1:] = (open_ms[w - 1:] - open_ms[:n - w + 1]) == (w - 1) * TF_MS
    out_s[~ok] = np.nan
    r2 = np.where(ok, r2, np.nan)
    return out_s, np.clip(r2, 0.0, 1.0)


def forward_slope(y, h, open_ms):
    """OLS slope per bar over bars t+1 .. t+h, NaN where that window is not a contiguous grid."""
    n = len(y)
    trail, _ = slope_r2(y, h, open_ms)          # trail[k] covers k-h+1 .. k
    out = np.full(n, np.nan)
    if n > h:
        out[:n - h] = trail[h:]                 # so out[t] covers t+1 .. t+h
        gap_ok = (open_ms[h:] - open_ms[:n - h]) == h * TF_MS
        out[:n - h] = np.where(gap_ok, out[:n - h], np.nan)
    out[n - h:] = np.nan
    return out


def build(panel):
    """Add the trailing slope/R2 features, the forward label, and the diagnostic forward return."""
    frames = []
    for sym, g in panel.groupby("symbol", sort=False):
        g = g.sort_values("bar_time").copy()
        y = np.log(g["close"].to_numpy(dtype=float))
        ms = g["open_time"].to_numpy(dtype="int64")
        # daily sigma, known at t. The same scaler is used for the label and every slope feature.
        dsig = g["vol_30d"].to_numpy(dtype=float) / ANNUAL
        with np.errstate(invalid="ignore", divide="ignore"):
            inv = np.where(dsig > 1e-6, 1.0 / dsig, np.nan)
        for d in TRAIL_D:
            s, r2 = slope_r2(y, d * BPD, ms)
            g[f"slope_{d}"] = s * BPD * inv       # per-bar -> per-day -> per-day-sigma
            g[f"r2_{d}"] = r2
        g["target"] = forward_slope(y, HORIZON, ms) * BPD * inv
        c = g["close"].to_numpy(dtype=float)
        fwd = np.full(len(c), np.nan)
        if len(c) > HORIZON:
            fwd[:len(c) - HORIZON] = c[HORIZON:] / c[:len(c) - HORIZON] - 1.0
            gap_ok = (ms[HORIZON:] - ms[:len(c) - HORIZON]) == HORIZON * TF_MS
            fwd[:len(c) - HORIZON] = np.where(gap_ok, fwd[:len(c) - HORIZON], np.nan)
        g["fwd_ret_14d"] = fwd
        frames.append(g)
    p = pd.concat(frames, ignore_index=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        p["vol_ratio_7_30"] = p["vol_7d"] / p["vol_30d"]
        p["log_vol_30d"] = np.log(p["vol_30d"].clip(lower=1e-6))
    btc = p[p["symbol"] == "BTCUSDT"].set_index("bar_time")[["slope_14", "slope_30"]]
    p = p.join(btc.add_prefix("btc_"), on="bar_time")
    return p.sort_values(["bar_time", "symbol"]).reset_index(drop=True)


# ---------------------------------------------------------------- part A: does it predict?

def folds(times):
    t0, t1 = times.min(), times.max()
    out, test_start = [], t0 + pd.Timedelta(days=TRAIN_D + PURGE_D)
    while test_start + pd.Timedelta(days=TEST_D) <= t1:
        out.append((test_start - pd.Timedelta(days=TRAIN_D + PURGE_D),   # train from
                    test_start - pd.Timedelta(days=PURGE_D),             # train to (exclusive)
                    test_start,                                          # test from
                    test_start + pd.Timedelta(days=TEST_D)))             # test to (exclusive)
        test_start = test_start + pd.Timedelta(days=TEST_D)
    return out


def _models():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge

    def gbm():
        return HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100,
            l2_regularization=1.0, early_stopping=False, random_state=0)
    return gbm, Ridge


def fit_predict(tr, te, feats, kind, rng=None, sink=None):
    """Fit on tr, predict te. Every statistic used comes from tr only."""
    gbm, Ridge = _models()
    X, y = tr[feats].to_numpy(dtype=float), tr["target"].to_numpy(dtype=float)
    Xt = te[feats].to_numpy(dtype=float)
    if kind == "gbm":
        return gbm().fit(X, y).predict(Xt)
    if kind == "gbm_xs":
        # target is the within-bar percentile; centre on 0 so the sign still means "above or
        # below the typical coin that bar", which is what the trading rule reads.
        yq = tr["target_pct"].to_numpy(dtype=float)
        return gbm().fit(X, yq).predict(Xt) - 0.5
    if kind == "gbm_shuffled":
        return gbm().fit(X, rng.permutation(y)).predict(Xt)
    if kind == "ridge":
        med = np.nanmedian(X, axis=0)
        mu, sd = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        Z = (np.where(np.isnan(X), med, X) - mu) / sd
        Zt = (np.where(np.isnan(Xt), med, Xt) - mu) / sd
        mdl = Ridge(alpha=10.0).fit(Z, y)
        if sink is not None:          # standardised coefficients: directly comparable to each other
            sink.append(dict(zip(feats, mdl.coef_)))
        return mdl.predict(Zt)
    raise ValueError(kind)


def calibrate(tr, te, col):
    """OLS target ~ a + b*col on the train window: the honest carry-forward."""
    a = tr[col].to_numpy(dtype=float)
    y = tr["target"].to_numpy(dtype=float)
    m = np.isfinite(a) & np.isfinite(y)
    if m.sum() < 100:
        return np.full(len(te), np.nan)
    b, c = np.polyfit(a[m], y[m], 1)
    return c + b * te[col].to_numpy(dtype=float)


def ic_series(oos, col, against="target"):
    """Per-bar_time Spearman between two columns, as a Series indexed by bar_time."""
    d = oos[["bar_time", col, against]].dropna()
    if d.empty:
        return pd.Series(dtype=float)
    g = d.groupby("bar_time")
    d = d.assign(a=g[col].rank(), b=g[against].rank())
    d["ab"] = d["a"] * d["b"]
    g = d.groupby("bar_time")
    n = g["a"].size()
    cov = g["ab"].mean() - g["a"].mean() * g["b"].mean()
    sa, sb = g["a"].std(ddof=0), g["b"].std(ddof=0)
    ic = (cov / (sa * sb)).replace([np.inf, -np.inf], np.nan)
    return ic[n >= 5].dropna()


def ic_bootstrap(oos, col, ref=None, against="target", block_days=HORIZON_D, n=400, seed=13):
    """P(this predictor's mean IC is not above zero / not above `ref`), on 14-day blocks of bars.

    The per-bar ICs are not independent - a 14-day label makes 84 consecutive bars nearly the same
    observation - so the block is one label length, the shortest that breaks the overlap.
    """
    a = ic_series(oos, col, against)
    if a.empty:
        return {}
    df = pd.DataFrame({"a": a})
    if ref:
        df["b"] = ic_series(oos, ref, against)
    df = df.dropna()
    df["blk"] = df.index.floor(f"{block_days}D")
    blocks = [g for _, g in df.groupby("blk")]
    rng = np.random.default_rng(seed)
    idx = np.arange(len(blocks))
    not_pos = not_better = 0
    for _ in range(n):
        q = pd.concat([blocks[i] for i in rng.choice(idx, len(idx), replace=True)])
        not_pos += q["a"].mean() <= 0
        if ref:
            not_better += q["a"].mean() <= q["b"].mean()
    out = {"p_ic_not_positive": not_pos / n}
    if ref:
        out[f"p_ic_not_above_{ref}"] = not_better / n
    return out


def xs_ic(oos, col, against="target"):
    """Mean within-bar_time Spearman rank correlation - the metric a RANKER lives or dies on.

    A pooled Spearman over the whole frame mixes "which coin is trending hardest today" with
    "is this a trending month", and only the first is tradeable: the strategy picks the best few
    coins available on one bar. Spearman is Pearson on ranks, so ranking inside each bar and then
    taking a grouped correlation is the same number and runs in one pass.
    """
    ic = ic_series(oos, col, against)
    if ic.empty:
        return {}
    return {"ic_mean": float(ic.mean()), "ic_median": float(ic.median()),
            "ic_std": float(ic.std()), "ic_share_positive": float((ic > 0).mean()),
            "n_bars": int(len(ic))}


def skill(p, verbose=True):
    rng = np.random.default_rng(7)
    need = ["target"] + SLOPE_COLS + R2_COLS
    rows = p[p["eligible"].fillna(False)].dropna(subset=need).copy()
    # Within-bar percentile of the label: "is this coin's forward trend strong RELATIVE to what
    # else was on the menu that bar". Computed inside a bar_time only, so it cannot see the future;
    # a whole bar always falls in the same fold, so it cannot leak across the train/test boundary.
    rows["target_pct"] = rows.groupby("bar_time")["target"].rank(pct=True)
    fl = folds(rows["bar_time"])
    preds, per_fold, coefs = [], [], []
    names = ["model", "gbm_slim", "ridge", "naive", "naive_cal", "shuffled"]
    for k, (tr0, tr1, te0, te1) in enumerate(fl):
        tr = rows[(rows["bar_time"] >= tr0) & (rows["bar_time"] < tr1)]
        te = rows[(rows["bar_time"] >= te0) & (rows["bar_time"] < te1)]
        if len(tr) < 5000 or len(te) < 200:
            continue
        t0 = time.time()
        out = pd.DataFrame({"bar_time": te["bar_time"].to_numpy(), "symbol": te["symbol"].to_numpy(),
                            "target": te["target"].to_numpy(), "fold": k,
                            "fwd_ret_14d": te["fwd_ret_14d"].to_numpy(),
                            "train_mean": float(tr["target"].mean())})
        out["target_pct"] = te["target_pct"].to_numpy()
        out["shipped_score"] = te["score"].to_numpy()     # the bot's existing ranker, as reference
        out["model"] = fit_predict(tr, te, FEATS, "gbm")
        out["gbm_slim"] = fit_predict(tr, te, SLIM, "gbm")
        out["ridge"] = fit_predict(tr, te, FEATS, "ridge", sink=coefs)
        out["gbm_xs"] = fit_predict(tr, te, FEATS, "gbm_xs")
        out["naive"] = te["slope_14"].to_numpy()
        out["naive_cal"] = calibrate(tr, te, "slope_14")
        out["shuffled"] = fit_predict(tr, te, FEATS, "gbm_shuffled", rng)
        preds.append(out)
        row = {"fold": k, "test_from": str(te0.date()), "test_to": str(te1.date()),
               "n_train": len(tr), "n_test": len(te), "secs": round(time.time() - t0, 1)}
        sst = float(((out["target"] - out["train_mean"]) ** 2).sum())
        for nm in names:
            sse = float(((out["target"] - out[nm]) ** 2).sum())
            row[nm] = 1 - sse / sst if sst > 0 else np.nan
        for nm in ALL_PRED:
            row["ic_" + nm] = xs_ic(out, nm).get("ic_mean", np.nan)
        per_fold.append(row)
        if verbose:
            print(f"  fold {k:2d} {row['test_from']}..{row['test_to']} "
                  f"train {len(tr):6d} test {len(te):5d}  "
                  + "  ".join(f"{nm} {row[nm]:+.4f}" for nm in names) + f"   {row['secs']}s",
                  flush=True)
    oos = pd.concat(preds, ignore_index=True)
    ff = pd.DataFrame(per_fold)

    # pooled R2 against each fold's TRAIN mean (no test information in the denominator)
    sst = float(((oos["target"] - oos["train_mean"]) ** 2).sum())
    sst_tm = float(((oos["target"] - oos["target"].mean()) ** 2).sum())
    pooled = {}
    from scipy import stats
    for nm in names:
        sse = float(((oos["target"] - oos[nm]) ** 2).sum())
        pooled[nm] = {
            "r2_vs_train_mean": 1 - sse / sst,
            "r2_vs_test_mean": 1 - sse / sst_tm,
            "rmse": float(np.sqrt(((oos["target"] - oos[nm]) ** 2).mean())),
            "spearman": float(stats.spearmanr(oos[nm], oos["target"]).statistic),
            "folds_beating_naive": int((ff[nm] > ff["naive"]).sum()),
            "folds_beating_naive_cal": int((ff[nm] > ff["naive_cal"]).sum()),
            "n_folds": len(ff),
        }
    # the ranking question, which is the one the strategy actually asks
    ic = {}
    for nm in ALL_PRED:
        row = xs_ic(oos, nm, "target")
        row["ic_vs_fwd_ret"] = xs_ic(oos, nm, "fwd_ret_14d").get("ic_mean", np.nan)
        row["folds_ic_positive"] = int((ff["ic_" + nm] > 0).sum())
        row["folds_ic_beats_naive"] = int((ff["ic_" + nm] > ff["ic_naive"]).sum())
        row.update(ic_bootstrap(oos, nm, ref="naive"))
        ic[nm] = row
    # Is the model saying anything the trailing slope and the shipped ranker do not already say?
    overlap = {}
    for nm in MODELS:
        overlap[nm] = {"vs_naive": xs_ic(oos, nm, "naive").get("ic_mean", np.nan),
                       "vs_shipped_score": xs_ic(oos, nm, "shipped_score").get("ic_mean", np.nan)}
    cf = pd.DataFrame(coefs)
    coef_tbl = pd.DataFrame({"mean": cf.mean(), "std": cf.std(),
                             "share_same_sign": cf.apply(lambda c: max((c > 0).mean(),
                                                                       (c < 0).mean()))})
    coef_tbl = coef_tbl.reindex(coef_tbl["mean"].abs().sort_values(ascending=False).index)
    return oos, ff, pooled, ic, overlap, coef_tbl


def block_bootstrap(oos, a, b, block_days=HORIZON_D, n=400, seed=11):
    """P(a is not better than b) on a moving block bootstrap of 14-day blocks of squared error.

    Adjacent 4h rows share 83 of their 84 forward bars, so the nominal row count is a fantasy. The
    block is as long as the label, which is the shortest block that breaks the overlap.
    """
    d = oos.copy()
    d["blk"] = d["bar_time"].dt.floor(f"{block_days}D")
    e = d.groupby("blk").apply(
        lambda g: pd.Series({"sse_a": ((g["target"] - g[a]) ** 2).sum(),
                             "sse_b": ((g["target"] - g[b]) ** 2).sum(),
                             "sst": ((g["target"] - g["train_mean"]) ** 2).sum()}),
        include_groups=False)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(e))
    worse = 0
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        q = e.iloc[s]
        r2a = 1 - q["sse_a"].sum() / q["sst"].sum()
        r2b = 1 - q["sse_b"].sum() / q["sst"].sum()
        worse += r2a <= r2b
    return worse / n


def quintiles(oos, col):
    d = oos[[col, "target", "fwd_ret_14d"]].dropna(subset=[col, "target"]).copy()
    d["q"] = pd.qcut(d[col], 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    g = d.groupby("q", observed=True).agg(n=(col, "size"), pred=(col, "mean"),
                                          realised=("target", "mean"),
                                          realised_med=("target", "median"),
                                          fwd_ret=("fwd_ret_14d", "mean"),
                                          share_up=("target", lambda x: float((x > 0).mean())))
    return g.reset_index().rename(columns={"q": "quintile"})


# ---------------------------------------------------------------- part B: does it make money?

def verify(panel, p, s, out):
    """ADVERSARIAL CHECKS. Every one of these has to pass or Part A and Part B mean nothing.

    1. truncation invariance - rebuild the features from a panel that has been cut off at a date.
       If a feature at time t secretly reads a bar after t, deleting those bars must change it.
       This is the only test that catches a lookahead bug rather than arguing about one.
    2. the purge really purges - the last training label must resolve before the test window opens.
    3. fold predictions are invariant to future data - retrain one fold on a truncated panel and
       demand bit-identical predictions.
    4. the stop is below the entry price on every entry, or sizing silently returns zero.
    5. random-coin / random-time placebos on the winning strategy's own trades.
    """
    res = {}
    print("\n=== VERIFICATION ===")

    cut = pd.Timestamp("2024-06-01", tz="UTC")
    trunc = build(panel[panel["bar_time"] <= cut].copy())
    cols = SLOPE_COLS + R2_COLS + ["vol_ratio_7_30", "log_vol_30d", "btc_slope_14", "btc_slope_30"]
    a = p[p["bar_time"] <= cut].set_index(["bar_time", "symbol"])[cols]
    b = trunc.set_index(["bar_time", "symbol"])[cols]
    common = a.index.intersection(b.index)
    d = (a.loc[common] - b.loc[common]).abs()
    worst = float(np.nanmax(d.to_numpy())) if len(common) else float("nan")
    nan_mismatch = int((a.loc[common].isna() != b.loc[common].isna()).to_numpy().sum())
    res["truncation_worst_diff"] = worst
    res["truncation_nan_mismatch"] = nan_mismatch
    print(f"1. truncation invariance at {cut.date()}: {len(common)} shared rows, "
          f"worst feature diff {worst:.3g}, NaN-pattern mismatches {nan_mismatch} "
          f"-> {'PASS' if worst < 1e-9 and nan_mismatch == 0 else 'FAIL'}")

    # the label at the truncation edge must be unknowable, i.e. NaN for the last HORIZON bars
    tail = trunc[trunc["bar_time"] > cut - pd.Timedelta(days=HORIZON_D)]
    res["edge_labels_all_nan"] = bool(tail["target"].isna().all())
    print(f"2a. labels inside {HORIZON_D}d of the data edge are all NaN: "
          f"{'PASS' if res['edge_labels_all_nan'] else 'FAIL'}")

    rows = p[p["eligible"].fillna(False)].dropna(subset=["target"] + SLOPE_COLS + R2_COLS)
    fl = folds(rows["bar_time"])
    gaps = [(te0 - tr1).days for tr0, tr1, te0, te1 in fl]
    res["min_purge_days"] = min(gaps)
    print(f"2b. purge gap {min(gaps)}d vs a {HORIZON_D}d label -> "
          f"{'PASS' if min(gaps) >= HORIZON_D else 'FAIL'}")

    # 3. one fold, retrained on a panel that cannot see past the fold's test window
    rows = rows.copy()
    rows["target_pct"] = rows.groupby("bar_time")["target"].rank(pct=True)
    k = len(fl) // 2
    tr0, tr1, te0, te1 = fl[k]
    tr = rows[(rows["bar_time"] >= tr0) & (rows["bar_time"] < tr1)]
    te = rows[(rows["bar_time"] >= te0) & (rows["bar_time"] < te1)]
    t2 = build(panel[panel["bar_time"] < te0].copy())
    t2 = t2[t2["eligible"].fillna(False)].dropna(subset=["target"] + SLOPE_COLS + R2_COLS)
    t2 = t2[(t2["bar_time"] >= tr0) & (t2["bar_time"] < tr1)]
    full = fit_predict(tr, te, FEATS, "ridge")
    cutd = fit_predict(t2, te, FEATS, "ridge")
    delta = float(np.nanmax(np.abs(full - cutd)))
    res["fold_retrain_worst_diff"] = delta
    print(f"3. fold {k} retrained with everything after {te0.date()} deleted: "
          f"{len(tr)} vs {len(t2)} train rows, worst prediction diff {delta:.3g} -> "
          f"{'PASS' if delta < 1e-9 else 'FAIL'}")

    # 4. the stop must sit below the close wherever the strategy is allowed to enter
    q = strategy_panel(p, "pred")
    ent = q["breakout"] & q["eligible"] & q["regime_on"].fillna(False)
    bad = int((ent & (q["stop_level"] >= q["close"])).sum())
    res["entries_with_stop_at_or_above_close"] = bad
    print(f"4. entry-eligible rows whose stop is not below the close: {bad} of {int(ent.sum())} "
          f"-> {'PASS' if bad == 0 else 'note: those rows size to zero and are skipped'}")

    # 5. did the ranking pick coins and moments better than chance?
    r = backtest.run(q, s, score_col="pred", flags=Flags(), max_names=3)
    rb = backtest.random_baselines(p, r.trades, s, n=300)
    res["random_baselines"] = {k2: float(v) for k2, v in rb.items()}
    print("5. placebos on the winning strategy's own trades (300 draws each):")
    for k2, v in rb.items():
        print(f"     {k2:<28} {v:+.4f}")
    (out / "trend_slope_verify.json").write_text(json.dumps(res, indent=2, default=str))
    return res


def fmt(m):
    return (f"CAGR {m['cagr']:+.4f}  Sharpe {m.get('sharpe', float('nan')):.4f}  "
            f"maxDD {m['max_drawdown']:.4f}  trades {m.get('trades', 0):>4}  "
            f"win {m.get('win_rate', float('nan')):.3f}  exp {m.get('avg_exposure', float('nan')):.3f}  "
            f"{m['start']}..{m['end']}")


def run_one(panel, s, label, score_col=None, flags=None, max_names=None, start=None):
    res = backtest.run(panel, s, score_col=score_col or "score", flags=flags if flags is not None
                       else SHIPPED, max_names=max_names, start=start)
    if len(res.equity) < 3:
        return {"label": label, "error": "no bars"}
    m = backtest.metrics(res.equity, res.trades, res.exposure)
    m = {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in m.items()}
    m["label"] = label
    return m


def ship_verdict(by_window):
    """The ship rule, verbatim: CAGR >= baseline AND Sharpe >= baseline AND maxDD not worse, on
    BOTH the standard panel and the held-out half. Anything short of all six is a reject."""
    out, ok = {}, True
    for w, want in SHIP_RULE.items():
        m = by_window.get(w)
        if not m or "error" in m:
            out[w] = "no result"
            ok = False
            continue
        c = m["cagr"] >= want["cagr"]
        sh = m["sharpe"] >= want["sharpe"]
        dd = m["max_drawdown"] >= want["maxdd"]      # less negative = not worse
        out[w] = ("CAGR " + ("ok" if c else f"short {m['cagr'] - want['cagr']:+.4f}")
                  + " | Sharpe " + ("ok" if sh else f"short {m['sharpe'] - want['sharpe']:+.4f}")
                  + " | maxDD " + ("ok" if dd else f"worse {m['max_drawdown'] - want['maxdd']:+.4f}"))
        ok &= c and sh and dd
    out["ship"] = ok
    return out


def rotation_panel(p):
    """THE CONTROL. Remove the breakout trigger and change nothing else: same shipped ranker,
    same exits, same filters, entry allowed on any bar the coin is in the top 5 by the shipped
    score. If this does as well as the model, then whatever the model's version gains comes from
    dropping the trigger, not from the forecast - and the forecast is decoration."""
    q = p.copy()
    q["breakout"] = q["eligible"].fillna(False)
    return q


def strategy_panel(p, pred_col, *, free_regime=False, relax_rs=False):
    """Express the prediction as a tradeable panel, by overwriting only the engine's input columns.

    `breakout` becomes "the model predicts a positive forward trend" - the 20-bar high is gone from
    the entry path entirely. `breakdown` becomes "the model predicts a negative forward trend".
    stop_level and eligible are left exactly as the shipped bot computes them, so position sizing
    and the universe are identical and the comparison stays apples to apples.
    """
    q = p.copy()
    pr = q[pred_col]
    q["breakout"] = (pr > 0).fillna(False)
    q["breakdown"] = (pr < 0).fillna(False)
    if "breakdown_exit" in q.columns:
        q["breakdown_exit"] = q["breakdown"]
    q["eligible"] = q["eligible"].fillna(False) & pr.notna()
    if free_regime:
        # "does the model subsume the shipped filters": no 200-day BTC gate on entry or exit
        q["regime_on"] = True
    if relax_rs:
        # backtest.run also demands rs_7d > 0. Neutralising it tests the prediction ALONE.
        # Nothing else in the run path reads rs_7d (only flags.rs_cap, which stays off), and the
        # shipped `score` column is not used because a prediction column is passed instead.
        q["rs_7d"] = 1e-9
    return q


def trade(p, s, pred_cols, out, verbose=True):
    first = p.loc[p[pred_cols[0]].notna(), "bar_time"].min()
    print(f"\nfirst out-of-sample prediction: {first.date()}  "
          f"(walk-forward needs {TRAIN_D}d of training plus a {PURGE_D}d purge, so the standard "
          f"panel carries {(first - p['bar_time'].min()).days} days this strategy cannot trade)")
    rows, verdicts = [], {}
    # "standard" starts at the panel's own start, so the untradeable training years count against
    # the candidate exactly as the ship rule intends. "matched" is the diagnostic that strips that
    # handicap out, and is never used to claim a pass.
    windows = (("standard", None), ("matched", first), ("held-out", HELD_OUT_START))

    def block(title, panel, label, score_col=None, flags=None, max_names=None, judge=True):
        print(f"\n{title}")
        got = {}
        for w, st in windows:
            m = run_one(panel, s, f"{label} [{w}]", score_col=score_col, flags=flags,
                        max_names=max_names, start=st)
            m["group"], m["window"], m["config"] = title.split(" -")[0], w, label
            got[w] = m
            rows.append(m)
            if verbose:
                print(f"  {w:<9} " + (fmt(m) if "error" not in m else m["error"]), flush=True)
        if judge:
            v = ship_verdict(got)
            verdicts[label] = v
            print(f"  ship rule: {'PASS' if v['ship'] else 'REJECT'}")
            for w in SHIP_RULE:
                print(f"     {w:<9} {v[w]}")
        return got

    block("REFERENCE - the shipped bot (breakout trigger, shipped ranker, shipped flags)",
          p, "shipped", judge=False)
    # Both flag settings, so the control always matches the candidate it is being compared with.
    for fl_lbl, fl in (("flags=shipped", SHIPPED), ("flags=none", Flags())):
        block(f"CONTROL - shipped ranker, breakout trigger REMOVED (pure rotation), {fl_lbl}",
              rotation_panel(p), f"rotation on shipped score {fl_lbl}", flags=fl, max_names=3)

    for pc in ("pred", "pred_naive"):
        if pc not in pred_cols:
            continue
        for n in (3, 5):
            for fl_lbl, fl in (("flags=shipped", SHIPPED), ("flags=none", Flags())):
                block(f"CANDIDATE - rank by {pc}, enter top {n}, no breakout trigger, {fl_lbl}",
                      strategy_panel(p, pc), f"{pc} topN={n} {fl_lbl}",
                      score_col=pc, flags=fl, max_names=n)

    if "pred_shuffled" in pred_cols:
        # THE PLACEBO. Identical machinery, identical exposure profile, but the model behind the
        # ranking was trained on a PERMUTED label, so it cannot know anything. If this scores like
        # the real one, the result belongs to the rotation mechanics and not to the forecast.
        block("PLACEBO - same rotation, ranked by a model trained on a SHUFFLED label",
              strategy_panel(p, "pred_shuffled"), "pred_shuffled topN=3 flags=none",
              score_col="pred_shuffled", flags=Flags(), max_names=3)

    for free, rs in ((True, False), (False, True)):
        tag = "no-regime-gate" if free else "no-rs7d-gate"
        block(f"VARIANT - pred topN=3, {tag}",
              strategy_panel(p, "pred", free_regime=free, relax_rs=rs),
              f"pred topN=3 {tag}", score_col="pred", max_names=3)

    # LEVERAGE NORMALISATION. The winning configuration runs roughly 1.5x the shipped bot's average
    # exposure, and a strategy that simply takes more risk is not a better strategy. risk_pct is
    # the one dial that changes size and nothing else - not which coin, not when - so turning it
    # down puts both bots on the same risk budget. Chosen to match the shipped bot's drawdown,
    # which is an in-sample choice and is reported as one.
    import dataclasses
    for rp in (0.0066, 0.005, 0.004):
        s2 = dataclasses.replace(s, risk_pct=rp)
        print(f"\nRISK-MATCHED - pred topN=3 flags=none at risk_pct={rp:g} "
              f"(shipped runs {s.risk_pct:g})")
        got = {}
        for w, st in windows:
            m = run_one(strategy_panel(p, "pred"), s2, f"pred risk={rp:g} [{w}]",
                        score_col="pred", flags=Flags(), max_names=3, start=st)
            m["group"], m["window"], m["config"] = "RISK-MATCHED", w, f"pred topN=3 risk={rp:g}"
            got[w], _ = m, rows.append(m)
            print(f"  {w:<9} " + (fmt(m) if "error" not in m else m["error"]), flush=True)
        v = ship_verdict(got)
        verdicts[f"pred topN=3 risk={rp:g}"] = v
        print(f"  ship rule: {'PASS' if v['ship'] else 'REJECT'}")
        for w in SHIP_RULE:
            print(f"     {w:<9} {v[w]}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "trend_slope_trades.csv", index=False)
    print("\n" + "=" * 110)
    print("SUMMARY - every configuration against the ship rule (standard AND held-out, all three "
          "metrics)")
    print("=" * 110)
    piv = df[df["group"] != "REFERENCE"].pivot_table(
        index="config", columns="window", values=["cagr", "sharpe", "max_drawdown"])
    print(piv.round(4).to_string())
    print("\npassed the ship rule: "
          + (", ".join(k for k, v in verdicts.items() if v["ship"]) or "NOTHING"))
    return df, verdicts


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["skill", "trade", "verify", "all"])
    ap.add_argument("--out", default=str(Path(tempfile.gettempdir()) / "trend_slope"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200, "display.max_columns", 50)

    s = config.load()
    u = json.loads(PINNED_UNIVERSE.read_text())
    cache = out / "panel.parquet"
    if cache.exists():
        panel = pd.read_parquet(cache)
    else:
        panel = funding.attach(features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars), s)
        panel.to_parquet(cache, index=False)
    p = build(panel)
    print(f"panel {p.shape}  {p['bar_time'].min().date()}..{p['bar_time'].max().date()}  "
          f"{p['symbol'].nunique()} symbols  labelled rows "
          f"{int(p['target'].notna().sum())}  eligible+labelled "
          f"{int((p['target'].notna() & p['eligible'].fillna(False)).sum())}")

    report = {}
    pred_path = out / "trend_slope_preds.parquet"
    if a.stage in ("skill", "all"):
        print(f"\n=== PART A: does it predict?  train {TRAIN_D}d / purge {PURGE_D}d / "
              f"test {TEST_D}d, label spans {HORIZON_D}d ===")
        oos, ff, pooled, ic, overlap, coef_tbl = skill(p)
        oos.to_parquet(pred_path, index=False)
        print(f"\npooled out-of-sample: {len(oos)} rows, {len(ff)} folds, "
              f"{oos['bar_time'].min().date()}..{oos['bar_time'].max().date()}")
        print("\nQ1: can the LEVEL of the forward slope be predicted? (R2, higher is better)")
        print(pd.DataFrame(pooled).T.round(4).to_string())
        print("\nQ2: can coins be RANKED by forward trend strength? (mean within-bar Spearman)")
        print(pd.DataFrame(ic).T.round(4).to_string())
        best = max(["model", "gbm_slim", "ridge"], key=lambda k: pooled[k]["r2_vs_train_mean"])
        best_ic = max(MODELS, key=lambda k: ic[k]["ic_mean"])
        print(f"\nbest by R2: {best}   best by ranking IC: {best_ic}   "
              f"naive IC {ic['naive']['ic_mean']:+.4f}   shuffled IC {ic['shuffled']['ic_mean']:+.4f}")
        print("\nis the model saying anything new? (within-bar rank corr of prediction vs ...)")
        print(pd.DataFrame(overlap).T.round(4).to_string())
        print("\nwhat the linear model actually reads (standardised coefficients, 27 folds):")
        print(coef_tbl.round(4).to_string())
        margins = {}
        for nm in ["model", "gbm_slim", "ridge"]:
            for bl in ["naive", "naive_cal"]:
                margins[f"{nm}_minus_{bl}"] = round(
                    pooled[nm]["r2_vs_train_mean"] - pooled[bl]["r2_vs_train_mean"], 4)
        print("\nmargins over the naive (log of the whole result):")
        for k, v in margins.items():
            print(f"  {k:<28} {v:+.4f}")
        boot = {f"{best}_vs_naive": block_bootstrap(oos, best, "naive"),
                f"{best}_vs_naive_cal": block_bootstrap(oos, best, "naive_cal")}
        print(f"\nblock bootstrap ({HORIZON_D}-day blocks, 400 draws), P(not better):")
        for k, v in boot.items():
            print(f"  {k:<28} {v:.4f}")
        for nm in dict.fromkeys([best, best_ic, "naive"]):
            tag = {best: "best by R2", best_ic: "best by ranking IC", "naive": "naive carry-forward"}
            print(f"\nquintiles of {nm} ({tag[nm]}) vs realised:")
            print(quintiles(oos, nm).round(4).to_string(index=False))
        report["skill"] = {"pooled": pooled, "ic": ic, "overlap": overlap,
                           "ridge_coefficients": coef_tbl.to_dict("index"),
                           "margins": margins, "bootstrap": boot,
                           "best": best, "best_ic": best_ic,
                           "n_oos": len(oos), "n_folds": len(ff),
                           "oos_from": str(oos["bar_time"].min().date()),
                           "oos_to": str(oos["bar_time"].max().date()),
                           "per_fold": ff.to_dict("records"),
                           "quintiles_best": quintiles(oos, best).to_dict("records"),
                           "quintiles_best_ic": quintiles(oos, best_ic).to_dict("records"),
                           "quintiles_naive": quintiles(oos, "naive").to_dict("records")}
        ff.to_csv(out / "trend_slope_folds.csv", index=False)

    if a.stage in ("trade", "all", "verify"):
        oos = pd.read_parquet(pred_path)
        sk = report.get("skill")
        if sk is None:        # --stage trade on its own: recover the choices from the cache
            cands = ["model", "gbm_slim", "ridge", "gbm_xs"]
            sk = {"best": max(cands, key=lambda k: -float(((oos["target"] - oos[k]) ** 2).sum())),
                  "best_ic": max(cands, key=lambda k: xs_ic(oos, k).get("ic_mean", -9))}
        print(f"trading the ranking model ({sk['best_ic']}), the best level model ({sk['best']}) "
              f"and the naive trailing slope")
        src = {"pred": sk["best_ic"], "pred_naive": "naive", "pred_shuffled": "shuffled"}
        pr = oos[["bar_time", "symbol"]].copy()
        for new, old in src.items():
            pr[new] = oos[old].to_numpy()
        p2 = p.merge(pr, on=["bar_time", "symbol"], how="left")
        assert len(p2) == len(p), "the prediction join duplicated panel rows"
        if a.stage == "verify":
            report["verify"] = verify(panel, p2, s, out)
        else:
            print("\n=== PART B: does it make money? ===")
            df, verdicts = trade(p2, s, list(src), out)
            report["trade"] = df.to_dict("records")
            report["verdicts"] = verdicts
            report["traded_models"] = src
            report["ship_rule"] = SHIP_RULE
            report["verify"] = verify(panel, p2, s, out)

    (out / "trend_slope.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {out / 'trend_slope.json'}")


if __name__ == "__main__":
    main()
