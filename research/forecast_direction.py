"""DIRECTION: the control experiment. Is the sign of a coin's next 7 days predictable at all?

WHY THIS EXISTS. Four separate attempts in this repo have failed to predict returns (walk-forward
ranker, ML veto on cross-sectional excess return, win-probability model that ran backwards to
profit, right-tail classifier with out-of-sample AUC 0.4914). Those were all *trade-level* studies
on the bot's own candidate population. This file asks the cleaner, more basic question on the WHOLE
panel, with a harness built to be audited rather than to win:

    Can anything in these features order coins by the sign of their next 7-day move?

It is deliberately the CONTROL. The expected answer is NO. The value of the file is not the answer,
it is the proof that the measuring instrument is honest, because the sibling studies (volatility,
regime persistence) use the same harness and their positive results are only believable if this one
comes back empty and the two calibration experiments below behave as they must.

TARGETS
  dir_abs   sign of the next 7-day return                          (all 17 symbols)
  dir_rel   sign of the next 7-day return MINUS BTC's              (BTC excluded: its own excess
                                                                    return is identically zero)

THREE CALIBRATIONS, and they are the whole point:
  1. NAIVE BASELINES, unfitted, using only features at t: momentum (ret_30d / rs_30d), short
     momentum (ret_7d / rs_7d), trend (dist_sma200). If a fitted model cannot beat a one-column
     sort, the fitting bought nothing.
  2. SHUFFLED NULL (no false positives). The train labels are permuted inside each fold and the
     model is then scored against the REAL test labels, 7 seeds. This traces the distribution of
     metrics a model with provably zero signal still produces in this exact harness. A real result
     inside that band is noise, and saying so is the deliverable.
  3. POSITIVE CONTROL (no false negatives). The same harness, same features, same folds, aimed at
     a target that is known to be predictable: is this coin's realised volatility over the next
     7 days above the cross-sectional median of that bar? Volatility clusters, so a working harness
     must find this easily. If the direction result is 0.50 AND the volatility control is 0.50, the
     harness is broken, not the market. Both numbers are needed to tell those two cases apart.

PROTOCOL (same as the sibling studies)
  * Walk-forward: train on a trailing 730 days, PURGE 10 days, test the next 91 days, roll. The
    purge exceeds the 7-day label horizon, so no training label resolves inside a test window.
  * Features at t use only bars closed at or before t (app/features.py guarantees this: rolling
    windows, don_* shifted by one bar, daily SMAs joined backward-asof). Targets look forward only.
  * Standardisation and NaN imputation use TRAINING-window statistics only, refit every fold.
  * Significance is computed ACROSS FOLDS, not across rows. Folds are disjoint in time; rows are
    not, because a 7-day label sampled daily overlaps itself ~7x. Row-level p-values here would be
    a lie by a factor of sqrt(7), so they are never quoted.
  * Overlap is then attacked directly: every headline target is re-run on a NON-OVERLAPPING weekly
    grid (one decision per symbol per 7 days) where consecutive labels cannot share a single bar.

DECISION GRID. One bar per symbol per day (the 00:00 UTC 4h bar), restricted to `eligible` rows
(>= 90 days of history, features present). Using all six 4h bars a day would multiply the row count
by six without adding a single independent observation.

METRICS. Per fold: AUC; accuracy at 0.5 against the majority-class accuracy of that fold; pooled
Spearman IC of prediction vs realised return; cross-sectional IC (Spearman within each bar_time,
then averaged); and the mean realised return of the top prediction decile minus the bottom decile.
Aggregated as mean +- standard error across folds, with a t-statistic. An IC of 0.02 +- 0.03 is
reported as noise, because it is noise.

KNOWN LIMITATIONS, stated rather than hidden:
  * SURVIVORSHIP. The universe is the top coins as of 2026-09; coins that died are absent. That
    biases the level of returns upward and would flatter any long-only result. It cannot create
    cross-sectional direction skill, and the shuffled null is subject to the identical bias, so the
    MARGIN over the null is unaffected.
  * The forward 7-day realised volatility of the positive control is bar-counted over 42 bars; the
    few hour-scale gaps in 2017-2019 make that slightly ragged, which can only weaken the control.

RUN
    cd crypto-bot && PYTHONIOENCODING=utf-8 python research/forecast_direction.py

Writes reports/forecast_direction.json.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config, features  # noqa: E402

BARS_PER_DAY = 6
ANNUAL = np.sqrt(BARS_PER_DAY * 365)
HORIZON_DAYS = 7
TRAIN_DAYS, TEST_DAYS, PURGE_DAYS = 730, 91, 10
MIN_TRAIN_ROWS = 400
N_SHUFFLE = 7

FEATS = list(features.FEATURE_COLS)        # 16 columns, all own-past or cross-section at t
GBM_HP = dict(max_iter=250, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=25,
              l2_regularization=1.0, random_state=0)


# ---------------------------------------------------------------- targets

def add_forward(panel: pd.DataFrame) -> pd.DataFrame:
    """Forward 7-day return and forward 7-day realised vol per symbol, plus BTC's forward return.

    The forward close is found by WALL-CLOCK time (bar_time + 7 days, nearest bar within 8h) rather
    than by counting 42 rows, so an exchange outage shifts nothing. Everything added here looks
    FORWARD and is therefore a target, never a feature.
    """
    n = HORIZON_DAYS * BARS_PER_DAY
    out = []
    for sym, g in panel.groupby("symbol", sort=False):
        g = g.sort_values("bar_time").reset_index(drop=True)
        tgt = pd.DataFrame({"bar_time": g["bar_time"],
                            "target_time": g["bar_time"] + pd.Timedelta(days=HORIZON_DAYS)})
        src = pd.DataFrame({"target_time": g["bar_time"], "fwd_close": g["close"]})
        m = pd.merge_asof(tgt.sort_values("target_time"), src.sort_values("target_time"),
                          on="target_time", direction="nearest",
                          tolerance=pd.Timedelta(hours=8))
        g = g.merge(m[["bar_time", "fwd_close"]], on="bar_time", how="left")
        g["fwd_ret"] = g["fwd_close"] / g["close"] - 1
        lr = np.log(g["close"]).diff()
        # vol of bars t+1 .. t+n, brought back to t
        g["fwd_vol"] = (lr.rolling(n).std() * ANNUAL).shift(-n)
        out.append(g)
    panel = pd.concat(out, ignore_index=True)

    btc = panel.loc[panel["symbol"] == "BTCUSDT", ["bar_time", "fwd_ret"]] \
               .rename(columns={"fwd_ret": "btc_fwd_ret"})
    panel = panel.merge(btc, on="bar_time", how="left")
    panel["fwd_excess"] = panel["fwd_ret"] - panel["btc_fwd_ret"]
    return panel


def build_rows(panel: pd.DataFrame) -> pd.DataFrame:
    """Daily decision grid: the 00:00 UTC bar of each day, eligible rows only."""
    d = panel[panel["eligible"] & (panel["bar_time"].dt.hour == 0)].copy()
    d = d.dropna(subset=["fwd_ret"]).sort_values(["bar_time", "symbol"]).reset_index(drop=True)
    # cross-sectional median of the FORWARD vol, for the positive control's label
    d["fwd_vol_xs_med"] = d.groupby("bar_time")["fwd_vol"].transform("median")
    return d


# ---------------------------------------------------------------- walk-forward

def folds_of(t: pd.Series):
    """(train mask, test mask) pairs. Train ends PURGE_DAYS before the test window opens."""
    start = t.min() + pd.Timedelta(days=TRAIN_DAYS)
    while start < t.max():
        end = start + pd.Timedelta(days=TEST_DAYS)
        tr = (t >= start - pd.Timedelta(days=TRAIN_DAYS)) & (t < start - pd.Timedelta(days=PURGE_DAYS))
        te = (t >= start) & (t < end)
        yield start, tr.to_numpy(), te.to_numpy()
        start = end


def fit_predict(X_tr, y_tr, X_te, kind):
    """One fold's model. All preprocessing statistics come from the training rows only."""
    if kind == "gbm":
        m = HistGradientBoostingClassifier(**GBM_HP)
        m.fit(X_tr, y_tr)
        return m.predict_proba(X_te)[:, 1]
    med = np.nanmedian(X_tr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    a = np.where(np.isfinite(X_tr), X_tr, med)
    b = np.where(np.isfinite(X_te), X_te, med)
    mu, sd = a.mean(axis=0), a.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    m = LogisticRegression(C=0.1, max_iter=2000)
    m.fit((a - mu) / sd, y_tr)
    return m.predict_proba((b - mu) / sd)[:, 1]


def fold_metrics(pred, y, realised, bar_time):
    """Scores for one test window. `realised` is the continuous return the label was cut from."""
    base = float(y.mean())
    r = {"n": int(len(y)), "base_rate": base,
         "majority_acc": float(max(base, 1 - base)),
         "acc": float(((pred > 0.5).astype(int) == y).mean())}
    r["auc"] = float(roc_auc_score(y, pred)) if 0 < y.sum() < len(y) else float("nan")
    r["mean_realised"] = float(np.nanmean(realised))
    r["ic_pooled"] = float(stats.spearmanr(pred, realised).statistic)
    ics = []
    for _, g in pd.DataFrame({"p": pred, "r": realised, "t": bar_time}).groupby("t"):
        if len(g) >= 5 and g["p"].nunique() > 1:
            ics.append(stats.spearmanr(g["p"], g["r"]).statistic)
    r["ic_xs"] = float(np.nanmean(ics)) if ics else float("nan")
    if len(np.unique(pred)) >= 10:
        q = pd.qcut(pd.Series(pred).rank(method="first"), 10, labels=False)
        top, bot = realised[q == 9], realised[q == 0]
        r["decile_spread"] = float(np.mean(top) - np.mean(bot))
        r["top_decile_ret"] = float(np.mean(top))
        r["bot_decile_ret"] = float(np.mean(bot))
    else:
        r["decile_spread"] = r["top_decile_ret"] = r["bot_decile_ret"] = float("nan")
    return r


def run(rows, label_col, realised_col, kind="gbm", shuffle_seed=None, feats=FEATS, naive=None):
    """Walk-forward over `rows`. `naive` names a feature column used as the score, unfitted."""
    t = rows["bar_time"]
    X = rows[feats].replace([np.inf, -np.inf], np.nan).to_numpy(float)
    y = rows[label_col].to_numpy(int)
    realised = rows[realised_col].to_numpy(float)
    rng = np.random.default_rng(0 if shuffle_seed is None else shuffle_seed)
    per_fold = []
    for start, tr, te in folds_of(t):
        if tr.sum() < MIN_TRAIN_ROWS or te.sum() < 50:
            continue
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        if naive is not None:
            p = rows[naive].to_numpy(float)[te]
            ok = np.isfinite(p)
            if ok.sum() < 50:
                continue
            m = fold_metrics(p[ok], y[te][ok], realised[te][ok], t.to_numpy()[te][ok])
        else:
            y_tr = y[tr]
            if shuffle_seed is not None:
                y_tr = rng.permutation(y_tr)       # train on noise, score on the real test labels
                if len(np.unique(y_tr)) < 2:
                    continue
            p = fit_predict(X[tr], y_tr, X[te], kind)
            m = fold_metrics(p, y[te], realised[te], t.to_numpy()[te])
        m["test_start"] = str(pd.Timestamp(start).date())
        per_fold.append(m)
    return per_fold


# ---------------------------------------------------------------- aggregation

AGG_KEYS = ["auc", "acc", "majority_acc", "base_rate", "ic_pooled", "ic_xs", "decile_spread"]


def agg(per_fold):
    """Mean +- standard error ACROSS FOLDS. Folds are disjoint in time; rows are not."""
    out = {"folds": len(per_fold), "rows": int(sum(f["n"] for f in per_fold))}
    for k in AGG_KEYS:
        v = np.array([f[k] for f in per_fold], float)
        v = v[np.isfinite(v)]
        if not len(v):
            out[k] = {"mean": float("nan"), "se": float("nan"), "t": float("nan")}
            continue
        se = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
        ref = 0.5 if k in ("auc",) else 0.0
        out[k] = {"mean": float(v.mean()), "se": se,
                  "t": float((v.mean() - ref) / se) if se and se > 0 else float("nan"),
                  "folds_above_ref": int((v > ref).sum()) if k == "auc" else None}
    return out


def fmt(a, k, pct=False):
    d = a[k]
    s = 100 if pct else 1
    u = "%" if pct else ""
    return f"{d['mean'] * s:+.4f}{u} +- {d['se'] * s:.4f}{u} (t={d['t']:+.2f})"


def band(seed_aggs, k):
    """The shuffled null's own spread: mean of seed means, and the min/max seen."""
    v = np.array([a[k]["mean"] for a in seed_aggs], float)
    v = v[np.isfinite(v)]
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "min": float(v.min()),
            "max": float(v.max()), "n_seeds": int(len(v))}


def paired_vs_null(real_folds, null_runs, k="auc"):
    """The HONEST margin over the null: paired per fold, with the error bar taken across folds.

    The seed-to-seed spread of the shuffled null (`band`) is NOT a valid error bar for the real
    metric. Every seed scores against the same fixed test labels, so that spread measures only the
    randomness of the permutation - it is blind to the dominant source of uncertainty here, which is
    that some 91-day windows are simply easier than others. It is therefore far too small (0.006 vs
    an across-fold standard error of 0.012) and a z computed against it overstates significance by
    roughly 2x.

    This pairs fold-by-fold instead: for each test window, real metric minus the mean metric the
    shuffled models got on that SAME window, then t-tests those differences across folds. Pairing
    cancels the easy-window / hard-window variance that both sides share.
    """
    nulls = {}
    for run_folds in null_runs:
        for f in run_folds:
            nulls.setdefault(f["test_start"], []).append(f[k])
    diffs = []
    for f in real_folds:
        v = nulls.get(f["test_start"])
        if v and np.isfinite(f[k]):
            diffs.append(f[k] - float(np.nanmean(v)))
    d = np.array(diffs, float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return {"folds": int(len(d)), "mean": float("nan"), "se": float("nan"), "t": float("nan")}
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    t = float(d.mean() / se) if se > 0 else float("nan")
    return {"folds": int(len(d)), "mean": float(d.mean()), "se": se, "t": t,
            "p_two_sided": float(2 * stats.t.sf(abs(t), len(d) - 1)),
            "folds_positive": int((d > 0).sum())}


# ---------------------------------------------------------------- main

def main():
    t0 = time.time()
    s = config.load()
    u = json.loads((ROOT / "reports" / "_parity_universe.json").read_text())
    panel = add_forward(features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars))
    rows = build_rows(panel)

    rows["y_abs"] = (rows["fwd_ret"] > 0).astype(int)
    rows["y_rel"] = (rows["fwd_excess"] > 0).astype(int)
    rows["y_vol"] = (rows["fwd_vol"] > rows["fwd_vol_xs_med"]).astype(int)
    rel = rows[rows["symbol"] != "BTCUSDT"].reset_index(drop=True)
    volrows = rows.dropna(subset=["fwd_vol", "fwd_vol_xs_med"]).reset_index(drop=True)

    print(f"panel {len(panel):,} 4h rows -> {len(rows):,} daily decisions, "
          f"{rows['symbol'].nunique()} symbols, {rows['bar_time'].min().date()} .. "
          f"{rows['bar_time'].max().date()}")
    print(f"features ({len(FEATS)}): {', '.join(FEATS)}")
    print(f"walk-forward {TRAIN_DAYS}d train / {PURGE_DAYS}d purge / {TEST_DAYS}d test, "
          f"horizon {HORIZON_DAYS}d")
    print(f"base rate up (abs) {rows['y_abs'].mean():.4f}   beat BTC (rel) {rel['y_rel'].mean():.4f}")

    R = {"config": {"train_days": TRAIN_DAYS, "test_days": TEST_DAYS, "purge_days": PURGE_DAYS,
                    "horizon_days": HORIZON_DAYS, "grid": "daily 00:00 UTC bar, eligible only",
                    "features": FEATS, "symbols": sorted(rows["symbol"].unique().tolist()),
                    "rows_daily": int(len(rows)), "n_shuffle_seeds": N_SHUFFLE,
                    "base_rate_abs": float(rows["y_abs"].mean()),
                    "base_rate_rel": float(rel["y_rel"].mean())},
         "targets": {}}

    specs = [
        ("dir_abs", rows, "y_abs", "fwd_ret", {"momentum_30d": "ret_30d", "momentum_7d": "ret_7d",
                                               "trend_sma200": "dist_sma200"}),
        ("dir_rel", rel, "y_rel", "fwd_excess", {"momentum_30d": "rs_30d", "momentum_7d": "rs_7d",
                                                 "trend_sma200": "dist_sma200"}),
    ]

    RAW = {}
    for name, df, ycol, rcol, naives in specs:
        print("\n" + "=" * 78)
        print(f"TARGET {name}: sign of next {HORIZON_DAYS}d "
              f"{'return' if name == 'dir_abs' else 'return minus BTC'}   ({len(df):,} rows)")
        print("=" * 78)
        T = {}
        raw = {}
        for kind in ("gbm", "logreg"):
            raw[kind] = run(df, ycol, rcol, kind=kind)
            a = agg(raw[kind])
            T[kind] = a
            print(f"\n  {kind:7s} {a['folds']} folds, {a['rows']:,} test rows")
            print(f"    AUC             {fmt(a, 'auc')}      naive(base rate) = 0.5000")
            print(f"    accuracy        {a['acc']['mean']:.4f}  vs majority-class "
                  f"{a['majority_acc']['mean']:.4f}  "
                  f"(margin {(a['acc']['mean'] - a['majority_acc']['mean']) * 100:+.2f} pp)")
            print(f"    IC pooled       {fmt(a, 'ic_pooled')}")
            print(f"    IC x-sectional  {fmt(a, 'ic_xs')}")
            print(f"    decile spread   {fmt(a, 'decile_spread', pct=True)}  "
                  f"(top {a['decile_spread']['mean'] * 100:+.2f}%  = "
                  f"D10 - D1 of realised {HORIZON_DAYS}d return)")
            print(f"    folds AUC>0.5   {a['auc']['folds_above_ref']} / {a['folds']}")

        print("\n  NAIVE, unfitted single-column sorts (same folds, same rows):")
        T["naive"] = {}
        for nm, col in naives.items():
            a = agg(run(df, ycol, rcol, naive=col))
            T["naive"][nm] = a
            print(f"    {nm:13s} ({col:12s}) AUC {a['auc']['mean']:.4f} +- {a['auc']['se']:.4f}"
                  f"   IC_xs {a['ic_xs']['mean']:+.4f}"
                  f"   decile {a['decile_spread']['mean'] * 100:+.2f}%")

        print(f"\n  SHUFFLED NULL, {N_SHUFFLE} seeds (train labels permuted, real test labels):")
        null_raw = [run(df, ycol, rcol, kind="gbm", shuffle_seed=sd) for sd in range(1, N_SHUFFLE + 1)]
        seeds = [agg(r) for r in null_raw]
        T["shuffled"] = {k: band(seeds, k) for k in ("auc", "ic_pooled", "ic_xs", "decile_spread")}
        for k, lab in (("auc", "AUC"), ("ic_xs", "IC x-sec"), ("decile_spread", "decile")):
            b = T["shuffled"][k]
            sc = 100 if k == "decile_spread" else 1
            print(f"    {lab:9s} null mean {b['mean'] * sc:+.4f}  sd {b['sd'] * sc:.4f}  "
                  f"range [{b['min'] * sc:+.4f}, {b['max'] * sc:+.4f}]")

        # the comparison that decides the question
        real, null = T["gbm"], T["shuffled"]
        T["verdict"] = {}
        for k in ("auc", "ic_xs", "decile_spread"):
            z = ((real[k]["mean"] - null[k]["mean"]) / null[k]["sd"]) if null[k]["sd"] else float("nan")
            T["verdict"][k] = {"real": real[k]["mean"], "null_mean": null[k]["mean"],
                               "null_sd": null[k]["sd"], "margin": real[k]["mean"] - null[k]["mean"],
                               "z_vs_null": z, "inside_null_range":
                                   bool(null[k]["min"] <= real[k]["mean"] <= null[k]["max"])}
        print("\n  REAL vs SHUFFLED NULL  (this is the result):")
        for k in ("auc", "ic_xs", "decile_spread"):
            v = T["verdict"][k]
            sc = 100 if k == "decile_spread" else 1
            print(f"    {k:14s} real {v['real'] * sc:+.4f}  null {v['null_mean'] * sc:+.4f}  "
                  f"margin {v['margin'] * sc:+.4f}  z={v['z_vs_null']:+.2f}  "
                  f"{'INSIDE null range' if v['inside_null_range'] else 'outside null range'}")

        # The honest margin: paired per fold, error bar taken across folds (see paired_vs_null).
        T["paired_vs_null"] = {k: paired_vs_null(raw["gbm"], null_raw, k)
                               for k in ("auc", "ic_xs", "decile_spread")}
        print("\n  PAIRED real-minus-null, error bar ACROSS FOLDS (the honest test):")
        for k in ("auc", "ic_xs", "decile_spread"):
            p = T["paired_vs_null"][k]
            sc = 100 if k == "decile_spread" else 1
            print(f"    {k:14s} margin {p['mean'] * sc:+.4f} +- {p['se'] * sc:.4f}  "
                  f"t={p['t']:+.2f}  p={p['p_two_sided']:.3f}  "
                  f"{p['folds_positive']}/{p['folds']} folds positive  "
                  f"{'SIGNIFICANT' if p['p_two_sided'] < 0.05 else 'not significant'}")

        # overlap attack: one decision per symbol per 7 days, labels cannot share a bar
        wk = df[(df["bar_time"].astype("int64") // 86_400_000_000_000) % HORIZON_DAYS == 0]
        a = agg(run(wk.reset_index(drop=True), ycol, rcol, kind="gbm"))
        T["non_overlapping_weekly"] = a
        print(f"\n  NON-OVERLAPPING weekly grid ({a['rows']:,} rows, {a['folds']} folds): "
              f"AUC {fmt(a, 'auc')}  IC_xs {fmt(a, 'ic_xs')}  "
              f"decile {a['decile_spread']['mean'] * 100:+.2f}%")

        # How many INDEPENDENT observations are really here? If every coin on a bar carries the same
        # label, the row count is fiction and the signal is one market-timing call, not 37k calls.
        agree = df.groupby("bar_time")[ycol].mean()
        herd = float(np.mean(np.maximum(agree, 1 - agree)))
        T["label_clustering"] = {"mean_share_of_coins_agreeing_on_a_bar": herd,
                                 "distinct_bars": int(df["bar_time"].nunique()),
                                 "rows": int(len(df)),
                                 "rows_per_bar": float(len(df) / df["bar_time"].nunique())}
        print(f"\n  LABEL CLUSTERING: on an average bar {herd * 100:.1f}% of coins carry the SAME "
              f"label\n    ({df['bar_time'].nunique():,} distinct bars, "
              f"{len(df) / df['bar_time'].nunique():.1f} coins per bar) -> the effective sample is "
              f"nearer the\n    bar count than the row count, and nearer the fold count once 7-day "
              f"overlap is charged.")

        # ABLATION: is the signal market timing (BTC columns) or anything coin-specific?
        btc_only = [c for c in FEATS if c.startswith("btc_")]
        coin_only = [c for c in FEATS if not c.startswith("btc_")]
        T["ablation"] = {}
        print("\n  FEATURE ABLATION (gbm, same folds):")
        for lab, fs in (("btc_only", btc_only), ("coin_only", coin_only)):
            aa = agg(run(df, ycol, rcol, kind="gbm", feats=fs))
            T["ablation"][lab] = {"features": fs, **aa}
            print(f"    {lab:10s} ({len(fs)} feats) AUC {aa['auc']['mean']:.4f} "
                  f"+- {aa['auc']['se']:.4f} (t={aa['auc']['t']:+.2f})   "
                  f"IC_xs {aa['ic_xs']['mean']:+.4f}   "
                  f"decile {aa['decile_spread']['mean'] * 100:+.2f}%")
        R["targets"][name] = T
        RAW[name] = raw

    # ------------------------------------------------ suspicion audit
    # Two numbers above look like signal and must be attacked, not banked:
    #   dir_abs coin_only AUC 0.5357 with t=+4.71, and the weekly non-overlapping AUC 0.5532.
    # A result this project is willing to believe has to survive being hunted.
    print("\n" + "=" * 78)
    print("SUSPICION AUDIT on dir_abs - hunting the two numbers that looked like skill")
    print("=" * 78)
    coin_only = [c for c in FEATS if not c.startswith("btc_")]
    A = {}

    # A1. The weekly grid has 7 possible phases (which weekday it samples). If the headline 0.5532
    # only shows up at one phase, it is a sampling artifact, not a non-overlapping confirmation.
    day = rows["bar_time"].astype("int64") // 86_400_000_000_000
    ph = []
    for k in range(HORIZON_DAYS):
        wk = rows[day % HORIZON_DAYS == k].reset_index(drop=True)
        a = agg(run(wk, "y_abs", "fwd_ret", kind="gbm"))
        ph.append({"phase": k, "auc": a["auc"]["mean"], "se": a["auc"]["se"],
                   "decile": a["decile_spread"]["mean"], "rows": a["rows"]})
    v = np.array([p["auc"] for p in ph], float)
    A["weekly_phase_sweep"] = {"phases": ph, "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                               "min": float(v.min()), "max": float(v.max())}
    print("\n  A1. weekly non-overlapping grid at all 7 sampling phases:")
    print("      " + "  ".join(f"p{p['phase']}={p['auc']:.4f}" for p in ph))
    print(f"      mean {v.mean():.4f}  sd {v.std(ddof=1):.4f}  spread [{v.min():.4f}, {v.max():.4f}]"
          f"  <- a real effect would not move {(v.max() - v.min()):.4f} AUC with the weekday")

    # A2. coin_only vs its OWN shuffled null, paired across folds. The raw t=+4.71 is measured
    # against 0.5, which assumes a model with no signal scores exactly 0.5 - it does not.
    co_raw = run(rows, "y_abs", "fwd_ret", kind="gbm", feats=coin_only)
    co_null = [run(rows, "y_abs", "fwd_ret", kind="gbm", feats=coin_only, shuffle_seed=sd)
               for sd in range(1, N_SHUFFLE + 1)]
    co_agg, co_nagg = agg(co_raw), [agg(r) for r in co_null]
    A["coin_only_paired"] = {k: paired_vs_null(co_raw, co_null, k)
                             for k in ("auc", "ic_xs", "decile_spread")}
    A["coin_only"] = co_agg
    A["coin_only_null"] = {k: band(co_nagg, k) for k in ("auc", "ic_xs", "decile_spread")}
    print(f"\n  A2. coin_only (14 feats) AUC {co_agg['auc']['mean']:.4f} +- {co_agg['auc']['se']:.4f}"
          f"   its shuffled null {A['coin_only_null']['auc']['mean']:.4f}")
    for k in ("auc", "ic_xs", "decile_spread"):
        p = A["coin_only_paired"][k]
        sc = 100 if k == "decile_spread" else 1
        print(f"      paired {k:14s} {p['mean'] * sc:+.4f} +- {p['se'] * sc:.4f}  t={p['t']:+.2f}  "
              f"p={p['p_two_sided']:.3f}  {p['folds_positive']}/{p['folds']} folds  "
              f"{'SIGNIFICANT' if p['p_two_sided'] < 0.05 else 'not significant'}")

    # A3. Is the AUC just riding the market? If fold AUC tracks how much the market rose in that
    # fold, the model is not ranking coins, it is being graded by the weather.
    fa = np.array([f["auc"] for f in RAW["dir_abs"]["gbm"]], float)
    fm = np.array([f["mean_realised"] for f in RAW["dir_abs"]["gbm"]], float)
    ok = np.isfinite(fa) & np.isfinite(fm)
    rho = float(stats.spearmanr(fa[ok], fm[ok]).statistic)
    up = fm > 0
    A["fold_auc_vs_market"] = {
        "spearman_fold_auc_vs_fold_mean_return": rho,
        "auc_in_up_folds": float(np.nanmean(fa[ok & up])), "n_up": int((ok & up).sum()),
        "auc_in_down_folds": float(np.nanmean(fa[ok & ~up])), "n_down": int((ok & ~up).sum())}
    print(f"\n  A3. fold AUC vs that fold's mean 7d return: Spearman {rho:+.3f}")
    print(f"      AUC in rising folds {np.nanmean(fa[ok & up]):.4f} (n={(ok & up).sum()})   "
          f"in falling folds {np.nanmean(fa[ok & ~up]):.4f} (n={(ok & ~up).sum()})")

    # A4. The only metric that pays. AUC is free; a tradeable long-short decile is not.
    ds = np.array([f["decile_spread"] for f in RAW["dir_abs"]["gbm"]], float)
    ds = ds[np.isfinite(ds)]
    A["tradeable"] = {"decile_spread_mean": float(ds.mean()),
                      "decile_spread_se": float(ds.std(ddof=1) / np.sqrt(len(ds))),
                      "folds_positive": int((ds > 0).sum()), "folds": int(len(ds)),
                      "round_trip_cost_pct": 0.44,
                      "net_of_cost": float(ds.mean() * 100 - 0.44)}
    print(f"\n  A4. tradeability: D10-D1 {ds.mean() * 100:+.2f}% per 7d, "
          f"{(ds > 0).sum()}/{len(ds)} folds positive, 0.44% round-trip cost "
          f"(0.10% fee + 7bps slip, both sides) -> net {ds.mean() * 100 - 0.44:+.2f}%")
    # A5. THE ACTUAL FINDING, found by accident. Every naive momentum/trend sort came back with AUC
    # BELOW 0.5 (0.4511 to 0.4859) - not noise around 0.5, consistently and significantly inverted.
    # An inverted sort is a signal with the sign flipped: 7-day SHORT-TERM REVERSAL. This is the one
    # effect in this file that clears its error bar, and it is unfitted, so there is no model to
    # overfit and no shuffled null to beat - the null for a fixed single-column sort is exactly 0.5.
    # Tested here across folds, and across the weekday phase that wrecked A1.
    print("\n  A5. the naive sorts are INVERTED - unfitted single columns, null is exactly 0.5:")
    A["reversal"] = {}
    for col in ("dist_sma200", "dist_sma50", "ret_7d", "ret_30d", "don_pos", "vol_30d"):
        a = agg(run(rows, "y_abs", "fwd_ret", naive=col))
        au, se = a["auc"]["mean"], a["auc"]["se"]
        ph = []
        for k in range(HORIZON_DAYS):
            wk = rows[day % HORIZON_DAYS == k].reset_index(drop=True)
            ph.append(agg(run(wk, "y_abs", "fwd_ret", naive=col))["auc"]["mean"])
        pv = np.array(ph, float)
        A["reversal"][col] = {"auc": au, "se": se, "t_vs_half": a["auc"]["t"],
                              "flipped_auc": 1 - au,
                              "decile_spread": a["decile_spread"]["mean"],
                              "weekly_phase_auc": [float(x) for x in pv],
                              "weekly_phase_mean": float(pv.mean()),
                              "weekly_phase_sd": float(pv.std(ddof=1)),
                              "all_phases_same_side": bool((pv < 0.5).all() or (pv > 0.5).all())}
        print(f"      {col:12s} AUC {au:.4f} +- {se:.4f} (t={a['auc']['t']:+.2f} vs 0.5)  "
              f"flipped {1 - au:.4f}  D10-D1 {a['decile_spread']['mean'] * 100:+.2f}%  "
              f"weekly phases {pv.mean():.4f}+-{pv.std(ddof=1):.4f} "
              f"{'ALL same side' if (pv < 0.5).all() or (pv > 0.5).all() else 'SIGN FLIPS by weekday'}")
    print("      A long-only SPOT bot cannot harvest a D10-D1 spread: it needs the short leg.")
    R["suspicion_audit_dir_abs"] = A

    # ------------------------------------------------ positive control
    print("\n" + "=" * 78)
    print("POSITIVE CONTROL: is next-7d realised vol above the cross-sectional median?")
    print("(same features, same folds, same code path - proves the harness CAN see signal)")
    print("=" * 78)
    a = agg(run(volrows, "y_vol", "fwd_vol", kind="gbm"))
    nv = agg(run(volrows, "y_vol", "fwd_vol", naive="vol_30d"))
    sh = [agg(run(volrows, "y_vol", "fwd_vol", kind="gbm", shuffle_seed=sd)) for sd in range(1, 4)]
    R["positive_control_vol"] = {"gbm": a, "naive_vol_30d": nv,
                                 "shuffled": {k: band(sh, k) for k in ("auc", "ic_xs")}}
    print(f"  gbm           AUC {fmt(a, 'auc')}   acc {a['acc']['mean']:.4f} vs majority "
          f"{a['majority_acc']['mean']:.4f}   IC_xs {fmt(a, 'ic_xs')}")
    print(f"  naive vol_30d AUC {nv['auc']['mean']:.4f} +- {nv['auc']['se']:.4f}"
          f"   IC_xs {nv['ic_xs']['mean']:+.4f}")
    print(f"  shuffled null AUC {R['positive_control_vol']['shuffled']['auc']['mean']:.4f}")
    print(f"  margin gbm over naive: {(a['auc']['mean'] - nv['auc']['mean']):+.4f} AUC")

    # ------------------------------------------------ conclusion
    ab, rl = R["targets"]["dir_abs"], R["targets"]["dir_rel"]
    pc = R["positive_control_vol"]
    honest = pc["gbm"]["auc"]["mean"] > 0.60 and abs(
        pc["shuffled"]["auc"]["mean"] - 0.5) < 0.03
    skill = []
    for nm, T in (("dir_abs", ab), ("dir_rel", rl)):
        # Every gate must hold: the paired across-fold margin significant, the raw AUC itself
        # significant against 0.5, and the coin-specific features carrying some of it rather than
        # the whole thing being one BTC-regime market-timing call.
        if (T["paired_vs_null"]["auc"]["p_two_sided"] < 0.05
                and T["paired_vs_null"]["auc"]["mean"] > 0.01
                and T["gbm"]["auc"]["t"] > 2.0
                and T["ablation"]["coin_only"]["auc"]["mean"] > 0.52):
            skill.append(nm)
    R["conclusion"] = {
        "harness_validated": bool(honest),
        "direction_skill_found": skill,
        "dir_abs_auc": ab["gbm"]["auc"]["mean"], "dir_abs_auc_se": ab["gbm"]["auc"]["se"],
        "dir_rel_auc": rl["gbm"]["auc"]["mean"], "dir_rel_auc_se": rl["gbm"]["auc"]["se"],
        "dir_abs_margin_over_null_auc": ab["verdict"]["auc"]["margin"],
        "dir_rel_margin_over_null_auc": rl["verdict"]["auc"]["margin"],
        "dir_abs_paired_margin_auc": ab["paired_vs_null"]["auc"],
        "dir_rel_paired_margin_auc": rl["paired_vs_null"]["auc"],
        "positive_control_auc": pc["gbm"]["auc"]["mean"],
        "positive_control_naive_auc": pc["naive_vol_30d"]["auc"]["mean"],
        "positive_control_gbm_minus_naive_auc": (pc["gbm"]["auc"]["mean"]
                                                 - pc["naive_vol_30d"]["auc"]["mean"]),
    }
    print("\n" + "=" * 78)
    print("CONCLUSION")
    print("=" * 78)
    print(f"  harness validated   {'YES' if honest else 'NO'}  "
          f"(vol control AUC {pc['gbm']['auc']['mean']:.4f}, its shuffled null "
          f"{pc['shuffled']['auc']['mean']:.4f})")
    for nm, T in (("dir_abs", ab), ("dir_rel", rl)):
        p = T["paired_vs_null"]["auc"]
        print(f"  {nm}  AUC {T['gbm']['auc']['mean']:.4f} +- {T['gbm']['auc']['se']:.4f} "
              f"(t={T['gbm']['auc']['t']:+.2f} vs 0.5)")
        print(f"           paired margin over null {p['mean']:+.4f} +- {p['se']:.4f} "
              f"(t={p['t']:+.2f}, p={p['p_two_sided']:.3f})")
        print(f"           coin-only features AUC {T['ablation']['coin_only']['auc']['mean']:.4f}, "
              f"btc-only AUC {T['ablation']['btc_only']['auc']['mean']:.4f}, "
              f"weekly non-overlapping AUC {T['non_overlapping_weekly']['auc']['mean']:.4f}")
        print(f"           accuracy {T['gbm']['acc']['mean']:.4f} vs always-up "
              f"{T['gbm']['majority_acc']['mean']:.4f} "
              f"({(T['gbm']['acc']['mean'] - T['gbm']['majority_acc']['mean']) * 100:+.2f} pp)")
    print(f"  direction skill     {skill if skill else 'NONE - direction is not predictable here'}")

    (ROOT / "reports" / "forecast_direction.json").write_text(json.dumps(R, indent=2, default=str))
    print(f"\nwrote reports/forecast_direction.json  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
