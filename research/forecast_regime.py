"""Is BTC REGIME PERSISTENCE predictable? Walk-forward, measured against persistence itself.

The bot's single biggest decision is binary: be in the market, or sit in cash. It flips when BTC
crosses its 200-day SMA, which means it always reacts AFTER the cross. If the state 7 or 30 days
ahead were forecastable, the bot could anticipate instead of react.

This script does NOT try to predict returns. That question is closed in this repo (ranker, ML veto,
win-probability model, right-tail classifier - all rejected). The question here is narrower:

    P(regime_on at t+h) given only BTC information closed at or before t.

The only number that matters is the MARGIN over the naive baselines:
  (a) PERSISTENCE - "the state in h days is the state now". Brutally strong here: the regime is a
      slow-moving trend filter, so persistence is right most of the time by construction. It has one
      weakness and only one: it can never, ever call a flip. Recall on flips is exactly zero.
  (b) BASE RATE - the training-window frequency of regime_on. AUC 0.50 by construction.
  (c) ALWAYS-ON (the degenerate "no flip ever" predictor) - checked explicitly, because a model that
      collapses to it will post a flattering accuracy and be worth nothing.
  (d) DIST_SMA200 ALONE and (e) DIST_SMA200 / VOL_30D ALONE - one free number, no model. This turned
      out to be the whole story. regime_on IS "dist_sma200 > 0", so forecasting the regime is mostly
      asking how near price already sits to the line that defines it, in units of how fast it moves.
      That is arithmetic the bot already has. The GBM loses to it at both horizons on every metric.

A DIRECTION control runs on the same folds, same features: predict the sign of BTC's next-7d return.
Honest answer there is AUC ~0.50. If the harness returns much more than that, the harness is broken,
not the market.

Causality. Features at bar t use bars closed at or before t (features.symbol_features guarantees
this; the daily SMAs are joined as-of backward). Labels look forward by a fixed bar count. Train and
test are separated by a purge gap equal to the horizon, so no training label overlaps a test bar.
Standardisation uses training-window statistics only.

Honesty about sample size. 4h bars with a 180-bar horizon are enormously overlapping: ~13,000 OOS
rows are not 13,000 observations. The real unit is the FLIP EPISODE, and nine years of BTC contains
only a few dozen of them (fewer still if you ignore one-day whipsaws). Every headline number here is
reported alongside the episode count, and the AUC margin gets a moving-block bootstrap so the
reader can see how wide the uncertainty really is.

    cd crypto-bot && PYTHONIOENCODING=utf-8 python research/forecast_regime.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, features  # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "forecast_regime.json"
BPD = 6                      # 4h bars per day
TRAIN_DAYS = 730
TEST_DAYS = 91
HORIZONS = {"7d": 7 * BPD, "30d": 30 * BPD}      # 42 and 180 bars
MIN_TRAIN_ROWS = 2000
SEED = 0


# ------------------------------------------------------------------ features

def btc_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per 4h bar of BTC, with the causal feature set from the brief."""
    b = (panel[panel["symbol"] == "BTCUSDT"]
         .sort_values("bar_time")
         .set_index("bar_time"))
    b = b[b["dist_sma200"].notna()].copy()       # regime undefined before 200 daily bars exist

    d = b["dist_sma200"]
    b["state"] = d > 0                           # == features.regime_on, recomputed locally
    # Slope of the trend filter: how fast BTC is approaching or leaving its 200-day line.
    b["slope_sma200_7d"] = d - d.shift(7 * BPD)
    b["slope_sma200_30d"] = d - d.shift(30 * BPD)
    b["ret_90d"] = b["close"].pct_change(90 * BPD)
    b["vol_ratio"] = b["vol_7d"] / b["vol_30d"]  # is volatility expanding right now?
    # Distance to the 200-day line measured in volatility units: "how many sigmas to the flip".
    b["dist_over_vol"] = b["dist_sma200"] / b["vol_30d"]
    # Drawdown from the running high: expanding (all history) and a rolling 1-year version, which is
    # the stationary one. Both use past bars only.
    b["dd_all"] = b["close"] / b["close"].cummax() - 1
    b["dd_1y"] = b["close"] / b["close"].rolling(365 * BPD, min_periods=30 * BPD).max() - 1
    # Bars since the last flip of the state (past-only: a run length never sees the future).
    run = b["state"].ne(b["state"].shift(1)).cumsum()
    b["days_since_flip"] = b.groupby(run).cumcount() / BPD
    return b


FEATS = ["dist_sma200", "dist_over_vol", "slope_sma200_7d", "slope_sma200_30d",
         "ret_7d", "ret_30d", "ret_90d",
         "vol_7d", "vol_30d", "vol_ratio",
         "dd_all", "dd_1y", "volume_ratio", "taker_ratio", "days_since_flip",
         "state_f"]


def design(b: pd.DataFrame) -> pd.DataFrame:
    X = b.reindex(columns=[c for c in FEATS if c != "state_f"]).copy()
    X["state_f"] = b["state"].astype(float)      # the model is allowed to know the current state
    return X.replace([np.inf, -np.inf], np.nan)


# ------------------------------------------------------------------ metrics

def brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def auc(y, p):
    y = np.asarray(y, int)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, p))


def acc(y, p, thr=0.5):
    return float(np.mean((np.asarray(p) >= thr).astype(int) == np.asarray(y, int)))


def confusion(y, p, thr=0.5):
    y = np.asarray(y, int)
    yh = (np.asarray(p) >= thr).astype(int)
    return {"tp": int(((y == 1) & (yh == 1)).sum()), "fp": int(((y == 0) & (yh == 1)).sum()),
            "fn": int(((y == 1) & (yh == 0)).sum()), "tn": int(((y == 0) & (yh == 0)).sum())}


def score_block(y, p, thr=0.5):
    return {"auc": auc(y, p), "accuracy": acc(y, p, thr), "brier": brier(y, p),
            "confusion": confusion(y, p, thr)}


# ------------------------------------------------------------------ walk-forward

def models():
    return {
        "gbm": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=7, min_samples_leaf=200,
            l2_regularization=1.0, random_state=SEED),
        "logit": None,       # built per fold (needs train-window standardisation)
    }


def fit_logit(Xtr, ytr, Xte, cols=None):
    if cols is not None:
        Xtr, Xte = Xtr[cols], Xte[cols]
    med = Xtr.median()
    mu, sd = Xtr.mean(), Xtr.std().replace(0, 1.0)
    A = ((Xtr.fillna(med) - mu) / sd).values
    B = ((Xte.fillna(med) - mu) / sd).values
    m = LogisticRegression(C=1.0, max_iter=2000)
    m.fit(A, ytr)
    return m.predict_proba(B)[:, 1]


def walk_forward(X, y, state, horizon, label, log=print):
    """Pooled out-of-sample predictions, plus the per-fold scores. One row per test bar."""
    t = X.index
    purge = pd.Timedelta(hours=4 * (horizon + 1))
    rows, folds = [], []
    test_start = t.min() + pd.Timedelta(days=TRAIN_DAYS)
    while test_start < t.max():
        test_end = test_start + pd.Timedelta(days=TEST_DAYS)
        tr = (t >= test_start - pd.Timedelta(days=TRAIN_DAYS)) & (t < test_start - purge) & y.notna()
        te = (t >= test_start) & (t < test_end) & y.notna()
        if tr.sum() >= MIN_TRAIN_ROWS and te.sum() > 0 and y[tr].nunique() > 1:
            Xtr, ytr = X[tr], y[tr].astype(int)
            Xte = X[te]
            gbm = models()["gbm"]
            gbm.fit(Xtr, ytr)
            p_gbm = gbm.predict_proba(Xte)[:, 1]
            p_log = fit_logit(Xtr, ytr, Xte)
            # Baseline (d): the ONE number the bot already has, free, with no model - how far price
            # is from the 200-day line. regime_on IS dist_sma200 > 0, so a bar sitting 0.3% above the
            # line is obviously more flip-prone than one sitting 30% above it. That is arithmetic,
            # not forecasting. If this single feature matches the GBM, the GBM has added nothing.
            p_d1 = fit_logit(Xtr, ytr, Xte, ["dist_sma200"])
            # (e) the same distance expressed in volatility units: how many sigmas to the line.
            p_d2 = fit_logit(Xtr, ytr, Xte, ["dist_over_vol"])

            # Naive baselines, all estimated on the TRAINING window only.
            base = float(ytr.mean())                               # (b) base rate
            # (a2) calibrated persistence: P(y=1 | state) from the training window
            cal = {}
            for s in (False, True):
                m = state[tr] == s
                cal[s] = float(ytr[m].mean()) if m.sum() > 30 else base
            st_te = state[te].astype(bool).values
            p_pers_hard = st_te.astype(float)                      # (a1) hard persistence 0/1
            p_pers_cal = np.where(st_te, cal[True], cal[False])
            p_base = np.full(te.sum(), base)
            p_always_on = np.ones(te.sum())                        # (c) degenerate no-flip-ever

            f = pd.DataFrame({"y": y[te].astype(int).values, "state": st_te,
                              "gbm": p_gbm, "logit": p_log, "dist_only": p_d1,
                              "dist_over_vol": p_d2,
                              "pers_hard": p_pers_hard, "pers_cal": p_pers_cal,
                              "base": p_base, "always_on": p_always_on},
                             index=t[te])
            f["fold"] = str(test_start.date())
            rows.append(f)
            folds.append({"fold": str(test_start.date()), "train_rows": int(tr.sum()),
                          "test_rows": int(te.sum()), "train_base_rate": base,
                          "test_base_rate": float(f["y"].mean()),
                          "auc_gbm": auc(f["y"], f["gbm"]),
                          "auc_pers": auc(f["y"], f["pers_hard"]),
                          "brier_gbm": brier(f["y"], f["gbm"]),
                          "brier_pers_cal": brier(f["y"], f["pers_cal"])})
            log(f"  [{label}] fold {test_start.date()} train={int(tr.sum())} test={int(te.sum())} "
                f"base={base:.3f} auc_gbm={folds[-1]['auc_gbm']:.3f} "
                f"auc_pers={folds[-1]['auc_pers']:.3f}")
        test_start = test_end
    oos = pd.concat(rows).sort_index() if rows else pd.DataFrame()
    return oos, folds


# ------------------------------------------------------------------ flip analysis

def flip_episodes(state: pd.Series, lo=None, hi=None):
    """Contiguous runs of the state. Returns one record per transition."""
    s = state.astype(bool)
    chg = s.ne(s.shift(1))
    chg.iloc[0] = False
    out = []
    for ts in s.index[chg]:
        i = s.index.get_loc(ts)
        run_end = s.index[i:]
        # length of the run that STARTS at ts
        nxt = s.iloc[i:].ne(s.iloc[i])
        n = int(nxt.values.argmax()) if nxt.any() else len(nxt)
        out.append({"time": ts, "to": bool(s.iloc[i]), "run_bars": n, "run_days": n / BPD})
    df = pd.DataFrame(out)
    if len(df) and lo is not None:
        df = df[(df["time"] >= lo) & (df["time"] <= hi)]
    return df.reset_index(drop=True)


def anticipation(oos: pd.DataFrame, state_full: pd.Series, horizon: int, col="gbm", thr=0.5):
    """Of the real flips inside the OOS span, how many did the model warn about BEFORE they
    happened, and what did the warnings cost in false alarms?

    A warning on an ON bar is p(regime_on at t+h) < thr. Persistence can NEVER issue one.
    """
    if oos.empty:
        return {}
    eps = flip_episodes(state_full, oos.index.min(), oos.index.max())
    on_off = eps[~eps["to"]] if len(eps) else eps          # transitions INTO off = the risk event
    off_on = eps[eps["to"]] if len(eps) else eps
    h = pd.Timedelta(hours=4 * horizon)

    caught = []
    for _, e in on_off.iterrows():
        w = oos[(oos.index >= e["time"] - h) & (oos.index < e["time"]) & oos["state"]]
        if w.empty:
            continue
        warned = w[col] < thr
        caught.append({"flip": str(e["time"]), "run_days_after": round(float(e["run_days"]), 1),
                       "bars_in_window": int(len(w)), "warned_bars": int(warned.sum()),
                       "warned_frac": float(warned.mean()),
                       "first_warning_days_before": (
                           float((e["time"] - w.index[warned][0]).total_seconds() / 86400)
                           if warned.any() else None)})
    rec = [c for c in caught if c["warned_bars"] > 0]

    # The cost side: on ON bars, a warning that did NOT precede a flip inside the horizon.
    on = oos[oos["state"]]
    warn_on = on[col] < thr
    false_alarm = (warn_on & (on["y"] == 1)).sum()
    true_alarm = (warn_on & (on["y"] == 0)).sum()

    # THE NULL for "it saw the flip coming". If the model warns on r% of ON bars, then a window of n
    # bars contains at least one warning with probability 1-(1-r)^n by pure chance. With r around
    # 10% and n = 42, that is ~99%: "warned before 42 of 48 flips" can be WORSE than random. Compare
    # the episode count to this expectation, and the warning density inside the pre-flip windows to
    # the density everywhere else.
    r = float(warn_on.mean()) if len(on) else 0.0
    exp_eps = sum(1 - (1 - r) ** c["bars_in_window"] for c in caught)
    dens = float(np.mean([c["warned_frac"] for c in caught])) if caught else float("nan")

    # Symmetric side: OFF bars, does the model see the recovery coming?
    off = oos[~oos["state"]]
    call_on = off[col] >= thr
    off_rec = {"off_bars": int(len(off)),
               "off_bars_that_do_flip_on": int((off["y"] == 1).sum()),
               "model_called_on": int(call_on.sum()),
               "recall_on_flips": float((call_on & (off["y"] == 1)).mean() * len(off)
                                        / max(1, (off["y"] == 1).sum())),
               "precision_on_flips": float((call_on & (off["y"] == 1)).sum() / max(1, call_on.sum())),
               "episodes_off_to_on": int(len(off_on))}

    return {
        "threshold": thr,
        "flip_episodes_on_to_off_in_oos": int(len(on_off)),
        "flip_episodes_on_to_off_lasting_10d_plus": int((on_off["run_days"] >= 10).sum())
        if len(on_off) else 0,
        "episodes_with_a_prior_warning": len(rec),
        "episode_recall": round(len(rec) / max(1, len(caught)), 3),
        "episodes_examined": len(caught),
        "warn_rate_on_all_on_bars": r,
        "expected_episodes_warned_if_random": round(float(exp_eps), 2),
        "episode_recall_lift_vs_random": round(len(rec) / exp_eps, 3) if exp_eps > 0 else None,
        "warn_density_inside_preflip_windows": dens,
        "warn_density_lift": round(dens / r, 3) if r > 0 and np.isfinite(dens) else None,
        "on_bars": int(len(on)),
        "on_bars_warned": int(warn_on.sum()),
        "warning_true": int(true_alarm), "warning_false": int(false_alarm),
        "warning_precision": float(true_alarm / max(1, warn_on.sum())),
        "warning_recall_bars": float(true_alarm / max(1, int((on["y"] == 0).sum()))),
        "per_episode": caught[:40],
        "off_side": off_rec,
    }


def conditional(oos: pd.DataFrame):
    """The question as the brief actually poses it, split by the current state.

    This is the sharpest test available. INSIDE one state, persistence is a CONSTANT - it says
    "stays on" for every ON bar - so its AUC is undefined/0.50 and its Brier is just the state's
    climatology. Any ranking ability the model has inside a state is ability persistence cannot
    supply. If conditional AUC is ~0.50, the pooled AUC of ~0.93 was nothing but the model
    rediscovering the state feature, and the model is worthless.
    """
    out = {}
    for key, m in (("regime_on_now", oos["state"]), ("regime_off_now", ~oos["state"])):
        d = oos[m]
        if len(d) < 50 or d["y"].nunique() < 2:
            out[key] = {"n": int(len(d)), "note": "not enough variation"}
            continue
        clim = float(d["y"].mean())
        out[key] = {
            "n": int(len(d)),
            "share_that_flips_by_t_plus_h": float(1 - clim) if key == "regime_on_now" else float(clim),
            "auc_gbm": auc(d["y"], d["gbm"]), "auc_logit": auc(d["y"], d["logit"]),
            "auc_dist_only": auc(d["y"], d["dist_only"]),
            "auc_dist_over_vol": auc(d["y"], d["dist_over_vol"]),
            "auc_persistence": 0.5,
            "margin_auc_over_dist_over_vol":
                auc(d["y"], d["gbm"]) - auc(d["y"], d["dist_over_vol"]),
            "brier_gbm": brier(d["y"], d["gbm"]),
            "brier_dist_over_vol": brier(d["y"], d["dist_over_vol"]),
            "brier_persistence_cal": brier(d["y"], d["pers_cal"]),
            "brier_within_state_climatology": brier(d["y"], np.full(len(d), clim)),
            "margin_auc_over_persistence": auc(d["y"], d["gbm"]) - 0.5,
        }
    return out


def deciles(oos: pd.DataFrame):
    """Among ON bars, the realised flip rate by decile of "sigmas to the line". If this is cleanly
    monotone, the predictability is mechanical: the only thing being forecast is how near the price
    already is to the threshold that DEFINES the regime. No model is needed to know that."""
    on = oos[oos["state"]].copy()
    if len(on) < 200:
        return {}
    on["q"] = pd.qcut(on["dist_over_vol"], 10, labels=False, duplicates="drop")
    g = on.groupby("q")
    t = pd.DataFrame({"n": g.size(), "flip_rate": 1 - g["y"].mean(),
                      "p_gbm_mean": g["gbm"].mean()})
    return {"flip_rate_by_decile_of_sigmas_to_line": [
        {"decile": int(i), "n": int(r["n"]), "realised_flip_rate": round(float(r["flip_rate"]), 4)}
        for i, r in t.iterrows()]}


def economics(oos: pd.DataFrame, b: pd.DataFrame, horizon: int, col="gbm", thr=0.5):
    """Cheap, engine-free read on whether the warnings are worth money: BTC's realised forward
    return over the horizon, split by what the model said, on ON bars only."""
    fwd = (b["close"].shift(-horizon) / b["close"] - 1).reindex(oos.index)
    on = oos[oos["state"]].copy()
    on["fwd"] = fwd.reindex(on.index)
    warn = on[col] < thr
    f = lambda m: (float(on.loc[m, "fwd"].mean()) if m.sum() else float("nan"), int(m.sum()))
    w_mean, w_n = f(warn)
    k_mean, k_n = f(~warn)
    med = lambda m: float(on.loc[m, "fwd"].median()) if m.sum() else float("nan")
    neg = lambda m: float((on.loc[m, "fwd"] < 0).mean()) if m.sum() else float("nan")
    return {"fwd_ret_when_warned": w_mean, "bars_warned": w_n,
            "fwd_ret_when_clear": k_mean, "bars_clear": k_n,
            "median_fwd_when_warned": med(warn), "median_fwd_when_clear": med(~warn),
            "share_negative_when_warned": neg(warn), "share_negative_when_clear": neg(~warn),
            "gap": (w_mean - k_mean) if np.isfinite(w_mean) and np.isfinite(k_mean) else None,
            "reading": ("warnings precede HIGHER returns - acting on them costs money"
                        if np.isfinite(w_mean) and np.isfinite(k_mean) and w_mean > k_mean
                        else "warnings precede lower returns")}


def block_bootstrap_auc(oos, a="gbm", b="pers_hard", block_days=90, n=500, seed=SEED):
    """Moving-block bootstrap of AUC(a) - AUC(b). Blocks respect the autocorrelation that makes
    the raw row count a lie."""
    if oos.empty:
        return {}
    rng = np.random.default_rng(seed)
    key = oos.index.floor("D").to_period("D").astype("int64") // block_days
    groups = [g for _, g in oos.groupby(key)]
    if len(groups) < 5:
        return {}
    d = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        s = pd.concat([groups[i] for i in pick])
        if s["y"].nunique() < 2:
            continue
        d.append(auc(s["y"], s[a]) - auc(s["y"], s[b]))
    d = np.array([x for x in d if np.isfinite(x)])
    if not len(d):
        return {}
    return {"blocks": len(groups), "resamples": int(len(d)), "mean_delta_auc": float(d.mean()),
            "p05": float(np.percentile(d, 5)), "p50": float(np.percentile(d, 50)),
            "p95": float(np.percentile(d, 95)),
            "share_positive": float((d > 0).mean())}


# ------------------------------------------------------------------ direction control

def direction_control(b, X, log=print):
    """Same features, same folds: can we call the SIGN of BTC's next 7 days? Expect AUC ~0.50."""
    h = 7 * BPD
    fwd = b["close"].shift(-h) / b["close"] - 1
    y = (fwd > 0).where(fwd.notna())
    oos, folds = walk_forward(X, y, b["state"], h, "direction", log=log)
    if oos.empty:
        return {}
    return {"n_oos_bars": int(len(oos)), "base_rate_up": float(oos["y"].mean()),
            "auc_gbm": auc(oos["y"], oos["gbm"]), "auc_logit": auc(oos["y"], oos["logit"]),
            "accuracy_gbm": acc(oos["y"], oos["gbm"]),
            "accuracy_base": float(max(oos["y"].mean(), 1 - oos["y"].mean())),
            "brier_gbm": brier(oos["y"], oos["gbm"]),
            "brier_base": brier(oos["y"], np.full(len(oos), oos["y"].mean())),
            "folds": len(folds)}


# ------------------------------------------------------------------ main

def main():
    s = config.load()
    u = json.loads((ROOT / "reports" / "_parity_universe.json").read_text())
    print("building panel ...")
    panel = features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars)
    b = btc_frame(panel)
    X = design(b)
    print(f"BTC bars with a defined regime: {len(b)}  {b.index.min().date()} -> {b.index.max().date()}")

    gaps = b.index.to_series().diff().dropna()
    bar_gaps = {"median_hours": float(gaps.median().total_seconds() / 3600),
                "bars_with_gap_over_4h": int((gaps > pd.Timedelta(hours=4)).sum()),
                "max_gap_hours": float(gaps.max().total_seconds() / 3600)}

    all_eps = flip_episodes(b["state"])
    print(f"regime flips in the whole sample: {len(all_eps)}  "
          f"(runs >= 10 days: {int((all_eps['run_days'] >= 10).sum())})")

    out = {"generated": pd.Timestamp.utcnow().isoformat(),
           "panel": {"symbols": len(u["symbols"]), "btc_bars": int(len(b)),
                     "start": str(b.index.min()), "end": str(b.index.max()),
                     "bar_gaps": bar_gaps},
           "method": {"train_days": TRAIN_DAYS, "test_days": TEST_DAYS,
                      "purge": "horizon + 1 bars", "features": FEATS,
                      "standardisation": "train-window mean/std (logit); trees need none"},
           "sample_truth": {
               "flip_episodes_all_sample": int(len(all_eps)),
               "flip_episodes_lasting_10d_plus": int((all_eps["run_days"] >= 10).sum()),
               "median_run_days": float(all_eps["run_days"].median()) if len(all_eps) else None,
               "note": "overlapping 4h bars; the independent unit is the episode, not the bar"},
           "horizons": {}}

    for name, h in HORIZONS.items():
        print(f"\n=== horizon {name} ({h} bars) ===")
        y = b["state"].shift(-h)
        y = y.where(y.notna())
        oos, folds = walk_forward(X, y, b["state"], h, name)
        if oos.empty:
            out["horizons"][name] = {"error": "no folds"}
            continue

        blocks = {k: score_block(oos["y"], oos[k]) for k in
                  ("gbm", "logit", "dist_only", "dist_over_vol",
                   "pers_hard", "pers_cal", "base", "always_on")}
        margin = {
            "auc_gbm_minus_persistence": blocks["gbm"]["auc"] - blocks["pers_hard"]["auc"],
            "auc_gbm_minus_baserate": blocks["gbm"]["auc"] - 0.5,
            "acc_gbm_minus_persistence": blocks["gbm"]["accuracy"] - blocks["pers_hard"]["accuracy"],
            "acc_gbm_minus_alwayson": blocks["gbm"]["accuracy"] - blocks["always_on"]["accuracy"],
            "brier_gbm_minus_persistence_cal": blocks["gbm"]["brier"] - blocks["pers_cal"]["brier"],
            "brier_gbm_minus_baserate": blocks["gbm"]["brier"] - blocks["base"]["brier"],
            "brier_skill_vs_persistence_cal":
                1 - blocks["gbm"]["brier"] / blocks["pers_cal"]["brier"],
            "brier_skill_vs_baserate": 1 - blocks["gbm"]["brier"] / blocks["base"]["brier"],
            "auc_gbm_minus_dist_only": blocks["gbm"]["auc"] - blocks["dist_only"]["auc"],
            "auc_gbm_minus_dist_over_vol": blocks["gbm"]["auc"] - blocks["dist_over_vol"]["auc"],
            "brier_gbm_minus_dist_over_vol":
                blocks["gbm"]["brier"] - blocks["dist_over_vol"]["brier"],
        }
        fold_df = pd.DataFrame(folds)
        per_fold = {"n": len(fold_df),
                    "auc_gbm_mean": float(fold_df["auc_gbm"].mean()),
                    "auc_gbm_median": float(fold_df["auc_gbm"].median()),
                    "auc_pers_mean": float(fold_df["auc_pers"].mean()),
                    "folds_gbm_beats_pers_auc": int((fold_df["auc_gbm"] > fold_df["auc_pers"]).sum()),
                    "folds_comparable": int(fold_df[["auc_gbm", "auc_pers"]].notna().all(1).sum()),
                    "folds_gbm_beats_pers_brier":
                        int((fold_df["brier_gbm"] < fold_df["brier_pers_cal"]).sum())}

        degenerate = {
            "share_predicted_on_at_0.5": float((oos["gbm"] >= 0.5).mean()),
            "share_prob_above_0.9": float((oos["gbm"] > 0.9).mean()),
            "share_prob_below_0.1": float((oos["gbm"] < 0.1).mean()),
            "prob_std": float(oos["gbm"].std()),
            "collapsed_to_always_on": bool((oos["gbm"] >= 0.5).mean() > 0.995)}

        thr_sweep = {f"{t:.2f}": {"gbm": score_block(oos["y"], oos["gbm"], t)}
                     for t in (0.3, 0.4, 0.5, 0.6, 0.7)}

        out["horizons"][name] = {
            "horizon_bars": h, "n_oos_bars": int(len(oos)),
            "oos_span": [str(oos.index.min()), str(oos.index.max())],
            "oos_base_rate_on": float(oos["y"].mean()),
            "scores": blocks, "margin_over_naive": margin, "per_fold": per_fold,
            "degenerate_check": degenerate,
            "threshold_sweep": thr_sweep,
            "bootstrap_auc_vs_persistence": block_bootstrap_auc(oos),
            "bootstrap_auc_vs_dist_over_vol": block_bootstrap_auc(oos, "gbm", "dist_over_vol"),
            "conditional_within_state": conditional(oos),
            "deciles_dist_over_vol_on_bars": deciles(oos),
            "anticipation": anticipation(oos, b["state"], h),
            "economics": economics(oos, b, h),
            "economics_best_predictor": economics(oos, b, h, col="dist_over_vol"),
            "folds": folds,
        }

        g, p = blocks["gbm"], blocks["pers_hard"]
        print(f"  pooled OOS rows {len(oos)}  base rate on={oos['y'].mean():.3f}")
        print(f"  AUC   gbm {g['auc']:.4f} | logit {blocks['logit']['auc']:.4f} | "
              f"persistence {p['auc']:.4f} | base rate 0.5000")
        print(f"  AUC   FREE BASELINES: dist_sma200 only {blocks['dist_only']['auc']:.4f} | "
              f"dist/vol only {blocks['dist_over_vol']['auc']:.4f}")
        print(f"  ACC   gbm {g['accuracy']:.4f} | persistence {p['accuracy']:.4f} | "
              f"always-on {blocks['always_on']['accuracy']:.4f}")
        print(f"  BRIER gbm {g['brier']:.4f} | persistence(cal) {blocks['pers_cal']['brier']:.4f} | "
              f"base {blocks['base']['brier']:.4f}")
        c = out["horizons"][name]["conditional_within_state"]
        for k, v in c.items():
            if "auc_gbm" in v:
                print(f"  CONDITIONAL {k}: n={v['n']} auc_gbm={v['auc_gbm']:.4f} "
                      f"| dist/vol alone {v['auc_dist_over_vol']:.4f} "
                      f"(margin over it {v['margin_auc_over_dist_over_vol']:+.4f}) "
                      f"| persistence 0.5000")
                print(f"      brier gbm {v['brier_gbm']:.4f} | dist/vol "
                      f"{v['brier_dist_over_vol']:.4f} | climatology "
                      f"{v['brier_within_state_climatology']:.4f}")
        a = out["horizons"][name]["anticipation"]
        print(f"  flips in OOS: {a.get('flip_episodes_on_to_off_in_oos')} "
              f"| warned before: {a.get('episodes_with_a_prior_warning')} "
              f"(random would warn {a.get('expected_episodes_warned_if_random')}) "
              f"| warning precision {a.get('warning_precision'):.3f} "
              f"({a.get('warning_true')} true / {a.get('warning_false')} false)")
        print(f"  warning density in pre-flip windows {a.get('warn_density_inside_preflip_windows'):.3f}"
              f" vs {a.get('warn_rate_on_all_on_bars'):.3f} everywhere -> lift "
              f"{a.get('warn_density_lift')}")
        e = out["horizons"][name]["economics"]
        print(f"  BTC fwd {name} when warned {e['fwd_ret_when_warned']:+.4f} "
              f"(n={e['bars_warned']}) vs clear {e['fwd_ret_when_clear']:+.4f} (n={e['bars_clear']})")

    print("\n=== direction control (expect AUC ~0.50) ===")
    out["direction_control"] = direction_control(b, X, log=lambda *a: None)
    dc = out["direction_control"]
    if dc:
        print(f"  AUC gbm {dc['auc_gbm']:.4f} | logit {dc['auc_logit']:.4f} | "
              f"base rate up {dc['base_rate_up']:.3f} | acc {dc['accuracy_gbm']:.4f} "
              f"vs majority {dc['accuracy_base']:.4f}")

    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
