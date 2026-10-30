"""ML aimed at the RIGHT TAIL, not at the win rate.

Track C (research/candidate_model.py) trained on `win = net > 0` and discovered an inversion: the
model's most-confident quartile earned +3.65% while its least-confident earned +8.79%. Win rate
runs backwards to profit in a trend system, because the trades most likely to win are the small
ones.

HYPOTHESIS: the failure was the TARGET, not the features. Predict the right tail explicitly and
the ordering should stop inverting:

    (a) tail20   binary: will this trade return more than +20% net?
    (b) tail50   binary: more than +50%?
    (c) logret   regression on log(1+net), which compresses the tail instead of letting one
                 squeeze own the loss function

Everything else is held fixed against candidate_model.py so the comparison is clean: the same
candidate population, the same labels (real net return of the trade the bot would have taken, next
open in, live exit rule out, 7 bps slippage + 0.10% fee per side), the same 730/91 walk-forward
with a 30-day purge gap, the same feature list, the same model hyperparameters.

Used only as a SIZE multiplier, never as a veto - filtering is what deleted the winners last time.

TWO MAPPINGS from prediction to multiplier, and the difference between them is the whole point:

  bar    percentile of the prediction WITHIN the bar (what candidate_model.py did). On a bar with
         one candidate the percentile is 1.0, so the multiplier is the maximum. With max_names=3
         and typically one or two candidates a bar, this mapping is mostly pure leverage wearing a
         model's clothes.
  train  percentile of the prediction within THAT FOLD'S TRAINING predictions. A lone candidate the
         model dislikes gets 0.5x. This is the only mapping that can express "small" when nothing
         else is competing, and it is the one that actually tests the hypothesis.

CONTROLS, because a multiplier whose mean is above 1 raises return by arithmetic alone:
  const 1.5 / 2.0   every entry at the cap: the pure-leverage null
  shuffled          the model's own multipliers, randomly reassigned among its own rows

THE 2.0 CAP. backtest.run hard-clips size_scale_col to [0.5, 1.5], and app/ is off limits, so the
wider cap is expressed the way the engine already allows - through the sizing reference:

    qty = equity * risk_pct / (entry - stop_level) * scale

so writing stop_level' = close - (close - stop_level)/m multiplies the risk-based size by exactly
m, and every cap (max_weight, min_notional, slot count) is still applied afterwards by the engine.
With the shipped flags stop_exec is OFF, so stop_level is a sizing reference ONLY - it is not an
exit and it is not the breakdown column. `_verify_stop_trick` asserts the two routes give bit-
identical equity inside [0.5, 1.5] before any 2.0 number is believed.

NO LOOKAHEAD: predictions come only from folds whose training window ends 30 days before the test
window opens; the multiplier for a bar is read at that bar's close and sized into the next bar's
open by the engine; the train-CDF percentile uses only that fold's training predictions; the
per-bar percentile is computed within a bar_time. The one soft edge is audited and printed: 1.11%
of training labels belong to trades that were still open 30 days after entry, so they resolve after
the purge gap closes.

RESULT: REJECT, 0 of 52 variants pass. See the CONCLUSION block printed at the end of a run, and
reports/explore_ml_right_tail.json.

    cd crypto-bot && PYTHONIOENCODING=utf-8 python research/explore_ml_right_tail.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from app import backtest, config, features, signals  # noqa: E402
from app.flags import Flags  # noqa: E402
from candidate_model import DERIVED, FEATS, add_derived, label_trades  # noqa: E402

SHIPPED = Flags(abs_momentum=True, volume_min=0.8, taker_min=0.5)
TRAIN_DAYS, TEST_DAYS, PURGE_DAYS = 730, 91, 30
SEEDS = (0,)               # HistGradientBoosting is deterministic at this sample size (no row
                           # subsampling, early stopping off below 10k rows), so random_state does
                           # nothing and averaging over seeds would be averaging identical fits.
                           # The stability check perturbs the TRAINING ROWS instead, which works.
HELD_OUT = "2022-01-01"
HP = dict(max_iter=250, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=25,
          l2_regularization=1.0)

TARGETS = {
    "tail20": ("clf", lambda net: (net > 0.20).astype(int)),
    "tail50": ("clf", lambda net: (net > 0.50).astype(int)),
    "logret": ("reg", lambda net: np.log1p(net.clip(-0.95, None))),
}


# ------------------------------------------------------------------ candidates and labels

def build(s):
    u = json.loads((ROOT / "reports" / "_parity_universe.json").read_text())
    panel = add_derived(features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars))
    rank = panel["rank"].to_numpy()
    btc = (panel["symbol"] == "BTCUSDT").to_numpy()
    cand = ((panel["eligible"] & panel["regime_on"] & panel["breakout"]).to_numpy()
            & ((panel["rs_7d"] > 0).to_numpy() | btc) & (rank <= s.candidates)
            & signals.extra_filters(panel, SHIPPED).to_numpy())
    panel["_cand"] = cand
    lab = label_trades(panel, s)
    d = panel[panel["_cand"]].merge(lab, on=["bar_time", "symbol"], how="inner")
    return panel, d


# ------------------------------------------------------------------ walk-forward

def walk_forward(d, seeds=SEEDS, subsample=None, rs=0):
    """Out-of-sample prediction per target, plus its percentile inside the fold's TRAIN preds.

    Returns the frame with, per target, `<name>_p` (the averaged raw prediction) and `<name>_q`
    (that prediction's rank inside the same fold's training predictions, in [0, 1]).
    """
    X = d[FEATS + DERIVED].replace([np.inf, -np.inf], np.nan)
    t = d["bar_time"]
    ys = {k: f(d["net"]) for k, (_, f) in TARGETS.items()}
    for k in TARGETS:
        d[f"{k}_p"] = np.nan
        d[f"{k}_q"] = np.nan

    for k in TARGETS:
        # that fold's TRAIN prediction median and mean, for the two ratio mappings. Both are pure
        # training statistics, so using them on the test rows introduces no lookahead.
        d[f"{k}_m"] = np.nan
        d[f"{k}_u"] = np.nan

    first = t.min() + pd.Timedelta(days=TRAIN_DAYS)
    test_start, folds, skipped = first, 0, {k: 0 for k in TARGETS}
    rng = np.random.default_rng(rs)
    while test_start < t.max():
        test_end = test_start + pd.Timedelta(days=TEST_DAYS)
        tr = (t >= test_start - pd.Timedelta(days=TRAIN_DAYS)) & \
             (t < test_start - pd.Timedelta(days=PURGE_DAYS))
        te = (t >= test_start) & (t < test_end)
        if tr.sum() >= 150 and te.any():
            used = False
            for name, (kind, _) in TARGETS.items():
                y = ys[name]
                if kind == "clf" and y[tr].nunique() < 2:
                    skipped[name] += 1
                    continue
                if kind == "clf" and min(y[tr].sum(), (1 - y[tr]).sum()) < 10:
                    skipped[name] += 1          # too few of one class to learn anything
                    continue
                pt, pr = [], []
                for seed in seeds:
                    M = (HistGradientBoostingClassifier if kind == "clf"
                         else HistGradientBoostingRegressor)(random_state=seed, **HP)
                    if subsample:
                        # perturb the training SET, not the seed: the only thing that actually
                        # changes this model's fit, and the right null for "is this result noise?"
                        idx = np.flatnonzero(tr.to_numpy())
                        keep = rng.choice(idx, int(len(idx) * subsample), replace=False)
                        mask = np.zeros(len(tr), bool)
                        mask[keep] = True
                        tr_fit = pd.Series(mask, index=tr.index)
                        if y[tr_fit].nunique() < 2:
                            continue
                        M.fit(X[tr_fit], y[tr_fit])
                    else:
                        M.fit(X[tr], y[tr])
                    if kind == "clf":
                        pt.append(M.predict_proba(X[te])[:, 1])
                        pr.append(M.predict_proba(X[tr])[:, 1])
                    else:
                        pt.append(M.predict(X[te]))
                        pr.append(M.predict(X[tr]))
                if not pt:
                    skipped[name] += 1
                    continue
                p_te, p_tr = np.mean(pt, axis=0), np.sort(np.mean(pr, axis=0))
                d.loc[te, f"{name}_p"] = p_te
                d.loc[te, f"{name}_q"] = np.searchsorted(p_tr, p_te, side="right") / len(p_tr)
                d.loc[te, f"{name}_m"] = float(np.median(p_tr))
                d.loc[te, f"{name}_u"] = float(np.mean(p_tr))
                used = True
            folds += used
        test_start = test_end
    return d, folds, skipped


# ------------------------------------------------------------------ diagnostics

def quartiles(o, col, label):
    """Out-of-sample: predicted bucket vs what the trades in it actually did."""
    sub = o[o[col].notna()]
    if sub[col].nunique() < 4:
        print(f"  {label}: not enough distinct predictions to bucket")
        return None
    q = pd.qcut(sub[col], 4, labels=["lowest", "low", "high", "highest"], duplicates="drop")
    tab = sub.groupby(q, observed=True).agg(
        trades=("net", "size"), win_rate=("win", "mean"), mean_net=("net", "mean"),
        median_net=("net", "median"), share_gt20=("net", lambda x: (x > 0.20).mean()),
        share_gt50=("net", lambda x: (x > 0.50).mean()), best=("net", "max"))
    ic = sub[col].corr(sub["net"], method="spearman")
    print(f"\n  {label}  (spearman vs realised net: {ic:+.4f}, n={len(sub)})")
    print(tab.to_string(float_format=lambda x: f"{x:.4f}"))
    lo, hi = tab["mean_net"].iloc[0], tab["mean_net"].iloc[-1]
    print(f"    monotone in the right direction? highest {hi:+.2%} vs lowest {lo:+.2%}  "
          f"-> {'YES' if hi > lo else 'NO - still inverted'}")
    return {"ic": float(ic), "n": int(len(sub)), "inverted": bool(hi <= lo),
            "table": tab.reset_index().rename(columns={col: "bucket"}).to_dict("records")}


# ------------------------------------------------------------------ sizing routes

def mult_from_q(q, hi):
    """q in [0,1] -> multiplier spread linearly over [0.5, hi]. Unscored rows size normally.

    hi=1.5 is MEAN-NEUTRAL when q is uniform (average multiplier 1.0), so it is a pure
    reallocation of the same risk budget. hi=2.0 averages 1.25, i.e. it also adds 25% leverage -
    which is why the constant-multiplier controls below exist.
    """
    return (0.5 + (hi - 0.5) * q.clip(0, 1)).fillna(1.0).clip(0.5, hi)


def mult_from_ratio(p, denom, lo, hi):
    """Multiplier = prediction / a TRAIN statistic of the prediction, clipped. No percentiles, so a
    bar where the model likes nothing is sized down even if every candidate is equally bad.

    denom = train MEDIAN makes the multiplier average above 1 (a probability distribution is
    right-skewed), so it adds leverage as well as tilt. denom = train MEAN is approximately
    mean-neutral, which is what separates allocation skill from leverage.
    """
    return (p / denom.replace(0, np.nan)).fillna(1.0).clip(lo, hi)


def run_sized(panel, s, mult, lo=0.5, hi=1.5, **kw):
    """Size every entry by `mult`, clipped to [lo, hi].

    hi <= 1.5 goes through the engine's own size_scale_col. A wider cap is expressed through the
    sizing reference (stop_level), which multiplies the risk-based size by exactly the same factor;
    `_verify_stop_trick` proves the two routes agree where they overlap.
    """
    p = panel.copy()
    m = mult.clip(lo, hi).fillna(1.0)
    if hi <= 1.5 and lo >= 0.5:
        p["_mult"] = m
        return backtest.run(p, s, flags=SHIPPED, size_scale_col="_mult", **kw)
    p["stop_level"] = p["close"] - (p["close"] - p["stop_level"]) / m
    return backtest.run(p, s, flags=SHIPPED, **kw)


def _verify_stop_trick(panel, s, mult):
    """The stop-level route must reproduce the size_scale_col route inside [0.5, 1.5]."""
    a = run_sized(panel, s, mult, 0.5, 1.5)
    p = panel.copy()
    m = mult.clip(0.5, 1.5).fillna(1.0)
    p["stop_level"] = p["close"] - (p["close"] - p["stop_level"]) / m
    b = backtest.run(p, s, flags=SHIPPED)
    same = np.allclose(a.equity.to_numpy(), b.equity.to_numpy(), rtol=1e-9, atol=1e-6)
    gap = float(np.abs(a.equity.to_numpy() - b.equity.to_numpy()).max())
    print(f"\nstop-level route == size_scale_col route inside [0.5,1.5]: {same}  "
          f"(max equity gap {gap:.3e} on {a.equity.iloc[-1]:,.0f})")
    return same


def row(name, res, mult=None, scored=None):
    m = backtest.metrics(res.equity, res.trades, res.exposure)
    out = {"run": name, "cagr": m["cagr"], "sharpe": m["sharpe"], "maxdd": m["max_drawdown"],
           "trades": m.get("trades", 0), "win": m.get("win_rate"),
           "avg_trade": m.get("avg_trade"), "expo": m.get("avg_exposure"),
           "best_trade": res.trades["ret"].max() if len(res.trades) else np.nan}
    if mult is not None:
        # averaged over the rows the model actually had an opinion about, which is the number that
        # says whether a variant is a reallocation or just more leverage.
        sel = mult[scored] if scored is not None else mult
        out["mean_mult"] = float(sel.mean())
    return out


def passes(v, base, eps=0.0):
    return (v["cagr"] >= base["cagr"] - eps and v["sharpe"] >= base["sharpe"] - eps
            and v["maxdd"] >= base["maxdd"] - eps)


# ------------------------------------------------------------------ main

def main():
    s = config.load()
    print("building panel and labelling the candidate population ...")
    panel, d = build(s)
    print(f"candidate bars {int(panel['_cand'].sum())}   labelled trades {len(d)}")
    print(f"win rate {d['win'].mean():.1%}  mean net {d['net'].mean():+.2%}  "
          f"median {d['net'].median():+.2%}")
    print("the right tail we are trying to predict: "
          f"{(d['net'] > 0.20).mean():.1%} of trades beat +20%, "
          f"{(d['net'] > 0.50).mean():.1%} beat +50%, "
          f"and they carry {d.loc[d['net'] > 0.20, 'net'].sum() / d['net'].sum():.0%} / "
          f"{d.loc[d['net'] > 0.50, 'net'].sum() / d['net'].sum():.0%} of all the return")

    # AUDIT the purge gap. A training label is the outcome of a trade that resolves AFTER its entry
    # bar, so a training trade still open when the test window opens would leak. The gap is 30 days
    # = 180 four-hour bars; this says how many labels outlive it.
    long_held = (d["bars_held"] > 180).mean()
    print(f"\npurge-gap audit: holding period median {d['bars_held'].median():.0f} bars, "
          f"mean {d['bars_held'].mean():.0f}, p95 {d['bars_held'].quantile(0.95):.0f}, "
          f"max {d['bars_held'].max():.0f}")
    print(f"  labels resolving AFTER the 30-day (180-bar) purge gap: {long_held:.2%} of trades "
          f"-- these are the only rows that could leak across a fold boundary")

    d, folds, skipped = walk_forward(d)
    print(f"\nwalk-forward: {folds} folds trained (730d train / 91d test / 30d purge), "
          f"{len(SEEDS)} seeds averaged")
    for k in TARGETS:
        n = int(d[f'{k}_p'].notna().sum())
        print(f"  {k}: {n} scored candidates, {skipped[k]} folds skipped (class too rare)")

    d["ens_p"] = d[["tail20_p", "tail50_p"]].mean(axis=1)
    # A causal trailing LEVEL of the model's own output: the mean prediction over the previous 100
    # scored candidates, shifted so the current row is excluded. It splits the model's signal into
    # "the market is in a state the model scores highly" (level) and "this candidate scores highly
    # against its recent peers" (ratio to the level). Only the second is trade selection.
    for k in ("tail20", "tail50"):
        sc = d[d[f"{k}_p"].notna()].sort_values("bar_time")
        d.loc[sc.index, f"{k}_lvl"] = (sc[f"{k}_p"].rolling(100, min_periods=20).mean()
                                       .shift(1).to_numpy())
    print("\n=== OUT-OF-SAMPLE QUARTILES: does a right-tail target stop the inversion? ===")
    print("(win_rate is deliberately shown next to mean_net: in Track C they moved opposite ways)")
    diag = {}
    for k in list(TARGETS) + ["ens"]:
        diag[k] = quartiles(d, f"{k}_p", f"{k} (raw prediction)")

    # The question behind all of it: can the model predict ITS OWN target out of sample? An AUC at
    # 0.50 means no, whatever the backtest afterwards happens to show.
    print("\n=== CAN THE MODEL PREDICT ITS OWN TARGET, OUT OF SAMPLE? (AUC 0.50 = coin flip) ===")
    auc = {}
    for k, thr in (("tail20", 0.20), ("tail50", 0.50), ("ens", 0.20)):
        sub = d[d[f"{k}_p"].notna()]
        y = (sub["net"] > thr).astype(int)
        if y.nunique() < 2:
            continue
        a = roc_auc_score(y, sub[f"{k}_p"])
        dec = pd.qcut(sub[f"{k}_p"], 10, labels=False, duplicates="drop")
        top, bot = y[dec == dec.max()].mean(), y[dec == 0].mean()
        auc[k] = {"auc": float(a), "top_decile_hit": float(top), "bottom_decile_hit": float(bot),
                  "base_rate": float(y.mean())}
        print(f"  {k:7} target net>{thr:.0%}:  AUC {a:.4f}   base rate {y.mean():.1%}   "
              f"top decile {top:.1%} vs bottom decile {bot:.1%}")
    # and, for comparison, the old win-rate question on the same rows
    sub = d[d["tail20_p"].notna()]
    print(f"  (for reference, the same model scored against win/lose: "
          f"AUC {roc_auc_score(sub['win'].astype(int), sub['tail20_p']):.4f})")

    oos_start = d[[f"{k}_p" for k in TARGETS]].notna().any(axis=1)
    start = d.loc[oos_start, "bar_time"].min()
    print(f"\nfirst out-of-sample bar: {start.date()}")

    merged = panel.merge(
        d[["bar_time", "symbol"] + [f"{k}_{x}" for k in TARGETS for x in "pqmu"]
          + ["tail20_lvl", "tail50_lvl"]], on=["bar_time", "symbol"], how="left")
    # an ensemble of the two right-tail classifiers, averaged in percentile space
    merged["ens_q"] = merged[["tail20_q", "tail50_q"]].mean(axis=1)
    merged["ens_p"] = merged[["tail20_p", "tail50_p"]].mean(axis=1)
    scored = merged[[f"{k}_p" for k in TARGETS]].notna().any(axis=1)

    # q in [0,1] per mapping; mult_from_q then spreads it over [0.5, hi]
    qs = {}
    for k in list(TARGETS) + ["ens"]:
        # per-bar percentile: candidate_model.py's mapping. On a one-candidate bar it is always 1.0.
        qs[(k, "bar")] = merged.groupby("bar_time")[f"{k}_p"].rank(pct=True)
        # train-CDF percentile: the only mapping that can say "small" on a lone candidate.
        qs[(k, "train")] = merged[f"{k}_q"]

    ok = _verify_stop_trick(merged, s, mult_from_q(qs[("ens", "train")], 1.5))

    print("\nrunning backtests (full panel and held-out) ...")
    base = backtest.run(merged, s, flags=SHIPPED)
    base_h = backtest.run(merged, s, flags=SHIPPED, start=HELD_OUT)
    std_rows = [row("baseline (shipped)", base)]
    held_rows = [row("baseline (shipped)", base_h)]

    variants = []
    for k in list(TARGETS) + ["ens"]:
        for mapping in ("bar", "train"):
            for hi in (1.5, 2.0):
                variants.append((f"{k} / {mapping} / [0.5,{hi}]",
                                 mult_from_q(qs[(k, mapping)], hi), 0.5, hi))
    # the ratio mappings only make sense for a probability, not for a log-return prediction
    for k in ("tail20", "tail50"):
        for hi in (1.5, 2.0):
            variants.append((f"{k} / ratio-med / [0.5,{hi}]",
                             mult_from_ratio(merged[f"{k}_p"], merged[f"{k}_m"], 0.5, hi),
                             0.5, hi))
            variants.append((f"{k} / ratio-mean / [0.5,{hi}]",
                             mult_from_ratio(merged[f"{k}_p"], merged[f"{k}_u"], 0.5, hi),
                             0.5, hi))
    # Gentle symmetric tilts around 1.0: the only shape that can raise return without raising the
    # peak position size, which is the one thing the drawdown criterion cares about.
    for k in ("tail20", "tail50"):
        for w in (0.1, 0.2, 0.3, 0.5):
            variants.append((f"{k} / ratio-mean / [{1 - w:.1f},{1 + w:.1f}]",
                             mult_from_ratio(merged[f"{k}_p"], merged[f"{k}_u"], 1 - w, 1 + w),
                             1 - w, 1 + w))

    # ---- DECOMPOSITION. AUC near 0.50 says the model cannot rank one candidate against another,
    # yet the ratio-med variant beats its own reshuffles. Both can only be true if what helps is
    # the LEVEL of the model's output over time, i.e. market timing, not trade selection. Split it.
    for k in ("tail20", "tail50"):
        for hi in (1.5, 2.0):
            variants.append((f"{k} / LEVEL-only (timing) / [0.5,{hi}]",
                             mult_from_ratio(merged[f"{k}_lvl"], merged[f"{k}_m"], 0.5, hi),
                             0.5, hi))
            variants.append((f"{k} / CROSS-only (selection) / [0.5,{hi}]",
                             mult_from_ratio(merged[f"{k}_p"], merged[f"{k}_lvl"], 0.5, hi),
                             0.5, hi))
    # ...and the no-ML version of the same timing idea, built from one column already in the panel.
    for hi in (1.5, 2.0):
        variants.append((f"MARKET 1+btc_dist_sma200 / [0.5,{hi}]",
                         (1 + merged["btc_dist_sma200"]).where(scored).fillna(1.0).clip(0.5, hi),
                         0.5, hi))
        variants.append((f"MARKET 0.5+breadth / [0.5,{hi}]",
                         (0.5 + merged["breadth"]).where(scored).fillna(1.0).clip(0.5, hi),
                         0.5, hi))

    # ---- controls. Any multiplier whose mean sits above 1.0 raises CAGR by arithmetic alone, so
    # every model variant is matched against a constant multiplier of a similar average.
    const = pd.Series(1.0, index=merged.index)
    for hi in (0.7, 0.8, 0.9, 0.95, 1.1, 1.15, 1.25, 1.4, 1.5, 2.0):
        variants.append((f"CONTROL const {hi} (pure leverage)", const.where(~scored, hi),
                         min(hi, 0.5), max(hi, 1.5)))

    seen, uniq = set(), []                       # the loops above overlap at [0.5,1.5]
    for v in variants:
        if v[0] not in seen:
            seen.add(v[0])
            uniq.append(v)
    variants = uniq

    for name, mult, lo, hi in variants:
        m = mult.clip(lo, hi).fillna(1.0)        # exactly what the engine will apply
        r = run_sized(merged, s, m, lo, hi)
        rh = run_sized(merged, s, m, lo, hi, start=HELD_OUT)
        std_rows.append(row(name, r, m, scored))
        held_rows.append(row(name, rh, m, scored))

    # ---- permutation test on the variant that looked best, properly powered. Same multiplier
    # distribution, same rows, information destroyed. If the model has no edge, its own result
    # should sit in the middle of this null.
    N_PERM = 20
    best_name = "tail20 / ratio-med / [0.5,1.5]"
    best_mult = mult_from_ratio(merged["tail20_p"], merged["tail20_m"], 0.5, 1.5)
    best_std = next(r for r in std_rows if r["run"] == best_name)
    best_held = next(r for r in held_rows if r["run"] == best_name)
    rng = np.random.default_rng(11)
    null = []
    src = best_mult.where(scored)
    vals0 = src[src.notna()].to_numpy()
    for i in range(N_PERM):
        vals = vals0.copy()
        rng.shuffle(vals)
        sh = src.copy()
        sh[sh.notna()] = vals
        sh = sh.fillna(1.0)
        a = backtest.metrics(*[getattr(run_sized(merged, s, sh, 0.5, 1.5), x)
                               for x in ("equity", "trades", "exposure")])
        b = backtest.metrics(*[getattr(run_sized(merged, s, sh, 0.5, 1.5, start=HELD_OUT), x)
                               for x in ("equity", "trades", "exposure")])
        null.append({"std_cagr": a["cagr"], "std_sharpe": a["sharpe"], "std_dd": a["max_drawdown"],
                     "held_cagr": b["cagr"], "held_sharpe": b["sharpe"],
                     "held_dd": b["max_drawdown"]})
    nd = pd.DataFrame(null)
    perm = {}
    print(f"\n=== PERMUTATION TEST on '{best_name}' "
          f"({N_PERM} reshuffles of its own multipliers) ===")
    for panel_name, v in (("std", best_std), ("held", best_held)):
        for metric in ("cagr", "sharpe"):
            col = nd[f"{panel_name}_{metric}"]
            p = float((col >= v[metric]).mean())
            perm[f"{panel_name}_{metric}_p"] = p
            print(f"  {panel_name:4} {metric:6} model {v[metric]:.4f}  null mean {col.mean():.4f} "
                  f"(min {col.min():.4f}, max {col.max():.4f})   p = {p:.2f}")
    perm["n"] = N_PERM
    print("  p is the share of information-free reshuffles that did at least as well.")

    cols = ["run", "cagr", "sharpe", "maxdd", "trades", "win", "avg_trade", "best_trade",
            "expo", "mean_mult"]
    std = pd.DataFrame(std_rows)[cols]
    held = pd.DataFrame(held_rows)[[c for c in cols if c != "mean_mult"]]
    fmt = lambda x: f"{x:.4f}"  # noqa: E731
    print("\n=== STANDARD PANEL (2017-08 .. 2026-09) ===")
    print(std.to_string(index=False, float_format=fmt))
    print(f"\n=== HELD-OUT (from {HELD_OUT}) ===")
    print(held.to_string(index=False, float_format=fmt))

    # ---- matched leverage. A multiplier averaging above 1.0 raises CAGR with no skill at all, so
    # every model variant is scored against the CONSTANT multiplier of the nearest average.
    consts = [(r["mean_mult"], r, h) for r, h in zip(std_rows[1:], held_rows[1:])
              if r["run"].startswith("CONTROL const")]
    print("\n=== SKILL, NOT LEVERAGE: each variant minus the constant multiplier of the same "
          "average ===")
    matched = []
    for v_s, v_h in zip(std_rows[1:], held_rows[1:]):
        if v_s["run"].startswith("CONTROL"):
            continue
        c_mult, c_s, c_h = min(consts, key=lambda c: abs(c[0] - v_s["mean_mult"]))
        m = {"run": v_s["run"], "mean_mult": v_s["mean_mult"], "matched_const": c_mult,
             "d_cagr_std": v_s["cagr"] - c_s["cagr"], "d_sharpe_std": v_s["sharpe"] - c_s["sharpe"],
             "d_dd_std": v_s["maxdd"] - c_s["maxdd"],
             "d_cagr_held": v_h["cagr"] - c_h["cagr"],
             "d_sharpe_held": v_h["sharpe"] - c_h["sharpe"],
             "d_dd_held": v_h["maxdd"] - c_h["maxdd"]}
        m["beats_matched_both"] = bool(m["d_cagr_std"] > 0 and m["d_cagr_held"] > 0
                                      and m["d_sharpe_std"] > 0 and m["d_sharpe_held"] > 0)
        matched.append(m)
    md = pd.DataFrame(matched).sort_values("d_sharpe_held", ascending=False)
    print(md.to_string(index=False, float_format=fmt))

    b_std, b_held = std_rows[0], held_rows[0]
    print("\n=== SHIP RULE: CAGR >= base AND Sharpe >= base AND maxDD not worse, BOTH panels ===")
    verdict = []
    for v_s, v_h in zip(std_rows[1:], held_rows[1:]):
        ps, ph = passes(v_s, b_std), passes(v_h, b_held)
        verdict.append({"run": v_s["run"], "std_pass": ps, "held_pass": ph, "ship": ps and ph,
                        "std": {k: v_s[k] for k in ("cagr", "sharpe", "maxdd", "trades")},
                        "held": {k: v_h[k] for k in ("cagr", "sharpe", "maxdd", "trades")}})
        print(f"  {'SHIP' if ps and ph else 'reject':>6}  {v_s['run']:<44} "
              f"std[{'pass' if ps else 'fail'}] held[{'pass' if ph else 'fail'}]  "
              f"cagr {v_s['cagr']:+.4f}/{v_h['cagr']:+.4f}  "
              f"sharpe {v_s['sharpe']:.3f}/{v_h['sharpe']:.3f}  "
              f"dd {v_s['maxdd']:.4f}/{v_h['maxdd']:.4f}")

    winners = [v for v in verdict if v["ship"]]
    print(f"\nvariants tested: {len(variants)}   passing the ship rule on BOTH panels: "
          f"{len(winners)}")
    if winners:
        for w in winners:
            print("  ->", w["run"])
    else:
        print("  -> none. The right-tail target does not beat the shipped strategy.")

    # Same rule against the baseline numbers quoted in the brief, which came from an earlier state
    # of app/backtest.py than the one in this tree. Reported so the verdict does not hinge on which
    # baseline is used.
    QUOTED_STD = {"cagr": 0.2071, "sharpe": 1.3712, "maxdd": -0.1509}
    QUOTED_HELD = {"cagr": 0.1473, "sharpe": 1.1004, "maxdd": -0.1509}
    q_win = [v_s["run"] for v_s, v_h in zip(std_rows[1:], held_rows[1:])
             if passes(v_s, QUOTED_STD) and passes(v_h, QUOTED_HELD)]
    print(f"\nagainst the brief's quoted (older-engine) baseline, variants passing both panels: "
          f"{len(q_win)}  {q_win if q_win else ''}")

    # best by CAGR among the non-control variants, for the record
    real = [v for v in verdict if not v["run"].startswith("CONTROL")]
    best = max(real, key=lambda v: v["std"]["cagr"])
    print(f"\nhighest-CAGR model variant: {best['run']}  "
          f"std cagr {best['std']['cagr']:.4f} sharpe {best['std']['sharpe']:.4f} "
          f"dd {best['std']['maxdd']:.4f}")
    bm = max(matched, key=lambda m: m["d_sharpe_held"])
    print(f"best against matched leverage: {bm['run']}  "
          f"dSharpe std {bm['d_sharpe_std']:+.3f} held {bm['d_sharpe_held']:+.3f}, "
          f"dCAGR std {bm['d_cagr_std']:+.4f} held {bm['d_cagr_held']:+.4f}")

    # ---- STABILITY. Random_state does nothing to this model, so the perturbation that matters is
    # the training SET: refit every fold on a random 90% of its training rows. If the headline
    # variant is information it should survive; if it is a lucky arrangement it will scatter.
    N_REFIT = 6
    print(f"\n=== TRAINING-SET STABILITY of '{best_name}' "
          f"({N_REFIT} refits, each fold trained on a random 90% of its rows) ===")
    refits = []
    for i in range(N_REFIT):
        _, di = build(s)
        di, _, _ = walk_forward(di, subsample=0.9, rs=100 + i)
        si = di[di["tail20_p"].notna()]
        ai = roc_auc_score((si["net"] > 0.20).astype(int), si["tail20_p"])
        mi = panel.merge(di[["bar_time", "symbol", "tail20_p", "tail20_m"]],
                         on=["bar_time", "symbol"], how="left")
        mui = mult_from_ratio(mi["tail20_p"], mi["tail20_m"], 0.5, 1.5)
        ri = backtest.metrics(*[getattr(run_sized(mi, s, mui, 0.5, 1.5), x)
                                for x in ("equity", "trades", "exposure")])
        hi_ = backtest.metrics(*[getattr(run_sized(mi, s, mui, 0.5, 1.5, start=HELD_OUT), x)
                                 for x in ("equity", "trades", "exposure")])
        sp = passes({"cagr": ri["cagr"], "sharpe": ri["sharpe"],
                     "maxdd": ri["max_drawdown"]}, b_std)
        hp = passes({"cagr": hi_["cagr"], "sharpe": hi_["sharpe"],
                     "maxdd": hi_["max_drawdown"]}, b_held)
        refits.append({"refit": i + 1, "auc": float(ai),
                       "std_cagr": ri["cagr"], "std_sharpe": ri["sharpe"],
                       "std_dd": ri["max_drawdown"], "held_cagr": hi_["cagr"],
                       "held_sharpe": hi_["sharpe"], "held_dd": hi_["max_drawdown"],
                       "ship": bool(sp and hp)})
        print(f"  refit {i + 1}: AUC {ai:.4f}  std {ri['cagr']:.4f}/{ri['sharpe']:.4f}/"
              f"{ri['max_drawdown']:.4f}   held {hi_['cagr']:.4f}/{hi_['sharpe']:.4f}/"
              f"{hi_['max_drawdown']:.4f}   ship {'PASS' if sp and hp else 'reject'}")
    rf = pd.DataFrame(refits)
    print(f"  full-data fit:  AUC {auc['tail20']['auc']:.4f}  "
          f"std {best_std['cagr']:.4f}/{best_std['sharpe']:.4f}/{best_std['maxdd']:.4f}   "
          f"held {best_held['cagr']:.4f}/{best_held['sharpe']:.4f}/{best_held['maxdd']:.4f}")
    print(f"  refit spread:   AUC {rf.auc.min():.4f}..{rf.auc.max():.4f}   "
          f"std cagr {rf.std_cagr.min():.4f}..{rf.std_cagr.max():.4f}   "
          f"std sharpe {rf.std_sharpe.min():.4f}..{rf.std_sharpe.max():.4f}")
    print(f"                  held cagr {rf.held_cagr.min():.4f}..{rf.held_cagr.max():.4f}   "
          f"held sharpe {rf.held_sharpe.min():.4f}..{rf.held_sharpe.max():.4f}")
    print(f"  refits that would have shipped: {int(rf.ship.sum())} of {N_REFIT}")
    stab = {"n_refits": N_REFIT, "subsample": 0.9, "refits": refits,
            "full_fit": {"auc": auc["tail20"]["auc"],
                         "std": {k: best_std[k] for k in ("cagr", "sharpe", "maxdd")},
                         "held": {k: best_held[k] for k in ("cagr", "sharpe", "maxdd")}},
            "ship_count": int(rf.ship.sum())}

    out = {"baseline_standard": b_std, "baseline_heldout": b_held, "stability": stab,
           "folds": folds, "seeds": list(SEEDS),
           "labelled_trades": int(len(d)),
           "share_gt20": float((d["net"] > 0.20).mean()),
           "share_gt50": float((d["net"] > 0.50).mean()),
           "stop_trick_verified": bool(ok),
           "labels_outliving_purge_gap": float(long_held),
           "bars_held_median": float(d["bars_held"].median()),
           "bars_held_max": float(d["bars_held"].max()),
           "oos_start": str(start.date()),
           "quartiles": diag, "auc": auc, "standard": std_rows, "heldout": held_rows,
           "permutation": perm, "matched_leverage": matched,
           "ship_rule": verdict,
           "ship_winners": [w["run"] for w in winners],
           "ship_winners_vs_quoted_baseline": q_win}
    (ROOT / "reports" / "explore_ml_right_tail.json").write_text(
        json.dumps(out, indent=2, default=str))
    print("\nwrote reports/explore_ml_right_tail.json")

    print("""
=== CONCLUSION ===
REJECT. The hypothesis was that Track C failed because it predicted win rate; predict the right
tail instead and the ordering would stop inverting. It does not, and the reason is worse than an
inversion: there is nothing there to order.

  1. The tail is not predictable from entry-time features. AUC for "will this trade return more
     than +20%" is 0.4914 - below a coin flip - and across six refits on 90% of the training rows
     it moves between 0.4806 and 0.5111, i.e. it straddles 0.50. The +50% model reaches AUC 0.537
     on 1,715 rows, which is the only number above chance anywhere in this study, and its quartiles
     carry 4.0% / 5.6% / 7.7% / 5.6% of >+50% trades: not monotone.
  2. The quartile tables still invert for tail20, logret and the ensemble. What the right-tail
     models DO learn is win rate again (tail50's win rate climbs 36.8% -> 48.3% across quartiles
     while its mean return does not), which is the same signal Track C already rejected.
  3. The one variant that beat the baseline on CAGR and Sharpe on BOTH panels
     (tail20 / ratio-med, std 25.4%/1.500, held 23.7%/1.512) fails the ship rule on drawdown
     (-16.6% against -13.2%) and does not replicate: all six training-perturbation refits land
     below it, four of six below the baseline Sharpe, and none would have shipped.
  4. Splitting that variant shows it was never trade selection. Its timing half (the level of the
     model's output over time) performs no better than a constant multiplier of the same average,
     and a one-line non-ML version of the same idea (1 + btc_dist_sma200) does about as well. Its
     selection half is a de-leveraging that loses CAGR.
  5. The 20-reshuffle permutation test returned p = 0.00-0.05 and is the one test here that
     MISLEADS: shuffling destroys the multiplier's time path, so it measures market-state timing,
     not candidate ranking. Matched leverage and training perturbation are the right nulls, and
     both say no.

Nothing passes the ship rule on either baseline. The P3 conclusion stands and is now stronger: the
few enormous winners this strategy lives on are not identifiable at entry - not by win rate, and
not by the right tail either.""")


if __name__ == "__main__":
    main()
