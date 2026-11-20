"""Search for a predictor on Binance history, as widely as wanted, without fooling ourselves.

    python research/predict_search.py

THE BRIEF. Train on past Binance data, predict, try combination after combination until something
accurate turns up - and do not let one train/test round contaminate the next.

THE TRAP IN THAT BRIEF, and how this avoids it. "Try again until it works" against one test set
guarantees a winner whether or not anything is predictable: with N tries the best of N noise
results looks like signal, and the test set has quietly become training data for the CHOICE of
model. The cure is to separate where the trying happens from where the judging happens:

    DEVELOPMENT  first bar .. 2024-12-31   try anything, as often as wanted. Walk-forward inside.
    HOLDOUT      2025-01-01 .. 2026-09-24  only the top 3 development candidates, once each.
    PRISTINE     2026-09-24 .. today       Binance candles downloaded AFTER every piece of
                                           research in this project was written. Touched once.

A candidate is called real only if it survives all three. The pristine window includes a 20% BTC
fall, so it is small but not gentle.

NOTHING CARRIES BETWEEN FOLDS. A fresh estimator per fold, fitted on that fold's rolling 730-day
training window only; imputer and scaler live inside the model pipeline; a purge gap of horizon+1
bars drops training labels that resolve inside the test window, and the same gap is held at the
development/holdout and holdout/pristine boundaries so no label in one period is computed from
prices in the next.

WHAT IS TRIED (every candidate is feature set x target x horizon x model x row subset):
    features   base    the 16 shipped features + cross-sectional ranks (what every earlier ML
                       rejection in this log was scored on)
               dots    base + 26 relationship features: BTC lead-lag, market breadth and
                       dispersion, rolling beta/correlation, bar microstructure, calendar
                       (research/dots_features.py, proven causal by truncation)
    targets    beats BTC / is up / top third of the cross-section / gains >10% / size of excess
               return, and "falls >10%" as the POSITIVE CONTROL - P6 showed losses are predictable
               here, so a harness that cannot see that one is broken
    horizons   1 bar (4h), 6 bars (1 day), 30 bars (5 days)
    models     gradient boosting, regularised linear
    subsets    all rows; bull-regime rows only; tradable (eligible) rows only

THREE ARTEFACTS THE FIRST GRID PRODUCED, each of which looked like signal and was not
(research/train_predict_grid.py, 20 Nov 2026, flagged "SIGNAL" on 17 of 20 cells):

  1. "beats BTC" scored AUC 0.58 with BTC's own rows in the sample. BTC cannot beat itself, so
     its label is always 0, and its relative-strength features are exactly 0, which identifies it.
     A block of trivially correct negatives lifts AUC from 0.50 to roughly 0.56-0.60 by itself.
     BTC rows are excluded from every excess-return target here.
  2. "gains more than 10%" scored 0.71 - and "falls more than 10%" scored 0.69. A model that
     predicts both tails equally well is predicting VOLATILITY, not direction.
  3. "is up" scored 0.53, which is about what the bull/bear drift difference alone would give -
     information the 200-day regime gate already uses.

So accuracy is not the yardstick. THE YARDSTICK IS MONEY: at every bar, take the 3 coins the model
scores highest and compare their forward return with the average coin at the same bar. That
spread ("edge") is zero for a model with no selection skill no matter how high its AUC, it is
immune to the volatility artefact (wilder coins are not better coins), and it has to exceed a
round trip of costs to be worth anything. Candidates are ranked by the z of that edge across
folds, and AUC is reported next to it.

THE LUCK BAR. With N return-predicting candidates, the best of N pure-noise results is expected
near z = sqrt(2 ln N). A development z below that is what luck looks like, and is printed as such.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from app import config, features, ml  # noqa: E402

import dots_features as dots  # noqa: E402
from train_predict_grid import add_targets, evaluate, make_model  # noqa: E402

DEV_END = pd.Timestamp("2025-01-01", tz="UTC")
HOLD_END = pd.Timestamp("2026-09-24 08:00", tz="UTC")     # where the research cache stopped
DAY_NS = 86_400 * 10**9
BAR_NS = 4 * 3600 * 10**9
COST = 0.0027                                              # round trip: 2 x (0.10% + 7 bps)

TARGETS = {                       # name: (metric, counts as return-predicting)
    "excess_sign": ("auc", True),
    "abs_sign": ("auc", True),
    "top_third": ("auc", True),
    "big_win": ("auc", True),
    "excess_ret": ("ic", True),
    "big_loss": ("auc", False),   # positive control
}
HORIZONS = (1, 6, 30)


def fold_edge(tt, pred, fwd, k=3, min_names=5):
    """Mean forward return of the k highest-scored coins per bar minus the all-coin mean.

    Zero in expectation for any model without selection skill, whatever its AUC.
    """
    d = pd.DataFrame({"t": tt, "pred": pred, "fwd": fwd}).dropna(subset=["fwd"])
    n = d.groupby("t")["fwd"].transform("count")
    d = d[n >= min_names]
    if d.empty:
        return float("nan")
    top = d.sort_values("pred", ascending=False).groupby("t").head(k).groupby("t")["fwd"].mean()
    return float((top - d.groupby("t")["fwd"].mean()).mean())


def walk(X, y, t, task, kind, horizon, t_from, t_to, mask=None, train_days=730, test_days=91,
         seed=11, threads=2, keep_pred=False, fwd=None):
    """Walk forward over test windows inside [t_from, t_to). Arrays only, so it parallelises.

    Returns {"scores", "nulls", "edges"} and, with keep_pred, the out-of-sample predictions by
    row index. `edges` is the per-fold pick spread (see fold_edge) when `fwd` is given.
    """
    from threadpoolctl import threadpool_limits
    purge = (horizon + 1) * BAR_NS
    window = train_days * DAY_NS
    ok = ~np.isnan(y)
    if mask is not None:
        ok &= mask
    rng = np.random.default_rng(seed)
    scores, nulls, edges, idx_all, pred_all = [], [], [], [], []
    cut = max(t_from, int(t[ok].min()) + window + purge)
    with threadpool_limits(limits=threads):
        while cut < t_to - purge:
            end = min(cut + test_days * DAY_NS, t_to - purge)
            tr = ok & (t >= cut - purge - window) & (t < cut - purge)
            te = ok & (t >= cut) & (t < end)
            cut += test_days * DAY_NS
            if tr.sum() < 3000 or te.sum() < 150:
                continue
            ytr = y[tr]
            if task == "auc" and len(np.unique(ytr)) < 2:
                continue
            model = make_model(kind, task)          # fresh, every fold, discarded below
            model.fit(X[tr], ytr)
            pred = model.predict_proba(X[te])[:, 1] if task == "auc" else model.predict(X[te])
            del model
            yte = y[te]
            s = evaluate(yte, pred, task)
            if s != s:
                continue
            n = evaluate(rng.permutation(yte), pred, task)
            scores.append(s)
            nulls.append(n if n == n else (0.5 if task == "auc" else 0.0))
            if fwd is not None:
                edges.append(fold_edge(t[te], pred, fwd[te]))
            if keep_pred:
                idx_all.append(np.flatnonzero(te))
                pred_all.append(pred)
    out = {"scores": scores, "nulls": nulls, "edges": edges}
    if keep_pred and idx_all:
        out["idx"], out["pred"] = np.concatenate(idx_all), np.concatenate(pred_all)
    return out


def zstat(scores, nulls, task):
    if len(scores) < 2:
        return float("nan")
    ref = max(float(np.mean(nulls)), 0.5 if task == "auc" else 0.0)
    se = float(np.std(scores, ddof=1)) / np.sqrt(len(scores))
    return (float(np.mean(scores)) - ref) / se if se > 0 else float("nan")


def edge_z(edges):
    e = np.array([x for x in edges if x == x])
    if len(e) < 2 or e.std(ddof=1) == 0:
        return float("nan"), float("nan")
    return float(e.mean()), float(e.mean() / (e.std(ddof=1) / np.sqrt(len(e))))


def perm_pvalue(y, pred, task, n=500, seed=5):
    """One-fold significance: how often do shuffled labels score at least as well?"""
    actual = evaluate(y, pred, task)
    rng = np.random.default_rng(seed)
    draws = np.array([evaluate(rng.permutation(y), pred, task) for _ in range(n)])
    return actual, float((draws >= actual).mean())


def pick_economics(p, idx, pred, k=3):
    """What the predictions are worth as picks: the top-k coins per bar against all coins.

    Mean forward return over the horizon of the k highest-scored coins at each bar, the same for
    every coin at those bars, and the difference net of one round trip. Windows overlap, so this is
    a description of the picks, not a tradable P&L.
    """
    d = p.iloc[idx][["bar_time", "fwd", "excess"]].copy()
    d["pred"] = pred
    d = d.dropna(subset=["fwd"])
    top = d.sort_values("pred", ascending=False).groupby("bar_time").head(k)
    by_bar_top = top.groupby("bar_time")["fwd"].mean()
    by_bar_all = d.groupby("bar_time")["fwd"].mean()
    spread = (by_bar_top - by_bar_all).dropna()
    return {"bars": int(len(spread)), "top_fwd": float(by_bar_top.mean()),
            "all_fwd": float(by_bar_all.mean()), "spread": float(spread.mean()),
            "spread_net": float(spread.mean() - COST),
            "hit": float((top["fwd"] > 0).mean())}


def main():
    t0 = time.time()
    s = config.load()
    kdir = Path(s.data_dir) / "klines_plus"
    if not kdir.exists():
        sys.exit("data/klines_plus is missing: the fresh-data download has not been run")
    syms = sorted({f.name.rsplit("_", 1)[0] for f in kdir.glob("*_4h.parquet")})
    print(f"building the panel from data/klines_plus ({len(syms)} symbols, Binance klines only)...")
    base = features.build_panel(kdir, syms, s.donchian_bars)
    print(f"{len(base):,} rows   {base.bar_time.min().date()} -> {base.bar_time.max().date()}")

    cells = dots.assert_causal(base[base["symbol"].isin(syms[:5] + ["BTCUSDT"])])
    print(f"relationship features causal: {cells:,} cells unchanged when the future is removed")
    panel = dots.add_dots(base)
    t_end = int(panel["bar_time"].max().value) + 1

    # ---- every (horizon) gets its own labels; features do not depend on the horizon ----------
    data = {}
    for h in HORIZONS:
        p = add_targets(panel, h)
        # BTC cannot beat itself: its excess return is identically 0 and its relative-strength
        # features are identically 0, so leaving it in hands the model a block of free negatives.
        is_btc = p["symbol"] == "BTCUSDT"
        ex = p["excess"].where(~is_btc)
        g = ex.groupby(p["bar_time"])
        pct, n = g.rank(pct=True), g.transform("count")
        p["t_top_third"] = np.where(n >= 6, (pct > 2 / 3).astype(float), np.nan)
        p.loc[ex.isna(), "t_top_third"] = np.nan
        for col in ("t_excess_sign", "t_excess_ret"):
            p.loc[is_btc, col] = np.nan
        data[h] = {
            "p": p,
            "t": p["bar_time"].astype("int64").to_numpy(),
            "X": {"base": ml.design(p).to_numpy(dtype=float),
                  "dots": dots.design_dots(p, ml.design).to_numpy(dtype=float)},
            "y": {k: p["t_" + k].to_numpy(dtype=float) for k in TARGETS},
            "fwd": p["fwd"].to_numpy(dtype=float),
            "mask": {"all": None,
                     "bull": p["regime_on"].fillna(False).to_numpy(dtype=bool),
                     "tradable": p["eligible"].fillna(False).to_numpy(dtype=bool)},
        }
    n_feat = {k: v.shape[1] for k, v in data[HORIZONS[0]]["X"].items()}
    print(f"features: base {n_feat['base']}, dots {n_feat['dots']}")
    print(f"periods: development < {DEV_END.date()} | holdout < {HOLD_END.date()} | "
          f"pristine to {panel.bar_time.max().date()}")

    # ---- the candidate list -------------------------------------------------------------------
    cands = []
    for fs in ("base", "dots"):                                   # rounds 1 and 2
        for tg in TARGETS:
            for h in HORIZONS:
                for kind in ("gbm", "linear"):
                    cands.append(dict(rnd=1 if fs == "base" else 2, fs=fs, target=tg, h=h,
                                      model=kind, subset="all"))
    for sub in ("bull", "tradable"):                              # round 3: row subsets
        for tg in TARGETS:
            if tg == "big_loss":
                continue
            for h in HORIZONS:
                cands.append(dict(rnd=3, fs="dots", target=tg, h=h, model="gbm", subset=sub))
    n_ret = sum(1 for c in cands if TARGETS[c["target"]][1])
    luck = float(np.sqrt(2 * np.log(n_ret)))
    print(f"\n{len(cands)} candidates ({n_ret} return-predicting). "
          f"Best-of-{n_ret} pure noise is expected near z = {luck:.2f}.\n")

    def run(c, t_from, t_to, keep_pred=False):
        d = data[c["h"]]
        return walk(d["X"][c["fs"]], d["y"][c["target"]], d["t"], TARGETS[c["target"]][0],
                    c["model"], c["h"], t_from, t_to, mask=d["mask"][c["subset"]],
                    keep_pred=keep_pred, fwd=d["fwd"])

    # ---- DEVELOPMENT: try everything ------------------------------------------------------------
    print("=" * 100)
    print("DEVELOPMENT (walk-forward, test windows before 2025). Every candidate, in parallel.")
    dev = Parallel(n_jobs=7, verbose=0)(
        delayed(run)(c, 0, int(DEV_END.value)) for c in cands)
    rows = []
    for c, r in zip(cands, dev):
        task = TARGETS[c["target"]][0]
        sc, nu = r["scores"], r["nulls"]
        em, ez = edge_z(r["edges"])
        rows.append(dict(c, metric=task, folds=len(sc),
                         mean=float(np.mean(sc)) if sc else float("nan"),
                         sd=float(np.std(sc, ddof=1)) if len(sc) > 1 else float("nan"),
                         null=float(np.mean(nu)) if nu else float("nan"),
                         z=zstat(sc, nu, task), edge=em, edge_z=ez))
    res = pd.DataFrame(rows)
    print(f"done in {time.time() - t0:.0f}s\n")

    def show(df, title):
        print(title)
        print(f"  {'rnd':>3s} {'features':8s} {'target':12s} {'h':>3s} {'model':7s} {'subset':9s} "
              f"{'folds':>5s} {'score':>8s} {'shuffled':>9s} {'z':>6s} | {'pick edge':>10s} "
              f"{'edge z':>7s} {'net':>9s}")
        for r in df.to_dict("records"):
            net = r["edge"] - COST if r["edge"] == r["edge"] else float("nan")
            print(f"  {r['rnd']:3d} {r['fs']:8s} {r['target']:12s} {r['h']:3d} {r['model']:7s} "
                  f"{r['subset']:9s} {r['folds']:5d} {r['mean']:8.4f} {r['null']:9.4f} "
                  f"{r['z']:6.2f} | {r['edge']:+10.3%} {r['edge_z']:7.2f} {net:+9.3%}")
        print()

    ctrl = res[res["target"] == "big_loss"].sort_values("z", ascending=False)
    show(ctrl.head(4), "POSITIVE CONTROL - 'falls more than 10%' (must be clearly above 0.50):")
    ctrl_ok = bool((ctrl["mean"] > 0.60).any())
    print(f"  -> {'the harness sees a real signal when one exists' if ctrl_ok else 'CONTROL FAILED: the harness is suspect and nothing below can be trusted'}\n")

    ret = res[res["target"].map(lambda k: TARGETS[k][1]) & (res["folds"] >= 10)]
    by_acc = ret.sort_values("z", ascending=False)
    show(by_acc.head(8), f"MOST ACCURATE, best 8 of {len(ret)} by the z of AUC / IC:")
    ret = ret.sort_values("edge_z", ascending=False)
    show(ret.head(12), "MOST PROFITABLE PICKS, best 12 by the z of the top-3 pick edge "
                       "(this is the ranking that counts):")
    print(f"  luck bar for the best of {n_ret}: z = {luck:.2f}. "
          f"{int((ret['edge_z'] > luck).sum())} candidate(s) above it on pick edge, "
          f"{int(((ret['edge'] - COST) > 0).sum())} with an edge larger than a round trip of "
          f"costs ({COST:.2%}).\n")
    by_family = ret.groupby(["target", "h"]).agg(
        best_score=("mean", "max"), best_edge=("edge", "max"), best_edge_z=("edge_z", "max"))
    print("  best development result by target and horizon:")
    print("  " + by_family.round(4).to_string().replace("\n", "\n  ") + "\n")

    # ---- HOLDOUT: the top 3, once each ---------------------------------------------------------
    top = ret.head(3).to_dict("records")
    print("=" * 100)
    print("HOLDOUT 2025-01 -> 2026-09. The top 3 development candidates, scored once each.")
    hold = Parallel(n_jobs=3)(
        delayed(run)(c, int(DEV_END.value), int(HOLD_END.value)) for c in top)
    final = []
    for c, r in zip(top, hold):
        task = TARGETS[c["target"]][0]
        z = zstat(r["scores"], r["nulls"], task)
        hem, hez = edge_z(r["edges"])
        c.update(hold_folds=len(r["scores"]),
                 hold_mean=float(np.mean(r["scores"])) if r["scores"] else float("nan"),
                 hold_null=float(np.mean(r["nulls"])) if r["nulls"] else float("nan"), hold_z=z,
                 hold_edge=hem, hold_edge_z=hez)
        print(f"  {c['fs']:5s} {c['target']:12s} h={c['h']:<3d} {c['model']:7s} {c['subset']:9s}")
        print(f"        accuracy  dev {c['mean']:.4f} (z {c['z']:.2f})  ->  holdout "
              f"{c['hold_mean']:.4f} over {c['hold_folds']} folds (shuffled {c['hold_null']:.4f}, "
              f"z {z:.2f})")
        print(f"        pick edge dev {c['edge']:+.3%} (z {c['edge_z']:.2f})  ->  holdout "
              f"{hem:+.3%} (z {hez:.2f}), net of costs {hem - COST:+.3%}")
        final.append(c)
    print()

    # ---- PRISTINE: data no research has seen ---------------------------------------------------
    print("=" * 100)
    print(f"PRISTINE {HOLD_END.date()} -> {panel.bar_time.max().date()}. "
          "Fresh Binance candles; one fit per candidate; never looked at before this line.")
    for c in final:
        d = data[c["h"]]
        task = TARGETS[c["target"]][0]
        r = walk(d["X"][c["fs"]], d["y"][c["target"]], d["t"], task, c["model"], c["h"],
                 int(HOLD_END.value), t_end + (c["h"] + 1) * BAR_NS,
                 mask=d["mask"][c["subset"]], test_days=400, keep_pred=True, threads=8)
        if "idx" not in r:
            print(f"  {c['target']} h={c['h']}: not enough pristine rows")
            c.update(pristine=float("nan"), pristine_p=float("nan"))
            continue
        y = d["y"][c["target"]][r["idx"]]
        actual, pv = perm_pvalue(y, r["pred"], task)
        eco = pick_economics(d["p"], r["idx"], r["pred"])
        c.update(pristine=actual, pristine_p=pv, pristine_rows=int(len(y)), **{
            "eco_" + k: v for k, v in eco.items()})
        print(f"  {c['fs']:5s} {c['target']:12s} h={c['h']:<3d} {c['model']:7s} {c['subset']:9s} "
              f"{task} {actual:.4f} on {len(y):,} rows, permutation p = {pv:.3f}")
        print(f"        top-3 picks per bar: {eco['top_fwd']:+.3%} forward vs {eco['all_fwd']:+.3%} "
              f"for all coins -> spread {eco['spread']:+.3%}, net of a round trip "
              f"{eco['spread_net']:+.3%}; {eco['hit']:.1%} of picks rose")
    print()

    # ---- verdict ----------------------------------------------------------------------------------
    print("=" * 100)
    print("VERDICT. Real = pick edge above the luck bar in development AND holdout pick edge")
    print("with z > 2.4 that is larger than a round trip of costs AND, on pristine data,")
    print("permutation p < 0.05 with picks that beat the average coin net of costs.")
    print("All three, or it is not a predictor worth trading.\n")
    any_real = False
    for c in final:
        dev_ok = c["edge_z"] > luck
        hold_ok = (c.get("hold_edge_z", float("nan")) > 2.4
                   and c.get("hold_edge", float("nan")) > COST)
        pri_ok = (c.get("pristine_p", 1.0) < 0.05
                  and c.get("eco_spread_net", float("nan")) > 0)
        real = dev_ok and hold_ok and pri_ok
        any_real |= real
        print(f"  {c['fs']} / {c['target']} / h={c['h']} / {c['model']} / {c['subset']}: "
              f"development {'pass' if dev_ok else 'FAIL'}, holdout {'pass' if hold_ok else 'FAIL'}, "
              f"pristine {'pass' if pri_ok else 'FAIL'}  ->  {'REAL' if real else 'not a predictor'}")
    print()
    if any_real:
        print("  Something survived all three. It is a candidate signal, not yet a strategy: it")
        print("  still has to be traded through backtest.run under the ship rule with costs.")
    else:
        print(f"  Nothing survived. {len(cands)} candidates, {n_ret} of them return-predicting, and")
        print("  none clears development, holdout and pristine data together.")
        if ctrl_ok:
            print("  The positive control worked, so this is a statement about the market, not about")
            print("  the harness: on Binance OHLCV at 4h-5d horizons, risk is forecastable and")
            print("  direction is not.")

    out = {"candidates": res.to_dict("records"), "promoted": final, "luck_bar": luck,
           "control_ok": ctrl_ok, "n_candidates": len(cands), "n_return": n_ret}
    (ROOT / "reports" / "predict_search.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"\n  written to reports/predict_search.json   ({time.time() - t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
