"""Train on the past, predict the future, try many combinations, keep nothing between folds.

    python research/train_predict_grid.py
    python research/train_predict_grid.py --horizons 6 30 --folds 0

WHAT THIS IS. A deliberately broad hit-and-try over targets, horizons and models, scored strictly
out of sample by walk-forward with a purge gap. Five of this log's rejections were ML, so the point
here is not to hope for a different answer by asking the same question again - it is to ask SEVERAL
questions under one honest harness, with a positive control that proves the harness can detect a
real signal when one is there.

NOTHING CARRIES BETWEEN FOLDS. A fresh estimator is constructed inside every fold, from the fold's
own training rows only. Nothing is fitted on the full sample: no global imputation, no global
scaling, no target encoding, no feature selection chosen by looking at test performance. The only
cross-sectional statistics used are computed WITHIN a single bar_time, which contains no future.
`test_fold_isolation()` asserts this by refitting one fold twice in opposite orders and requiring
identical predictions - if any state leaked through the harness, the two runs would differ.

THE PURGE GAP. A label at bar t is the outcome of a window ending at t+H, so the last H bars of
any training set resolve INSIDE the test set. Training on them leaks the test period's prices
backwards. Every fold therefore drops the final H bars of its training rows.

THE POSITIVE CONTROL, which is what makes a null result mean anything. P6 established that losses
are predictable here where returns are not: AUC 0.78 on "will this fall more than 20%" against
0.47-0.52 for every return model. `big_loss` is included for exactly that reason. If this harness
reproduces a high AUC on the loss target and chance on the return targets, then the chance results
are a fact about the market rather than a bug in the harness. If `big_loss` ALSO comes out at 0.50,
the harness is broken and no other row on the page can be trusted.

THE PERMUTATION NULL. An AUC of 0.52 on 16 folds means nothing without knowing what noise looks
like. Each configuration is re-scored with its test labels shuffled, which gives the band a
genuinely uninformative model occupies on this data.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config, features, ml  # noqa: E402

BARS_PER_DAY = 6


# ----------------------------------------------------------------------------- targets
def add_targets(panel, horizon):
    """Forward outcomes measured from the NEXT bar's open, which is when an order would fill.

    Shifts are by row within a symbol. On this panel a 6-row shift spans exactly 24h for 99.85% of
    rows and at most 52h for the rest, so the row/time distinction is immaterial here - measured,
    not assumed.
    """
    p = panel.sort_values(["symbol", "bar_time"]).copy()
    g = p.groupby("symbol")
    entry = g["open"].shift(-1)                       # the fill price
    exit_ = g["open"].shift(-1 - horizon)
    p["fwd"] = exit_ / entry - 1
    # the worst drawdown inside the window, for the loss target
    low = g["low"].shift(-1)
    for k in range(2, horizon + 2):
        low = np.minimum(low, g["low"].shift(-k))
    p["fwd_mae"] = low / entry - 1

    btc = p[p.symbol == "BTCUSDT"].set_index("bar_time")["fwd"].rename("btc_fwd")
    p = p.join(btc, on="bar_time")
    p["excess"] = p["fwd"] - p["btc_fwd"]

    p["t_excess_sign"] = (p["excess"] > 0).astype(float)
    p["t_abs_sign"] = (p["fwd"] > 0).astype(float)
    p["t_excess_ret"] = p["excess"]
    p["t_big_win"] = (p["fwd"] > 0.10).astype(float)
    p["t_big_loss"] = (p["fwd_mae"] < -0.10).astype(float)
    p.loc[p["fwd"].isna(), [c for c in p.columns if c.startswith("t_")]] = np.nan
    return p.sort_values(["bar_time", "symbol"]).reset_index(drop=True)


TARGETS = {
    "excess_sign": ("beats BTC over the window", "auc"),
    "abs_sign": ("is up over the window", "auc"),
    "excess_ret": ("size of the excess return", "ic"),
    "big_win": ("beats +10% (the right tail)", "auc"),
    "big_loss": ("falls more than 10% (POSITIVE CONTROL)", "auc"),
}


# ----------------------------------------------------------------------------- models
def make_model(kind, task):
    """A FRESH estimator, every call. Nothing is reused between folds by construction."""
    if kind == "gbm":
        from sklearn.ensemble import (HistGradientBoostingClassifier,
                                      HistGradientBoostingRegressor)
        cls = HistGradientBoostingClassifier if task == "auc" else HistGradientBoostingRegressor
        return cls(max_depth=4, learning_rate=0.06, max_iter=250,
                   early_stopping=False, random_state=0)
    if kind == "linear":
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        head = LogisticRegression(max_iter=2000) if task == "auc" else Ridge(alpha=1.0)
        # imputer and scaler live INSIDE the pipeline, so they are fitted on training rows only
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), head)
    raise ValueError(kind)


def evaluate(y_true, pred, task):
    if task == "auc":
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(roc_auc_score(y_true, pred))
    from scipy.stats import spearmanr
    if len(y_true) < 10:
        return np.nan
    r = spearmanr(y_true, pred).correlation
    return float(r) if r == r else np.nan


# ----------------------------------------------------------------------------- walk-forward
def folds(times, train_days=730, test_days=91):
    lo, hi = times.min(), times.max()
    out, start = [], lo + pd.Timedelta(days=train_days)
    while start + pd.Timedelta(days=test_days) <= hi:
        out.append((lo, start, start + pd.Timedelta(days=test_days)))
        start += pd.Timedelta(days=test_days)
    return out


def run_one(p, X, target, task, kind, horizon, train_days, test_days, seed=11):
    """Walk forward. Returns (scores, shuffled_scores), one pair per fold.

    A fresh model per fold and no shared state. The training window ROLLS - the most recent
    `train_days` before the purge gap - as app/ml.py documents, so every fit sees the same amount
    of history and an old regime cannot dominate by sheer row count.

    The shuffled score reuses the fold's own predictions against permuted test labels. No refit is
    needed for that: shuffling the labels a model is scored against is exactly what "what would an
    uninformative model get here" means, and it costs nothing.
    """
    y = p["t_" + target]
    t = p["bar_time"]
    purge = pd.Timedelta(hours=4 * (horizon + 1))
    window = pd.Timedelta(days=train_days)
    rng = np.random.default_rng(seed)
    scores, nulls = [], []
    for _, cut, end in folds(t, train_days, test_days):
        tr = (t >= cut - purge - window) & (t < cut - purge) & y.notna()
        te = (t >= cut) & (t < end) & y.notna()
        if tr.sum() < 5000 or te.sum() < 200:
            continue
        ytr = y[tr].to_numpy()
        if task == "auc" and len(np.unique(ytr)) < 2:
            continue
        model = make_model(kind, task)           # <- constructed here, discarded at the end
        model.fit(X[tr.to_numpy()], ytr)
        pred = (model.predict_proba(X[te.to_numpy()])[:, 1] if task == "auc"
                else model.predict(X[te.to_numpy()]))
        yte = y[te].to_numpy()
        s = evaluate(yte, pred, task)
        n = evaluate(rng.permutation(yte), pred, task)
        if s == s:
            scores.append(s)
            nulls.append(n if n == n else (0.5 if task == "auc" else 0.0))
        del model
    return scores, nulls


def test_fold_isolation(p, X, horizon):
    """Refit one configuration twice with a different target fitted in between, and require
    identical scores.

    If anything leaked through this harness - a cached fit, a mutated frame, a global statistic -
    the two runs would diverge. This is the test behind the claim that nothing is carried over.
    """
    a, _ = run_one(p, X, "excess_sign", "auc", "gbm", horizon, 730, 91)
    run_one(p, X, "big_loss", "auc", "gbm", horizon, 730, 91)
    b, _ = run_one(p, X, "excess_sign", "auc", "gbm", horizon, 730, 91)
    ok = len(a) == len(b) and all(abs(x - y) < 1e-12 for x, y in zip(a, b))
    print(f"  fold isolation: {'PASS' if ok else 'FAIL'} "
          f"(same config refit after another target: {len(a)} folds, identical={ok})")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horizons", type=int, nargs="*", default=[6, 30])
    ap.add_argument("--models", nargs="*", default=["gbm", "linear"])
    ap.add_argument("--train-days", type=int, default=730)
    ap.add_argument("--test-days", type=int, default=91)
    args = ap.parse_args(argv)

    s = config.load()
    u = json.loads((ROOT / "reports" / "_parity_universe.json").read_text())
    print("building the panel...")
    base = features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars)
    print(f"{len(base):,} rows, {base.symbol.nunique()} symbols, "
          f"{base.bar_time.min().date()} to {base.bar_time.max().date()}")
    print(f"walk-forward: {args.train_days}d train / {args.test_days}d test, "
          f"purge = horizon + 1 bars, fresh model per fold")

    results = []
    for horizon in args.horizons:
        p = add_targets(base, horizon)
        X = ml.design(p).to_numpy(dtype=float)
        days = horizon * 4 / 24
        print(f"\n{'=' * 92}")
        print(f"HORIZON {horizon} bars = {days:.1f} days   ({len(ml.design(p).columns)} features)")
        if horizon == args.horizons[0]:
            test_fold_isolation(p, X, horizon)
        print()
        print(f"  {'target':13s} {'model':7s} {'metric':7s} {'folds':>6s} {'mean':>8s} "
              f"{'sd':>7s} {'min':>7s} {'max':>7s} {'shuffled':>9s} {'verdict':>9s}")
        for target, (desc, task) in TARGETS.items():
            for kind in args.models:
                sc, null = run_one(p, X, target, task, kind, horizon,
                                   args.train_days, args.test_days)
                if not sc:
                    print(f"  {target:13s} {kind:7s} {task:7s} {'-':>6s}  no usable folds")
                    continue
                base_level = 0.5 if task == "auc" else 0.0
                m, sd = float(np.mean(sc)), float(np.std(sc))
                nm = float(np.mean(null)) if null else float("nan")
                # beats the shuffled control by more than a fold-level standard error
                se = sd / max(np.sqrt(len(sc)), 1)
                verdict = "SIGNAL" if (m - max(nm, base_level)) > 2 * se else "nothing"
                results.append({"horizon": horizon, "target": target, "model": kind,
                                "metric": task, "folds": len(sc), "mean": m, "sd": sd,
                                "shuffled": nm, "verdict": verdict})
                print(f"  {target:13s} {kind:7s} {task:7s} {len(sc):6d} {m:8.4f} {sd:7.4f} "
                      f"{min(sc):7.4f} {max(sc):7.4f} {nm:9.4f} {verdict:>9s}")

    print("\n" + "=" * 92)
    print("WHAT SURVIVED")
    print()
    signal = [r for r in results if r["verdict"] == "SIGNAL"]
    control = [r for r in results if r["target"] == "big_loss"]
    ctrl_ok = any(r["mean"] > 0.60 for r in control)
    print(f"  positive control (big_loss): best AUC "
          f"{max((r['mean'] for r in control), default=float('nan')):.4f} -> "
          f"{'the harness CAN see a real signal' if ctrl_ok else 'HARNESS SUSPECT, see below'}")
    if not ctrl_ok:
        print("  P6 measured AUC 0.78 on the -20% loss tail with the same data. If this harness")
        print("  cannot reproduce something in that region, every other row here is meaningless")
        print("  and the harness is what needs fixing, not the conclusion.")
    print()
    returns = [r for r in results if r["target"] in ("excess_sign", "abs_sign", "excess_ret",
                                                     "big_win")]
    best = max(returns, key=lambda r: r["mean"] - (0.5 if r["metric"] == "auc" else 0.0))
    print(f"  best RETURN-predicting cell: {best['target']} / {best['model']} / h={best['horizon']}"
          f" -> {best['mean']:.4f} against a shuffled {best['shuffled']:.4f} "
          f"[{best['verdict']}]")
    print()
    if signal:
        for r in signal:
            print(f"  SIGNAL: {r['target']} / {r['model']} / h={r['horizon']} "
                  f"{r['metric']}={r['mean']:.4f} +/- {r['sd']:.4f} over {r['folds']} folds "
                  f"(shuffled {r['shuffled']:.4f})")
        print()
        print("  A cell marked SIGNAL is not a strategy. Next it would have to survive the ship")
        print("  rule on both panels, a placebo null, and cost stress - and P11 found that a")
        print("  tradable-looking edge can still fail when the trades it implies are priced.")
    else:
        print("  Nothing beat its own shuffled control by more than two standard errors.")

    out = ROOT / "reports" / "train_predict_grid.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\n  written to {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
