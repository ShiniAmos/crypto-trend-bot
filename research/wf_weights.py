"""P3-L: are the ranker's four weights worth re-fitting, walked forward?

Method, deliberately conservative:
  - the four score components are percentiles and do not depend on the weights, so they are
    computed once; only the blend changes;
  - folds match app/ml.py: 730 days train, 91 days test, with a purge gap of H+1 bars so the last
    training labels cannot overlap the test window;
  - on each training window the weight set with the best mean information coefficient is chosen
    and frozen for the NEXT fold only;
  - the search is a small simplex around the shipped 0.35 / 0.25 / 0.20 / 0.20, not a grid sweep;
  - an optional variant replaces component C (distance from the 50-day average) with acceleration,
    percentile(rs_7d - rs_30d).

Ship rule for this experiment: two consecutive held-out folds must beat the baseline on CAGR,
Sharpe and max drawdown. Anything less keeps the shipped weights.

    python research/wf_weights.py [--accel]
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import backtest, config, features, ml  # noqa: E402
from app.flags import Flags  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SHIPPED = Flags(abs_momentum=True, volume_min=0.8)
BASE_W = (0.35, 0.25, 0.20, 0.20)
STEP = 0.05
HORIZON = 6
TRAIN_DAYS, TEST_DAYS = 730, 91


def weight_grid(base=BASE_W, step=STEP):
    """A small simplex: each weight may move one step either way, and they must still sum to 1."""
    out = []
    for deltas in itertools.product((-step, 0.0, step), repeat=4):
        w = tuple(round(b + d, 10) for b, d in zip(base, deltas))
        if min(w) >= 0.05 and abs(sum(w) - 1.0) < 1e-9:
            out.append(w)
    return out


def components(panel, accel=False):
    """The four percentile components, computed once. Identical to app/features.score."""
    el = panel["eligible"]
    g = panel[el].groupby("bar_time")
    parts = pd.DataFrame(index=panel.index, columns=list("abcd"), dtype=float)
    parts.loc[el, "a"] = g["rs_7d"].rank(pct=True)
    parts.loc[el, "b"] = g["rs_30d"].rank(pct=True)
    if accel:
        acc = panel["rs_7d"] - panel["rs_30d"]
        parts.loc[el, "c"] = acc[el].groupby(panel.loc[el, "bar_time"]).rank(pct=True)
    else:
        parts.loc[el, "c"] = g["dist_sma50"].rank(pct=True)
    ra = panel["ret_30d"] / panel["vol_30d"]
    parts.loc[el, "d"] = ra[el].groupby(panel.loc[el, "bar_time"]).rank(pct=True)
    return parts


def mean_ic(parts, target, mask, w):
    """Mean per-bar rank correlation between a blended score and the forward excess return."""
    s = (parts[list("abcd")] * np.array(w)).sum(axis=1)
    d = pd.DataFrame({"score": s, "target": target, "bar": parts.index.map(lambda i: i)})[mask]
    d = d.dropna()
    if d.empty:
        return np.nan
    return d.groupby(BAR[d.index]).apply(
        lambda g: g["score"].rank().corr(g["target"].rank()) if len(g) >= 5 else np.nan,
        include_groups=False).mean()


def metrics_for(panel, s, flags, score_col, start, end):
    res = backtest.run(panel, s, flags=flags, score_col=score_col, start=start, end=end)
    if len(res.equity) < 3:
        return None
    return backtest.metrics(res.equity, res.trades, res.exposure)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accel", action="store_true",
                    help="replace component C with percentile(rs_7d - rs_30d)")
    args = ap.parse_args()

    s = config.load()
    u = json.loads((ROOT / "reports" / "_parity_universe.json").read_text())
    panel = features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars)

    global BAR
    BAR = panel["bar_time"]
    parts = components(panel, accel=args.accel)
    tgt = ml.add_target(panel.assign(_i=panel.index), HORIZON).set_index("_i")["target"]
    target = tgt.reindex(panel.index)

    grid = weight_grid()
    print(f"universe {len(u['symbols'])} coins | {len(grid)} weight sets | "
          f"component C = {'acceleration' if args.accel else 'dist_sma50'}")

    t = panel["bar_time"]
    first = t.min() + pd.Timedelta(days=TRAIN_DAYS)
    purge = pd.Timedelta(hours=4 * (HORIZON + 1))
    panel["score_wf"] = np.nan
    folds, test_start = [], first
    while test_start < t.max():
        test_end = test_start + pd.Timedelta(days=TEST_DAYS)
        tr = (t >= test_start - pd.Timedelta(days=TRAIN_DAYS)) & (t < test_start - purge)
        te = (t >= test_start) & (t < test_end)
        if tr.sum() > 5000 and te.any():
            scores = {}
            for w in grid:
                blended = (parts[list("abcd")] * np.array(w)).sum(axis=1)
                d = pd.DataFrame({"s": blended, "y": target, "bar": t})[tr].dropna()
                ic = d.groupby("bar").apply(
                    lambda g: g["s"].rank().corr(g["y"].rank()) if len(g) >= 5 else np.nan,
                    include_groups=False).mean()
                scores[w] = ic
            best = max(scores, key=lambda k: (scores[k] if pd.notna(scores[k]) else -9))
            blended = (parts[list("abcd")] * np.array(best)).sum(axis=1) * 100
            panel.loc[te, "score_wf"] = blended[te]
            folds.append({"test_start": test_start, "test_end": test_end,
                          "weights": best, "train_ic": scores[best],
                          "is_baseline_w": best == BASE_W})
        test_start = test_end

    oos = panel["score_wf"].notna()
    if not oos.any():
        sys.exit("no out-of-sample folds were produced")
    start = panel.loc[oos, "bar_time"].min()
    print(f"\n{len(folds)} folds, out-of-sample from {start.date()}")
    picked = pd.Series([f["weights"] for f in folds]).value_counts()
    print(f"weights chosen: {len(picked)} distinct; most common {picked.index[0]} "
          f"({picked.iloc[0]}/{len(folds)} folds); baseline weights chosen in "
          f"{sum(f['is_baseline_w'] for f in folds)} folds")

    base = metrics_for(panel, s, SHIPPED, "score", start, None)
    cand = metrics_for(panel, s, SHIPPED, "score_wf", start, None)
    print("\nOut-of-sample span, shipped filters both sides:")
    for tag, m in (("baseline weights", base), ("walk-forward weights", cand)):
        print(f"  {tag:<22} CAGR {m['cagr']:+.4f}  Sharpe {m['sharpe']:.4f}  "
              f"maxDD {m['max_drawdown']:.4f}  trades {m['trades']}")

    wins = []
    for f in folds:
        b = metrics_for(panel, s, SHIPPED, "score", f["test_start"], f["test_end"])
        c = metrics_for(panel, s, SHIPPED, "score_wf", f["test_start"], f["test_end"])
        if not b or not c:
            wins.append(False)
            continue
        wins.append(c["cagr"] >= b["cagr"] and c["sharpe"] >= b["sharpe"]
                    and c["max_drawdown"] >= b["max_drawdown"])
    streak = best_streak = 0
    for w in wins:
        streak = streak + 1 if w else 0
        best_streak = max(best_streak, streak)
    print(f"\nfolds beating baseline on all three metrics: {sum(wins)}/{len(wins)}; "
          f"longest consecutive run: {best_streak}")
    print("VERDICT:", "CANDIDATE (two consecutive folds)" if best_streak >= 2 else
          "REJECT - keep the shipped weights")
    out = ROOT / "reports" / f"wf_weights{'_accel' if args.accel else ''}.json"
    out.write_text(json.dumps({"folds": [{**f, "test_start": str(f["test_start"]),
                                          "test_end": str(f["test_end"]),
                                          "weights": list(f["weights"])} for f in folds],
                               "baseline": base, "candidate": cand,
                               "fold_wins": wins, "best_streak": best_streak}, indent=2,
                              default=str))
    print("saved", out)


if __name__ == "__main__":
    main()
