"""What is the 4-hour signal, and can it be turned into money? Development data only.

    python research/signal_anatomy.py

predict_search.py found one thing that survived development, holdout and fresh data alike:
gradient boosting on the relationship features predicts which coins land in the top third of the
cross-section over the NEXT 4-HOUR BAR. AUC 0.533 / 0.529 / 0.544 and a top-3 pick edge of
+0.074% / +0.052% / +0.047% per bar across the three periods. Real, stable - and a fifth of the
0.27% a round trip costs, so the pre-registered verdict was "not a predictor worth trading".

That verdict stands. This script asks the next question, and asks it ONLY of development data
(test windows before 2025) because the holdout and the fresh data have each been looked at once
for this family and choosing a trading rule by peeking at them again would spend them:

  1. WHERE DOES THE SIGNAL COME FROM? Refit with each feature group removed. If one group carries
     it, that says what the effect is - short-term reversal, BTC lead-lag, market state.
  2. IS THE EDGE CONCENTRATED? Mean forward spread by the model's rank and by its confidence. An
     edge that lives in the top few percent of predictions can be traded rarely, and a rule that
     trades rarely pays costs rarely.
  3. HOW LONG DOES IT LAST? Forward spread of picks made at bar t over the next 1, 2, 3, 6 and 12
     bars. If it keeps accumulating, a pick can be held and the cost amortised. If it is gone
     after one bar, it cannot.
  4. HOW FAST DO THE PICKS CHANGE? Turnover of the top-3 set per bar. Cost is paid on turnover,
     not on time.

Everything printed here is a description of development data. Nothing here is a result.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from app import config, features, ml  # noqa: E402

import dots_features as dots  # noqa: E402
import predict_search as ps  # noqa: E402
from train_predict_grid import add_targets  # noqa: E402

H = 1
GROUPS = {
    "own microstructure": ["ret_4h", "ret_4h_l1", "ret_4h_l2", "bar_pos", "vol_ratio_7_30",
                           "dd_30d", "up_30d", "vol_surge", "taker_chg_1d"],
    "BTC lead-lag": ["btc_ret_4h", "btc_ret_4h_l1", "btc_ret_1d", "btc_ret_30d", "btc_vol_7d",
                     "btc_bar_pos", "btc_taker_ratio"],
    "market state": ["breadth_sma50", "breadth_up_1d", "disp_1d", "mkt_ret_1d", "mkt_ret_7d",
                     "rel_mkt_1d"],
    "co-movement": ["beta_30d", "corr_30d"],
    "calendar": ["slot", "dow"],
}


def build():
    s = config.load()
    kdir = Path(s.data_dir) / "klines_plus"
    syms = sorted({f.name.rsplit("_", 1)[0] for f in kdir.glob("*_4h.parquet")})
    base = features.build_panel(kdir, syms, s.donchian_bars)
    p = add_targets(dots.add_dots(base), H)
    is_btc = p["symbol"] == "BTCUSDT"
    ex = p["excess"].where(~is_btc)
    g = ex.groupby(p["bar_time"])
    pct, n = g.rank(pct=True), g.transform("count")
    p["t_top_third"] = np.where(n >= 6, (pct > 2 / 3).astype(float), np.nan)
    p.loc[ex.isna(), "t_top_third"] = np.nan
    # forward returns over several holding periods, all entered at the next bar's open
    gs = p.groupby("symbol")
    entry = gs["open"].shift(-1)
    for k in (1, 2, 3, 6, 12):
        p[f"fwd_{k}"] = gs["open"].shift(-1 - k) / entry - 1
    return p


def main():
    print("building (development analysis only; nothing after 2024 is scored)...")
    p = build()
    Xdf = dots.design_dots(p, ml.design)
    X = Xdf.to_numpy(dtype=float)
    cols = list(Xdf.columns)
    y = p["t_top_third"].to_numpy(dtype=float)
    t = p["bar_time"].astype("int64").to_numpy()
    fwd = p["fwd"].to_numpy(dtype=float)
    dev_end = int(ps.DEV_END.value)

    # ---- 1. which feature group carries it -----------------------------------------------------
    print("\n" + "=" * 92)
    print("1. WHERE DOES THE SIGNAL COME FROM? The model refitted with one feature group removed.")
    print()

    def ablate(name, drop):
        keep = [i for i, c in enumerate(cols)
                if c not in drop and not (c.endswith("_xrank") and c[:-6] in drop)]
        r = ps.walk(X[:, keep], y, t, "auc", "gbm", H, 0, dev_end, fwd=fwd)
        em, ez = ps.edge_z(r["edges"])
        return name, len(keep), float(np.mean(r["scores"])), em, ez

    jobs = [("everything (57 features)", [])]
    jobs += [(f"without {g}", cs) for g, cs in GROUPS.items()]
    all_dots = [c for cs in GROUPS.values() for c in cs]
    jobs += [("base features only", all_dots)]
    for g, cs in GROUPS.items():
        jobs.append((f"base + {g} only", [c for c in all_dots if c not in cs]))
    out = Parallel(n_jobs=6)(delayed(ablate)(n, d) for n, d in jobs)
    print(f"  {'model':34s} {'features':>8s} {'AUC':>8s} {'pick edge':>10s} {'edge z':>7s}")
    for name, k, auc, em, ez in out:
        print(f"  {name:34s} {k:8d} {auc:8.4f} {em:+10.3%} {ez:7.2f}")

    # ---- out-of-sample predictions for the full model, development only -------------------------
    r = ps.walk(X, y, t, "auc", "gbm", H, 0, dev_end, fwd=fwd, keep_pred=True, threads=8)
    d = p.iloc[r["idx"]][["bar_time", "symbol"] + [f"fwd_{k}" for k in (1, 2, 3, 6, 12)]].copy()
    d["pred"] = r["pred"]
    d = d.dropna(subset=["fwd_1"])
    d = d[d.groupby("bar_time")["fwd_1"].transform("count") >= 8]
    for k in (1, 2, 3, 6, 12):
        d[f"x_{k}"] = d[f"fwd_{k}"] - d.groupby("bar_time")[f"fwd_{k}"].transform("mean")
    d["rank"] = d.groupby("bar_time")["pred"].rank(ascending=False, method="first")
    n_bars = d["bar_time"].nunique()
    print(f"\n  out-of-sample predictions: {len(d):,} rows over {n_bars:,} bars "
          f"({d.bar_time.min().date()} to {d.bar_time.max().date()})")

    # ---- 2. is the edge concentrated -------------------------------------------------------------
    print("\n" + "=" * 92)
    print("2. IS THE EDGE CONCENTRATED? Next-bar return minus the all-coin average, by model rank.")
    print()
    print(f"  {'rank in the bar':18s} {'mean spread':>12s} {'t-stat':>8s} {'share up':>9s}")
    for lo, hi, label in ((1, 1, "the top pick"), (2, 2, "second"), (3, 3, "third"),
                          (4, 6, "4th to 6th"), (7, 99, "everything else")):
        sub = d[(d["rank"] >= lo) & (d["rank"] <= hi)]
        by_bar = sub.groupby("bar_time")["x_1"].mean()
        tstat = by_bar.mean() / (by_bar.std(ddof=1) / np.sqrt(len(by_bar)))
        print(f"  {label:18s} {by_bar.mean():+12.3%} {tstat:8.2f} {(sub['fwd_1'] > 0).mean():9.1%}")

    print()
    print("  by the model's confidence (all predictions pooled, cut into tenths):")
    d["decile"] = pd.qcut(d["pred"], 10, labels=False, duplicates="drop")
    dec = d.groupby("decile").agg(pred=("pred", "mean"), spread=("x_1", "mean"), n=("x_1", "size"))
    print(f"  {'tenth':>6s} {'mean score':>11s} {'mean spread':>12s} {'rows':>8s}")
    for i, row in dec.iterrows():
        print(f"  {int(i) + 1:6d} {row['pred']:11.4f} {row['spread']:+12.3%} {int(row['n']):8d}")
    top1pct = d[d["pred"] >= d["pred"].quantile(0.99)]
    top5pct = d[d["pred"] >= d["pred"].quantile(0.95)]
    print(f"\n  the most confident 5% of predictions: {top5pct['x_1'].mean():+.3%} per bar "
          f"({len(top5pct):,} rows)")
    print(f"  the most confident 1% of predictions: {top1pct['x_1'].mean():+.3%} per bar "
          f"({len(top1pct):,} rows)")

    # ---- 3. how long does it last -----------------------------------------------------------------
    print("\n" + "=" * 92)
    print("3. HOW LONG DOES IT LAST? Cumulative spread of the top-3 picks made at bar t.")
    print()
    top3 = d[d["rank"] <= 3]
    print(f"  {'held for':>10s} {'cumulative spread':>18s} {'per bar':>10s} {'vs 0.27% cost':>14s}")
    for k in (1, 2, 3, 6, 12):
        m = top3.groupby("bar_time")[f"x_{k}"].mean().mean()
        print(f"  {k * 4:8d}h {m:+18.3%} {m / k:+10.3%} {m - ps.COST:+14.3%}")

    # ---- 4. turnover ---------------------------------------------------------------------------------
    print("\n" + "=" * 92)
    print("4. HOW FAST DO THE PICKS CHANGE? Cost is paid on turnover, not on time.")
    print()
    sets = top3.groupby("bar_time")["symbol"].apply(frozenset).sort_index()
    prev = sets.shift(1)
    changed = [len(a - b) / 3 for a, b in zip(sets.iloc[1:], prev.iloc[1:]) if isinstance(b, frozenset)]
    turn = float(np.mean(changed))
    edge = top3.groupby("bar_time")["x_1"].mean().mean()
    print(f"  share of the top-3 replaced each bar: {turn:.1%}")
    print(f"  cost per bar at that turnover: {turn * ps.COST:.3%}   edge per bar: {edge:+.3%}   "
          f"net: {edge - turn * ps.COST:+.3%}")
    for label, cost in (("0.27% (taker, as modelled)", 0.0027), ("0.15% (BNB fee discount, less slippage)", 0.0015),
                        ("0.08% (maker orders, no spread crossed)", 0.0008)):
        print(f"    at a round trip of {label}: net {edge - turn * cost:+.3%} per bar")
    print(f"\n  break-even round-trip cost at this turnover: {edge / turn:.3%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
