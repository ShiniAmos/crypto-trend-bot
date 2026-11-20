"""Where does the ranking skill live, and is any of it usable by a spot account?

    python research/ranking_anatomy.py

THE PUZZLE. magnitude_model_pit.py trained gradient boosting on the survivorship-free universe to
predict each coin's 24-hour return against the day's average coin. Its ranking is right in every
quarter it was scored on - Spearman IC +0.076 in development (22 of 22 quarters positive) and
+0.063 on the holdout (7 of 7) - and yet buying the top of that ranking lost 0.48% a trade on the
holdout (z -2.33) and 80% as a portfolio.

Both can only be true if the skill is somewhere other than the top. This script looks, and then
asks the only question that matters for a spot account, which cannot sell what it does not own:
does the ranking say anything about the trades this bot ALREADY takes?

  1. THE SHAPE. Realised 24h spread by the model's within-bar score decile, development and
     holdout. If the bottom deciles carry the correlation, the model is a detector of coins about
     to do badly, and a long-only rule was reading the wrong end of it.
  2. THE TOP, UP CLOSE. Mean against median and the 5th percentile in the top decile: how a bucket
     can win more often than it loses and still lose money.
  3. THE BOT'S OWN ENTRIES. Every breakout-rotation trade the shipped configuration takes on the
     pit panel, joined to the model's score at the bar the order was decided on. If low-scored
     entries lose and high-scored ones win, the ranking is a filter for trades already being paid
     for - the one use that costs nothing extra.

EVERYTHING HERE IS DESCRIPTION. The holdout has now been read for this model three times and no
rule in this file was chosen in advance; nothing below is a result, and a filter suggested by
section 3 would have to earn its place under the ship rule and on forward data like anything else.
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from app import backtest, config, ml  # noqa: E402
from app.flags import Flags  # noqa: E402

import dots_features as dots  # noqa: E402
import magnitude_model_pit as mm  # noqa: E402
import modern_era as me  # noqa: E402
import predict_search as ps  # noqa: E402
from core_only import SHIPPED  # noqa: E402

K = mm.K
PREDS = ROOT / "reports" / "_pit_rank_preds.parquet"


def predictions():
    p, _, _ = mm.load()
    if PREDS.exists():
        pr = pd.read_parquet(PREDS)
    else:
        X = dots.design_dots(p, ml.design).to_numpy(dtype=float)
        t = p["bar_time"].astype("int64").to_numpy()
        y = (p[f"fwd_{K}"] - p.groupby("bar_time")[f"fwd_{K}"].transform("mean")).to_numpy(dtype=float)
        print("walk-forward (gradient boosting, fresh model per fold)...")
        idx, pred = mm.walk_reg(X, y, t, "gbm", int(t.max()) + 1)
        pr = p.iloc[idx][["bar_time", "symbol"]].copy()
        pr["pred"] = pred
        pr.to_parquet(PREDS)
    d = pr.merge(p[["bar_time", "symbol", "dead", f"fwd_{K}"]], on=["bar_time", "symbol"], how="left")
    d[f"x_{K}"] = d[f"fwd_{K}"] - d.groupby("bar_time")[f"fwd_{K}"].transform("mean")
    d["pct"] = d.groupby("bar_time")["pred"].rank(pct=True)
    d["n"] = d.groupby("bar_time")["pred"].transform("count")
    return d


def main():
    d = predictions()
    first, last = d["bar_time"].min(), d["bar_time"].max()
    dev_hi = ps.DEV_END - pd.Timedelta(hours=4 * 13)
    hold_hi = min(ps.HOLD_END, last) - pd.Timedelta(hours=4 * 13)
    periods = (("development", first, dev_hi), ("holdout", ps.DEV_END, hold_hi))
    print(f"{len(d):,} out-of-sample scores, {first.date()} to {last.date()}")

    # ---- 1. the shape ---------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("1. THE SHAPE. Realised 24h return against the day's average coin, by within-bar score decile.")
    print()
    w = d[d["n"] >= 15].dropna(subset=[f"x_{K}"]).copy()
    w["decile"] = np.minimum((w["pct"] * 10).astype(int), 9)
    print(f"  {'decile':>7s} | {'development':>12s} {'median':>9s} {'win':>6s} | "
          f"{'holdout':>12s} {'median':>9s} {'win':>6s}")
    tab = {}
    for label, lo, hi in periods:
        seg = w[(w["bar_time"] >= lo) & (w["bar_time"] < hi)]
        tab[label] = seg.groupby("decile")[f"x_{K}"].agg(["mean", "median", lambda v: (v > 0).mean()])
    for dec in range(10):
        a, b = tab["development"].loc[dec], tab["holdout"].loc[dec]
        name = "lowest" if dec == 0 else ("highest" if dec == 9 else str(dec + 1))
        print(f"  {name:>7s} | {a.iloc[0]:+12.3%} {a.iloc[1]:+9.3%} {a.iloc[2]:6.1%} | "
              f"{b.iloc[0]:+12.3%} {b.iloc[1]:+9.3%} {b.iloc[2]:6.1%}")

    print()
    print("  long the highest decile, short the lowest, gross, per quarter (a description of the")
    print("  ranking, not a strategy: it needs a venue that allows shorting, and pays two sets of costs):")
    for label, lo, hi in periods:
        seg = w[(w["bar_time"] >= lo) & (w["bar_time"] < hi)]
        q = seg["bar_time"].dt.tz_localize(None).dt.to_period("Q")
        top = seg[seg["decile"] == 9].groupby(q)[f"x_{K}"].mean()
        bot = seg[seg["decile"] == 0].groupby(q)[f"x_{K}"].mean()
        ls = (top - bot).dropna()
        z = ls.mean() / (ls.std(ddof=1) / np.sqrt(len(ls)))
        print(f"    {label:12s} highest {top.mean():+.3%}   lowest {bot.mean():+.3%}   "
              f"difference {ls.mean():+.3%} a day, z {z:.2f}, positive in "
              f"{int((ls > 0).sum())} of {len(ls)} quarters")

    # ---- 2. the top, up close -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("2. THE TOP AND THE BOTTOM, UP CLOSE. How a bucket wins about half the time and loses money.")
    print()
    print(f"  {'bucket':16s} {'period':12s} {'mean':>9s} {'median':>9s} {'5th pct':>9s} {'95th pct':>9s} "
          f"{'worst':>9s} {'best':>9s}")
    for dec, name in ((9, "highest decile"), (0, "lowest decile")):
        for label, lo, hi in periods:
            v = w[(w["decile"] == dec) & (w["bar_time"] >= lo) & (w["bar_time"] < hi)][f"x_{K}"]
            print(f"  {name:16s} {label:12s} {v.mean():+9.3%} {v.median():+9.3%} "
                  f"{v.quantile(0.05):+9.2%} {v.quantile(0.95):+9.2%} {v.min():+9.1%} {v.max():+9.1%}")

    # ---- 3. the bot's own entries ------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("3. THE BOT'S OWN ENTRIES. Breakout-rotation trades of the shipped configuration on the pit")
    print("   panel, by the model's score at the bar each order was decided on.")
    print()
    s = replace(config.load(), max_names=5)
    pit = me.build_pit(s, top_n=30)
    r = backtest.run(pit, s, flags=Flags(**SHIPPED))
    tr = r.trades[r.trades["entry_reason"].astype(str).str.startswith("breakout")].copy()
    tr["bar_time"] = pd.to_datetime(tr["entry_time"], utc=True) - pd.Timedelta(hours=4)
    j = tr.merge(d[["bar_time", "symbol", "pct"]], on=["bar_time", "symbol"], how="left")
    matched = j["pct"].notna().mean()
    print(f"  {len(tr)} breakout trades in the backtest; {matched:.0%} have a model score at their "
          "decision bar (the rest predate the first out-of-sample fold)")
    j = j.dropna(subset=["pct"])
    j["bucket"] = pd.cut(j["pct"], [0, 1 / 3, 2 / 3, 1.0001], labels=["bottom third", "middle third",
                                                                   "top third"])
    print()
    print(f"  {'model score':14s} | {'development':>11s} {'mean':>8s} {'win':>6s} {'total':>9s} | "
          f"{'holdout':>8s} {'mean':>8s} {'win':>6s} {'total':>9s}")
    for b in ("bottom third", "middle third", "top third"):
        cells = []
        for label, lo, hi in periods:
            v = j[(j["bucket"] == b) & (j["bar_time"] >= lo) & (j["bar_time"] < hi)]["ret"]
            cells.append((len(v), v.mean() if len(v) else float("nan"),
                          (v > 0).mean() if len(v) else float("nan"), v.sum() if len(v) else 0.0))
        (n1, m1, w1, t1), (n2, m2, w2, t2) = cells
        print(f"  {b:14s} | {n1:11d} {m1:+8.2%} {w1:6.1%} {t1:+9.1%} | "
              f"{n2:8d} {m2:+8.2%} {w2:6.1%} {t2:+9.1%}")
    for label, lo, hi in periods:
        seg = j[(j["bar_time"] >= lo) & (j["bar_time"] < hi)]
        if len(seg) > 10:
            ic = seg["pct"].corr(seg["ret"], method="spearman")
            print(f"  {label}: rank correlation between score and trade return {ic:+.3f} "
                  f"over {len(seg)} trades")
    print()
    print("  where the bot's entries sit in the model's ranking:")
    print("    " + "   ".join(f"{b}: {(j['bucket'] == b).mean():.0%}"
                             for b in ("bottom third", "middle third", "top third")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
