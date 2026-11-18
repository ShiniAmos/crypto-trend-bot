"""Is core-only's held-out advantage a property, or one episode?

    python research/core_only_robustness.py

WHY THIS IS THE RIGHT FOLLOW-UP TO P18. P18 found core-only better on every risk measure on the
held-out pit sample: Sharpe 0.6911 -> 0.9436, maxDD -23.44% -> -11.13%, Calmar 0.510 -> 0.864, for
80% of the return from 23% of the trades. It failed the ship rule on CAGR, so nothing can be acted
on, but the gradient across panels was suggestive enough to be worth attacking.

Two warnings already in this log say attack it rather than believe it:

  "Passes both panels is weak evidence, now quantified. Placebo nulls, information-free universe
   cuts with matched persistence, pass the same ship rule 7.5% to 22% of the time."

  "The two panels are not independent: the worst drawdown in nine years falls after 2022, so
   standard and held-out max drawdown are the same episode for nearly every variant."

That second one is the whole question. If the -23.44% and the -11.13% are the SAME drawdown episode
seen twice, then core-only's headline advantage is one observation, not a property, and it should
be read as "the core happened to be out during that fall" rather than "the core draws down less".

WHAT IS MEASURED
  1. WHEN each arm's worst drawdown happens, and how deep each arm was during the other's worst
     window. Same episode or different.
  2. Calendar-year CAGR, Sharpe and maxDD for both arms. An advantage present in one year only is
     an episode; present in most years it is a property.
  3. The drawdown distribution, not just its maximum: how many separate 5%+ falls each arm has, and
     their median depth. A single deep hole and a habit of shallow ones are different risks.

No new signal is computed anywhere here; this reads the two equity curves P18 produced.
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from app import backtest, config  # noqa: E402
from app.flags import Flags  # noqa: E402

import modern_era as me  # noqa: E402
from core_only import ARMS, BASE, CAND  # noqa: E402
from report_panels import HELD_OUT  # noqa: E402


def drawdown(eq):
    return eq / eq.cummax() - 1.0


def worst_window(eq):
    """(peak date, trough date, depth) of the deepest peak-to-trough fall."""
    dd = drawdown(eq)
    trough = dd.idxmin()
    peak = eq.loc[:trough].idxmax()
    return peak, trough, float(dd.min())


def episodes(eq, threshold=0.05):
    """Every separate fall deeper than `threshold`, as (peak, trough, depth).

    A new episode starts only once equity has made a fresh high, so one long grind is one episode
    rather than dozens.
    """
    dd = drawdown(eq)
    out, in_ep, peak_i, worst_i = [], False, None, None
    for i, v in dd.items():
        if v < 0 and not in_ep:
            in_ep, peak_i, worst_i = True, i, i
        elif in_ep and v < dd.loc[worst_i]:
            worst_i = i
        if in_ep and v >= 0:                      # recovered to a new high: episode closed
            if dd.loc[worst_i] <= -threshold:
                out.append((peak_i, worst_i, float(dd.loc[worst_i])))
            in_ep = False
    if in_ep and dd.loc[worst_i] <= -threshold:   # still underwater at the end
        out.append((peak_i, worst_i, float(dd.loc[worst_i])))
    return out


def yearly(eq):
    rows = {}
    for year, seg in eq.groupby(eq.index.year):
        if len(seg) < 50:
            continue
        r = np.log(seg.clip(lower=1e-12)).diff().dropna()
        years = max((seg.index[-1] - seg.index[0]).days / 365.25, 1e-9)
        rows[year] = {
            "cagr": float((seg.iloc[-1] / seg.iloc[0]) ** (1 / years) - 1),
            "sharpe": float(r.mean() / r.std() * np.sqrt(6 * 365)) if r.std() > 0 else 0.0,
            "maxdd": float(drawdown(seg).min()),
        }
    return rows


def main():
    s = replace(config.load(), max_names=5)
    print("building the pit panel...")
    pit = me.build_pit(s, top_n=30)
    start = pd.Timestamp(HELD_OUT, tz="UTC")

    curves = {}
    for arm in (BASE, CAND):
        r = backtest.run(pit, s, flags=Flags(**ARMS[arm]), start=start)
        eq = r.equity.dropna()
        eq.index = pd.to_datetime(eq.index, utc=True)
        curves[arm] = eq

    # ---- 1. is it the same episode? ---------------------------------------------------------
    print("\n" + "=" * 84)
    print("1. THE WORST DRAWDOWN: same episode, or different?")
    print()
    windows = {}
    for arm, eq in curves.items():
        peak, trough, depth = worst_window(eq)
        windows[arm] = (peak, trough, depth)
        print(f"  {arm:24s} {depth:+7.2%}   peak {peak.date()}  ->  trough {trough.date()}")

    print()
    for arm, (peak, trough, depth) in windows.items():
        other = CAND if arm == BASE else BASE
        seg = curves[other].loc[peak:trough]
        if len(seg):
            fall = float(seg.iloc[-1] / curves[other].loc[:peak].iloc[-1] - 1)
            print(f"  during {arm}'s worst window, {other} fell {fall:+.2%} "
                  f"(its own worst is {windows[other][2]:+.2%})")

    same = abs((windows[BASE][1] - windows[CAND][1]).days) <= 60
    print()
    print(f"  troughs are {abs((windows[BASE][1] - windows[CAND][1]).days)} days apart -> "
          f"{'THE SAME EPISODE' if same else 'DIFFERENT EPISODES'}")
    if same:
        print("  So the maxDD difference is one fall measured twice, not a general property.")
        print("  It says the core was less exposed in that particular fall.")

    # ---- 2. year by year -------------------------------------------------------------------
    print("\n" + "=" * 84)
    print("2. YEAR BY YEAR. An edge in one year is an episode; in most years it is a property.")
    print()
    yb, yc = yearly(curves[BASE]), yearly(curves[CAND])
    print(f"  {'year':6s} {'shipped CAGR':>13s} {'core CAGR':>11s} | "
          f"{'shipped Sh':>11s} {'core Sh':>9s} | {'shipped DD':>11s} {'core DD':>9s} | better")
    wins = {"cagr": 0, "sharpe": 0, "maxdd": 0}
    n = 0
    for year in sorted(set(yb) & set(yc)):
        b, c = yb[year], yc[year]
        n += 1
        for k in wins:
            if c[k] >= b[k]:
                wins[k] += 1
        flags = "".join(k[0].upper() if c[k] >= b[k] else "." for k in ("cagr", "sharpe", "maxdd"))
        print(f"  {year:<6d} {b['cagr']:+13.2%} {c['cagr']:+11.2%} | "
              f"{b['sharpe']:11.3f} {c['sharpe']:9.3f} | "
              f"{b['maxdd']:+11.2%} {c['maxdd']:+9.2%} | {flags}")
    print()
    print(f"  core-only was no worse in: CAGR {wins['cagr']}/{n} years, "
          f"Sharpe {wins['sharpe']}/{n}, maxDD {wins['maxdd']}/{n}")
    print("  (C/S/M in the last column = core-only no worse on that measure that year)")

    # ---- 3. the shape of the risk, not just its worst point --------------------------------
    print("\n" + "=" * 84)
    print("3. DRAWDOWN DISTRIBUTION. One deep hole and a habit of shallow ones differ.")
    print()
    for arm, eq in curves.items():
        eps = episodes(eq, 0.05)
        depths = [d for _, _, d in eps]
        under = float((drawdown(eq) < -0.01).mean())
        print(f"  {arm:24s} {len(eps):2d} falls past 5%   "
              f"median {np.median(depths) if depths else float('nan'):+.2%}   "
              f"worst {min(depths) if depths else float('nan'):+.2%}   "
              f"{under:.0%} of bars more than 1% underwater")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
