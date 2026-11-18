"""Is the breakout rotation earning its keep, or is the trend core the whole strategy?

    python research/core_only.py

THE QUESTION. The per-engine attribution in P12 says that on the held-out pit sample the core made
60.3% of the profit from a twelfth of the trades, with the shallower drawdown of the two engines.
The rotation produces eleven twelfths of the trades, under 40% of the profit, and it is where every
defect in this project has lived: three decision paths, Donchian levels, the ranking, four filters,
the slot count, the 30 Oct churn.

Every one of the ~660 variants in RESEARCH_LOG.md kept both engines. Nothing had ever asked whether
the rotation should exist at all, which made it the cheapest high-information experiment left.

HOW THE ARMS ARE BUILT, without touching shipped code:

    shipped        the configuration that cleared the ship rule (P15) and now runs in every book
    core_only      the same, with `capital_slices` on and `slice_max_names=0`. That sets the
                   rotation's slot cap to 0 so it never enters, while the core sleeve is sized from
                   `trend_core` and `signals.core_weight()` and is untouched by either flag.
    rotation_only  the same, with `trend_core=0`, which empties core_syms and turns the core off.

`max_names=0` would be the obvious way to silence the rotation and does not work: backtest.run has
`max_names = max_names or settings.max_names`, so 0 falls back to the configured value. Worth
knowing before trying it.

TWO QUESTIONS, NOT ONE. The ship rule answers "is core-only strictly better" - no worse on CAGR,
Sharpe AND maxDD on both panels, held-out not failing. It does not answer "is the rotation worth
what it costs", which is a different question with a different answer, so both are printed.

A CAVEAT ON THE ABSOLUTE NUMBERS. The pit panel is read from the cache in reports/, which
research/modern_era.py returns unconditionally without checking whether its inputs changed. That
does not affect this comparison - every arm is scored on the same panel, which is the point - but
do not quote the absolute figures from here as freshly built.
"""

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from app import config  # noqa: E402
from app.flags import Flags  # noqa: E402

import modern_era as me  # noqa: E402
from report_panels import HELD_OUT, one, standard_panel  # noqa: E402

# The shipped configuration, as P15 defined it and as .env now sets for every book.
SHIPPED = dict(abs_momentum=True, volume_min=0.8, funding_max_pct=0.8,
               trend_core=0.25, trend_core_symbols="BTCUSDT,ETHUSDT", core_dual_ma=True)

ARMS = {
    "shipped (both engines)": dict(SHIPPED),
    "core only": dict(SHIPPED, capital_slices=True, slice_max_names=0),
    "rotation only": dict(SHIPPED, trend_core=0.0),
}

BASE = "shipped (both engines)"
CAND = "core only"
HEAD = "  {:24s} {:>8s} {:>8s} {:>8s} {:>7s} {:>7s} {:>7s}".format(
    "arm", "CAGR", "Sharpe", "maxDD", "Calmar", "expo", "trades")


def calmar(m):
    """Return per unit of worst drawdown. The number that decides whether a book is sittable."""
    return (m["cagr"] / abs(m["maxdd"])) if m["maxdd"] else float("nan")


def fmt(m):
    return ("{:+8.2%} {:8.4f} {:+8.2%} {:7.3f} {:7.1%} {:7d}".format(
        m["cagr"], m["sharpe"], m["maxdd"], calmar(m),
        m.get("exposure") or 0.0, int(m.get("trades") or 0)))


def main():
    s = replace(config.load(), max_names=5)          # the shipped slot count
    print("costs: {:.2%} fee + {:.0f} bps slippage per side, max_names {}".format(
        s.fee_rate, s.slippage * 1e4, s.max_names))

    print("\nbuilding panels...")
    std, n_std = standard_panel(s)
    pit = me.build_pit(s, top_n=30)
    panels = {"standard ({} coins)".format(n_std): std, "pit (survivorship-free)": pit}
    periods = (("full", None), ("held-out " + HELD_OUT, HELD_OUT))

    results = {}
    for pname, panel in panels.items():
        for period, start in periods:
            print("\n{}, {}".format(pname, period))
            print(HEAD)
            for arm, kw in ARMS.items():
                m = one(panel, s, Flags(**kw), start=start)
                results[(pname, period, arm)] = m
                print("  {:24s} {}".format(arm, fmt(m)))

    # Cost stress, pit only. A thin edge dies first at higher costs, and the rotation trades
    # roughly 3.4x as often as the core, so it pays roughly 3.4x the fees for its return.
    print("\npit at 2x costs, full sample")
    print(HEAD)
    s2 = replace(s, fee_rate=s.fee_rate * 2, slippage_bps=s.slippage_bps * 2)
    for arm, kw in ARMS.items():
        m = one(pit, s2, Flags(**kw), start=None)
        results[("pit 2x costs", "full", arm)] = m
        print("  {:24s} {}".format(arm, fmt(m)))

    # ---- question 1: does core-only pass the ship rule, i.e. is it strictly better? ----------
    print("\n" + "=" * 86)
    print("SHIP RULE, applied to DELETING the rotation")
    print("core_only must be no worse on CAGR, Sharpe AND maxDD on BOTH panels,")
    print("with the held-out period not failing.")
    print()

    cells, all_ok = [], True
    for pname in panels:
        for period, _ in periods:
            b, c = results[(pname, period, BASE)], results[(pname, period, CAND)]
            # maxdd is negative, so "no worse" means not more negative
            checks = {"cagr": c["cagr"] >= b["cagr"],
                      "sharpe": c["sharpe"] >= b["sharpe"],
                      "maxdd": c["maxdd"] >= b["maxdd"]}
            ok = all(checks.values())
            all_ok &= ok
            cells.append((pname, period, b, c, checks))
            detail = " ".join("{}={}".format(k, "ok" if v else "WORSE") for k, v in checks.items())
            print("  {:26s} {:20s} {}   {}".format(
                pname, period, "PASS" if ok else "FAIL", detail))
            print("      dCAGR {:+.2%}   dSharpe {:+.4f}   dmaxDD {:+.2%}   dCalmar {:+.3f}".format(
                c["cagr"] - b["cagr"], c["sharpe"] - b["sharpe"],
                c["maxdd"] - b["maxdd"], calmar(c) - calmar(b)))

    print()
    if all_ok:
        print("VERDICT: the rotation adds nothing that survives measurement.")
        print("Deleting it removes most of this system's fragile surface area at no cost.")
    else:
        failed = sorted({k for _, _, _, _, ch in cells for k, v in ch.items() if not v})
        print("VERDICT: core_only does NOT pass. It is worse on: " + ", ".join(failed) + ".")
        print("That settles the ship rule and not the decision - see below.")

    # ---- question 2: what does the rotation buy, and what does it cost? ----------------------
    print("\n" + "=" * 86)
    print("WHAT THE ROTATION BUYS, AND WHAT IT COSTS")
    print("The ship rule asks 'strictly better'. This asks 'worth it', which is not the same.")
    print()
    for pname, period, b, c, _ in cells:
        print("  {}, {}".format(pname, period))
        print("      keeping it:  CAGR {:+.2%}  Sharpe {:.4f}  maxDD {:+.2%}  Calmar {:.3f}  "
              "{} trades".format(b["cagr"], b["sharpe"], b["maxdd"], calmar(b),
                                 int(b.get("trades") or 0)))
        print("      core only :  CAGR {:+.2%}  Sharpe {:.4f}  maxDD {:+.2%}  Calmar {:.3f}  "
              "{} trades".format(c["cagr"], c["sharpe"], c["maxdd"], calmar(c),
                                 int(c.get("trades") or 0)))
        tb = max(int(b.get("trades") or 1), 1)
        print("      core only keeps {:.0%} of the return for {:.0%} of the drawdown "
              "and {:.0%} of the trades".format(
                  c["cagr"] / b["cagr"] if b["cagr"] else float("nan"),
                  abs(c["maxdd"]) / abs(b["maxdd"]) if b["maxdd"] else float("nan"),
                  int(c.get("trades") or 0) / tb))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
