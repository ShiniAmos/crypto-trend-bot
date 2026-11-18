"""Is the breakout rotation an edge, or just exposure? If it is exposure, buy it more cheaply.

    python research/core_instead_of_rotation.py

THE CHAIN OF REASONING THAT LEADS HERE

P18: core-only keeps 80% of the held-out pit return for 47% of the drawdown and 23% of the trades.
It fails the ship rule on CAGR, so nothing could be acted on.

core_only_robustness.py then attacked that, because this log's own warning says the two panels'
worst drawdown is usually the same episode seen twice. It is not, here:

    shipped worst  -23.44%   Mar 2024 -> Oct 2024
    core only      -11.13%   Dec 2024 -> Apr 2025        178 days apart, different falls

and in EACH arm's worst window the other arm did better. Year by year, core-only has the shallower
drawdown in 5 of 5 years and the better Sharpe in 4 of 5. It has 5 falls past 5% against 10. The
drawdown advantage is a property, not an episode.

What it costs is CAGR, and only in the strong years: 2024 +18.45% -> +11.04%, 2026 +28.32% ->
+13.80%. The rotation adds return when things go well and drawdown when they do not.

THAT IS THE DEFINITION OF LEVERAGE, and this log already measured what leverage does here:

    "Every 'more return' knob is leverage. Across 43 portfolio runs exposure explains CAGR
     (r=+0.90) and drawdown (r=+0.93) nearly equally, but drawdown rises 1.29x faster per unit
     of exposure, because the 8% kill switch fires more often."

So the hypothesis is that the rotation is not a second edge but the exposure knob, bought
expensively - 77% of the trades, all of the fragile machinery, and a drawdown that rises faster
than the return.

If that is right, buying the same exposure through the core instead should dominate it: raise
`trend_core` until core-only reaches the shipped configuration's return, and see what the drawdown
does. The core has no stop, no rank, no breakout and one predicate, so the same exposure costs
almost no code and almost no trades.

THE TEST. core-only at trend_core from 0.25 to 0.60 against the shipped configuration, both panels,
full and held out. Reported with exposure and with return per unit of exposure, because comparing
two books at different exposure on CAGR alone is the mistake this log warns about twice.
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
from core_only import SHIPPED, calmar  # noqa: E402
from report_panels import HELD_OUT, one, standard_panel  # noqa: E402

WEIGHTS = (0.25, 0.35, 0.45, 0.60)

ARMS = {"shipped (both engines)": dict(SHIPPED)}
for w in WEIGHTS:
    ARMS[f"core only, trend_core={w:.2f}"] = dict(
        SHIPPED, trend_core=w, capital_slices=True, slice_max_names=0)

HEAD = "  {:28s} {:>8s} {:>8s} {:>8s} {:>7s} {:>7s} {:>8s} {:>7s}".format(
    "arm", "CAGR", "Sharpe", "maxDD", "Calmar", "expo", "CAGR/exp", "trades")


def row(arm, m):
    expo = m.get("exposure") or 0.0
    per = (m["cagr"] / expo) if expo else float("nan")
    return "  {:28s} {:+8.2%} {:8.4f} {:+8.2%} {:7.3f} {:7.1%} {:8.3f} {:7d}".format(
        arm, m["cagr"], m["sharpe"], m["maxdd"], calmar(m), expo, per,
        int(m.get("trades") or 0))


def main():
    s = replace(config.load(), max_names=5)
    print("building panels...")
    std, n_std = standard_panel(s)
    pit = me.build_pit(s, top_n=30)
    panels = {f"standard ({n_std} coins)": std, "pit (survivorship-free)": pit}

    results = {}
    for pname, panel in panels.items():
        for period, start in (("full", None), (f"held-out {HELD_OUT}", HELD_OUT)):
            print(f"\n{pname}, {period}")
            print(HEAD)
            for arm, kw in ARMS.items():
                m = one(panel, s, Flags(**kw), start=start)
                results[(pname, period, arm)] = m
                print(row(arm, m))

    # ---- the ship rule, for each weight, in the direction of replacing the rotation ---------
    print("\n" + "=" * 100)
    print("SHIP RULE: can core-only at a higher trend_core REPLACE the rotation outright?")
    print("No worse on CAGR, Sharpe AND maxDD on BOTH panels, held-out not failing.")
    print()
    base = "shipped (both engines)"
    verdicts = {}
    for arm in ARMS:
        if arm == base:
            continue
        cells, ok_all = [], True
        for pname in panels:
            for period, _ in (("full", None), (f"held-out {HELD_OUT}", HELD_OUT)):
                b, c = results[(pname, period, base)], results[(pname, period, arm)]
                checks = (c["cagr"] >= b["cagr"], c["sharpe"] >= b["sharpe"],
                          c["maxdd"] >= b["maxdd"])
                ok_all &= all(checks)
                cells.append((pname, period, b, c, checks))
        verdicts[arm] = ok_all
        print(f"  {arm}: {'SHIPS' if ok_all else 'no'}")
        for pname, period, b, c, checks in cells:
            names = ("cagr", "sharpe", "maxdd")
            detail = " ".join(f"{n}={'ok' if v else 'WORSE'}" for n, v in zip(names, checks))
            print(f"      {pname:26s} {period:20s} {detail}")
            print(f"          dCAGR {c['cagr'] - b['cagr']:+.2%}  "
                  f"dSharpe {c['sharpe'] - b['sharpe']:+.4f}  "
                  f"dmaxDD {c['maxdd'] - b['maxdd']:+.2%}  "
                  f"dCalmar {calmar(c) - calmar(b):+.3f}")
        print()

    winners = [a for a, v in verdicts.items() if v]
    print("=" * 100)
    if winners:
        print("VERDICT: the rotation IS replaceable. These ship over it:")
        for a in winners:
            print(f"  - {a}")
        print("\nThat would delete the rotation's three decision paths, its Donchian levels, its")
        print("ranking, its four filters and its slot count, and buy the same return through one")
        print("predicate with no stop and no rank.")
    else:
        print("VERDICT: no core weight replaces the rotation on the full rule.")
        print("Read dCalmar and CAGR/exp above: if those favour the core while CAGR does not, the")
        print("rotation is buying return with exposure rather than with an edge, and the right")
        print("question becomes which risk the owner of the money would rather hold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
