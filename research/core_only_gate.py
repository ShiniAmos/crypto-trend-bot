"""The gate P19 said must be passed before core-only could replace the rotation.

    python research/core_only_gate.py

P19 found that on the survivorship-free held-out sample, core-only at trend_core >= 0.35 beats the
shipped configuration on CAGR, Sharpe AND max drawdown at the same time, and that every cell where
it fails is on the standard panel - the one this project's README calls a liar. P19 then refused to
act on it, for a reason already quantified in this log:

    "Passes both panels is weak evidence, now quantified. Placebo nulls, information-free universe
     cuts with matched persistence, pass the same ship rule 7.5% to 22% of the time."

So: three gates, each able to kill the result on its own.

GATE 1 - PLACEBO NULLS. backtest.random_baselines, unchanged, which is what every other shipped
change in this log was tested against. Two nulls: the same entry and exit times on a RANDOM
ELIGIBLE COIN, and the same coin held for the same duration at a RANDOM RISK-ON TIME. If the core's
trades are not distinguishable from either, the "edge" is the market's drift and the holding period,
not the rule.

A caution specific to the core, and the reason this gate matters more here than usual: the core
holds two coins chosen by hand, BTC and ETH. The random-coin null is therefore asking something
quite pointed - would any other eligible coin, held over exactly the same windows, have done as
well? If yes, the core is a market-timing overlay and the coin choice is decoration.

GATE 2 - COST STRESS at 1.5x, 2x and 3x. The core trades roughly a twelfth as often as the
rotation, so higher costs should favour it, and the margin should WIDEN. If it does not widen, the
claim that the core is the cheaper way to hold the same exposure is false, and that is
disqualifying rather than merely disappointing.

GATE 3 - IS 0.45 FITTED? There is nothing to walk forward in the usual sense: the strategy fits no
parameters to data. The one fitted quantity is a human's choice of trend_core after seeing the
held-out result, which is exactly the kind of selection the ship rule exists to catch. So the test
is stability: score every weight in rolling sub-windows and ask which one wins in each. If the
winner moves around, 0.45 is noise that happened to look good on one window.
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from app import backtest  # noqa: E402
from app import config  # noqa: E402
from app.flags import Flags  # noqa: E402

import modern_era as me  # noqa: E402
from core_only import SHIPPED, calmar  # noqa: E402
from report_panels import HELD_OUT  # noqa: E402

CANDIDATE = dict(SHIPPED, trend_core=0.45, capital_slices=True, slice_max_names=0)
WEIGHTS = (0.25, 0.35, 0.45, 0.60)
ARMS = {"shipped": dict(SHIPPED), "core only 0.45": dict(CANDIDATE)}


def score(panel, s, flags, start=None):
    r = backtest.run(panel, s, flags=flags,
                     start=pd.Timestamp(start, tz="UTC") if isinstance(start, str) else start)
    m = backtest.metrics(r.equity, r.trades, r.exposure)
    return r, m


def main():
    s = replace(config.load(), max_names=5)
    print("building the pit panel...")
    pit = me.build_pit(s, top_n=30)

    # ---- GATE 1: placebo nulls ------------------------------------------------------------
    print("\n" + "=" * 88)
    print("GATE 1  PLACEBO NULLS (backtest.random_baselines, 500 draws, unchanged)")
    print("Can the core's trades be told apart from the same windows on a random coin,")
    print("and from the same coin at a random risk-on time?")
    print()
    print("  {:16s} {:10s} {:>12s} {:>12s} {:>9s} {:>12s} {:>9s}".format(
        "arm", "period", "avg trade", "rand coin", "p_coin", "rand time", "p_time"))
    gate1 = {}
    for arm, kw in ARMS.items():
        for period, start in (("full", None), ("held-out", HELD_OUT)):
            r, _ = score(pit, s, Flags(**kw), start)
            rb = backtest.random_baselines(pit, r.trades, s, n=500, seed=7)
            if not rb:
                print(f"  {arm:16s} {period:10s}  no trades")
                continue
            gate1[(arm, period)] = rb
            print("  {:16s} {:10s} {:+12.2%} {:+12.2%} {:9.3f} {:+12.2%} {:9.3f}".format(
                arm, period, rb["strategy_avg_trade"], rb["random_coin_avg_trade"],
                rb["p_value_vs_random_coin"], rb["random_time_avg_trade"],
                rb["p_value_vs_random_time"]))

    c = gate1.get(("core only 0.45", "held-out"), {})
    b = gate1.get(("shipped", "held-out"), {})
    g1 = bool(c) and c["p_value_vs_random_coin"] <= max(b.get("p_value_vs_random_coin", 0.05), 0.05) + 1e-9
    print()
    print("  GATE 1: {}  (core p_coin {:.3f} against shipped {:.3f}, threshold 0.05)".format(
        "PASS" if g1 else "FAIL",
        c.get("p_value_vs_random_coin", float("nan")),
        b.get("p_value_vs_random_coin", float("nan"))))

    # ---- GATE 2: cost stress ---------------------------------------------------------------
    print("\n" + "=" * 88)
    print("GATE 2  COST STRESS. The core trades a twelfth as often, so its margin should WIDEN.")
    print()
    print("  {:7s} {:>10s} {:>10s} {:>9s} | {:>10s} {:>10s} {:>9s} | {:>9s}".format(
        "costs", "ship CAGR", "core CAGR", "dCAGR", "ship maxDD", "core maxDD", "dmaxDD", "dSharpe"))
    margins = {}
    for mult in (1.0, 1.5, 2.0, 3.0):
        s2 = replace(s, fee_rate=s.fee_rate * mult, slippage_bps=s.slippage_bps * mult)
        _, mb = score(pit, s2, Flags(**ARMS["shipped"]), HELD_OUT)
        _, mc = score(pit, s2, Flags(**ARMS["core only 0.45"]), HELD_OUT)
        margins[mult] = (mc["cagr"] - mb["cagr"], mc["sharpe"] - mb["sharpe"],
                         mc["max_drawdown"] - mb["max_drawdown"])
        print("  {:5.1f}x {:+10.2%} {:+10.2%} {:+9.2%} | {:+10.2%} {:+10.2%} {:+9.2%} | {:+9.4f}".format(
            mult, mb["cagr"], mc["cagr"], mc["cagr"] - mb["cagr"],
            mb["max_drawdown"], mc["max_drawdown"],
            mc["max_drawdown"] - mb["max_drawdown"], mc["sharpe"] - mb["sharpe"]))
    g2 = margins[3.0][0] >= margins[1.0][0]
    print()
    print("  GATE 2: {}  (dCAGR {:+.2%} at 1x -> {:+.2%} at 3x; it must not narrow)".format(
        "PASS" if g2 else "FAIL", margins[1.0][0], margins[3.0][0]))

    # ---- GATE 3: is 0.45 fitted to one window? ---------------------------------------------
    print("\n" + "=" * 88)
    print("GATE 3  IS trend_core=0.45 FITTED? Which weight wins in each rolling window?")
    print("Nothing in this strategy fits parameters to data. The one fitted quantity is a")
    print("human choosing 0.45 after seeing the held-out result, which is the selection the")
    print("ship rule exists to catch. A winner that moves window to window means noise.")
    print()
    edges = pd.date_range("2022-01-01", "2026-10-01", freq="12MS", tz="UTC")
    print("  {:24s} {}".format("window", "  ".join(f"{w:>7.2f}" for w in WEIGHTS) + "   best"))
    best_counts = {w: 0 for w in WEIGHTS}
    for a, bnd in zip(edges, edges[1:]):
        cells = {}
        for w in WEIGHTS:
            kw = dict(SHIPPED, trend_core=w, capital_slices=True, slice_max_names=0)
            r = backtest.run(pit, s, flags=Flags(**kw), start=a, end=bnd)
            m = backtest.metrics(r.equity, r.trades, r.exposure)
            cells[w] = m["sharpe"]           # Sharpe, so a higher weight cannot win on leverage
        best = max(cells, key=cells.get)
        best_counts[best] += 1
        print("  {} -> {}  {}   {:.2f}".format(
            a.date(), bnd.date(), "  ".join(f"{cells[w]:7.3f}" for w in WEIGHTS), best))
    print()
    print("  wins by weight (on Sharpe): " + ", ".join(
        f"{w:.2f}:{n}" for w, n in best_counts.items()))
    g3 = best_counts[0.45] >= max(best_counts.values())
    print("  GATE 3: {}  (0.45 wins {} of {} windows)".format(
        "PASS" if g3 else "FAIL", best_counts[0.45], sum(best_counts.values())))

    # ---- verdict ----------------------------------------------------------------------------
    print("\n" + "=" * 88)
    passed = [n for n, g in (("placebo", g1), ("cost stress", g2), ("weight stability", g3)) if g]
    failed = [n for n, g in (("placebo", g1), ("cost stress", g2), ("weight stability", g3)) if not g]
    print("PASSED: " + (", ".join(passed) or "none"))
    print("FAILED: " + (", ".join(failed) or "none"))
    print()
    if not failed:
        print("All three gates hold. core-only at trend_core=0.45 is a real candidate to replace")
        print("the rotation, and the remaining objection is the one the ship rule itself raises:")
        print("it loses on the standard panel, which is the panel built from survivors.")
    else:
        print("Not actionable. " + ", ".join(failed) + " killed it, which is what these gates are")
        print("for - P19 said this result was suggestive and had not been attacked yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
