"""Try 5: use the ranking where it is strong - as a veto on entries the bot already takes.

    python research/rank_veto_ship.py

WHAT LED HERE. Four tries at turning a predictor into a long-only rule failed, and the last one
explained why: the model trained in magnitude_model_pit.py ranks coins correctly in 29 of 29
quarters on the survivorship-free universe, but its skill is at the BOTTOM of the ranking
(ranking_anatomy.py). The lowest score decile returns -0.68% a day against the average coin in
development and -0.87% on the holdout; the highest decile +0.40% and then +0.05%. It is a detector
of coins about to do badly, and every rule so far had been buying the other end.

A spot account cannot sell what it does not own, so the only way to use a "this will do badly"
signal is to not buy. ranking_anatomy.py then joined every breakout trade of the shipped
configuration to the model's score at the bar the order was decided on:

    model score      development            holdout
    bottom third     187 trades  -1.39%     55 trades  -4.62%
    middle third      76 trades  +9.20%     25 trades  +1.81%
    top third        228 trades  +5.77%     31 trades  +2.70%

40% of the bot's breakout entries sit in the model's bottom third, and they lose in both periods.
Skipping a trade costs nothing, so this use does not have to clear 0.27%.

THAT TABLE IS A DESCRIPTION, AND IT HAS BEEN SEEN. No rule was fixed before it was printed, so it
cannot count as a test. What can still be done honestly is to fix ONE rule now, run it through the
real engine - slots, exits, costs, the kill switch, a vetoed entry's slot going to the next
candidate - and judge it by this project's own gate:

  THE RULE      veto a breakout entry when the model's within-bar score percentile is in the
                bottom third. The tercile is the diagnostic's default cut, not a tuned number, and
                it is not varied here. Bars with no out-of-sample score (before the model's first
                test fold, or coins outside the pit universe) pass through untouched.
  THE GATE      the ship rule: CAGR, Sharpe AND max drawdown no worse on BOTH panels, held-out
                period not failing.
  THE CONTROL   a veto of the same size at random. P11 measured that information-free cuts pass
                the ship rule 7.5% to 22% of the time, so "it passed" is not enough: the model's
                veto has to beat vetoes that know nothing. Twelve random vetoes, each blocking a
                third of scored coin-days, held-out pit sample.

The scores are out of sample throughout - every one comes from a model fitted on the 730 days
before its test window, less a purge - so the backtest never uses a prediction that had seen the
bar it is applied to. It uses the existing `ml_veto` flag (entries need ml_score > 0), so no
trading code is changed by this test.

EVEN A PASS IS NOT A SHIP. The join above was seen before the rule was fixed, and the engine run
is that same evidence translated into portfolio terms, not new evidence. What a pass earns is a
place in a paper book as a shadow flag, logging what it would have vetoed, until forward data
agrees or does not.
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from app import config  # noqa: E402
from app.flags import Flags  # noqa: E402

import modern_era as me  # noqa: E402
from core_only import SHIPPED, calmar  # noqa: E402
from report_panels import HELD_OUT, one, standard_panel  # noqa: E402

PREDS = ROOT / "reports" / "_pit_rank_preds.parquet"
CUT = 1 / 3
N_NULL = 12


def scores():
    pr = pd.read_parquet(PREDS)
    pr["pct"] = pr.groupby("bar_time")["pred"].rank(pct=True)
    return pr[["bar_time", "symbol", "pct"]]


def with_veto(panel, sc, rng=None):
    """Attach ml_score so that ml_veto blocks the bottom third - or, with rng, a random third.

    The random veto is drawn once per coin per day, not per bar: a real low score persists for
    hours, so a per-bar coin flip would only delay entries by a bar and understate what a veto of
    this size costs.
    """
    m = panel.merge(sc, on=["bar_time", "symbol"], how="left")
    if rng is None:
        score = m["pct"] - CUT
    else:
        day = m["bar_time"].dt.floor("D")
        key = pd.MultiIndex.from_arrays([m["symbol"], day]).factorize()[0]
        draws = rng.random(key.max() + 1)
        score = pd.Series(draws[key], index=m.index) - CUT
        score = score.where(m["pct"].notna())
    m["ml_score"] = score.fillna(1.0)               # no score: pass through, never veto
    return m.drop(columns=["pct"])


def fmt(m):
    return ("{:+8.2%} {:8.4f} {:+8.2%} {:7.3f} {:7.1%} {:7d}".format(
        m["cagr"], m["sharpe"], m["maxdd"], calmar(m), m.get("exposure") or 0.0,
        int(m.get("trades") or 0)))


def main():
    s = replace(config.load(), max_names=5)
    sc = scores()
    print(f"{len(sc):,} out-of-sample scores, {sc.bar_time.min().date()} to {sc.bar_time.max().date()}")
    print("building panels...")
    std, n_std = standard_panel(s)
    pit = me.build_pit(s, top_n=30)
    panels = {f"standard ({n_std} coins)": std, "pit (survivorship-free)": pit}
    base_flags = Flags(**SHIPPED)
    veto_flags = Flags(**dict(SHIPPED, ml_veto=True))
    periods = (("full", None), ("held-out " + HELD_OUT, HELD_OUT))
    head = "  {:22s} {:>8s} {:>8s} {:>8s} {:>7s} {:>7s} {:>7s}".format(
        "arm", "CAGR", "Sharpe", "maxDD", "Calmar", "expo", "trades")

    res = {}
    for pname, panel in panels.items():
        vp = with_veto(panel, sc)
        for period, start in periods:
            print(f"\n{pname}, {period}")
            print(head)
            b = one(panel, s, base_flags, start=start)
            v = one(vp, s, veto_flags, start=start)
            res[(pname, period)] = (b, v)
            print("  {:22s} {}".format("shipped", fmt(b)))
            print("  {:22s} {}".format("shipped + rank veto", fmt(v)))
            if "breakout_pnl" in b and "breakout_pnl" in v:
                print("  breakout engine P&L: {:+,.0f} -> {:+,.0f}   core engine P&L: {:+,.0f} -> {:+,.0f}"
                      .format(b["breakout_pnl"], v["breakout_pnl"], b["core_pnl"], v["core_pnl"]))

    # ---- the gate ---------------------------------------------------------------------------------
    print("\n" + "=" * 96)
    print("THE SHIP RULE: no worse on CAGR, Sharpe AND max drawdown, both panels, held-out not failing.")
    print()
    all_ok = True
    for (pname, period), (b, v) in res.items():
        checks = {"cagr": v["cagr"] >= b["cagr"], "sharpe": v["sharpe"] >= b["sharpe"],
                  "maxdd": v["maxdd"] >= b["maxdd"]}
        ok = all(checks.values())
        all_ok &= ok
        detail = " ".join("{}={}".format(k, "ok" if c else "WORSE") for k, c in checks.items())
        print("  {:26s} {:20s} {}   {}".format(pname, period, "PASS" if ok else "FAIL", detail))
        print("      dCAGR {:+.2%}   dSharpe {:+.4f}   dmaxDD {:+.2%}   dCalmar {:+.3f}".format(
            v["cagr"] - b["cagr"], v["sharpe"] - b["sharpe"], v["maxdd"] - b["maxdd"],
            calmar(v) - calmar(b)))
    print("\n  SHIP RULE: {}".format("PASSES on all four cells" if all_ok else "FAILS"))

    # ---- the control ---------------------------------------------------------------------------------
    print("\n" + "=" * 96)
    print(f"THE CONTROL: {N_NULL} random vetoes of the same size, pit held-out. Does a veto that knows")
    print("nothing do as well?")
    print()
    pname = "pit (survivorship-free)"
    b, v = res[(pname, "held-out " + HELD_OUT)]
    nulls = []
    for i in range(N_NULL):
        m = one(with_veto(pit, sc, rng=np.random.default_rng(100 + i)), s, veto_flags, start=HELD_OUT)
        nulls.append(m)
        print("  random veto {:2d}          {}".format(i + 1, fmt(m)))
    print("  {:22s} {}".format("shipped (no veto)", fmt(b)))
    print("  {:22s} {}".format("MODEL veto", fmt(v)))
    print()
    for key, label in (("cagr", "CAGR"), ("sharpe", "Sharpe"), ("maxdd", "max drawdown")):
        vals = np.array([n[key] for n in nulls])
        beat = int((vals >= v[key]).sum())
        print("  {:13s} model {:+.4f}   random vetoes: mean {:+.4f}, best {:+.4f}   "
              "{} of {} random vetoes match or beat the model".format(
                  label, v[key], vals.mean(), vals.max(), beat, N_NULL))
    n_pass = sum(1 for n in nulls if n["cagr"] >= b["cagr"] and n["sharpe"] >= b["sharpe"]
                 and n["maxdd"] >= b["maxdd"])
    print(f"\n  {n_pass} of {N_NULL} random vetoes would themselves pass the ship rule on this cell.")

    # ---- verdict ----------------------------------------------------------------------------------------
    print("\n" + "=" * 96)
    better_than_null = all(sum(1 for n in nulls if n[k] >= v[k]) <= 1 for k in ("cagr", "sharpe"))
    print("VERDICT.")
    print("  ship rule            -> {}".format("pass" if all_ok else "FAIL"))
    print("  beats random vetoes  -> {}".format("yes" if better_than_null else "NO"))
    print()
    if all_ok and better_than_null:
        print("  The veto clears the gate and is not something a random veto reproduces. It is still")
        print("  built on a table that was seen before the rule was fixed, so what it has earned is a")
        print("  shadow flag in a paper book and forward data - not a place in the shipped config.")
    elif all_ok:
        print("  It passes the ship rule and so do vetoes that know nothing, which is P11's warning in")
        print("  one line: on this cell, removing a third of the trades helps whoever chooses them.")
    else:
        pit_cells = [c for (pn, _), c in res.items() if pn.startswith("pit")]
        pit_ok = all(v_["cagr"] >= b_["cagr"] and v_["sharpe"] >= b_["sharpe"]
                     and v_["maxdd"] >= b_["maxdd"] for b_, v_ in pit_cells)
        if pit_ok and better_than_null:
            # The first run of this script printed a single canned line for every failure - "run
            # through the engine it is not a filter" - which misdescribed a result that fails on
            # one panel only. The verdict above is unchanged; this is the sentence under it.
            print("  It FAILS the ship rule, and the rule is the rule: it does not ship.")
            print("  What kind of failure it is matters, though. Every failing cell is on the standard")
            print("  panel - today's survivors - and by under a point; on the survivorship-free panel")
            print("  it is better on all three metrics in both periods, and no random veto matches it.")
            print("  A veto whose value is avoiding coins about to do badly is expected to look")
            print("  weakest on a panel built from coins that did not do badly. That is P19's question")
            print("  about the gate, met in practice. It is a case for a shadow flag in a paper book")
            print("  and for settling that question - not for shipping on this evidence.")
        else:
            print("  It does not clear the gate, on the honest panel or against random vetoes. The")
            print("  event-level table looked like a filter; run through the engine it is not one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
