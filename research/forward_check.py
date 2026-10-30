"""Would it still work next year? The four tests a rule must pass to be called forward-robust.

research/modern_era.py asks whether a rule works in 2022-2026. That is necessary and nowhere near
sufficient: fitting the modern era is the same mistake as fitting 2020, one sample later. A rule
that "stands in the market and stays profitable" has to survive the things that will actually
differ about next year, and only four of those can be tested on history:

  1. ROLLING STABILITY   Sharpe and return in every rolling 12-month window from 2022 on, stepped
                         quarterly. The average over an era hides a rule that made everything in
                         one quarter. The number that matters is the SHARE OF WINDOWS POSITIVE and
                         the WORST window - a rule you can hold through next year is one whose bad
                         windows you could sit through.
  2. COST STRESS         1.5x and 2.0x fees and slippage. Spreads widen, size grows, a $100 book
                         pays worse than a backtest assumes. An edge that dies at 1.5x was never an
                         edge, it was a rebate.
  3. TIME SPLIT          the 2025-2026 half, never used to choose anything. The only honest proxy
                         for "next year" that history contains.
  4. PLACEBO NULLS       random-coin and random-time baselines over the same bars, so "it works" is
                         distinguished from "the market went up while it was invested".

EXPOSURE is carried through every table, because a rule that improves Sharpe by standing in cash
is not standing in the market, and standing in the market was the requirement.

PASS is pre-declared, before any number is seen:
    - positive in at least 70% of rolling 12-month windows
    - worst rolling window no worse than the baseline's worst
    - still clears the baseline at 1.5x costs
    - clears the baseline on the 2025-2026 half
    - beats the random-coin null at p <= 0.05
Anything less is reported as it is and left OFF.

    python research/forward_check.py                      # the shipped configuration
    python research/forward_check.py vol_target=0.8       # shipped + one flag
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import backtest, config  # noqa: E402
from app.flags import Flags  # noqa: E402

import modern_era as me  # noqa: E402

MODERN = me.MODERN
WINDOW_MONTHS = 12
STEP_MONTHS = 3
MIN_WIN_RATE = 0.70


def parse(args):
    """`trend_core=0.25 trend_core_symbols=BTCUSDT` -> flags on top of the shipped set.

    Typed from the dataclass field rather than guessed from the text, so a symbol list stays a
    string and a weight stays a float.
    """
    from dataclasses import fields
    types = {f.name: type(f.default) for f in fields(Flags)}
    extra = {}
    for a in args:
        k, _, v = a.partition("=")
        if not v:
            extra[k] = True
            continue
        cast = types.get(k, str)
        extra[k] = v if cast is str else (cast(float(v)) if cast is int else cast(v))
    return extra


def _ts(x):
    """Accept a date string or an already-tz-aware Timestamp; the rolling loop passes the latter."""
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def one(panel, s, flags, start, end=None):
    r = backtest.run(panel, s, flags=flags, start=_ts(start),
                     end=_ts(end) if end is not None else None)
    m = backtest.metrics(r.equity, r.trades, r.exposure)
    return m, r


def rolling(panel, s, flags, label):
    """Every rolling 12-month window, stepped a quarter at a time."""
    lo = pd.Timestamp(MODERN, tz="UTC")
    hi = panel["bar_time"].max()
    rows = []
    start = lo
    while start + pd.DateOffset(months=WINDOW_MONTHS) <= hi:
        end = start + pd.DateOffset(months=WINDOW_MONTHS)
        m, _ = one(panel, s, flags, start, end)
        rows.append({"from": start.date(), "to": end.date(), "return": m["cagr"],
                     "sharpe": m["sharpe"], "maxdd": m["max_drawdown"],
                     "exposure": m.get("avg_exposure"), "trades": m.get("trades", 0)})
        start = start + pd.DateOffset(months=STEP_MONTHS)
    df = pd.DataFrame(rows)
    df.attrs["label"] = label
    return df


def summarise(df):
    r = df["return"]
    return {"windows": len(df), "share_positive": float((r > 0).mean()),
            "median_return": float(r.median()), "worst_return": float(r.min()),
            "best_return": float(r.max()),
            "worst_sharpe": float(df["sharpe"].min()),
            "mean_exposure": float(df["exposure"].mean())}


def main():
    extra = parse(sys.argv[1:])
    s = config.load()
    panel = me.build_pit(s)
    base_flags = Flags(**me.SHIPPED)
    test_flags = Flags(**{**me.SHIPPED, **extra})
    name = "shipped + " + ", ".join(f"{k}={v}" for k, v in extra.items()) if extra else "shipped"
    print(f"testing: {name}")
    print(f"panel {panel['bar_time'].min().date()} .. {panel['bar_time'].max().date()}\n")

    # ---------------------------------------------------------------- 1. rolling stability
    rb = rolling(panel, s, base_flags, "baseline")
    rt = rolling(panel, s, test_flags, "candidate") if extra else rb
    print(f"1. ROLLING {WINDOW_MONTHS}-MONTH WINDOWS from {MODERN}, stepped {STEP_MONTHS} months")
    print(rt.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    sb, st = summarise(rb), summarise(rt)
    cmp_roll = pd.DataFrame([{"run": "baseline", **sb}] + ([{"run": "candidate", **st}]
                                                           if extra else []))
    print("\n" + cmp_roll.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"   -> share of windows positive: {st['share_positive']:.0%} "
          f"(bar is {MIN_WIN_RATE:.0%})")
    print(f"   -> worst 12-month window: {st['worst_return']:+.2%} "
          f"(baseline {sb['worst_return']:+.2%})")

    # ---------------------------------------------------------------- 2. cost stress
    print("\n2. COST STRESS over the whole modern era")
    cost_rows = []
    for mult in (1.0, 1.5, 2.0):
        ss = replace(s, fee_rate=s.fee_rate * mult, slippage_bps=s.slippage_bps * mult)
        arms = [("baseline", base_flags)] + ([("candidate", test_flags)] if extra else [])
        for tag, fl in arms:
            m, _ = one(panel, ss, fl, MODERN)
            cost_rows.append({"costs": f"{mult:g}x", "run": tag, "cagr": m["cagr"],
                              "sharpe": m["sharpe"], "maxdd": m["max_drawdown"],
                              "exposure": m.get("avg_exposure"), "trades": m.get("trades", 0)})
    cdf = pd.DataFrame(cost_rows)
    print(cdf.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---------------------------------------------------------------- 3. the time split
    print("\n3. TIME SPLIT - 2025-2026 was never used to choose anything")
    split_rows = []
    arms2 = [("baseline", base_flags)] + ([("candidate", test_flags)] if extra else [])
    for tag, lo, hi in (("discover 2022-2024", MODERN, me.DISCOVER_END),
                        ("validate 2025-2026", me.VALIDATE_START, None)):
        for run_tag, fl in arms2:
            m, _ = one(panel, s, fl, lo, hi)
            split_rows.append({"half": tag, "run": run_tag, "cagr": m["cagr"],
                               "sharpe": m["sharpe"], "maxdd": m["max_drawdown"],
                               "exposure": m.get("avg_exposure"), "trades": m.get("trades", 0)})
    sdf = pd.DataFrame(split_rows)
    print(sdf.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---------------------------------------------------------------- 4. placebo nulls
    print("\n4. PLACEBO NULLS over the modern era")
    m, r = one(panel, s, test_flags, MODERN)
    nulls = backtest.random_baselines(panel, r.trades, s) or {}
    print("   " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in nulls.items()}, indent=2).replace("\n", "\n   "))

    # ---------------------------------------------------------------- verdict
    print("\nFORWARD-ROBUSTNESS SCORECARD")
    c15 = cdf[(cdf.costs == "1.5x") & (cdf.run == ("candidate" if extra else "baseline"))].iloc[0]
    b15 = cdf[(cdf.costs == "1.5x") & (cdf.run == "baseline")].iloc[0]
    val = sdf[(sdf.half == "validate 2025-2026")
              & (sdf.run == ("candidate" if extra else "baseline"))].iloc[0]
    bval = sdf[(sdf.half == "validate 2025-2026") & (sdf.run == "baseline")].iloc[0]
    checks = {
        f"positive in >={MIN_WIN_RATE:.0%} of rolling windows":
            st["share_positive"] >= MIN_WIN_RATE,
        "worst window no worse than baseline":
            st["worst_return"] >= sb["worst_return"] - 1e-9,
        "survives 1.5x costs (CAGR > 0)": c15["cagr"] > 0,
        "at 1.5x costs, no worse than baseline":
            (c15["sharpe"] >= b15["sharpe"] - 1e-9) if extra else True,
        "profitable on the 2025-2026 half": val["cagr"] > 0,
        "2025-2026 no worse than baseline":
            (val["sharpe"] >= bval["sharpe"] - 1e-9) if extra else True,
        "beats the random-coin null at p<=0.05":
            (nulls.get("p_value_vs_random_coin", 1.0) <= 0.05),
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    verdict = "FORWARD-ROBUST" if all(checks.values()) else "NOT FORWARD-ROBUST"
    print(f"\nVERDICT: {name} -> {verdict}")

    out = ROOT / "reports" / ("forward_test.json" if not extra else
                              "forward_test_" + "_".join(extra) + ".json")
    out.write_text(json.dumps({"name": name, "rolling": rt.to_dict("records"),
                               "rolling_summary": {"baseline": sb, "candidate": st},
                               "costs": cdf.to_dict("records"), "splits": sdf.to_dict("records"),
                               "nulls": nulls, "checks": checks, "verdict": verdict},
                              indent=2, default=str))


if __name__ == "__main__":
    main()
