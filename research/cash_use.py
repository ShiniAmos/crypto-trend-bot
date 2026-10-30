"""The cash is idle. Four ways to spend it, and only one of them should be taken.

Measured on the pit panel, shipped flags, 2022-2026:

    max_names 3, risk_pct 1%, max_weight 30%
    mean position weight at entry      6.90%      <- three full slots is ~21% of the book
    MEDIAN exposure                    0.00%      <- half the time nothing is held at all
    mean exposure                      5.84%
    highest exposure ever              52.81%

So there are two separate problems wearing one coat, and they need different answers:

    (a) half the time the engine holds NOTHING, because it is waiting for a breakout
    (b) when it is as full as it gets, three slots only account for about a fifth of the book,
        because a 1%-risk position with a stop ~14% below entry is a 7% position

THE TRAP, stated before any number: scaling a position size up multiplies return AND drawdown by
the same factor. That is leverage, not edge, and it is only worth doing if the thing being scaled
makes money. On this panel the rotation engine earns +0.54% a year over the modern era and
**-2.06% a year on the 2025-2026 holdout**. Tripling a loss gives a bigger loss. So options B and C
below are expected to fail, and they are run anyway because "expected to fail" is not a measurement.

    A  baseline                     3 slots, 1% risk
    B  more risk per slot           risk_pct 2%, 3%     - same trades, bigger
    C  more slots                   max_names 5, 8      - more trades, same size
    D  the trend core               trend_core 0.25 / 0.50 / 0.75, now that trend_core is its own
                                    cap rather than being clipped to max_weight
    E  core + more slots            the two combined, to see whether they overlap or add

Judged on the modern era AND the 2025-2026 holdout, which chose nothing. Calmar (return per unit of
drawdown) is the metric that matters here, because every option on this list raises exposure and
the ship rule's "maxDD not worse" would reject all of them by construction.

    python research/cash_use.py
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from app import backtest, config  # noqa: E402
from app.flags import Flags  # noqa: E402

import modern_era as me  # noqa: E402

WINDOWS = [("modern 2022-2026", "2022-01-01", None),
           ("HOLDOUT 2025-2026", "2025-01-01", None)]

# (label, settings overrides, flag overrides)
GRID = [
    ("A baseline 3x1%", {}, {}),
    ("B risk 2%", dict(risk_pct=0.02), {}),
    ("B risk 3%", dict(risk_pct=0.03), {}),
    ("C slots 5", dict(max_names=5), {}),
    ("C slots 8", dict(max_names=8), {}),
    ("D core 0.25", {}, dict(trend_core=0.25, trend_core_symbols="BTCUSDT")),
    ("D core 0.50", {}, dict(trend_core=0.50, trend_core_symbols="BTCUSDT")),
    ("D core 0.75", {}, dict(trend_core=0.75, trend_core_symbols="BTCUSDT")),
    ("D core 0.50 B+E", {}, dict(trend_core=0.50, trend_core_symbols="BTCUSDT,ETHUSDT")),
    ("D core 0.75 B+E", {}, dict(trend_core=0.75, trend_core_symbols="BTCUSDT,ETHUSDT")),
    ("E core 0.25 + 5 slots", dict(max_names=5),
     dict(trend_core=0.25, trend_core_symbols="BTCUSDT")),
    ("E core 0.25 + 8 slots", dict(max_names=8),
     dict(trend_core=0.25, trend_core_symbols="BTCUSDT")),
    ("E core 0.35 + 5 slots", dict(max_names=5),
     dict(trend_core=0.35, trend_core_symbols="BTCUSDT")),
    ("E core 0.50 + 5 slots", dict(max_names=5),
     dict(trend_core=0.50, trend_core_symbols="BTCUSDT")),
    ("E core 0.75 + 5 slots", dict(max_names=5),
     dict(trend_core=0.75, trend_core_symbols="BTCUSDT")),
]


def main():
    base_s = config.load()
    panel = me.build_pit(base_s)
    out, rows_by_window = {}, {}

    for wname, lo, hi in WINDOWS:
        rows = []
        for label, sov, fov in GRID:
            s = replace(base_s, **sov) if sov else base_s
            fl = Flags(**{**me.SHIPPED, **fov})
            r = backtest.run(panel, s, flags=fl, start=pd.Timestamp(lo, tz="UTC"),
                             end=pd.Timestamp(hi, tz="UTC") if hi else None)
            m = backtest.metrics(r.equity, r.trades, r.exposure)
            rows.append({"run": label, "cagr": m["cagr"], "sharpe": m["sharpe"],
                         "maxdd": m["max_drawdown"], "calmar": m.get("calmar"),
                         "expo_mean": m.get("avg_exposure"),
                         "pct_bars_invested": float((r.exposure > 0).mean()),
                         "expo_max": float(r.exposure.max()),
                         "trades": m.get("trades", 0)})
        df = pd.DataFrame(rows)
        rows_by_window[wname] = df
        out[wname] = df.to_dict("records")
        print(f"=== {wname} ===")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print()

    mod, hold = rows_by_window["modern 2022-2026"], rows_by_window["HOLDOUT 2025-2026"]
    b_mod = mod[mod.run == "A baseline 3x1%"].iloc[0]
    b_hold = hold[hold.run == "A baseline 3x1%"].iloc[0]
    cmp_rows = []
    for label, _, _ in GRID[1:]:
        m, h = mod[mod.run == label].iloc[0], hold[hold.run == label].iloc[0]
        cmp_rows.append({
            "run": label,
            "d_cagr_mod": m.cagr - b_mod.cagr, "d_calmar_mod": (m.calmar or 0) - (b_mod.calmar or 0),
            "holdout_cagr": h.cagr, "d_cagr_hold": h.cagr - b_hold.cagr,
            "d_sharpe_hold": h.sharpe - b_hold.sharpe,
            "expo_mod": m.expo_mean,
            "holdout_profitable": bool(h.cagr > 0),
            "better_both": bool(m.cagr > b_mod.cagr and h.cagr > b_hold.cagr)})
    cdf = pd.DataFrame(cmp_rows)
    print("=== against the baseline; the last two columns are the whole decision ===")
    print(cdf.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    good = cdf[cdf.better_both & cdf.holdout_profitable]
    print(f"\nraise the modern era AND the holdout, with the holdout actually profitable: "
          f"{', '.join(good['run']) if len(good) else 'none'}")
    # "median exposure" was the first statistic here and it was a bad one: a book invested on 48.9%
    # of bars has a median of zero, which reads as "never invested". The fraction of bars with any
    # position is the honest version.
    b_inv = mod[mod.run == "A baseline 3x1%"].pct_bars_invested.iloc[0]
    print(f"\nThe idle cash itself: the baseline holds something on only {b_inv:.1%} of bars, and "
          "averages 5.84% of the book at work, so most of the answer is not sizing at all - it is "
          "having something worth holding.")
    best = good.sort_values("expo_mod", ascending=False)
    if len(best):
        top = best.iloc[0]
        print(f"Most cash put to work among the options that survive the holdout: {top['run']} at "
              f"{top['expo_mod']:.1%} mean exposure.")

    (ROOT / "reports" / "cash_use.json").write_text(json.dumps(
        {"windows": out, "comparison": cmp_rows,
         "passes": good["run"].tolist()}, indent=2, default=str))


if __name__ == "__main__":
    main()
