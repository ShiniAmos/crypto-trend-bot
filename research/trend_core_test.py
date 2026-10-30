"""Does the trend core earn its place - and does it hold up on data it never chose anything on?

The case for it, from research/btc_core.py: over 2022-2026, through real costs, holding BTC
whenever it is above its own 200-day average returned +31.12% a year at Sharpe 0.829 while the
alt-rotation engine returned +2.24% at 0.263 and was invested 6.8% of the time.

That was a standalone simulation of a single rule. This is the real engine: app/backtest.py, the
survivorship-free panel, the shipped filters, the slot limits, the drawdown kill, the minimum
order size, the fees - everything that makes a backtest less flattering than a spreadsheet.

THE GRID
    core weight   0 (baseline), 0.25, 0.50, 0.75
    core coins    BTC only, and BTC+ETH
Judged on four windows, and the last one is the only one that really counts:

    full sample            2018-2026, where the alt engine's 2019-2021 era still dominates
    modern                 2022-2026
    discover               2022-2024
    HOLDOUT                2025-2026, which chose nothing - the forward test

WHAT WOULD MAKE THIS A FAILURE, declared before the numbers are read:
  - the holdout. If the core only wins 2022-2024 it is another era fit and it stays off.
  - exposure without return. Being invested more is the point, but not at a worse Sharpe than the
    baseline - that would just be leverage with extra steps.
  - the drawdown. The core WILL deepen it; BTC fell 38% below its own 200-day average inside this
    sample. The question is whether return per unit of drawdown (Calmar) improves, so that is
    reported rather than the ship rule's "maxDD not worse", which no always-invested variant can
    ever satisfy against a strategy that sits in cash.

    python research/trend_core_test.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from app import backtest, config  # noqa: E402
from app.flags import Flags  # noqa: E402

import modern_era as me  # noqa: E402

WINDOWS = [("full 2018-2026", "2018-01-01", None),
           ("modern 2022-2026", "2022-01-01", None),
           ("discover 2022-2024", "2022-01-01", "2024-12-31"),
           ("HOLDOUT 2025-2026", "2025-01-01", None)]
GRID = [("baseline (no core)", {}),
        ("core 0.25 BTC", dict(trend_core=0.25, trend_core_symbols="BTCUSDT")),
        ("core 0.50 BTC", dict(trend_core=0.50, trend_core_symbols="BTCUSDT")),
        ("core 0.75 BTC", dict(trend_core=0.75, trend_core_symbols="BTCUSDT")),
        ("core 0.50 BTC+ETH", dict(trend_core=0.50, trend_core_symbols="BTCUSDT,ETHUSDT")),
        ("core 0.75 BTC+ETH", dict(trend_core=0.75, trend_core_symbols="BTCUSDT,ETHUSDT"))]


def main():
    s = config.load()
    panel = me.build_pit(s)
    print(f"panel {panel['bar_time'].min().date()} .. {panel['bar_time'].max().date()}, "
          f"{panel['symbol'].nunique()} symbols")
    print(f"shipped filters: {Flags(**me.SHIPPED).label}\n")

    out = {}
    for wname, lo, hi in WINDOWS:
        rows = []
        for gname, extra in GRID:
            fl = Flags(**{**me.SHIPPED, **extra})
            r = backtest.run(panel, s, flags=fl, start=pd.Timestamp(lo, tz="UTC"),
                             end=pd.Timestamp(hi, tz="UTC") if hi else None)
            m = backtest.metrics(r.equity, r.trades, r.exposure)
            core_trades = 0
            if len(r.trades):
                core_trades = int(r.trades["entry_reason"].astype(str)
                                  .str.startswith("core").sum())
            rows.append({"run": gname, "cagr": m["cagr"], "sharpe": m["sharpe"],
                         "maxdd": m["max_drawdown"], "calmar": m.get("calmar"),
                         "exposure": m.get("avg_exposure"), "trades": m.get("trades", 0),
                         "core_trades": core_trades})
        df = pd.DataFrame(rows)
        out[wname] = df.to_dict("records")
        print(f"=== {wname} ===")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print()

    # the decision table: every variant against the baseline, per window
    print("=== against the baseline, per window (d = candidate - baseline) ===")
    cmp_rows = []
    for wname in out:
        recs = {r["run"]: r for r in out[wname]}
        b = recs["baseline (no core)"]
        for gname, _ in GRID[1:]:
            c = recs[gname]
            cmp_rows.append({"window": wname, "run": gname,
                             "d_cagr": c["cagr"] - b["cagr"],
                             "d_sharpe": c["sharpe"] - b["sharpe"],
                             "d_calmar": (c["calmar"] or 0) - (b["calmar"] or 0),
                             "d_expo": c["exposure"] - b["exposure"]})
    cdf = pd.DataFrame(cmp_rows)
    print(cdf.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    print("\n=== the test that counts: better on BOTH the modern era and the holdout ===")
    piv = cdf.pivot_table(index="run", columns="window", values="d_sharpe")
    good = []
    for run in piv.index:
        mod = piv.loc[run, "modern 2022-2026"]
        hold = piv.loc[run, "HOLDOUT 2025-2026"]
        ok = mod > 0 and hold > 0
        good.append({"run": run, "d_sharpe_modern": mod, "d_sharpe_holdout": hold,
                     "passes_both": bool(ok)})
    gdf = pd.DataFrame(good).sort_values("d_sharpe_holdout", ascending=False)
    print(gdf.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    winners = gdf[gdf.passes_both]["run"].tolist()
    print(f"\nimprove Sharpe on the modern era AND the holdout: "
          f"{', '.join(winners) if winners else 'none'}")

    (ROOT / "reports" / "trend_core.json").write_text(json.dumps(
        {"windows": out, "deltas": cmp_rows, "passes_both": winners}, indent=2, default=str))


if __name__ == "__main__":
    main()
