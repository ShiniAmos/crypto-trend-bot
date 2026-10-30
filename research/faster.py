"""Track A: does deciding every hour beat deciding every four hours?

Same strategy, same shipped filters, same wall-clock horizons - only the decision granularity
changes. A 20-bar Donchian channel on 4h bars is 3.33 days, so on 1h bars it is 80 bars. Returns,
volatility and volume windows are already expressed in days and scale automatically.

Both sides pay identical costs (7 bps slippage, 0.10% fee per side) and are measured over the same
dates. Sharpe is annualised with the right bar count for each timeframe.

    python research/faster.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import backtest, config, features  # noqa: E402
from app.flags import Flags  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SHIPPED = Flags(abs_momentum=True, volume_min=0.8, taker_min=0.5)


def main():
    s = config.load()
    u = json.loads((ROOT / "reports" / "_parity_universe.json").read_text())

    print("building 4h panel...")
    p4 = features.build_panel(s.klines_dir, u["symbols"], 20, tf="4h")
    print("building 1h panel (same horizons, 80-bar channel)...")
    p1 = features.build_panel(s.klines_dir, u["symbols"], 80, tf="1h")

    start = max(p4["bar_time"].min(), p1["bar_time"].min())
    end = min(p4["bar_time"].max(), p1["bar_time"].max())
    print(f"common window: {start.date()} to {end.date()}\n")

    rows = []
    for tag, panel, bpy in (("4h decisions", p4, 6 * 365), ("1h decisions", p1, 24 * 365)):
        res = backtest.run(panel, s, flags=SHIPPED, start=start, end=end)
        m = backtest.metrics(res.equity, res.trades, res.exposure, bars_per_year=bpy)
        fees = res.log["fee"].sum() if len(res.log) else 0.0
        years = max((res.equity.index[-1] - res.equity.index[0]).days / 365.25, 1e-9)
        rb = backtest.random_baselines(panel, res.trades, s) or {}
        rows.append({"run": tag, "cagr": m["cagr"], "sharpe": m["sharpe"],
                     "maxdd": m["max_drawdown"], "trades": m.get("trades", 0),
                     "trades/yr": round(m.get("trades", 0) / years, 1),
                     "win": m.get("win_rate"), "avg_trade": m.get("avg_trade"),
                     "fees_paid": round(fees, 0),
                     "fees_%_of_start": round(100 * fees / s.start_equity, 1),
                     "p_coin": rb.get("p_value_vs_random_coin"),
                     "p_time": rb.get("p_value_vs_random_time")})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    b, c = rows[0], rows[1]
    checks = {"CAGR": c["cagr"] >= b["cagr"], "Sharpe": c["sharpe"] >= b["sharpe"],
              "maxDD": c["maxdd"] >= b["maxdd"],
              "p_coin": c["p_coin"] <= max(b["p_coin"], 0.05) + 1e-9}
    print("\n-> " + ("PASS" if all(checks.values()) else "FAIL") + "  " +
          "  ".join(f"{k}:{'ok' if v else 'NO'}" for k, v in checks.items()))
    (ROOT / "reports" / "faster.json").write_text(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
