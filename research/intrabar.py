"""Track A2: same 4h signal, but act the moment price crosses the line.

The ranking, the filters and the channel all come from the last CLOSED 4h bar - exactly what the
bot knows today. The only change is that the entry no longer waits for the 4h close: any hourly
close above that channel triggers it, and any hourly close below the lower channel exits.

Anti-lookahead: the 4h columns are merged onto the 1h bars with merge_asof(direction="backward"),
and a 4h bar stamped 12:00 is only attached to 1h bars from 12:00 onward, which is when it is
genuinely known. Fills still happen at the NEXT 1h open, never at the trigger price.

    python research/intrabar.py
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
CARRY = ["don_high", "don_low", "stop_level", "rank", "score", "eligible", "regime_on",
         "rs_7d", "rs_30d", "ret_30d", "volume_ratio", "taker_ratio", "vol_30d", "dist_sma50",
         "n_eligible", "btc_ret_30d"]


def hybrid(p4, p1):
    """1h bars carrying the last closed 4h bar's signal columns."""
    out = []
    for sym, g1 in p1.groupby("symbol", sort=False):
        g4 = p4[p4["symbol"] == sym]
        if g4.empty:
            continue
        g1 = g1.sort_values("bar_time")
        g4 = g4.sort_values("bar_time")[["bar_time"] + [c for c in CARRY if c in g4.columns]]
        m = pd.merge_asof(g1[["bar_time", "open", "high", "low", "close", "ret_4h", "age_bars"]],
                          g4, on="bar_time", direction="backward", allow_exact_matches=True)
        m["symbol"] = sym
        out.append(m)
    h = pd.concat(out, ignore_index=True)
    # The trigger: an hourly close through the 4h channel, instead of a 4h close through it.
    h["breakout"] = h["close"] > h["don_high"]
    h["breakdown"] = h["close"] < h["don_low"]
    h["eligible"] = h["eligible"].fillna(False).astype(bool)
    h["regime_on"] = h["regime_on"].fillna(False).astype(bool)
    return h.sort_values(["bar_time", "symbol"]).reset_index(drop=True)


def main():
    s = config.load()
    u = json.loads((ROOT / "reports" / "_parity_universe.json").read_text())
    print("building panels...")
    p4 = features.build_panel(s.klines_dir, u["symbols"], 20, tf="4h")
    p1 = features.build_panel(s.klines_dir, u["symbols"], 80, tf="1h")
    h = hybrid(p4, p1)
    start = max(p4["bar_time"].min(), h["bar_time"].min())
    end = min(p4["bar_time"].max(), h["bar_time"].max())
    print(f"common window {start.date()} to {end.date()}\n")

    rows = []
    for tag, panel, bpy in (("4h close trigger (shipped)", p4, 6 * 365),
                            ("1h trigger on the 4h channel", h, 24 * 365)):
        res = backtest.run(panel, s, flags=SHIPPED, start=start, end=end)
        m = backtest.metrics(res.equity, res.trades, res.exposure, bars_per_year=bpy)
        years = max((res.equity.index[-1] - res.equity.index[0]).days / 365.25, 1e-9)
        rb = backtest.random_baselines(panel, res.trades, s) or {}
        rows.append({"run": tag, "cagr": m["cagr"], "sharpe": m["sharpe"],
                     "maxdd": m["max_drawdown"], "trades": m.get("trades", 0),
                     "trades/yr": round(m.get("trades", 0) / years, 1),
                     "win": m.get("win_rate"), "avg_trade": m.get("avg_trade"),
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
    (ROOT / "reports" / "intrabar.json").write_text(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
