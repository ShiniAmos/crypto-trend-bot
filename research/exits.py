"""P4: how long to hold. Every experiment so far changed WHICH trade is taken, not its length.

Baseline is the shipped state. Each variant changes exactly one thing about the exit, and is
measured on the standard panel and the held-out half with the usual ship rule. The average winner
runs about 41 bars and the average loser dies much sooner, so if an asymmetry exists anywhere left
in this strategy, it is here.

    python research/exits.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import backtest, config, features  # noqa: E402
from app.flags import Flags  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SHIPPED = dict(abs_momentum=True, volume_min=0.8, taker_min=0.5)
HELD_OUT = "2022-01-01"
EPS_CAGR, EPS_DD = 0.005, 0.01


def measure(panel, s, flags, start=None):
    res = backtest.run(panel, s, flags=flags, start=start)
    m = backtest.metrics(res.equity, res.trades, res.exposure)
    rb = backtest.random_baselines(panel, res.trades, s) or {}
    t = res.trades
    return {"cagr": m["cagr"], "sharpe": m["sharpe"], "maxdd": m["max_drawdown"],
            "trades": m.get("trades", 0), "win": m.get("win_rate"),
            "avg_bars": m.get("avg_bars_held"),
            "best": float(t["ret"].max()) if len(t) else float("nan"),
            "p_coin": rb.get("p_value_vs_random_coin", float("nan"))}


def verdict(b, c, held_out=False):
    ec, ed = (EPS_CAGR, EPS_DD) if held_out else (0.0, 0.0)
    checks = {"CAGR": c["cagr"] >= b["cagr"] - ec, "Sharpe": c["sharpe"] >= b["sharpe"],
              "maxDD": c["maxdd"] >= b["maxdd"] - ed}
    if not held_out:
        checks["p_coin"] = c["p_coin"] <= max(b["p_coin"], 0.05) + 1e-9
    return all(checks.values()), checks


def main():
    s = config.load()
    u = json.loads((ROOT / "reports" / "_parity_universe.json").read_text())
    base_panel = features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars)

    variants = [("baseline", Flags(**SHIPPED), 0),
                ("exit channel 30", Flags(**SHIPPED, exit_donchian=30), 30),
                ("exit channel 40", Flags(**SHIPPED, exit_donchian=40), 40),
                ("exit channel 55", Flags(**SHIPPED, exit_donchian=55), 55),
                ("exit rank 15", Flags(**SHIPPED, exit_rank=15), 0),
                ("exit rank 20", Flags(**SHIPPED, exit_rank=20), 0),
                ("hold through regime", Flags(**SHIPPED, hold_through_regime=True), 0),
                ("time stop 12 bars", Flags(**SHIPPED, time_stop_bars=12), 0),
                ("time stop 30 bars", Flags(**SHIPPED, time_stop_bars=30), 0)]

    panels = {0: base_panel}
    rows, held = [], []
    for name, flags, don in variants:
        if don and don not in panels:
            panels[don] = features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars,
                                               exit_donchian=don)
        p = panels[don]
        rows.append({"run": name, **measure(p, s, flags)})
        held.append({"run": name, **measure(p, s, flags, start=HELD_OUT)})

    for tag, data in (("STANDARD PANEL", rows), ("HELD-OUT 2022-2026", held)):
        print(f"\n=== {tag} ===")
        print(pd.DataFrame(data).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== ship rule ===")
    b_std, b_held = rows[0], held[0]
    out = []
    for r_std, r_held in zip(rows[1:], held[1:]):
        ok_s, cs = verdict(b_std, r_std)
        ok_h, ch = verdict(b_held, r_held, held_out=True)
        out.append({"run": r_std["run"], "standard": "PASS" if ok_s else "FAIL",
                    "held_out": "PASS" if ok_h else "FAIL",
                    "verdict": "run pit30" if (ok_s and ok_h) else "REJECT",
                    "fails": ",".join(k for k, v in {**cs, **ch}.items() if not v)})
    print(pd.DataFrame(out).to_string(index=False))
    (ROOT / "reports" / "exits.json").write_text(
        json.dumps({"standard": rows, "held_out": held, "verdicts": out}, indent=2, default=str))


if __name__ == "__main__":
    main()
