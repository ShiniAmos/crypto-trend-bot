"""Each engine's own drawdown, separate from the book's.

The core's pit drawdown is quoted as -25.54%, but that is the BOOK's drawdown while the core was
running - it includes whatever the breakout engine was doing at the same time. The question the
core's risk actually raises is narrower: how bad does the CORE's own P&L get, on its own?

`Result.engine_pnl` is a cumulative P&L curve per engine, realised plus open, starting at zero. A
drawdown on a P&L curve is measured in money against its own high-water mark, and then expressed
against starting equity so it is comparable with the book's percentage drawdown:

    dd_engine = min(pnl_t - cummax(pnl)) / start_equity

That is an attribution, not a standalone backtest: an engine run alone would have had all the cash
and might have taken different positions. It answers "how much of the book's pain came from here",
which is the question, and it does not pretend to answer "what would this engine do alone".

    python research/engine_split.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from app import backtest, config, features  # noqa: E402
from app.flags import Flags  # noqa: E402

import modern_era as me  # noqa: E402

HELD_OUT = "2022-01-01"
DEMO = dict(abs_momentum=True, volume_min=0.8, funding_max_pct=0.8,
            trend_core=0.25, trend_core_symbols="BTCUSDT")


def split(panel, s, flags, start=None):
    r = backtest.run(panel, s, flags=flags,
                     start=pd.Timestamp(start, tz="UTC") if start else None)
    m = backtest.metrics(r.equity, r.trades, r.exposure)
    e0 = r.equity.iloc[0]
    out = {"book_cagr": m["cagr"], "book_sharpe": m["sharpe"], "book_maxdd": m["max_drawdown"]}
    for eng in ("core", "breakout"):
        if eng not in r.engine_pnl:
            continue
        pnl = r.engine_pnl[eng]
        dd_money = (pnl - pnl.cummax()).min()
        out[f"{eng}_pnl"] = float(pnl.iloc[-1])
        out[f"{eng}_pnl_pct"] = float(pnl.iloc[-1] / e0)
        out[f"{eng}_maxdd_pct"] = float(dd_money / e0)
    tot = (out.get("core_pnl", 0.0) + out.get("breakout_pnl", 0.0))
    out["core_share"] = (out.get("core_pnl", 0.0) / tot) if tot else float("nan")
    # the reconciliation: attribution that does not add up is decoration
    out["reconciles"] = abs(tot - (r.equity.iloc[-1] - e0)) < 1e-6
    return out


def main():
    s = config.load()
    u = json.loads((ROOT / "reports" / "_parity_universe.json").read_text())
    std = features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars)
    pit = me.build_pit(s)

    rows = []
    for pname, panel, start in (("standard full", std, None),
                                ("pit full", pit, None),
                                ("pit held-out 2022-26", pit, HELD_OUT)):
        rows.append({"panel": pname, **split(panel, s, Flags(**DEMO), start)})
    df = pd.DataFrame(rows)

    print("demo configuration: " + Flags(**DEMO).label)
    print("\n=== the book, and each engine's own worst run ===")
    cols = ["panel", "book_cagr", "book_sharpe", "book_maxdd",
            "core_maxdd_pct", "breakout_maxdd_pct"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== who made the money ===")
    cols = ["panel", "core_pnl_pct", "breakout_pnl_pct", "core_share", "reconciles"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nRead the drawdown columns as: the worst peak-to-trough fall in THAT engine's own")
    print("cumulative P&L, as a fraction of starting equity. They do not sum to the book's")
    print("drawdown - the two engines' worst runs happen at different times, which is the point.")

    (ROOT / "reports" / "engine_split.json").write_text(
        json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
