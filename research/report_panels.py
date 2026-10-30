"""Step 2: one command that prints standard and pit side by side, on the flags that are LIVE.

The number quoted for this strategy has drifted from the number it would actually produce, for two
reasons that are easy to miss and easy to fix:

  1. the STANDARD panel applies today's coin list to all of history, so it never holds a coin that
     was later delisted. It flatters everything. The pit panel rebuilds each day's universe from
     the volume known that morning, including coins that later died.
  2. the numbers in this log were published against the flags of the day. Filters have shipped
     since. A CAGR quoted for `abs_momentum` alone is not the CAGR of the configuration running.

So this script takes the flags actually in force - read from the environment the same way the live
bot reads them, or named explicitly - and prints both panels for the same dates, universe, costs
and risk settings. It also prints the per-engine split once the core is on, because "which engine
earned this" is not answerable from a single equity curve.

    python research/report_panels.py                      # whatever ENABLE_* says, as the bot sees it
    python research/report_panels.py --account demo       # that book's per-account overrides too
    python research/report_panels.py trend_core=0.25      # an explicit set, for comparison
"""

import json
import sys
from dataclasses import fields, replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from app import backtest, config, features  # noqa: E402
from app.flags import Flags  # noqa: E402

import modern_era as me  # noqa: E402

HELD_OUT = "2022-01-01"


def parse_flags(args):
    types = {f.name: type(f.default) for f in fields(Flags)}
    extra = {}
    for a in args:
        k, _, v = a.partition("=")
        if k.startswith("--"):
            continue
        if not v:
            extra[k] = True
            continue
        cast = types.get(k, str)
        extra[k] = v if cast is str else (cast(float(v)) if cast is int else cast(v))
    return extra


def standard_panel(s):
    u = json.loads((ROOT / "reports" / "_parity_universe.json").read_text())
    return features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars), len(u["symbols"])


def one(panel, s, flags, start=None):
    r = backtest.run(panel, s, flags=flags,
                     start=pd.Timestamp(start, tz="UTC") if start else None)
    m = backtest.metrics(r.equity, r.trades, r.exposure)
    out = {"cagr": m["cagr"], "sharpe": m["sharpe"], "maxdd": m["max_drawdown"],
           "exposure": m.get("avg_exposure"), "trades": m.get("trades", 0),
           "calmar": m.get("calmar")}
    # per-engine attribution: which engine's trades produced the money
    if len(r.trades) and "entry_reason" in r.trades:
        core = r.trades["entry_reason"].astype(str).str.startswith("core")
        out["core_trades"] = int(core.sum())
        out["core_pnl"] = float(r.trades.loc[core, "pnl"].sum())
        out["breakout_pnl"] = float(r.trades.loc[~core, "pnl"].sum())
    return out


def main():
    argv = sys.argv[1:]
    account = None
    if "--account" in argv:
        account = argv[argv.index("--account") + 1]
    explicit = parse_flags([a for a in argv if "=" in a])

    s = config.load()
    if account:
        s = replace(s, account_name=account)
    flags = Flags(**explicit) if explicit else Flags.from_env(s, account=account)
    print(f"flags in force: {flags.label or 'baseline'}")
    print(f"  risk_pct {s.risk_pct:.2%}  max_names {s.max_names}  max_weight {s.max_weight:.0%}  "
          f"fee {s.fee_rate:.3%}  slippage {s.slippage:.3%}  kill {s.kill_drawdown:.0%}\n")

    std, n_std = standard_panel(s)
    pit = me.build_pit(s)

    rows = []
    for label, panel, start in (
            ("standard, full", std, None),
            ("standard, held-out 2022-2026", std, HELD_OUT),
            ("PIT, full", pit, None),
            ("PIT, held-out 2022-2026", pit, HELD_OUT)):
        rows.append({"panel": label, **one(panel, s, flags, start)})

    df = pd.DataFrame(rows)
    print(f"standard panel: {n_std} coins (today's list, applied to all history - flattering)")
    print(f"pit panel:      {pit['symbol'].nunique()} coins ever in the top 30, "
          "rebuilt daily including ones later delisted\n")
    cols = ["panel", "cagr", "sharpe", "maxdd", "calmar", "exposure", "trades"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if "core_trades" in df:
        print("\nper-engine attribution (PIT, full):")
        p = df[df.panel == "PIT, full"].iloc[0]
        tot = (p.get("core_pnl") or 0) + (p.get("breakout_pnl") or 0)
        for k in ("core", "breakout"):
            v = p.get(f"{k}_pnl") or 0
            print(f"  {k:9s} {v:>12,.2f} USDT"
                  + (f"  ({v / tot:+.1%} of the total)" if tot else ""))
        print(f"  core trades: {int(p.get('core_trades') or 0)} of {int(p['trades'])}")

    print("\nQUOTE THE PIT ROW. The standard panel never holds a coin that was later delisted, "
          "which is why it reads higher.")
    (ROOT / "reports" / "panels_report.json").write_text(json.dumps(
        {"flags": flags.label, "account": account, "rows": rows}, indent=2, default=str))


if __name__ == "__main__":
    main()
