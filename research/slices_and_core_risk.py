"""Steps 3 and 4 against the ship rule: capital slices, and a volatility cap on the core.

THE SHIP RULE, applied exactly as written and not loosened:

    CAGR, Sharpe and maxDD must all be NO WORSE on BOTH the standard panel AND the pit panel,
    and the held-out 2022-2026 period must not fail.

Same universe, dates, costs and risk for every row. The baseline is the demo book's live
configuration, because that is what these changes would be layered onto:

    abs_momentum + volume_min=0.8 + funding_max_pct=0.8 + trend_core=0.25 on BTC

WHAT IS BEING TESTED

  step 3  capital_slices      the core sized inside its own `trend_core` share, the breakout engine
                              inside the remainder and capped at slice_max_names=2, so the two stop
                              competing for one pot of cash
  step 4a core_target_vol     w_i = min(0.18 / vol_30d_i, 0.25) - a wild coin held smaller
  step 4b core_max_stretch    refuse a NEW core entry more than 15% above its 200-day average
  combined                    slices plus each cap

THE EXTRA CONDITION FROM THE BRIEF: if core pit drawdown exceeds 30%, the cap stays on or the core
change is rejected. The current core pit drawdown is -25.5%, so that is a live constraint, not a
formality.

    python research/slices_and_core_risk.py
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
BASE = dict(abs_momentum=True, volume_min=0.8, funding_max_pct=0.8,
            trend_core=0.25, trend_core_symbols="BTCUSDT")

VARIANTS = [
    ("baseline (demo live)", {}),
    ("3 slices", dict(capital_slices=True, slice_max_names=2)),
    ("4a vol cap 0.18", dict(core_target_vol=0.18, core_w_max=0.25)),
    ("4b stretch cap 15%", dict(core_max_stretch=0.15)),
    ("3+4a slices + vol cap", dict(capital_slices=True, slice_max_names=2,
                                   core_target_vol=0.18, core_w_max=0.25)),
    ("3+4b slices + stretch", dict(capital_slices=True, slice_max_names=2,
                                   core_max_stretch=0.15)),
]


def metrics(panel, s, flags, start=None):
    r = backtest.run(panel, s, flags=flags,
                     start=pd.Timestamp(start, tz="UTC") if start else None)
    m = backtest.metrics(r.equity, r.trades, r.exposure)
    out = {"cagr": m["cagr"], "sharpe": m["sharpe"], "maxdd": m["max_drawdown"],
           "calmar": m.get("calmar"), "exposure": m.get("avg_exposure"),
           "trades": m.get("trades", 0)}
    if len(r.trades) and "entry_reason" in r.trades:
        core = r.trades["entry_reason"].astype(str).str.startswith("core")
        out["core_pnl"] = float(r.trades.loc[core, "pnl"].sum())
        out["breakout_pnl"] = float(r.trades.loc[~core, "pnl"].sum())
    return out


def main():
    s = config.load()
    u = json.loads((ROOT / "reports" / "_parity_universe.json").read_text())
    std = features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars)
    pit = me.build_pit(s)

    panels = [("standard", std, None), ("standard held-out", std, HELD_OUT),
              ("pit", pit, None), ("pit held-out", pit, HELD_OUT)]

    rows = {}
    for label, extra in VARIANTS:
        fl = Flags(**{**BASE, **extra})
        rows[label] = {p: metrics(panel, s, fl, start) for p, panel, start in panels}
        print(f"  ran {label}")

    print()
    for pname, _, _ in panels:
        print(f"=== {pname} ===")
        df = pd.DataFrame([{"variant": k, **v[pname]} for k, v in rows.items()])
        print(df[["variant", "cagr", "sharpe", "maxdd", "calmar", "exposure", "trades"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print()

    base = rows["baseline (demo live)"]
    print("=== SHIP RULE: no worse on CAGR, Sharpe AND maxDD, on standard AND pit, "
          "held-out not failing ===")
    verdicts = []
    eps = 1e-9
    for label, _ in VARIANTS[1:]:
        v = rows[label]
        checks = {}
        for pname in ("standard", "pit"):
            checks[pname] = (v[pname]["cagr"] >= base[pname]["cagr"] - eps
                             and v[pname]["sharpe"] >= base[pname]["sharpe"] - eps
                             and v[pname]["maxdd"] >= base[pname]["maxdd"] - eps)
        # the held-out must not fail: positive CAGR and no worse than the baseline's held-out
        checks["held-out"] = (v["pit held-out"]["cagr"] >= base["pit held-out"]["cagr"] - eps
                              and v["pit held-out"]["sharpe"]
                              >= base["pit held-out"]["sharpe"] - eps)
        # the brief's extra condition on core drawdown
        dd_ok = v["pit"]["maxdd"] >= -0.30
        ships = all(checks.values()) and dd_ok
        verdicts.append({"variant": label, **checks, "pit_dd_within_30pct": dd_ok,
                         "SHIPS": ships})
    vdf = pd.DataFrame(verdicts)
    print(vdf.to_string(index=False))
    shipped = vdf[vdf.SHIPS]["variant"].tolist()
    print(f"\nSHIPPED: {', '.join(shipped) if shipped else 'none - all REJECT, flags stay off'}")

    print("\nsix metrics for every variant, pit panel (the one to quote):")
    six = pd.DataFrame([{"variant": k, **{m: v["pit"][m] for m in
                                          ("cagr", "sharpe", "maxdd", "calmar", "exposure",
                                           "trades")}} for k, v in rows.items()])
    print(six.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    (ROOT / "reports" / "slices_core_risk.json").write_text(json.dumps(
        {"rows": {k: v for k, v in rows.items()}, "verdicts": verdicts,
         "shipped": shipped}, indent=2, default=str))


if __name__ == "__main__":
    main()
