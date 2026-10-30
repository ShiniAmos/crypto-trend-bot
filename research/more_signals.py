"""Trade more often WITHOUT leverage: does a lower entry bar actually help?

The complaint is right and already measured: the book is invested 5.8% of the time and holds
nothing on 60.5% of bars. Two very different cures get proposed for that, and only one of them has
any evidence behind it:

    MORE SIGNALS   lower the bar, so more coins qualify more often. Untested on the modern era -
                   every Donchian and candidate-count experiment in this log was scored on the full
                   sample, where 2019-2021 dominates. That is what this script measures.
    MORE LEVERAGE  put more money behind the same signals. Already measured in research/cash_use.py
                   and it is the worst option on the board: risk_pct 2% takes the 2025-26 holdout
                   to -7.46%, risk_pct 3% to -15.32% with a -41% drawdown, because multiplying a
                   losing engine multiplies the loss. Futures would be this, with liquidation.

THE GRID - every lever that produces MORE trades, without touching position size:

    donchian 10 / 15 / 20     a shorter breakout channel: a 10-bar high is cleared far more often
                              than a 20-bar high, so more coins trigger
    candidates 5 / 8 / 12     how deep the ranking is allowed to reach
    max_names 3 / 5 / 8       how many can be held at once
    rs gate on / off          dropping "must beat BTC over 7 days" admits far more names

REPORTED FOR EACH: trades, exposure, bars invested - and then whether the extra activity was worth
having. Activity is not the goal; the goal is return per unit of drawdown while being invested more
of the time, so Calmar and exposure are shown together.

SCORED ON 2022-2024 ONLY. The 2025-2026 holdout was declared spent in P9 after three rounds of
tuning, and it is not consulted here. Anything that looks good in this table is a CANDIDATE for a
forward test, not a result - which is exactly the discipline that would have saved the four shipped
filters from being 2019-2021 artifacts.

    python research/more_signals.py
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from app import backtest, config, features, pit, signals  # noqa: E402
from app.flags import Flags  # noqa: E402

import modern_era as me  # noqa: E402

DISCOVER = ("2022-01-01", "2024-12-31")
SHIPPED = dict(me.SHIPPED)
CORE = dict(trend_core=0.25, trend_core_symbols="BTCUSDT")


def panel_for(s, donchian, cache={}):
    """A pit panel built with a given entry channel. Cached: each build is minutes and ~600MB."""
    if donchian in cache:
        return cache[donchian]
    cp = ROOT / "reports" / f"_pit_panel_d{donchian}.parquet"
    if cp.exists():
        cache[donchian] = pd.read_parquet(cp)
        return cache[donchian]
    kdir = s.data_dir / "pit" / "klines"
    syms = sorted({p.name.rsplit("_", 1)[0] for p in kdir.glob("*_1d.parquet")})
    syms = [x for x in syms if pit.keep(x, s.exclude_memes)]
    member = pit.membership(kdir, syms, 30)
    ever = sorted(set(member.columns[member.any()]) | {"BTCUSDT"})
    print(f"  building panel for donchian={donchian} ({len(ever)} coins)...")
    panel = features.build_panel(kdir, ever, donchian)
    panel["day"] = (panel["bar_time"] - pd.Timedelta(milliseconds=1)).dt.normalize()
    m = member.rename_axis(index="day", columns="symbol").stack().rename("member").reset_index()
    panel = panel.merge(m, on=["day", "symbol"], how="left")
    panel["member"] = panel["member"].astype("boolean").fillna(False).astype(bool)
    panel["eligible"] = panel["eligible"] & (panel["member"] | (panel["symbol"] == "BTCUSDT"))
    panel = features.score(panel.drop(columns=["score", "rank", "n_eligible"]))
    import funding_pit
    panel = funding_pit.attach(panel, s.data_dir / "funding_pit")
    panel.to_parquet(cp)
    cache[donchian] = panel
    return panel


def run(panel, s, flags, drop_rs=False):
    if drop_rs:
        # "must beat BTC over 7 days" is applied inside backtest.run from the rs_7d column, so the
        # honest way to switch it off is to neutralise the column rather than fork the engine.
        panel = panel.copy()
        panel["rs_7d"] = panel["rs_7d"].abs() + 1e-9
    r = backtest.run(panel, s, flags=flags,
                     start=pd.Timestamp(DISCOVER[0], tz="UTC"),
                     end=pd.Timestamp(DISCOVER[1], tz="UTC"))
    m = backtest.metrics(r.equity, r.trades, r.exposure)
    return {"cagr": m["cagr"], "sharpe": m["sharpe"], "maxdd": m["max_drawdown"],
            "calmar": m.get("calmar"), "trades": m.get("trades", 0),
            "exposure": m.get("avg_exposure"),
            "bars_invested": float((r.exposure > 0).mean())}


def main():
    s0 = config.load()
    rows = []

    print("baseline and the core, for reference:")
    p20 = panel_for(s0, 20)
    rows.append({"variant": "baseline (d20, c5, 3 slots)", **run(p20, s0, Flags(**SHIPPED))})
    rows.append({"variant": "+ core 0.25 + 5 slots",
                 **run(p20, replace(s0, max_names=5), Flags(**SHIPPED, **CORE))})

    print("shorter breakout channel (more signals):")
    for d in (10, 15):
        p = panel_for(s0, d)
        rows.append({"variant": f"donchian {d}", **run(p, s0, Flags(**SHIPPED))})
        rows.append({"variant": f"donchian {d} + core + 5 slots",
                     **run(p, replace(s0, max_names=5), Flags(**SHIPPED, **CORE))})

    print("deeper ranking and more slots:")
    for c in (8, 12):
        rows.append({"variant": f"candidates {c}",
                     **run(p20, replace(s0, candidates=c), Flags(**SHIPPED))})
    for n in (5, 8):
        rows.append({"variant": f"max_names {n}",
                     **run(p20, replace(s0, max_names=n), Flags(**SHIPPED))})

    print("dropping the relative-strength gate:")
    rows.append({"variant": "no RS gate", **run(p20, s0, Flags(**SHIPPED), drop_rs=True)})
    rows.append({"variant": "no RS gate + core + 5 slots",
                 **run(p20, replace(s0, max_names=5), Flags(**SHIPPED, **CORE), drop_rs=True)})

    df = pd.DataFrame(rows)
    base = df.iloc[0]
    df["d_trades"] = df["trades"] - base["trades"]
    df["d_calmar"] = df["calmar"] - base["calmar"]
    print(f"\n=== DISCOVER {DISCOVER[0]} .. {DISCOVER[1]} (the holdout is NOT consulted) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nDid trading more often pay? (more trades AND better return per unit of drawdown)")
    better = df[(df.trades > base["trades"]) & (df.calmar > base["calmar"])]
    print(better[["variant", "trades", "exposure", "calmar"]].to_string(index=False,
          float_format=lambda x: f"{x:.4f}") if len(better) else "  nothing: more trades, no better")
    worse = df[(df.trades > base["trades"]) & (df.calmar <= base["calmar"])]
    if len(worse):
        print(f"\nTraded more and did WORSE: {', '.join(worse['variant'])}")

    (ROOT / "reports" / "more_signals.json").write_text(
        json.dumps({"window": DISCOVER, "rows": df.to_dict("records")}, indent=2, default=str))


if __name__ == "__main__":
    main()
