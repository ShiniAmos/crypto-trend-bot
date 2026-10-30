"""Which years actually paid? Calendar-year attribution of the shipped configuration.

A multi-year CAGR is an average, and an average hides the thing you most need to know: whether the
return came from every year or from one. A strategy whose whole edge sits in 2021 is a bet that
2021 comes back, not a strategy.

Both panels are reported because they answer different questions:

    standard   today's coin list applied to all history. Flattering - those coins are the
               survivors - but it is the panel every number in RESEARCH_LOG.md's headline rows uses.
    pit        survivorship-free: each day's universe is that day's top 30 by trailing volume,
               including coins later delisted. The honest one, and the reason the shipped CAGR is
               quoted as 10.71% rather than 27%.

Offline: both panels are rebuilt from the local cache (data/klines and data/pit/klines), so this
makes no network calls and the pit universe is taken from the cached symbol set rather than
re-listed from the archive.

    python research/year_attribution.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import backtest, config, features, pit  # noqa: E402
from app.flags import Flags  # noqa: E402

SHIPPED = Flags(abs_momentum=True, volume_min=0.8, taker_min=0.5, funding_max_pct=0.8)


def year_table(equity: pd.Series, trades: pd.DataFrame) -> pd.DataFrame:
    """Per calendar year: return, the year's own worst drawdown, trades, and share of the total."""
    eq = equity.dropna()
    eq.index = pd.to_datetime(eq.index, utc=True)
    rows = []
    for y, g in eq.groupby(eq.index.year):
        start, end = g.iloc[0], g.iloc[-1]
        prev = eq[eq.index < g.index[0]]
        open_eq = prev.iloc[-1] if len(prev) else start
        ret = end / open_eq - 1.0
        dd = (g / g.cummax() - 1.0).min()
        row = {"year": y, "return": ret, "maxdd_in_year": dd,
               "growth_mult": end / open_eq}
        if trades is not None and len(trades):
            t = trades.copy()
            tcol = "exit_time" if "exit_time" in t else ("time" if "time" in t else None)
            if tcol:
                t[tcol] = pd.to_datetime(t[tcol], utc=True)
                ty = t[t[tcol].dt.year == y]
                row["trades"] = len(ty)
                for c in ("net", "pnl", "return"):
                    if c in ty and len(ty):
                        row["win_rate"] = float((ty[c] > 0).mean())
                        break
        rows.append(row)
    df = pd.DataFrame(rows)
    total = df["growth_mult"].prod()
    # share of the compounded result each year is responsible for, in log space so it adds to 100%
    lg = np.log(df["growth_mult"].clip(lower=1e-9))
    df["share_of_profit"] = lg / lg.sum() if lg.sum() != 0 else np.nan
    df.attrs["total_mult"] = total
    return df


def report(name, equity, trades, exposure):
    m = backtest.metrics(equity, trades, exposure)
    df = year_table(equity, trades)
    print(f"\n{'=' * 78}\n{name}")
    print(f"  overall: CAGR {m['cagr']:.2%}  Sharpe {m['sharpe']:.4f}  "
          f"maxDD {m['max_drawdown']:.2%}  trades {m.get('trades', 0)}  "
          f"total x{df.attrs['total_mult']:.2f}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    pos = df[df["return"] > 0]
    print(f"  profitable years: {len(pos)}/{len(df)}")
    top = df.loc[df["share_of_profit"].idxmax()]
    print(f"  single biggest year: {int(top['year'])} at {top['return']:+.1%}, "
          f"{top['share_of_profit']:.1%} of the compounded result")
    # the honest stress test: drop the best year and see what is left
    without = df[df["year"] != top["year"]]["growth_mult"].prod()
    yrs = len(without_years := df[df["year"] != top["year"]])
    cagr_wo = without ** (1 / max(yrs, 1)) - 1 if yrs else float("nan")
    print(f"  without {int(top['year'])}: x{without:.2f} over {len(without_years)} years "
          f"= {cagr_wo:.2%} a year")

    # The era split. 2019-2021 was the altcoin bull; everything after it is a different market,
    # and the question "is this strategy still working" is answered here and nowhere else.
    eras = {}
    for label, lo, hi in (("2019-2021", 2019, 2021), ("2022-2026", 2022, 2026)):
        g = df[(df["year"] >= lo) & (df["year"] <= hi)]
        n = int((g["trades"].fillna(0) > 0).sum()) if "trades" in g else len(g)
        mult = g["growth_mult"].prod()
        span = max(len(g), 1)
        eras[label] = {"mult": float(mult), "years": len(g), "active_years": n,
                       "cagr": float(mult ** (1 / span) - 1)}
        print(f"  {label}: x{mult:.2f} over {len(g)} years = "
              f"{eras[label]['cagr']:.2%} a year  ({n} years with trades)")
    return {"name": name, "cagr": m["cagr"], "sharpe": m["sharpe"],
            "maxdd": m["max_drawdown"], "years": df.to_dict("records"),
            "profitable_years": f"{len(pos)}/{len(df)}", "best_year": int(top["year"]),
            "cagr_without_best": float(cagr_wo), "eras": eras}


def standard(s):
    u = json.loads((ROOT / "reports" / "_parity_universe.json").read_text())
    panel = features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars)
    r = backtest.run(panel, s, flags=SHIPPED)
    return report(f"STANDARD panel - {len(u['symbols'])} coins, today's list over all history",
                  r.equity, r.trades, r.exposure)


def pit_panel(s, top_n=30):
    kdir = s.data_dir / "pit" / "klines"
    # the cache is flat files, <SYMBOL>_<tf>.parquet, not one directory per symbol
    syms = sorted({p.name.rsplit("_", 1)[0] for p in kdir.glob("*_1d.parquet")}
                  if kdir.exists() else set())
    syms = [x for x in syms if pit.keep(x, s.exclude_memes)]
    if not syms:
        print("\n(no cached pit klines - skipping the survivorship-free panel)")
        return None
    member = pit.membership(kdir, syms, top_n)
    ever = sorted(set(member.columns[member.any()]) | {"BTCUSDT"})
    panel = features.build_panel(kdir, ever, s.donchian_bars)
    panel["day"] = (panel["bar_time"] - pd.Timedelta(milliseconds=1)).dt.normalize()
    m = member.rename_axis(index="day", columns="symbol").stack().rename("member").reset_index()
    panel = panel.merge(m, on=["day", "symbol"], how="left")
    panel["member"] = panel["member"].astype("boolean").fillna(False).astype(bool)
    panel["eligible"] = panel["eligible"] & (panel["member"] | (panel["symbol"] == "BTCUSDT"))
    panel = features.score(panel.drop(columns=["score", "rank", "n_eligible"]))
    r = backtest.run(panel, s, flags=SHIPPED)
    return report(f"PIT panel - survivorship-free, {len(ever)} coins ever in the top {top_n}",
                  r.equity, r.trades, r.exposure)


def main():
    s = config.load()
    print(f"shipped flags: {SHIPPED.label}")
    out = [standard(s)]
    p = pit_panel(s)
    if p:
        out.append(p)
    (ROOT / "reports" / "year_attribution.json").write_text(
        json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
