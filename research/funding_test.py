"""Does funding - crowding information from outside the price - improve the strategy?

Funding is paid every 8 hours by the crowded side of the perpetual market. High positive funding
means leveraged longs are paying to stay long, which is the closest thing crypto has to a direct
measure of how one-sided positioning has become.

Two opposite stories are plausible and both are tested, because assuming one is how you fool
yourself:
  - EXHAUSTION: high funding means the move is crowded and late, so avoid it
  - CONFIRMATION: high funding means real demand, so prefer it

Every variant keeps the engine, the costs, the sizing and the exits identical and changes only
which candidates pass or how they rank. Baseline is re-measured over the SAME dates, because
funding history starts in 2020 and comparing against a 2017-start baseline would be meaningless.

    python research/funding_test.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import backtest, config, features  # noqa: E402
from app.flags import Flags  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SHIPPED = Flags(abs_momentum=True, volume_min=0.8, taker_min=0.5)
HELD_OUT = "2022-01-01"
PER_DAY = 3                     # funding is paid every 8 hours


def attach_funding(panel, settings):
    """Merge each coin's funding history onto its 4h bars, backward only.

    A funding payment stamped 08:00 is known at 08:00, so merge_asof(direction='backward') with
    exact matches allowed is correct and cannot see the future.
    """
    out = []
    fdir = settings.data_dir / "funding"
    for sym, g in panel.groupby("symbol", sort=False):
        g = g.sort_values("bar_time")
        path = fdir / f"{sym}.parquet"
        if not path.exists():
            g = g.assign(funding=np.nan, funding_7d=np.nan, funding_30d=np.nan, funding_z=np.nan)
            out.append(g)
            continue
        f = pd.read_parquet(path)
        f["bar_time"] = pd.to_datetime(f["funding_time"], unit="ms", utc=True)
        f = f.sort_values("bar_time")
        # trailing statistics computed on the funding series itself, before the join
        f["funding_7d"] = f["funding_rate"].rolling(7 * PER_DAY).mean()
        f["funding_30d"] = f["funding_rate"].rolling(30 * PER_DAY).mean()
        mu = f["funding_rate"].rolling(90 * PER_DAY).mean()
        sd = f["funding_rate"].rolling(90 * PER_DAY).std()
        f["funding_z"] = (f["funding_7d"] - mu) / sd
        merged = pd.merge_asof(g, f[["bar_time", "funding_rate", "funding_7d", "funding_30d",
                                     "funding_z"]],
                               on="bar_time", direction="backward", allow_exact_matches=True)
        merged = merged.rename(columns={"funding_rate": "funding"})
        out.append(merged)
    p = pd.concat(out, ignore_index=True).sort_values(["bar_time", "symbol"]).reset_index(drop=True)
    # cross-sectional crowding: where does this coin sit against the others that bar
    el = p["eligible"] & p["funding_7d"].notna()
    p["funding_pct"] = np.nan
    p.loc[el, "funding_pct"] = p[el].groupby("bar_time")["funding_7d"].rank(pct=True)
    return p


def measure(panel, s, flags, start, label, score_col="score"):
    res = backtest.run(panel, s, flags=flags, start=start, score_col=score_col)
    m = backtest.metrics(res.equity, res.trades, res.exposure)
    held = backtest.run(panel, s, flags=flags, start=HELD_OUT, score_col=score_col)
    mh = backtest.metrics(held.equity, held.trades, held.exposure)
    rb = backtest.random_baselines(panel, res.trades, s) or {}
    return {"run": label, "cagr": m["cagr"], "sharpe": m["sharpe"], "maxdd": m["max_drawdown"],
            "trades": m.get("trades", 0), "win": m.get("win_rate"),
            "ho_cagr": mh["cagr"], "ho_sharpe": mh["sharpe"], "ho_maxdd": mh["max_drawdown"],
            "p_coin": rb.get("p_value_vs_random_coin", np.nan)}


def main():
    s = config.load()
    u = json.loads((ROOT / "reports" / "_parity_universe.json").read_text())
    base_panel = features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars)
    p = attach_funding(base_panel, s)

    have = p.groupby("bar_time")["funding_7d"].apply(lambda x: x.notna().sum())
    start = have[have >= 8].index.min()
    print(f"funding covers 8+ coins from {start}")
    cov = p["funding_7d"].notna().mean()
    print(f"{cov:.1%} of panel rows have funding; "
          f"{p.loc[p['eligible'], 'funding_7d'].notna().mean():.1%} of eligible rows\n")

    # Is funding even related to what happens next? Check before building strategies on it.
    d = p[p["eligible"] & p["funding_7d"].notna() & p["ret_7d"].notna()].copy()
    d["fwd"] = d.groupby("symbol")["close"].shift(-42) / d["close"] - 1
    d = d.dropna(subset=["fwd"])
    q = pd.qcut(d["funding_pct"], 5, labels=["least crowded", "2", "3", "4", "most crowded"],
                duplicates="drop")
    print("Forward 7-day return by funding crowding (all eligible bars):")
    print(d.groupby(q, observed=True)["fwd"].agg(["size", "mean", "median"]).to_string(
        float_format=lambda x: f"{x:.4f}"))
    print(f"\nSpearman(funding percentile, forward 7d return): "
          f"{d['funding_pct'].corr(d['fwd'], method='spearman'):+.4f}\n")

    rows = [measure(p, s, SHIPPED, start, "baseline (same dates)")]

    # EXHAUSTION: refuse the most crowded candidates
    for cut in (0.8, 0.9):
        v = p.copy()
        v["breakout"] = v["breakout"] & ~(v["funding_pct"] > cut).fillna(False)
        rows.append(measure(v, s, SHIPPED, start, f"skip funding_pct > {cut:g}"))
    for z in (1.5, 2.0):
        v = p.copy()
        v["breakout"] = v["breakout"] & ~(v["funding_z"] > z).fillna(False)
        rows.append(measure(v, s, SHIPPED, start, f"skip funding_z > {z:g}"))
    v = p.copy()
    v["breakout"] = v["breakout"] & ~(v["funding_7d"] > 0.0005).fillna(False)
    rows.append(measure(v, s, SHIPPED, start, "skip funding_7d > 0.05%/8h"))

    # CONFIRMATION: the opposite - demand only the crowded ones
    v = p.copy()
    v["breakout"] = v["breakout"] & (v["funding_pct"] > 0.5).fillna(False)
    rows.append(measure(v, s, SHIPPED, start, "require funding_pct > 0.5"))

    # RANKING: tilt the score away from (then toward) crowding
    for w, name in ((-0.20, "score minus 0.20 x crowding"), (+0.20, "score plus 0.20 x crowding")):
        v = p.copy()
        v["score_f"] = v["score"] + w * 100 * v["funding_pct"].fillna(0.5)
        v["score_f"] = v["score_f"].where(v["eligible"])
        rows.append(measure(v, s, SHIPPED, start, name, score_col="score_f"))

    # SIZING: smaller when crowded, never deleting a candidate
    v = p.copy()
    v["fsize"] = (1.5 - v["funding_pct"].fillna(0.5)).clip(0.5, 1.5)
    res = backtest.run(v, s, flags=SHIPPED, start=start, size_scale_col="fsize")
    m = backtest.metrics(res.equity, res.trades, res.exposure)
    held = backtest.run(v, s, flags=SHIPPED, start=HELD_OUT, size_scale_col="fsize")
    mh = backtest.metrics(held.equity, held.trades, held.exposure)
    rows.append({"run": "size down when crowded", "cagr": m["cagr"], "sharpe": m["sharpe"],
                 "maxdd": m["max_drawdown"], "trades": m.get("trades", 0),
                 "win": m.get("win_rate"), "ho_cagr": mh["cagr"], "ho_sharpe": mh["sharpe"],
                 "ho_maxdd": mh["max_drawdown"], "p_coin": np.nan})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    b = rows[0]
    print("\nShip rule against the same-dates baseline (CAGR, Sharpe, maxDD on BOTH panels):")
    for r in rows[1:]:
        ok = (r["cagr"] >= b["cagr"] and r["sharpe"] >= b["sharpe"] and r["maxdd"] >= b["maxdd"]
              and r["ho_cagr"] >= b["ho_cagr"] and r["ho_sharpe"] >= b["ho_sharpe"]
              and r["ho_maxdd"] >= b["ho_maxdd"])
        print(f"  {'PASS' if ok else 'fail'}  {r['run']}")
    (ROOT / "reports" / "funding_test.json").write_text(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
