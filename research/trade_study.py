"""What separates a winning trade from a losing one?

Research only: nothing here is imported by the bot. The method is deliberately conservative.

    discover on 2017-2021, validate on 2022-2026

A relationship that only shows up in the discovery half is noise, and there are enough features
here that a few will look good by chance. Anything that survives both halves still has to clear the
ship rule on the standard and pit panels before it can be considered a strategy change.

    python research/trade_study.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import backtest, config, features, universe  # noqa: E402

SPLIT = "2022-01-01"
# Everything available at the moment of the entry decision. No forward-looking columns.
CANDIDATES = ["rank", "score", "rs_7d", "rs_30d", "ret_7d", "ret_30d", "dist_sma50",
              "dist_sma200", "vol_30d", "vol_7d", "atr_pct", "volume_ratio", "taker_ratio",
              "don_pos", "btc_dist_sma200", "btc_ret_7d", "btc_ret_30d",
              "breadth", "n_eligible", "rs_accel", "vol_ratio_btc", "stop_distance_pct"]


def build():
    s = config.load()
    u = universe.load(s)
    panel = features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars)

    # Derived columns that the bot does not compute today, but could.
    el = panel["eligible"] & (panel["symbol"] != "BTCUSDT")
    breadth = (panel[el].assign(pos=lambda d: d["rs_30d"] > 0)
               .groupby("bar_time")["pos"].mean().rename("breadth"))
    panel = panel.join(breadth, on="bar_time")
    panel["rs_accel"] = panel["rs_7d"] - panel["rs_30d"]          # is the lead widening?
    btc_vol = (panel[panel.symbol == "BTCUSDT"].set_index("bar_time")["vol_30d"]
               .rename("btc_vol_30d"))
    panel = panel.join(btc_vol, on="bar_time")
    panel["vol_ratio_btc"] = panel["vol_30d"] / panel["btc_vol_30d"]
    panel["stop_distance_pct"] = 1 - panel["stop_level"] / panel["close"]

    res = backtest.run(panel, s)
    trades = res.trades.copy()
    # The decision was taken on the bar BEFORE the fill: that is the row to study.
    bars = sorted(panel["bar_time"].unique())
    prev = {t: bars[i - 1] for i, t in enumerate(bars) if i}
    trades["signal_bar"] = trades["entry_time"].map(prev)
    keyed = panel.set_index(["bar_time", "symbol"])
    rows = []
    for t in trades.itertuples():
        try:
            row = keyed.loc[(t.signal_bar, t.symbol)]
        except KeyError:
            continue
        rows.append({**{c: row.get(c, np.nan) for c in CANDIDATES},
                     "ret": t.ret, "win": t.ret > 0, "bars_held": t.bars_held,
                     "entry_time": t.entry_time, "symbol": t.symbol})
    return pd.DataFrame(rows), panel, s


def terciles(df, col, label):
    """Average trade return in the bottom / middle / top third of a feature."""
    d = df[[col, "ret", "win"]].dropna()
    if len(d) < 60 or d[col].nunique() < 10:
        return None
    q = pd.qcut(d[col], 3, labels=["low", "mid", "high"], duplicates="drop")
    if q.nunique() < 3:
        return None
    g = d.groupby(q, observed=True)
    out = {"feature": col, "half": label, "n": len(d)}
    for name in ("low", "mid", "high"):
        sub = g.get_group(name) if name in g.groups else None
        out[f"{name}_ret"] = sub["ret"].mean() if sub is not None else np.nan
        out[f"{name}_win"] = sub["win"].mean() if sub is not None else np.nan
    out["spread"] = out["high_ret"] - out["low_ret"]
    hi = g.get_group("high")["ret"] if "high" in g.groups else pd.Series(dtype=float)
    lo = g.get_group("low")["ret"] if "low" in g.groups else pd.Series(dtype=float)
    pooled = np.sqrt(hi.var() / max(len(hi), 1) + lo.var() / max(len(lo), 1))
    out["t_stat"] = (hi.mean() - lo.mean()) / pooled if pooled > 0 else np.nan
    return out


def main():
    df, panel, s = build()
    early = df[df["entry_time"] < SPLIT]
    late = df[df["entry_time"] >= SPLIT]
    print(f"{len(df)} trades: {len(early)} before {SPLIT} (discovery), {len(late)} after (validation)")
    print(f"average trade: all {df.ret.mean():+.2%} | discovery {early.ret.mean():+.2%} "
          f"| validation {late.ret.mean():+.2%}")

    rows = []
    for col in CANDIDATES:
        for name, part in (("discovery", early), ("validation", late)):
            r = terciles(part, col, name)
            if r:
                rows.append(r)
    res = pd.DataFrame(rows)
    wide = res.pivot(index="feature", columns="half",
                     values=["spread", "t_stat", "high_ret", "low_ret"])
    wide.columns = ["_".join(c) for c in wide.columns]
    wide = wide.dropna()
    # Survives = the top third beats the bottom third in BOTH halves, same direction.
    wide["same_sign"] = np.sign(wide["spread_discovery"]) == np.sign(wide["spread_validation"])
    wide["min_abs_t"] = wide[["t_stat_discovery", "t_stat_validation"]].abs().min(axis=1)
    wide = wide.sort_values(["same_sign", "min_abs_t"], ascending=[False, False])
    cols = ["spread_discovery", "spread_validation", "t_stat_discovery", "t_stat_validation",
            "same_sign"]
    print("\nTop third minus bottom third, average trade return:\n")
    print(wide[cols].to_string(float_format=lambda x: f"{x:+.3f}"))

    survivors = wide[wide["same_sign"] & (wide["min_abs_t"] > 1.0)]
    print(f"\nSurvive both halves with |t| > 1: {list(survivors.index) or 'none'}")
    out = Path(__file__).resolve().parent.parent / "reports" / "trade_study.csv"
    wide.to_csv(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
