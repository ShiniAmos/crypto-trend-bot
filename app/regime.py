"""Market regimes, labelled causally, so performance can be read per environment.

Every label at bar t uses only BTC data closed by t: its position against the 200-day average and
its trailing 90-day return. Nothing here may look forward, or the regime table would flatter the
strategy by hindsight-selecting the good periods.
"""

import numpy as np
import pandas as pd

BARS_PER_DAY = 6
STRONG = 0.25      # 90-day BTC return above this, while above trend, is a strong bull
FLAT = 0.05        # above trend but going nowhere
REGIMES = ("strong bull", "weak bull", "sideways/chop", "recovery", "bear")


def label_bars(panel: pd.DataFrame) -> pd.Series:
    """bar_time -> regime label, from BTC only."""
    btc = panel[panel["symbol"] == "BTCUSDT"].set_index("bar_time").sort_index()
    if btc.empty:
        raise RuntimeError("BTC history is required to label regimes")
    above = btc["dist_sma200"] > 0
    ret90 = btc["close"].pct_change(90 * BARS_PER_DAY)
    out = pd.Series("sideways/chop", index=btc.index, name="regime")
    out[above & (ret90 > STRONG)] = "strong bull"
    out[above & (ret90 <= STRONG) & (ret90 >= FLAT)] = "weak bull"
    out[above & (ret90 < FLAT)] = "sideways/chop"
    out[~above & (ret90 >= 0)] = "recovery"
    out[~above & (ret90 < 0)] = "bear"
    out[ret90.isna()] = "sideways/chop"      # first 90 days: no trailing return yet
    return out


def per_regime(equity: pd.Series, labels: pd.Series, trades: pd.DataFrame = None,
               exposure: pd.Series = None) -> pd.DataFrame:
    """Metrics inside each regime. Segments are not contiguous, so drawdown is measured across the
    concatenated bars of that regime and returns are compounded across them."""
    lab = labels.reindex(equity.index).ffill()
    r = equity.pct_change()
    rows = []
    for name in REGIMES:
        m = lab == name
        bars = int(m.sum())
        if bars < 2:
            continue
        rr = r[m].dropna()
        curve = (1 + rr).cumprod()
        years = bars / (BARS_PER_DAY * 365)
        total = curve.iloc[-1] - 1 if len(curve) else 0.0
        dd = (curve / curve.cummax() - 1).min() if len(curve) else 0.0
        row = {"regime": name, "bars": bars, "years": round(years, 2),
               "total_return": total,
               "cagr": (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else np.nan,
               "sharpe": rr.mean() / rr.std() * np.sqrt(BARS_PER_DAY * 365) if rr.std() > 0 else 0.0,
               "max_drawdown": dd}
        if exposure is not None:
            row["exposure"] = exposure[m].mean()
        if trades is not None and len(trades):
            t = trades[trades["entry_time"].map(lambda x: lab.get(pd.Timestamp(x), None)) == name]
            row["trades"] = len(t)
            if len(t):
                wins, losses = t.loc[t.pnl > 0, "pnl"].sum(), -t.loc[t.pnl <= 0, "pnl"].sum()
                row["win_rate"] = (t.pnl > 0).mean()
                row["avg_trade"] = t.ret.mean()
                row["profit_factor"] = wins / losses if losses > 0 else np.inf
        rows.append(row)
    return pd.DataFrame(rows)
