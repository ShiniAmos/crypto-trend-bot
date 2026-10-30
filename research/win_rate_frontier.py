"""What does a 90% win rate cost? The frontier between how OFTEN you win and how MUCH you make.

The request: not 100% profit, but a win rate above 90-95%. That is achievable - trivially, in fact
- and the only honest way to answer is to show the price rather than argue about it.

THE ARITHMETIC THAT GOVERNS ALL OF THIS

    expectancy = P(win) x average win  -  P(loss) x average loss

Those four numbers are not free of one another. Raising P(win) means closing winners sooner (which
shrinks the average win) and holding losers longer (which grows the average loss). A strategy with
a 95% win rate and a 1:40 win/loss ratio loses money; one with a 20% win rate and a 1:10 ratio
makes a fortune. The win rate on its own carries no information about profit.

So this sweeps a take-profit / stop-loss grid on real data and reports BOTH numbers for every cell.
At one end sit the high win rates; the CAGR column says what they are worth.

    take profit   +1% .. +50%     how far a trade must run before it is banked
    stop loss     -5% .. -60%     how far it may fall before it is cut
    entry         the shipped trend entry: above its own 50-day AND 200-day average

Also reported, because it reframes the question usefully: the BREAKEVEN WIN RATE for each cell -
the win rate that would be needed, at that cell's actual win/loss sizes, merely to not lose money.

    python research/win_rate_frontier.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402

BPD = 6
BPY = BPD * 365
COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
START = "2022-01-01"
TAKES = [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 0.35, 0.50]
STOPS = [0.05, 0.10, 0.20, 0.35, 0.60]


def load(sym):
    d = pd.read_parquet(ROOT / "data" / "pit" / "klines" / f"{sym}_4h.parquet")
    d["t"] = pd.to_datetime(d["open_time"], unit="ms", utc=True)
    return d.set_index("t").sort_index()[["open", "high", "low", "close"]].astype(float)


def run(px, entry, take, stop, cost):
    """Long on `entry`, out at +take or -stop, intrabar, filled next open on entry.

    The exits are checked against the bar's HIGH and LOW, so a target or a stop inside the bar
    counts - the conservative reading, and the one that stops a high win rate being an artefact of
    only ever looking at closes.
    """
    o, h, l, c = (px[k].to_numpy() for k in ("open", "high", "low", "close"))
    ent = entry.to_numpy()
    n = len(c)
    eq = np.ones(n)
    cash, units, pos, epx = 1.0, 0.0, 0.0, 0.0
    rets = []
    for i in range(1, n):
        if pos == 0.0 and ent[i - 1]:
            epx = o[i] * (1 + cost)
            units, cash, pos = cash / epx, 0.0, 1.0
        elif pos == 1.0:
            tp, sl = epx * (1 + take), epx * (1 - stop)
            hit_sl = l[i] <= sl
            hit_tp = h[i] >= tp
            # If both are touched in one bar, assume the stop first: the pessimistic order.
            px_out = sl if hit_sl else (tp if hit_tp else None)
            if px_out is not None:
                fill = px_out * (1 - cost)
                cash = units * fill
                rets.append(fill / epx - 1)
                units, pos = 0.0, 0.0
        eq[i] = cash if pos == 0.0 else units * c[i]
    eq = pd.Series(eq, index=px.index)
    r = np.array(rets)
    yrs = (px.index[-1] - px.index[0]).days / 365.25
    wins, losses = r[r > 0], r[r <= 0]
    out = {"trades": len(r),
           "win_rate": float((r > 0).mean()) if len(r) else np.nan,
           "avg_win": float(wins.mean()) if len(wins) else np.nan,
           "avg_loss": float(-losses.mean()) if len(losses) else np.nan,
           "cagr": float(eq.iloc[-1] ** (1 / yrs) - 1) if yrs > 0 else np.nan,
           "maxdd": float((eq / eq.cummax() - 1).min())}
    # the win rate this cell would NEED, at its own win/loss sizes, just to break even
    if len(wins) and len(losses):
        w, L = out["avg_win"], out["avg_loss"]
        out["breakeven_win_rate"] = float(L / (w + L))
        out["edge"] = out["win_rate"] - out["breakeven_win_rate"]
    return out


def main():
    s = config.load()
    cost = s.fee_rate + s.slippage
    rows = []
    for sym in COINS:
        px = load(sym)[START:]
        c = px["close"]
        sma50 = c.rolling(50 * BPD, min_periods=50 * BPD).mean()
        sma200 = c.rolling(200 * BPD, min_periods=200 * BPD).mean()
        entry = ((c > sma50) & (c > sma200)).shift(1).fillna(False)
        for take in TAKES:
            for stop in STOPS:
                r = run(px, entry, take, stop, cost)
                rows.append({"coin": sym.replace("USDT", ""), "take": take, "stop": stop, **r})
    df = pd.DataFrame(rows)

    print(f"{len(COINS)} coins, {START} onward, costs {cost:.4%} per side, "
          f"{len(TAKES)}x{len(STOPS)} = {len(TAKES) * len(STOPS)} cells each\n")

    agg = (df.groupby(["take", "stop"])
             .agg(win_rate=("win_rate", "mean"), cagr=("cagr", "mean"),
                  maxdd=("maxdd", "mean"), trades=("trades", "mean"),
                  avg_win=("avg_win", "mean"), avg_loss=("avg_loss", "mean"))
             .reset_index())

    print("=== WIN RATE by take-profit (rows) and stop-loss (columns) ===")
    print((agg.pivot(index="take", columns="stop", values="win_rate") * 100)
          .round(1).to_string())
    print("\n=== CAGR for the same cells, % a year ===")
    print((agg.pivot(index="take", columns="stop", values="cagr") * 100)
          .round(1).to_string())

    hi = agg[agg.win_rate >= 0.90].sort_values("win_rate", ascending=False)
    print(f"\n=== every cell with a win rate of 90% or more ({len(hi)} of {len(agg)}) ===")
    if len(hi):
        print(hi[["take", "stop", "win_rate", "cagr", "maxdd", "avg_win", "avg_loss", "trades"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"\n  best CAGR among them: {hi.cagr.max():+.2%}   "
              f"median: {hi.cagr.median():+.2%}   worst: {hi.cagr.min():+.2%}")
    best = agg.sort_values("cagr", ascending=False).head(6)
    print("\n=== the six most PROFITABLE cells, and what their win rates are ===")
    print(best[["take", "stop", "cagr", "win_rate", "maxdd", "avg_win", "avg_loss"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    d = df.dropna(subset=["win_rate", "cagr"])
    print(f"\ncorrelation across all {len(d)} coin-cells, win rate vs CAGR: "
          f"{d['win_rate'].corr(d['cagr']):+.4f}")
    (ROOT / "reports" / "win_rate_frontier.json").write_text(
        json.dumps(agg.to_dict("records"), indent=2, default=str))


if __name__ == "__main__":
    main()
