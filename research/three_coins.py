"""SOL, XRP and BNB: medium holds, and the honest cost of never selling at a loss.

The brief: hold for a MEDIUM period, never sell below the buying price, test on SOL/XRP/BNB, and
show a positive backtest. Three of those four are reasonable and one is a trap, so all four are
measured rather than argued about.

WHY "NEVER SELL AT A LOSS" NEEDS MEASURING, NOT ASSERTING
It cannot lose a trade: every closed trade is a winner, by construction. What it does instead is
refuse to close losers, so they stop being trades and start being a frozen portfolio. The money is
still gone - it is just no longer counted. Two numbers expose it, and neither is the win rate:

    max drawdown    the mark-to-market truth, which does not care what has been "realised"
    time in losers  capital locked in a position that is underwater and going nowhere

HOLDING PERIOD, made explicit rather than implied
    short    exit on a 10-bar Donchian low        (~1.7 days of channel)
    medium   exit on a 30-bar low, or 60 bars max (~5-10 days)    <- what was asked for
    long     exit only when it loses its 200-day average          (the trend core)

STRATEGIES TESTED on each coin, single-asset, long only, real costs:
    buy_hold        the benchmark that must be beaten; any rule that loses to it is noise
    sma200          long while above its own 200-day average
    sma50_200       long while above both - a medium-term trend filter
    donchian_med    buy a 20-bar high, sell a 30-bar low
    donchian_med_nl the same, but never selling below the entry price
    sma200_nl       sma200, but never selling below the entry price

Costs: the live settings, 0.10% fee and 7 bps slippage per side, charged on every change of
position. Causal throughout: a weight held during a bar is decided by bars that had already closed.

    python research/three_coins.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402

BPD = 6                      # 4h bars per day
BPY = BPD * 365
COINS = ["SOLUSDT", "XRPUSDT", "BNBUSDT", "BTCUSDT", "ETHUSDT"]
ERAS = {"full": (None, None), "2022-2026 (modern)": ("2022-01-01", None),
        "2025-2026": ("2025-01-01", None)}


def load(sym):
    d = pd.read_parquet(ROOT / "data" / "pit" / "klines" / f"{sym}_4h.parquet")
    d["t"] = pd.to_datetime(d["open_time"], unit="ms", utc=True)
    d = d.set_index("t").sort_index()
    return d[["open", "high", "low", "close"]].astype(float)


def simulate(px, entry_sig, exit_sig, cost, never_loss=False, max_bars=None):
    """Long/flat on explicit entry and exit signals, filled at the NEXT bar's open.

    Returns the equity curve, the trade list, and - the number that matters for `never_loss` - the
    fraction of bars spent holding a position that is underwater.
    """
    o = px["open"].to_numpy()
    c = px["close"].to_numpy()
    n = len(c)
    ent = entry_sig.to_numpy()
    ex = exit_sig.to_numpy()
    eq = np.ones(n)
    pos = 0.0
    entry_px = 0.0
    bars_held = 0
    cash = 1.0
    units = 0.0
    trades, underwater = [], 0
    for i in range(1, n):
        # act at THIS bar's open on the PREVIOUS bar's signal
        if pos == 0.0 and ent[i - 1]:
            entry_px = o[i] * (1 + cost)
            units = cash / entry_px
            cash = 0.0
            pos = 1.0
            bars_held = 0
        elif pos == 1.0:
            bars_held += 1
            want_out = ex[i - 1] or (max_bars is not None and bars_held >= max_bars)
            if want_out and never_loss and c[i - 1] <= entry_px:
                want_out = False            # refuse to close below the buying price
            if want_out:
                exit_px = o[i] * (1 - cost)
                cash = units * exit_px
                trades.append({"entry": entry_px, "exit": exit_px, "bars": bars_held,
                               "ret": exit_px / entry_px - 1})
                units, pos = 0.0, 0.0
        eq[i] = cash if pos == 0.0 else units * c[i]
        if pos == 1.0 and c[i] < entry_px:
            underwater += 1
    return (pd.Series(eq, index=px.index), pd.DataFrame(trades),
            underwater / max(n, 1))


def stats(eq, trades, underwater, label):
    r = np.log(eq).diff().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    out = {"strategy": label,
           "cagr": eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan,
           "sharpe": float(r.mean() / r.std() * np.sqrt(BPY)) if r.std() > 0 else 0.0,
           "maxdd": float((eq / eq.cummax() - 1).min()),
           "trades": len(trades),
           "win_rate": float((trades["ret"] > 0).mean()) if len(trades) else np.nan,
           "med_hold_days": float(trades["bars"].median() / BPD) if len(trades) else np.nan,
           "pct_bars_underwater": underwater}
    out["calmar"] = out["cagr"] / abs(out["maxdd"]) if out["maxdd"] < 0 else np.nan
    return out


def strategies(px):
    c = px["close"]
    sma50 = c.rolling(50 * BPD, min_periods=50 * BPD).mean()
    sma200 = c.rolling(200 * BPD, min_periods=200 * BPD).mean()
    hi20 = px["high"].rolling(20).max().shift(1)
    lo30 = px["low"].rolling(30).min().shift(1)
    above200 = (c > sma200).fillna(False)
    above50 = (c > sma50).fillna(False)
    always = pd.Series(True, index=c.index)
    never = pd.Series(False, index=c.index)
    return {
        "buy_hold":        (always, never, False, None),
        "sma200":          (above200, ~above200, False, None),
        "sma50_200":       (above200 & above50, ~(above200 & above50), False, None),
        "donchian_med":    ((c > hi20).fillna(False), (c < lo30).fillna(False), False, 60),
        "donchian_med_nl": ((c > hi20).fillna(False), (c < lo30).fillna(False), True, 60),
        "sma200_nl":       (above200, ~above200, True, None),
    }


def main():
    s = config.load()
    cost = s.fee_rate + s.slippage
    print(f"costs {cost:.4%} per side (fee {s.fee_rate:.3%} + slippage {s.slippage:.3%})\n")
    all_rows = []
    for era, (lo, hi) in ERAS.items():
        print("=" * 100)
        print(f"ERA {era}")
        for sym in COINS:
            px = load(sym)
            px = px[lo:hi] if (lo or hi) else px
            if len(px) < 300 * BPD:
                print(f"  {sym}: not enough history in this era, skipped")
                continue
            rows = []
            for name, (ent, ex, nl, mb) in strategies(px).items():
                eq, tr, uw = simulate(px, ent.reindex(px.index).fillna(False),
                                      ex.reindex(px.index).fillna(False), cost,
                                      never_loss=nl, max_bars=mb)
                rows.append({"coin": sym.replace("USDT", ""), **stats(eq, tr, uw, name), "era": era})
            df = pd.DataFrame(rows)
            all_rows += rows
            print(f"\n  --- {sym.replace('USDT','')} ---")
            print(df[["strategy", "cagr", "sharpe", "maxdd", "calmar", "trades",
                      "win_rate", "med_hold_days", "pct_bars_underwater"]]
                  .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print()
    (ROOT / "reports" / "three_coins.json").write_text(
        json.dumps(all_rows, indent=2, default=str))


if __name__ == "__main__":
    main()
