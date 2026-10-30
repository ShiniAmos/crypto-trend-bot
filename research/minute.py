"""Can this strategy trade on the minute?

Two questions, kept separate because they are different:

  A. same horizons, decided every minute - a 3.3-day channel checked 1,440 times a day instead
     of 6. This is what LEVEL_TF=1m would do.
  B. minute-scale horizons - an 80-minute channel, 7-hour and 30-hour momentum. This is a
     different strategy that happens to share the rules.

Both pay the real costs: 7 bps slippage and 0.10% fee per side, 0.27% round trip. Five liquid
coins, 120 days of genuine 1-minute candles.

    python research/minute.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import backtest, config, features  # noqa: E402
from app.flags import Flags  # noqa: E402

SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "LTCUSDT"]
SHIPPED = Flags(abs_momentum=True, volume_min=0.8, taker_min=0.5)
MIN_PER_DAY = 1440


def panel(donchian, bpd, min_age):
    frames = []
    for sym in SYMS:
        b = features.data.load(config.load().klines_dir, sym, "1m")
        if b.empty:
            continue
        d = features.data.load(config.load().klines_dir, sym, "1d")
        f = features.symbol_features(b, d, donchian=donchian, bpd=bpd, tf_ms=60_000)
        f["symbol"] = sym
        frames.append(f)
    return features.add_cross_section(pd.concat(frames, ignore_index=True), min_age_bars=min_age)


def main():
    s = config.load()
    rows = []

    print("A. same horizons (3.3-day channel), decided every minute...")
    pa = panel(donchian=20 * 240, bpd=MIN_PER_DAY, min_age=35 * MIN_PER_DAY)
    ra = backtest.run(pa, s, flags=SHIPPED)
    ma = backtest.metrics(ra.equity, ra.trades, ra.exposure, bars_per_year=MIN_PER_DAY * 365)
    rows.append(("A: 3.3-day channel, minute decisions", ma, ra))

    print("B. minute-scale horizons (80-minute channel)...")
    pb = panel(donchian=80, bpd=MIN_PER_DAY // 24, min_age=3 * MIN_PER_DAY)
    rb = backtest.run(pb, s, flags=SHIPPED)
    mb = backtest.metrics(rb.equity, rb.trades, rb.exposure, bars_per_year=MIN_PER_DAY * 365)
    rows.append(("B: 80-minute channel", mb, rb))

    print("\n120 days, five coins, costs on both legs:\n")
    out = []
    for name, m, res in rows:
        days = max((res.equity.index[-1] - res.equity.index[0]).days, 1)
        fees = res.log["fee"].sum() if len(res.log) else 0.0
        out.append({"run": name, "total_return": m["total_return"], "cagr": m["cagr"],
                    "sharpe": m["sharpe"], "maxdd": m["max_drawdown"],
                    "trades": m.get("trades", 0),
                    "trades/day": round(m.get("trades", 0) / days, 2),
                    "avg_trade": m.get("avg_trade"),
                    "fees_paid": round(fees), "fees_%start": round(100 * fees / s.start_equity, 1)})
    print(pd.DataFrame(out).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
