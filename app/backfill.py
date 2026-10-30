"""Phase 3: full Binance kline history for every universe symbol, resumable (section 4.1)."""

import pandas as pd

from . import data

TIMEFRAMES = ("1d", "4h", "1h")


def backfill_symbol(klines_dir, symbol, tf, log=print):
    """Download from the last stored close (or the first Binance candle) up to the last closed bar."""
    df = data.load(klines_dir, symbol, tf)
    start = int(df["open_time"].iloc[-1]) + data.TF_MS[tf] if len(df) else 0
    pages = []
    while True:
        page = data.fetch_klines(symbol, tf, start_ms=start)
        if page.empty:
            break
        pages.append(page)
        start = int(page["open_time"].iloc[-1]) + data.TF_MS[tf]
        if len(page) < 1000:
            break
    if pages:
        df = data.save(klines_dir, symbol, tf, pd.concat(([df] if len(df) else []) + pages, ignore_index=True))
        log(f"  {symbol} {tf}: +{sum(len(p) for p in pages)} bars, {len(df)} total")
    return df


def run(settings, symbols, timeframes=TIMEFRAMES, log=print):
    for sym in symbols:
        for tf in timeframes:
            backfill_symbol(settings.klines_dir, sym, tf, log=log)


def check(settings, symbols, tf="4h"):
    """Exit check for phase 3: row counts, first date, no duplicate timestamps, no gaps invented."""
    rows = []
    for sym in symbols:
        df = data.load(settings.klines_dir, sym, tf)
        if df.empty:
            rows.append({"symbol": sym, "bars": 0})
            continue
        gaps = int((df["open_time"].diff().dropna() != data.TF_MS[tf]).sum())
        rows.append({"symbol": sym, "bars": len(df),
                     "first": pd.to_datetime(df["open_time"].iloc[0], unit="ms").date(),
                     "last_close": pd.to_datetime(df["close_time"].iloc[-1], unit="ms"),
                     "duplicates": int(df["open_time"].duplicated().sum()),
                     "gaps(exchange outages)": gaps})
    return pd.DataFrame(rows)
