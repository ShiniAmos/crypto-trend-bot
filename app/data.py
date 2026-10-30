"""Public Binance REST access and local parquet kline storage (section 4)."""

import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

from .config import BINANCE_URLS

TF_MS = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote", "ignore"]
KEEP = ["open_time", "open", "high", "low", "close", "volume", "quote_volume",
        "taker_buy_volume", "close_time"]

_session = requests.Session()


class RateLimited(Exception):
    pass


# Public market data is served by two Binance hosts; if one is down or unreachable, use the other.
PUBLIC_BASES = [BINANCE_URLS["public"], "https://data-api.binance.vision"]


def get(path, params=None, base=BINANCE_URLS["public"], tries=6):
    """GET with backoff on 429/418/5xx and network errors, switching to the backup host for
    public data when the main one fails."""
    delay = 1.0
    bases = PUBLIC_BASES if base == BINANCE_URLS["public"] else [base]
    for attempt in range(tries):
        host = bases[attempt % len(bases)]
        try:
            r = _session.get(host + path, params=params, timeout=10)
        except requests.RequestException:
            if attempt == tries - 1:
                raise
            if attempt % len(bases) == len(bases) - 1:  # every host failed this round: wait
                time.sleep(delay)
                delay *= 2
            continue
        if r.status_code in (451, 403) and len(bases) > 1 and attempt < tries - 1:
            continue  # region-blocked on this host: the backup host serves the same data
        if r.status_code in (429, 418):
            wait = float(r.headers.get("Retry-After", delay))
            if attempt == tries - 1:
                raise RateLimited(f"{path} rate limited")
            time.sleep(wait)
            delay *= 2
            continue
        if r.status_code >= 500 and attempt < tries - 1:
            time.sleep(delay)
            delay *= 2
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("unreachable")


def now_ms() -> int:
    return int(time.time() * 1000)


def fetch_klines(symbol, tf, start_ms=0, end_ms=None, limit=1000):
    """One page of klines as a DataFrame (closed candles only)."""
    params = {"symbol": symbol, "interval": tf, "startTime": start_ms, "limit": limit}
    if end_ms:
        params["endTime"] = end_ms
    rows = get("/api/v3/klines", params)
    df = pd.DataFrame(rows, columns=KLINE_COLS)[KEEP] if rows else pd.DataFrame(columns=KEEP)
    return closed_only(_types(df))


def _types(df):
    for c in ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume"):
        df[c] = df[c].astype(float)
    for c in ("open_time", "close_time"):
        df[c] = df[c].astype("int64")
    return df


def closed_only(df, at_ms=None):
    """Rule: unfinished candles are never used as history."""
    at_ms = at_ms or now_ms()
    return df[df["close_time"] < at_ms].reset_index(drop=True)


def first_candle_ms(symbol, tf="1d"):
    rows = get("/api/v3/klines", {"symbol": symbol, "interval": tf, "startTime": 0, "limit": 1})
    return int(rows[0][0]) if rows else None


def path_for(klines_dir: Path, symbol, tf) -> Path:
    return Path(klines_dir) / f"{symbol}_{tf}.parquet"


def load(klines_dir, symbol, tf) -> pd.DataFrame:
    p = path_for(klines_dir, symbol, tf)
    if not p.exists():
        return pd.DataFrame(columns=KEEP)
    return pd.read_parquet(p)


def save(klines_dir, symbol, tf, df):
    df = (df.drop_duplicates("open_time", keep="last")
            .sort_values("open_time").reset_index(drop=True))
    dest = path_for(klines_dir, symbol, tf)
    tmp = dest.with_suffix(f".{os.getpid()}.tmp")   # per-process, so two bots never clash
    df.to_parquet(tmp, index=False)
    for attempt in range(5):                        # Windows: the file may be briefly locked
        try:
            tmp.replace(dest)
            break
        except PermissionError:
            if attempt == 4:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.5)
    return df


def exchange_info():
    return get("/api/v3/exchangeInfo", {"permissions": "SPOT"})


def ticker_24h():
    return {t["symbol"]: t for t in get("/api/v3/ticker/24hr", {"type": "MINI"})}


def price(symbol) -> float:
    return float(get("/api/v3/ticker/price", {"symbol": symbol})["price"])


def prices(symbols) -> dict:
    """Live price for several symbols in one request, for the live cycle."""
    symbols = list(symbols)
    if not symbols:
        return {}
    if len(symbols) == 1:
        return {symbols[0]: price(symbols[0])}
    rows = get("/api/v3/ticker/price", {"symbols": json.dumps(symbols, separators=(",", ":"))})
    return {r["symbol"]: float(r["price"]) for r in rows}
