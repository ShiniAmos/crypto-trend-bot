"""Perpetual-futures funding rates: the first signal in this bot that is not derived from price.

Funding is what leveraged traders pay every eight hours to hold a perpetual position. When
everyone is long with borrowed money, funding goes positive and longs bleed to shorts. It is a
direct reading of how crowded a trade has become, and nothing in a candle contains it.

That matters because nine families of price-derived ideas and six machine-learning formulations
found nothing in this project. Funding found something on the first attempt: skipping a breakout
whose funding sits in the top fifth of the universe improved return, Sharpe and drawdown on the
standard panel, the held-out half and the survivorship-free panel, survived 2x costs, and beat a
40-draw placebo null 40-0.

Spot trading needs no futures account for this - the funding history is public market data.
"""

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

FAPI = "https://fapi.binance.com/fapi/v1/fundingRate"
PER_DAY = 3                 # funding is paid every 8 hours
WINDOW_DAYS = 7             # the trailing average the filter reads
FIRST_MS = 1_500_000_000_000
_session = requests.Session()


def dir_for(settings):
    d = settings.data_dir / "funding"
    d.mkdir(parents=True, exist_ok=True)
    return d


def path_for(settings, symbol):
    return dir_for(settings) / f"{symbol}.parquet"


def load(settings, symbol) -> pd.DataFrame:
    p = path_for(settings, symbol)
    if not p.exists():
        return pd.DataFrame(columns=["funding_time", "funding_rate"])
    return pd.read_parquet(p)


def backfill_symbol(settings, symbol, log=print):
    """Download funding history from the last stored payment onward. Resumable, like the klines."""
    have = load(settings, symbol)
    start = int(have["funding_time"].iloc[-1]) + 1 if len(have) else FIRST_MS
    rows = []
    while True:
        try:
            r = _session.get(FAPI, params={"symbol": symbol, "startTime": start, "limit": 1000},
                             timeout=30)
            if r.status_code >= 400:
                return have          # no perpetual contract for this coin, which is not an error
            page = r.json()
        except requests.RequestException as e:
            log(f"  funding {symbol}: {e}")
            return have
        if not page:
            break
        rows += page
        if len(page) < 1000:
            break
        start = int(page[-1]["fundingTime"]) + 1
        time.sleep(0.1)
    if not rows:
        return have
    new = pd.DataFrame(rows)
    new["funding_time"] = new["fundingTime"].astype("int64")
    new["funding_rate"] = new["fundingRate"].astype(float)
    out = (pd.concat([have, new[["funding_time", "funding_rate"]]], ignore_index=True)
           .drop_duplicates("funding_time").sort_values("funding_time").reset_index(drop=True))
    out.to_parquet(path_for(settings, symbol), index=False)
    return out


def backfill(settings, symbols, log=print):
    done = 0
    for sym in symbols:
        if len(backfill_symbol(settings, sym, log)):
            done += 1
    log(f"funding: {done} of {len(symbols)} symbols have a perpetual contract")
    return done


def attach(panel: pd.DataFrame, settings) -> pd.DataFrame:
    """Add funding_7d and funding_pct to a panel, without ever looking forward.

    A funding payment stamped 08:00 is known at 08:00, so the join is backward with exact matches
    allowed. funding_pct is the coin's rank among the eligible coins on that bar, so the filter
    asks "is this one of the most crowded trades available right now", not "is funding high in
    absolute terms", which drifts with the whole market.

    A coin with no perpetual contract keeps a NaN and is never filtered: the rule cannot act on
    data that does not exist, and the backtest was measured the same way.
    """
    out = []
    for sym, g in panel.groupby("symbol", sort=False):
        g = g.sort_values("bar_time")
        f = load(settings, sym)
        if f.empty:
            out.append(g.assign(funding_7d=np.nan))
            continue
        f = f.copy()
        f["bar_time"] = pd.to_datetime(f["funding_time"], unit="ms", utc=True)
        f = f.sort_values("bar_time")
        f["funding_7d"] = f["funding_rate"].rolling(WINDOW_DAYS * PER_DAY).mean()
        out.append(pd.merge_asof(g, f[["bar_time", "funding_7d"]], on="bar_time",
                                 direction="backward", allow_exact_matches=True))
    p = pd.concat(out, ignore_index=True).sort_values(["bar_time", "symbol"]).reset_index(drop=True)
    el = p["eligible"] & p["funding_7d"].notna()
    p["funding_pct"] = np.nan
    if el.any():
        p.loc[el, "funding_pct"] = p[el].groupby("bar_time")["funding_7d"].rank(pct=True)
    return p


def crowding_now(settings, panel: pd.DataFrame) -> dict:
    """What the website shows: how crowded each coin's trade is on the latest bar."""
    p = attach(panel, settings)
    last = p[p["bar_time"] == p["bar_time"].max()]
    return {r.symbol: {"funding_7d": None if pd.isna(r.funding_7d) else float(r.funding_7d),
                       "crowding_pct": None if pd.isna(r.funding_pct) else float(r.funding_pct)}
            for r in last.itertuples()}


def age_days(settings, symbol, now=None) -> float:
    """How stale this symbol's funding history is, so a silent download failure is visible."""
    f = load(settings, symbol)
    if f.empty:
        return float("inf")
    now = now or datetime.now(timezone.utc)
    last = datetime.fromtimestamp(int(f["funding_time"].iloc[-1]) / 1000, timezone.utc)
    return (now - last).total_seconds() / 86_400
