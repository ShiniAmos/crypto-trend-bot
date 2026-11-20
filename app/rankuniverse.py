"""The universe the ranking model is trained on and scored over: each day's top COINS.

The survivorship-free research universe is "the top N USDT pairs by trailing 30-day volume, as
known that morning" (app/pit.py). That rule was written when a USDT pair on Binance was a
cryptocurrency. By late 2026 it is not: on 19 Nov 2026 the top 30 by volume held 7 tokenized
equities (GOOGLB, NVDAB, SOXLB, MUB, SNDKB, SPCXB, CRCLB) and 2 dollar-pegged tokens (U, RWUSD)
next to 21 coins. A model ranking coins against "the average member" would be ranking them
against Nvidia and a stablecoin.

Binance's exchangeInfo carries no asset-class field - checked 21 Nov 2026, the only differences
between NVDABUSDT and BTCUSDT are the names and the permission groups - so both classes are
recognised from Binance's own data:

  TOKENIZED STOCK   the base asset ends in "B" AND the pair first traded on or after STOCK_LAUNCH.
                    Measured on every USDT pair in the archive: 113 have a base ending in "B".
                    The 9 listed before 2026 are all coins (BNB, ARB, CKB, DGB, TRB, PHB, VIB,
                    AMB, MOB; the newest from March 2023). All 104 listed since are equities and
                    ETFs - NVDAB, TSLAB, QQQB, SPYB, TQQQB, UVXYB ... - beginning with five on
                    11 June 2026. A future coin whose name happens to end in "B" would be wrongly
                    excluded; for a model that only scores, that costs one coin's score.

                    The first version of this rule used price behaviour instead (a stock barely
                    moves at weekends). It was checked on 90 days of 2026 data and looked perfect;
                    applied to nine years of rolling windows it flagged BNB and ARB as stocks at
                    some point. Recorded so nobody reaches for it again.

  PEGGED            the last 30 closed daily bars moved less than 0.5% a day (standard deviation
                    of log returns). The pegged tokens sit at 0.02%; the quietest real coin, TRX,
                    at 1.8%. Catches a new stablecoin without anyone adding it to a list, and a
                    coin that was pegged for a while (UST before May 2022) only while it was.

Exclusions are applied BEFORE ranking by volume, so the universe stays N coins wide instead of
shrinking as non-coins crowd in. Everything here uses days strictly before the day it applies to.

This module is the single definition: research/train_rank_model.py builds the training universe
from it and app/rankscore.py builds the live one from it.
"""

import numpy as np
import pandas as pd

STOCK_LAUNCH = pd.Timestamp("2026-06-01", tz="UTC")
PEG_WINDOW = 30
PEG_MAX_DAILY_VOL = 0.005


def is_stock_name(symbol: str) -> bool:
    base = symbol[:-4] if symbol.endswith("USDT") else symbol
    return len(base) >= 3 and base.endswith("B")


def tokenized_stocks(symbols, listed: dict) -> set:
    """Symbols that are tokenized equities: a "B" base listed on or after STOCK_LAUNCH.

    `listed` maps symbol -> first trading day (UTC Timestamp). A symbol with no known listing date
    is not called a stock: an unknown should not silently remove a coin.
    """
    out = set()
    for s in symbols:
        t = listed.get(s)
        if t is not None and is_stock_name(s) and pd.Timestamp(t) >= STOCK_LAUNCH:
            out.add(s)
    return out


def pegged(close: pd.DataFrame) -> pd.DataFrame:
    """date x symbol -> True where the previous PEG_WINDOW closed days were dollar-flat."""
    r = np.log(close.where(close > 0)).diff()
    vol = r.rolling(PEG_WINDOW, min_periods=PEG_WINDOW - 5).std()
    return (vol < PEG_MAX_DAILY_VOL).shift(1, fill_value=False).astype(bool)


def membership(volume: pd.DataFrame, close: pd.DataFrame, top_n: int, listed: dict,
               extend_to=None) -> pd.DataFrame:
    """date x symbol -> True if in that day's top N COINS by trailing 30-day quote volume.

    The volume rule is app.pit.membership's, unchanged: a 30-day sum over days already closed.
    Symbols that are tokenized stocks, or pegged on that day, are removed before the ranking.

    `extend_to` adds rows up to that day. Today's membership depends only on days already closed,
    but there is no daily bar for today until it closes, so without this the frame stops at
    yesterday and today's 4h bars would find no membership at all. History is unaffected: every
    past day already has its row.
    """
    volume = volume.sort_index()
    if extend_to is not None:
        end = pd.Timestamp(extend_to).normalize()
        if end > volume.index.max():
            idx = volume.index.append(pd.date_range(volume.index.max() + pd.Timedelta(days=1),
                                                    end, freq="D", tz=volume.index.tz))
            volume = volume.reindex(idx)
    close = close.reindex(index=volume.index, columns=volume.columns)
    trailing = volume.rolling(30, min_periods=30).sum().shift(1)
    bad = pegged(close)
    stocks = sorted(tokenized_stocks(volume.columns, listed))
    if stocks:
        bad.loc[:, stocks] = True
    trailing = trailing.where(~bad)
    return trailing.rank(axis=1, ascending=False) <= top_n


def daily_frames(kdir, symbols, loader):
    """(quote volume, close, listing dates) from stored daily klines.

    The listing date is the first stored daily bar, which is the first bar Binance has when the
    store was filled from startTime=0, as app/pit.fetch does.
    """
    vols, closes, listed = {}, {}, {}
    for s in symbols:
        d = loader(kdir, s, "1d")
        if len(d):
            idx = pd.to_datetime(d["open_time"], unit="ms", utc=True)
            vols[s] = pd.Series(d["quote_volume"].to_numpy(dtype=float), index=idx)
            closes[s] = pd.Series(d["close"].to_numpy(dtype=float), index=idx)
            listed[s] = idx.min()
    return pd.DataFrame(vols).sort_index(), pd.DataFrame(closes).sort_index(), listed
