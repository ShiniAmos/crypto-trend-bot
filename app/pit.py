"""Survivorship-free re-test: the universe as it was at each point in time.

The normal backtest uses *today's* top coins over all years, which flatters the past (those coins
are the survivors). Here every USDT pair Binance has ever listed is considered, including ones
later delisted, and on each day the universe is that day's top N by trailing 30-day USDT volume,
computed only from days already closed. The same filters as the live universe apply.
"""

import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from . import backfill, data, features
from .universe import MEMES, STABLES, WRAPPED

ARCHIVE = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
FIAT = {"EUR", "GBP", "AUD", "TRY", "BRL", "RUB", "UAH", "NGN", "ZAR", "BIDR", "IDRT", "BKRW",
        "BVND", "JPY", "MXN", "ARS", "COP", "PLN", "RON", "CZK", "BUSD", "USDSB", "USDS", "SUSD"}
LEVERAGED = re.compile(r"(UP|DOWN|BULL|BEAR)$")


def all_usdt_symbols():
    """Every USDT spot pair in Binance's public archive (listed or delisted)."""
    out, marker = [], ""
    while True:
        r = requests.get(ARCHIVE, params={"delimiter": "/", "prefix": "data/spot/monthly/klines/",
                                          "marker": marker}, timeout=30)
        r.raise_for_status()
        out += re.findall(r"<Prefix>data/spot/monthly/klines/([A-Z0-9]+)/</Prefix>", r.text)
        nxt = re.search(r"<NextMarker>([^<]+)</NextMarker>", r.text)
        if not nxt:
            break
        marker = nxt.group(1)
    return sorted(s for s in set(out) if s.endswith("USDT"))


def keep(symbol, exclude_memes=True):
    base = symbol[:-4]
    return not (base in STABLES or base in FIAT or base in WRAPPED or LEVERAGED.search(base)
                or (exclude_memes and base in MEMES) or base.startswith("USD"))


def fetch(settings, symbols, tf, log=print, workers=8):
    kdir = settings.data_dir / "pit" / "klines"
    kdir.mkdir(parents=True, exist_ok=True)

    def one(sym):
        try:
            backfill.backfill_symbol(kdir, sym, tf, log=lambda *_: None)
        except Exception as e:  # a few archive names never traded vs USDT on the API
            return f"{sym}: {e}"
    with ThreadPoolExecutor(workers) as ex:
        errors = [e for e in ex.map(one, symbols) if e]
    log(f"  {tf}: {len(symbols)} symbols, {len(errors)} unavailable")
    return kdir


def membership(kdir, symbols, top_n):
    """date x symbol -> True if in that day's top N by trailing 30-day volume (known that morning)."""
    vols = {}
    for s in symbols:
        d = data.load(kdir, s, "1d")
        if len(d):
            vols[s] = d.set_index(pd.to_datetime(d["open_time"], unit="ms", utc=True))["quote_volume"]
    v = pd.DataFrame(vols).sort_index()
    trailing = v.rolling(30, min_periods=30).sum().shift(1)   # only days already closed
    rank = trailing.rank(axis=1, ascending=False)
    return rank <= top_n


def run(settings, top_n=30, log=print, flags=None):
    from . import backtest
    log("Listing every USDT pair Binance has ever had...")
    syms = [s for s in all_usdt_symbols() if keep(s, settings.exclude_memes)]
    log(f"  {len(syms)} pairs after removing stablecoins, fiat, leveraged, wrapped and meme coins")
    kdir = fetch(settings, syms, "1d", log)
    member = membership(kdir, syms, top_n)
    ever = sorted(set(member.columns[member.any()]) | {"BTCUSDT"})
    log(f"  {len(ever)} coins were in the top {top_n} at some point; fetching their 4h history...")
    fetch(settings, ever, "4h", log)

    panel = features.build_panel(kdir, ever, settings.donchian_bars)
    panel["day"] = (panel["bar_time"] - pd.Timedelta(milliseconds=1)).dt.normalize()
    m = member.rename_axis(index="day", columns="symbol").stack().rename("member").reset_index()
    panel = panel.merge(m, on=["day", "symbol"], how="left")
    panel["member"] = panel["member"].astype("boolean").fillna(False).astype(bool)
    panel["eligible"] = panel["eligible"] & (panel["member"] | (panel["symbol"] == "BTCUSDT"))
    panel = features.score(panel.drop(columns=["score", "rank", "n_eligible"]))
    res = backtest.run(panel, settings, flags=flags)
    return res, panel, member


def equal_weight_universe(panel, index):
    """Benchmark: hold every coin in that day's universe, equal weights (no fees)."""
    p = panel[panel["eligible"]]
    r = p.groupby("bar_time")["ret_4h"].mean().reindex(index).fillna(0)
    return (1 + r).cumprod()
