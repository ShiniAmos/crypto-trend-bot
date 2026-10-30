"""Phase 2: the authenticated top-30 universe, rebuilt daily (section 3).

A coin is kept only if it is in CoinGecko's top CG_TOP by market cap, has an official Binance
spot COINUSDT pair with status TRADING, trades above the volume floor, and is not a stablecoin,
leveraged token, wrapped/staked stand-in or (by default) a meme coin.
"""

import json
import re
from datetime import datetime, timezone

import numpy as np
import requests

from . import data

STABLES = {"USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDD", "USDE", "PYUSD", "BUSD", "USDP",
           "USD1", "USDS", "GUSD", "FRAX", "LUSD", "EURC", "EURI", "AEUR", "RLUSD", "USTC",
           "XAUT", "PAXG", "USDX", "SUSD", "USDF", "USDG", "USDTB", "BFUSD"}
WRAPPED = {"WBTC", "WETH", "WBETH", "STETH", "WSTETH", "WEETH", "CBBTC", "CBETH", "RETH",
           "BETH", "BNSOL", "JITOSOL", "MSOL", "LSETH", "EZETH", "RSETH", "SOLVBTC", "LBTC",
           "TBTC", "WTRX", "WBNB", "HYPE_WRAPPED", "SUSDE", "SUSDS", "BSC-USD", "FBTC"}
MEMES = {"DOGE", "SHIB", "PEPE", "BONK", "WIF", "FLOKI", "TRUMP", "PENGU", "FARTCOIN",
         "BRETT", "POPCAT", "MEME", "BOME", "NEIRO", "TURBO", "PNUT", "MOG", "SPX"}
LEVERAGED = re.compile(r"^[A-Z]{3,}(UP|DOWN|BULL|BEAR)$|[0-9][LS]$")

CG_URL = "https://api.coingecko.com/api/v3/coins/markets"


def coingecko_top(n):
    r = requests.get(CG_URL, params={"vs_currency": "usd", "order": "market_cap_desc",
                                     "per_page": n, "page": 1}, timeout=30)
    r.raise_for_status()
    return r.json()


def binance_usdt_pairs():
    """base asset -> symbol info, for SPOT USDT pairs that are TRADING."""
    out = {}
    for s in data.exchange_info()["symbols"]:
        if s["quoteAsset"] == "USDT" and s["status"] == "TRADING" and s.get("isSpotTradingAllowed"):
            out[s["baseAsset"]] = s
    return out


def is_stable(coin) -> bool:
    sym = coin["symbol"].upper()
    if sym in STABLES:
        return True
    price = coin.get("current_price") or 0
    return "usd" in coin.get("id", "") and 0.97 < price < 1.03


def volume_history(settings, observations=None):
    """Rolling record of each symbol's observed 24h volume, one entry per rebuild.

    The file is the memory that lets the list judge a coin on its habit rather than on the single
    reading it happened to take this morning.
    """
    path = settings.data_dir / "universe_volume.json"
    hist = json.loads(path.read_text()) if path.exists() else {}
    if observations:
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        keep = max(settings.volume_window_days * 2, 4)
        for sym, vol in observations.items():
            rows = [r for r in hist.get(sym, []) if r[0] != stamp]
            rows.append([stamp, float(vol)])
            hist[sym] = rows[-keep:]
        path.write_text(json.dumps(hist))
    return hist


def typical_volume(hist, symbol, latest, window):
    """Median of the recent observations, so one dead day cannot eject an established coin."""
    rows = [v for _, v in hist.get(symbol, [])][-window:]
    if latest is not None:
        rows = rows + [float(latest)]
    return float(np.median(rows)) if rows else float(latest or 0.0)


def build(settings, today=None, refresh=False, held_symbols=None):
    """Return today's universe, reusing it until it is stale so two runs agree.

    v2 rules (4 Oct 2026):
      - a coin is judged on the MEDIAN of its last VOLUME_WINDOW_DAYS volume readings;
      - a coin already on the list only leaves when that median falls below
        VOLUME_EXIT_RATIO x the floor, so joining is harder than staying (hysteresis);
      - a coin currently held is never dropped for volume alone;
      - if fewer than UNIVERSE_MIN_NAMES survive, the list is topped up by volume rather than
        allowed to collapse in a market-wide drought.
    """
    today = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = settings.universe_path
    if path.exists() and not refresh:
        cached = json.loads(path.read_text())
        age_h = float("inf")
        if cached.get("built_at"):
            age_h = (datetime.now(timezone.utc)
                     - datetime.fromisoformat(cached["built_at"])).total_seconds() / 3600
        if cached.get("date") == today and age_h < settings.universe_refresh_hours:
            return cached

    try:
        pairs = binance_usdt_pairs()
        tickers = data.ticker_24h()
        top = coingecko_top(settings.cg_top)
    except Exception:
        if path.exists():  # offline or a source is down: keep trading yesterday's list
            return json.loads(path.read_text())
        raise
    previous = json.loads(path.read_text()) if path.exists() else {}
    incumbents = {c["symbol"] for c in previous.get("coins", [])}
    held = set(held_symbols or ())
    observed = {pairs[c["symbol"].upper()]["symbol"]:
                float(tickers.get(pairs[c["symbol"].upper()]["symbol"], {}).get("quoteVolume", 0))
                for c in top if c["symbol"].upper() in pairs}
    hist = volume_history(settings, observed)

    kept, dropped, benched = [], [], []
    for coin in top:
        base = coin["symbol"].upper()
        reason = None
        if is_stable(coin):
            reason = "stablecoin"
        elif base in WRAPPED or coin["id"].startswith(("wrapped-", "staked-")):
            reason = "wrapped/staked stand-in"
        elif settings.exclude_memes and base in MEMES:
            reason = "meme coin"
        elif LEVERAGED.search(base):
            reason = "leveraged token"
        elif base not in pairs:
            reason = "no TRADING Binance USDT spot pair"
        if reason:
            dropped.append({"coin": base, "cg_rank": coin["market_cap_rank"], "reason": reason})
            continue
        symbol = pairs[base]["symbol"]
        vol = float(tickers.get(symbol, {}).get("quoteVolume", 0))
        # `hist` already contains today's reading, so it is not passed again: double-weighting
        # today would defeat the point of taking a median.
        typical = typical_volume(hist, symbol, None, settings.volume_window_days)
        # Staying is easier than joining, and a coin we hold is never ejected for volume alone.
        floor = settings.min_volume_usdt
        bar = floor * settings.volume_exit_ratio if symbol in incumbents else floor
        if symbol in held:
            bar = 0.0
        if typical < bar:
            row = {"coin": base, "cg_rank": coin["market_cap_rank"], "symbol": symbol,
                   "typical_volume_usdt": round(typical),
                   "reason": (f"{settings.volume_window_days}-reading median volume "
                              f"${typical/1e6:.1f}M below "
                              f"{'exit' if symbol in incumbents else 'entry'} bar ${bar/1e6:.0f}M")}
            dropped.append(row)
            benched.append(row)          # eligible to be recalled if the list would be too short
            continue
        first = data.first_candle_ms(symbol)
        days = (data.now_ms() - first) / 86_400_000 if first else 0
        kept.append({"symbol": symbol, "base": base, "cg_rank": coin["market_cap_rank"],
                     "cg_id": coin["id"], "volume_24h_usdt": round(vol),
                     "typical_volume_usdt": round(typical),
                     "first_candle": first, "history_days": int(days),
                     "live_eligible": days >= settings.min_history_days})
        if len(kept) >= settings.universe_size:
            break

    # A market-wide drought must not collapse the list: top it up with the most traded of the
    # coins that just missed, rather than leaving the bot with nothing to choose from.
    if len(kept) < settings.universe_min_names and benched:
        for row in sorted(benched, key=lambda r: -r["typical_volume_usdt"]):
            if len(kept) >= settings.universe_min_names:
                break
            sym = row["symbol"]
            first = data.first_candle_ms(sym)
            days = (data.now_ms() - first) / 86_400_000 if first else 0
            kept.append({"symbol": sym, "base": row["coin"], "cg_rank": row["cg_rank"],
                         "cg_id": "", "volume_24h_usdt": row["typical_volume_usdt"],
                         "typical_volume_usdt": row["typical_volume_usdt"],
                         "first_candle": first, "history_days": int(days),
                         "live_eligible": days >= settings.min_history_days,
                         "recalled": True})
            dropped = [d for d in dropped if d.get("symbol") != sym]

    if not any(k["symbol"] == "BTCUSDT" for k in kept):
        raise RuntimeError("BTCUSDT failed the universe filters; refusing to continue")
    result = {"date": today, "built_at": datetime.now(timezone.utc).isoformat(),
              "symbols": [k["symbol"] for k in kept], "coins": kept, "dropped": dropped}
    path.write_text(json.dumps(result, indent=2))
    return result


def load(settings):
    if not settings.universe_path.exists():
        raise FileNotFoundError("No universe.json yet. Run: python -m app.main universe")
    return json.loads(settings.universe_path.read_text())


def tradable_status(symbol) -> bool:
    """True if Binance still lists the pair as TRADING (checked before every live action)."""
    info = data.get("/api/v3/exchangeInfo", {"symbol": symbol})
    return bool(info["symbols"]) and info["symbols"][0]["status"] == "TRADING"
