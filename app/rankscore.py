"""Shadow scorer: rank every coin in the universe each 4h bar, record it, change nothing.

    python -m app.rankscore            # score every closed 4h bar not yet scored
    python -m app.rankscore --status   # what has been recorded

WHAT THIS IS FOR. RESEARCH_LOG.md P22 found a model that ranks coins by their next-24-hour return
against the average coin, correctly in 29 of 29 quarters out of sample - with all of its skill at
the bottom of the ranking. As a veto on the breakout entries the bot already takes (skip a coin in
the model's bottom third) it improved the survivorship-free backtest and beat twelve random vetoes,
and it FAILED the ship rule on the survivor panel. So it does not trade. It earns forward data.

This process produces that data. It never touches a book, an order, a signal or a setting of the
trading bot: it runs as its own service, reads Binance's public market data into its own store,
and writes to its own database. The fleet does not import it. If it crashes, nothing else notices.

EACH RUN
  1. the universe: every TRADING USDT spot pair, ranked by trailing 30-day volume over days already
     closed, minus tokenized stocks and pegged tokens (app.rankuniverse) - the top 30 coins
  2. features for those coins from a short tail of candles (app.rankfeatures, the exact code the
     model was trained with; tests/test_rank_shadow.py proves tail and full history agree)
  3. the model's prediction for each coin and its percentile within the bar (app.rankmodel,
     numpy only)
  4. one row per (bar, coin) in shadow.sqlite, and one row per run saying what happened

Bars missed while the server was down are scored on the next run, up to MAX_BACKFILL of them.

WHAT IS RECORDED is enough to answer both forward questions without trusting this process's own
arithmetic: research/shadow_report.py recomputes every outcome from candles, and joins the scores
to the trades the bot actually made.
"""

import argparse
import json
import sqlite3
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, data, pit, rankfeatures as rf, rankmodel, rankuniverse as ru, universe

TOP_N = 30
MAX_BACKFILL = 12            # bars: two days of downtime is recovered, more is reported as a gap
DAILY_KEEP = rf.TAIL_1D + 10
H4_KEEP = rf.TAIL_4H + MAX_BACKFILL + 10
VETO_PCT = 1 / 3             # the P22 rule: a percentile at or below this would have been vetoed
WORKERS = 4
BAR = pd.Timedelta(hours=4)
ROOT = Path(__file__).resolve().parent.parent


def paths(s):
    base = Path(s.data_dir) / "rank"
    return {"base": base, "klines": base / "klines", "db": base / "shadow.sqlite",
            "listed": base / "listed.json", "model": ROOT / "models" / "rank_model.json"}


def connect(db):
    con = sqlite3.connect(db)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS scores (
            bar_time TEXT NOT NULL, symbol TEXT NOT NULL, pred REAL NOT NULL, pct REAL NOT NULL,
            would_veto INTEGER NOT NULL, close REAL, model_id TEXT NOT NULL, scored_at TEXT NOT NULL,
            PRIMARY KEY (bar_time, symbol));
        CREATE TABLE IF NOT EXISTS runs (
            started_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL, model_id TEXT,
            bars_scored INTEGER, rows_written INTEGER, members TEXT, message TEXT);
    """)
    return con


def _fetch_tail(kdir, symbol, tf, keep, now_ms):
    """Bring one symbol's stored tail up to the last closed bar and trim it to `keep` rows."""
    have = data.load(kdir, symbol, tf)
    step = data.TF_MS[tf]
    if len(have):
        start = int(have["open_time"].iloc[-1]) + step
    else:
        start = now_ms - (keep + 2) * step
    pages = []
    while start < now_ms - step:
        page = data.fetch_klines(symbol, tf, start_ms=start)
        if page.empty:
            break
        pages.append(page)
        start = int(page["open_time"].iloc[-1]) + step
        if len(page) < 1000:
            break
    if not pages:
        return have
    df = pd.concat(([have] if len(have) else []) + pages, ignore_index=True)
    df = df.drop_duplicates("open_time", keep="last").sort_values("open_time").tail(keep)
    return data.save(kdir, symbol, tf, df)


def _listing_dates(store, symbols, log):
    """First trading day for symbols that could be tokenized stocks. Fetched once, kept forever."""
    known = json.loads(store.read_text()) if store.exists() else {}
    need = [x for x in symbols if ru.is_stock_name(x) and x not in known]
    for x in need:
        try:
            ms = data.first_candle_ms(x, "1d")
            if ms is not None:
                known[x] = int(ms)
        except Exception as e:                                       # noqa: BLE001
            log(f"  listing date for {x} unavailable: {e}")
    if need:
        store.write_text(json.dumps(known, indent=0))
    return {k: pd.Timestamp(v, unit="ms", tz="UTC") for k, v in known.items()}


def score_once(s, log=print, now=None):
    """One run. Returns a dict describing what was done; raises only on programming errors."""
    P = paths(s)
    P["klines"].mkdir(parents=True, exist_ok=True)
    model = rankmodel.load(P["model"])
    con = connect(P["db"])
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    now_ms = int((now or time.time()) * 1000)

    # ---- the universe ----------------------------------------------------------------------
    pairs = universe.binance_usdt_pairs()
    symbols = sorted(i["symbol"] for i in pairs.values() if pit.keep(i["symbol"], s.exclude_memes))
    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(lambda x: _fetch_tail(P["klines"], x, "1d", DAILY_KEEP, now_ms), symbols))
    listed = _listing_dates(P["listed"], symbols, log)
    vol, close, _ = ru.daily_frames(P["klines"], symbols, data.load)

    btc = _fetch_tail(P["klines"], "BTCUSDT", "4h", H4_KEEP, now_ms)
    latest = pd.Timestamp(int(btc["close_time"].max()) + 1, unit="ms", tz="UTC")
    member = ru.membership(vol, close, TOP_N, listed, extend_to=latest)

    last_scored = con.execute("SELECT MAX(bar_time) FROM scores").fetchone()[0]
    first = latest - (MAX_BACKFILL - 1) * BAR
    if last_scored:
        first = max(first, pd.Timestamp(last_scored) + BAR)
    targets = pd.date_range(first, latest, freq="4h")
    if len(targets) == 0:
        con.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?)",
                    (started, datetime.now(timezone.utc).isoformat(timespec="seconds"), "nothing new",
                     model["model_id"], 0, 0, None, f"latest bar {latest} already scored"))
        con.commit()
        return {"status": "nothing new", "latest": str(latest)}
    gap = None
    if last_scored and pd.Timestamp(last_scored) + BAR < first:
        gap = f"bars after {last_scored} and before {first} were missed and are not recoverable"

    days = sorted({(t - pd.Timedelta(milliseconds=1)).normalize() for t in targets})
    needed = sorted(set().union(*[set(member.columns[member.loc[d]]) for d in days if d in member.index]))

    # ---- features from tails, with the training code ----------------------------------------
    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(lambda x: _fetch_tail(P["klines"], x, "4h", H4_KEEP, now_ms), needed))
    full = rf.build(P["klines"], needed, s.donchian_bars)
    p = rf.attach_members(full, member)
    p = p[p["bar_time"].isin(targets)].copy()
    if p.empty:
        raise RuntimeError(f"no member rows at {targets[0]} .. {targets[-1]}")
    p = rf.add_market_state(p).sort_values(["bar_time", "symbol"]).reset_index(drop=True)
    X = rf.design(p, model["features"])
    p["pred"] = rankmodel.predict(model, X)
    p["pct"] = p.groupby("bar_time")["pred"].rank(pct=True)
    p["would_veto"] = (p["pct"] <= VETO_PCT).astype(int)

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = [(str(r.bar_time), r.symbol, float(r.pred), float(r.pct), int(r.would_veto),
             float(r.close), model["model_id"], stamp) for r in p.itertuples()]
    con.executemany("INSERT OR REPLACE INTO scores VALUES (?,?,?,?,?,?,?,?)", rows)
    members_now = sorted(member.columns[member.loc[days[-1]]]) if days[-1] in member.index else []
    msg = gap or ""
    con.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?)",
                (started, stamp, "ok", model["model_id"], int(p["bar_time"].nunique()), len(rows),
                 ",".join(members_now), msg))
    con.commit()
    return {"status": "ok", "bars": int(p["bar_time"].nunique()), "rows": len(rows),
            "latest": str(latest), "members": len(members_now), "gap": gap}


def status(s):
    P = paths(s)
    if not P["db"].exists():
        print("no shadow database yet")
        return
    con = connect(P["db"])
    n, bars, lo, hi = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT bar_time), MIN(bar_time), MAX(bar_time) FROM scores").fetchone()
    print(f"{n} scores over {bars} bars, {lo} to {hi}")
    for r in con.execute("SELECT started_at, status, bars_scored, rows_written, message FROM runs "
                         "ORDER BY started_at DESC LIMIT 8"):
        print("  ", r)
    print("latest bar, lowest-ranked third (would be vetoed):")
    for r in con.execute("SELECT symbol, ROUND(pct, 3), ROUND(pred * 100, 3) FROM scores WHERE "
                         "bar_time = (SELECT MAX(bar_time) FROM scores) ORDER BY pct LIMIT 10"):
        print("   ", r)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args(argv)
    s = config.load()
    if args.status:
        status(s)
        return 0
    try:
        out = score_once(s)
        print(json.dumps(out))
        return 0
    except Exception as e:                                           # noqa: BLE001
        # A failed run is recorded and reported, and never retried in a loop: the next timer tick
        # is the retry. Nothing here can affect trading, so failing loudly is the safe choice.
        P = paths(s)
        try:
            con = connect(P["db"])
            con.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?)",
                        (datetime.now(timezone.utc).isoformat(timespec="seconds"), None, "error",
                         None, 0, 0, None, f"{type(e).__name__}: {e}"[:2000]))
            con.commit()
        except Exception:                                            # noqa: BLE001
            pass
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
