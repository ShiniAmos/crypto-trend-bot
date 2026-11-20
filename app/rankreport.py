"""What the shadow record says so far. Read-only.

    python -m app.rankreport

Two questions, answered from data/rank/shadow.sqlite and Binance's public candles:

  1. IS THE MODEL STILL RIGHT? For every scored bar at least 24 hours old, the realised 24-hour
     return of each scored coin against that bar's average scored coin - the model's own label,
     recomputed here from candles rather than trusted from the scorer. Rank correlation per bar,
     and the bottom third against the rest. This accumulates 30 observations a bar, 180 a day,
     so it says something within weeks.
  2. WOULD THE VETO HAVE HELPED THE BOT? Every breakout entry the three books took since scoring
     began, joined to the model's percentile at the bar the order was decided on, and the trade's
     realised return. This accumulates only as fast as the bot trades - slowly - and it is the
     question that decides whether the veto ever ships.

Overlapping 24-hour windows are not independent, so significance is computed on one bar per day
(the 00:00 close), and the means on all bars.

Expectations from training (research/train_rank_model.py, corrected universe): the bottom third
returned -0.29% a day against the average coin in development, -0.24% on the holdout and -0.29% on
seven weeks of unseen data. A forward figure well inside that range is the model holding up; a
figure near zero over a few months is the model not holding up.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, data

H = 6
EXPECT = {"development": -0.0029, "holdout": -0.0024, "unseen 7 weeks": -0.0029}
BOOKS = ("paper", "paper100", "demo")


def _outcomes(kdir, symbols, first, last):
    """24h forward return per (bar_time, symbol), entered at the next bar's open."""
    kdir.mkdir(parents=True, exist_ok=True)
    out = []
    start_ms = int((first - pd.Timedelta(hours=8)).value // 10**6)
    for sym in symbols:
        have = data.load(kdir, sym, "4h")
        need_from = int(have["open_time"].iloc[-1]) + data.TF_MS["4h"] if len(have) else start_ms
        pages = []
        while True:
            page = data.fetch_klines(sym, "4h", start_ms=need_from)
            if page.empty:
                break
            pages.append(page)
            need_from = int(page["open_time"].iloc[-1]) + data.TF_MS["4h"]
            if len(page) < 1000:
                break
        if pages:
            have = data.save(kdir, sym, "4h", pd.concat(([have] if len(have) else []) + pages,
                                                         ignore_index=True))
        if have.empty:
            continue
        o = have.set_index(pd.to_datetime(have["open_time"], unit="ms", utc=True))["open"].astype(float)
        # the scored bar's close time b is the next bar's open time; exit at the open 24h later
        entry = o
        exit_ = o.reindex(o.index + pd.Timedelta(hours=4 * H)).to_numpy()
        f = pd.DataFrame({"bar_time": o.index, "symbol": sym, "fwd": exit_ / entry.to_numpy() - 1})
        out.append(f[(f["bar_time"] >= first) & (f["bar_time"] <= last)])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["bar_time", "symbol", "fwd"])


def model_health(sc, oc):
    d = sc.merge(oc, on=["bar_time", "symbol"], how="inner").dropna(subset=["fwd"])
    if d.empty:
        print("  no scored bar is 24 hours old yet")
        return
    d["x"] = d["fwd"] - d.groupby("bar_time")["fwd"].transform("mean")
    ic = d.groupby("bar_time").apply(lambda g: g["pred"].corr(g["x"], method="spearman"),
                                     include_groups=False).dropna()
    third = pd.cut(d["pct"], [0, 1 / 3, 2 / 3, 1.0001], labels=["bottom", "middle", "top"])
    by = d.groupby(third, observed=False)["x"].mean()
    spread = (d[d["pct"] <= 1 / 3].groupby("bar_time")["x"].mean()
              - d[d["pct"] > 1 / 3].groupby("bar_time")["x"].mean()).dropna()
    daily = spread[spread.index.hour == 0]
    z = (daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily)))) if len(daily) > 2 else float("nan")
    print(f"  {d['bar_time'].nunique()} bars matured, {len(d)} coin-bars, "
          f"{d['bar_time'].min()} to {d['bar_time'].max()}")
    print(f"  rank correlation per bar: mean {ic.mean():+.4f}, positive on {int((ic > 0).sum())} of "
          f"{len(ic)} bars")
    print(f"  24h return against the average scored coin:  bottom third {by['bottom']:+.3%}   "
          f"middle {by['middle']:+.3%}   top third {by['top']:+.3%}")
    print(f"  bottom third minus the rest: {spread.mean():+.3%} a day; on {len(daily)} "
          f"non-overlapping days z = {z:.2f}")
    print("  training expectation for the bottom third: " + ", ".join(
        f"{k} {v:+.2%}" for k, v in EXPECT.items()))
    if len(daily) < 30:
        print(f"  ({len(daily)} independent days is too few to conclude anything; a month of data"
              " is the minimum for this line to mean something)")


def veto_check(sc, data_dir, first):
    rows = []
    for book in BOOKS:
        db = Path(data_dir) / f"state_{book}.sqlite"
        if not db.exists():
            continue
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        f = pd.read_sql("SELECT time, symbol, side, qty, price, fee, reason FROM fills ORDER BY id", con)
        con.close()
        f["time"] = pd.to_datetime(f["time"], utc=True, format="ISO8601")
        f = f[f["time"] >= first]
        for sym, g in f.groupby("symbol"):
            g = g.reset_index(drop=True)
            for i, r in g.iterrows():
                if r["side"] != "BUY" or not str(r["reason"]).startswith("breakout"):
                    continue
                exit_ = g[(g.index > i) & (g["side"] == "SELL")].head(1)
                ret = (float(exit_["price"].iloc[0]) / float(r["price"]) - 1.0 - 0.002
                       if len(exit_) else float("nan"))
                rows.append({"book": book, "symbol": sym, "time": r["time"],
                             "bar_time": r["time"].floor("4h"), "ret": ret, "open": not len(exit_)})
    if not rows:
        print("  no breakout entries since scoring began")
        return
    e = pd.DataFrame(rows).merge(sc[["bar_time", "symbol", "pct", "would_veto"]],
                                 on=["bar_time", "symbol"], how="left")
    print(f"  {len(e)} breakout entries across the books since {first.date()}; "
          f"{int(e['pct'].notna().sum())} have a score at their decision bar")
    for label, sub in (("would have been VETOED", e[e["would_veto"] == 1]),
                       ("would have been taken", e[e["would_veto"] == 0])):
        closed = sub.dropna(subset=["ret"])
        print(f"  {label:24s} {len(sub):3d} entries, {len(closed):3d} closed, "
              f"mean return {closed['ret'].mean() if len(closed) else float('nan'):+.2%}")
    print("  (paper and paper100 take the same signals, so their entries are one sample, not two)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="")
    args = ap.parse_args(argv)
    s = config.load()
    base = Path(s.data_dir) / "rank"
    db = Path(args.db) if args.db else base / "shadow.sqlite"
    if not db.exists():
        sys.exit(f"no shadow record at {db}")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    sc = pd.read_sql("SELECT bar_time, symbol, pred, pct, would_veto FROM scores", con)
    runs = pd.read_sql("SELECT * FROM runs", con)
    con.close()
    sc["bar_time"] = pd.to_datetime(sc["bar_time"], utc=True)
    first, last = sc["bar_time"].min(), sc["bar_time"].max()
    print(f"shadow record: {len(sc)} scores over {sc['bar_time'].nunique()} bars, {first} to {last}")
    print(f"runs: {len(runs)} ({int((runs['status'] == 'ok').sum())} ok, "
          f"{int((runs['status'] == 'error').sum())} errors)")
    gaps = runs[runs["message"].fillna("").str.contains("missed")]
    if len(gaps):
        print(f"  {len(gaps)} run(s) reported unrecoverable missed bars")

    print("\n1. IS THE MODEL STILL RIGHT?")
    mature = sc[sc["bar_time"] <= pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=4 * H + 4)]
    oc = _outcomes(base / "outcomes", sorted(mature["symbol"].unique()), first, last)
    model_health(mature, oc)

    print("\n2. WOULD THE VETO HAVE HELPED THE BOT?")
    veto_check(sc, s.data_dir, first)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
