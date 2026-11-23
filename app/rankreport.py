"""What the shadow record says so far. Read-only.

    python -m app.rankreport

Two questions, answered from data/rank/shadow.sqlite and Binance's public candles:

  1. IS THE MODEL STILL RIGHT? For every scored bar at least 24 hours old, the realised 24-hour
     return of each scored coin against that bar's average scored coin - the model's own label,
     recomputed here from candles rather than trusted from the scorer. Rank correlation per bar,
     and the bottom third against the rest.
  2. WOULD THE VETO HAVE HELPED THE BOT? Every breakout entry the three books took since scoring
     began, joined to the model's percentile at the bar the order was decided on, and the trade's
     realised return. This accumulates only as fast as the bot trades - slowly - and it is the
     question that decides whether the veto ever ships.

HOW LONG IT TAKES. The 30 coins in a bar move together, and each bar's 24-hour window overlaps the
next five, so the record holds about ONE independent observation a day - not 180, as the first
version of this file claimed. Measured on the 2025-26 walk-forward, one bar a day (689 days):

    daily rank correlation       mean +0.076, day-to-day sd 0.26   ->  about 48 days to z = 2
    bottom third minus the rest  mean -0.38%, day-to-day sd 2.5%   ->  about 168 days to z = 2

So the rank correlation answers "does the model still rank coins" in about seven weeks, while the
bottom-third spread - the thing the veto actually uses - needs about five and a half months. Both
are printed with their progress on every run; significance uses the 00:00 bar of each day only.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, data

H = 6
BOOKS = ("paper", "paper100", "demo")
# What each measure looks like one bar a day when the model works: the 2025-26 walk-forward of
# research/train_rank_model.py, 23 Nov 2026. A model exported with `forward_expectation` in its
# meta overrides these.
EXPECT = {"ic_mean": 0.0759, "ic_sd": 0.2615, "spread_mean": -0.00383, "spread_sd": 0.02478}


def days_to_verdict(mean, sd, z=2.0):
    """Independent days before a measure averaging `mean` with daily sd `sd` reads z."""
    return int(np.ceil((z * sd / abs(mean)) ** 2)) if mean else None


def progress_line(label, values, mean, sd):
    n = len(values)
    need = days_to_verdict(mean, sd)
    if n < 3:
        return f"  {label}: {n} independent days so far; a verdict needs about {need}"
    m, se = float(values.mean()), float(values.std(ddof=1)) / np.sqrt(n)
    z = m / se if se else float("nan")
    return (f"  {label}: {m:+.4f} over {n} days (z {z:+.2f}). A working model averages "
            f"{mean:+.4f}; a verdict needs about {need} days - {min(n / need, 1):.0%} of the way")


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


def model_health(sc, oc, expect=EXPECT):
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
    print(f"  {d['bar_time'].nunique()} bars matured, {len(d)} coin-bars, "
          f"{d['bar_time'].min()} to {d['bar_time'].max()}")
    print(f"  24h return against the average scored coin, all bars:  bottom third "
          f"{by['bottom']:+.3%}   middle {by['middle']:+.3%}   top third {by['top']:+.3%}")
    print()
    print(progress_line("rank correlation", ic[ic.index.hour == 0],
                        expect["ic_mean"], expect["ic_sd"]))
    print(progress_line("bottom third minus the rest", spread[spread.index.hour == 0],
                        expect["spread_mean"], expect["spread_sd"]))


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
    sc = pd.read_sql("SELECT bar_time, symbol, pred, pct, would_veto, model_id FROM scores", con)
    runs = pd.read_sql("SELECT * FROM runs", con)
    con.close()
    sc["bar_time"] = pd.to_datetime(sc["bar_time"], utc=True)
    first, last = sc["bar_time"].min(), sc["bar_time"].max()
    print(f"shadow record: {len(sc)} scores over {sc['bar_time'].nunique()} bars, {first} to {last}")
    print(f"runs: {len(runs)} ({int((runs['status'] == 'ok').sum())} ok, "
          f"{int((runs['status'] == 'error').sum())} errors)")
    # Retraining on one extra day moved the rankings to a mean rank correlation of 0.83 with the
    # previous fit and flipped 15% of veto decisions (RESEARCH_LOG.md P23). A record that mixes
    # models measures an average of noisy fits, so the evaluation runs one frozen model and this
    # says loudly when that has not held.
    ids = sc.groupby("model_id")["bar_time"].agg(["min", "max", "count"])
    if len(ids) > 1:
        print(f"WARNING: {len(ids)} different models in this record - results below mix them:")
        for mid, r_ in ids.iterrows():
            print(f"  model {mid}: {int(r_['count'])} scores, {r_['min']} to {r_['max']}")
    else:
        print(f"model: {ids.index[0]} for the whole record")
    gaps = runs[runs["message"].fillna("").str.contains("missed")]
    if len(gaps):
        print(f"  {len(gaps)} run(s) reported unrecoverable missed bars")

    print("\n1. IS THE MODEL STILL RIGHT?")
    expect = dict(EXPECT)
    model_file = Path(__file__).resolve().parent.parent / "models" / "rank_model.json"
    if model_file.exists():
        import json
        meta = json.loads(model_file.read_text(encoding="utf-8")).get("meta", {})
        expect.update(meta.get("forward_expectation", {}))
    mature = sc[sc["bar_time"] <= pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=4 * H + 4)]
    oc = _outcomes(base / "outcomes", sorted(mature["symbol"].unique()), first, last)
    model_health(mature, oc, expect)

    print("\n2. WOULD THE VETO HAVE HELPED THE BOT?")
    veto_check(sc, s.data_dir, first)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
