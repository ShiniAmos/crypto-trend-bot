"""Does the confidence rule survive on coins that did NOT survive?

    python research/signal_rule_pit.py

signal_rule.py produced the first result in this project to pass on data it was not chosen on:
buy the model's most confident 1% of calls and hold 24 hours. Net of costs, +0.56% a trade in
development, +0.75% on the 2025-26 holdout (z 2.78), +1.65% on eight weeks of fresh data; as a
portfolio +80% on the holdout while the average coin made +1%.

A result that good has one obvious way of being false, and this project already owns the tool that
detects it. Those numbers come from 17 coins that are all alive today. A model that buys extreme
moments looks brilliant on coins that recovered, because the ones that did not recover - LUNA to
0.00005, FTT, UST, SRM - are not in the sample. Every "it bounced" in that panel is conditioned on
the coin still existing in November 2026.

THE TEST. Same model, same features, same rule, nothing re-chosen: top 1% confidence, hold 24h,
0.27% a round trip. But the universe is the survivorship-free one - each day's top 30 by the
trading volume known that morning, 383 coins over the life of the panel, the dead included.

Three things are done differently from the survivor panel, each so that a dead coin costs what it
would have cost:

  1. A position is marked on the coin's FULL price history, not only on the bars where it was
     still in the top 30. A coin that collapses and falls out of the universe mid-hold keeps
     losing money in this simulation, as it would in an account.
  2. If a coin's data ends while a position is open (delisting), the position is closed at the
     last price that traded. It is not dropped from the sample, which is what a NaN would do and
     which would delete exactly the worst outcomes.
  3. Cross-sectional features and the top-third label are computed among that day's members only,
     so the model is never told how a coin ranks against coins nobody could have been trading.

WHAT COUNTS. The rule was chosen on the survivor panel's development period; here it is scored on
the pit panel's development AND holdout periods. If the net spread stays positive with a holdout z
above 2, survivorship does not explain the result. If it collapses, the result was the bias.
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from app import config, ml  # noqa: E402

import dots_features as dots  # noqa: E402
import modern_era as me  # noqa: E402
import predict_search as ps  # noqa: E402
import signal_rule as sr  # noqa: E402

Q, K = 0.99, 6                     # the rule chosen on the survivor panel. Not re-chosen here.
KEEP = ["symbol", "bar_time", "open", "high", "low", "close", "quote_volume", "taker_ratio",
        "vol_7d", "vol_30d", "dist_sma50", "dist_sma200", "ret_4h", "ret_1d", "ret_7d", "ret_30d",
        "don_pos", "atr_pct", "volume_ratio", "rs_1d", "rs_7d", "rs_30d", "btc_ret_1d",
        "btc_ret_7d", "btc_ret_30d", "btc_dist_sma200", "breakout", "regime_on", "eligible",
        "member"]
MARKET = ["breadth_sma50", "breadth_up_1d", "disp_1d", "mkt_ret_1d", "mkt_ret_7d", "rel_mkt_1d"]


def build():
    s = replace(config.load(), max_names=5)
    pit = me.build_pit(s, top_n=30)[KEEP]
    print(f"  pit panel: {len(pit):,} rows, {pit.symbol.nunique()} coins, "
          f"{pit.bar_time.min().date()} to {pit.bar_time.max().date()}")

    # per-coin features on each coin's FULL history
    full = dots.add_dots(pit).sort_values(["symbol", "bar_time"]).reset_index(drop=True)
    del pit

    # Forward returns on the full history, entered at the next bar's open - computed WITHIN A RUN
    # OF CONSECUTIVE BARS, never across a gap.
    #
    # The first version shifted by rows, and a row shift does not know that trading stopped. When
    # Terra collapsed, LUNAUSDT's last bar before the stop is 13 May 2022 with a close of 0.00005,
    # and the SAME TICKER resumes on 31 May, eighteen days later, at 1.0 - a different coin. One
    # row later, a twenty-thousand-fold "gain" (measured in data/pit/klines, not remembered). A position bought into the collapse and held "24 hours" by row count was credited
    # with it: development showed +908% a trade, with the median trade at -0.27% and 100% of the
    # profit coming from the best 1% of trades. The mean was one ticker being reused.
    #
    # So: a run ends wherever the next bar is not exactly four hours later. A position still open
    # when its run ends - suspension, delisting, ticker reuse - is closed at the last price that
    # traded before the stop, which is the most a holder could have had. Only a run that reaches
    # the end of the data leaves the outcome unknown.
    g = full.groupby("symbol", sort=False)
    step = full["bar_time"] - g["bar_time"].shift(1)
    full["run"] = (step != pd.Timedelta(hours=4)).groupby(full["symbol"], sort=False).cumsum()
    gr = full.groupby(["symbol", "run"], sort=False)
    entry = gr["open"].shift(-1)
    run_last_close = gr["close"].transform("last")
    run_end = gr["bar_time"].transform("max")
    stopped = run_end < (full["bar_time"].max() - pd.Timedelta(days=2))
    for k in (1, 2, 3, 6, 12):
        exit_ = gr["open"].shift(-1 - k)
        exit_ = exit_.where(exit_.notna() | ~stopped, run_last_close)
        full[f"fwd_{k}"] = exit_ / entry - 1
    last_bar = g["bar_time"].transform("max")
    full["dead"] = last_bar < (full["bar_time"].max() - pd.Timedelta(days=30))
    # After a stop of more than a week the series is, for practical purposes, a new listing: every
    # rolling feature straddles the gap and means nothing. Those rows are not tradable for 30 days.
    gap_days = step.dt.total_seconds().div(86400)
    relist = (gap_days > 7).astype(int).groupby(full["symbol"], sort=False).cumsum()
    bars_since = full.groupby(["symbol", relist], sort=False).cumcount()
    full["fresh_relist"] = (relist > 0) & (bars_since < 180)
    n_gaps = int((gap_days > 7).sum())
    print(f"  {n_gaps} trading stops longer than a week found "
          f"({full.loc[gap_days > 7, 'symbol'].nunique()} coins); positions across them are closed "
          "at the last traded price")
    r1_full = full.pivot(index="bar_time", columns="symbol", values="fwd_1")

    # the tradable universe: that day's top 30, as known that morning
    p = full[full["member"] & ~full["fresh_relist"]].copy()
    del full
    gb = p.groupby("bar_time", sort=False)
    p["breadth_sma50"] = gb["dist_sma50"].transform(lambda x: (x > 0).mean())
    p["breadth_up_1d"] = gb["ret_1d"].transform(lambda x: (x > 0).mean())
    p["disp_1d"] = gb["ret_1d"].transform("std")
    p["mkt_ret_1d"] = gb["ret_1d"].transform("median")
    p["mkt_ret_7d"] = gb["ret_7d"].transform("median")
    p["rel_mkt_1d"] = p["ret_1d"] - p["mkt_ret_1d"]

    btc = p.loc[p["symbol"] == "BTCUSDT"].set_index("bar_time")["fwd_1"]
    p["excess"] = (p["fwd_1"] - p["bar_time"].map(btc)).where(p["symbol"] != "BTCUSDT")
    gx = p["excess"].groupby(p["bar_time"])
    pct, n = gx.rank(pct=True), gx.transform("count")
    p["t_top_third"] = np.where(n >= 6, (pct > 2 / 3).astype(float), np.nan)
    p.loc[p["excess"].isna(), "t_top_third"] = np.nan
    p = p.sort_values(["bar_time", "symbol"]).reset_index(drop=True)
    return p, r1_full, btc


def main():
    print("building the survivorship-free panel (a few minutes)...")
    p, r1_full, btc_r1 = build()
    print(f"  tradable rows: {len(p):,}   coins ever tradable: {p.symbol.nunique()}   "
          f"of which later dead: {p.loc[p['dead'], 'symbol'].nunique()}")

    X = dots.design_dots(p, ml.design).to_numpy(dtype=float)
    y = p["t_top_third"].to_numpy(dtype=float)
    t = p["bar_time"].astype("int64").to_numpy()
    fwd = p["fwd_1"].to_numpy(dtype=float)
    t_end = int(t.max()) + 1 + 4 * ps.BAR_NS
    print("walk-forward predictions (fresh model per fold, 730-day window)...")
    r = ps.walk(X, y, t, "auc", "gbm", 1, 0, t_end, keep_pred=True, threads=12, fwd=fwd)
    em, ez = ps.edge_z(r["edges"])
    print(f"  the raw signal on this panel: AUC {np.mean(r['scores']):.4f} over {len(r['scores'])} "
          f"folds (shuffled {np.mean(r['nulls']):.4f}), top-3 pick edge {em:+.3%} a bar, z {ez:.2f}")
    print("  on the survivor panel it was: AUC 0.533, pick edge +0.074% a bar, z 6.36")

    d = p.iloc[r["idx"]][["bar_time", "symbol", "dead"] + [f"fwd_{k}" for k in (1, 2, 3, 6, 12)]].copy()
    d["pred"] = r["pred"]
    for k in (1, 2, 3, 6, 12):
        d[f"x_{k}"] = d[f"fwd_{k}"] - d.groupby("bar_time")[f"fwd_{k}"].transform("mean")
    d = d.sort_values(["bar_time", "symbol"]).reset_index(drop=True)
    d = d.join(sr.causal_thresholds(d), on="bar_time")
    first, last = d["bar_time"].min(), d["bar_time"].max()
    dev_hi = ps.DEV_END - pd.Timedelta(hours=4 * 13)
    hold_hi = min(ps.HOLD_END, last) - pd.Timedelta(hours=4 * 13)

    # ---- the pre-chosen rule -----------------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"THE RULE CHOSEN ON THE SURVIVOR PANEL, UNCHANGED: top {1 - Q:.0%} confidence, hold "
          f"{K * 4}h, 0.27% a round trip")
    print()
    print(f"  {'period':26s} {'trades':>7s} {'spread':>9s} {'net spread':>11s} {'abs return':>11s} "
          f"{'net abs':>9s} {'win':>6s} {'z':>6s}")
    ed = sr.event_study(d, Q, K, first, dev_hi)
    eh = sr.event_study(d, Q, K, ps.DEV_END, hold_hi)
    for label, e in (("development (to 2024)", ed), ("holdout (2025 -> cache end)", eh)):
        if e is None:
            print(f"  {label:26s}  fewer than 5 trades")
            continue
        print(f"  {label:26s} {e['trades']:7d} {e['spread']:+9.3%} {e['net_spread']:+11.3%} "
              f"{e['abs_ret']:+11.3%} {e['net_abs']:+9.3%} {e['win']:6.1%} {e['z']:6.2f}")
    print()
    print("  the same trades without leaning on the mean (a mean is one bad series away from"
          " meaningless):")
    for label, lo, hi in (("development", first, dev_hi), ("holdout", ps.DEV_END, hold_hi)):
        w = d[(d["bar_time"] >= lo) & (d["bar_time"] < hi) & (d["pred"] >= d[f"th_{Q}"])]
        v = (w[f"fwd_{K}"] - ps.COST).dropna()
        if len(v) < 5:
            continue
        clip = v.clip(v.quantile(0.01), v.quantile(0.99)).mean()
        best = v.nlargest(max(len(v) // 100, 1)).sum()
        print(f"    {label:12s} median {v.median():+.3%}   1%-clipped mean {clip:+.3%}   "
              f"worst {v.min():+.1%}   best {v.max():+.1%}   "
              f"sum of all trades {v.sum():+.1%}, of which the best 1% are {best:+.1%}")
    print()
    print("  for comparison, the same rule on the 17 survivors:")
    print("    development                  1450   +0.827%     +0.557%     +1.245%   +0.975%  52.9%   2.41")
    print("    holdout                       582   +1.020%     +0.750%     +1.347%   +1.077%  54.0%   2.78")

    print()
    print("  as a portfolio (five equal slots, positions marked on full price history):")
    sr.show_sim("development", sr.simulate(d, Q, K, first, dev_hi, btc_r1=btc_r1, r1_full=r1_full))
    sr.show_sim("holdout", sr.simulate(d, Q, K, ps.DEV_END, hold_hi, btc_r1=btc_r1, r1_full=r1_full))

    # ---- where the survivorship would show --------------------------------------------------------
    print("\n" + "=" * 100)
    print("WHERE SURVIVORSHIP WOULD SHOW: the same trades, split by what became of the coin.")
    print()
    survivors = {f.name.rsplit("_", 1)[0]
                 for f in (ROOT / "data" / "klines_plus").glob("*_4h.parquet")}
    sig = d[(d["pred"] >= d[f"th_{Q}"]) & (d["bar_time"] < hold_hi)].dropna(subset=[f"fwd_{K}"]).copy()
    sig["net"] = sig[f"fwd_{K}"] - ps.COST
    sig["netx"] = sig[f"x_{K}"] - ps.COST
    sig["class"] = np.where(sig["dead"], "later delisted",
                            np.where(sig["symbol"].isin(survivors), "one of the 17 survivors",
                                     "alive, not in the 17"))
    print(f"  {'what became of the coin':26s} {'trades':>7s} {'coins':>6s} {'net spread':>11s} "
          f"{'net abs':>9s} {'win':>6s} {'worst trade':>12s}")
    for cls, sub in sig.groupby("class"):
        print(f"  {cls:26s} {len(sub):7d} {sub.symbol.nunique():6d} {sub['netx'].mean():+11.3%} "
              f"{sub['net'].mean():+9.3%} {(sub['net'] > 0).mean():6.1%} {sub['net'].min():+12.2%}")
    print(f"  {'ALL':26s} {len(sig):7d} {sig.symbol.nunique():6d} {sig['netx'].mean():+11.3%} "
          f"{sig['net'].mean():+9.3%} {(sig['net'] > 0).mean():6.1%} {sig['net'].min():+12.2%}")

    print()
    print("  by year (net absolute return per trade, and how many trades):")
    yr = sig.groupby(sig["bar_time"].dt.year).agg(trades=("net", "size"), net=("net", "mean"),
                                                  netx=("netx", "mean"), win=("net", lambda v: (v > 0).mean()))
    for y_, row in yr.iterrows():
        print(f"    {y_}   {int(row['trades']):5d} trades   net abs {row['net']:+7.3%}   "
              f"net spread {row['netx']:+7.3%}   win {row['win']:5.1%}")

    print()
    print("  the ten worst trades:")
    worst = sig.nsmallest(10, "net")[["bar_time", "symbol", "class", "net"]]
    for row in worst.itertuples():
        print(f"    {str(row.bar_time)[:16]}  {row.symbol:12s} {row._3:26s} {row.net:+8.2%}")

    # robustness to the average being carried by a few trades
    print()
    med = sig["net"].median()
    trimmed = sig["net"].clip(sig["net"].quantile(0.01), sig["net"].quantile(0.99)).mean()
    top_share = sig.nlargest(max(len(sig) // 100, 1), "net")["net"].sum() / sig["net"].sum()
    print(f"  median trade {med:+.3%}   mean with the top and bottom 1% clipped {trimmed:+.3%}   "
          f"share of all profit from the best 1% of trades {top_share:.0%}")

    # ---- verdict ------------------------------------------------------------------------------------
    print("\n" + "=" * 100)
    ok_d = ed is not None and ed["net_spread"] > 0
    ok_h = eh is not None and eh["net_spread"] > 0 and eh["z"] > 2
    print("VERDICT on survivorship.")
    print(f"  development net spread {'%+.3f%%' % (ed['net_spread'] * 100) if ed else 'n/a'}  ->  "
          f"{'positive' if ok_d else 'NOT positive'}")
    print(f"  holdout     net spread {'%+.3f%%' % (eh['net_spread'] * 100) if eh else 'n/a'}, "
          f"z {'%.2f' % eh['z'] if eh else 'n/a'}  ->  {'pass' if ok_h else 'FAIL'}")
    if ok_d and ok_h:
        print("\n  The rule survives the coins that did not. Survivorship is not what produced it.")
    else:
        print("\n  The rule does not survive the survivorship-free universe. What looked like a")
        print("  predictor on 17 coins was, at least in part, the fact that those 17 are still here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
