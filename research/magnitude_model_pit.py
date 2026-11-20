"""Try 4: learn HOW MUCH, on the universe that includes the dead.

    python research/magnitude_model_pit.py

WHAT THE FIRST THREE TRIES ESTABLISHED (research/predict_search.py, signal_rule.py,
signal_rule_pit.py, timing_rule.py):

  - there is a real short-horizon cross-sectional signal: AUC 0.53-0.55 for "lands in the top third
    over the next bar", in development, on the holdout, on fresh data and on the survivorship-free
    universe alike;
  - a rule built on its most confident calls made +80% on the holdout of 17 surviving coins and
    lost 70% on the real universe, because those calls were crash-buying: LUNA six times on the
    way to zero, FTT the day it failed;
  - timing BTC and ETH with the same machinery had information in development and none on the
    holdout.

THE DIAGNOSIS THIS TRY ACTS ON. The model that failed was trained to predict a FREQUENCY - top
third, yes or no - on a panel with no dead coins in it. A collapsing coin that bounces on six bars
out of ten and loses half its value on the other four is "top third" most of the time, and a
frequency label scores it as a good pick. Nothing in that setup could have taught the model
otherwise: the label throws away the size of the loss, and the training data had no examples of a
coin that did not come back.

So both are changed, and nothing else:

  universe   each day's top 30 by the volume known that morning, 383 coins over the panel's life,
             93 of them later dead. The model TRAINS on the collapses once they are in the past.
  target     the 24-hour forward return minus the average of that day's tradable coins - a
             magnitude, clipped to the training fold's own 1st and 99th percentiles so that one
             bad series cannot own the loss function. Clip limits come from training rows only.
  horizon    24 hours, the holding period the earlier analysis pointed to, so that the thing
             predicted is the thing traded.

Returns are computed within runs of consecutive bars, positions are marked on full price history,
and a position open when trading stops is closed at the last price that traded.

THE PROTOCOL, fixed before the numbers exist:

  1. Two models (gradient boosting, ridge) x four rules = eight candidates, scored on DEVELOPMENT
     (before 2025) only:
        daily     the 3 highest-scored coins at each day's first bar, held 24h
        top 5% / 2.5% / 1%   coins whose score clears a trailing 180-day quantile, held 24h
  2. ONE is chosen by code: the highest quarterly z of spread net of 0.27% a round trip, at least
     150 trades. With eight candidates the best of pure noise is expected near z = 2.0.
  3. That one candidate, ONCE, on the holdout (2025 to the end of the panel). It passes only with
     a positive net spread and a quarterly z above 2.

THE HOLDOUT HAS BEEN USED ONCE BEFORE on this universe, by the rule that failed. This is its
second use. A pass here would be worth less than a first-use pass and would still need forward
data; a failure needs no such caveat.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from app import ml  # noqa: E402

import dots_features as dots  # noqa: E402
import predict_search as ps  # noqa: E402
import signal_rule as sr  # noqa: E402
import signal_rule_pit as srp  # noqa: E402
from train_predict_grid import make_model  # noqa: E402

K = 6                                             # 24 hours, label and holding period alike
CACHE_P = ROOT / "reports" / "_pit_members_dots.parquet"
CACHE_R = ROOT / "reports" / "_pit_r1_full.parquet"


def load():
    """The tradable pit rows with relationship features, built once and cached."""
    if CACHE_P.exists() and CACHE_R.exists():
        p = pd.read_parquet(CACHE_P)
        r1_full = pd.read_parquet(CACHE_R)
    else:
        print("building the survivorship-free panel (first run only, a few minutes)...")
        p, r1_full, _ = srp.build()
        p.to_parquet(CACHE_P)
        r1_full.to_parquet(CACHE_R)
    btc = p.loc[p["symbol"] == "BTCUSDT"].set_index("bar_time")["fwd_1"]
    return p, r1_full, btc


def walk_reg(X, y, t, kind, t_to, train_days=730, test_days=91, threads=12):
    """Walk-forward regression. A fresh model per fold; the clip limits are the fold's own."""
    from threadpoolctl import threadpool_limits
    purge = (K + 1) * ps.BAR_NS
    window = train_days * ps.DAY_NS
    ok = ~np.isnan(y)
    cut = int(t[ok].min()) + window + purge
    idx_all, pred_all = [], []
    with threadpool_limits(limits=threads):
        while cut < t_to:
            end = cut + test_days * ps.DAY_NS
            tr = ok & (t >= cut - purge - window) & (t < cut - purge)
            te = (t >= cut) & (t < end)
            cut = end
            if tr.sum() < 3000 or te.sum() < 150:
                continue
            lo, hi = np.quantile(y[tr], [0.01, 0.99])       # training rows only
            model = make_model(kind, "ic")
            model.fit(X[tr], np.clip(y[tr], lo, hi))
            idx_all.append(np.flatnonzero(te))
            pred_all.append(model.predict(X[te]))
            del model
    return np.concatenate(idx_all), np.concatenate(pred_all)


def study(d, mask, lo, hi):
    """Every row in `mask` within [lo, hi) as one 24-hour trade."""
    s = d[mask & (d["bar_time"] >= lo) & (d["bar_time"] < hi)].dropna(subset=[f"fwd_{K}", f"x_{K}"])
    if len(s) < 5:
        return None
    net = s[f"x_{K}"] - ps.COST
    by_q = net.groupby(s["bar_time"].dt.tz_localize(None).dt.to_period("Q")).mean()
    z = by_q.mean() / (by_q.std(ddof=1) / np.sqrt(len(by_q))) if len(by_q) > 2 else float("nan")
    nabs = s[f"fwd_{K}"] - ps.COST
    return {"trades": int(len(s)), "coins": int(s["symbol"].nunique()),
            "spread": float(s[f"x_{K}"].mean()), "net_spread": float(net.mean()),
            "net_abs": float(nabs.mean()), "median": float(nabs.median()),
            "clipped": float(nabs.clip(nabs.quantile(0.01), nabs.quantile(0.99)).mean()),
            "win": float((nabs > 0).mean()), "worst": float(nabs.min()),
            "quarters": int(len(by_q)), "z": float(z)}


def frame(p, idx, pred):
    d = p.iloc[idx][["bar_time", "symbol", "dead"] + [f"fwd_{k}" for k in (1, 2, 3, 6, 12)]].copy()
    d["pred"] = pred
    for k in (1, 2, 3, 6, 12):
        d[f"x_{k}"] = d[f"fwd_{k}"] - d.groupby("bar_time")[f"fwd_{k}"].transform("mean")
    d = d.sort_values(["bar_time", "symbol"]).reset_index(drop=True)
    d = d.join(sr.causal_thresholds(d), on="bar_time")
    # the daily rule as a threshold too, so one simulator serves every rule: the score of the
    # third-ranked coin at each day's first bar, and unreachable on every other bar
    rank = d.groupby("bar_time")["pred"].rank(ascending=False, method="first")
    third = d["pred"].where(rank == 3).groupby(d["bar_time"]).transform("max")
    d["th_daily"] = np.where(d["bar_time"].dt.hour == 0, third, np.inf)
    return d


RULES = {"daily top 3": "daily", "top 5% score": 0.95, "top 2.5% score": 0.975,
         "top 1% score": 0.99}


def main():
    p, r1_full, btc_r1 = load()
    print(f"tradable rows {len(p):,}   coins {p.symbol.nunique()}   later dead "
          f"{p.loc[p['dead'], 'symbol'].nunique()}   {p.bar_time.min().date()} to "
          f"{p.bar_time.max().date()}")
    X = dots.design_dots(p, ml.design).to_numpy(dtype=float)
    t = p["bar_time"].astype("int64").to_numpy()
    y = (p[f"fwd_{K}"] - p.groupby("bar_time")[f"fwd_{K}"].transform("mean")).to_numpy(dtype=float)
    t_to = int(t.max()) + 1

    frames = {}
    for kind in ("gbm", "linear"):
        print(f"walk-forward, {kind}: predicting the 24h return against the day's average coin...")
        idx, pred = walk_reg(X, y, t, kind, t_to)
        frames[kind] = frame(p, idx, pred)

    first = min(f["bar_time"].min() for f in frames.values())
    last = max(f["bar_time"].max() for f in frames.values())
    dev_hi = ps.DEV_END - pd.Timedelta(hours=4 * 13)
    hold_hi = min(ps.HOLD_END, last) - pd.Timedelta(hours=4 * 13)

    # ---- does the model rank coins at all, on this universe ---------------------------------------
    print("\n" + "=" * 104)
    print("DOES IT RANK COINS? Spearman correlation between score and realised 24h spread, per quarter.")
    print()
    for kind, d in frames.items():
        for label, lo, hi in (("development", first, dev_hi), ("holdout", ps.DEV_END, hold_hi)):
            w = d[(d["bar_time"] >= lo) & (d["bar_time"] < hi)].dropna(subset=[f"x_{K}"])
            ic = w.groupby(w["bar_time"].dt.tz_localize(None).dt.to_period("Q")).apply(
                lambda g: g["pred"].corr(g[f"x_{K}"], method="spearman"))
            z = ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))
            print(f"  {kind:7s} {label:12s} IC {ic.mean():+.4f} over {len(ic)} quarters, z {z:5.2f}, "
                  f"positive in {int((ic > 0).sum())} of {len(ic)}")

    # ---- 1. eight candidates on development --------------------------------------------------------
    print("\n" + "=" * 104)
    print("1. DEVELOPMENT (before 2025). Eight candidates, every signal one 24h trade, net of 0.27%.")
    print()
    print(f"  {'model':7s} {'rule':15s} {'trades':>7s} {'coins':>6s} {'spread':>8s} {'net spread':>11s} "
          f"{'net abs':>8s} {'median':>8s} {'clipped':>8s} {'win':>6s} {'worst':>8s} {'z':>6s}")
    cands = []
    for kind, d in frames.items():
        for name, q in RULES.items():
            e = study(d, d["pred"] >= d[f"th_{q}"], first, dev_hi)
            if e is None:
                continue
            cands.append((kind, name, q, e))
            print(f"  {kind:7s} {name:15s} {e['trades']:7d} {e['coins']:6d} {e['spread']:+8.3%} "
                  f"{e['net_spread']:+11.3%} {e['net_abs']:+8.3%} {e['median']:+8.3%} "
                  f"{e['clipped']:+8.3%} {e['win']:6.1%} {e['worst']:+8.1%} {e['z']:6.2f}")
    ok = [c for c in cands if c[3]["trades"] >= 150 and c[3]["z"] == c[3]["z"]]
    kind, name, q, best = max(ok, key=lambda c: c[3]["z"])
    luck = float(np.sqrt(2 * np.log(len(cands))))
    print(f"\n  CHOSEN BY THE RULE: {kind} / {name}  (development net spread {best['net_spread']:+.3%}, "
          f"z {best['z']:.2f}; best-of-{len(cands)} noise is expected near z = {luck:.2f})")
    d = frames[kind]
    mask = d["pred"] >= d[f"th_{q}"]

    # ---- 2. the chosen candidate, once, on the holdout ----------------------------------------------
    print("\n" + "=" * 104)
    print("2. THE CHOSEN CANDIDATE, ONCE, ON THE HOLDOUT (2025 to the end of the panel).")
    print()
    eh = study(d, mask, ps.DEV_END, hold_hi)
    print(f"  {'period':12s} {'trades':>7s} {'coins':>6s} {'spread':>8s} {'net spread':>11s} "
          f"{'net abs':>8s} {'median':>8s} {'clipped':>8s} {'win':>6s} {'worst':>8s} {'z':>6s}")
    for label, e in (("development", best), ("holdout", eh)):
        if e is None:
            print(f"  {label:12s} fewer than 5 trades")
            continue
        print(f"  {label:12s} {e['trades']:7d} {e['coins']:6d} {e['spread']:+8.3%} "
              f"{e['net_spread']:+11.3%} {e['net_abs']:+8.3%} {e['median']:+8.3%} "
              f"{e['clipped']:+8.3%} {e['win']:6.1%} {e['worst']:+8.1%} {e['z']:6.2f}")
    print()
    print("  as a portfolio (five equal slots, marked on full price history, costs on every trade):")
    sr.show_sim("development", sr.simulate(d, q, K, first, dev_hi, btc_r1=btc_r1, r1_full=r1_full))
    sr.show_sim("holdout", sr.simulate(d, q, K, ps.DEV_END, hold_hi, btc_r1=btc_r1, r1_full=r1_full))

    # ---- did the change of target and universe fix the crash-buying -----------------------------------
    print("\n" + "=" * 104)
    print("DID IT STOP BUYING COLLAPSES? The chosen rule's trades, by what became of the coin.")
    print()
    survivors = {f.name.rsplit("_", 1)[0] for f in (ROOT / "data" / "klines_plus").glob("*_4h.parquet")}
    sig = d[mask & (d["bar_time"] < hold_hi)].dropna(subset=[f"fwd_{K}"]).copy()
    sig["net"] = sig[f"fwd_{K}"] - ps.COST
    sig["netx"] = sig[f"x_{K}"] - ps.COST
    sig["class"] = np.where(sig["dead"], "later delisted",
                            np.where(sig["symbol"].isin(survivors), "one of the 17 survivors",
                                     "alive, not in the 17"))
    print(f"  {'what became of the coin':26s} {'trades':>7s} {'coins':>6s} {'net spread':>11s} "
          f"{'net abs':>9s} {'win':>6s} {'worst':>9s}")
    for cls, sub in sig.groupby("class"):
        print(f"  {cls:26s} {len(sub):7d} {sub.symbol.nunique():6d} {sub['netx'].mean():+11.3%} "
              f"{sub['net'].mean():+9.3%} {(sub['net'] > 0).mean():6.1%} {sub['net'].min():+9.1%}")
    print("\n  the eight worst trades:")
    for row in sig.nsmallest(8, "net")[["bar_time", "symbol", "class", "net"]].itertuples():
        print(f"    {str(row.bar_time)[:16]}  {row.symbol:13s} {row._3:26s} {row.net:+8.2%}")
    luna = sig[sig["symbol"].isin(["LUNAUSDT", "USTUSDT", "FTTUSDT"])]
    print(f"\n  trades in LUNA, UST or FTT: {len(luna)}, averaging "
          f"{luna['net'].mean() if len(luna) else float('nan'):+.2%} "
          "(the frequency model's ten worst trades were LUNA six times and UST once)")

    # ---- verdict ----------------------------------------------------------------------------------------
    print("\n" + "=" * 104)
    print("VERDICT, by the criteria at the top of this file.")
    dev_ok = best["net_spread"] > 0 and best["z"] > luck
    hold_ok = eh is not None and eh["net_spread"] > 0 and eh["z"] > 2
    print(f"  development: net spread {best['net_spread']:+.3%}, z {best['z']:.2f} against a luck bar "
          f"of {luck:.2f}  ->  {'pass' if dev_ok else 'FAIL'}")
    print(f"  holdout    : net spread {('%+.3f%%' % (eh['net_spread'] * 100)) if eh else 'n/a'}, "
          f"z {('%.2f' % eh['z']) if eh else 'n/a'}  ->  {'pass' if hold_ok else 'FAIL'}")
    print()
    if dev_ok and hold_ok:
        print("  It passes on the universe with the dead coins in it. This is the holdout's second")
        print("  use, so it is a candidate for forward testing, not a result.")
    else:
        print("  It does not pay its costs on the real universe. Changing what the model learns and")
        print("  what it learns from did not turn the short-horizon signal into money.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
