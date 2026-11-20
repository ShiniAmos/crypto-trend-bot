"""Turn the 4-hour signal into a rule that pays its own costs - or show that it cannot.

    python research/signal_rule.py

WHERE THIS STARTS. predict_search.py found a real predictor: which coins land in the top third of
the cross-section over the next 4-hour bar (AUC 0.533 / 0.529 / 0.544 in development, holdout and
fresh data). As a top-3-every-bar rule it earns +0.05% to +0.07% a bar against a 0.27% round trip,
so the pre-registered verdict was "not worth trading". signal_anatomy.py then showed, on
development data only, the two properties a tradable version would have to lean on:

    the edge is concentrated   the most confident 1% of predictions earn +0.37% a bar, the top
                               pick +0.115%, anything ranked 4th or lower nothing
    the edge accumulates       top-3 picks: +0.07% after 4h, +0.16% after 24h, +0.21% after 48h

THE RULE FAMILY. Buy a coin when its score is in the top q of the scores seen over the previous
180 days, and hold it for K bars. Two knobs, twelve combinations:

    q   the most confident 5%, 2.5% or 1%
    K   hold for 1, 3, 6 or 12 bars (4h to 48h)

The threshold is CAUSAL: at bar t it is a quantile of predictions made strictly before t, so the
rule never knows what "high" will turn out to mean.

THE PROTOCOL, fixed before the numbers exist:

  1. Score all twelve on DEVELOPMENT only (test windows before 2025).
  2. Choose ONE: the highest z of net-of-cost spread across calendar quarters, with at least 150
     trades. Chosen by code, not by eye.
  3. Run that one rule ONCE on the holdout (2025-01 to 2026-09) and ONCE on fresh data
     (2026-09-24 onward), as a portfolio with five equal slots and 0.135% charged per side.
  4. It works only if the holdout AND the fresh data both show a positive net spread per trade,
     and the holdout's quarterly z is above 2. Otherwise the signal is real and untradeable at
     these costs, which is also an answer.

The model itself has no setting chosen here: gradient boosting, the 57 relationship features, a
rolling 730-day window, a fresh fit per fold - exactly as in predict_search.py.
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
import signal_anatomy as sa  # noqa: E402

QS = (0.95, 0.975, 0.99)
KS = (1, 3, 6, 12)
LOOKBACK_BARS = 180 * 6
SLOTS = 5
SIDE = ps.COST / 2
BARS_PER_YEAR = 6 * 365


def predictions():
    """One continuous walk-forward across the whole timeline. Every prediction is out of sample:
    the model that made it was fitted on the 730 days before its test window, minus the purge."""
    p = sa.build()
    Xdf = dots.design_dots(p, ml.design)
    X = Xdf.to_numpy(dtype=float)
    y = p["t_top_third"].to_numpy(dtype=float)
    t = p["bar_time"].astype("int64").to_numpy()
    t_end = int(t.max()) + 1 + 2 * ps.BAR_NS
    r = ps.walk(X, y, t, "auc", "gbm", sa.H, 0, t_end + 2 * ps.BAR_NS, keep_pred=True, threads=12)
    d = p.iloc[r["idx"]][["bar_time", "symbol"] + [f"fwd_{k}" for k in (1, 2, 3, 6, 12)]].copy()
    d["pred"] = r["pred"]
    for k in (1, 2, 3, 6, 12):
        d[f"x_{k}"] = d[f"fwd_{k}"] - d.groupby("bar_time")[f"fwd_{k}"].transform("mean")
    d = d.sort_values(["bar_time", "symbol"]).reset_index(drop=True)
    # BTC has no label for a beats-BTC target, so it is absent from `d`. Its own bar returns are
    # carried separately for the benchmark - the first version of this script read BTC from `d`,
    # found nothing, and silently printed the all-coin figure twice under two names.
    d.attrs["btc_r1"] = p.loc[p["symbol"] == "BTCUSDT"].set_index("bar_time")["fwd_1"]
    return d


def causal_thresholds(d):
    """For every bar, the q-quantile of the predictions made over the previous 180 days."""
    by_bar = d.groupby("bar_time")["pred"].apply(np.asarray).sort_index()
    bars = by_bar.index
    out = {q: np.full(len(bars), np.nan) for q in QS}
    window, size = [], 0
    for i in range(len(bars)):
        if size >= 3000:                          # at least ~200 bars of history before trading
            pool = np.concatenate(window)
            for q in QS:
                out[q][i] = np.quantile(pool, q)
        window.append(by_bar.iloc[i])             # appended AFTER use: strictly-before semantics
        size += len(by_bar.iloc[i])
        if len(window) > LOOKBACK_BARS:
            size -= len(window.pop(0))
    th = pd.DataFrame(out, index=bars)
    th.columns = [f"th_{q}" for q in QS]
    return th


def event_study(d, q, k, lo, hi):
    """Every signal in [lo, hi) as one trade: spread and absolute return over K bars, net of costs."""
    m = (d["bar_time"] >= lo) & (d["bar_time"] < hi) & (d["pred"] >= d[f"th_{q}"])
    s = d.loc[m].dropna(subset=[f"fwd_{k}"])
    if len(s) < 5:
        return None
    quarter = s["bar_time"].dt.tz_localize(None).dt.to_period("Q")
    by_q = (s[f"x_{k}"] - ps.COST).groupby(quarter).mean()
    z = by_q.mean() / (by_q.std(ddof=1) / np.sqrt(len(by_q))) if len(by_q) > 2 else float("nan")
    years = max((hi - lo).days / 365.25, 1e-9)
    return {"q": q, "k": k, "trades": int(len(s)), "per_year": len(s) / years,
            "spread": float(s[f"x_{k}"].mean()), "net_spread": float(s[f"x_{k}"].mean() - ps.COST),
            "abs_ret": float(s[f"fwd_{k}"].mean()), "net_abs": float(s[f"fwd_{k}"].mean() - ps.COST),
            "win": float((s[f"fwd_{k}"] - ps.COST > 0).mean()), "quarters": int(len(by_q)),
            "z": float(z)}


def simulate(d, q, k, lo, hi, btc_r1=None, r1_full=None):
    """Five equal slots, enter at the next open, hold K bars, 0.135% per side.

    A coin that fires again while held has its clock reset at no cost, which is what a trader
    would do and is the main way this rule keeps turnover down.
    """
    w = d[(d["bar_time"] >= lo) & (d["bar_time"] < hi)]
    r1 = w.pivot(index="bar_time", columns="symbol", values="fwd_1")
    # Returns for a HELD coin must come from its full history, not only from the bars on which it
    # was in the tradable universe: a coin that collapses and drops out of the universe mid-hold
    # would otherwise show no return at all for exactly the bars that hurt.
    held_r = r1 if r1_full is None else r1_full.reindex(r1.index)
    fire = w[w["pred"] >= w[f"th_{q}"]].sort_values("pred", ascending=False)
    fire_by_bar = fire.groupby("bar_time")["symbol"].apply(list).to_dict()
    held, equity, curve, n_trades, exposure = {}, 1.0, [], 0, []
    for bar in r1.index:
        cost = 0.0
        for sym in [s for s, left in held.items() if left <= 0]:
            del held[sym]
            cost += SIDE / SLOTS
        for sym in fire_by_bar.get(bar, []):
            if sym in held:
                held[sym] = k
            elif len(held) < SLOTS:
                held[sym] = k
                cost += SIDE / SLOTS
                n_trades += 1
        row = held_r.loc[bar]
        gain = sum(row.get(sym, 0.0) for sym in held if row.get(sym, np.nan) == row.get(sym, np.nan))
        equity *= (1.0 + gain / SLOTS - cost)
        curve.append(equity)
        exposure.append(len(held) / SLOTS)
        for sym in held:
            held[sym] -= 1
    eq = pd.Series(curve, index=r1.index)
    if len(eq) < 10:
        return None
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    ret = eq.pct_change().dropna()
    bench = (1 + r1.mean(axis=1).fillna(0)).cumprod()
    if btc_r1 is None:
        btc_r1 = d.attrs.get("btc_r1")
    btc = ((1 + btc_r1.reindex(r1.index).fillna(0)).cumprod() if btc_r1 is not None
           else pd.Series(np.nan, index=r1.index))
    return {"total": float(eq.iloc[-1] - 1), "cagr": float(eq.iloc[-1] ** (1 / years) - 1),
            "maxdd": float((eq / eq.cummax() - 1).min()),
            "sharpe": float(ret.mean() / ret.std() * np.sqrt(BARS_PER_YEAR)) if ret.std() > 0 else 0.0,
            "exposure": float(np.mean(exposure)), "trades": n_trades, "years": years,
            "all_coins": float(bench.iloc[-1] - 1), "btc": float(btc.iloc[-1] - 1),
            "all_coins_dd": float((bench / bench.cummax() - 1).min())}


def show_sim(label, s):
    if s is None:
        print(f"  {label}: too few bars")
        return
    print(f"  {label}")
    print(f"      rule       total {s['total']:+8.2%}   CAGR {s['cagr']:+8.2%}   maxDD {s['maxdd']:+7.2%}   "
          f"Sharpe {s['sharpe']:5.2f}   invested {s['exposure']:5.1%}   {s['trades']} trades "
          f"over {s['years']:.2f}y")
    print(f"      all coins  total {s['all_coins']:+8.2%}   maxDD {s['all_coins_dd']:+7.2%}      "
          f"BTC total {s['btc']:+8.2%}")


def main():
    print("walk-forward predictions over the whole timeline (every one out of sample)...")
    d = predictions()
    btc_r1 = d.attrs.get("btc_r1")
    d = d.join(causal_thresholds(d), on="bar_time")
    d.attrs["btc_r1"] = btc_r1
    first, last = d["bar_time"].min(), d["bar_time"].max()
    dev_hi = ps.DEV_END - pd.Timedelta(hours=4 * 13)        # no development label reaches 2025
    hold_hi = ps.HOLD_END - pd.Timedelta(hours=4 * 13)
    print(f"{len(d):,} predictions, {first.date()} to {last.date()}")

    # ---- 1. development: all twelve ------------------------------------------------------------
    print("\n" + "=" * 100)
    print("1. DEVELOPMENT (before 2025). All twelve rules, every signal counted as one trade.")
    print()
    print(f"  {'confidence':>10s} {'hold':>5s} {'trades':>7s} {'/year':>6s} {'spread':>9s} "
          f"{'net spread':>11s} {'abs return':>11s} {'net abs':>9s} {'win':>6s} {'z (quarters)':>13s}")
    rows = []
    for q in QS:
        for k in KS:
            e = event_study(d, q, k, first, dev_hi)
            if e:
                rows.append(e)
                print(f"  {'top ' + format(1 - q, '.1%'):>10s} {k * 4:4d}h {e['trades']:7d} "
                      f"{e['per_year']:6.0f} {e['spread']:+9.3%} {e['net_spread']:+11.3%} "
                      f"{e['abs_ret']:+11.3%} {e['net_abs']:+9.3%} {e['win']:6.1%} {e['z']:13.2f}")
    ok = [r for r in rows if r["trades"] >= 150 and r["z"] == r["z"]]
    if not ok:
        print("\n  no rule has 150 development trades; nothing to choose")
        return 0
    best = max(ok, key=lambda r: r["z"])
    q, k = best["q"], best["k"]
    print(f"\n  CHOSEN BY THE RULE (highest quarterly z of net spread, >= 150 trades): "
          f"top {1 - q:.1%} confidence, hold {k * 4}h")
    print(f"  development: net spread {best['net_spread']:+.3%} per trade, z {best['z']:.2f}, "
          f"{best['per_year']:.0f} trades a year")
    n_pos = sum(1 for r in rows if r["net_spread"] > 0)
    print(f"  ({n_pos} of {len(rows)} rules have a positive net spread in development)")

    # ---- 2. the one chosen rule, once on each untouched period ------------------------------------
    print("\n" + "=" * 100)
    print("2. THE CHOSEN RULE, ONCE, ON DATA IT WAS NOT CHOSEN ON.")
    print()
    eh = event_study(d, q, k, ps.DEV_END, hold_hi)
    ep = event_study(d, q, k, ps.HOLD_END, last + pd.Timedelta(hours=4))
    print(f"  {'period':22s} {'trades':>7s} {'spread':>9s} {'net spread':>11s} {'abs return':>11s} "
          f"{'net abs':>9s} {'win':>6s} {'z':>6s}")
    for label, e in (("development", best), ("holdout 2025-26", eh), ("fresh data", ep)):
        if e is None:
            print(f"  {label:22s}  fewer than 5 trades")
            continue
        print(f"  {label:22s} {e['trades']:7d} {e['spread']:+9.3%} {e['net_spread']:+11.3%} "
              f"{e['abs_ret']:+11.3%} {e['net_abs']:+9.3%} {e['win']:6.1%} {e['z']:6.2f}")

    print()
    print("  as a portfolio (five equal slots, costs charged on every entry and exit):")
    show_sim("development", simulate(d, q, k, first, dev_hi))
    show_sim("holdout 2025-01 -> 2026-09", simulate(d, q, k, ps.DEV_END, hold_hi))
    show_sim("fresh data 2026-09-24 ->", simulate(d, q, k, ps.HOLD_END, last + pd.Timedelta(hours=4)))

    # ---- 3. verdict ----------------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("3. VERDICT, by the criteria written at the top of this file before any of it ran.")
    print()
    h_ok = eh is not None and eh["net_spread"] > 0 and eh["z"] > 2
    p_ok = ep is not None and ep["net_spread"] > 0
    print(f"  holdout: net spread {'%+.3f%%' % (eh['net_spread'] * 100) if eh else 'n/a'}, "
          f"z {'%.2f' % eh['z'] if eh else 'n/a'}  ->  {'pass' if h_ok else 'FAIL'}")
    print(f"  fresh  : net spread {'%+.3f%%' % (ep['net_spread'] * 100) if ep else 'n/a'}"
          f"  ->  {'pass' if p_ok else 'FAIL'}")
    print()
    if h_ok and p_ok:
        print("  The rule pays its costs on data it was not chosen on. It is a candidate, and the")
        print("  next gate is the one every change to this bot faces: backtest.run under the ship")
        print("  rule, the survivorship-free panel, and a placebo null.")
    else:
        print("  The signal is real and this rule does not turn it into money after costs.")
        print("  What the numbers above say about why is the finding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
