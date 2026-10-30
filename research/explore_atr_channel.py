"""Volatility-scaled entry and exit: an ATR band instead of the fixed 20-bar Donchian channel.

HYPOTHESIS
    A fixed 20-bar high/low channel is the wrong shape. The same N bars mean very different things
    in a quiet tape and a wild one. A band anchored on the N-bar mean close and widened by k * ATR
    adapts: in quiet markets a smaller move clears the band, in wild markets the band is wider and
    only a real move counts.

WHAT IS CHANGED
    Per symbol, with everything known strictly before the bar it judges:
        ma    = close.rolling(N).mean().shift(1)        # mean of bars t-N .. t-1
        atr   = (atr_pct * close).shift(1)              # 14-bar mean true range as of t-1
        upper = ma + k * atr
        lower = ma - k * atr
        breakout  = close_t > upper
        breakdown = close_t < lower
        stop_level = lower                              # the sizing denominator
    That is exactly the shipped shape with `don_high`/`don_low` swapped for the band, so the
    shift(1) convention matches `features.symbol_features` byte for byte.

WHAT IS NOT CHANGED
    The engine. Costs (7bps slip + 10bps fee per side), next-open fills, the score, the rank gate,
    the regime gate, rs_7d > 0, the three shipped filters (abs_momentum, volume_min 0.8,
    taker_min 0.5), the 1%-risk sizing formula, the 30% cap, 3 names. Only the three panel columns
    above are overwritten, on a copy of the panel.

    python research/explore_atr_channel.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import backtest, config, features, signals  # noqa: E402
from app.flags import Flags  # noqa: E402

SHIPPED = dict(abs_momentum=True, volume_min=0.8, taker_min=0.5)
HELD_OUT = "2022-01-01"
KS = (1.0, 1.5, 2.0)
NS = (20, 40)


# ------------------------------------------------------------------ the band

def atr_band(panel, k, n):
    """(upper, lower) as Series aligned to `panel`'s index. Nothing at bar t uses bar t."""
    p = panel.sort_values(["symbol", "bar_time"])
    by = p.groupby("symbol", sort=False)
    ma = by["close"].transform(lambda x: x.rolling(n).mean().shift(1))
    # atr_pct is the 14-bar mean true range divided by that bar's close, so atr_pct*close is the
    # range in price units. shift(1) makes it an as-of-t-1 quantity.
    atr = (p["atr_pct"] * p["close"]).groupby(p["symbol"], sort=False).shift(1)
    upper = (ma + k * atr).reindex(panel.index)
    lower = (ma - k * atr).reindex(panel.index)
    return upper, lower


def band_panel(panel, k, n, stop_from_band=True):
    """A copy of the panel whose breakout / breakdown / stop_level express the ATR band."""
    upper, lower = atr_band(panel, k, n)
    p2 = panel.copy()
    p2["breakout"] = (panel["close"] > upper).fillna(False)
    p2["breakdown"] = (panel["close"] < lower).fillna(False)
    if stop_from_band:
        p2["stop_level"] = lower
    p2["_upper"], p2["_lower"] = upper, lower
    return p2


# ------------------------------------------------------------------ guards

def entry_mask(p, s, flags):
    """The engine's own entry predicate, rebuilt here so the guards test the rows that matter."""
    btc = p["symbol"] == "BTCUSDT"
    ok = (p["eligible"] & p["regime_on"] & p["breakout"]
          & ((p["rs_7d"] > 0) | btc) & (p["rank"] <= s.candidates))
    ok &= signals.extra_filters(p, flags)
    return signals.add_btc_sleeve(p, ok, flags).fillna(False)


def assert_sane(p, s, flags, label):
    """stop_level must sit strictly below the price sizing uses, or every size is zero."""
    m = entry_mask(p, s, flags)
    n_entry = int(m.sum())
    if n_entry == 0:
        raise AssertionError(f"{label}: no entry candidates at all")
    bad = m & ~(p["stop_level"] < p["close"])
    if bad.any():
        raise AssertionError(f"{label}: stop_level >= close on {int(bad.sum())}/{n_entry} "
                             f"entry candidate rows - sizing would return zero")
    dist = ((p.loc[m, "close"] - p.loc[m, "stop_level"]) / p.loc[m, "close"])
    return {"entry_rows": n_entry, "median_stop_dist_pct": float(dist.median()),
            "p90_stop_dist_pct": float(dist.quantile(0.9))}


def lookahead_audit(panel, k, n, sym="ETHUSDT", probes=400, seed=11):
    """Recompute the band at random bars from a slice that ENDS at t-1 and demand it matches."""
    one = panel[panel["symbol"] == sym].sort_values("bar_time").reset_index(drop=True)
    upper, lower = atr_band(panel, k, n)
    ref_u = upper.reindex(panel[panel["symbol"] == sym].sort_values("bar_time").index).to_numpy()
    c = one["close"].to_numpy()
    atr_price = (one["atr_pct"] * one["close"]).to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.arange(n + 2, len(one)), size=min(probes, len(one) - n - 2), replace=False)
    worst = 0.0
    for t in idx:
        past_only = c[t - n:t]                   # bars t-n .. t-1, bar t untouched
        want = past_only.mean() + k * atr_price[t - 1]
        got = ref_u[t]
        if np.isnan(want) or np.isnan(got):
            continue
        worst = max(worst, abs(want - got) / max(abs(want), 1e-12))
    return {"symbol": sym, "probes": int(len(idx)), "worst_rel_err": float(worst)}


# ------------------------------------------------------------------ measuring

def measure(panel, s, flags, start=None, with_p=False):
    res = backtest.run(panel, s, flags=flags, start=start)
    m = backtest.metrics(res.equity, res.trades, res.exposure)
    out = {"cagr": float(m["cagr"]), "sharpe": float(m["sharpe"]),
           "maxdd": float(m["max_drawdown"]), "trades": int(m.get("trades", 0)),
           "win_rate": float(m.get("win_rate", float("nan"))),
           "avg_trade": float(m.get("avg_trade", float("nan"))),
           "avg_bars_held": float(m.get("avg_bars_held", float("nan"))),
           "avg_exposure": float(m.get("avg_exposure", float("nan"))),
           "total_return": float(m["total_return"])}
    t = res.trades
    out["best_trade"] = float(t["ret"].max()) if len(t) else float("nan")
    out["top1_share_of_pnl"] = (float(t["pnl"].max() / t["pnl"].clip(lower=0).sum())
                               if len(t) and t["pnl"].clip(lower=0).sum() > 0 else float("nan"))
    if with_p:
        rb = backtest.random_baselines(panel, res.trades, s) or {}
        out["p_coin"] = float(rb.get("p_value_vs_random_coin", float("nan")))
        out["p_time"] = float(rb.get("p_value_vs_random_time", float("nan")))
    return out, res


def ship(base, cand):
    """The non-negotiable rule: no metric may get worse. Strict, no epsilon."""
    return {"cagr": cand["cagr"] >= base["cagr"],
            "sharpe": cand["sharpe"] >= base["sharpe"],
            "maxdd": cand["maxdd"] >= base["maxdd"]}


def fmt(name, m):
    return (f"  {name:<34} CAGR {m['cagr']:+.4f}  Sharpe {m['sharpe']:.4f}  "
            f"maxDD {m['maxdd']:.4f}  trades {m['trades']:>4}  "
            f"expo {m['avg_exposure']:.2f}  bars {m['avg_bars_held']:.0f}")


# ------------------------------------------------------------------ main

def main():
    s = config.load()
    u = json.loads((ROOT / "reports" / "_parity_universe.json").read_text())
    print(f"Universe: {len(u['symbols'])} coins from reports/_parity_universe.json")
    panel = features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars)
    print(f"Panel: {len(panel):,} rows, {panel['bar_time'].min().date()} .. "
          f"{panel['bar_time'].max().date()}")
    flags = Flags(**SHIPPED)

    print("\n=== BASELINE (shipped Donchian 20) ===")
    base_std, base_res = measure(panel, s, flags, with_p=True)
    base_held, _ = measure(panel, s, flags, start=HELD_OUT)
    print(fmt("standard panel", base_std) + f"  p_coin {base_std['p_coin']:.3f}")
    print(fmt("held-out 2022+", base_held))
    base_guard = assert_sane(panel, s, flags, "baseline")
    print(f"  baseline stop distance: median {base_guard['median_stop_dist_pct']:.2%} "
          f"p90 {base_guard['p90_stop_dist_pct']:.2%}")

    print("\n=== ATR BAND GRID (band replaces the channel for entry, exit AND the stop) ===")
    rows = []
    for n in NS:
        for k in KS:
            label = f"ATR band k={k} N={n}"
            p2 = band_panel(panel, k, n)
            guard = assert_sane(p2, s, flags, label)
            std, _ = measure(p2, s, flags)
            held, _ = measure(p2, s, flags, start=HELD_OUT)
            c_std, c_held = ship(base_std, std), ship(base_held, held)
            row = {"variant": label, "k": k, "n": n, "mode": "band entry + band exit + band stop",
                   "standard": std, "held_out": held,
                   "checks_standard": c_std, "checks_held_out": c_held,
                   "passes_ship_rule": all(c_std.values()) and all(c_held.values()),
                   "guard": guard}
            rows.append(row)
            mark = "PASS" if row["passes_ship_rule"] else "fail"
            print(fmt(label, std))
            print(fmt("   held-out", held) + f"   -> {mark}"
                  f"  [std {''.join(kk[0].upper() if v else kk[0] for kk, v in c_std.items())}]")
            print(f"       stop distance median {guard['median_stop_dist_pct']:.2%}, "
                  f"entry-candidate rows {guard['entry_rows']:,}")

    # --- diagnostics. The mandated grid changes three things at once (trigger, exit, stop).
    # These split it apart, and then scan the ONE split that came close across the whole grid so a
    # near-pass cannot be a single lucky cell. Everything here is exploratory: it spends extra
    # degrees of freedom on the same history, so a win must be read as a hypothesis, not a result.
    print("\n=== DIAGNOSTIC A: pure sizing effect (shipped trigger, band as the stop) ===")
    diags = []
    for n in NS:
        for k in KS:
            _, lower = atr_band(panel, k, n)
            p_size = panel.copy()
            # min() keeps the stop below the close wherever the band would sit above it
            p_size["stop_level"] = np.minimum(panel["stop_level"], lower)
            label = f"shipped trigger + band stop k={k} N={n}"
            assert_sane(p_size, s, flags, label)
            e_std, _ = measure(p_size, s, flags)
            e_held, _ = measure(p_size, s, flags, start=HELD_OUT)
            c_s, c_h = ship(base_std, e_std), ship(base_held, e_held)
            diags.append({"variant": label, "k": k, "n": n, "family": "sizing only",
                          "standard": e_std, "held_out": e_held,
                          "checks_standard": c_s, "checks_held_out": c_h,
                          "passes_ship_rule": all(c_s.values()) and all(c_h.values())})
            print(fmt(f"k={k} N={n}", e_std))

    print("\n=== DIAGNOSTIC B: band trigger only, shipped Donchian stop ===")
    for n in NS:
        for k in KS:
            label = f"band trigger k={k} N={n} + shipped stop"
            p_trig = band_panel(panel, k, n, stop_from_band=False)
            assert_sane(p_trig, s, flags, label)
            d_std, _ = measure(p_trig, s, flags)
            d_held, _ = measure(p_trig, s, flags, start=HELD_OUT)
            c_s, c_h = ship(base_std, d_std), ship(base_held, d_held)
            diags.append({"variant": label, "k": k, "n": n, "family": "trigger only",
                          "standard": d_std, "held_out": d_held,
                          "checks_standard": c_s, "checks_held_out": c_h,
                          "passes_ship_rule": all(c_s.values()) and all(c_h.values())})
            print(fmt(f"k={k} N={n}", d_std))
            print(fmt("   held-out", d_held))

    print("\n=== DIAGNOSTIC C: Donchian AND band must agree (either one exits) ===")
    for n in NS:
        for k in KS:
            label = f"Donchian AND band k={k} N={n}"
            p_both = band_panel(panel, k, n, stop_from_band=False)
            p_both["breakout"] = p_both["breakout"] & panel["breakout"]
            p_both["breakdown"] = p_both["breakdown"] | panel["breakdown"]
            assert_sane(p_both, s, flags, label)
            f_std, _ = measure(p_both, s, flags)
            f_held, _ = measure(p_both, s, flags, start=HELD_OUT)
            c_s, c_h = ship(base_std, f_std), ship(base_held, f_held)
            diags.append({"variant": label, "k": k, "n": n, "family": "trigger AND",
                          "standard": f_std, "held_out": f_held,
                          "checks_standard": c_s, "checks_held_out": c_h,
                          "passes_ship_rule": all(c_s.values()) and all(c_h.values())})
            print(fmt(f"k={k} N={n}", f_std))
            print(fmt("   held-out", f_held) +
                  f"   std {c_s} held {c_h}")

    d_pass = [d for d in diags if d["passes_ship_rule"]]
    print(f"\nDiagnostic variants clearing the ship rule on BOTH panels: "
          f"{[d['variant'] for d in d_pass] or 'none'}")

    # --- the winner on the ship rule, else the least-bad by Sharpe on the held-out half
    passing = [r for r in rows if r["passes_ship_rule"]]
    best = (max(passing, key=lambda r: r["held_out"]["sharpe"]) if passing
            else max(rows, key=lambda r: min(r["standard"]["sharpe"], r["held_out"]["sharpe"])))
    print(f"\n=== VERDICT: {'PASS' if passing else 'FAIL'} ===")
    print(f"Best variant: {best['variant']}  (ship rule "
          f"{'PASSED' if best['passes_ship_rule'] else 'FAILED'})")
    for panel_name, key, bref in (("standard", "checks_standard", base_std),
                                  ("held-out", "checks_held_out", base_held)):
        got = best["standard"] if panel_name == "standard" else best["held_out"]
        print(f"  {panel_name:<9} CAGR {got['cagr']:+.4f} vs {bref['cagr']:+.4f} | "
              f"Sharpe {got['sharpe']:.4f} vs {bref['sharpe']:.4f} | "
              f"maxDD {got['maxdd']:.4f} vs {bref['maxdd']:.4f} | {best[key]}")

    # p-value only for the best, it is the expensive one
    pb = band_panel(panel, best["k"], best["n"])
    best_std_p, _ = measure(pb, s, flags, with_p=True)
    best["standard"]["p_coin"] = best_std_p["p_coin"]
    best["standard"]["p_time"] = best_std_p["p_time"]
    print(f"  best variant p_coin {best_std_p['p_coin']:.3f} (baseline {base_std['p_coin']:.3f}), "
          f"p_time {best_std_p['p_time']:.3f}")

    # --- is any winner carried by one year or one trade?
    print("\n=== YEAR BY YEAR (does a win depend on one lucky year?) ===")
    years = {}
    _, r_base = measure(panel, s, flags)
    years["baseline"] = {str(y): float(v) for y, v in backtest.yearly(r_base.equity).items()}
    # the closest cell gets a row even though it fails, because that is the lead worth reading
    p_near0 = band_panel(panel, 1.0, 20, stop_from_band=False)
    p_near0["breakout"] = p_near0["breakout"] & panel["breakout"]
    p_near0["breakdown"] = p_near0["breakdown"] | panel["breakdown"]
    _, r_near = measure(p_near0, s, flags)
    years["Donchian AND band k=1.0 N=20"] = {str(y): float(v)
                                             for y, v in backtest.yearly(r_near.equity).items()}
    for d in d_pass:
        p_both = band_panel(panel, d["k"], d["n"], stop_from_band=False)
        if d["family"] == "trigger AND":
            p_both["breakout"] = p_both["breakout"] & panel["breakout"]
            p_both["breakdown"] = p_both["breakdown"] | panel["breakdown"]
        elif d["family"] == "sizing only":
            _, lower = atr_band(panel, d["k"], d["n"])
            p_both = panel.copy()
            p_both["stop_level"] = np.minimum(panel["stop_level"], lower)
        _, rr = measure(p_both, s, flags)
        years[d["variant"]] = {str(y): float(v) for y, v in backtest.yearly(rr.equity).items()}
    _, r_best = measure(band_panel(panel, best["k"], best["n"]), s, flags)
    years[best["variant"]] = {str(y): float(v) for y, v in backtest.yearly(r_best.equity).items()}
    allyears = sorted({y for v in years.values() for y in v})
    print(f"  {'variant':<34}" + "".join(f"{y:>9}" for y in allyears))
    for name, v in years.items():
        print(f"  {name[:34]:<34}" + "".join(f"{v.get(y, float('nan')):>+9.1%}" for y in allyears))
    for name in years:
        src = best if name == best["variant"] else next(
            (d for d in d_pass if d["variant"] == name), None)
        if src:
            print(f"  {name[:50]}: biggest winner is "
                  f"{src['standard']['top1_share_of_pnl']:.0%} of all gross profit")

    # --- split stability. The single near-miss cell (Donchian AND band, k=1.0 N=20) clears every
    # test on the 2022+ half and misses CAGR by a hair on the full panel. If that is real it should
    # survive moving the split; if it is noise it will not.
    print("\n=== SPLIT STABILITY of the closest cell (Donchian AND band k=1.0 N=20) ===")
    p_near = band_panel(panel, 1.0, 20, stop_from_band=False)
    p_near["breakout"] = p_near["breakout"] & panel["breakout"]
    p_near["breakdown"] = p_near["breakdown"] | panel["breakdown"]
    splits = {}
    for cut in (None, "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01"):
        b, _ = measure(panel, s, flags, start=cut)
        c, _ = measure(p_near, s, flags, start=cut)
        chk = ship(b, c)
        splits[cut or "full"] = {"baseline": b, "candidate": c, "checks": chk,
                                 "passes": all(chk.values())}
        print(f"  from {str(cut or 'inception'):<12} "
              f"CAGR {c['cagr']:+.4f} vs {b['cagr']:+.4f} | "
              f"Sharpe {c['sharpe']:.3f} vs {b['sharpe']:.3f} | "
              f"maxDD {c['maxdd']:.4f} vs {b['maxdd']:.4f} | "
              f"{'PASS' if all(chk.values()) else 'fail ' + ','.join(k for k, v in chk.items() if not v)}")
    n_pass = sum(v["passes"] for v in splits.values())
    failed = [k for k, v in splits.items() if not v["passes"]]
    print(f"  clears the ship rule on {n_pass}/{len(splits)} splits; fails on {failed or 'none'}")
    # Neighbour sensitivity: a real edge should be a plateau, not a spike. Compare the cell with
    # its two grid neighbours in the SAME family.
    nb = {d["variant"]: d for d in diags if d["family"] == "trigger AND"}
    print("  neighbour sensitivity in the same family (a spike is a warning, a plateau is not):")
    for key in ("Donchian AND band k=1.0 N=20", "Donchian AND band k=1.5 N=20",
                "Donchian AND band k=1.0 N=40"):
        d = nb[key]
        print(f"    {key:<34} std Sharpe {d['standard']['sharpe']:.3f}  "
              f"held Sharpe {d['held_out']['sharpe']:.3f}  std CAGR {d['standard']['cagr']:+.4f}")

    # --- is the ATR doing the work, or is it just a moving-average confirmation?
    # k=1.0 N=20 sits in the CORNER of the mandated grid, so "spike or plateau edge" is
    # undecidable from the grid alone. Walking k down to 0 answers a sharper question: at k=0 the
    # band condition degenerates to "close > mean(close, N)", a plain trend filter with no
    # volatility scaling in it at all. If the numbers keep improving as k shrinks, the ATR term is
    # not the active ingredient and the hypothesis is wrong even where the equity curve is better.
    print("\n=== IS THE ATR THE ACTIVE INGREDIENT? (AND family, k walked down to 0) ===")
    gradient = []
    for n in (10, 20):
        for k in (0.0, 0.25, 0.5, 0.75, 1.0):
            pg = band_panel(panel, k, n, stop_from_band=False)
            pg["breakout"] = pg["breakout"] & panel["breakout"]
            pg["breakdown"] = pg["breakdown"] | panel["breakdown"]
            # at k=0 upper==lower==ma, so the band degenerates to a moving-average filter
            assert_sane(pg, s, flags, f"AND k={k} N={n}")
            g_std, _ = measure(pg, s, flags)
            g_held, _ = measure(pg, s, flags, start=HELD_OUT)
            c_s, c_h = ship(base_std, g_std), ship(base_held, g_held)
            gradient.append({"k": k, "n": n, "standard": g_std, "held_out": g_held,
                             "checks_standard": c_s, "checks_held_out": c_h,
                             "passes_ship_rule": all(c_s.values()) and all(c_h.values())})
            print(f"  k={k:<5} N={n:<3} std CAGR {g_std['cagr']:+.4f} Sharpe {g_std['sharpe']:.3f} "
                  f"DD {g_std['maxdd']:.4f} | held CAGR {g_held['cagr']:+.4f} "
                  f"Sharpe {g_held['sharpe']:.3f} DD {g_held['maxdd']:.4f} | trades {g_std['trades']:>4}"
                  f" | {'PASS' if all(c_s.values()) and all(c_h.values()) else 'fail'}")
    g_pass = [g for g in gradient if g["passes_ship_rule"]]
    print(f"  gradient cells clearing the ship rule: {len(g_pass)}/{len(gradient)}")

    audit = [lookahead_audit(panel, best["k"], best["n"], sym)
             for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT")]
    print("\nLookahead audit (band rebuilt from bars <= t-1 only):")
    for a in audit:
        print(f"  {a['symbol']}: {a['probes']} probes, worst relative error {a['worst_rel_err']:.2e}")
    assert max(a["worst_rel_err"] for a in audit) < 1e-9, "band does not match a past-only rebuild"

    out = {"hypothesis": "an ATR-scaled band adapts to volatility better than a fixed 20-bar "
                         "Donchian channel, for entry, exit and the sizing stop",
           "universe": u["symbols"], "shipped_flags": SHIPPED, "held_out_start": HELD_OUT,
           "grid": {"k": list(KS), "N": list(NS)},
           "baseline": {"standard": base_std, "held_out": base_held, "guard": base_guard},
           "variants": rows, "diagnostics": diags,
           "diagnostics_passing": [d["variant"] for d in d_pass],
           "best_variant": best["variant"], "best": best,
           "passes_ship_rule": bool(passing),
           "yearly": years,
           "atr_gradient": gradient,
           "split_stability_closest_cell": {"variant": "Donchian AND band k=1.0 N=20",
                                            "splits": splits,
                                            "passes": int(n_pass), "of": len(splits)},
           "brief_baseline_for_reference": {
               "note": "the numbers quoted in the task brief; they do NOT match this data vintage, "
                       "the live-measured baseline above does and agrees with reports/exits.json",
               "standard": {"cagr": 0.2071, "sharpe": 1.3712, "maxdd": -0.1509, "trades": 398,
                            "p_coin": 0.040},
               "held_out": {"cagr": 0.1473, "sharpe": 1.1004, "maxdd": -0.1509, "trades": 233}},
           "lookahead_audit": audit}
    path = ROOT / "reports" / "explore_atr_channel.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
