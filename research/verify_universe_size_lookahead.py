"""Adversarial audit of research/explore_universe_size.py - LOOKAHEAD + MEASUREMENT lens.

Nothing here is modified outside this file. Six mechanical checks, then a placebo suite.

MECHANICAL (does the measurement itself hold up?)
  1. costs: slippage 7bps + fee 0.10% applied once per leg, read back out of the fill log
  2. fill timing: every BUY fills at the open of the bar AFTER the bar whose close triggered it
  3. trigger shift: breakout at t compares close(t) to max(high) over t-20..t-1 only
  4. dollar_vol_30d at t uses bars <= t only (recomputed independently from raw klines)
  5. the narrowing rank is taken strictly WITHIN bar_time, and only among eligible rows
  6. the baseline and the variant are measured over identical bar ranges on both panels

PLACEBO (is the N=15 pass a volume effect, or does any 4-coin cut do it?)
  - lag the volume signal by 1 bar and by 1 day
  - rank by a different but equally legitimate volume window (7d)
  - invert the signal: keep the 15 SMALLEST by dollar volume
  - 24 pseudorandom cuts that keep the same binding structure (top 15 of the eligible set each
    bar, chosen by a slowly-drifting random key instead of by volume)
  If random cuts pass the ship rule at a high rate, N=15 passing is selection noise, not an edge.
"""

import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\amos2\Downloads\New folder (3)\crypto-bot")
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app import backtest, config, data, features  # noqa: E402
from app.flags import Flags  # noqa: E402

SHIPPED = Flags(abs_momentum=True, volume_min=0.8, taker_min=0.5)
TWO = Flags(abs_momentum=True, volume_min=0.8)
HELD = "2022-01-01"

# the two candidate bars, both taken from reports/experiments/ and reproduced below
BASE_QUOTED = {"cagr": 0.2071, "sharpe": 1.3712, "max_drawdown": -0.1509}
BASE_QUOTED_H = {"cagr": 0.1473, "sharpe": 1.1004, "max_drawdown": -0.1509}
BASE_STRICT = {"cagr": 0.2073, "sharpe": 1.4295, "max_drawdown": -0.1324}
BASE_STRICT_H = {"cagr": 0.1548, "sharpe": 1.2229, "max_drawdown": -0.1324}

CLAIM = {"cagr": 0.2121, "sharpe": 1.4507, "max_drawdown": -0.1115}
CLAIM_H = {"cagr": 0.1636, "sharpe": 1.2688, "max_drawdown": -0.1115}


def m(res):
    d = backtest.metrics(res.equity, res.trades, res.exposure)
    return {"cagr": float(d["cagr"]), "sharpe": float(d["sharpe"]),
            "max_drawdown": float(d["max_drawdown"]), "trades": int(d.get("trades", 0)),
            "first_bar": str(res.equity.index[0]), "last_bar": str(res.equity.index[-1]),
            "n_bars": int(len(res.equity))}


def beats(c, b):
    return (c["cagr"] >= b["cagr"] and c["sharpe"] >= b["sharpe"]
            and c["max_drawdown"] >= b["max_drawdown"])


def narrow_by(panel, top_n, keycol):
    """The researcher's narrow(), but ranking on any column, so placebos share the plumbing."""
    p = panel.copy()
    r = p[p["eligible"]].groupby("bar_time")[keycol].rank(ascending=False, method="first")
    keep = pd.Series(False, index=p.index)
    keep.loc[r.index] = (r <= top_n).to_numpy()
    p["eligible"] = p["eligible"] & (keep | (p["symbol"] == "BTCUSDT"))
    return features.score(p.drop(columns=["score", "rank", "n_eligible"]))


def main():
    s = config.load()
    u = json.loads((ROOT / "reports" / "_parity_universe.json").read_text())
    syms = u["symbols"]
    panel = features.build_panel(s.klines_dir, syms, s.donchian_bars)
    out = {"universe_file": "reports/_parity_universe.json", "n_symbols": len(syms),
           "panel_first": str(panel["bar_time"].min()), "panel_last": str(panel["bar_time"].max())}

    # ---------------------------------------------------------------- reproduce both baselines
    base = backtest.run(panel, s, flags=SHIPPED)
    base_h = backtest.run(panel, s, flags=SHIPPED, start=HELD)
    two = backtest.run(panel, s, flags=TWO)
    two_h = backtest.run(panel, s, flags=TWO, start=HELD)
    out["repro"] = {"three_filter_full": m(base), "three_filter_held": m(base_h),
                    "two_filter_full": m(two), "two_filter_held": m(two_h)}
    print("3-filter full  ", {k: round(v, 4) for k, v in m(base).items() if isinstance(v, float)})
    print("2-filter full  ", {k: round(v, 4) for k, v in m(two).items() if isinstance(v, float)})
    out["quoted_baseline_matches"] = {
        "two_filter": all(abs(m(two)[k] - BASE_QUOTED[k]) < 5e-4 for k in BASE_QUOTED),
        "three_filter": all(abs(m(base)[k] - BASE_QUOTED[k]) < 5e-4 for k in BASE_QUOTED)}
    print("quoted baseline is the 2-filter run:", out["quoted_baseline_matches"])

    # ------------------------------------------------------- 1) costs, read back out of the log
    lg = base.log
    buys = lg[lg["side"] == "BUY"].copy()
    sells = lg[(lg["side"] == "SELL")].copy()
    op = panel.set_index(["bar_time", "symbol"])["open"]
    bi = op.reindex(pd.MultiIndex.from_arrays([buys["time"], buys["symbol"]])).to_numpy()
    si = op.reindex(pd.MultiIndex.from_arrays([sells["time"], sells["symbol"]])).to_numpy()
    buy_slip = buys["price"].to_numpy() / bi - 1
    sell_slip = 1 - sells["price"].to_numpy() / si
    fee_rate_buy = buys["fee"].to_numpy() / (buys["qty"].to_numpy() * buys["price"].to_numpy())
    fee_rate_sell = sells["fee"].to_numpy() / (sells["qty"].to_numpy() * sells["price"].to_numpy())
    out["check1_costs"] = {
        "buy_slippage_unique": sorted(set(np.round(buy_slip[~np.isnan(buy_slip)], 8).tolist()))[:4],
        "sell_slippage_unique": sorted(set(np.round(sell_slip[~np.isnan(sell_slip)], 8).tolist()))[:4],
        "fee_rate_unique": sorted(set(np.round(np.r_[fee_rate_buy, fee_rate_sell], 8).tolist()))[:4],
        "ok": bool(np.allclose(buy_slip[~np.isnan(buy_slip)], 0.0007)
                   and np.allclose(sell_slip[~np.isnan(sell_slip)], 0.0007)
                   and np.allclose(np.r_[fee_rate_buy, fee_rate_sell], 0.001))}
    print("1) costs once per leg:", out["check1_costs"]["ok"], out["check1_costs"]["fee_rate_unique"])

    # --------------------------------------------- 2) fill timing: signal bar strictly before fill
    bt = np.sort(panel["bar_time"].unique())
    prev = pd.Series(np.r_[pd.NaT, bt[:-1]], index=bt)
    bo = panel.set_index(["bar_time", "symbol"])[["breakout", "regime_on", "eligible", "rank"]]
    sig_t = prev.reindex(buys["time"]).to_numpy()
    key = pd.MultiIndex.from_arrays([sig_t, buys["symbol"]])
    sig = bo.reindex(key)
    same = bo.reindex(pd.MultiIndex.from_arrays([buys["time"], buys["symbol"]]))
    out["check2_fill_timing"] = {
        "n_buys": int(len(buys)),
        "share_prior_bar_was_a_breakout": float(sig["breakout"].fillna(False).mean()),
        "share_fill_bar_itself_was_a_breakout": float(same["breakout"].fillna(False).mean()),
        "share_prior_bar_regime_on": float(sig["regime_on"].fillna(False).mean()),
        "share_prior_bar_rank_le_5": float((sig["rank"] <= 5).fillna(False).mean())}
    print("2) fill timing: prior bar breakout share",
          round(out["check2_fill_timing"]["share_prior_bar_was_a_breakout"], 4),
          "| fill bar breakout share",
          round(out["check2_fill_timing"]["share_fill_bar_itself_was_a_breakout"], 4))

    # -------------------------------------------------- 3) breakout trigger uses prior bars only
    sym0 = "ETHUSDT"
    b4 = data.load(s.klines_dir, sym0, "4h")
    h = b4["high"]
    don_prior = h.rolling(20).max().shift(1)          # bars t-20..t-1
    don_incl = h.rolling(20).max()                    # includes bar t  (would be lookahead-ish)
    eng = panel[panel["symbol"] == sym0].sort_values("bar_time")
    bk_prior = (b4["close"] > don_prior).to_numpy()
    bk_incl = (b4["close"] > don_incl).to_numpy()
    out["check3_trigger_shift"] = {
        "symbol": sym0,
        "engine_breakout_equals_shifted": bool(np.array_equal(
            eng["breakout"].to_numpy(), bk_prior)),
        "engine_breakout_equals_unshifted": bool(np.array_equal(
            eng["breakout"].to_numpy(), bk_incl)),
        "n_breakouts_shifted": int(np.nansum(bk_prior)),
        "n_breakouts_unshifted": int(np.nansum(bk_incl))}
    print("3) breakout == .shift(1) channel:",
          out["check3_trigger_shift"]["engine_breakout_equals_shifted"])

    # ------------------------------------------ 4) dollar_vol_30d uses bars <= t, recomputed raw
    qv = b4["quote_volume"]
    mine = (qv.rolling(180).sum() / 30).to_numpy()
    fwd = (qv.shift(-1).rolling(180).sum() / 30).to_numpy()   # a version that peeks one bar ahead
    engv = eng["dollar_vol_30d"].to_numpy()
    ok = np.allclose(engv[~np.isnan(engv)], mine[~np.isnan(mine)])
    out["check4_volume_causal"] = {
        "equals_trailing_window_ending_at_t": bool(ok),
        "equals_window_peeking_one_bar_ahead": bool(np.allclose(
            engv[200:-2], fwd[200:-2], equal_nan=True)),
        "note": "window ends at bar t; bar t has closed when the decision is taken, fill is t+1"}
    print("4) dollar_vol_30d causal:", ok)

    # --------------------------------------- 5) narrowing rank is within bar_time, eligible only
    n15 = narrow_by(panel, 15, "dollar_vol_30d")
    g = n15[n15["eligible"]].groupby("bar_time")["symbol"].nunique()
    gr = n15[n15["eligible"]].groupby("bar_time")["rank"]
    orig = panel[panel["eligible"]].groupby("bar_time")["symbol"].nunique()
    out["check5_rank_within_bar"] = {
        "max_eligible_per_bar_after_narrow": int(g.max()),
        "bars_exceeding_15": int((g > 15).sum()),
        "max_rank_equals_count_every_bar": bool((gr.max() == g).all()),
        "min_rank_is_1_every_bar": bool((gr.min() == 1).all()),
        "bars_where_cut_binds": float((orig > 15).mean()),
        "first_binding_bar": str(orig[orig > 15].index[0]) if (orig > 15).any() else None,
        "narrow_only_removes_rows": bool(
            (n15.sort_values(["bar_time", "symbol"])["eligible"].to_numpy()
             <= panel.sort_values(["bar_time", "symbol"])["eligible"].to_numpy()).all())}
    print("5) ranks within bar_time:", out["check5_rank_within_bar"]["max_rank_equals_count_every_bar"],
          "| cut binds on", round(out["check5_rank_within_bar"]["bars_where_cut_binds"], 3),
          "of bars, first", out["check5_rank_within_bar"]["first_binding_bar"])

    # --------------------------------------------------------- reproduce the claimed best variant
    v15 = backtest.run(n15, s, flags=SHIPPED)
    v15_h = backtest.run(n15, s, flags=SHIPPED, start=HELD)
    out["claim_repro"] = {"full": m(v15), "held": m(v15_h),
                          "claimed_full": CLAIM, "claimed_held": CLAIM_H,
                          "full_matches": all(abs(m(v15)[k] - CLAIM[k]) < 5e-4 for k in CLAIM),
                          "held_matches": all(abs(m(v15_h)[k] - CLAIM_H[k]) < 5e-4 for k in CLAIM_H),
                          "ships_vs_strict": beats(m(v15), BASE_STRICT) and beats(m(v15_h), BASE_STRICT_H),
                          "ships_vs_quoted": beats(m(v15), BASE_QUOTED) and beats(m(v15_h), BASE_QUOTED_H)}
    print("N=15 full", {k: round(v, 4) for k, v in m(v15).items() if isinstance(v, float)},
          "matches claim", out["claim_repro"]["full_matches"])
    print("N=15 held", {k: round(v, 4) for k, v in m(v15_h).items() if isinstance(v, float)},
          "matches claim", out["claim_repro"]["held_matches"])

    # ------------------------------------------------------- 6) identical measurement windows
    out["check6_same_window"] = {
        "full_same_first_bar": m(base)["first_bar"] == m(v15)["first_bar"],
        "full_same_last_bar": m(base)["last_bar"] == m(v15)["last_bar"],
        "full_same_n_bars": m(base)["n_bars"] == m(v15)["n_bars"],
        "held_same_first_bar": m(base_h)["first_bar"] == m(v15_h)["first_bar"],
        "held_same_last_bar": m(base_h)["last_bar"] == m(v15_h)["last_bar"],
        "held_same_n_bars": m(base_h)["n_bars"] == m(v15_h)["n_bars"]}
    print("6) identical windows:", out["check6_same_window"])

    # ================================================================= PLACEBO / FRAGILITY SUITE
    print("\n--- placebo suite: does ANY 4-coin cut pass? ---")
    p = panel.copy()
    # lagged volume signals (strictly prior information)
    p["dv_lag1"] = p.groupby("symbol")["dollar_vol_30d"].shift(1)
    p["dv_lag6"] = p.groupby("symbol")["dollar_vol_30d"].shift(6)
    # a 7-day window instead of 30, same construction (sum of quote volume / days)
    qvp = p.groupby("symbol")["dollar_vol_30d"]  # placeholder to keep groupby import warm
    del qvp
    p["dv_7d"] = p.groupby("symbol")["dollar_vol_30d"].transform(lambda x: x)  # overwritten below
    # rebuild a true 7d dollar volume per symbol from the raw klines
    sevens = []
    for sym in syms:
        b = data.load(s.klines_dir, sym, "4h")
        if b.empty:
            continue
        t = pd.to_datetime(b["open_time"] + 4 * 3_600_000, unit="ms", utc=True)
        sevens.append(pd.DataFrame({"bar_time": t, "symbol": sym,
                                    "dv7": (b["quote_volume"].rolling(42).sum() / 7).to_numpy()}))
    sev = pd.concat(sevens, ignore_index=True)
    p = p.drop(columns=["dv_7d"]).merge(sev, on=["bar_time", "symbol"], how="left")
    p["dv_neg"] = -p["dollar_vol_30d"]              # keep the 15 SMALLEST instead

    placebos = {}
    for label, col in (("volume lagged 1 bar", "dv_lag1"), ("volume lagged 1 day", "dv_lag6"),
                       ("7-day volume window", "dv7"), ("inverted: keep 15 smallest", "dv_neg")):
        pn = narrow_by(p, 15, col)
        f, hh = m(backtest.run(pn, s, flags=SHIPPED)), m(backtest.run(pn, s, flags=SHIPPED, start=HELD))
        placebos[label] = {"full": f, "held": hh,
                           "ships_vs_strict": beats(f, BASE_STRICT) and beats(hh, BASE_STRICT_H),
                           "ships_vs_quoted": beats(f, BASE_QUOTED) and beats(hh, BASE_QUOTED_H)}
        print(f"  {label:28s} full {f['cagr']:+.4f}/{f['sharpe']:.4f}/{f['max_drawdown']:+.4f} "
              f"held {hh['cagr']:+.4f}/{hh['sharpe']:.4f}/{hh['max_drawdown']:+.4f} "
              f"ships(strict) {placebos[label]['ships_vs_strict']} "
              f"ships(quoted) {placebos[label]['ships_vs_quoted']}")
    out["placebo_signals"] = placebos

    # random cuts that keep the same binding structure: a slowly drifting random key per symbol
    print("\n  24 pseudorandom top-15 cuts (same binding structure, random ordering):")
    block = (p["bar_time"].astype("int64") // (90 * 24 * 3_600_000_000)).to_numpy()
    codes = pd.factorize(p["symbol"])[0]
    rnd = []
    for seed in range(24):
        rs = np.random.default_rng(1000 + seed)
        tbl = rs.random((block.max() + 2, codes.max() + 1))
        p["rkey"] = tbl[block, codes]
        pn = narrow_by(p, 15, "rkey")
        f = m(backtest.run(pn, s, flags=SHIPPED))
        hh = m(backtest.run(pn, s, flags=SHIPPED, start=HELD))
        row = {"seed": seed, "full": f, "held": hh,
               "ships_vs_strict": beats(f, BASE_STRICT) and beats(hh, BASE_STRICT_H),
               "ships_vs_quoted": beats(f, BASE_QUOTED) and beats(hh, BASE_QUOTED_H)}
        rnd.append(row)
        print(f"   seed {seed:>2} full {f['cagr']:+.4f}/{f['sharpe']:.4f}/{f['max_drawdown']:+.4f}"
              f" held {hh['cagr']:+.4f}/{hh['sharpe']:.4f}/{hh['max_drawdown']:+.4f}"
              f"  strict {row['ships_vs_strict']}  quoted {row['ships_vs_quoted']}")
    out["placebo_random_cuts"] = rnd
    n_s = sum(r["ships_vs_strict"] for r in rnd)
    n_q = sum(r["ships_vs_quoted"] for r in rnd)
    out["placebo_pass_rates"] = {"n": len(rnd), "pass_vs_strict": n_s, "pass_vs_quoted": n_q,
                                 "rate_vs_strict": n_s / len(rnd), "rate_vs_quoted": n_q / len(rnd),
                                 "best_random_sharpe": max(r["full"]["sharpe"] for r in rnd),
                                 "n_random_beating_N15_sharpe": sum(
                                     r["full"]["sharpe"] >= m(v15)["sharpe"] for r in rnd)}
    print(f"\n  random cuts passing the ship rule: {n_s}/{len(rnd)} vs strict, "
          f"{n_q}/{len(rnd)} vs quoted; best random Sharpe "
          f"{out['placebo_pass_rates']['best_random_sharpe']:.4f} vs N=15 {m(v15)['sharpe']:.4f}")

    o = ROOT / "reports" / "verify_universe_size_lookahead.json"
    o.write_text(json.dumps(out, indent=2, default=str))
    print("\nwrote", o)


if __name__ == "__main__":
    main()
