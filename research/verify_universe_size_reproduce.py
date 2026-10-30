"""Adversarial verification of explore_universe_size's one passing variant (N=15 narrowing).

LENS: reproduce. Two independent jobs.
  1. Re-run their script and check the published numbers come back.
  2. Rebuild the winning strategy FROM SCRATCH here - my own trailing-dollar-volume measure
     computed off the panel's raw quote_volume, my own within-bar ranking of it, my own
     percentile-blend score and rank columns - and see whether the same CAGR falls out.
     Nothing is imported from research/explore_universe_size.py.

Also probes the things a single grid point should survive if it is a mechanism and not luck:
  - strictly-lagged volume (decide on volume through bar t-1 instead of through bar t)
  - a different volume window (7d, 14d, 60d, 90d) - their caveat 12 says this was never run
  - a different tie-break in the volume rank (first / min / dense / average)
  - the random_baselines seed, swept, for the baseline and for the variant

Writes nothing except reports/explore_universe_size_verify.json. Reads only; no file under
app/, data/, tests/ is touched.
"""

import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\amos2\Downloads\New folder (3)\crypto-bot")
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app import backtest, config, features  # noqa: E402
from app.flags import Flags  # noqa: E402

SHIPPED = Flags(abs_momentum=True, volume_min=0.8, taker_min=0.5)
HELD = "2022-01-01"
OUT = ROOT / "reports" / "explore_universe_size_verify.json"

# what the claim says, so the diff is mechanical
CLAIM_FULL = {"cagr": 0.2121, "sharpe": 1.4507, "max_drawdown": -0.1115}
CLAIM_HELD = {"cagr": 0.1636, "sharpe": 1.2688, "max_drawdown": -0.1115}
# the three-filter baseline they judged against (reports/experiments/standard_taker.json)
BASE_FULL = {"cagr": 0.2073371876462211, "sharpe": 1.4295214518563957,
             "max_drawdown": -0.13235652870874526, "trades": 366}
BASE_HELD = {"cagr": 0.1547862573707699, "sharpe": 1.2228570418134788,
             "max_drawdown": -0.13235652870874404, "trades": 210}


def mm(res):
    m = backtest.metrics(res.equity, res.trades, res.exposure)
    return {"cagr": float(m["cagr"]), "sharpe": float(m["sharpe"]),
            "max_drawdown": float(m["max_drawdown"]), "trades": int(m.get("trades", 0)),
            "total_return": float(m["total_return"]),
            "start": str(m["start"]), "end": str(m["end"])}


def beats(c, b):
    return (c["cagr"] >= b["cagr"] and c["sharpe"] >= b["sharpe"]
            and c["max_drawdown"] >= b["max_drawdown"])


# ----------------------------------------------------------------- my own score, from the spec

def my_score(panel):
    """The documented blend, written from the spec rather than called from features.score().

    0.35*pct(rs_7d) + 0.25*pct(rs_30d) + 0.20*pct(dist_sma50) + 0.20*pct(ret_30d/vol_30d),
    percentiles taken among the ELIGIBLE rows of that bar_time only, x100. Then rank descending
    within the same bar_time. Percentile is average-rank / n, computed by hand.
    """
    p = panel.copy()
    el = p["eligible"].to_numpy()
    bt = p["bar_time"]

    def pct_within_bar(col):
        s = p[col].where(p["eligible"])
        # average rank among the bar's eligible, divided by the bar's eligible count
        r = s.groupby(bt).rank(method="average")
        n = s.notna().groupby(bt).transform("sum")
        return r / n.replace(0, np.nan)

    a = pct_within_bar("rs_7d")
    b = pct_within_bar("rs_30d")
    c = pct_within_bar("dist_sma50")
    p["_ra"] = p["ret_30d"] / p["vol_30d"]
    d = pct_within_bar("_ra")
    sc = 100.0 * (0.35 * a + 0.25 * b + 0.20 * c + 0.20 * d)
    p["score"] = sc.where(p["eligible"])
    p["rank"] = p["score"].where(p["eligible"]).groupby(bt).rank(ascending=False, method="first")
    p["n_eligible"] = p["eligible"].astype(int).groupby(bt).transform("sum")
    assert el.sum() == int(p["eligible"].sum())
    return p.drop(columns=["_ra"]).sort_values(["bar_time", "rank"]).reset_index(drop=True)


def my_dollar_vol(panel, days=30, lag_bars=0):
    """Trailing mean daily quote volume, built here from the panel's raw quote_volume.

    Does NOT use the engine's dollar_vol_30d column. `lag_bars=1` makes the measure strictly
    prior to the decision bar, which is the paranoid reading of the timing.
    """
    p = panel[["symbol", "bar_time", "quote_volume"]].sort_values(["symbol", "bar_time"])
    win = 6 * days
    g = p.groupby("symbol", sort=False)["quote_volume"]
    v = g.transform(lambda x: x.rolling(win, min_periods=win).sum() / days)
    if lag_bars:
        v = v.groupby(p["symbol"], sort=False).shift(lag_bars)
    return v.reindex(panel.index)


def my_narrow(panel, top_n, days=30, lag_bars=0, tie="first"):
    """Keep the top N eligible coins of each bar by MY trailing dollar volume, BTC always kept."""
    p = panel.copy()
    p["_dv"] = my_dollar_vol(p, days=days, lag_bars=lag_bars)
    dv = p["_dv"].where(p["eligible"])
    r = dv.groupby(p["bar_time"]).rank(ascending=False, method=tie)
    keep = (r <= top_n).fillna(False)
    p["eligible"] = p["eligible"] & (keep | (p["symbol"] == "BTCUSDT"))
    return my_score(p.drop(columns=["_dv"]))


def main():
    s = config.load()
    u = json.loads((ROOT / "reports" / "_parity_universe.json").read_text())
    print(f"universe: {len(u['symbols'])} symbols from reports/_parity_universe.json")
    panel = features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars)
    print(f"panel: {len(panel)} rows, {panel.bar_time.min()} .. {panel.bar_time.max()}")

    doc = {"lens": "reproduce", "claim": {"full": CLAIM_FULL, "held_out": CLAIM_HELD}}

    # ---- 0) does my own score reproduce the engine's baseline bit for bit?
    print("\n=== 0) my score/rank vs the engine's, on the unmodified panel ===")
    mine = my_score(panel.drop(columns=["score", "rank", "n_eligible"]))
    j = panel[["bar_time", "symbol", "score", "rank"]].merge(
        mine[["bar_time", "symbol", "score", "rank"]], on=["bar_time", "symbol"],
        suffixes=("_eng", "_mine"))
    dsc = (j["score_eng"] - j["score_mine"]).abs()
    drk = (j["rank_eng"].fillna(-1) - j["rank_mine"].fillna(-1)).abs()
    print(f"  max |score diff| {np.nanmax(dsc):.3e}   rank mismatches {int((drk > 0).sum())}")
    doc["score_reimplementation_check"] = {"max_abs_score_diff": float(np.nanmax(dsc)),
                                           "rank_mismatches": int((drk > 0).sum())}

    base = mm(backtest.run(panel, s, flags=SHIPPED))
    base_h = mm(backtest.run(panel, s, flags=SHIPPED, start=HELD))
    mine_base = mm(backtest.run(mine, s, flags=SHIPPED))
    mine_base_h = mm(backtest.run(mine, s, flags=SHIPPED, start=HELD))
    print(f"  engine baseline  CAGR {base['cagr']:.4f} Sh {base['sharpe']:.4f} "
          f"DD {base['max_drawdown']:.4f} tr {base['trades']}")
    print(f"  my-score  "
          f"       CAGR {mine_base['cagr']:.4f} Sh {mine_base['sharpe']:.4f} "
          f"DD {mine_base['max_drawdown']:.4f} tr {mine_base['trades']}")
    doc["baseline"] = {"engine_full": base, "engine_held": base_h,
                       "my_score_full": mine_base, "my_score_held": mine_base_h,
                       "published_full": BASE_FULL, "published_held": BASE_HELD}

    # ---- 1) the winning variant, rebuilt from scratch
    print("\n=== 1) N=15 narrowing, rebuilt independently (my volume, my ranking, my score) ===")
    doc["independent_narrow"] = []
    for n in (13, 14, 15, 16, 17, 19):
        pn = my_narrow(panel, n)
        f = mm(backtest.run(pn, s, flags=SHIPPED))
        h = mm(backtest.run(pn, s, flags=SHIPPED, start=HELD))
        row = {"universe": n, "full": f, "held_out": h,
               "ships_vs_published_baseline": bool(beats(f, BASE_FULL) and beats(h, BASE_HELD))}
        doc["independent_narrow"].append(row)
        print(f"  N={n:>2}: full CAGR {f['cagr']:.4f} Sh {f['sharpe']:.4f} "
              f"DD {f['max_drawdown']:.4f} tr {f['trades']:>4} | held CAGR {h['cagr']:.4f} "
              f"Sh {h['sharpe']:.4f} DD {h['max_drawdown']:.4f} tr {h['trades']:>4} | "
              f"ships {row['ships_vs_published_baseline']}")

    got = next(r for r in doc["independent_narrow"] if r["universe"] == 15)
    doc["reproduction"] = {
        "claimed_full_cagr": CLAIM_FULL["cagr"], "my_full_cagr": got["full"]["cagr"],
        "abs_cagr_gap_points": abs(got["full"]["cagr"] - CLAIM_FULL["cagr"]) * 100,
        "claimed_held_cagr": CLAIM_HELD["cagr"], "my_held_cagr": got["held_out"]["cagr"],
        "within_1_cagr_point": abs(got["full"]["cagr"] - CLAIM_FULL["cagr"]) <= 0.01}
    print(f"\n  claimed full CAGR {CLAIM_FULL['cagr']:.4f} vs mine {got['full']['cagr']:.4f}  "
          f"gap {abs(got['full']['cagr'] - CLAIM_FULL['cagr']) * 100:.3f} CAGR points")

    # ---- 2) does the notch survive a strictly-lagged volume measure?
    print("\n=== 2) strictly-lagged volume (through bar t-1) ===")
    doc["lagged_volume"] = []
    for n in (14, 15, 16):
        pn = my_narrow(panel, n, lag_bars=1)
        f = mm(backtest.run(pn, s, flags=SHIPPED))
        h = mm(backtest.run(pn, s, flags=SHIPPED, start=HELD))
        row = {"universe": n, "lag_bars": 1, "full": f, "held_out": h,
               "ships": bool(beats(f, BASE_FULL) and beats(h, BASE_HELD))}
        doc["lagged_volume"].append(row)
        print(f"  N={n} lag1: full {f['cagr']:.4f}/{f['sharpe']:.4f}/{f['max_drawdown']:.4f} | "
              f"held {h['cagr']:.4f}/{h['sharpe']:.4f}/{h['max_drawdown']:.4f} | "
              f"ships {row['ships']}")

    # ---- 3) a different volume window: the check they said they skipped
    print("\n=== 3) other volume windows at N=15 (their caveat 12) ===")
    doc["volume_window"] = []
    for days in (7, 14, 30, 60, 90):
        pn = my_narrow(panel, 15, days=days)
        f = mm(backtest.run(pn, s, flags=SHIPPED))
        h = mm(backtest.run(pn, s, flags=SHIPPED, start=HELD))
        row = {"days": days, "full": f, "held_out": h,
               "ships": bool(beats(f, BASE_FULL) and beats(h, BASE_HELD))}
        doc["volume_window"].append(row)
        print(f"  {days:>2}d: full {f['cagr']:.4f}/{f['sharpe']:.4f}/{f['max_drawdown']:.4f} "
              f"tr {f['trades']} | held {h['cagr']:.4f}/{h['sharpe']:.4f}/"
              f"{h['max_drawdown']:.4f} | ships {row['ships']}")

    # ---- 4) tie-break in the volume rank: an arbitrary implementation choice
    print("\n=== 4) tie-break method in the volume rank, N=15 ===")
    doc["tie_break"] = []
    for tie in ("first", "min", "dense", "average"):
        pn = my_narrow(panel, 15, tie=tie)
        f = mm(backtest.run(pn, s, flags=SHIPPED))
        h = mm(backtest.run(pn, s, flags=SHIPPED, start=HELD))
        row = {"tie": tie, "full": f, "held_out": h,
               "ships": bool(beats(f, BASE_FULL) and beats(h, BASE_HELD))}
        doc["tie_break"].append(row)
        print(f"  {tie:>7}: full {f['cagr']:.4f}/{f['sharpe']:.4f}/{f['max_drawdown']:.4f} | "
              f"held {h['cagr']:.4f}/{h['sharpe']:.4f}/{h['max_drawdown']:.4f} | "
              f"ships {row['ships']}")

    # ---- 5) seed sweep on random_baselines
    print("\n=== 5) random_baselines seed sweep (baseline vs N=15) ===")
    pn15 = my_narrow(panel, 15)
    r_base = backtest.run(panel, s, flags=SHIPPED)
    r_var = backtest.run(pn15, s, flags=SHIPPED)
    doc["seed_sweep"] = []
    for seed in (7, 1, 2, 3, 11, 42, 1234, 2026):
        rb = backtest.random_baselines(panel, r_base.trades, s, seed=seed)
        rv = backtest.random_baselines(pn15, r_var.trades, s, seed=seed)
        row = {"seed": seed,
               "baseline_p_coin": float(rb["p_value_vs_random_coin"]),
               "baseline_p_time": float(rb["p_value_vs_random_time"]),
               "variant_p_coin": float(rv["p_value_vs_random_coin"]),
               "variant_p_time": float(rv["p_value_vs_random_time"]),
               "baseline_avg_trade": float(rb["strategy_avg_trade"]),
               "variant_avg_trade": float(rv["strategy_avg_trade"])}
        doc["seed_sweep"].append(row)
        print(f"  seed {seed:>5}: base p_coin {row['baseline_p_coin']:.3f} "
              f"p_time {row['baseline_p_time']:.3f} | N=15 p_coin {row['variant_p_coin']:.3f} "
              f"p_time {row['variant_p_time']:.3f}")

    # ---- 6) how thin is the margin? per-year and binding share
    print("\n=== 6) margin forensics ===")
    nel = panel.groupby("bar_time")["n_eligible"].first()
    binds = nel > 15
    doc["binding"] = {"share_bars": float(binds.mean()),
                      "first_bar": str(binds[binds].index[0]) if binds.any() else None,
                      "median_eligible": float(nel.median())}
    print(f"  N=15 binds on {binds.mean():.1%} of bars, first {str(doc['binding']['first_bar'])[:10]}")
    yb = backtest.yearly(r_base.equity)
    yv = backtest.yearly(r_var.equity)
    doc["yearly"] = {"baseline": {str(k): float(v) for k, v in yb.items()},
                     "variant": {str(k): float(v) for k, v in yv.items()}}
    for k in yb.index:
        print(f"  {k}: base {yb[k]:+.2%}  N=15 {yv[k]:+.2%}  diff {yv[k] - yb[k]:+.2%}")

    # margin over the baseline in basis points, both panels
    f, h = got["full"], got["held_out"]
    doc["margins_bp"] = {
        "full_cagr": (f["cagr"] - BASE_FULL["cagr"]) * 10000,
        "full_sharpe": (f["sharpe"] - BASE_FULL["sharpe"]),
        "held_cagr": (h["cagr"] - BASE_HELD["cagr"]) * 10000,
        "held_sharpe": (h["sharpe"] - BASE_HELD["sharpe"])}
    print(f"  margin: full CAGR {doc['margins_bp']['full_cagr']:+.0f}bp "
          f"Sharpe {doc['margins_bp']['full_sharpe']:+.4f} | "
          f"held CAGR {doc['margins_bp']['held_cagr']:+.0f}bp "
          f"Sharpe {doc['margins_bp']['held_sharpe']:+.4f}")

    OUT.write_text(json.dumps(doc, indent=2, default=str))
    print(f"\nwrote {OUT}")
    return doc


def null_distribution(n_draws=60, top_n=15):
    """THE decisive test. 'Top 15 of 19 by dollar volume' drops 4 names per bar. So does any other
    persistent 4-name cut. Give every symbol a random constant key, rank within each bar among the
    eligible, keep the top 15 - same binding frequency, same portfolio machinery, zero information.
    If a fair share of those random cuts also pass the ship rule, then N=15 by dollar volume is a
    draw from this distribution and not an edge.
    """
    s = config.load()
    u = json.loads((ROOT / "reports" / "_parity_universe.json").read_text())
    panel = features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars)
    syms = sorted(panel["symbol"].unique())
    print(f"null: {len(syms)} symbols, keeping top {top_n} per bar by a random constant key")

    real = my_narrow(panel, top_n)
    rf, rh = mm(backtest.run(real, s, flags=SHIPPED)), \
        mm(backtest.run(real, s, flags=SHIPPED, start=HELD))
    print(f"  REAL dollar-volume cut: full {rf['cagr']:.4f}/{rf['sharpe']:.4f}/"
          f"{rf['max_drawdown']:.4f} | held {rh['cagr']:.4f}/{rh['sharpe']:.4f}/"
          f"{rh['max_drawdown']:.4f}")

    rows = []
    for d in range(n_draws):
        rng = np.random.default_rng(1000 + d)
        key = dict(zip(syms, rng.permutation(len(syms))))
        p = panel.copy()
        k = p["symbol"].map(key).where(p["eligible"])
        r = k.groupby(p["bar_time"]).rank(ascending=False, method="first")
        keep = (r <= top_n).fillna(False)
        p["eligible"] = p["eligible"] & (keep | (p["symbol"] == "BTCUSDT"))
        p = my_score(p)
        f = mm(backtest.run(p, s, flags=SHIPPED))
        h = mm(backtest.run(p, s, flags=SHIPPED, start=HELD))
        ok = beats(f, BASE_FULL) and beats(h, BASE_HELD)
        # the cut keeps the HIGHEST keys, so the names that lose their place are the lowest ones.
        # (The first written run of this file labelled the highest keys "dropped" by mistake; the
        # mechanism was always "keep the top `top_n` by key", which is what the null needs.)
        n_cut = max(0, len(syms) - top_n)
        rows.append({"draw": d, "dropped": [x for x in syms if key[x] < n_cut],
                     "n_names_the_cut_can_remove": n_cut,
                     "full": f, "held_out": h, "ships": bool(ok)})
        print(f"  draw {d:>2}: full {f['cagr']:.4f}/{f['sharpe']:.4f}/{f['max_drawdown']:+.4f} | "
              f"held {h['cagr']:.4f}/{h['sharpe']:.4f}/{h['max_drawdown']:+.4f} | ships {ok}",
              flush=True)

    cg = np.array([r["full"]["cagr"] for r in rows])
    sh = np.array([r["full"]["sharpe"] for r in rows])
    dd = np.array([r["full"]["max_drawdown"] for r in rows])
    npass = sum(r["ships"] for r in rows)
    out = {"n_draws": n_draws, "top_n": top_n,
           "real": {"full": rf, "held_out": rh},
           "null_full_cagr": {"mean": float(cg.mean()), "sd": float(cg.std(ddof=1)),
                              "min": float(cg.min()), "max": float(cg.max()),
                              "p90": float(np.quantile(cg, 0.9))},
           "null_full_sharpe": {"mean": float(sh.mean()), "sd": float(sh.std(ddof=1)),
                                "max": float(sh.max())},
           "null_full_maxdd": {"mean": float(dd.mean()), "best": float(dd.max())},
           "n_random_cuts_passing_ship_rule": int(npass),
           "share_random_cuts_passing": npass / n_draws,
           "share_random_cuts_with_cagr_at_least_real": float((cg >= rf["cagr"]).mean()),
           "share_random_cuts_with_sharpe_at_least_real": float((sh >= rf["sharpe"]).mean()),
           "draws": rows}
    print(f"\n  null full CAGR  mean {cg.mean():.4f} sd {cg.std(ddof=1):.4f} "
          f"max {cg.max():.4f}  (real {rf['cagr']:.4f})")
    print(f"  null full Sharpe mean {sh.mean():.4f} sd {sh.std(ddof=1):.4f} "
          f"max {sh.max():.4f}  (real {rf['sharpe']:.4f})")
    print(f"  random cuts passing the FULL ship rule: {npass}/{n_draws} "
          f"({npass / n_draws:.0%})")
    print(f"  random cuts with CAGR >= the real cut: "
          f"{(cg >= rf['cagr']).mean():.0%};  Sharpe >= real: {(sh >= rf['sharpe']).mean():.0%}")
    path = ROOT / "reports" / "explore_universe_size_verify_null.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {path}")
    return out


def my_membership(kdir, symbols, top_n):
    """My own point-in-time membership mask, written from the definition rather than imported.

    day -> symbol -> True if that symbol was in the top N by the 30-day quote-volume sum over days
    that had ALREADY CLOSED before that day. Indexed by the daily bar's OPEN day, and shifted one
    day, so the mask for day D is built from D-30..D-1.
    """
    from app import data
    cols = {}
    for s in symbols:
        d = data.load(kdir, s, "1d")
        if len(d):
            idx = pd.to_datetime(d["open_time"], unit="ms", utc=True)
            cols[s] = pd.Series(d["quote_volume"].to_numpy(), index=idx)
    v = pd.DataFrame(cols).sort_index()
    trailing = v.rolling(30, min_periods=30).sum().shift(1)
    # rank across symbols within a day: strictly a cross-sectional operation, never across days
    return trailing.rank(axis=1, ascending=False, method="first") <= top_n


def verify_pit(top_n):
    """Independent rebuild of their headline survivorship-free run at one universe size."""
    import glob
    import os
    from app import pit
    s = config.load()
    kdir = s.data_dir / "pit" / "klines"
    cached = sorted(os.path.basename(p)[: -len("_1d.parquet")]
                    for p in glob.glob(str(kdir / "*_1d.parquet")))
    syms = [x for x in cached if pit.keep(x, s.exclude_memes)]
    member = my_membership(kdir, syms, top_n)
    ever = sorted(set(member.columns[member.any()]) | {"BTCUSDT"})
    print(f"pit top {top_n}: {len(syms)} cached pairs, {len(ever)} ever in the universe")
    panel = features.build_panel(kdir, ever, s.donchian_bars)
    panel["day"] = (panel["bar_time"] - pd.Timedelta(milliseconds=1)).dt.normalize()
    m = member.rename_axis(index="day", columns="symbol").stack().rename("member").reset_index()
    panel = panel.merge(m, on=["day", "symbol"], how="left")
    panel["member"] = panel["member"].astype("boolean").fillna(False).astype(bool)
    panel["eligible"] = panel["eligible"] & (panel["member"] | (panel["symbol"] == "BTCUSDT"))
    panel = my_score(panel.drop(columns=["score", "rank", "n_eligible"]))
    res = backtest.run(panel, s, flags=SHIPPED)
    f = mm(res)
    h = mm(backtest.run(panel, s, flags=SHIPPED, start=HELD))
    rb = {sd: float(backtest.random_baselines(panel, res.trades, s,
                                              seed=sd)["p_value_vs_random_coin"])
          for sd in (7, 1, 42, 2026)}
    out = {"universe": top_n, "n_symbols_in_panel": int(panel["symbol"].nunique()),
           "full": f, "held_out": h, "p_coin_by_seed": rb}
    print(f"  full  {f['cagr']:.4f}/{f['sharpe']:.4f}/{f['max_drawdown']:.4f} tr {f['trades']}")
    print(f"  held  {h['cagr']:.4f}/{h['sharpe']:.4f}/{h['max_drawdown']:.4f} tr {h['trades']}")
    print(f"  p_coin by seed {rb}")
    path = ROOT / "reports" / f"explore_universe_size_verify_pit{top_n}.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {path}")
    return out


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "pit":
        verify_pit(int(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == "null":
        null_distribution(int(sys.argv[2]) if len(sys.argv) > 2 else 60)
    else:
        main()
