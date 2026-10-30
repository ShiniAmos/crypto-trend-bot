"""Universe size and slot count: is breadth a bigger lever than any entry filter?

HYPOTHESIS
----------
An early survivorship-free test of this bot (before the three shipped filters existed) scored
+13.4%/yr on a top-10 universe against +5.7% on top-30 - the largest spread ever seen in this
project. It has never been re-tested with abs_momentum + volume_min=0.8 + taker_min=0.5 on.
This script re-tests it, and separately asks whether holding more or fewer names helps.

WHAT IS MEASURED
----------------
A) survivorship-free (point-in-time) universes at N = 10, 15, 20, 30, shipped filters on.
   Full history and held out from 2022-01-01.
B) standard panel (the pinned 19-coin parity universe), max_names K = 2..6, full + held out.
C) standard panel with the universe itself narrowed to the top N coins by trailing 30-day dollar
   volume on each bar (N = 5, 8, 10, 12, 15, 19) - the standard-panel analogue of (A), so the
   ship rule can actually be evaluated on both required panels.
D) the best (universe size x slot count) combination from B and C, re-checked on both panels.

NO-LOOKAHEAD NOTES
------------------
* Nothing in this file computes a new signal. Every column used already exists in the engine's
  panel and was built from closed bars only (don_high/don_low are .shift(1)ed in features.py).
* The point-in-time membership mask is app.pit.membership unchanged: a trailing 30-day volume sum
  that is .shift(1)ed, so a coin joins the universe on the morning AFTER the data that qualified
  it, and the daily mask is joined to 4h bars on the bar's own completed day.
* (C) narrows eligibility with dollar_vol_30d, which at bar t is the mean daily quote volume over
  the 30 days ending at bar t. The decision is taken on the close of t and fills at the open of
  t+1, so this is information the bot has in hand. Ranked WITHIN each bar_time only.
* Scores and ranks are always recomputed with features.score() after eligibility changes, which
  ranks inside a bar_time group. No cross-bar ranking anywhere.

WHY THE PIT RUNS ARE DONE OFFLINE
---------------------------------
app.main pit refetches every Binance USDT pair from the archive on each call, which would (a)
re-download ~20 minutes of data per universe size and (b) move the data end date between runs, so
N=10 and the published N=30 number would no longer share a snapshot. This script instead reuses the
already-downloaded cache in data/pit/klines and calls exactly the same pit.membership /
features.build_panel / features.score / backtest.run chain that app.pit.run calls, with the symbol
list taken from the cached 1d files (pit.keep() applied, same as app.pit.run). All four universe
sizes therefore share one snapshot, and N=30 is re-run here rather than quoted, so the comparison
is apples to apples. Nothing under app/, data/, tests/ is written or modified.

Usage:
    python research/explore_universe_size.py            # everything (pit parts in subprocesses)
    python research/explore_universe_size.py pit 10      # one point-in-time universe size
    python research/explore_universe_size.py standard    # parts B, C, D
"""

import glob
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(r"C:\Users\amos2\Downloads\New folder (3)\crypto-bot")
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from app import backtest, config, features, pit  # noqa: E402
from app.flags import Flags  # noqa: E402

SHIPPED = Flags(abs_momentum=True, volume_min=0.8, taker_min=0.5)
HELD_OUT = "2022-01-01"
PARTS = Path(tempfile.gettempdir()) / "explore_universe_size_parts"
OUT = ROOT / "reports" / "explore_universe_size.json"

# The numbers quoted to this experiment as "the baseline, flags abs_momentum + volume_min=0.8 +
# taker_min=0.5". They are NOT that flag set: they are reports/experiments/standard_vol08.json,
# i.e. abs_momentum + volume_min=0.8 with taker_min OFF (verified exactly, see
# reports/explore_universe_size.json -> quoted_baseline_is_actually). The genuine three-filter
# shipped run is standard_taker.json, reproduced bit for bit below as BASE_STRICT. Every ship
# decision in this file is taken against BASE_STRICT, which is the harder bar on Sharpe and
# drawdown; the comparison against the quoted figures is reported alongside, never instead.
BASE_QUOTED = {"cagr": 0.2071, "sharpe": 1.3712, "max_drawdown": -0.1509, "trades": 398}
BASE_QUOTED_HELD = {"cagr": 0.1473, "sharpe": 1.1004, "max_drawdown": -0.1509, "trades": 233}
BASE_STRICT = {"cagr": 0.2073, "sharpe": 1.4295, "max_drawdown": -0.1324, "trades": 366}
BASE_STRICT_HELD = {"cagr": 0.1548, "sharpe": 1.2229, "max_drawdown": -0.1324, "trades": 210}
BASE_PIT30 = {"cagr": 0.1018, "sharpe": 0.8796, "max_drawdown": -0.1713}

KEEP_COLS = ["bar_time", "symbol", "open", "high", "low", "close", "ret_4h", "eligible",
             "regime_on", "breakout", "breakdown", "stop_level", "rank", "score", "rs_7d",
             "ret_30d", "volume_ratio", "taker_ratio", "btc_ret_30d", "don_pos", "atr_pct",
             "vol_30d", "rs_30d", "dist_sma50", "dollar_vol_30d", "age_bars", "n_eligible"]


def m_of(res, with_p=False, panel=None, settings=None):
    """metrics() plus, optionally, the random-coin p-value the project reports as p_coin."""
    m = backtest.metrics(res.equity, res.trades, res.exposure)
    out = {"cagr": float(m["cagr"]), "sharpe": float(m["sharpe"]),
           "max_drawdown": float(m["max_drawdown"]),
           "trades": int(m.get("trades", 0)),
           "win_rate": float(m.get("win_rate", float("nan"))),
           "profit_factor": float(m.get("profit_factor", float("nan"))),
           "avg_exposure": float(m.get("avg_exposure", float("nan"))),
           "total_return": float(m["total_return"]),
           "start": str(m["start"]), "end": str(m["end"])}
    if with_p and panel is not None and len(res.trades):
        rb = backtest.random_baselines(panel, res.trades, settings)
        out["p_coin"] = float(rb.get("p_value_vs_random_coin", float("nan")))
        out["p_time"] = float(rb.get("p_value_vs_random_time", float("nan")))
    return out


def beats(cand, base):
    """The ship rule on one panel: CAGR >= base AND Sharpe >= base AND drawdown no worse."""
    return (cand["cagr"] >= base["cagr"] and cand["sharpe"] >= base["sharpe"]
            and cand["max_drawdown"] >= base["max_drawdown"])


def dd_window(equity):
    """(peak time, trough time, depth) of the deepest drawdown."""
    dd = equity / equity.cummax() - 1
    trough = dd.idxmin()
    peak = equity.loc[:trough].idxmax()
    return peak, trough, float(dd.min())


def trade_diff(base_trades, var_trades):
    """Round trips the baseline took that the variant did not, and the reverse, keyed symbol+entry.

    This is how a 'win' that is really one avoided loss gets caught: if the variant differs by a
    handful of trades and one of them explains the whole drawdown gap, it is noise, not an edge.
    """
    def key(t):
        return set(zip(t["symbol"], t["entry_time"])) if len(t) else set()
    bk, vk = key(base_trades), key(var_trades)
    only_base = base_trades[[k in (bk - vk) for k in zip(base_trades["symbol"],
                                                         base_trades["entry_time"])]]
    only_var = var_trades[[k in (vk - bk) for k in zip(var_trades["symbol"],
                                                       var_trades["entry_time"])]]
    def rows(t):
        return [{"symbol": r.symbol, "entry_time": str(r.entry_time), "ret": float(r.ret),
                 "pnl": float(r.pnl), "exit_reason": r.exit_reason} for r in t.itertuples()]
    return {"n_only_baseline": len(only_base), "n_only_variant": len(only_var),
            "only_baseline": rows(only_base.nsmallest(min(8, len(only_base)), "ret"))
            if len(only_base) else [],
            "only_variant": rows(only_var.nsmallest(min(8, len(only_var)), "ret"))
            if len(only_var) else []}


# ------------------------------------------------------------------ A) point-in-time universes

def pit_symbols(settings):
    kdir = settings.data_dir / "pit" / "klines"
    cached = sorted(os.path.basename(p)[: -len("_1d.parquet")]
                    for p in glob.glob(str(kdir / "*_1d.parquet")))
    return kdir, [s for s in cached if pit.keep(s, settings.exclude_memes)]


def pit_panel(settings, top_n):
    """app.pit.run's panel, rebuilt from the cache with no refetch (same code path otherwise)."""
    kdir, syms = pit_symbols(settings)
    member = pit.membership(kdir, syms, top_n)
    ever = sorted(set(member.columns[member.any()]) | {"BTCUSDT"})
    print(f"  top {top_n}: {len(syms)} pairs scanned, {len(ever)} were ever in the universe")
    panel = features.build_panel(kdir, ever, settings.donchian_bars)
    panel["day"] = (panel["bar_time"] - pd.Timedelta(milliseconds=1)).dt.normalize()
    m = member.rename_axis(index="day", columns="symbol").stack().rename("member").reset_index()
    panel = panel.merge(m, on=["day", "symbol"], how="left")
    panel["member"] = panel["member"].astype("boolean").fillna(False).astype(bool)
    panel["eligible"] = panel["eligible"] & (panel["member"] | (panel["symbol"] == "BTCUSDT"))
    panel = features.score(panel.drop(columns=["score", "rank", "n_eligible"]))
    return panel[[c for c in KEEP_COLS if c in panel.columns]].copy()


def run_pit_extra(top_n):
    """Sub-period stability for one point-in-time universe size: the headline claim is that top 10
    beats top 30, so it has to beat it in more than one window."""
    s = config.load()
    panel = pit_panel(s, top_n)
    row = {"universe": top_n, "thirds": [], "start_sweep": []}
    for a, b in (("2017-08-17", "2020-12-31"), ("2021-01-01", "2023-06-30"),
                 ("2023-07-01", "2026-10-04")):
        m = m_of(backtest.run(panel, s, flags=SHIPPED, start=a, end=b))
        row["thirds"].append({"from": a, "to": b, **m})
        print(f"  top {top_n} {a[:7]}..{b[:7]}: CAGR {m['cagr']:+.4f} Sh {m['sharpe']:+.3f} "
              f"DD {m['max_drawdown']:+.4f} tr {m['trades']}")
    for a in ("2020-01-01", "2021-01-01", "2023-01-01", "2024-01-01"):
        m = m_of(backtest.run(panel, s, flags=SHIPPED, start=a))
        row["start_sweep"].append({"start": a, **m})
        print(f"  top {top_n} from {a[:7]}: CAGR {m['cagr']:+.4f} Sh {m['sharpe']:+.3f} "
              f"DD {m['max_drawdown']:+.4f} tr {m['trades']}")
    PARTS.mkdir(parents=True, exist_ok=True)
    (PARTS / f"pit_extra_{top_n}.json").write_text(json.dumps(row, indent=2))
    return row


def run_pit(top_n):
    s = config.load()
    panel = pit_panel(s, top_n)
    row = {"universe": top_n, "n_symbols_in_panel": int(panel["symbol"].nunique())}
    res = backtest.run(panel, s, flags=SHIPPED)
    row["full"] = m_of(res, with_p=True, panel=panel, settings=s)
    held = backtest.run(panel, s, flags=SHIPPED, start=HELD_OUT)
    row["held_out"] = m_of(held)
    # sanity: a stop above the entry price would mean the sizing reference is broken
    t = res.trades
    row["sanity"] = {"nan_equity": bool(res.equity.isna().any()),
                     "min_equity": float(res.equity.min()),
                     "losing_trades": int((t.pnl <= 0).sum()) if len(t) else 0}
    PARTS.mkdir(parents=True, exist_ok=True)
    (PARTS / f"pit_{top_n}.json").write_text(json.dumps(row, indent=2))
    print(f"  top {top_n} FULL : CAGR {row['full']['cagr']:.4f}  Sharpe {row['full']['sharpe']:.4f}"
          f"  maxDD {row['full']['max_drawdown']:.4f}  trades {row['full']['trades']}"
          f"  p_coin {row['full'].get('p_coin', float('nan')):.3f}")
    print(f"  top {top_n} HELD : CAGR {row['held_out']['cagr']:.4f}"
          f"  Sharpe {row['held_out']['sharpe']:.4f}"
          f"  maxDD {row['held_out']['max_drawdown']:.4f}  trades {row['held_out']['trades']}")
    return row


# ------------------------------------- B/C/D) standard panel: slots, and a narrowed universe

def std_panel(settings):
    u = json.loads((ROOT / "reports" / "_parity_universe.json").read_text())
    return features.build_panel(settings.klines_dir, u["symbols"], settings.donchian_bars)


def narrow(panel, top_n):
    """Keep only the top N coins by trailing 30-day dollar volume on each bar, then re-rank.

    dollar_vol_30d at bar t covers the 30 days ending at t, a closed-bar quantity, and the rank is
    taken within that bar_time only. BTC is always kept, mirroring app.pit.run.
    """
    p = panel.copy()
    r = p[p["eligible"]].groupby("bar_time")["dollar_vol_30d"].rank(ascending=False,
                                                                   method="first")
    keep = pd.Series(False, index=p.index)
    keep.loc[r.index] = (r <= top_n).to_numpy()
    p["eligible"] = p["eligible"] & (keep | (p["symbol"] == "BTCUSDT"))
    return features.score(p.drop(columns=["score", "rank", "n_eligible"]))


def line(tag, f, h, ships):
    return (f"  {tag}: full CAGR {f['cagr']:.4f} Sh {f['sharpe']:.4f} DD {f['max_drawdown']:.4f} "
            f"tr {f['trades']:>4} | held CAGR {h['cagr']:.4f} Sh {h['sharpe']:.4f} "
            f"DD {h['max_drawdown']:.4f} tr {h['trades']:>4} | ships {ships}")


def run_standard():
    s = config.load()
    panel = std_panel(s)
    out = {"slots": [], "narrow": [], "combo": []}

    print("\n=== check: which record do the quoted baseline numbers actually come from? ===")
    two = backtest.run(panel, s, flags=Flags(abs_momentum=True, volume_min=0.8))
    two_h = backtest.run(panel, s, flags=Flags(abs_momentum=True, volume_min=0.8), start=HELD_OUT)
    base_res = backtest.run(panel, s, flags=SHIPPED)
    base_hres = backtest.run(panel, s, flags=SHIPPED, start=HELD_OUT)
    base, base_h = m_of(base_res), m_of(base_hres)
    out["quoted_baseline_is_actually"] = {
        "note": "the figures quoted as the 3-filter baseline are the 2-filter run "
                "(abs_momentum + volume_min=0.8, taker_min OFF) = standard_vol08.json",
        "two_filter_full": m_of(two), "two_filter_held_out": m_of(two_h),
        "quoted_full": BASE_QUOTED, "quoted_held_out": BASE_QUOTED_HELD}
    out["baseline_reproduced"] = {"full": base, "held_out": base_h}
    print(f"  2-filter full : CAGR {m_of(two)['cagr']:.4f} Sh {m_of(two)['sharpe']:.4f} "
          f"DD {m_of(two)['max_drawdown']:.4f} tr {m_of(two)['trades']}   "
          f"<- quoted as baseline ({BASE_QUOTED['cagr']}/{BASE_QUOTED['sharpe']}/"
          f"{BASE_QUOTED['max_drawdown']}/{BASE_QUOTED['trades']})")
    print(f"  3-filter full : CAGR {base['cagr']:.4f} Sh {base['sharpe']:.4f} "
          f"DD {base['max_drawdown']:.4f} tr {base['trades']}   <- the REAL shipped baseline")
    print(f"  3-filter held : CAGR {base_h['cagr']:.4f} Sh {base_h['sharpe']:.4f} "
          f"DD {base_h['max_drawdown']:.4f} tr {base_h['trades']}")

    print("\n=== B) how many names to hold at once (standard panel) ===")
    for k in (2, 3, 4, 5, 6):
        f = m_of(backtest.run(panel, s, flags=SHIPPED, max_names=k))
        h = m_of(backtest.run(panel, s, flags=SHIPPED, max_names=k, start=HELD_OUT))
        row = {"max_names": k, "full": f, "held_out": h,
               "ships": beats(f, BASE_STRICT) and beats(h, BASE_STRICT_HELD),
               "ships_vs_quoted": beats(f, BASE_QUOTED) and beats(h, BASE_QUOTED_HELD)}
        out["slots"].append(row)
        print(line(f"K={k}", f, h, row["ships"]))

    print("\n=== C) narrowing the standard universe by trailing dollar volume (fine grid) ===")
    narrowed, var_res = {}, {}
    for n in (5, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19):
        pn = narrow(panel, n)
        narrowed[n] = pn
        rf = backtest.run(pn, s, flags=SHIPPED)
        var_res[n] = rf
        f = m_of(rf)
        h = m_of(backtest.run(pn, s, flags=SHIPPED, start=HELD_OUT))
        row = {"universe": n, "full": f, "held_out": h,
               "ships": beats(f, BASE_STRICT) and beats(h, BASE_STRICT_HELD),
               "ships_vs_quoted": beats(f, BASE_QUOTED) and beats(h, BASE_QUOTED_HELD)}
        out["narrow"].append(row)
        print(line(f"N={n:>2}", f, h, row["ships"]))

    print("\n=== C2) is the best N a real effect or one trade? ===")
    cand = [r for r in out["narrow"] if r["universe"] != 19 and r["ships"]]
    best_n = (max(cand, key=lambda r: r["full"]["sharpe"])["universe"] if cand
              else max([r for r in out["narrow"] if r["universe"] != 19],
                       key=lambda r: r["full"]["sharpe"])["universe"])
    out["forensics"] = {"universe": best_n}
    d = trade_diff(base_res.trades, var_res[best_n].trades)
    out["forensics"]["trade_diff"] = d
    print(f"  N={best_n} vs N=19: {d['n_only_baseline']} trades dropped, "
          f"{d['n_only_variant']} trades added")
    for r in d["only_baseline"][:5]:
        print(f"    dropped {r['symbol']:10s} {r['entry_time'][:10]} ret {r['ret']:+.3f} "
              f"pnl {r['pnl']:+.1f}")
    for r in d["only_variant"][:5]:
        print(f"    added   {r['symbol']:10s} {r['entry_time'][:10]} ret {r['ret']:+.3f} "
              f"pnl {r['pnl']:+.1f}")
    bp, bt, bd = dd_window(base_res.equity)
    vp, vt, vd = dd_window(var_res[best_n].equity)
    out["forensics"]["dd_window"] = {"baseline": [str(bp), str(bt), bd],
                                     "variant": [str(vp), str(vt), vd]}
    print(f"  baseline worst DD {bd:.4f} from {bp.date()} to {bt.date()}")
    print(f"  N={best_n}  worst DD {vd:.4f} from {vp.date()} to {vt.date()}")
    # sub-period stability: does the narrowed universe win in each third of the history?
    out["forensics"]["thirds"] = []
    for a, b in (("2017-08-17", "2020-12-31"), ("2021-01-01", "2023-06-30"),
                 ("2023-07-01", "2026-09-24")):
        bm = m_of(backtest.run(panel, s, flags=SHIPPED, start=a, end=b))
        vm = m_of(backtest.run(narrowed[best_n], s, flags=SHIPPED, start=a, end=b))
        wins = vm["cagr"] >= bm["cagr"] and vm["sharpe"] >= bm["sharpe"]
        out["forensics"]["thirds"].append({"from": a, "to": b, "baseline": bm, "variant": vm,
                                           "variant_wins": bool(wins)})
        print(f"  {a[:7]}..{b[:7]}: base CAGR {bm['cagr']:+.4f} Sh {bm['sharpe']:+.3f} "
              f"tr {bm['trades']:>3} | N={best_n} CAGR {vm['cagr']:+.4f} Sh {vm['sharpe']:+.3f} "
              f"tr {vm['trades']:>3} | variant wins {wins}")

    # How often does narrowing to best_n bind at all? If the pinned universe rarely has more than
    # best_n eligible names, the "win" can only come from a short stretch of the history.
    nel = panel.groupby("bar_time")["n_eligible"].first()
    binds = (nel > best_n)
    first_bind = str(binds[binds].index[0]) if binds.any() else None
    out["forensics"]["binding"] = {"share_of_bars_where_filter_binds": float(binds.mean()),
                                   "first_bar_where_it_binds": first_bind,
                                   "median_eligible_names": float(nel.median())}
    print(f"  the N={best_n} cut only binds on {binds.mean():.1%} of bars, first at "
          f"{str(first_bind)[:10]} (median eligible names {nel.median():.0f})")

    print("\n=== C3) does the win survive moving the held-out start? ===")
    out["forensics"]["start_sweep"] = []
    for a in ("2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01"):
        bm = m_of(backtest.run(panel, s, flags=SHIPPED, start=a))
        vm = m_of(backtest.run(narrowed[best_n], s, flags=SHIPPED, start=a))
        ok = beats(vm, bm)
        out["forensics"]["start_sweep"].append({"start": a, "baseline": bm, "variant": vm,
                                                "variant_passes_ship_rule": bool(ok)})
        print(f"  from {a[:7]}: base CAGR {bm['cagr']:+.4f} Sh {bm['sharpe']:+.3f} "
              f"DD {bm['max_drawdown']:+.4f} | N={best_n} CAGR {vm['cagr']:+.4f} "
              f"Sh {vm['sharpe']:+.3f} DD {vm['max_drawdown']:+.4f} | passes {ok}")

    print("\n=== D) narrowed universe x slot count ===")
    for n in sorted({best_n, 10, 15}):
        for k in sorted({2, 4, 6}):
            pn = narrowed[n]
            f = m_of(backtest.run(pn, s, flags=SHIPPED, max_names=k))
            h = m_of(backtest.run(pn, s, flags=SHIPPED, max_names=k, start=HELD_OUT))
            row = {"universe": n, "max_names": k, "full": f, "held_out": h,
                   "ships": beats(f, BASE_STRICT) and beats(h, BASE_STRICT_HELD),
                   "ships_vs_quoted": beats(f, BASE_QUOTED) and beats(h, BASE_QUOTED_HELD)}
            out["combo"].append(row)
            print(line(f"N={n:>2} K={k}", f, h, row["ships"]))

    PARTS.mkdir(parents=True, exist_ok=True)
    (PARTS / "standard.json").write_text(json.dumps(out, indent=2, default=str))
    return out


# ------------------------------------------------------------------------------- orchestration

def main():
    PARTS.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for n in (10, 15, 20, 30):
        if (PARTS / f"pit_{n}.json").exists():
            print(f"pit {n}: cached")
            continue
        print(f"\n=== A) survivorship-free universe, top {n} ===", flush=True)
        subprocess.run([sys.executable, __file__, "pit", str(n)], cwd=str(ROOT), env=env,
                       check=True)
    pits = [json.loads((PARTS / f"pit_{n}.json").read_text()) for n in (10, 15, 20, 30)]
    for n in (10, 30):
        if not (PARTS / f"pit_extra_{n}.json").exists():
            subprocess.run([sys.executable, __file__, "pitextra", str(n)], cwd=str(ROOT), env=env,
                           check=True)
    extra = {n: json.loads((PARTS / f"pit_extra_{n}.json").read_text()) for n in (10, 30)}
    std = (json.loads((PARTS / "standard.json").read_text())
           if (PARTS / "standard.json").exists() else run_standard())

    pit30 = next(r for r in pits if r["universe"] == 30)
    for r in pits:
        r["ships_vs_pit30_same_snapshot"] = (beats(r["full"], pit30["full"])
                                            and beats(r["held_out"], pit30["held_out"]))
        r["ships_vs_pit30_published"] = beats(r["full"], BASE_PIT30)
    cands = std["slots"] + std["narrow"] + std["combo"]
    winners = [c for c in cands if c["ships"]]
    best = max(cands, key=lambda c: (c["ships"], c["full"]["sharpe"]))

    doc = {
        "experiment": "explore_universe_size",
        "hypothesis": "universe size is a bigger lever than any entry filter; and the 3-slot "
                      "portfolio may be the wrong width",
        "ship_rule": "CAGR >= base AND Sharpe >= base AND maxDD not worse, on BOTH the standard "
                     "panel and the held-out period from 2022-01-01",
        "baseline_used_for_ship_rule": {"standard": BASE_STRICT, "held_out": BASE_STRICT_HELD,
                                        "source": "reports/experiments/standard_taker.json, "
                                                  "reproduced exactly by this script"},
        "quoted_baseline_is_actually": std["quoted_baseline_is_actually"],
        "baseline_reproduced": std["baseline_reproduced"],
        "A_point_in_time_universe_size": pits,
        "B_slot_count_standard": std["slots"],
        "C_narrowed_standard_universe": std["narrow"],
        "A2_pit_subperiod_stability": {"top10": extra[10], "top30": extra[30]},
        "C2_forensics_on_best_narrowed": std["forensics"],
        "D_combinations": std["combo"],
        "variants_tested": len(pits) + len(cands),
        "beats_baseline_on_both_panels": bool(winners),
        "best_variant": best,
        "verdict": {
            "hypothesis_on_the_survivorship_free_panel": "CONFIRMED and it survives the three "
                "shipped filters. Point-in-time top 10 scores CAGR 16.93% / Sharpe 1.279 / "
                "maxDD -13.74% against top 30's 9.90% / 0.857 / -17.13% on the same cached "
                "snapshot, and it wins from every start date tested (2020, 2021, 2023, 2024) and "
                "in two of three thirds of the history. The one third it loses, 2021-01..2023-06, "
                "it loses on return while halving the drawdown.",
            "hypothesis_on_the_shipped_panels": "FAILED. The same narrowing applied to the pinned "
                "19-coin parity universe makes things worse, not better: top 10 by trailing dollar "
                "volume gives 19.42% / 1.361 / -16.59% against the baseline's 20.73% / 1.430 / "
                "-13.24%, and it fails the ship rule on both panels.",
            "why_the_two_panels_disagree": "universe SIZE is not the lever; universe QUALITY is. "
                "The pit top-30 list is the top 30 of every USDT pair Binance ever listed, so it "
                "is full of brief high-volume listings and pump names, and cutting to 10 removes "
                "them. The pinned 19-coin universe is already a hand-filtered mega-cap list "
                "(stablecoins, fiat, wrapped, leveraged and meme coins removed, 90-day history "
                "and $20M turnover floors), so the quality cut has already been made and "
                "narrowing further only removes diversification. The shipped universe builder "
                "already captures essentially all of this lever.",
            "slot_count": "3 is already the best choice. K=2 halves the return, K=4/5/6 buy more "
                "CAGR only by deepening the drawdown past the baseline, so every K fails the ship "
                "rule. K=6 is the most tempting (25.28% CAGR) and the clearest reject: Sharpe "
                "1.316 < 1.430 and maxDD -19.79% against -13.24%.",
            "only_passing_variant": "N=15 on the standard panel, 3 slots. It passes mechanically "
                "but should NOT ship: see caveats_for_reviewer.",
            "caveats_for_reviewer": [
                "the N=15 pass is a single grid point: N=14 fails (held-out CAGR 0.1541 against "
                "the baseline's 0.1548, short by 7 bp) and N=16 fails (CAGR 0.2053 < 0.2073). "
                "N=17, 18 and 19 are bit-identical to the baseline, so the whole effect lives in "
                "a two-wide notch.",
                "the N=15 cut binds on only 34.6% of bars and first binds on 2023-08-01, because "
                "the pinned universe rarely has more than 15 coins with enough history before "
                "then. Two of the three thirds of the history are bit-identical to the baseline, "
                "so a 9.1-year number is being moved by 3.2 years of data and 62 differing trades.",
                "it fails the ship rule from a 2025-01-01 start (CAGR 0.1522 vs 0.1622, Sharpe "
                "1.297 vs 1.369), the one start date where the 3.2-year window is all there is.",
                "the point-in-time sizes are not monotonic: top 15 (0.0997 / 0.834) and top 20 "
                "(0.1020 / 0.888) are no better than top 30 (0.0990 / 0.857), and top 15 is "
                "worse on drawdown. A smooth 'fewer coins is better' curve does not exist; there "
                "is a step at 10 and nothing in between, which is what noise in a 3-slot, "
                "kill-switched, path-dependent portfolio looks like.",
                "top 10's random-coin p-value is 0.084, against 0.002 for top 30. On a 10-coin "
                "universe the ranker is no longer distinguishable from drawing a random eligible "
                "coin at the same moments, so whatever top 10 earns is the universe's doing and "
                "not the score's. That is an argument about the strategy's edge, not for it.",
                "the pit numbers here are a fresh offline recomputation on the 2026-10-04 cache, "
                "not the published 2026-09-27 run. The N=30 control reproduces to 0.0990 / 0.857 "
                "/ -0.1713 against the published 0.1018 / 0.880 / -0.1713, so the harness is "
                "sound, but every pit figure in this file is 20-30 bp of CAGR below its published "
                "twin for that reason. All four sizes share one snapshot, which is the point.",
                "the standard panel ends 2026-09-24 because data/klines has not been refreshed "
                "since; only the pit cache is current to 2026-10-04.",
            ],
        },
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2, default=str))
    print(f"\nwrote {OUT}")
    print(f"variants tested: {doc['variants_tested']}  "
          f"any variant ships: {doc['beats_baseline_on_both_panels']}")
    return doc


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pit":
        run_pit(int(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == "pitextra":
        run_pit_extra(int(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == "standard":
        run_standard()
    else:
        main()
