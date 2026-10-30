"""ADVERSARIAL VERIFICATION, robustness lens: the N=15 narrowed-universe "win".

The claim under test: on the pinned 19-coin parity panel, narrowing eligibility each bar to the
top 15 names by trailing 30-day dollar volume passes the ship rule on both the full panel
(CAGR 0.2121 / Sharpe 1.4507 / maxDD -0.1115) and the held-out window from 2022-01-01
(0.1636 / 1.2688 / -0.1115).

This script does not re-litigate the mechanism. It attacks ROBUSTNESS only:
  1  reproduce the baseline and the variant exactly (otherwise nothing else means anything)
  2  split the history: 2017-08..2021-12 and 2022-01..end, each judged on its own
  3  split the variant's own TRADE LIST in half chronologically and judge each half's window
  4  count trades (per window, and in the differing set) - thin cells cannot support the claim
  5  remove the single best trade, two ways:
       a) its share of total P&L, and of the baseline-vs-variant P&L gap
       b) an in-engine counterfactual: make that symbol ineligible at the one decision bar that
          triggered the entry, so the slot goes to the next candidate and the path re-runs
  6  double the costs (slippage 7->14 bps, fee 0.10%->0.20% per side) and re-judge both panels
  7  bonus, the check the author admitted skipping: re-run the N grid with a 7-day volume window
     instead of 30-day. If the "best N" notch moves, the grid point is noise.

Nothing outside research/ and reports/ is written. No app/ file is modified.
"""

import copy
import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\amos2\Downloads\New folder (3)\crypto-bot")
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from app import backtest, config, features  # noqa: E402
from app.flags import Flags  # noqa: E402

SHIPPED = Flags(abs_momentum=True, volume_min=0.8, taker_min=0.5)
HELD_OUT = "2022-01-01"
BEST_N = 15
OUT = ROOT / "reports" / "explore_universe_size_robustness.json"

CLAIM_FULL = {"cagr": 0.2121, "sharpe": 1.4507, "max_drawdown": -0.1115}
CLAIM_HELD = {"cagr": 0.1636, "sharpe": 1.2688, "max_drawdown": -0.1115}
# The brief's baseline (which the author showed is really the 2-filter run) and the author's
# stricter 3-filter baseline. Both are checked.
BRIEF_BASE = {"cagr": 0.2071, "sharpe": 1.3712, "max_drawdown": -0.1509, "trades": 398}
BRIEF_BASE_HELD = {"cagr": 0.1473, "sharpe": 1.1004, "max_drawdown": -0.1509, "trades": 233}


def m_of(res):
    m = backtest.metrics(res.equity, res.trades, res.exposure)
    return {"cagr": round(float(m["cagr"]), 4), "sharpe": round(float(m["sharpe"]), 4),
            "max_drawdown": round(float(m["max_drawdown"]), 4),
            "trades": int(m.get("trades", 0)),
            "total_return": round(float(m["total_return"]), 4),
            "win_rate": round(float(m.get("win_rate", float("nan"))), 4),
            "profit_factor": round(float(m.get("profit_factor", float("nan"))), 3),
            "start": str(m["start"]), "end": str(m["end"])}


def beats(cand, base):
    """ship rule on one window: CAGR >= base AND Sharpe >= base AND drawdown no worse."""
    return bool(cand["cagr"] >= base["cagr"] and cand["sharpe"] >= base["sharpe"]
                and cand["max_drawdown"] >= base["max_drawdown"])


def why_fail(cand, base):
    bad = []
    if cand["cagr"] < base["cagr"]:
        bad.append(f"CAGR {cand['cagr']:.4f}<{base['cagr']:.4f}")
    if cand["sharpe"] < base["sharpe"]:
        bad.append(f"Sharpe {cand['sharpe']:.4f}<{base['sharpe']:.4f}")
    if cand["max_drawdown"] < base["max_drawdown"]:
        bad.append(f"maxDD {cand['max_drawdown']:.4f}<{base['max_drawdown']:.4f}")
    return "; ".join(bad) or "passes"


def narrow(panel, top_n, vol_col="dollar_vol_30d"):
    """The author's narrowing, copied verbatim except for the volume column being a parameter.

    Ranked WITHIN each bar_time on a closed-bar quantity; BTC always kept; score/rank rebuilt.
    """
    p = panel.copy()
    r = p[p["eligible"]].groupby("bar_time")[vol_col].rank(ascending=False, method="first")
    keep = pd.Series(False, index=p.index)
    keep.loc[r.index] = (r <= top_n).to_numpy()
    p["eligible"] = p["eligible"] & (keep | (p["symbol"] == "BTCUSDT"))
    return features.score(p.drop(columns=["score", "rank", "n_eligible"]))


def line(tag, base, var):
    return (f"  {tag:<34} base {base['cagr']:+.4f}/{base['sharpe']:+.3f}/"
            f"{base['max_drawdown']:+.4f}/tr{base['trades']:<4} | "
            f"var {var['cagr']:+.4f}/{var['sharpe']:+.3f}/{var['max_drawdown']:+.4f}/"
            f"tr{var['trades']:<4} | {'PASS' if beats(var, base) else 'FAIL'} "
            f"({why_fail(var, base)})")


def main():
    s = config.load()
    u = json.loads((ROOT / "reports" / "_parity_universe.json").read_text())
    print(f"universe: {len(u['symbols'])} symbols from reports/_parity_universe.json", flush=True)
    panel = features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars)
    pv = narrow(panel, BEST_N)
    out = {"lens": "robustness", "variant": f"standard panel, eligibility narrowed to top "
           f"{BEST_N} of 19 by trailing 30d dollar volume each bar, 3 slots"}

    # --------------------------------------------------------------- 1) reproduce
    print("\n=== 1) reproduction ===", flush=True)
    bres, vres = backtest.run(panel, s, flags=SHIPPED), backtest.run(pv, s, flags=SHIPPED)
    bf, vf = m_of(bres), m_of(vres)
    bh = m_of(backtest.run(panel, s, flags=SHIPPED, start=HELD_OUT))
    vh = m_of(backtest.run(pv, s, flags=SHIPPED, start=HELD_OUT))
    two = m_of(backtest.run(panel, s, flags=Flags(abs_momentum=True, volume_min=0.8)))
    out["repro"] = {"baseline_3filter_full": bf, "baseline_3filter_held": bh,
                    "variant_full": vf, "variant_held": vh,
                    "claimed_variant_full": CLAIM_FULL, "claimed_variant_held": CLAIM_HELD,
                    "brief_baseline_is_2filter_run": two, "brief_baseline_quoted": BRIEF_BASE,
                    "variant_full_matches_claim":
                        all(abs(vf[k] - CLAIM_FULL[k]) < 2e-4 for k in CLAIM_FULL),
                    "variant_held_matches_claim":
                        all(abs(vh[k] - CLAIM_HELD[k]) < 2e-4 for k in CLAIM_HELD),
                    "brief_baseline_is_really_2filter":
                        all(abs(two[k] - BRIEF_BASE[k]) < 2e-4 for k in
                            ("cagr", "sharpe", "max_drawdown"))}
    print(line("FULL  (variant vs 3-filter base)", bf, vf), flush=True)
    print(line("HELD  (variant vs 3-filter base)", bh, vh), flush=True)
    print(f"  variant reproduces claim? full={out['repro']['variant_full_matches_claim']} "
          f"held={out['repro']['variant_held_matches_claim']}", flush=True)
    print(f"  brief's 'baseline' is really the 2-filter run? "
          f"{out['repro']['brief_baseline_is_really_2filter']} "
          f"({two['cagr']:.4f}/{two['sharpe']:.4f}/{two['max_drawdown']:.4f})", flush=True)
    print(f"  edge over real baseline: dCAGR {vf['cagr'] - bf['cagr']:+.4f} "
          f"dSharpe {vf['sharpe'] - bf['sharpe']:+.4f} "
          f"(held {vh['cagr'] - bh['cagr']:+.4f} / {vh['sharpe'] - bh['sharpe']:+.4f})",
          flush=True)

    # --------------------------------------------------------------- 2) halves of the history
    print("\n=== 2) sub-periods, each judged on its own ===", flush=True)
    out["subperiods"] = []
    for a, b in (("2017-08-17", "2021-12-31"), ("2022-01-01", "2026-09-24")):
        bm = m_of(backtest.run(panel, s, flags=SHIPPED, start=a, end=b))
        vm = m_of(backtest.run(pv, s, flags=SHIPPED, start=a, end=b))
        out["subperiods"].append({"from": a, "to": b, "baseline": bm, "variant": vm,
                                  "variant_passes": beats(vm, bm), "why": why_fail(vm, bm)})
        print(line(f"{a[:7]}..{b[:7]}", bm, vm), flush=True)

    # --------------------------------------------------------------- 3) halves of the TRADE LIST
    print("\n=== 3) halves of the variant's own trade list ===", flush=True)
    vt = vres.trades.sort_values("entry_time").reset_index(drop=True)
    mid = len(vt) // 2
    split = pd.Timestamp(vt.loc[mid, "entry_time"])
    h1 = vt.iloc[:mid]
    h2 = vt.iloc[mid:]
    out["trade_halves"] = {
        "n_trades_variant": int(len(vt)), "split_at": str(split),
        "first_half": {"n": int(len(h1)), "avg_ret": round(float(h1.ret.mean()), 4),
                       "sum_pnl": round(float(h1.pnl.sum()), 1),
                       "win_rate": round(float((h1.pnl > 0).mean()), 4)},
        "second_half": {"n": int(len(h2)), "avg_ret": round(float(h2.ret.mean()), 4),
                        "sum_pnl": round(float(h2.pnl.sum()), 1),
                        "win_rate": round(float((h2.pnl > 0).mean()), 4)}}
    print(f"  variant trades {len(vt)}, split at {split.date()}", flush=True)
    for k in ("first_half", "second_half"):
        d = out["trade_halves"][k]
        print(f"  {k:<12} n={d['n']:<4} avg_ret {d['avg_ret']:+.4f} "
              f"sum_pnl {d['sum_pnl']:+.1f} win {d['win_rate']:.3f}", flush=True)
    # and the same split as backtest windows, so the ship rule can be applied to each half
    out["trade_halves"]["windows"] = []
    for a, b in ((str(vt.loc[0, "entry_time"])[:10], str(split)[:10]),
                 (str(split)[:10], "2026-09-24")):
        bm = m_of(backtest.run(panel, s, flags=SHIPPED, start=a, end=b))
        vm = m_of(backtest.run(pv, s, flags=SHIPPED, start=a, end=b))
        out["trade_halves"]["windows"].append({"from": a, "to": b, "baseline": bm, "variant": vm,
                                              "variant_passes": beats(vm, bm),
                                              "why": why_fail(vm, bm)})
        print(line(f"trade-half window {a}..{b}", bm, vm), flush=True)

    # --------------------------------------------------------------- 4) where the edge lives
    print("\n=== 4) trade-level diff and concentration ===", flush=True)
    bt = bres.trades.sort_values("entry_time").reset_index(drop=True)

    def keyset(t):
        return set(zip(t["symbol"], t["entry_time"]))
    bk, vk = keyset(bt), keyset(vt)
    only_b = bt[[k not in vk for k in zip(bt["symbol"], bt["entry_time"])]]
    only_v = vt[[k not in bk for k in zip(vt["symbol"], vt["entry_time"])]]
    out["diff"] = {"n_baseline_trades": int(len(bt)), "n_variant_trades": int(len(vt)),
                   "n_only_baseline": int(len(only_b)), "n_only_variant": int(len(only_v)),
                   "pnl_only_baseline": round(float(only_b.pnl.sum()), 1),
                   "pnl_only_variant": round(float(only_v.pnl.sum()), 1),
                   "first_differing_entry": str(min(list(only_b.entry_time)
                                                    + list(only_v.entry_time))),
                   "identical_trades": int(len(bk & vk))}
    print(f"  {len(only_b)} dropped / {len(only_v)} added, {len(bk & vk)} identical; "
          f"first difference {out['diff']['first_differing_entry'][:10]}", flush=True)
    print(f"  P&L of dropped {only_b.pnl.sum():+.1f} vs added {only_v.pnl.sum():+.1f}", flush=True)

    # --------------------------------------------------------------- 5) kill the best trade
    print("\n=== 5) remove the single best trade ===", flush=True)
    best = vt.loc[vt.pnl.idxmax()]
    top5 = vt.nlargest(5, "pnl")
    gross_win = float(vt.loc[vt.pnl > 0, "pnl"].sum())
    out["best_trade"] = {
        "symbol": str(best.symbol), "entry_time": str(best.entry_time),
        "exit_time": str(best.exit_time), "ret": round(float(best.ret), 4),
        "pnl": round(float(best.pnl), 1),
        "share_of_net_pnl": round(float(best.pnl / vt.pnl.sum()), 4),
        "share_of_gross_wins": round(float(best.pnl / gross_win), 4),
        "top5_share_of_net_pnl": round(float(top5.pnl.sum() / vt.pnl.sum()), 4),
        "in_baseline_too": bool((best.symbol, best.entry_time) in bk),
        "pnl_gap_variant_minus_baseline": round(float(vt.pnl.sum() - bt.pnl.sum()), 1)}
    print(f"  best trade {best.symbol} {str(best.entry_time)[:10]} ret {best.ret:+.3f} "
          f"pnl {best.pnl:+.1f} = {out['best_trade']['share_of_net_pnl']:.1%} of net P&L; "
          f"also in baseline: {out['best_trade']['in_baseline_too']}", flush=True)
    print(f"  top 5 trades = {out['best_trade']['top5_share_of_net_pnl']:.1%} of net P&L; "
          f"variant-minus-baseline P&L gap "
          f"{out['best_trade']['pnl_gap_variant_minus_baseline']:+.1f}", flush=True)

    # in-engine counterfactual: block that symbol at the decision bar that triggered the entry
    bars = sorted(panel["bar_time"].unique())
    idx = bars.index(pd.Timestamp(best.entry_time))
    dec_bar = bars[idx - 1]

    def block(p, sym, bar):
        q = p.copy()
        q.loc[(q["symbol"] == sym) & (q["bar_time"] == bar), "eligible"] = False
        return features.score(q.drop(columns=["score", "rank", "n_eligible"]))
    vf_nb = m_of(backtest.run(block(pv, best.symbol, dec_bar), s, flags=SHIPPED))
    # same surgery on the baseline, so the comparison stays apples to apples
    bbest = bt.loc[bt.pnl.idxmax()]
    bdec = bars[bars.index(pd.Timestamp(bbest.entry_time)) - 1]
    bf_nb = m_of(backtest.run(block(panel, bbest.symbol, bdec), s, flags=SHIPPED))
    out["best_trade"]["counterfactual"] = {
        "decision_bar_blocked": str(dec_bar),
        "variant_without_its_best": vf_nb, "baseline_without_its_best": bf_nb,
        "variant_still_passes": beats(vf_nb, bf_nb), "why": why_fail(vf_nb, bf_nb),
        "variant_without_best_vs_untouched_baseline": beats(vf_nb, bf),
        "why_vs_untouched": why_fail(vf_nb, bf),
        "baseline_best_trade": {"symbol": str(bbest.symbol),
                                "entry_time": str(bbest.entry_time),
                                "pnl": round(float(bbest.pnl), 1)}}
    print(line("drop-best: var vs base(drop-best)", bf_nb, vf_nb), flush=True)
    print(line("drop-best: var vs base(untouched)", bf, vf_nb), flush=True)

    # --------------------------------------------------------------- 6) cost stress
    print("\n=== 6) doubled costs (14 bps slippage, 0.20% fee per side) ===", flush=True)
    s2 = copy.copy(s)
    s2.slippage_bps = 14.0
    s2.fee_rate = 0.002
    out["cost_stress"] = {"slippage_bps": 14.0, "fee_rate": 0.002, "windows": []}
    for tag, kw in (("full", {}), ("held_out", {"start": HELD_OUT})):
        bm = m_of(backtest.run(panel, s2, flags=SHIPPED, **kw))
        vm = m_of(backtest.run(pv, s2, flags=SHIPPED, **kw))
        out["cost_stress"]["windows"].append({"window": tag, "baseline": bm, "variant": vm,
                                              "variant_passes": beats(vm, bm),
                                              "why": why_fail(vm, bm)})
        print(line(f"2x costs {tag}", bm, vm), flush=True)
    # 1.5x as an intermediate point
    s15 = copy.copy(s)
    s15.slippage_bps = 10.5
    s15.fee_rate = 0.0015
    out["cost_stress"]["x1_5"] = []
    for tag, kw in (("full", {}), ("held_out", {"start": HELD_OUT})):
        bm = m_of(backtest.run(panel, s15, flags=SHIPPED, **kw))
        vm = m_of(backtest.run(pv, s15, flags=SHIPPED, **kw))
        out["cost_stress"]["x1_5"].append({"window": tag, "baseline": bm, "variant": vm,
                                          "variant_passes": beats(vm, bm),
                                          "why": why_fail(vm, bm)})
        print(line(f"1.5x costs {tag}", bm, vm), flush=True)

    # --------------------------------------------------------------- 7) does the notch move?
    print("\n=== 7) same narrowing, 7-day volume window instead of 30-day ===", flush=True)
    pw = panel.copy()
    bpd = 6
    pw["dollar_vol_7d"] = (pw.sort_values(["symbol", "bar_time"])
                           .groupby("symbol")["quote_volume"]
                           .transform(lambda x: x.rolling(7 * bpd).sum() / 7))
    out["volume_window_7d"] = []
    for n in (13, 14, 15, 16, 17):
        p7 = narrow(pw, n, vol_col="dollar_vol_7d")
        f = m_of(backtest.run(p7, s, flags=SHIPPED))
        h = m_of(backtest.run(p7, s, flags=SHIPPED, start=HELD_OUT))
        row = {"universe": n, "full": f, "held_out": h,
               "passes": beats(f, bf) and beats(h, bh),
               "why_full": why_fail(f, bf), "why_held": why_fail(h, bh)}
        out["volume_window_7d"].append(row)
        print(f"  7d N={n:>2}: full {f['cagr']:+.4f}/{f['sharpe']:+.3f}/{f['max_drawdown']:+.4f} "
              f"held {h['cagr']:+.4f}/{h['sharpe']:+.3f}/{h['max_drawdown']:+.4f} "
              f"passes {row['passes']}", flush=True)

    # ------------------------------------------- 8) the ONLY window where the variant differs
    # 333 of 366 trades are identical and the first difference is late 2023, so the full panel
    # and the held-out panel are not two tests - they share the same identical prefix. The real
    # test is the binding window on its own, and halves of it.
    print("\n=== 8) the binding window only (first difference onward) ===", flush=True)
    first_diff = str(out["diff"]["first_differing_entry"])[:10]
    out["binding_window"] = {"starts": first_diff, "windows": [], "start_sweep": []}
    bd_half = "2025-03-01"
    for a, b in ((first_diff, "2026-09-24"), (first_diff, bd_half), (bd_half, "2026-09-24")):
        bm = m_of(backtest.run(panel, s, flags=SHIPPED, start=a, end=b))
        vm = m_of(backtest.run(pv, s, flags=SHIPPED, start=a, end=b))
        out["binding_window"]["windows"].append({"from": a, "to": b, "baseline": bm,
                                                "variant": vm, "variant_passes": beats(vm, bm),
                                                "why": why_fail(vm, bm)})
        print(line(f"binding {a}..{b}", bm, vm), flush=True)
    for a in ("2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01"):
        bm = m_of(backtest.run(panel, s, flags=SHIPPED, start=a))
        vm = m_of(backtest.run(pv, s, flags=SHIPPED, start=a))
        out["binding_window"]["start_sweep"].append({"start": a, "baseline": bm, "variant": vm,
                                                    "variant_passes": beats(vm, bm),
                                                    "why": why_fail(vm, bm)})
        print(line(f"from {a}", bm, vm), flush=True)

    # 9) how concentrated is the +P&L gap inside the 62 differing trades?
    print("\n=== 9) concentration inside the differing trades ===", flush=True)
    add = only_v.sort_values("pnl", ascending=False)
    drop = only_b.sort_values("pnl", ascending=False)
    gap = float(vt.pnl.sum() - bt.pnl.sum())
    out["diff_concentration"] = {
        "net_pnl_gap": round(gap, 1),
        "top_added": [{"symbol": r.symbol, "entry": str(r.entry_time)[:10],
                       "ret": round(float(r.ret), 4), "pnl": round(float(r.pnl), 1)}
                      for r in add.head(6).itertuples()],
        "top_dropped": [{"symbol": r.symbol, "entry": str(r.entry_time)[:10],
                         "ret": round(float(r.ret), 4), "pnl": round(float(r.pnl), 1)}
                        for r in drop.head(6).itertuples()],
        "best_added_pnl_as_share_of_gap": round(float(add.pnl.iloc[0] / gap), 3)
        if gap != 0 else None,
        "top3_added_pnl_as_share_of_gap": round(float(add.pnl.head(3).sum() / gap), 3)
        if gap != 0 else None}
    for r in out["diff_concentration"]["top_added"][:4]:
        print(f"    added   {r['symbol']:<10} {r['entry']} ret {r['ret']:+.3f} "
              f"pnl {r['pnl']:+.1f}", flush=True)
    for r in out["diff_concentration"]["top_dropped"][:4]:
        print(f"    dropped {r['symbol']:<10} {r['entry']} ret {r['ret']:+.3f} "
              f"pnl {r['pnl']:+.1f}", flush=True)
    print(f"  net gap {gap:+.1f}; best single ADDED trade is "
          f"{out['diff_concentration']['best_added_pnl_as_share_of_gap']:.2f}x the whole gap, "
          f"top 3 added = {out['diff_concentration']['top3_added_pnl_as_share_of_gap']:.2f}x",
          flush=True)

    # in-engine: deny the variant its single best ADDED trade (the one the baseline never took)
    ba = add.iloc[0]
    bdec2 = bars[bars.index(pd.Timestamp(ba.entry_time)) - 1]
    vf_na = m_of(backtest.run(block(pv, ba.symbol, bdec2), s, flags=SHIPPED))
    vh_na = m_of(backtest.run(block(pv, ba.symbol, bdec2), s, flags=SHIPPED, start=HELD_OUT))
    out["diff_concentration"]["without_best_added"] = {
        "blocked": f"{ba.symbol} decision bar {bdec2}",
        "variant_full": vf_na, "variant_held": vh_na,
        "passes_full": beats(vf_na, bf), "why_full": why_fail(vf_na, bf),
        "passes_held": beats(vh_na, bh), "why_held": why_fail(vh_na, bh)}
    print(line("no best-added: full vs base", bf, vf_na), flush=True)
    print(line("no best-added: held vs base", bh, vh_na), flush=True)

    # 10) micro-perturbation: lag the volume ranking by one bar. If the notch is a mechanism this
    # is a no-op; if it is a lucky alignment of 62 trades it will move.
    print("\n=== 10) one-bar lag on the volume ranking ===", flush=True)
    pl = panel.copy()
    pl["dv_lag"] = (pl.sort_values(["symbol", "bar_time"]).groupby("symbol")["dollar_vol_30d"]
                    .shift(1))
    out["lagged_volume"] = []
    for n in (14, 15, 16):
        pn = narrow(pl, n, vol_col="dv_lag")
        f = m_of(backtest.run(pn, s, flags=SHIPPED))
        h = m_of(backtest.run(pn, s, flags=SHIPPED, start=HELD_OUT))
        row = {"universe": n, "full": f, "held_out": h,
               "passes": beats(f, bf) and beats(h, bh),
               "why_full": why_fail(f, bf), "why_held": why_fail(h, bh)}
        out["lagged_volume"].append(row)
        print(f"  lag1 N={n}: full {f['cagr']:+.4f}/{f['sharpe']:+.3f}/{f['max_drawdown']:+.4f} "
              f"held {h['cagr']:+.4f}/{h['sharpe']:+.3f}/{h['max_drawdown']:+.4f} "
              f"passes {row['passes']}", flush=True)

    # 11) attribution: when was the whole edge actually earned?
    # The relative equity curve (variant / baseline, both normalised to 1.0) is the cleanest
    # possible statement of the claim: a real edge accumulates, a lucky episode is a step.
    print("\n=== 11) attribution: relative equity and monthly diff P&L ===", flush=True)
    rel = (vres.equity / vres.equity.iloc[0]) / (bres.equity / bres.equity.iloc[0])
    yr = rel.resample("YE").last()
    ob, ov = only_b.copy(), only_v.copy()
    ob["m"] = pd.to_datetime(ob.entry_time).dt.tz_localize(None).dt.to_period("M").astype(str)
    ov["m"] = pd.to_datetime(ov.entry_time).dt.tz_localize(None).dt.to_period("M").astype(str)
    g = pd.DataFrame({"added_n": ov.groupby("m").size(), "added_pnl": ov.groupby("m").pnl.sum(),
                      "dropped_n": ob.groupby("m").size(),
                      "dropped_pnl": ob.groupby("m").pnl.sum()}).fillna(0)
    g["net"] = g.added_pnl - g.dropped_pnl
    nov = g.loc[[i for i in g.index if i in ("2024-11", "2024-12")]]
    out["attribution"] = {
        "relative_equity_year_end": {str(k.year): round(float(v) - 1, 4)
                                     for k, v in yr.items()},
        "relative_equity_peak": round(float(rel.max()), 4),
        "relative_equity_peak_date": str(rel.idxmax().date()),
        "relative_equity_final": round(float(rel.iloc[-1]), 4),
        "years_bit_identical": float((rel == 1.0).mean()),
        "monthly_diff_pnl": {i: {k: round(float(r[k]), 1) for k in
                                 ("added_n", "added_pnl", "dropped_n", "dropped_pnl", "net")}
                             for i, r in g.iterrows()},
        "nov_dec_2024_share_of_diff_net_pnl": round(float(nov.net.sum() / g.net.sum()), 3),
        "nov_dec_2024_differing_trades": int(nov.added_n.sum() + nov.dropped_n.sum()),
        "total_differing_trades": int(len(ob) + len(ov)),
        "implied_cagr_from_a_one_off_step":
            round(float(rel.iloc[-1] ** (1 / 9.1) - 1), 4)}
    print(f"  relative equity: identical for {(rel == 1.0).mean():.1%} of all bars; "
          f"year ends " + ", ".join(f"{k.year}:{v - 1:+.4f}" for k, v in yr.items()), flush=True)
    print(f"  peak {rel.max():.4f} on {rel.idxmax().date()}, final {rel.iloc[-1]:.4f} "
          f"(a one-off step of that size over 9.1y = "
          f"{out['attribution']['implied_cagr_from_a_one_off_step']:+.4f} CAGR, i.e. the whole "
          f"claimed +0.0048 edge)", flush=True)
    print(f"  Nov+Dec 2024 = {out['attribution']['nov_dec_2024_share_of_diff_net_pnl']:.0%} of "
          f"the differing-trade net P&L from "
          f"{out['attribution']['nov_dec_2024_differing_trades']} of "
          f"{out['attribution']['total_differing_trades']} differing trades", flush=True)

    out["verdict"] = {
        "refuted": True,
        "survived": ["exact reproduction of both claimed rows",
                     "2x cost stress (14bps slippage, 0.20% fee): variant still passes on both "
                     "windows and the gap widens slightly",
                     "1.5x cost stress",
                     "removal of the single best trade (SOLUSDT 2023-10, which both runs take)",
                     "a one-bar lag on the volume ranking (N=15 unchanged)"],
        "killed_by": [
            "5.4 of the 9.1 years are BIT-IDENTICAL to the baseline (relative equity exactly "
            "1.0000 until 2023-10-02), so the full panel and the held-out panel are not two "
            "tests: the held-out window's first 21 months are identical too",
            "the entire cumulative edge is a +3.65% relative level step, and 78% of the "
            "differing-trade net P&L is earned in Nov-Dec 2024 by 15 of 62 differing trades",
            "the second half of the only window where the rule binds FAILS all three metrics "
            "(2025-03..2026-09: 0.1844/1.519/-0.1046 vs baseline 0.1957/1.597/-0.1034)",
            "relative equity peaked at +11.2% on 2025-09-21 and has decayed to +3.65%: the "
            "edge has been reversing for the last 12 months",
            "fails the ship rule from 2024-07, 2025-01 and 2025-07 starts",
            "the diff is a near-wash of opposite piles (dropped trades +3022 P&L, added +4514, "
            "net +1866 on ~43k); the top 3 added trades alone are 1.85x the whole gap"]}

    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
