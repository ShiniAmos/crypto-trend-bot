"""Pure rotation: is the 20-bar breakout trigger an edge, or a tax?

HYPOTHESIS
    The shipped strategy only buys a top-ranked coin on the bar its close clears the prior
    20-bar high. Classic cross-sectional momentum does not wait for a price trigger at all: it
    simply HOLDS the top N ranked names and rebalances. If the trigger is a tax (it delays every
    entry and skips names that never print a fresh high while still ranking well), removing it
    should raise the return.

HOW IT IS EXPRESSED
    The engine is not touched. backtest.run reads the `breakout` column for entry and the
    `breakdown` column for the price exit, so a copy of the panel with

        panel2["breakout"] = True        -> entry reduces to "regime on, ranked top N, filters"
        panel2["breakdown"] = False      -> the 20-bar-low price exit is removed as well

    expresses the rotation exactly, with identical costs, fills, slot logic and risk limits.
    Setting breakout to a constant True can only REMOVE information from the decision, so this
    construction cannot introduce lookahead. Nothing else about the panel is changed: the sizing
    reference `stop_level` is still the 20-bar low including the decision bar (a closed bar), and
    the ranks are still the per-bar_time cross-sectional ranks built by features.score.

VARIANTS TESTED
    candidates (entry rank cut)   : 3, 5
    exit_rank  (rank-drop exit)   : 5, 10, 15
    price exit                    : keep the 20-bar low (the honest middle ground) / remove it
    plus, as an extra middle ground, rotation with the intrabar hard stop at the ratcheting
    20-bar low (flags.stop_exec), which is the strictest reading of "still uses the 20-bar low".
    Each rotation arm is paired with the SAME-PARAMETER breakout strategy so the breakout effect
    is isolated rather than confounded with the rank parameters.

SHIP RULE
    A variant beats the baseline only if CAGR >= baseline AND Sharpe >= baseline AND max drawdown
    is not worse, on BOTH the standard panel and the held-out period from 2022-01-01.

Run:  cd <repo> && PYTHONIOENCODING=utf-8 python research/explore_rotation.py
"""

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from app import backtest, config, features  # noqa: E402
from app.flags import Flags  # noqa: E402

UNIVERSE = ROOT / "reports" / "_parity_universe.json"
OUT = ROOT / "reports" / "explore_rotation.json"
HELD_START = "2022-01-01"

# The shipped flag set. NOTE on the baseline reference numbers: the task brief quotes
#   standard CAGR 0.2071 / Sharpe 1.3712 / maxDD -0.1509 / 398 trades
# as belonging to abs_momentum + volume_min=0.8 + taker_min=0.5. Measured here, those exact
# numbers belong to abs_momentum + volume_min=0.8 ONLY (they are the "standard baseline" row in
# RESEARCH_LOG.md's P1-F section). Adding taker_min=0.5 gives 0.2073 / 1.4295 / -0.1324 / 366.
# So there are two defensible bars. Every variant below is run with the full SHIPPED set, so the
# apples-to-apples bar is the measured one; the quoted one is the softer of the two and is also
# checked. A variant must clear a bar on CAGR, Sharpe and maxDD on BOTH panels to count.
SHIPPED = dict(abs_momentum=True, volume_min=0.8, taker_min=0.5)

# Softer bar: the numbers quoted in the brief (= abs_momentum + volume_min 0.8, measured).
BASELINE = {
    "standard": {"cagr": 0.2071, "sharpe": 1.3712, "max_drawdown": -0.1509, "trades": 398},
    "heldout": {"cagr": 0.1473, "sharpe": 1.1004, "max_drawdown": -0.1509, "trades": 233},
}
# Harder bar: the same flags the variants run with, measured in this script's own parity run.
BASELINE_SHIPPED = {
    "standard": {"cagr": 0.2073, "sharpe": 1.4295, "max_drawdown": -0.1324, "trades": 366},
    "heldout": {"cagr": 0.1548, "sharpe": 1.2229, "max_drawdown": -0.1324, "trades": 210},
}

KEEP = ["cagr", "sharpe", "max_drawdown", "calmar", "total_return", "trades", "win_rate",
        "avg_trade", "median_trade", "profit_factor", "avg_bars_held", "avg_exposure"]


def settings_with(base, **kw):
    """A shallow copy of Settings with fields overridden (no __post_init__ re-run, no mkdir)."""
    s = copy.copy(base)
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def rotation_panel(panel, drop_price_exit: bool):
    """A copy of the panel with the breakout trigger removed, optionally the price exit too."""
    p = panel.copy()
    p["breakout"] = True
    if drop_price_exit:
        p["breakdown"] = False
    return p


def measure(panel, settings, flags, start=None):
    res = backtest.run(panel, settings, flags=flags, start=start)
    m = backtest.metrics(res.equity, res.trades, res.exposure)
    out = {k: (float(m[k]) if isinstance(m.get(k), (int, float)) else m.get(k))
           for k in KEEP if k in m}
    out.setdefault("trades", 0)
    return out, res


def verdict(std, held, bar=BASELINE):
    """Ship rule: CAGR, Sharpe and maxDD must all be at least as good on BOTH panels."""
    checks = {}
    for name, m, b in (("standard", std, bar["standard"]), ("heldout", held, bar["heldout"])):
        checks[name] = {
            "d_cagr": round(m["cagr"] - b["cagr"], 4),
            "d_sharpe": round(m["sharpe"] - b["sharpe"], 4),
            "d_maxdd": round(m["max_drawdown"] - b["max_drawdown"], 4),
            "pass": bool(m["cagr"] >= b["cagr"] and m["sharpe"] >= b["sharpe"]
                         and m["max_drawdown"] >= b["max_drawdown"]),
        }
    return bool(checks["standard"]["pass"] and checks["heldout"]["pass"]), checks


def sanity(res, label):
    """Catch an obviously broken run before anyone reads its CAGR."""
    notes = []
    eq = res.equity
    if eq.isna().any():
        notes.append("NaN in the equity curve")
    if not len(res.trades):
        notes.append("zero trades")
    else:
        t = res.trades
        if (t["entry_price"] <= 0).any() or (t["exit_price"] <= 0).any():
            notes.append("non-positive fill price")
        # Same-bar exit_time == entry_time is legitimate under flags.stop_exec (the engine
        # documents that a position bought at an open can be stopped inside that same bar).
        # An exit strictly BEFORE its entry would be a real bug.
        if (t["exit_time"] < t["entry_time"]).any():
            notes.append("an exit before its own entry")
    buys = res.log[res.log["side"] == "BUY"] if len(res.log) else res.log
    if len(buys):
        # stop_level must sit below the entry fill, or the sizing formula is nonsense.
        bad = 0
        for row in buys.itertuples():
            txt = str(row.reason)
            if "stop " in txt:
                try:
                    stop = float(txt.rsplit("stop ", 1)[1])
                except ValueError:
                    continue
                if stop >= row.price:
                    bad += 1
        if bad:
            notes.append(f"{bad} entries sized off a stop at or above the entry price")
    if notes:
        print(f"    !! {label}: " + "; ".join(notes))
    return notes


def main():
    s = config.load()
    u = json.loads(UNIVERSE.read_text())
    print(f"universe: {len(u['symbols'])} symbols from {UNIVERSE.name}")
    panel = features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars)
    print(f"panel: {len(panel):,} rows, {panel['bar_time'].min().date()} .. "
          f"{panel['bar_time'].max().date()}")

    rot_keep = rotation_panel(panel, drop_price_exit=False)
    rot_nostop = rotation_panel(panel, drop_price_exit=True)
    panels = {"breakout": panel, "rotation": rot_keep, "rotation_no_price_exit": rot_nostop}

    rows = []

    def run_variant(name, panel_key, candidates, exit_rank, extra_flags=None, risk_pct=None,
                    collect=True):
        over = {"candidates": candidates}
        if risk_pct is not None:
            over["risk_pct"] = risk_pct
        st = settings_with(s, **over)
        fl = Flags(**SHIPPED, exit_rank=exit_rank, **(extra_flags or {}))
        std, res_std = measure(panels[panel_key], st, fl)
        held, res_held = measure(panels[panel_key], st, fl, start=HELD_START)
        notes = sanity(res_std, name) + sanity(res_held, name + " (held-out)")
        beats, checks = verdict(std, held, BASELINE)
        beats_hard, checks_hard = verdict(std, held, BASELINE_SHIPPED)
        row = {"name": name, "panel": panel_key, "candidates": candidates,
               "exit_rank": exit_rank, "extra_flags": extra_flags or {},
               "risk_pct": risk_pct if risk_pct is not None else s.risk_pct,
               "standard": std, "heldout": held,
               "checks_vs_quoted_baseline": checks, "beats_quoted_baseline": beats,
               "checks_vs_shipped_flags_baseline": checks_hard,
               "beats_shipped_flags_baseline": beats_hard,
               "checks": checks, "beats_baseline": bool(beats and beats_hard),
               "sanity": notes}
        if collect:
            rows.append(row)
        tag = "  <-- BEATS BOTH BARS" if (beats and beats_hard) else (
            "  (clears quoted bar only)" if beats else "")
        print(f"  {name:<42} std CAGR {std['cagr']:+.4f} Sh {std['sharpe']:.4f} "
              f"DD {std['max_drawdown']:+.4f} n={std['trades']:>4} | "
              f"held CAGR {held['cagr']:+.4f} Sh {held['sharpe']:.4f} "
              f"DD {held['max_drawdown']:+.4f} n={held['trades']:>4}{tag}")
        return row

    print("\n0) parity: the shipped strategy as the repo ships it")
    base = run_variant("SHIPPED baseline (breakout, c=5, xr=10)", "breakout", 5, 0, collect=False)
    # The flag set the brief's quoted numbers actually belong to, measured here for the record.
    sf = Flags(abs_momentum=True, volume_min=0.8)
    soft_std, _ = measure(panel, s, sf)
    soft_held, _ = measure(panel, s, sf, start=HELD_START)
    print(f"  {'quoted-bar flags (abs+vol0.8, no taker)':<42} "
          f"std CAGR {soft_std['cagr']:+.4f} Sh {soft_std['sharpe']:.4f} "
          f"DD {soft_std['max_drawdown']:+.4f} n={soft_std['trades']:>4} | "
          f"held CAGR {soft_held['cagr']:+.4f} Sh {soft_held['sharpe']:.4f} "
          f"DD {soft_held['max_drawdown']:+.4f} n={soft_held['trades']:>4}")
    parity = {}
    for k, b in (("standard", BASELINE_SHIPPED["standard"]), ("heldout", BASELINE_SHIPPED["heldout"])):
        got = base[k]
        ok = (abs(got["cagr"] - b["cagr"]) < 1e-3 and abs(got["sharpe"] - b["sharpe"]) < 1e-3
              and abs(got["max_drawdown"] - b["max_drawdown"]) < 1e-3
              and got["trades"] == b["trades"])
        parity[k] = bool(ok)
        print(f"    parity {k} (shipped flags): {'OK' if ok else 'MISMATCH'}")

    print("\n1) same-parameter breakout controls (so the rank grid is not confounded)")
    for cand in (3, 5):
        for xr in (5, 10, 15):
            run_variant(f"breakout  c={cand} xr={xr}", "breakout", cand, xr)

    print("\n2) rotation, no breakout trigger, 20-bar-low price exit KEPT (middle ground)")
    for cand in (3, 5):
        for xr in (5, 10, 15):
            run_variant(f"rotation  c={cand} xr={xr} (low exit)", "rotation", cand, xr)

    print("\n3) rotation, no breakout trigger, no price exit (rank/regime only)")
    for cand in (3, 5):
        for xr in (5, 10, 15):
            run_variant(f"rotation  c={cand} xr={xr} (no low exit)", "rotation_no_price_exit",
                        cand, xr)

    print("\n4) rotation with the intrabar hard stop at the ratcheting 20-bar low")
    for cand in (3, 5):
        for xr in (10, 15):
            run_variant(f"rotation  c={cand} xr={xr} (+stop_exec)", "rotation", cand, xr,
                        {"stop_exec": True})
        run_variant(f"rotation  c={cand} xr=15 (no low exit, +stop_exec)",
                    "rotation_no_price_exit", cand, 15, {"stop_exec": True})

    # ---- 5) de-levered rotation: match the baseline's drawdown budget, then compare ------
    # The rotation's only failing metric is drawdown, so the fair follow-up question is whether
    # its extra CAGR survives being risk-scaled down to the baseline's drawdown. risk_pct scales
    # every position size (max_weight still caps it), so this is a pure leverage knob and changes
    # no decision. If the rotation held a real edge, a de-levered version should still out-earn
    # the baseline at the same drawdown.
    print("\n5) de-levered rotation (risk_pct turned down to match the baseline drawdown)")
    best_rot_arm = max((r for r in rows if r["panel"] != "breakout"),
                       key=lambda r: r["standard"]["cagr"])
    delev = []
    for rp in (0.01, 0.008, 0.007, 0.006, 0.005, 0.004):
        r = run_variant(f"rotation c={best_rot_arm['candidates']} "
                        f"xr={best_rot_arm['exit_rank']} risk={rp:.3f}",
                        best_rot_arm["panel"], best_rot_arm["candidates"],
                        best_rot_arm["exit_rank"], best_rot_arm["extra_flags"], risk_pct=rp,
                        collect=False)
        delev.append(r)
    # The de-levered arms that actually respect the baseline's drawdown on both panels.
    dd_ok = [r for r in delev
             if r["standard"]["max_drawdown"] >= BASELINE["standard"]["max_drawdown"]
             and r["heldout"]["max_drawdown"] >= BASELINE["heldout"]["max_drawdown"]]
    if dd_ok:
        b = max(dd_ok, key=lambda r: r["standard"]["cagr"])
        print(f"  best arm inside the baseline drawdown budget: {b['name']} "
              f"std CAGR {b['standard']['cagr']:+.4f} vs baseline "
              f"{BASELINE['standard']['cagr']:+.4f} "
              f"({b['standard']['cagr'] - BASELINE['standard']['cagr']:+.4f})")
        dd_match = {"name": b["name"], "risk_pct": b["risk_pct"],
                    "standard": b["standard"], "heldout": b["heldout"],
                    "d_cagr_vs_quoted_baseline": b["standard"]["cagr"] - BASELINE["standard"]["cagr"]}
    else:
        dd_match = None
        print("  no risk_pct down to 0.004 brought the rotation inside the baseline's drawdown "
              "on both panels")

    # Exposure-matched: the baseline spends ~9.5% of the time invested. The rotation is invested
    # more, which on its own raises both CAGR and drawdown. Picking the de-levered arm with the
    # same average exposure removes that confound entirely.
    base_expo = base["standard"]["avg_exposure"]
    em = min(delev, key=lambda r: abs(r["standard"]["avg_exposure"] - base_expo))
    expo_matched = {
        "name": em["name"], "risk_pct": em["risk_pct"],
        "baseline_avg_exposure": base_expo,
        "rotation_avg_exposure": em["standard"]["avg_exposure"],
        "standard": em["standard"], "heldout": em["heldout"],
        "d_cagr": em["standard"]["cagr"] - base["standard"]["cagr"],
        "d_sharpe": em["standard"]["sharpe"] - base["standard"]["sharpe"],
        "d_maxdd": em["standard"]["max_drawdown"] - base["standard"]["max_drawdown"],
    }
    print(f"  exposure-matched ({em['risk_pct']:.3f}): rotation expo "
          f"{em['standard']['avg_exposure']:.3f} vs baseline {base_expo:.3f}  ->  "
          f"CAGR {em['standard']['cagr']:+.4f} vs {base['standard']['cagr']:+.4f} "
          f"({expo_matched['d_cagr']:+.4f}), Sharpe {em['standard']['sharpe']:.4f} vs "
          f"{base['standard']['sharpe']:.4f} ({expo_matched['d_sharpe']:+.4f}), "
          f"DD {em['standard']['max_drawdown']:+.4f} vs "
          f"{base['standard']['max_drawdown']:+.4f} ({expo_matched['d_maxdd']:+.4f})")

    # Per-trade quality is the cleanest read on whether the trigger is a tax or an edge:
    # it is free of exposure, leverage and compounding.
    pt = {"baseline_avg_trade": base["standard"]["avg_trade"],
          "baseline_trades": base["standard"]["trades"],
          "baseline_win_rate": base["standard"]["win_rate"],
          "baseline_profit_factor": base["standard"]["profit_factor"],
          "best_rotation_avg_trade": best_rot_arm["standard"]["avg_trade"],
          "best_rotation_trades": best_rot_arm["standard"]["trades"],
          "best_rotation_win_rate": best_rot_arm["standard"]["win_rate"],
          "best_rotation_profit_factor": best_rot_arm["standard"]["profit_factor"]}
    print(f"  per-trade quality: baseline {pt['baseline_avg_trade']:+.4f} over "
          f"{pt['baseline_trades']:.0f} trades (win {pt['baseline_win_rate']:.1%}, "
          f"PF {pt['baseline_profit_factor']:.2f})  vs  best rotation "
          f"{pt['best_rotation_avg_trade']:+.4f} over {pt['best_rotation_trades']:.0f} "
          f"(win {pt['best_rotation_win_rate']:.1%}, PF {pt['best_rotation_profit_factor']:.2f})")

    # ---- pick the best variant honestly -------------------------------------------------
    rot = [r for r in rows if r["panel"] != "breakout"]
    winners = [r for r in rot if r["beats_baseline"]]
    # Ranking for the report: anything that ships wins; otherwise the one that comes closest
    # without being allowed to buy its CAGR with drawdown.
    def closeness(r):
        c = r["checks"]
        return min(c["standard"]["d_cagr"], c["heldout"]["d_cagr"])
    best = max(winners, key=closeness) if winners else max(rot, key=closeness)
    # Also report the best rotation arm by raw standard CAGR, for context.
    best_cagr = max(rot, key=lambda r: r["standard"]["cagr"])

    print("\n" + "=" * 100)
    print(f"variants tested (rotation arms): {len(rot)}   that beat the baseline: {len(winners)}")
    print(f"best by ship rule / closeness : {best['name']}")
    print(f"best rotation by raw std CAGR : {best_cagr['name']} "
          f"(std {best_cagr['standard']['cagr']:+.4f}, held {best_cagr['heldout']['cagr']:+.4f})")

    # Head-to-head at matched parameters: does removing the trigger help or hurt?
    print("\nbreakout vs rotation at matched (candidates, exit_rank), standard panel:")
    head = []
    for cand in (3, 5):
        for xr in (5, 10, 15):
            bo = next(r for r in rows if r["panel"] == "breakout" and r["candidates"] == cand
                      and r["exit_rank"] == xr)
            ro = next(r for r in rows if r["panel"] == "rotation" and r["candidates"] == cand
                      and r["exit_rank"] == xr and not r["extra_flags"])
            rn = next(r for r in rows if r["panel"] == "rotation_no_price_exit"
                      and r["candidates"] == cand and r["exit_rank"] == xr)
            d_keep = ro["standard"]["cagr"] - bo["standard"]["cagr"]
            d_none = rn["standard"]["cagr"] - bo["standard"]["cagr"]
            head.append({"candidates": cand, "exit_rank": xr,
                         "breakout_cagr": bo["standard"]["cagr"],
                         "rotation_lowexit_cagr": ro["standard"]["cagr"],
                         "rotation_nolowexit_cagr": rn["standard"]["cagr"],
                         "delta_lowexit": d_keep, "delta_nolowexit": d_none,
                         "breakout_heldout_cagr": bo["heldout"]["cagr"],
                         "rotation_lowexit_heldout_cagr": ro["heldout"]["cagr"],
                         "rotation_nolowexit_heldout_cagr": rn["heldout"]["cagr"]})
            print(f"  c={cand} xr={xr:<2}  breakout {bo['standard']['cagr']:+.4f}  "
                  f"rotation(low exit) {ro['standard']['cagr']:+.4f} ({d_keep:+.4f})  "
                  f"rotation(no low exit) {rn['standard']['cagr']:+.4f} ({d_none:+.4f})")
    mean_keep = sum(h["delta_lowexit"] for h in head) / len(head)
    mean_none = sum(h["delta_nolowexit"] for h in head) / len(head)
    print(f"  mean CAGR effect of removing the trigger: {mean_keep:+.4f} (low exit kept), "
          f"{mean_none:+.4f} (no low exit)")

    # ---- robustness on the best rotation arm --------------------------------------------
    print("\nrobustness on the best rotation arm")
    st = settings_with(s, candidates=best["candidates"])
    fl = Flags(**SHIPPED, exit_rank=best["exit_rank"], **best["extra_flags"])
    pk = panels[best["panel"]]
    res = backtest.run(pk, st, flags=fl)
    yr = backtest.yearly(res.equity)
    print("  yearly returns: " + ", ".join(f"{y}:{v:+.1%}" for y, v in yr.items()))
    extra = {"yearly": {int(y): float(v) for y, v in yr.items()}}
    # split-half: does the arm work in both halves, or is it one era?
    mid = res.equity.index[len(res.equity) // 2]
    for half, eq in (("first_half", res.equity.loc[:mid]), ("second_half", res.equity.loc[mid:])):
        mm = backtest.metrics(eq / eq.iloc[0] * s.start_equity)
        extra[half] = {"cagr": float(mm["cagr"]), "sharpe": float(mm["sharpe"]),
                       "max_drawdown": float(mm["max_drawdown"]),
                       "start": str(mm["start"]), "end": str(mm["end"])}
        print(f"  {half}: CAGR {mm['cagr']:+.4f} Sharpe {mm['sharpe']:.4f} "
              f"DD {mm['max_drawdown']:+.4f}")
    if len(res.trades):
        t = res.trades.sort_values("pnl", ascending=False)
        tot = float(res.trades["pnl"].sum())
        top1 = float(t["pnl"].iloc[0])
        extra["pnl_total"] = tot
        extra["top1_pnl_share"] = top1 / tot if tot else None
        extra["top5_pnl_share"] = float(t["pnl"].head(5).sum()) / tot if tot else None
        print(f"  biggest single trade is {extra['top1_pnl_share']:.1%} of total P&L; "
              f"top 5 are {extra['top5_pnl_share']:.1%}")
        # drop the single best trade and see what is left
        try:
            rb = backtest.random_baselines(pk, res.trades, st, n=300, seed=7)
            extra["random_baselines"] = {k: float(v) for k, v in rb.items()}
            print(f"  p(random coin) {rb['p_value_vs_random_coin']:.3f}  "
                  f"p(random time) {rb['p_value_vs_random_time']:.3f}  "
                  f"avg trade {rb['strategy_avg_trade']:+.4f} vs random coin "
                  f"{rb['random_coin_avg_trade']:+.4f}")
        except Exception as e:
            print(f"  random baselines unavailable: {e}")

    out = {
        "slug": "rotation",
        "hypothesis": "the 20-bar breakout trigger is a tax, not an edge; a pure top-N "
                      "cross-sectional momentum rotation with no price trigger should do better",
        "generated": pd.Timestamp.utcnow().isoformat(),
        "universe_file": str(UNIVERSE),
        "held_out_start": HELD_START,
        "shipped_flags": SHIPPED,
        "baseline_quoted_in_brief": BASELINE,
        "baseline_quoted_belongs_to_flags": "abs_momentum + volume_min=0.8 (no taker_min)",
        "baseline_shipped_flags_measured": BASELINE_SHIPPED,
        "parity_check": {"standard": base["standard"], "heldout": base["heldout"],
                         "matches_shipped_flags_baseline": parity,
                         "quoted_bar_flags_remeasured": {"standard": soft_std,
                                                         "heldout": soft_held}},
        "delevered": [{"name": r["name"], "risk_pct": r["risk_pct"],
                       "standard": r["standard"], "heldout": r["heldout"]} for r in delev],
        "drawdown_matched": dd_match,
        "exposure_matched": expo_matched,
        "per_trade_quality": pt,
        "incidental_findings": [
            {"what": "breakout c=5 exit_rank=5 (NOT this hypothesis: the shipped breakout "
                     "strategy with a tighter rank-drop exit)",
             "standard": next(r["standard"] for r in rows if r["panel"] == "breakout"
                              and r["candidates"] == 5 and r["exit_rank"] == 5),
             "heldout": next(r["heldout"] for r in rows if r["panel"] == "breakout"
                             and r["candidates"] == 5 and r["exit_rank"] == 5),
             "note": "clears both bars on CAGR, Sharpe and maxDD on both panels, but it is one "
                     "cell of a 6-point grid, it is an exit-side change that belongs to the exit "
                     "study, and it has not been checked on the survivorship-free panel or for "
                     "p_coin. Not a ship claim."},
            {"what": "rotation + stop_exec makes the close-based 20-bar-low exit redundant",
             "note": "the (no low exit, +stop_exec) arms are bit-identical to the (+stop_exec) "
                     "arms: a close below the prior 20-bar low implies that bar's low already "
                     "pierced the ratcheting stop, so the intrabar stop always fires first."},
        ],
        "variants": rows,
        "n_rotation_variants": len(rot),
        "n_beating_baseline": len(winners),
        "best": {"name": best["name"], "panel": best["panel"],
                 "candidates": best["candidates"], "exit_rank": best["exit_rank"],
                 "extra_flags": best["extra_flags"], "standard": best["standard"],
                 "heldout": best["heldout"], "checks": best["checks"],
                 "beats_baseline": best["beats_baseline"]},
        "best_by_raw_cagr": {"name": best_cagr["name"], "standard": best_cagr["standard"],
                             "heldout": best_cagr["heldout"]},
        "head_to_head": head,
        "mean_cagr_effect_of_removing_trigger": {"low_exit_kept": mean_keep,
                                                 "no_low_exit": mean_none},
        "best_robustness": extra,
        "verdict": ("BEATS baseline under the ship rule" if winners
                    else "REJECT: no rotation variant beats the baseline on both panels"),
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT}")
    print(out["verdict"])


if __name__ == "__main__":
    main()
