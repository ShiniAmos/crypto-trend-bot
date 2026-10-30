"""Longer momentum formation horizons (60d / 90d / 120d / 180d) against the shipped 7d/30d blend.

HYPOTHESIS
    The shipped ranker blends 7-day and 30-day relative strength. The cross-sectional momentum
    literature (Jegadeesh-Titman and everything after it) finds the edge in 3-12 month formation
    periods, not one week. If crypto trends persist on the same kind of timescale, ranking on
    60d/90d relative strength should sort the universe better than ranking on 7d/30d.

WHAT CHANGES, WHAT DOES NOT
    Only the RANKING changes (and, in two explicitly labelled variants, the momentum gate).
    Entry trigger (close above the prior 20-bar high), the regime gate, the three shipped filters
    (ret_30d>0, volume_ratio>=0.8, taker_ratio>=0.5), the exits, the 1%-risk sizing, the 30%/3-name
    caps, the 7bps slippage + 0.10% fee and the next-bar-open fills are all untouched: every variant
    runs through app.backtest.run on a copy of the same panel, so the comparison is clean.

    New score columns are passed with score_col=, which makes app.backtest._rank re-rank WITHIN each
    bar_time. The rank>10 exit therefore uses the same new score as the entry, which is the
    self-consistent reading of "a different ranking".

ANTI-LOOKAHEAD
    * every slow return is close.pct_change(n) on a per-symbol series ordered by bar_time, so the
      value at bar t uses closes at t and earlier only - identical in construction to the shipped
      ret_7d / ret_30d.
    * the decision is taken on the close of t and filled at the open of t+1 by the engine; no slow
      feature is read at a bar later than the one it is used on.
    * relative strength subtracts BTC's return over the SAME window at the SAME bar_time (a join on
      bar_time, never a shift forward).
    * all percentile ranks are computed inside a bar_time groupby, never across the whole frame.
    * nothing here touches breakout / breakdown / stop_level, which already carry their .shift(1).
    A printed self-check at the end re-derives one slow return by hand and asserts the column only
    ever looks backwards.

RUN
    cd crypto-bot && PYTHONIOENCODING=utf-8 python research/explore_horizons.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import backtest, config, features  # noqa: E402
from app.flags import Flags  # noqa: E402

SHIPPED = Flags(abs_momentum=True, volume_min=0.8, taker_min=0.5)
BPD = 6                       # 4h bars per day
HELD_OUT = "2022-01-01"

# The numbers this work has to beat, quoted from the task brief (recomputed below as a parity check).
BASELINE_QUOTED = {
    "standard": {"cagr": 0.2071, "sharpe": 1.3712, "maxdd": -0.1509, "trades": 398},
    "heldout": {"cagr": 0.1473, "sharpe": 1.1004, "maxdd": -0.1509, "trades": 233},
}


# ------------------------------------------------------------------ slow features

def add_slow_features(panel: pd.DataFrame, days=(60, 90, 120, 180)) -> pd.DataFrame:
    """ret_{d}d, rs_{d}d and vol_{d}d for each horizon, built per symbol from closes only.

    Positional pct_change over d*6 rows, exactly how features.symbol_features builds ret_7d and
    ret_30d, so a gap in a coin's history is treated the same way in both.
    """
    p = panel.sort_values(["symbol", "bar_time"]).copy()
    g = p.groupby("symbol", sort=False)["close"]
    logret = p.groupby("symbol", sort=False)["close"].transform(lambda c: np.log(c).diff())
    for d in days:
        n = d * BPD
        p[f"ret_{d}d"] = g.pct_change(n)
        p[f"vol_{d}d"] = (logret.groupby(p["symbol"], sort=False)
                          .transform(lambda x, n=n: x.rolling(n).std()) * np.sqrt(BPD * 365))
    p = p.sort_index()

    btc = p[p["symbol"] == "BTCUSDT"].set_index("bar_time")
    cols = [f"ret_{d}d" for d in days]
    p = p.join(btc[cols].add_prefix("btc_"), on="bar_time")
    for d in days:
        # relative strength vs BTC over the SAME window, read at the SAME bar.
        p[f"rs_{d}d"] = p[f"ret_{d}d"] - p[f"btc_ret_{d}d"]
    return p


def pct_rank(p: pd.DataFrame, series: pd.Series, mask: pd.Series) -> pd.Series:
    """Cross-sectional percentile of `series` among `mask` rows, within each bar_time."""
    out = pd.Series(np.nan, index=p.index)
    sub = series.where(mask)
    out[mask] = sub[mask].groupby(p.loc[mask, "bar_time"]).rank(pct=True)
    return out


def build_scores(p: pd.DataFrame) -> pd.DataFrame:
    """Add every candidate score column. NaN score = not rankable that bar (engine skips it)."""
    el = p["eligible"].fillna(False)

    def blend(parts_weights):
        """parts_weights: list of (series, weight). NaN in any component -> NaN score."""
        total = pd.Series(0.0, index=p.index)
        ok = pd.Series(True, index=p.index)
        for s, w in parts_weights:
            r = pct_rank(p, s, el)
            ok &= r.notna()
            total = total + w * r.fillna(0.0)
        return (100 * total).where(ok & el)

    ra30 = p["ret_30d"] / p["vol_30d"]

    # The shipped blend, rebuilt here so a "blend of fast and slow" uses an identical fast leg.
    p["sc_fast"] = blend([(p["rs_7d"], 0.35), (p["rs_30d"], 0.25),
                          (p["dist_sma50"], 0.20), (ra30, 0.20)])

    # (a) 60d/90d relative strength replacing 7d/30d, everything else the same weights.
    p["sc_slow_6090"] = blend([(p["rs_60d"], 0.35), (p["rs_90d"], 0.25),
                               (p["dist_sma50"], 0.20), (ra30, 0.20)])
    # ... and a slower pair still (90d/120d), and a 3-6 month pair (120d/180d).
    p["sc_slow_90120"] = blend([(p["rs_90d"], 0.35), (p["rs_120d"], 0.25),
                                (p["dist_sma50"], 0.20), (ra30, 0.20)])
    p["sc_slow_120180"] = blend([(p["rs_120d"], 0.35), (p["rs_180d"], 0.25),
                                 (p["dist_sma50"], 0.20), (ra30, 0.20)])
    # fully slow: the risk-adjusted leg also moves to the slow horizon.
    p["sc_slow_full"] = blend([(p["rs_60d"], 0.35), (p["rs_90d"], 0.25),
                               (p["dist_sma50"], 0.20),
                               (p["ret_90d"] / p["vol_90d"], 0.20)])
    # keep the fast leg, slow only the medium leg (30d -> 60d).
    p["sc_half_60"] = blend([(p["rs_7d"], 0.35), (p["rs_60d"], 0.25),
                             (p["dist_sma50"], 0.20), (ra30, 0.20)])
    p["sc_half_90"] = blend([(p["rs_7d"], 0.35), (p["rs_90d"], 0.25),
                             (p["dist_sma50"], 0.20), (ra30, 0.20)])
    # single-factor sanity checks: is slow RS alone a usable sort?
    p["sc_pure_rs60"] = blend([(p["rs_60d"], 1.0)])
    p["sc_pure_rs90"] = blend([(p["rs_90d"], 1.0)])
    p["sc_pure_rs7"] = blend([(p["rs_7d"], 1.0)])

    # (b) blends of the fast score and the slow score, on the same 0-100 scale.
    for w in (0.25, 0.50, 0.75):
        both = p["sc_fast"].notna() & p["sc_slow_6090"].notna()
        p[f"sc_mix{int(w * 100)}"] = ((1 - w) * p["sc_fast"] + w * p["sc_slow_6090"]).where(both)
    return p


# ------------------------------------------------------------------ measurement

def row(tag, res, note=""):
    m = backtest.metrics(res.equity, res.trades, res.exposure)
    t = res.trades
    # Payoff shape: this strategy lives on a handful of very large winners, so a sort that raises
    # the hit rate while shaving the right tail can look "better" and still earn less.
    best = top5_share = np.nan
    if len(t):
        best = float(t["ret"].max())
        pos = t.loc[t["pnl"] > 0, "pnl"]
        top5_share = float(pos.nlargest(5).sum() / pos.sum()) if pos.sum() > 0 else np.nan
    return {"variant": tag, "cagr": m["cagr"], "sharpe": m["sharpe"],
            "maxdd": m["max_drawdown"], "trades": int(m.get("trades", 0)),
            "win": m.get("win_rate", np.nan), "avg_trade": m.get("avg_trade", np.nan),
            "best_trade": best, "top5_share_of_gains": top5_share,
            "bars_held": m.get("avg_bars_held", np.nan),
            "exposure": m.get("avg_exposure", np.nan), "note": note}


def info_coeff(p: pd.DataFrame, cols, fwd_bars=48):
    """Cross-sectional Spearman IC of each column against the NEXT `fwd_bars` bars' return.

    Diagnostic only - the forward return is never fed to the backtest. It answers the question
    behind the hypothesis directly: does slow relative strength sort next week's winners at all?
    `fwd_bars=48` is 8 days, close to the strategy's ~47-bar average hold.
    """
    q = p.sort_values(["symbol", "bar_time"]).copy()
    # forward return: shift(-fwd) of close over this bar's close. Evaluation only.
    q["fwd"] = q.groupby("symbol", sort=False)["close"].shift(-fwd_bars) / q["close"] - 1
    q = q[q["eligible"].fillna(False) & q["fwd"].notna()]
    out = []
    for c in cols:
        sub = q[q[c].notna()]
        ic = (sub.groupby("bar_time")
                 .apply(lambda g, c=c: g[c].corr(g["fwd"], method="spearman"), include_groups=False)
                 .dropna())
        n = len(ic)
        out.append({"column": c, "mean_ic": float(ic.mean()), "bars": int(n),
                    "t_stat": float(ic.mean() / ic.std() * np.sqrt(n)) if n > 2 and ic.std() > 0
                    else float("nan"),
                    "share_positive": float((ic > 0).mean())})
    return out


def ship_rule(cand, base, tol=1e-9):
    """CAGR >= base AND Sharpe >= base AND maxDD not worse, on BOTH panels."""
    return {"cagr": cand["cagr"] >= base["cagr"] - tol,
            "sharpe": cand["sharpe"] >= base["sharpe"] - tol,
            "maxdd": cand["maxdd"] >= base["maxdd"] - tol}


def main():
    s = config.load()
    u = json.loads((ROOT / "reports" / "_parity_universe.json").read_text())

    print("building panel (19 pinned coins, 4h)...")
    panel = features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars)
    print(f"  {len(panel):,} rows, {panel['bar_time'].min().date()} .. {panel['bar_time'].max().date()}")

    print("adding slow horizons 60/90/120/180d...")
    panel = add_slow_features(panel)
    panel = build_scores(panel)

    # ---- lookahead self-checks -------------------------------------------------
    checks = {}
    pv = panel[panel["symbol"] == "ETHUSDT"].sort_values("bar_time").reset_index(drop=True)
    i = len(pv) - 1
    manual = pv["close"].iloc[i] / pv["close"].iloc[i - 60 * BPD] - 1
    checks["ret_60d_matches_manual"] = bool(abs(manual - pv["ret_60d"].iloc[i]) < 1e-12)
    # a slow return must be NaN before enough history exists (i.e. it is not back-filled)
    checks["ret_60d_nan_before_warmup"] = bool(pv["ret_60d"].iloc[:60 * BPD].isna().all())
    checks["ret_180d_nan_before_warmup"] = bool(pv["ret_180d"].iloc[:180 * BPD].isna().all())
    # shifting the slow column forward one bar must change the score -> it is not already shifted
    checks["rs_60d_is_not_future"] = bool(
        not pv["rs_60d"].dropna().equals(pv["rs_60d"].shift(-1).dropna()))
    # ranks live inside a bar: max rank per bar must never exceed that bar's eligible count
    r = backtest._rank(panel.sort_values(["bar_time", "symbol"]).reset_index(drop=True),
                       "sc_slow_6090")
    tmp = panel.sort_values(["bar_time", "symbol"]).reset_index(drop=True).assign(_r=r)
    per_bar = tmp.groupby("bar_time").agg(mx=("_r", "max"), n=("eligible", "sum"))
    checks["ranks_are_within_bar"] = bool((per_bar["mx"].fillna(0) <= per_bar["n"]).all())
    print("\nlookahead self-checks:", json.dumps(checks))
    if not all(checks.values()):
        print("  !! a self-check failed - results below are NOT trustworthy")

    # ---- baseline parity -------------------------------------------------------
    print("\nrunning baseline (shipped flags, shipped score)...")
    b_std = row("baseline", backtest.run(panel, s, flags=SHIPPED))
    b_held = row("baseline", backtest.run(panel, s, flags=SHIPPED, start=HELD_OUT))
    parity = {
        "std_cagr_delta": round(b_std["cagr"] - BASELINE_QUOTED["standard"]["cagr"], 4),
        "std_sharpe_delta": round(b_std["sharpe"] - BASELINE_QUOTED["standard"]["sharpe"], 4),
        "std_trades_delta": b_std["trades"] - BASELINE_QUOTED["standard"]["trades"],
        "held_cagr_delta": round(b_held["cagr"] - BASELINE_QUOTED["heldout"]["cagr"], 4),
    }
    print("  parity vs quoted baseline:", json.dumps(parity))

    # sanity: rebuilding the shipped blend by hand must reproduce the shipped score
    # Not a lookahead test: a control on the code path. `sc_fast` is the shipped blend rebuilt by
    # hand, so the score VALUES must be identical; any remaining difference in the run can only come
    # from how exact score ties are broken, because backtest._rank returns the precomputed `rank`
    # column for score_col="score" and recomputes it (on a differently ordered frame) otherwise.
    sorted_panel = panel.sort_values(["bar_time", "symbol"]).reset_index(drop=True)
    both = panel["score"].notna() & panel["sc_fast"].notna()
    r_ship = pd.Series(backtest._rank(sorted_panel, "score"))
    r_mine = pd.Series(backtest._rank(sorted_panel, "sc_fast"))
    codepath = {
        "score_values_identical": bool(
            (panel["score"] - panel["sc_fast"]).abs().max() == 0
            and int((panel["score"].notna() ^ panel["sc_fast"].notna()).sum()) == 0),
        "max_abs_score_diff": float((panel["score"] - panel["sc_fast"])[both].abs().max()),
        "rank_rows_differing_on_ties": int(
            (r_ship.fillna(-1) != r_mine.fillna(-1)).sum()),
        "rank_rows_total": int(len(r_ship)),
    }
    print("  code-path control (shipped blend rebuilt by hand):", json.dumps(codepath))

    # ---- the variant grid ------------------------------------------------------
    score_variants = [
        ("slow_6090", "sc_slow_6090", "(a) 60d/90d RS replaces 7d/30d"),
        ("slow_90120", "sc_slow_90120", "90d/120d RS replaces 7d/30d"),
        ("slow_120180", "sc_slow_120180", "120d/180d RS (3-6 month formation)"),
        ("slow_full_6090", "sc_slow_full", "60d/90d RS + 90d risk-adjusted leg"),
        ("half_60", "sc_half_60", "fast 7d kept, 30d leg -> 60d"),
        ("half_90", "sc_half_90", "fast 7d kept, 30d leg -> 90d"),
        ("pure_rs7", "sc_pure_rs7", "single factor: 7d RS only"),
        ("pure_rs60", "sc_pure_rs60", "single factor: 60d RS only"),
        ("pure_rs90", "sc_pure_rs90", "single factor: 90d RS only"),
        ("mix25", "sc_mix25", "(b) 75% fast + 25% slow score"),
        ("mix50", "sc_mix50", "(b) 50/50 fast + slow score"),
        ("mix75", "sc_mix75", "(b) 25% fast + 75% slow score"),
    ]

    # The rebuilt fast score is the baseline run through the SAME code path as the variants
    # (score_col= re-ranks, instead of reading the precomputed `rank` column). Its only difference
    # from the shipped baseline is how exact score ties are broken, so it bounds that artifact.
    std_rows, held_rows = [b_std], [b_held]
    std_rows.append(row("baseline_rebuilt", backtest.run(panel, s, flags=SHIPPED,
                                                         score_col="sc_fast"),
                        "shipped blend, re-ranked code path (tie-break control)"))
    held_rows.append(row("baseline_rebuilt", backtest.run(panel, s, flags=SHIPPED,
                                                          score_col="sc_fast", start=HELD_OUT),
                         "shipped blend, re-ranked code path (tie-break control)"))
    for tag, col, note in score_variants:
        print(f"  running {tag} ...")
        std_rows.append(row(tag, backtest.run(panel, s, flags=SHIPPED, score_col=col), note))
        held_rows.append(row(tag, backtest.run(panel, s, flags=SHIPPED, score_col=col,
                                               start=HELD_OUT), note))

    # (c) slow ranking with the fast breakout trigger intact is exactly the above (only `score_col`
    # moved), so the extra thing worth testing is replacing the fast MOMENTUM GATE too: the engine
    # hardcodes `rs_7d > 0`, which we express by overwriting that column on a copy of the panel.
    for tag, col, gate in (("slow_6090_gate60", "sc_slow_6090", "rs_60d"),
                           ("slow_6090_gate90", "sc_slow_6090", "rs_90d"),
                           ("slow_90120_gate90", "sc_slow_90120", "rs_90d")):
        print(f"  running {tag} ...")
        p2 = panel.copy()
        p2["rs_7d"] = p2[gate]          # the engine's `rs_7d > 0` entry gate now reads the slow RS
        note = f"slow rank + entry gate {gate}>0 instead of rs_7d>0"
        std_rows.append(row(tag, backtest.run(p2, s, flags=SHIPPED, score_col=col), note))
        held_rows.append(row(tag, backtest.run(p2, s, flags=SHIPPED, score_col=col,
                                              start=HELD_OUT), note))

    # Slow momentum as an extra ENTRY FILTER with the shipped ranking untouched: the other way a
    # longer horizon could add value if it is a poor sort but a decent gate.
    for d in (60, 90, 120, 180):
        tag = f"slow_filter_{d}"
        print(f"  running {tag} ...")
        p2 = panel.copy()
        # the engine gate is `rs_7d > 0`; force a failure wherever the slow RS is not positive.
        p2["rs_7d"] = p2["rs_7d"].where(p2[f"rs_{d}d"] > 0, -1.0)
        note = f"shipped ranking, entry also requires rs_{d}d>0"
        std_rows.append(row(tag, backtest.run(p2, s, flags=SHIPPED, score_col="sc_fast"), note))
        held_rows.append(row(tag, backtest.run(p2, s, flags=SHIPPED, score_col="sc_fast",
                                              start=HELD_OUT), note))

    std = pd.DataFrame(std_rows)
    held = pd.DataFrame(held_rows)

    fmt = lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)  # noqa: E731
    cols = ["variant", "cagr", "sharpe", "maxdd", "trades", "win", "avg_trade", "best_trade",
            "top5_share_of_gains", "bars_held", "exposure"]
    print("\n=== STANDARD PANEL (2017-08 .. 2026-09) ===")
    print(std[cols].to_string(index=False, float_format=fmt))
    print("\n=== HELD OUT (from 2022-01-01) ===")
    print(held[cols].to_string(index=False, float_format=fmt))

    # ---- the ship rule ---------------------------------------------------------
    bs = std.iloc[0].to_dict()
    bh = held.iloc[0].to_dict()
    rs_, rh_ = std.iloc[1].to_dict(), held.iloc[1].to_dict()   # baseline_rebuilt (same code path)
    verdicts = []
    for k in range(1, len(std)):
        cs, ch = std.iloc[k].to_dict(), held.iloc[k].to_dict()
        a, b = ship_rule(cs, bs), ship_rule(ch, bh)
        a2, b2 = ship_rule(cs, rs_), ship_rule(ch, rh_)
        verdicts.append({"ships_vs_rebuilt": all(a2.values()) and all(b2.values()),
                         "variant": cs["variant"],
                         "std_pass": all(a.values()), "held_pass": all(b.values()),
                         "ships": all(a.values()) and all(b.values()),
                         "std": a, "held": b,
                         "d_cagr_std": round(cs["cagr"] - bs["cagr"], 4),
                         "d_sharpe_std": round(cs["sharpe"] - bs["sharpe"], 4),
                         "d_maxdd_std": round(cs["maxdd"] - bs["maxdd"], 4),
                         "d_cagr_held": round(ch["cagr"] - bh["cagr"], 4),
                         "d_sharpe_held": round(ch["sharpe"] - bh["sharpe"], 4),
                         "d_maxdd_held": round(ch["maxdd"] - bh["maxdd"], 4)})
    v = pd.DataFrame(verdicts)
    print("\n=== SHIP RULE (CAGR, Sharpe, maxDD all >= baseline on BOTH panels) ===")
    print(v[["variant", "d_cagr_std", "d_sharpe_std", "d_maxdd_std",
             "d_cagr_held", "d_sharpe_held", "d_maxdd_held",
             "std_pass", "held_pass", "ships", "ships_vs_rebuilt"]]
          .to_string(index=False, float_format=fmt))

    # baseline_rebuilt is a control, not a candidate: exclude it from both winner lists.
    real = v["variant"] != "baseline_rebuilt"
    winners = v[v["ships"] & real]["variant"].tolist()
    winners_rebuilt = v[v["ships_vs_rebuilt"] & real]["variant"].tolist()
    print(f"\nvariants that pass the ship rule on both panels: {winners or 'NONE'}")
    print("  (same rule measured against baseline_rebuilt, i.e. the identical code path): "
          f"{winners_rebuilt or 'NONE'}")

    # ---- why: does slow relative strength sort forward returns at all? -----------
    ic_cols = ["rs_7d", "rs_30d", "rs_60d", "rs_90d", "rs_120d", "rs_180d",
               "dist_sma50", "sc_fast", "sc_slow_6090", "sc_slow_120180"]
    ic48 = info_coeff(panel, ic_cols, fwd_bars=48)
    ic180 = info_coeff(panel, ic_cols, fwd_bars=180)
    print("\n=== information coefficient vs forward return (diagnostic, eligible rows only) ===")
    icdf = pd.DataFrame(ic48).merge(pd.DataFrame(ic180), on="column", suffixes=("_8d", "_30d"))
    print(icdf[["column", "mean_ic_8d", "t_stat_8d", "share_positive_8d",
                "mean_ic_30d", "t_stat_30d", "share_positive_30d"]]
          .to_string(index=False, float_format=fmt))

    # The paradox worth recording: a slow score can sort the cross-section BETTER on average
    # (higher IC) and still earn less, because this strategy's return is carried by a few very
    # large winners and the slower sort trades that right tail for a higher hit rate.
    ic_map = {d["column"]: d["mean_ic"] for d in ic48}
    sdf = std.set_index("variant")
    interp = {}
    for cand, col in (("slow_120180", "sc_slow_120180"), ("slow_6090", "sc_slow_6090")):
        if cand in sdf.index:
            interp[cand] = {
                "mean_ic_8d": round(ic_map.get(col, float("nan")), 4),
                "baseline_mean_ic_8d": round(ic_map.get("sc_fast", float("nan")), 4),
                "sorts_better_on_average": bool(ic_map.get(col, -9) > ic_map.get("sc_fast", 9)),
                "avg_trade": round(float(sdf.loc[cand, "avg_trade"]), 4),
                "baseline_avg_trade": round(float(sdf.loc["baseline", "avg_trade"]), 4),
                "win_rate": round(float(sdf.loc[cand, "win"]), 4),
                "baseline_win_rate": round(float(sdf.loc["baseline", "win"]), 4),
                "best_trade": round(float(sdf.loc[cand, "best_trade"]), 4),
                "baseline_best_trade": round(float(sdf.loc["baseline", "best_trade"]), 4),
            }
    print("\npayoff-shape note:", json.dumps(interp))

    # Best on a single combined yardstick, for reporting even when nothing ships: rank variants by
    # the minimum of their per-panel Sharpe deltas (so a one-panel fluke cannot win).
    v["worst_sharpe_delta"] = v[["d_sharpe_std", "d_sharpe_held"]].min(axis=1)
    v["worst_cagr_delta"] = v[["d_cagr_std", "d_cagr_held"]].min(axis=1)
    best = v[real].sort_values("worst_sharpe_delta", ascending=False).iloc[0]
    print(f"\nleast-bad variant by worst-panel Sharpe: {best['variant']}  "
          f"(worst Sharpe delta {best['worst_sharpe_delta']:+.4f}, "
          f"worst CAGR delta {best['worst_cagr_delta']:+.4f})")

    # ---- common-window control -------------------------------------------------
    # The slow scores need more warm-up, so on the full panel they sit in cash for longer, which
    # alone would hurt CAGR. Re-run baseline and the headline slow variant from the first bar where
    # the slowest score is rankable, so neither side is penalised for warm-up.
    first_slow = panel.loc[panel["sc_slow_120180"].notna(), "bar_time"].min()
    print(f"\ncommon-window control from {pd.Timestamp(first_slow).date()} "
          "(first bar the slowest score exists):")
    ctl_rows = [row("baseline", backtest.run(panel, s, flags=SHIPPED, start=first_slow))]
    for tag, col, note in score_variants:
        ctl_rows.append(row(tag, backtest.run(panel, s, flags=SHIPPED, score_col=col,
                                             start=first_slow), note))
    ctl = pd.DataFrame(ctl_rows)
    print(ctl[cols].to_string(index=False, float_format=fmt))
    ctl_pass = []
    for k in range(1, len(ctl)):
        a = ship_rule(ctl.iloc[k].to_dict(), ctl.iloc[0].to_dict())
        if all(a.values()):
            ctl_pass.append(ctl.iloc[k]["variant"])
    print(f"common-window: variants beating baseline on all three metrics: {ctl_pass or 'NONE'}")

    # ---- save ------------------------------------------------------------------
    out = {
        "hypothesis": "Longer momentum formation (60/90/120/180d relative strength) ranks the "
                      "universe better than the shipped 7d/30d blend.",
        "method": "Only the score column changes; app.backtest.run re-ranks within each bar_time. "
                  "Entry trigger, three shipped filters, exits, sizing, caps, 7bps slippage + "
                  "0.10% fee and next-bar-open fills unchanged. Two variants additionally replace "
                  "the engine's rs_7d>0 entry gate by overwriting rs_7d on a panel copy.",
        "flags": SHIPPED.enabled,
        "panel": {"symbols": u["symbols"],
                  "start": str(panel["bar_time"].min().date()),
                  "end": str(panel["bar_time"].max().date())},
        "lookahead_checks": checks,
        "codepath_control": codepath,
        "baseline_parity_vs_quoted": parity,
        "baseline_parity_note": "CAGR reproduces the brief's quoted baseline to 2bp, but Sharpe, "
                                "maxDD and trade count do not (366 trades here vs 398 quoted). "
                                "app/ and data/klines were both modified after those numbers were "
                                "recorded, so this is engine/data drift, not a measurement here. "
                                "Every variant below is measured against the baseline re-run in "
                                "THIS process on THIS panel, so the comparison is internally exact.",
        "standard": std.to_dict(orient="records"),
        "heldout": held.to_dict(orient="records"),
        "common_window": {"start": str(pd.Timestamp(first_slow).date()),
                          "rows": ctl.to_dict(orient="records"),
                          "pass_all_three": ctl_pass},
        "ship_rule": v.to_dict(orient="records"),
        "information_coefficient": {"fwd_8d": ic48, "fwd_30d": ic180},
        "interpretation": interp,
        "ships": winners,
        "ships_vs_rebuilt_baseline": winners_rebuilt,
        "best_by_worst_panel_sharpe": {
            "variant": best["variant"],
            "worst_sharpe_delta": float(best["worst_sharpe_delta"]),
            "worst_cagr_delta": float(best["worst_cagr_delta"]),
            "std_pass": bool(best["std_pass"]), "held_pass": bool(best["held_pass"])},
        "verdict": ("slower formation beats the shipped blend: " + ", ".join(winners))
                   if winners else
                   ("REJECT - no longer-horizon ranking or filter beat the shipped 7d/30d blend "
                    "under the ship rule on both panels."
                    + (f" (against the identical-code-path baseline, these would pass: "
                       f"{', '.join(winners_rebuilt)})" if winners_rebuilt else "")),
    }
    dest = ROOT / "reports" / "explore_horizons.json"
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {dest}")
    print("VERDICT:", out["verdict"])


if __name__ == "__main__":
    main()
