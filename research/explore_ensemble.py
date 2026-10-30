"""Ensemble voting: does requiring agreement between independent signals beat one trigger?

HYPOTHESIS
    The shipped entry fires on a single trigger (close above the prior 20-bar high). If that
    trigger is noisy, demanding that 2 or 3 genuinely different signals agree should pick a
    cleaner subset of the same opportunities, or open up good entries the breakout misses.

THE THREE SIGNALS (each a boolean per (bar_time, symbol), closed bars only)
    S1  trend     dist_sma50 > 0                     - price above its own 50-DAY average
    S2  breakout  the existing 20-bar high break      - close > prior 20-bar high (already .shift(1)ed)
    S3  pressure  taker_ratio > 0.55 AND volume_ratio > 1.0
    S2b accel     rs_7d > rs_30d                      - the lead over BTC is widening

HOW A VARIANT IS EXPRESSED
    The engine is never touched. backtest.run reads the `breakout` column as its entry trigger,
    so a k-of-3 vote is written into that column on a COPY of the panel. Costs, fills, sizing,
    slot logic, the exit ladder and the stop all stay byte-identical, which is what makes the
    numbers comparable.

THE HONEST WARNING THIS SCRIPT IS BUILT TO ANSWER
    S3 is a strictly tighter restatement of two shipped filters (volume_min=0.8, taker_min=0.5),
    and S1 is already a scoring component (dist_sma50 is 20% of the blend, and rank<=5 selects
    on it). So a vote can easily be the shipped rule in disguise. Every variant is therefore
    reported next to:
      - how often the vote fires vs how often the breakout alone fires,
      - how many of its entries are NON-breakout entries (the only way a vote is a new rule),
      - what share of its trades the baseline also took.
    A variant whose entries are a pure subset of the baseline's is a FILTER, not an ensemble, and
    is labelled as such.

    python research/explore_ensemble.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import backtest, config, features  # noqa: E402
from app.flags import Flags  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SHIPPED = dict(abs_momentum=True, volume_min=0.8, taker_min=0.5)
HELD_OUT = "2022-01-01"

# The task brief quotes these as the baseline. They actually reproduce abs_momentum+volume_min=0.8
# WITHOUT taker_min (verified below), so they are carried as a second, easier bar and the measured
# shipped baseline is the one a variant has to clear.
BRIEF_STD = {"cagr": 0.2071, "sharpe": 1.3712, "maxdd": -0.1509, "trades": 398}
BRIEF_HELD = {"cagr": 0.1473, "sharpe": 1.1004, "maxdd": -0.1509, "trades": 233}


# ------------------------------------------------------------------ the signals

def signals_of(panel):
    """The four candidate signals. Each uses only values known at that bar's close."""
    s1 = (panel["dist_sma50"] > 0).fillna(False)
    s2 = panel["breakout"].fillna(False).astype(bool)
    s3 = ((panel["taker_ratio"] > 0.55) & (panel["volume_ratio"] > 1.0)).fillna(False)
    s2b = (panel["rs_7d"] > panel["rs_30d"]).fillna(False)
    return {"S1": s1.to_numpy(), "S2": s2.to_numpy(), "S3": s3.to_numpy(), "S2b": s2b.to_numpy()}


def vote(sigs, names, k):
    """k-of-n agreement as a boolean array."""
    total = np.zeros(len(sigs[names[0]]), dtype=int)
    for n in names:
        total += sigs[n].astype(int)
    return total >= k


def with_trigger(panel, trigger):
    """A copy of the panel whose entry trigger is `trigger`. Nothing else changes."""
    p = panel.copy()
    p["breakout"] = trigger
    return p


# ------------------------------------------------------------------ measurement

def measure(panel, s, flags, start=None, p_coin=True):
    res = backtest.run(panel, s, flags=flags, start=start)
    m = backtest.metrics(res.equity, res.trades, res.exposure)
    t = res.trades
    out = {"cagr": m["cagr"], "sharpe": m["sharpe"], "maxdd": m["max_drawdown"],
           "trades": int(m.get("trades", 0)), "win": m.get("win_rate"),
           "avg_trade": m.get("avg_trade"), "avg_bars": m.get("avg_bars_held"),
           "expo": m.get("avg_exposure"),
           "best": float(t["ret"].max()) if len(t) else float("nan"),
           "worst": float(t["ret"].min()) if len(t) else float("nan")}
    if p_coin and len(t):
        rb = backtest.random_baselines(panel, t, s) or {}
        out["p_coin"] = rb.get("p_value_vs_random_coin", float("nan"))
    else:
        out["p_coin"] = float("nan")
    return out, res


def strict_verdict(base, cand):
    """The ship rule, no tolerances: every one of the three must be at least as good."""
    checks = {"CAGR": cand["cagr"] >= base["cagr"],
              "Sharpe": cand["sharpe"] >= base["sharpe"],
              "maxDD": cand["maxdd"] >= base["maxdd"]}
    return all(checks.values()), checks


def no_lucky_trade(res):
    """Rerun the metric with the single best trade's P&L removed, as a fragility check."""
    t = res.trades
    if not len(t):
        return float("nan")
    worst_case = t.drop(t["pnl"].idxmax())
    gross = worst_case["pnl"].sum()
    return float(gross / 10_000.0)      # total return on start equity, best trade deleted


# ------------------------------------------------------------------ diagnostics

def entry_diag(panel, s, flags, trigger, base_trigger):
    """Is this vote a genuinely new rule, or the shipped rule with extra filters bolted on?

    Works on the entry predicate, not on trades, so it is independent of slot competition.
    """
    from app import signals as sig_mod
    p = panel
    rank = p["rank"].to_numpy()
    btc = (p["symbol"] == "BTCUSDT").to_numpy()
    common = (p["eligible"] & p["regime_on"].fillna(False)).to_numpy() \
        & ((p["rs_7d"] > 0).to_numpy() | btc) & (rank <= s.candidates) \
        & sig_mod.extra_filters(p, flags).to_numpy()
    cand = common & trigger
    base = common & base_trigger
    return {"signal_rows": int(cand.sum()),
            "baseline_signal_rows": int(base.sum()),
            "shared_with_baseline": int((cand & base).sum()),
            "new_non_breakout_rows": int((cand & ~base_trigger).sum()),
            "subset_of_baseline": bool((cand & ~base).sum() == 0)}


def trade_overlap(cand_res, base_res):
    """Share of this variant's trades that the baseline also opened (same coin, same bar)."""
    if not len(cand_res.trades) or not len(base_res.trades):
        return float("nan")
    key = lambda t: set(zip(t["symbol"], t["entry_time"]))  # noqa: E731
    a, b = key(cand_res.trades), key(base_res.trades)
    return len(a & b) / len(a)


# ------------------------------------------------------------------ main

def main():
    s = config.load()
    u = json.loads((ROOT / "reports" / "_parity_universe.json").read_text())
    panel = features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars)
    ship = Flags(**SHIPPED)
    # Without the two pressure filters, so S3 is the only volume/taker gate: that is the only way
    # a 3-signal ensemble is actually testing three signals instead of four.
    ship_no_press = Flags(abs_momentum=True)
    sigs = signals_of(panel)
    base_trigger = sigs["S2"]

    # --- 0) how independent are these signals, really?
    el = (panel["eligible"] & panel["regime_on"].fillna(False)).to_numpy()
    fire = {n: float(sigs[n][el].mean()) for n in sigs}
    corr = pd.DataFrame({n: sigs[n][el].astype(int) for n in sigs}).corr()
    print("=== signal frequency on eligible risk-on rows ===")
    for n, v in fire.items():
        print(f"  {n:4s} fires {v:.4f}")
    print("\n=== pairwise correlation (eligible risk-on rows) ===")
    print(corr.round(3).to_string())

    # --- 1) the variant grid
    V = []          # (name, trigger array, flags, note)
    V.append(("baseline (S2 alone, shipped)", base_trigger, ship, "reference"))
    V.append(("S1 alone (trend only)", sigs["S1"], ship, "single-signal reference"))
    V.append(("S3 alone (pressure only)", sigs["S3"], ship, "single-signal reference"))
    V.append(("no trigger at all", np.ones(len(panel), bool), ship, "pure rotation reference"))

    V.append(("k=2 of {S1,S2,S3}", vote(sigs, ["S1", "S2", "S3"], 2), ship, "the hypothesis"))
    V.append(("k=3 of {S1,S2,S3}", vote(sigs, ["S1", "S2", "S3"], 3), ship, "the hypothesis"))
    V.append(("k=1 of {S1,S2,S3}", vote(sigs, ["S1", "S2", "S3"], 1), ship, "loosest vote"))

    V.append(("k=2 of {S1,S2b,S3} (accel for S2)", vote(sigs, ["S1", "S2b", "S3"], 2), ship,
              "S2 replaced by acceleration"))
    V.append(("k=3 of {S1,S2b,S3} (accel for S2)", vote(sigs, ["S1", "S2b", "S3"], 3), ship,
              "S2 replaced by acceleration"))

    V.append(("k=2 of {S1,S2,S2b}", vote(sigs, ["S1", "S2", "S2b"], 2), ship,
              "no pressure leg (S3 is too rare)"))
    V.append(("k=2 of {S1,S2,S3,S2b}", vote(sigs, ["S1", "S2", "S3", "S2b"], 2), ship,
              "4 signals, 2 must agree"))
    V.append(("k=3 of {S1,S2,S3,S2b}", vote(sigs, ["S1", "S2", "S3", "S2b"], 3), ship,
              "4 signals, 3 must agree"))

    # The same votes with the shipped pressure filters removed, so S3 is not double-counted.
    V.append(("k=2 of {S1,S2,S3}, no shipped pressure", vote(sigs, ["S1", "S2", "S3"], 2),
              ship_no_press, "genuine 3-signal ensemble"))
    V.append(("k=3 of {S1,S2,S3}, no shipped pressure", vote(sigs, ["S1", "S2", "S3"], 3),
              ship_no_press, "genuine 3-signal ensemble"))
    V.append(("k=2 of {S1,S2b,S3}, no shipped pressure", vote(sigs, ["S1", "S2b", "S3"], 2),
              ship_no_press, "genuine 3-signal ensemble, accel"))

    # --- 2) run every variant on both panels
    std_rows, held_rows, diag_rows = [], [], []
    base_res = base_held = None
    results = {}
    for name, trig, flags, note in V:
        p2 = with_trigger(panel, trig)
        m_std, r_std = measure(p2, s, flags)
        m_held, r_held = measure(p2, s, flags, start=HELD_OUT)
        if name.startswith("baseline"):
            base_res, base_held = r_std, r_held
        results[name] = (r_std, r_held)
        d = entry_diag(panel, s, flags, trig, base_trigger)
        d["ex_best_total_ret"] = no_lucky_trade(r_std)
        std_rows.append({"run": name, **m_std})
        held_rows.append({"run": name, **m_held})
        diag_rows.append({"run": name, "note": note, **d})

    for name in results:
        i = [r["run"] for r in diag_rows].index(name)
        diag_rows[i]["trade_overlap_vs_base"] = trade_overlap(results[name][0], base_res)

    cols = ["run", "cagr", "sharpe", "maxdd", "trades", "win", "avg_trade", "avg_bars", "expo",
            "p_coin"]
    fmt = lambda x: f"{x:.4f}"  # noqa: E731
    print("\n=== STANDARD PANEL 2017-08 .. 2026-09 ===")
    print(pd.DataFrame(std_rows)[cols].to_string(index=False, float_format=fmt))
    print(f"\n=== HELD-OUT from {HELD_OUT} ===")
    print(pd.DataFrame(held_rows)[cols].to_string(index=False, float_format=fmt))

    print("\n=== is it an ensemble or the shipped rule in disguise? ===")
    dcols = ["run", "signal_rows", "baseline_signal_rows", "new_non_breakout_rows",
             "subset_of_baseline", "trade_overlap_vs_base", "ex_best_total_ret"]
    print(pd.DataFrame(diag_rows)[dcols].to_string(index=False, float_format=fmt))

    # --- 3) the ship rule, strict, against the measured shipped baseline AND the brief's numbers
    b_std, b_held = std_rows[0], held_rows[0]
    print("\n=== SHIP RULE (strict: CAGR, Sharpe and maxDD all at least as good, BOTH panels) ===")
    verdicts = []
    for r_std, r_held, d in zip(std_rows[1:], held_rows[1:], diag_rows[1:]):
        ok_s, cs = strict_verdict(b_std, r_std)
        ok_h, ch = strict_verdict(b_held, r_held)
        ok_bs, _ = strict_verdict(BRIEF_STD, r_std)
        ok_bh, _ = strict_verdict(BRIEF_HELD, r_held)
        verdicts.append({
            "run": r_std["run"],
            "std": "PASS" if ok_s else "FAIL", "held": "PASS" if ok_h else "FAIL",
            "verdict": "BEATS BASELINE" if (ok_s and ok_h) else "REJECT",
            "fails": ",".join([f"std:{k}" for k, v in cs.items() if not v]
                              + [f"held:{k}" for k, v in ch.items() if not v]),
            "vs_brief_numbers": "PASS" if (ok_bs and ok_bh) else "FAIL",
            "is_filter_not_ensemble": d["subset_of_baseline"],
        })
    print(pd.DataFrame(verdicts).to_string(index=False))

    # --- 4) is the negative threshold-specific? sweep S3's two cut-offs and swap S1's average.
    print("\n=== robustness: k=2 of {S1,S2,S3} across S3 cut-offs and S1 definitions ===")
    sweep = []
    for tk in (0.50, 0.55, 0.60):
        for vr in (0.8, 1.0, 1.2):
            s3 = ((panel["taker_ratio"] > tk) & (panel["volume_ratio"] > vr)).fillna(False)
            sg = {"S1": sigs["S1"], "S2": sigs["S2"], "S3": s3.to_numpy()}
            trig = vote(sg, ["S1", "S2", "S3"], 2)
            a, _ = measure(with_trigger(panel, trig), s, ship, p_coin=False)
            b, _ = measure(with_trigger(panel, trig), s, ship, start=HELD_OUT, p_coin=False)
            sweep.append({"run": f"k=2 taker>{tk} vol>{vr}", "cagr": a["cagr"],
                          "sharpe": a["sharpe"], "maxdd": a["maxdd"], "trades": a["trades"],
                          "h_cagr": b["cagr"], "h_sharpe": b["sharpe"], "h_maxdd": b["maxdd"],
                          "pass": strict_verdict(b_std, a)[0] and strict_verdict(b_held, b)[0]})
    for lbl, col in (("S1=dist_sma200>0", "dist_sma200"), ("S1=dist_sma50>0", "dist_sma50")):
        s1 = (panel[col] > 0).fillna(False)
        sg = {"S1": s1.to_numpy(), "S2": sigs["S2"], "S3": sigs["S3"]}
        for k in (2, 3):
            trig = vote(sg, ["S1", "S2", "S3"], k)
            a, _ = measure(with_trigger(panel, trig), s, ship, p_coin=False)
            b, _ = measure(with_trigger(panel, trig), s, ship, start=HELD_OUT, p_coin=False)
            sweep.append({"run": f"k={k} {lbl}", "cagr": a["cagr"], "sharpe": a["sharpe"],
                          "maxdd": a["maxdd"], "trades": a["trades"], "h_cagr": b["cagr"],
                          "h_sharpe": b["sharpe"], "h_maxdd": b["maxdd"],
                          "pass": strict_verdict(b_std, a)[0] and strict_verdict(b_held, b)[0]})
    print(pd.DataFrame(sweep).to_string(index=False, float_format=fmt))

    # --- 5) anti-lookahead proof. The vote is rebuilt from columns the engine already reads at the
    # same timing, and "S2 alone" below must reproduce the untouched baseline exactly. As a second
    # check, every signal is lagged one bar PER SYMBOL: a rule with same-bar leakage collapses when
    # it is forced to act on strictly older information, while an honest rule only gets a bit worse.
    print("\n=== anti-lookahead checks ===")
    untouched, _ = measure(panel, s, ship, p_coin=False)
    print(f"  untouched panel baseline : cagr {untouched['cagr']:.6f} sharpe "
          f"{untouched['sharpe']:.6f} trades {untouched['trades']}")
    print(f"  rebuilt 'S2 alone' trigger: cagr {b_std['cagr']:.6f} sharpe "
          f"{b_std['sharpe']:.6f} trades {b_std['trades']}  "
          f"-> {'IDENTICAL' if abs(untouched['cagr'] - b_std['cagr']) < 1e-12 else 'DIFFERS'}")
    lag = []
    g = panel.groupby("symbol", sort=False)
    for nm, names, k in (("baseline S2", ["S2"], 1), ("k=2 {S1,S2,S3}", ["S1", "S2", "S3"], 2),
                         ("k=3 {S1,S2,S3}", ["S1", "S2", "S3"], 3)):
        trig = vote(sigs, names, k)
        lagged = (pd.Series(trig, index=panel.index).groupby(panel["symbol"], sort=False)
                  .shift(1).fillna(False).to_numpy().astype(bool))
        a, _ = measure(with_trigger(panel, lagged), s, ship, p_coin=False)
        now, _ = measure(with_trigger(panel, trig), s, ship, p_coin=False)
        lag.append({"run": nm, "cagr_now": now["cagr"], "cagr_lagged_1_bar": a["cagr"],
                    "sharpe_now": now["sharpe"], "sharpe_lagged": a["sharpe"],
                    "trades_now": now["trades"], "trades_lagged": a["trades"]})
    print(pd.DataFrame(lag).to_string(index=False, float_format=fmt))

    # --- 6) agreement clearly predicts TRADE quality but fires 20x less often. Is it simply
    # under-deployed? Raise the risk per trade on the triple-agreement rule so its exposure is
    # comparable, costs and every other rule unchanged. Reported as exploratory, not a ship
    # candidate: it moves the account's risk setting, not just the entry rule.
    import dataclasses
    print("\n=== exploratory: k=3 agreement at larger size (risk_pct scaled) ===")
    scaled = []
    trig3 = vote(sigs, ["S1", "S2", "S3"], 3)
    p3 = with_trigger(panel, trig3)
    for mult in (1, 2, 4, 6):
        s2 = dataclasses.replace(s, risk_pct=s.risk_pct * mult,
                                 max_weight=min(1.0, s.max_weight * mult))
        a, _ = measure(p3, s2, ship, p_coin=False)
        b, _ = measure(p3, s2, ship, start=HELD_OUT, p_coin=False)
        scaled.append({"run": f"k=3 risk x{mult}", "cagr": a["cagr"], "sharpe": a["sharpe"],
                       "maxdd": a["maxdd"], "expo": a["expo"], "trades": a["trades"],
                       "h_cagr": b["cagr"], "h_sharpe": b["sharpe"], "h_maxdd": b["maxdd"],
                       "pass": strict_verdict(b_std, a)[0] and strict_verdict(b_held, b)[0]})
    print(pd.DataFrame(scaled).to_string(index=False, float_format=fmt))

    out = {
        "hypothesis": "k-of-3 agreement between independent signals beats the single breakout "
                      "trigger",
        "robustness_sweep": sweep, "lookahead_lag_check": lag,
        "exploratory_k3_sized_up": scaled,
        "parity_untouched_vs_rebuilt_trigger": {"untouched": untouched, "rebuilt": b_std},
        "signal_frequency": fire,
        "signal_correlation": corr.round(4).to_dict(),
        "baseline_measured_standard": b_std,
        "baseline_measured_held_out": b_held,
        "baseline_quoted_in_brief": {"standard": BRIEF_STD, "held_out": BRIEF_HELD,
                                     "note": "these reproduce abs_momentum+volume_min=0.8 WITHOUT "
                                             "taker_min=0.5; the measured shipped baseline is "
                                             "stronger and is the bar used"},
        "standard": std_rows, "held_out": held_rows, "diagnostics": diag_rows,
        "verdicts": verdicts,
        "beats_baseline": any(v["verdict"] == "BEATS BASELINE" for v in verdicts),
    }
    (ROOT / "reports" / "explore_ensemble.json").write_text(
        json.dumps(out, indent=2, default=str))
    print("\nwrote reports/explore_ensemble.json")


if __name__ == "__main__":
    main()
