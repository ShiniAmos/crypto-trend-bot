"""Cross-sectional reversal and dip-buying: a deliberate attempt to disprove the bot's premise.

The shipped bot buys strength (top 5 of a momentum blend, on a 20-bar breakout). The hypothesis
here is the opposite: in crypto the worst recent performers bounce hardest, and buying a breakout
is buying exhaustion. If reversal beats momentum on both panels, the whole project is pointed the
wrong way.

Nothing in app/ is touched. Every variant is expressed by overwriting columns on a COPY of the
panel that backtest.run reads (rank / score, breakout, breakdown_exit, stop_level, rs_7d), so the
costs, the fills at the next bar's open, the risk limits, the slot logic and the exit ladder stay
byte-for-byte the shipped ones.

Two engine gates have to be neutralised to let a reversal rule exist at all, and both are declared
per variant in the output table:
  rs7   - backtest.run hardcodes `rs_7d > 0` on every entry. A rule that buys 7-day losers cannot
          coexist with it, so those variants set rs_7d to a constant 1.0 AFTER the score is
          computed (score/rank are untouched; rs_cap is off, so the column is read nowhere else).
  trig  - the 20-bar breakout trigger. "breakout=True" removes the trigger and makes the variant a
          plain rotation; a momentum rotation with no trigger is included as a reference run so the
          effect of removing the trigger is never confused with the effect of inverting the rank.

    python research/explore_reversal.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import backtest, config, features  # noqa: E402
from app.flags import Flags  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SHIPPED = dict(abs_momentum=True, volume_min=0.8, taker_min=0.5)
HELD_OUT = "2022-01-01"
EXIT_WIDE = 40          # bars, for the "wider exit channel" sensitivity runs
FIXED_STOP = 0.10       # risk-matched sizing: a flat 10% stop distance, no fitting, no lookahead
P_COIN_DRAWS = 200


# ---------------------------------------------------------------- panel surgery helpers

def rank_within_bar(p, col, ascending):
    """Cross-sectional rank of `col` inside each bar_time, eligible rows only.

    ascending=True  -> the SMALLEST value gets rank 1 (worst performer first: reversal).
    ascending=False -> the LARGEST value gets rank 1 (the shipped direction).
    Never ranks across bars; ineligible rows and NaNs stay NaN, which the engine reads as
    "not a candidate" on entry and as "fell out of the top 10" on exit.
    """
    el = (p["eligible"] & p[col].notna()).to_numpy()
    r = pd.Series(np.nan, index=p.index)
    sub = p.loc[el]
    r.loc[el] = sub.groupby("bar_time")[col].rank(ascending=ascending, method="first")
    return r


def drop_rs_gate(p):
    """Neutralise backtest.run's hardcoded `rs_7d > 0` entry gate. Score is already computed."""
    p["rs_7d"] = 1.0
    return p


def risk_match(p):
    """Flat FIXED_STOP distance below the close, so a dip entry is not sized 3x the baseline.

    Buying a coin sitting on its own 20-bar low makes (entry - stop_level) tiny, and the 1%-risk
    formula then demands a position so large it pins at the 30% weight cap. That is a leverage
    change masquerading as an alpha change. This gives every reversal entry the same stop distance
    instead, which is a constant and therefore cannot look ahead.
    """
    p["stop_level"] = p["close"] * (1 - FIXED_STOP)
    return p


def measure(panel, s, flags, start=None, p_coin=False):
    res = backtest.run(panel, s, flags=flags, start=start)
    m = backtest.metrics(res.equity, res.trades, res.exposure)
    out = {"cagr": m["cagr"], "sharpe": m["sharpe"], "maxdd": m["max_drawdown"],
           "trades": int(m.get("trades", 0)), "win": m.get("win_rate", float("nan")),
           "expo": m.get("avg_exposure", float("nan")),
           "bars": m.get("avg_bars_held", float("nan")),
           "best": float(res.trades["ret"].max()) if len(res.trades) else float("nan")}
    if p_coin and len(res.trades):
        rb = backtest.random_baselines(panel, res.trades, s, n=P_COIN_DRAWS) or {}
        out["p_coin"] = rb.get("p_value_vs_random_coin", float("nan"))
    else:
        out["p_coin"] = float("nan")
    return out


def ship_rule(base, cand):
    """CAGR >= baseline AND Sharpe >= baseline AND maxDD not worse. No epsilon, no trade-offs."""
    return {"CAGR": bool(cand["cagr"] >= base["cagr"]),
            "Sharpe": bool(cand["sharpe"] >= base["sharpe"]),
            "maxDD": bool(cand["maxdd"] >= base["maxdd"])}


# ---------------------------------------------------------------- the variants

def build(name, base, wide, s):
    """Return (panel, flags, notes) for one named variant. `base` / `wide` are never mutated."""
    SH = Flags(**SHIPPED)
    SH_WIDE = Flags(**SHIPPED, exit_donchian=EXIT_WIDE)

    # ---- reference runs (context only, never claimed as a win)
    if name == "R0 momentum shipped (baseline)":
        return base, SH, "top 5 by score, 20-bar breakout - the bar to beat"
    if name == "R1 momentum rotation, no trigger":
        p = base.copy()
        p["breakout"] = True
        return p, SH, "trig: isolates the breakout trigger from the rank direction"

    # ---- (a) buy the BOTTOM 5 by the same score: pure inversion of the rank column
    if name == "A1 bottom 5, breakout kept, rs7 kept":
        p = base.copy()
        p["rank"] = rank_within_bar(p, "score", ascending=True)
        return p, SH, "only the rank direction changed, nothing else"
    if name == "A2 bottom 5, breakout kept":
        p = drop_rs_gate(base.copy())
        p["rank"] = rank_within_bar(p, "score", ascending=True)
        return p, SH, "rs7"
    if name == "A3 bottom 5 rotation":
        p = drop_rs_gate(base.copy())
        p["rank"] = rank_within_bar(p, "score", ascending=True)
        p["breakout"] = True
        return p, SH, "rs7 + trig: the honest mirror of the bot"
    if name == "A3b bottom 5 rotation, risk-matched":
        p = risk_match(drop_rs_gate(base.copy()))
        p["rank"] = rank_within_bar(p, "score", ascending=True)
        p["breakout"] = True
        return p, SH, f"rs7 + trig + flat {FIXED_STOP:.0%} stop for sizing"
    if name == "A4 bottom 5 rotation, rs7 kept":
        p = base.copy()
        p["rank"] = rank_within_bar(p, "score", ascending=True)
        p["breakout"] = True
        return p, SH, "trig"

    # ---- (b) buy a dip inside an uptrend
    if name == "B1 dip in uptrend, momentum rank":
        p = base.copy()
        p["breakout"] = ((p["dist_sma50"] > 0) & p["breakdown"] & (p["rs_30d"] > 0)).fillna(False)
        return p, SH, "shipped rank still gates at top 5"
    if name == "B2 dip in uptrend, rs30 rank":
        p = drop_rs_gate(base.copy())
        p["breakout"] = ((p["dist_sma50"] > 0) & p["breakdown"] & (p["rs_30d"] > 0)).fillna(False)
        p["rank"] = rank_within_bar(p, "rs_30d", ascending=False)
        return p, SH, "rs7: dip, ranked by the strongest 30d trend"
    if name == "B2b dip in uptrend, rs30 rank, wide exit":
        p = drop_rs_gate(wide.copy())
        p["breakout"] = ((p["dist_sma50"] > 0) & p["breakdown"] & (p["rs_30d"] > 0)).fillna(False)
        p["rank"] = rank_within_bar(p, "rs_30d", ascending=False)
        return p, SH_WIDE, f"rs7 + {EXIT_WIDE}-bar exit channel (EXITS DIFFER)"
    if name == "B2c dip in uptrend, rs30 rank, risk-matched":
        p = risk_match(drop_rs_gate(base.copy()))
        p["breakout"] = ((p["dist_sma50"] > 0) & p["breakdown"] & (p["rs_30d"] > 0)).fillna(False)
        p["rank"] = rank_within_bar(p, "rs_30d", ascending=False)
        return p, SH, f"rs7 + flat {FIXED_STOP:.0%} stop for sizing"
    if name == "B3 lowest in channel, above sma50":
        # A softer dip: no breakdown required, just the bottom of the 20-bar range while the
        # 50-day trend is still up. don_pos is (close - prior low) / (prior high - prior low).
        p = drop_rs_gate(base.copy())
        p["breakout"] = ((p["dist_sma50"] > 0) & (p["don_pos"] < 0.25)).fillna(False)
        p["rank"] = rank_within_bar(p, "don_pos", ascending=True)
        return p, SH, "rs7 + trig: deepest in its channel first"

    # ---- (c) the biggest 7-day losers whose 30-day return is still positive
    if name == "C1 worst 7d, 30d still up":
        p = drop_rs_gate(base.copy())
        p["breakout"] = True
        p["rank"] = rank_within_bar(p, "ret_7d", ascending=True)
        return p, SH, "rs7 + trig (abs_momentum flag enforces ret_30d > 0)"
    if name == "C2 worst 7d vs BTC, 30d still up":
        p = base.copy()
        p["rank"] = rank_within_bar(p, "rs_7d", ascending=True)
        p["breakout"] = True
        return drop_rs_gate(p), SH, "rs7 + trig"
    if name == "C1b worst 7d, risk-matched":
        p = risk_match(drop_rs_gate(base.copy()))
        p["breakout"] = True
        p["rank"] = rank_within_bar(p, "ret_7d", ascending=True)
        return p, SH, f"rs7 + trig + flat {FIXED_STOP:.0%} stop"
    if name == "C1c worst 7d, wide exit":
        p = drop_rs_gate(wide.copy())
        p["breakout"] = True
        p["rank"] = rank_within_bar(p, "ret_7d", ascending=True)
        return p, SH_WIDE, f"rs7 + trig + {EXIT_WIDE}-bar exit channel (EXITS DIFFER)"
    if name == "D1 worst 7d, no 30d filter":
        # The strongest form of the hypothesis: pure short-term reversal, no trend filter at all.
        p = drop_rs_gate(base.copy())
        p["breakout"] = True
        p["rank"] = rank_within_bar(p, "ret_7d", ascending=True)
        return p, Flags(volume_min=0.8, taker_min=0.5), "rs7 + trig + abs_momentum OFF (FLAGS DIFFER)"
    raise KeyError(name)


VARIANTS = ["R0 momentum shipped (baseline)",
            "R1 momentum rotation, no trigger",
            "A1 bottom 5, breakout kept, rs7 kept",
            "A2 bottom 5, breakout kept",
            "A3 bottom 5 rotation",
            "A3b bottom 5 rotation, risk-matched",
            "A4 bottom 5 rotation, rs7 kept",
            "B1 dip in uptrend, momentum rank",
            "B2 dip in uptrend, rs30 rank",
            "B2b dip in uptrend, rs30 rank, wide exit",
            "B2c dip in uptrend, rs30 rank, risk-matched",
            "B3 lowest in channel, above sma50",
            "C1 worst 7d, 30d still up",
            "C2 worst 7d vs BTC, 30d still up",
            "C1b worst 7d, risk-matched",
            "C1c worst 7d, wide exit",
            "D1 worst 7d, no 30d filter"]


# ---------------------------------------------------------------- lookahead self-check

def lookahead_checks(base):
    """Prove the columns a variant writes are built from closed bars only."""
    out = {}
    p = base.sort_values(["symbol", "bar_time"])
    g = p.groupby("symbol", sort=False)
    # the entry channel must be the PRIOR 20 bars: don_high == shift(1) of a 20-bar rolling max
    want_hi = g["high"].apply(lambda x: x.rolling(20).max().shift(1)).to_numpy()
    want_lo = g["low"].apply(lambda x: x.rolling(20).min().shift(1)).to_numpy()
    out["don_high is prior-20 shifted"] = bool(
        np.allclose(p["don_high"].to_numpy(), want_hi, equal_nan=True))
    out["don_low is prior-20 shifted"] = bool(
        np.allclose(p["don_low"].to_numpy(), want_lo, equal_nan=True))
    out["breakdown uses the shifted low"] = bool(
        (p["breakdown"].fillna(False) == (p["close"] < p["don_low"]).fillna(False)).all())
    # a rank written per bar_time must never see another bar
    r = rank_within_bar(base, "ret_7d", ascending=True)
    chk = base.assign(_r=r).dropna(subset=["_r"])
    sizes = chk.groupby("bar_time")["_r"].agg(["max", "count"])
    out["ranks are within-bar only"] = bool((sizes["max"] == sizes["count"]).all())
    # inverting the rank must be a permutation of the same eligible rows, same bar
    inv = rank_within_bar(base, "score", ascending=True)
    out["inverted rank covers the same rows"] = bool(
        (inv.notna() == (base["eligible"] & base["score"].notna())).all())
    return out


# ---------------------------------------------------------------- engine-free evidence

def forward_return_study(base, horizons=(1, 7, 30), bpd=6):
    """Top vs bottom of the cross-section, measured as raw forward returns.

    This deliberately ignores the bot: no exits, no stops, no slot limits, no sizing. Each risk-on
    bar's eligible coins are split into the top 5 and bottom 5 by the shipped score, and the return
    from the NEXT bar's open to the open h days later is averaged. If reversal is real, the bottom
    bucket must out-earn the top bucket here - and no choice of exit rule can hide that.

    No lookahead in the signal: the bucket is decided on bar t's closed data and the position is
    priced from bar t+1's open, exactly as the engine fills. The forward window is measurement, not
    information available to the rule.
    """
    opens = base.pivot_table(index="bar_time", columns="symbol", values="open")
    rows = []
    for h in horizons:
        fwd = (opens.shift(-(1 + h * bpd)) / opens.shift(-1) - 1).stack(future_stack=True)
        fwd.name = f"fwd_{h}d"
        rows.append(fwd)
    fwd = pd.concat(rows, axis=1)
    fwd.index.names = ["bar_time", "symbol"]
    j = base.set_index(["bar_time", "symbol"]).join(fwd)
    j = j[j["eligible"] & j["regime_on"].fillna(False) & j["rank"].notna()]
    n = j.groupby(level=0)["rank"].transform("max")
    bucket = pd.Series(np.nan, index=j.index, dtype=object)
    bucket[j["rank"] <= 5] = "top 5 (bot buys these)"
    bucket[j["rank"] > n - 5] = "bottom 5 (reversal buys these)"
    out = {}
    for h in horizons:
        col, res = f"fwd_{h}d", {}
        for label in ("top 5 (bot buys these)", "bottom 5 (reversal buys these)"):
            x = j.loc[bucket == label, col].dropna()
            res[label] = {"mean": float(x.mean()), "median": float(x.median()),
                          "n": int(len(x)),
                          "t_stat": float(x.mean() / (x.std() / np.sqrt(len(x)))) if len(x) > 1
                          else float("nan")}
        res["top_minus_bottom"] = res["top 5 (bot buys these)"]["mean"] \
            - res["bottom 5 (reversal buys these)"]["mean"]
        out[f"{h}d"] = res
    return out


def main():
    t0 = time.time()
    s = config.load()
    u = json.loads((ROOT / "reports" / "_parity_universe.json").read_text())
    base = features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars)
    wide = features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars,
                               exit_donchian=EXIT_WIDE)
    print(f"panel {base.shape[0]} rows, {base['bar_time'].min().date()} -> "
          f"{base['bar_time'].max().date()}, built in {time.time() - t0:.1f}s")

    print("\n=== lookahead self-check ===")
    checks = lookahead_checks(base)
    for k, v in checks.items():
        print(f"  {'OK  ' if v else 'FAIL'} {k}")

    # the median stop distance the shipped rule actually uses, for context on the sizing argument
    el = base["eligible"] & base["breakout"] & base["regime_on"] & (base["rank"] <= 5)
    d = ((base.loc[el, "close"] - base.loc[el, "stop_level"]) / base.loc[el, "close"]).dropna()
    print(f"\nshipped entries' stop distance: median {d.median():.1%}, mean {d.mean():.1%} "
          f"(risk-matched runs use a flat {FIXED_STOP:.0%})")

    print("\n=== engine-free check: forward returns by cross-sectional bucket (risk-on bars) ===")
    fwd = forward_return_study(base)
    for h, res in fwd.items():
        print(f"  horizon {h}:")
        for label in ("top 5 (bot buys these)", "bottom 5 (reversal buys these)"):
            r = res[label]
            print(f"    {label:<32} mean {r['mean']:+.4f}  median {r['median']:+.4f}  "
                  f"t {r['t_stat']:+.2f}  n {r['n']}")
        print(f"    top minus bottom                 {res['top_minus_bottom']:+.4f}")

    std, held, notes = {}, {}, {}
    for name in VARIANTS:
        p, flags, note = build(name, base, wide, s)
        notes[name] = note
        std[name] = measure(p, s, flags, p_coin=True)
        held[name] = measure(p, s, flags, start=HELD_OUT)
        del p
        print(f"  ran {name:<46} cagr {std[name]['cagr']:+.4f} "
              f"sharpe {std[name]['sharpe']:+.4f} dd {std[name]['maxdd']:+.4f} "
              f"tr {std[name]['trades']:>5}")

    cols = ["cagr", "sharpe", "maxdd", "trades", "win", "expo", "bars", "best", "p_coin"]
    for tag, d_ in (("STANDARD PANEL 2017-2026", std), (f"HELD-OUT from {HELD_OUT}", held)):
        print(f"\n=== {tag} ===")
        t = pd.DataFrame(d_).T[cols]
        print(t.to_string(float_format=lambda x: f"{x:.4f}"))

    b_std, b_held = std[VARIANTS[0]], held[VARIANTS[0]]
    print("\n=== ship rule: CAGR >= base AND Sharpe >= base AND maxDD not worse, on BOTH ===")
    rows = []
    for name in VARIANTS[1:]:
        cs, ch = ship_rule(b_std, std[name]), ship_rule(b_held, held[name])
        passed = all(cs.values()) and all(ch.values())
        rows.append({"variant": name, "notes": notes[name],
                     "std": "".join(k[0] if v else k[0].lower() for k, v in cs.items()),
                     "held": "".join(k[0] if v else k[0].lower() for k, v in ch.items()),
                     "verdict": "PASS" if passed else "REJECT",
                     "std_checks": cs, "held_checks": ch, "passed": passed})
    print(pd.DataFrame(rows)[["variant", "notes", "std", "held", "verdict"]]
          .to_string(index=False))
    print("  (letters: C=CAGR S=Sharpe M=maxDD; UPPER = met the baseline, lower = failed it)")

    # "best" reversal variant = the one with the highest held-out Sharpe among the reversal runs
    rev = [n for n in VARIANTS if n[0] in "ABCD"]
    best = max(rev, key=lambda n: (held[n]["sharpe"], held[n]["cagr"]))
    print(f"\nbest reversal variant by held-out Sharpe: {best}")
    print(f"  standard : cagr {std[best]['cagr']:.4f} sharpe {std[best]['sharpe']:.4f} "
          f"dd {std[best]['maxdd']:.4f} trades {std[best]['trades']}")
    print(f"  held-out : cagr {held[best]['cagr']:.4f} sharpe {held[best]['sharpe']:.4f} "
          f"dd {held[best]['maxdd']:.4f} trades {held[best]['trades']}")
    print(f"  baseline : cagr {b_std['cagr']:.4f}/{b_held['cagr']:.4f} "
          f"sharpe {b_std['sharpe']:.4f}/{b_held['sharpe']:.4f} "
          f"dd {b_std['maxdd']:.4f}/{b_held['maxdd']:.4f}")
    any_pass = [r["variant"] for r in rows if r["passed"]]
    print(f"\nvariants passing the ship rule on both panels: {any_pass or 'NONE'}")

    out = {"slug": "reversal",
           "hypothesis": "crypto reverses: the worst recent performers bounce hardest, so buying "
                         "the bottom of the cross-section (or a dip inside an uptrend) should beat "
                         "the shipped momentum-breakout rule",
           "universe": u["symbols"], "held_out_start": HELD_OUT,
           "panel_start": str(base["bar_time"].min().date()),
           "panel_end": str(base["bar_time"].max().date()),
           "baseline_flags": SHIPPED,
           "baseline_measured": {"standard": b_std, "held_out": b_held},
           "baseline_quoted_in_task": {"cagr": 0.2071, "sharpe": 1.3712, "maxdd": -0.1509,
                                       "trades": 398,
                                       "note": "those numbers reproduce with abs_momentum+"
                                               "volume_min=0.8 only; adding taker_min=0.5 (the "
                                               "shipped flag set) gives the stricter baseline "
                                               "measured above, which is what every variant here "
                                               "is judged against"},
           "lookahead_checks": checks,
           "forward_return_by_bucket": fwd,
           "shipped_stop_distance": {"median": float(d.median()), "mean": float(d.mean()),
                                     "risk_matched_flat": FIXED_STOP},
           "notes": notes,
           "standard": std, "held_out": held,
           "ship_rule": [{k: r[k] for k in ("variant", "notes", "verdict", "std_checks",
                                            "held_checks")} for r in rows],
           "best_reversal_variant": best,
           "variants_tested": len(VARIANTS) - 1,
           "any_variant_passes": bool(any_pass),
           "runtime_s": round(time.time() - t0, 1)}
    path = ROOT / "reports" / "explore_reversal.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {path}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
