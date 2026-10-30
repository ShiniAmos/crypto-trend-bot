"""P5-regime-switch: does this strategy need a DIFFERENT rule in sideways/chop?

Hypothesis. The regime report says the shipped bot earns ~+78%/yr annualised in strong bull and
~+50%/yr in weak bull while bleeding in sideways/chop. Chop is the only regime where it is both
active and losing (recovery and bear have regime_on = False, so the bot is already in cash there
and nothing an entry rule does can change them). So: switch behaviour by regime.

Three families, all asked for in the task, plus the obvious controls:
  (a) refuse every entry while the label is chop          -> eligible = False on those bars
  (b) in chop, invert the trigger to mean reversion       -> breakout = breakdown on those bars
      (b1 literal, b2 also suppressing the breakdown exit so the new entry is not stopped out on
       the very next bar, b3 b2 plus an ATR-based stop so the size is not pinned at the 30% cap)
  (c) chop entries allowed but at half size              -> size_scale_col = 0.5 on those bars
  controls: go flat in chop; chop top-2 only; 1.5x size in strong bull; the engine's own
            regime_risk flag, which is (c) extended to recovery and bear.

Everything is expressed by overwriting columns on a COPY of the panel, so costs, fills, slot logic
and risk limits are byte-for-byte the shipped engine and the numbers stay comparable.

Causality. regime.label_bars uses BTC only: its position against the 200-day SMA (joined as-of,
backward) and its trailing 90-day return, both closed at or before bar t. The label at t gates a
decision taken on t's close, which fills at t+1's open. No bar sees its own future.

    python research/explore_regime_switch.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import backtest, config, features, regime  # noqa: E402
from app.flags import Flags  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SHIPPED = dict(abs_momentum=True, volume_min=0.8, taker_min=0.5)
HELD_OUT = "2022-01-01"
CHOP = "sideways/chop"

# The numbers quoted in the task brief, for reference only. The panel has since been extended, so
# the ship rule is judged against the baseline re-measured on TODAY's panel, which is the only
# apples-to-apples comparison. Both are reported.
QUOTED = {"standard": {"cagr": 0.2071, "sharpe": 1.3712, "maxdd": -0.1509, "trades": 398},
          "held_out": {"cagr": 0.1473, "sharpe": 1.1004, "maxdd": -0.1509, "trades": 233}}


# ---------------------------------------------------------------- measurement

def measure(panel, s, flags, start=None, p_coin=False, labels=None, **kw):
    res = backtest.run(panel, s, flags=flags, start=start, **kw)
    m = backtest.metrics(res.equity, res.trades, res.exposure)
    out = {"cagr": float(m["cagr"]), "sharpe": float(m["sharpe"]),
           "maxdd": float(m["max_drawdown"]), "trades": int(m.get("trades", 0)),
           "win_rate": float(m.get("win_rate", np.nan)),
           "avg_trade": float(m.get("avg_trade", np.nan)),
           "avg_exposure": float(m.get("avg_exposure", np.nan)),
           "total_return": float(m["total_return"]),
           "kills": int(res.kills), "halts": int(res.halts)}
    if p_coin:
        rb = backtest.random_baselines(panel, res.trades, s) or {}
        out["p_coin"] = float(rb.get("p_value_vs_random_coin", np.nan))
    if labels is not None and len(res.equity) > 2:
        per = regime.per_regime(res.equity, labels, res.trades, res.exposure)
        row = per[per["regime"] == CHOP]
        if len(row):
            out["chop_cagr"] = float(row["cagr"].iloc[0])
            out["chop_total"] = float(row["total_return"].iloc[0])
            out["chop_dd"] = float(row["max_drawdown"].iloc[0])
            out["chop_trades"] = int(row.get("trades", pd.Series([0])).iloc[0] or 0)
    return out, res


def verdict(base, cand):
    """The ship rule: no metric may get worse, on this panel."""
    eps = 1e-9
    checks = {"cagr": cand["cagr"] >= base["cagr"] - eps,
              "sharpe": cand["sharpe"] >= base["sharpe"] - eps,
              "maxdd": cand["maxdd"] >= base["maxdd"] - eps}
    return all(checks.values()), checks


# ---------------------------------------------------------------- panel surgery

def with_labels(panel):
    labels = regime.label_bars(panel)
    lab = panel["bar_time"].map(labels)
    return labels, lab.fillna("unknown")


def variants(panel, lab):
    """name -> (panel, flags, extra kwargs for backtest.run). Each panel is a fresh copy."""
    chop = (lab == CHOP).to_numpy()
    out = {}

    out["baseline (shipped)"] = (panel, Flags(**SHIPPED), {})

    # (a) refuse entries while the market chops. Exits are untouched, so a position opened in a
    # bull regime still runs its normal ladder if the label turns to chop while it is held.
    p = panel.copy()
    p["eligible"] = p["eligible"].to_numpy() & ~chop
    out["a: no entries in chop"] = (p, Flags(**SHIPPED), {})

    # Control for (a): also refuse in recovery and bear. Should be a no-op, because regime_on is
    # False in both by construction - if it is not a no-op, the label and the regime disagree.
    p = panel.copy()
    dead = lab.isin(["recovery", "bear"]).to_numpy()
    p["eligible"] = p["eligible"].to_numpy() & ~(chop | dead)
    out["a2: no entries in chop+recovery+bear"] = (p, Flags(**SHIPPED), {})

    # Control: go flat in chop. No entries AND every held name is sold on the first chop bar.
    p = panel.copy()
    p["eligible"] = p["eligible"].to_numpy() & ~chop
    p["breakdown"] = p["breakdown"].to_numpy() | chop
    out["ctl: flat in chop"] = (p, Flags(**SHIPPED), {})

    # Control: chop entries allowed, but only the very best name (rank 1-2) instead of the top 5.
    p = panel.copy()
    p["eligible"] = p["eligible"].to_numpy() & ~(chop & (p["rank"].to_numpy() > 2))
    out["ctl: chop top-2 only"] = (p, Flags(**SHIPPED), {})

    # (b) mean reversion in chop: buy the break of the prior 20-bar LOW instead of the high, while
    # the name is still strong (rs_7d > 0 and ret_30d > 0 are already required by the engine and
    # the shipped flags). Momentum is untouched in every other regime.
    bo, bd = panel["breakout"].to_numpy(), panel["breakdown"].to_numpy()
    p = panel.copy()
    p["breakout"] = np.where(chop, bd, bo)
    out["b1: chop mean-reversion (literal)"] = (p, Flags(**SHIPPED), {})

    # b2: the literal version buys a close below the channel low and then the SAME condition is the
    # exit, so the trade is sold at the next open. Suppress the breakdown exit on chop bars so the
    # position is actually given a chance; it still exits on regime off or a rank drop.
    p = panel.copy()
    p["breakout"] = np.where(chop, bd, bo)
    p["breakdown"] = np.where(chop, False, bd)
    out["b2: chop mean-reversion, no chop breakdown exit"] = (p, Flags(**SHIPPED), {})

    # b3: b2 plus a sane sizing reference. A dip entry sits almost on the 20-bar low, so
    # (entry - stop) collapses and the 1%-risk size is pinned at the 30% weight cap. Use the wider
    # of (20-bar low, close - 2 ATR), i.e. the LOWER price, which can only shrink the position.
    p = panel.copy()
    p["breakout"] = np.where(chop, bd, bo)
    p["breakdown"] = np.where(chop, False, bd)
    atr_stop = (p["close"] * (1 - 2 * p["atr_pct"].fillna(0.05))).to_numpy()
    p["stop_level"] = np.where(chop, np.minimum(p["stop_level"].to_numpy(), atr_stop),
                               p["stop_level"].to_numpy())
    out["b3: b2 + 2-ATR stop reference"] = (p, Flags(**SHIPPED), {})

    # b4: b3 gets only a handful of chop entries, because a close under the 20-bar low almost never
    # also shows a day of above-average volume with more taker buying than selling. Neutralise
    # exactly those two filters on chop bars (the columns are read nowhere else in the engine) so
    # the mean-reversion idea is judged on a real sample instead of six trades. ret_30d > 0 stays.
    p = panel.copy()
    p["breakout"] = np.where(chop, bd, bo)
    p["breakdown"] = np.where(chop, False, bd)
    p["volume_ratio"] = np.where(chop, 1.0, p["volume_ratio"].to_numpy())
    p["taker_ratio"] = np.where(chop, 1.0, p["taker_ratio"].to_numpy())
    atr_stop = (p["close"] * (1 - 2 * p["atr_pct"].fillna(0.05))).to_numpy()
    p["stop_level"] = np.where(chop, np.minimum(p["stop_level"].to_numpy(), atr_stop),
                               p["stop_level"].to_numpy())
    out["b4: b3 + volume/taker filters off in chop"] = (p, Flags(**SHIPPED), {})

    # b5/b6: a milder dip. "Below the 20-bar low" is an extreme event; don_pos is where the close
    # sits inside the prior channel (0 = at the low, 1 = at the high), both bounds shifted, so this
    # is causal. Buying the bottom fifth of the channel gives mean reversion a real sample while
    # the ordinary 20-bar-low exit still protects the trade.
    dp = panel["don_pos"].to_numpy()
    for thr in (0.2, 0.5):
        p = panel.copy()
        p["breakout"] = np.where(chop, np.nan_to_num(dp, nan=1.0) < thr, bo)
        atr_stop = (p["close"] * (1 - 2 * p["atr_pct"].fillna(0.05))).to_numpy()
        p["stop_level"] = np.where(chop, np.minimum(p["stop_level"].to_numpy(), atr_stop),
                                   p["stop_level"].to_numpy())
        out[f"b5: chop buy-the-dip don_pos<{thr}"] = (p, Flags(**SHIPPED), {})

    # (c) chop entries at half size. size_scale_col multiplies the risk-based quantity and is
    # clipped to [0.5, 1.5] by the engine, so 0.5 is the smallest stake it will take.
    p = panel.copy()
    p["rs_scale"] = np.where(chop, 0.5, 1.0)
    out["c1: chop entries at half size"] = (p, Flags(**SHIPPED), {"size_scale_col": "rs_scale"})

    p = panel.copy()
    p["rs_scale"] = np.where(chop | dead, 0.5, 1.0)
    out["c2: half size in chop+recovery+bear"] = (p, Flags(**SHIPPED), {"size_scale_col": "rs_scale"})

    # The engine's own version of c2, as a cross-check that the column trick reproduces it.
    out["c3: engine regime_risk flag"] = (panel, Flags(**SHIPPED, regime_risk=True), {})

    # The dual of (c): instead of shrinking in the bad regime, lean into the good one.
    p = panel.copy()
    p["rs_scale"] = np.where((lab == "strong bull").to_numpy(), 1.5, 1.0)
    out["d: 1.5x size in strong bull"] = (p, Flags(**SHIPPED), {"size_scale_col": "rs_scale"})

    # Best-of-both: no entries in chop AND bigger in strong bull.
    p = panel.copy()
    p["eligible"] = p["eligible"].to_numpy() & ~chop
    p["rs_scale"] = np.where((lab == "strong bull").to_numpy(), 1.5, 1.0)
    out["e: no chop entries + 1.5x strong bull"] = (p, Flags(**SHIPPED), {"size_scale_col": "rs_scale"})

    # (d) is the only direction that lifted CAGR on both panels, so probe it properly: a milder
    # lean, and the risk-neutral pair (bigger in the good regime, smaller in the bad one) which is
    # the version most likely to keep Sharpe and the drawdown intact.
    sb = (lab == "strong bull").to_numpy()
    for up in (1.15, 1.25):
        p = panel.copy()
        p["rs_scale"] = np.where(sb, up, 1.0)
        out[f"f: {up}x size in strong bull"] = (p, Flags(**SHIPPED), {"size_scale_col": "rs_scale"})
    for up in (1.25, 1.5):
        p = panel.copy()
        p["rs_scale"] = np.where(sb, up, np.where(chop, 0.5, 1.0))
        out[f"g: {up}x strong bull + 0.5x chop"] = (p, Flags(**SHIPPED), {"size_scale_col": "rs_scale"})

    # Anti-lookahead control: run (a) again with the label LAGGED one bar, so the decision uses a
    # label that was already a bar old. If (a)'s numbers barely move, the label is not being read at
    # a moment it could not have been known.
    p = panel.copy()
    lagged = (pd.Series(chop, index=panel.index).groupby(panel["symbol"]).shift(1)
              .astype("boolean").fillna(False).astype(bool))
    p["eligible"] = p["eligible"].to_numpy() & ~lagged.to_numpy()
    out["ctl: (a) with the label lagged 1 bar"] = (p, Flags(**SHIPPED), {})

    return out


def noise_floor(panel, s, labels):
    """How much do these metrics move for a change that means nothing?

    The 8% drawdown kill flattens the book and pauses for seven days, so a tiny change in position
    size can move the day the kill fires, and everything after it. Scaling every position by
    +/-1-2% is economically meaningless, so the spread it produces is the measurement's own noise.
    Any variant whose effect is inside this band is not evidence of anything.
    """
    rows = []
    for g in (0.98, 0.99, 1.0, 1.01, 1.02):
        p = panel.copy()
        p["rs_scale"] = g
        std, _ = measure(p, s, Flags(**SHIPPED), size_scale_col="rs_scale")
        held, _ = measure(p, s, Flags(**SHIPPED), start=HELD_OUT, size_scale_col="rs_scale")
        rows.append({"global_size": g, "std_cagr": std["cagr"], "std_sharpe": std["sharpe"],
                     "std_maxdd": std["maxdd"], "std_kills": std["kills"],
                     "held_cagr": held["cagr"], "held_sharpe": held["sharpe"],
                     "held_maxdd": held["maxdd"]})
    df = pd.DataFrame(rows)
    spread = {f"{c}_spread": float(df[c].max() - df[c].min())
              for c in ("std_cagr", "std_sharpe", "std_maxdd", "held_cagr", "held_sharpe",
                        "held_maxdd")}
    return {"runs": json.loads(df.to_json(orient="records")), "spread": spread}


# ---------------------------------------------------------------- diagnosis

def chop_attribution(res, labels):
    """Where does the chop bleed actually come from - trades BORN in chop, or trades born in a bull
    regime that were still open when the market turned sideways?

    per_regime attributes equity to BARS, not to entries, so the two are easily confused. This
    splits the baseline's closed trades by the regime they were entered in and by whether they were
    still open during at least one chop bar.
    """
    t = res.trades.copy()
    if t.empty:
        return {}
    idx = labels.index
    chop_pos = np.flatnonzero((labels == CHOP).to_numpy())
    t["entry_regime"] = t["entry_time"].map(lambda x: labels.get(pd.Timestamp(x), "unknown"))
    lo = idx.searchsorted(pd.DatetimeIndex(t["entry_time"]), side="left")
    hi = idx.searchsorted(pd.DatetimeIndex(t["exit_time"]), side="right")
    held_chop = [bool(np.any((chop_pos >= a) & (chop_pos < b))) for a, b in zip(lo, hi)]
    t["held_through_chop"] = held_chop
    g = t.groupby(["entry_regime", "held_through_chop"])
    rows = g.agg(n=("ret", "size"), mean_ret=("ret", "mean"), total_pnl=("pnl", "sum"),
                 win_rate=("pnl", lambda x: float((x > 0).mean())),
                 avg_bars=("bars_held", "mean")).reset_index()
    return {"by_entry_regime_and_chop_overlap": json.loads(rows.to_json(orient="records")),
            "chop_entered": {"n": int((t.entry_regime == CHOP).sum()),
                             "mean_ret": float(t.loc[t.entry_regime == CHOP, "ret"].mean()),
                             "total_pnl": float(t.loc[t.entry_regime == CHOP, "pnl"].sum())},
            "bull_entered_held_into_chop": {
                "n": int(((t.entry_regime.isin(["strong bull", "weak bull"])) & t.held_through_chop).sum()),
                "total_pnl": float(t.loc[t.entry_regime.isin(["strong bull", "weak bull"])
                                         & t.held_through_chop, "pnl"].sum())}}


# ---------------------------------------------------------------- main

def main():
    s = config.load()
    u = json.loads((ROOT / "reports" / "_parity_universe.json").read_text())
    panel = features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars)
    labels, lab = with_labels(panel)
    print(f"panel {panel.shape[0]} rows, {panel['bar_time'].nunique()} bars, "
          f"{panel['bar_time'].min().date()} .. {panel['bar_time'].max().date()}")
    print("bars per regime:", labels.value_counts().to_dict())
    print(f"unlabelled panel rows: {int((lab == 'unknown').sum())}\n")

    vs = variants(panel, lab)
    rows, attribution = [], None
    for name, (p, flags, kw) in vs.items():
        std, res = measure(p, s, flags, labels=labels, p_coin=True, **kw)
        if attribution is None:
            attribution = chop_attribution(res, labels)
            print("chop attribution (baseline):")
            print(pd.DataFrame(attribution["by_entry_regime_and_chop_overlap"]).to_string(), "\n")
        held, _ = measure(p, s, flags, start=HELD_OUT, labels=labels, **kw)
        rows.append({"variant": name, "std": std, "held": held})
        print(f"{name:46s} std CAGR {std['cagr']:+.4f} Sharpe {std['sharpe']:.4f} "
              f"DD {std['maxdd']:+.4f} n={std['trades']:4d} | held CAGR {held['cagr']:+.4f} "
              f"Sharpe {held['sharpe']:.4f} DD {held['maxdd']:+.4f} n={held['trades']:4d} | "
              f"chop {std.get('chop_cagr', float('nan')):+.3f}/yr ({std.get('chop_trades', 0)} ent)"
              f" | kills {std['kills']} p_coin {std.get('p_coin', float('nan')):.3f}")

    print("\n--- noise floor: the same strategy, every position scaled by +/-1-2% ---")
    nf = noise_floor(panel, s, labels)
    print(pd.DataFrame(nf["runs"]).to_string())
    print("spread:", {k: round(v, 4) for k, v in nf["spread"].items()})

    base = rows[0]
    print("\n--- ship rule vs the baseline re-measured on this panel ---")
    results = []
    for r in rows[1:]:
        ok_s, cs = verdict(base["std"], r["std"])
        ok_h, ch = verdict(base["held"], r["held"])
        ok_q = (r["std"]["cagr"] >= QUOTED["standard"]["cagr"]
                and r["std"]["sharpe"] >= QUOTED["standard"]["sharpe"]
                and r["std"]["maxdd"] >= QUOTED["standard"]["maxdd"]
                and r["held"]["cagr"] >= QUOTED["held_out"]["cagr"]
                and r["held"]["sharpe"] >= QUOTED["held_out"]["sharpe"]
                and r["held"]["maxdd"] >= QUOTED["held_out"]["maxdd"])
        ships = ok_s and ok_h
        results.append({**r, "ship_standard": ok_s, "ship_held_out": ok_h, "ships": ships,
                        "ship_vs_quoted_numbers": bool(ok_q),
                        "checks_standard": {k: bool(v) for k, v in cs.items()},
                        "checks_held_out": {k: bool(v) for k, v in ch.items()}})
        fails = [k for k, v in {**{f"std.{k}": v for k, v in cs.items()},
                               **{f"held.{k}": v for k, v in ch.items()}}.items() if not v]
        print(f"{'SHIP' if ships else 'REJECT':7s} {r['variant']:46s} "
              f"{'' if ships else 'fails ' + ', '.join(fails)}")

    # Best = the one that ships. If none does, the one CLOSEST to shipping: most of the six checks
    # passed, then the smallest worst-case relative shortfall on the checks it failed. Picking by
    # CAGR alone would crown a variant that doubled the drawdown.
    for r in results:
        b, worst, passed = (base["std"], base["held"]), 0.0, 0
        for panel_key, bb in (("std", base["std"]), ("held", base["held"])):
            for metric in ("cagr", "sharpe", "maxdd"):
                cv, bv = r[panel_key][metric], bb[metric]
                if cv >= bv - 1e-9:
                    passed += 1
                else:
                    worst = max(worst, abs(bv - cv) / max(abs(bv), 1e-9))
        r["checks_passed"] = passed
        r["worst_relative_shortfall"] = float(worst)
    shipped = [r for r in results if r["ships"]]
    best = (max(shipped, key=lambda r: r["std"]["sharpe"] + r["held"]["sharpe"]) if shipped
            else sorted(results, key=lambda r: (-r["checks_passed"],
                                                r["worst_relative_shortfall"]))[0])
    print(f"\nclosest to shipping: {best['variant']}  ships={best['ships']}  "
          f"checks passed {best['checks_passed']}/6  "
          f"worst shortfall {best['worst_relative_shortfall']:.1%}")
    for r in sorted(results, key=lambda r: (-r["checks_passed"], r["worst_relative_shortfall"]))[:5]:
        print(f"   {r['checks_passed']}/6  shortfall {r['worst_relative_shortfall']:6.1%}  {r['variant']}")

    payload = {
        "slug": "regime_switch",
        "hypothesis": "One rule cannot fit every market: the shipped momentum strategy bleeds in "
                      "sideways/chop, so switch behaviour (refuse / invert / half-size) on the "
                      "causal BTC regime label.",
        "panel": {"rows": int(panel.shape[0]), "bars": int(panel["bar_time"].nunique()),
                  "start": str(panel["bar_time"].min().date()),
                  "end": str(panel["bar_time"].max().date()),
                  "symbols": u["symbols"], "universe_file": "reports/_parity_universe.json"},
        "bars_per_regime": {k: int(v) for k, v in labels.value_counts().to_dict().items()},
        "quoted_baseline_from_brief": QUOTED,
        "measured_baseline": {"standard": base["std"], "held_out": base["held"]},
        "chop_attribution": attribution,
        "noise_floor": nf,
        "variants": results,
        "best": {"variant": best["variant"], "ships": best["ships"],
                 "checks_passed_of_6": best["checks_passed"],
                 "worst_relative_shortfall": best["worst_relative_shortfall"],
                 "standard": best["std"], "held_out": best["held"]},
        "any_variant_ships": bool(shipped),
        "verdict": ("SHIP" if shipped else
                    "REJECT - no regime switch beat the shipped strategy on both panels"),
    }
    outp = ROOT / "reports" / "explore_regime_switch.json"
    outp.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {outp}")
    return payload


if __name__ == "__main__":
    main()
