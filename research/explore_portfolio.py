"""Portfolio construction: is the edge real but the portfolio wasting it?

The shipped bot holds at most 3 names, sizes every entry to the same 1% risk, and as a result sits
at ~9.5% average exposure - over 90% in cash. Three levers are tested here, none of which touches
which trade is taken:

  (a) slots      - max_names 3 (shipped), 4, 5, 6, 8
  (b) conviction - size by the candidate's rank that bar, via size_scale_col (engine clips [0.5,1.5])
  (c) risk       - risk_pct 1.0% (shipped) -> 1.5% / 2.0% / 2.5% / 3.0% at 3 slots
  (d) the best combination of the three

Nothing in the engine is modified: slots go through `max_names=`, conviction through
`size_scale_col=` on a COPY of the panel, risk through a COPY of the settings object. Costs, fills,
the stop, the kill switch and the slot logic are byte-identical to the shipped run, which is what
makes the numbers comparable.

Anti-lookahead note: the only new information used is `rank`, which the engine already reads from
the same bar to decide the entry, and `score`, which built it. Nothing from bar t+1 is touched; the
candidate-position column is derived with the engine's own `signals.extra_filters`, grouped inside
each bar_time. Raising risk_pct and max_names uses no data at all.

    python research/explore_portfolio.py
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import backtest, config, features, signals  # noqa: E402
from app.flags import Flags  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SHIPPED = Flags(abs_momentum=True, volume_min=0.8, taker_min=0.5)
HELD_OUT = "2022-01-01"

# The baseline as recorded in RESEARCH_LOG.md. The kline files have been refreshed since, so these
# do not reproduce bar for bar any more; the recomputed baseline below is the honest bar to clear,
# and every variant is checked against BOTH.
STATED = {"standard": {"cagr": 0.2071, "sharpe": 1.3712, "maxdd": -0.1509, "trades": 398},
          "held_out": {"cagr": 0.1473, "sharpe": 1.1004, "maxdd": -0.1509, "trades": 233}}


# ---------------------------------------------------------------- conviction-sizing columns

def add_conviction(panel, s, flags):
    """Three size multipliers, each a per-bar tilt only (mean 1.0 across the positions it scales).

    conv_rank  - straight from the score rank: rank 1 -> 1.5x, rank 5 -> 0.5x.
    conv_mild  - the same shape, half the spread: rank 1 -> 1.25x, rank 5 -> 0.75x.
    conv_cand  - position inside that bar's ACTUAL candidate list (after the shipped filters), so a
                 bar with one candidate gets 1.0x rather than a free 1.5x.
    conv_inv   - deliberately upside down (worst candidate biggest). A falsification check: if the
                 tilt is picking up real conviction, this one must be worse.
    """
    p = panel.copy()
    r = p["rank"]
    cand = s.candidates
    span = max(cand - 1, 1)
    p["conv_rank"] = (1.5 - (r - 1) * (1.0 / span)).clip(0.5, 1.5)
    p["conv_mild"] = (1.25 - (r - 1) * (0.5 / span)).clip(0.75, 1.25)
    p["conv_inv"] = (0.5 + (r - 1) * (1.0 / span)).clip(0.5, 1.5)

    # The engine's own entry predicate, reused rather than re-implemented so the two cannot drift.
    btc = p["symbol"] == "BTCUSDT"
    ok = (p["eligible"] & p["regime_on"] & p["breakout"]
          & ((p["rs_7d"] > 0) | btc) & (r <= cand)
          & signals.extra_filters(p, flags)).fillna(False)
    # Position within the bar's candidate list, computed WITHIN bar_time only.
    pos = p[ok].groupby("bar_time")["score"].rank(ascending=False, method="first")
    n = p[ok].groupby("bar_time")["score"].transform("size")
    scale = pd.Series(1.0, index=p.index)
    many = n > 1
    scale.loc[pos.index[many]] = 1.5 - (pos[many] - 1) * (1.0 / (n[many] - 1))
    p["conv_cand"] = scale.clip(0.5, 1.5)
    return p


# ---------------------------------------------------------------- measurement

def measure(panel, s, start=None, max_names=None, size_scale_col=None, with_p=False):
    res = backtest.run(panel, s, flags=SHIPPED, start=start, max_names=max_names,
                       size_scale_col=size_scale_col)
    m = backtest.metrics(res.equity, res.trades, res.exposure)
    t = res.trades
    row = {"cagr": float(m["cagr"]), "sharpe": float(m["sharpe"]),
           "maxdd": float(m["max_drawdown"]), "calmar": float(m["calmar"]),
           "trades": int(m.get("trades", 0)), "win": float(m.get("win_rate", np.nan)),
           "expo": float(m.get("avg_exposure", np.nan)),
           "pf": float(m.get("profit_factor", np.nan)),
           "kills": int(res.kills), "halts": int(res.halts),
           "best_trade": float(t["ret"].max()) if len(t) else float("nan"),
           "top3_share": float("nan")}
    if len(t):
        # How much of the whole result rests on three trades: a one-lucky-trade check.
        gross = t["pnl"].clip(lower=0).sum()
        row["top3_share"] = float(t["pnl"].nlargest(3).sum() / gross) if gross > 0 else float("nan")
    if with_p:
        rb = backtest.random_baselines(panel, res.trades, s) or {}
        row["p_coin"] = rb.get("p_value_vs_random_coin", float("nan"))
        row["p_time"] = rb.get("p_value_vs_random_time", float("nan"))
    return row


def ship(base, cand):
    """The ship rule, strict: CAGR, Sharpe and drawdown must each be no worse."""
    checks = {"CAGR": cand["cagr"] >= base["cagr"],
              "Sharpe": cand["sharpe"] >= base["sharpe"],
              "maxDD": cand["maxdd"] >= base["maxdd"]}
    return all(checks.values()), checks


def fails(base, cand):
    _, c = ship(base, cand)
    return ",".join(k for k, v in c.items() if not v)


def show(title, rows, cols=None):
    cols = cols or ["run", "cagr", "sharpe", "maxdd", "calmar", "expo", "trades", "win",
                    "pf", "kills", "top3_share"]
    df = pd.DataFrame(rows)[[c for c in cols if c in pd.DataFrame(rows).columns]]
    print(f"\n=== {title} ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main():
    s = config.load()
    u = json.loads((ROOT / "reports" / "_parity_universe.json").read_text())
    raw = features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars)
    panel = add_conviction(raw, s, SHIPPED)

    # Sanity: the extra columns must not change the shipped result at all.
    a = measure(raw, s)
    b = measure(panel, s)
    assert abs(a["cagr"] - b["cagr"]) < 1e-12 and a["trades"] == b["trades"], "panel copy drifted"
    # Parity: a size_scale_col of all ones must reproduce the shipped run exactly, which proves the
    # conviction results come from the tilt and not from the plumbing.
    panel["conv_flat"] = 1.0
    f = measure(panel, s, size_scale_col="conv_flat")
    assert abs(f["cagr"] - b["cagr"]) < 1e-12 and f["trades"] == b["trades"], "size_scale parity"
    print(f"parity ok: baseline cagr {b['cagr']:.6f}, {b['trades']} trades, "
          f"exposure {b['expo']:.4f} (so {1 - b['expo']:.1%} of the book sits in cash)")

    # Causality: rebuild the conviction columns from a panel truncated at a mid-history date and
    # check every value up to that date is unchanged. If any of them peeked at a later bar, the
    # truncated build would differ. (max_names and risk_pct use no data at all, so nothing to test.)
    cut = pd.Timestamp("2022-06-01", tz="UTC")
    trunc = add_conviction(raw[raw["bar_time"] <= cut].copy(), s, SHIPPED)
    keys = ["bar_time", "symbol"]
    cols = ["conv_rank", "conv_mild", "conv_cand", "conv_inv"]
    j = (panel[panel["bar_time"] <= cut].set_index(keys)[cols]
         .join(trunc.set_index(keys)[cols], rsuffix="_t", how="inner"))
    for c in cols:
        d = (j[c].fillna(-9) - j[c + "_t"].fillna(-9)).abs().max()
        assert d < 1e-12, f"{c} changed when future bars were removed: max diff {d}"
    print(f"causality ok: conviction columns identical on {len(j):,} rows built without any bar "
          f"after {cut.date()}")

    def settings_at(risk_pct):
        s2 = copy.copy(s)
        s2.risk_pct = risk_pct
        return s2

    # variants: (label, settings, max_names, size_scale_col, group)
    variants = [("baseline 3 slots 1.0% risk", s, None, None, "base")]
    for k in (4, 5, 6, 8):
        variants.append((f"(a) {k} slots", s, k, None, "slots"))
    for col, nice in (("conv_rank", "rank 1.5->0.5"), ("conv_mild", "rank 1.25->0.75"),
                      ("conv_cand", "candidate pos 1.5->0.5"), ("conv_inv", "INVERTED (sanity)")):
        variants.append((f"(b) conviction {nice}", s, None, col, "conv"))
    for rp in (0.015, 0.02, 0.025, 0.03):
        variants.append((f"(c) risk {rp:.1%} @3 slots", settings_at(rp), None, None, "risk"))
    # (d) combinations
    for k in (4, 5, 6):
        for rp in (0.01, 0.015, 0.02):
            for col in (None, "conv_rank"):
                tag = f"(d) {k} slots, risk {rp:.1%}" + (", conviction" if col else "")
                variants.append((tag, settings_at(rp), k, col, "combo"))
    # the two extremes of the curve, to show where it breaks
    variants.append(("(d) 8 slots, risk 2.0%", settings_at(0.02), 8, None, "combo"))
    variants.append(("(d) 5 slots, risk 3.0%", settings_at(0.03), 5, None, "combo"))
    # (e) the cleanest form of the hypothesis: spread the SAME total risk budget over more names.
    # K * risk_pct is held at the shipped 3 x 1% = 3%, so any improvement is diversification and
    # not leverage. If more slots genuinely help, this is where it shows.
    for k in (4, 5, 6, 8):
        variants.append((f"(e) {k} slots, risk {0.03 / k:.2%} (total risk fixed at 3%)",
                         settings_at(0.03 / k), k, None, "fixed"))
    # (f) neighbourhood of the one near-miss cell, to see whether it is a smooth optimum or a spike.
    for rp in (0.012, 0.013, 0.014, 0.016, 0.017, 0.018):
        variants.append((f"(f) 5 slots, risk {rp:.1%} (scan)", settings_at(rp), 5, None, "scan"))

    std, held = [], []
    for name, st, mn, col, grp in variants:
        std.append({"run": name, "group": grp, **measure(panel, st, None, mn, col)})
        held.append({"run": name, "group": grp,
                     **measure(panel, st, HELD_OUT, mn, col)})

    show("STANDARD PANEL 2017-08..2026-09", std)
    show("HELD-OUT from 2022-01-01", held)

    # ---------------------------------------------------------------- ship rule
    b_std, b_held = std[0], held[0]
    rows = []
    for rs, rh in zip(std[1:], held[1:]):
        ok_s, _ = ship(b_std, rs)
        ok_h, _ = ship(b_held, rh)
        ok_s2, _ = ship(STATED["standard"], rs)
        ok_h2, _ = ship(STATED["held_out"], rh)
        rows.append({"run": rs["run"],
                     "std_cagr": rs["cagr"], "std_sharpe": rs["sharpe"], "std_dd": rs["maxdd"],
                     "ho_cagr": rh["cagr"], "ho_sharpe": rh["sharpe"], "ho_dd": rh["maxdd"],
                     "vs_recomputed": "PASS" if (ok_s and ok_h) else "FAIL",
                     "vs_stated": "PASS" if (ok_s2 and ok_h2) else "FAIL",
                     "fails_std": fails(b_std, rs), "fails_ho": fails(b_held, rh)})
    print("\n=== SHIP RULE (strict: CAGR, Sharpe and maxDD all no worse, BOTH panels) ===")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---------------------------------------------------------------- risk/return curve
    curve = [{"run": r["run"], "expo": r["expo"], "cagr": r["cagr"], "sharpe": r["sharpe"],
              "maxdd": r["maxdd"], "calmar": r["calmar"], "kills": r["kills"],
              "ho_cagr": h["cagr"], "ho_sharpe": h["sharpe"], "ho_dd": h["maxdd"]}
             for r, h in zip(std, held)]
    print("\n=== RISK / RETURN CURVE (sorted by exposure) ===")
    print(pd.DataFrame(sorted(curve, key=lambda x: x["expo"]))
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---------------------------------------------------------------- is the tilt real?
    # Conviction raises exposure as a side effect (the top-ranked name gets the big size and fills
    # the slot first). Compare each conviction run against the plain run whose exposure is nearest,
    # so the question becomes "is the tilt better than simply staking more?" rather than
    # "is more risk better than less?".
    plain = [r for r in std if r["group"] in ("base", "risk")]
    pairs = []
    for r in [x for x in std if x["group"] == "conv"]:
        near = min(plain, key=lambda x: abs(x["expo"] - r["expo"]))
        pairs.append({"conviction": r["run"], "expo": r["expo"], "cagr": r["cagr"],
                      "sharpe": r["sharpe"], "maxdd": r["maxdd"],
                      "matched_plain": near["run"], "m_expo": near["expo"],
                      "m_cagr": near["cagr"], "m_sharpe": near["sharpe"], "m_maxdd": near["maxdd"],
                      "tilt_adds_cagr": r["cagr"] - near["cagr"],
                      "tilt_adds_sharpe": r["sharpe"] - near["sharpe"]})
    print("\n=== CONVICTION TILT vs AN EXPOSURE-MATCHED PLAIN RUN ===")
    print(pd.DataFrame(pairs).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---------------------------------------------------------------- is the curve just leverage?
    # If raising exposure only levers the same equity curve, CAGR rises in proportion to drawdown
    # and Sharpe stays flat. Printing CAGR/exposure and the drawdown-per-unit-exposure makes that
    # visible: a real diversification gain would show rising Sharpe, not a constant one.
    lev = [{"run": r["run"], "expo": r["expo"], "cagr": r["cagr"],
            "cagr_per_expo": r["cagr"] / r["expo"] if r["expo"] else np.nan,
            "dd_per_expo": r["maxdd"] / r["expo"] if r["expo"] else np.nan,
            "sharpe": r["sharpe"], "calmar": r["calmar"], "kills": r["kills"]}
           for r in std]
    print("\n=== LEVERAGE CHECK (return and drawdown per unit of exposure) ===")
    print(pd.DataFrame(sorted(lev, key=lambda x: x["expo"]))
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---------------------------------------------------------------- scan around the near miss
    scan = [r for r in std if r["group"] in ("scan",) or r["run"].startswith("(d) 5 slots")]
    scan_h = {h["run"]: h for h in held}
    print("\n=== NEIGHBOURHOOD SCAN, 5 slots (is any winner a smooth optimum or one lucky cell?) ===")
    print(pd.DataFrame([{"run": r["run"], "cagr": r["cagr"], "sharpe": r["sharpe"],
                         "maxdd": r["maxdd"], "kills": r["kills"],
                         "ho_cagr": scan_h[r["run"]]["cagr"],
                         "ho_sharpe": scan_h[r["run"]]["sharpe"]}
                        for r in sorted(scan, key=lambda x: x["run"])])
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---------------------------------------------------------------- significance for the leaders
    winners = [r["run"] for r in rows if r["vs_recomputed"] == "PASS"]
    lead = sorted([r for r in std[1:] if r["sharpe"] >= b_std["sharpe"] * 0.98],
                  key=lambda x: -x["calmar"])[:4]
    sig = [{"run": "baseline", **measure(panel, s, with_p=True)}]
    look = {v[0]: v for v in variants}
    for r in lead:
        _, st, mn, col, _ = look[r["run"]]
        sig.append({"run": r["run"], **measure(panel, st, None, mn, col, with_p=True)})
    print("\n=== RANDOM-BASELINE SIGNIFICANCE (standard panel, leaders only) ===")
    print(pd.DataFrame(sig)[["run", "cagr", "sharpe", "maxdd", "p_coin", "p_time"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---------------------------------------------------------------- frontier: leverage or alpha?
    e = np.array([r["expo"] for r in std])
    cg = np.array([r["cagr"] for r in std])
    dd = np.array([-r["maxdd"] for r in std])
    sh = np.array([r["sharpe"] for r in std])
    frontier = {"corr_expo_cagr": float(np.corrcoef(e, cg)[0, 1]),
                "corr_expo_maxdd": float(np.corrcoef(e, dd)[0, 1]),
                "corr_expo_sharpe": float(np.corrcoef(e, sh)[0, 1]),
                "cagr_per_expo_slope": float(np.polyfit(e, cg, 1)[0]),
                "maxdd_per_expo_slope": float(np.polyfit(e, dd, 1)[0]),
                "baseline_sharpe_rank": int(sum(1 for x in sh if x > b_std["sharpe"]) + 1),
                "n_variants": len(std) - 1}
    print("\n=== FRONTIER: is the whole dimension just leverage? ===")
    for k, v in frontier.items():
        print(f"  {k:24s} {v}")
    print(f"  baseline Sharpe {b_std['sharpe']:.4f} ranks "
          f"{frontier['baseline_sharpe_rank']} of {len(std)} runs tested")

    # The nuance that matters for an honest read: on the RECENT half alone, more slots is
    # consistently better on return AND on Sharpe. It is the full history plus the drawdown clause
    # that rejects it, i.e. a split-sample contradiction rather than a flat negative.
    ho_only = [{"run": h["run"], "ho_cagr": h["cagr"], "ho_sharpe": h["sharpe"],
                "ho_dd": h["maxdd"], "std_cagr": r["cagr"], "std_sharpe": r["sharpe"],
                "std_dd": r["maxdd"], "dd_gap_vs_base": h["maxdd"] - b_held["maxdd"]}
               for r, h in zip(std[1:], held[1:])
               if h["cagr"] >= b_held["cagr"] and h["sharpe"] >= b_held["sharpe"]]
    print("\n=== BEATS BASELINE ON CAGR *AND* SHARPE ON THE HELD-OUT HALF ALONE ===")
    print(pd.DataFrame(ho_only).to_string(index=False, float_format=lambda x: f"{x:.4f}")
          if ho_only else "  none")

    # Is there anything at all that raises CAGR without deepening the drawdown, on either panel?
    free = [r["run"] for r in std[1:]
            if r["cagr"] > b_std["cagr"] and r["maxdd"] >= b_std["maxdd"]]
    print("\nvariants with MORE CAGR and NO deeper drawdown (standard panel):", free or "NONE")

    print("\nvariants that PASS the strict ship rule vs the recomputed baseline:",
          winners or "NONE")

    out = {"baseline_recomputed": {"standard": b_std, "held_out": b_held},
           "baseline_stated_research_log": STATED,
           "standard": std, "held_out": held, "ship_rule": rows,
           "risk_return_curve": curve, "conviction_vs_matched_exposure": pairs,
           "significance": sig, "winners_strict": winners, "leverage_check": lev,
           "frontier": frontier, "held_out_only_winners": ho_only,
           "free_lunch_candidates": free,
           "notes": ("Shipped flags abs_momentum+volume_min0.8+taker_min0.5. The RESEARCH_LOG "
                     "baseline (398 trades) does not reproduce on the current kline files (366 "
                     "trades); the recomputed baseline is the bar used, and it is harder on "
                     "Sharpe and drawdown than the stated one.")}
    (ROOT / "reports" / "explore_portfolio.json").write_text(
        json.dumps(out, indent=2, default=str))
    print("\nwrote reports/explore_portfolio.json")


if __name__ == "__main__":
    main()
