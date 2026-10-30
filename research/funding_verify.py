"""Try to kill the funding-crowding filter.

One variant passed the ship rule: skip a candidate whose 7-day funding sits in the top 20% of the
universe that bar. Before that can be believed it has to survive the four tests this project
adopted after two earlier passes turned out to be noise:

  1. GRID    - do neighbouring thresholds also work, or is it one lucky cell?
  2. PLACEBO - how often does an information-free filter of the same shape pass? If the real one
               sits inside that distribution it is not evidence.
  3. PERIOD  - is the gain spread across the history or concentrated in one quarter?
  4. COSTS   - does it survive 1.5x and 2x the assumed slippage and fees?

    python research/funding_verify.py
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import backtest, config, features  # noqa: E402
from app.flags import Flags  # noqa: E402
from funding_test import attach_funding  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SHIPPED = Flags(abs_momentum=True, volume_min=0.8, taker_min=0.5)
HELD_OUT = "2022-01-01"
CUT = 0.8
SEED = 11


def run(panel, s, start, size_col=None):
    res = backtest.run(panel, s, flags=SHIPPED, start=start, size_scale_col=size_col)
    m = backtest.metrics(res.equity, res.trades, res.exposure)
    h = backtest.run(panel, s, flags=SHIPPED, start=HELD_OUT, size_scale_col=size_col)
    mh = backtest.metrics(h.equity, h.trades, h.exposure)
    return m, mh, res


def passes(b, bh, c, ch):
    return (c["cagr"] >= b["cagr"] and c["sharpe"] >= b["sharpe"]
            and c["max_drawdown"] >= b["max_drawdown"] and ch["cagr"] >= bh["cagr"]
            and ch["sharpe"] >= bh["sharpe"] and ch["max_drawdown"] >= bh["max_drawdown"])


def apply_cut(panel, mask):
    v = panel.copy()
    v["breakout"] = v["breakout"] & ~mask.fillna(False)
    return v


def main():
    s = config.load()
    u = json.loads((ROOT / "reports" / "_parity_universe.json").read_text())
    p = attach_funding(features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars), s)
    have = p.groupby("bar_time")["funding_7d"].apply(lambda x: x.notna().sum())
    start = have[have >= 8].index.min()

    base, baseh, base_res = run(p, s, start)
    print(f"baseline  CAGR {base['cagr']:.4f}  Sharpe {base['sharpe']:.4f}  "
          f"DD {base['max_drawdown']:.4f}  |  held-out {baseh['cagr']:.4f} / "
          f"{baseh['sharpe']:.4f} / {baseh['max_drawdown']:.4f}\n")

    # 1. GRID -----------------------------------------------------------------
    print("1. GRID - neighbouring thresholds")
    grid = []
    for cut in (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        c, ch, _ = run(apply_cut(p, p["funding_pct"] > cut), s, start)
        ok = passes(base, baseh, c, ch)
        grid.append({"cut": cut, "cagr": c["cagr"], "sharpe": c["sharpe"],
                     "maxdd": c["max_drawdown"], "ho_cagr": ch["cagr"],
                     "ho_sharpe": ch["sharpe"], "pass": ok})
        print(f"   {cut:.2f}  CAGR {c['cagr']:.4f}  Sharpe {c['sharpe']:.4f}  "
              f"DD {c['max_drawdown']:.4f}  held-out {ch['cagr']:.4f}  {'PASS' if ok else ''}")
    n_pass = sum(g["pass"] for g in grid)
    print(f"   -> {n_pass} of {len(grid)} thresholds pass\n")

    # 2. PLACEBO --------------------------------------------------------------
    print("2. PLACEBO - information-free filters of the same shape")
    rng = np.random.default_rng(SEED)
    real = apply_cut(p, p["funding_pct"] > CUT)
    real_m, real_h, _ = run(real, s, start)
    removed = int((p["breakout"] & (p["funding_pct"] > CUT).fillna(False)).sum())
    print(f"   the real filter removes {removed:,} breakout bars")
    hits = 0
    draws = 40
    for i in range(draws):
        # same shape, no information: a constant random key per symbol, re-ranked each bar
        keys = {sym: rng.random() for sym in p["symbol"].unique()}
        fake = p["symbol"].map(keys) + rng.normal(0, 0.01, len(p))
        pct = pd.Series(fake, index=p.index).groupby(p["bar_time"]).rank(pct=True)
        c, ch, _ = run(apply_cut(p, pct > CUT), s, start)
        if passes(base, baseh, c, ch):
            hits += 1
    print(f"   {hits} of {draws} placebo filters also pass the ship rule "
          f"({hits / draws:.0%} false-positive rate)\n")

    # 3. PERIOD ---------------------------------------------------------------
    print("3. PERIOD - where does the gain come from")
    eq_b = base_res.equity
    _, _, real_res = run(real, s, start)
    eq_r = real_res.equity.reindex(eq_b.index).ffill()
    rel = (eq_r / eq_r.iloc[0]) / (eq_b / eq_b.iloc[0])
    yearly = rel.resample("YE").last().pct_change().fillna(rel.resample("YE").last().iloc[0] - 1)
    print("   relative performance by year (positive = filter helped):")
    for d, v in yearly.items():
        print(f"     {d.year}  {v:+.2%}")
    best = yearly.abs().idxmax()
    share = abs(yearly[best]) / yearly.abs().sum() if yearly.abs().sum() else 0
    print(f"   largest single year {best.year} carries {share:.0%} of the total movement\n")

    # 4. COSTS ----------------------------------------------------------------
    print("4. COSTS - does it survive worse execution")
    for mult in (1.5, 2.0):
        s2 = copy.copy(s)
        s2.slippage_bps = s.slippage_bps * mult
        s2.fee_rate = s.fee_rate * mult
        b2, bh2, _ = run(p, s2, start)
        c2, ch2, _ = run(real, s2, start)
        ok = passes(b2, bh2, c2, ch2)
        print(f"   {mult:g}x costs: baseline {b2['cagr']:.4f} vs filter {c2['cagr']:.4f}  "
              f"held-out {bh2['cagr']:.4f} vs {ch2['cagr']:.4f}  -> {'PASS' if ok else 'FAIL'}")

    out = {"grid": grid, "placebo_hits": hits, "placebo_draws": draws,
           "yearly_relative": {str(d.year): float(v) for d, v in yearly.items()}}
    (ROOT / "reports" / "funding_verify.json").write_text(json.dumps(out, indent=2, default=str))
    print("\nsaved reports/funding_verify.json")


if __name__ == "__main__":
    main()
