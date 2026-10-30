"""What to do when the 200-day rule says no - the long stretches in cash.

The trend core holds BTC while it is above its own 200-day average. The complaint is about the rest
of the time: in 2022 the book took ZERO trades for twelve months. That is the regime filter doing
its job, and it is also a year of capital earning nothing.

So: measure the idle periods honestly, then test whether any rule earns anything in them.

    IDLE = BTC below its own 200-day average, decided causally from closed bars.

FIVE CANDIDATE FALLBACKS, each only allowed to act while IDLE:

    cash            do nothing (the baseline, and the thing to beat)
    sma50           hold BTC when it is above its 50-day average even though it is below its 200d.
                    Catches the start of a recovery early, at the cost of being whipsawed in a
                    downtrend - which is precisely what the 200-day filter exists to avoid, so this
                    is the honest test of whether that caution costs more than it saves.
    sma50_half      the same at half size: if the edge is real but noisy, smaller is the fix
    bounce          mean reversion: buy a coin trading far below its own 50-day average, exit when
                    it returns to it. Bear markets are where mean reversion is supposed to work.
    alt_rs          hold the strongest alt by 30-day relative strength, regardless of BTC. Some
                    names trend through a bear; this asks whether enough of them do.

THE BAR THESE MUST CLEAR, declared first:
  1. make money DURING the idle windows (if it loses there it is strictly worse than cash)
  2. not make the FULL-PERIOD drawdown worse than the core-only configuration
  3. survive 2x costs - a bear-market rule trades against the trend and will be whipsawed

ON SAMPLING: the idle windows are 2018-2019, most of 2022, and stretches of 2024-2025. The
2025-2026 holdout was declared spent in P9, so a rule that only works there is not believable.
Each candidate is therefore reported per-window as well as in aggregate: a fallback that earns its
keep in ONE window is a curve fit, and the per-window table is what exposes that.

    python research/idle_regime.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from app import config  # noqa: E402

import modern_era as me  # noqa: E402

BPD = 6                     # 4h bars per day in the panel
BPY = BPD * 365
FEE, SLIP = None, None      # filled from settings in main()


def idle_mask(btc):
    """True where BTC closed below its own 200-day average, known from the previous bar."""
    sma = btc["close"].rolling(200 * BPD, min_periods=200 * BPD).mean()
    return (btc["close"] < sma).shift(1).fillna(False)


def windows(mask, index, min_days=30):
    """Contiguous idle stretches of at least `min_days`, as (start, end) timestamps."""
    out, start = [], None
    vals = mask.to_numpy()
    for i, v in enumerate(vals):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if (index[i] - index[start]).days >= min_days:
                out.append((index[start], index[i]))
            start = None
    if start is not None and (index[-1] - index[start]).days >= min_days:
        out.append((index[start], index[-1]))
    return out


def equity_from_weights(ret, w, turn_cost):
    """Compound a weight series against per-bar returns, charging cost on every weight change."""
    turn = w.diff().abs().fillna(w.abs())
    return np.exp((ret * w - turn * turn_cost).cumsum())


def stats(eq, w, label):
    eq = eq.dropna()
    if len(eq) < 3:
        return {"rule": label, "return": 0.0, "sharpe": 0.0, "maxdd": 0.0, "exposure": 0.0}
    r = np.log(eq).diff().dropna()
    return {"rule": label,
            "return": float(eq.iloc[-1] / eq.iloc[0] - 1),
            "sharpe": float(r.mean() / r.std() * np.sqrt(BPY)) if r.std() > 0 else 0.0,
            "maxdd": float((eq / eq.cummax() - 1).min()),
            "exposure": float(w.mean())}


def build_rules(panel, btc, idle):
    """Target weight per bar for each candidate, zero outside the idle windows."""
    c = btc["close"]
    sma50 = c.rolling(50 * BPD, min_periods=50 * BPD).mean()
    above50 = (c > sma50).shift(1).fillna(False)

    rules = {}
    rules["cash"] = pd.Series(0.0, index=c.index)
    rules["sma50"] = (above50 & idle).astype(float)
    rules["sma50_half"] = rules["sma50"] * 0.5

    # bounce: BTC more than 15% below its own 50-day average, exit when it gets back to it.
    # Expressed as a state so it needs no position bookkeeping: "deeply below" -> long.
    deep = (c / sma50 - 1).shift(1)
    rules["bounce"] = ((deep < -0.15) & idle).astype(float)
    return rules


def alt_rs_weight(panel, idle_index, idle):
    """Equal weight in the single best alt by 30-day relative strength, while idle.

    Rebuilt per bar from the panel's own rs_30d column, so it uses the same definition of
    relative strength the live ranker uses rather than a second one invented here.
    """
    el = panel[panel["eligible"] & (panel["symbol"] != "BTCUSDT")]
    if el.empty:
        return None, None
    best = (el.sort_values(["bar_time", "rs_30d"], ascending=[True, False])
              .groupby("bar_time").head(1).set_index("bar_time"))
    ret = (panel.pivot_table(index="bar_time", columns="symbol", values="close")
           .pct_change().fillna(0.0))
    pick = best["symbol"].reindex(idle_index).shift(1)
    r = pd.Series(0.0, index=idle_index)
    for t, s in pick.dropna().items():
        if s in ret.columns and t in ret.index:
            r.loc[t] = ret.at[t, s]
    # positive rs only: holding the "least bad" alt in a crash is not a strategy. Multiplied
    # rather than &=, because these are already float weights and `&` on floats is a TypeError.
    strong = (best["rs_30d"].reindex(idle_index).shift(1).fillna(-1) > 0).astype(float)
    w = (idle.reindex(idle_index).fillna(False) & pick.notna()).astype(float) * strong
    return np.log1p(r.clip(-0.95, 10)), w


def main():
    s = config.load()
    cost = s.fee_rate + s.slippage
    panel = me.build_pit(s)
    btc = (panel[panel["symbol"] == "BTCUSDT"].sort_values("bar_time")
           .set_index("bar_time"))
    idle = idle_mask(btc)
    idx = btc.index

    total_days = (idx[-1] - idx[0]).days
    idle_days = int(idle.sum() / BPD)
    print(f"panel {idx[0].date()} .. {idx[-1].date()}  ({total_days} days)")
    print(f"BTC below its 200-day average on {idle.mean():.1%} of bars "
          f"(~{idle_days} days). That is the time the book currently sits in cash.\n")

    wins = windows(idle, idx)
    print("idle stretches of 30 days or more:")
    for a, b in wins:
        px = btc["close"]
        print(f"  {a.date()} .. {b.date()}  ({(b - a).days:>4} days)  "
              f"BTC {px.loc[b] / px.loc[a] - 1:+7.1%} over the window")

    ret = np.log(btc["close"]).diff().fillna(0.0)
    rules = build_rules(panel, btc, idle)
    rs_ret, rs_w = alt_rs_weight(panel, idx, idle)
    if rs_w is not None:
        rules["alt_rs"] = rs_w

    # ---------------------------------------------------------------- aggregate
    print("\n=== across ALL idle time, 1x costs ===")
    rows = []
    for k, w in rules.items():
        r = rs_ret if k == "alt_rs" else ret
        m = stats(equity_from_weights(r[idle], w[idle], cost), w[idle], k)
        rows.append(m)
    agg = pd.DataFrame(rows)
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---------------------------------------------------------------- per window
    print("\n=== per idle window (a rule that only works in one of these is a curve fit) ===")
    per = []
    for a, b in wins:
        m = (idx >= a) & (idx <= b)
        row = {"window": f"{a.date()}..{b.date()}"}
        for k, w in rules.items():
            r = rs_ret if k == "alt_rs" else ret
            e = equity_from_weights(r[m], w[m], cost)
            row[k] = float(e.iloc[-1] / e.iloc[0] - 1) if len(e) > 2 else 0.0
        per.append(row)
    pdf = pd.DataFrame(per)
    print(pdf.to_string(index=False, float_format=lambda x: f"{x:+.2%}"))

    # ---------------------------------------------------------------- cost stress
    print("\n=== 2x costs, across all idle time ===")
    rows2 = []
    for k, w in rules.items():
        r = rs_ret if k == "alt_rs" else ret
        rows2.append(stats(equity_from_weights(r[idle], w[idle], cost * 2), w[idle], k))
    print(pd.DataFrame(rows2).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---------------------------------------------------------------- verdict
    print("\n=== verdict ===")
    names = [k for k in rules if k != "cash"]
    for k in names:
        a = agg[agg.rule == k].iloc[0]
        wins_pos = sum(1 for row in per if row[k] > 0)
        s2 = [x for x in rows2 if x["rule"] == k][0]
        ok = a["return"] > 0 and wins_pos >= max(2, len(per) - 1) and s2["return"] > 0
        print(f"  {k:12s} total {a['return']:+7.2%}  positive in {wins_pos}/{len(per)} windows  "
              f"2x-cost {s2['return']:+7.2%}  -> {'CANDIDATE' if ok else 'REJECT'}")

    (ROOT / "reports" / "idle_regime.json").write_text(json.dumps(
        {"idle_share": float(idle.mean()), "windows": [[str(a), str(b)] for a, b in wins],
         "aggregate": rows, "per_window": per, "cost2x": rows2}, indent=2, default=str))


if __name__ == "__main__":
    main()
