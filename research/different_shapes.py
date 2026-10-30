"""Different SHAPES of strategy, not more parameters on the current one.

The complaint is structural and correct: the median 18-day window returns +0.00%, only 26.4% are
positive, and exposure is ~18%. The bot spends most of its life in cash by design. Sixteen research
programs have tuned that design; none of them changed its shape.

So this tests shapes. Each is a different ANSWER to "what should the book hold right now", not a
different setting:

    cash_when_unsure   today's bot: hold the strongest names only when the regime permits,
                       otherwise cash                                        (the benchmark)
    always_rotate      ALWAYS hold the top N by 30-day momentum. No regime gate, no breakout,
                       never in cash. This is the direct answer to the complaint: if sitting in
                       cash is the problem, stop sitting in cash.
    always_eq          always hold an equal-weight basket of the whole universe, rebalanced
                       monthly. The dumbest always-invested rule there is, included because any
                       cleverness that cannot beat it is not cleverness.
    reversion          BUY WEAKNESS: hold the most oversold names (worst 30-day return), exit when
                       they revert to their 50-day average. The opposite signal to everything this
                       repository has ever tested, and the one most likely to trade often.
    vol_scaled_btc_eth always hold BTC and ETH, sized inversely to volatility, no trend gate at
                       all - risk control through SIZE instead of through being flat.
    trend_no_cash      the current trend ranking, but when nothing qualifies hold BTC rather than
                       cash. A minimal change that removes only the idleness.

Judged on what the complaint is actually about, alongside return:
    exposure           fraction of capital at work
    pct_months_green   how often the operator sees progress
    median 18d window  the number that is currently +0.00%

Pit panel, survivorship-free, real costs, next-bar fills. Full sample and held-out 2022-2026.

    python research/different_shapes.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from app import backtest, config  # noqa: E402
from app.flags import Flags  # noqa: E402

import modern_era as me  # noqa: E402

BPD = 6
BPY = BPD * 365
TOP_N = 5
HELD = "2022-01-01"


def panel_matrices(panel):
    """Wide price / score matrices, one column per coin, so a basket is a weight row per bar."""
    close = panel.pivot_table(index="bar_time", columns="symbol", values="close")
    elig = panel.pivot_table(index="bar_time", columns="symbol", values="eligible").fillna(0) > 0
    ret30 = panel.pivot_table(index="bar_time", columns="symbol", values="ret_30d")
    d50 = panel.pivot_table(index="bar_time", columns="symbol", values="dist_sma50")
    vol = panel.pivot_table(index="bar_time", columns="symbol", values="vol_30d")
    return close, elig, ret30, d50, vol


def weights_from_rank(score, elig, n, ascending=False):
    """Equal weight across the best n eligible names each bar, decided on the PREVIOUS bar."""
    s = score.where(elig)
    rank = s.rank(axis=1, ascending=ascending, method="first")
    w = (rank <= n).astype(float)
    w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return w.shift(1).fillna(0.0)


def run_weights(close, w, cost):
    """Compound a weight matrix against per-bar returns, charging cost on every change."""
    r = close.pct_change().fillna(0.0).clip(-0.95, 5.0)
    port = (r * w).sum(axis=1)
    turn = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    eq = (1.0 + port - turn * cost).cumprod()
    return eq, w.sum(axis=1)


def stats(eq, expo, label):
    eq = eq.dropna()
    r = np.log(eq.clip(lower=1e-9)).diff().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    mon = eq.resample("ME").last().pct_change().dropna()
    w18 = (eq.shift(-18 * BPD) / eq - 1).dropna()
    return {"shape": label,
            "cagr": float(eq.iloc[-1] ** (1 / yrs) - 1) if yrs > 0 else np.nan,
            "sharpe": float(r.mean() / r.std() * np.sqrt(BPY)) if r.std() > 0 else 0.0,
            "maxdd": float((eq / eq.cummax() - 1).min()),
            "exposure": float(expo.mean()),
            "pct_months_green": float((mon > 0).mean()) if len(mon) else np.nan,
            "median_18d": float(w18.median()) if len(w18) else np.nan}


def main():
    s = config.load()
    cost = s.fee_rate + s.slippage
    panel = me.build_pit(s)
    out_all = {}

    for era, start in (("full 2018-2026", None), ("held-out 2022-2026", HELD)):
        p = panel[panel["bar_time"] >= pd.Timestamp(start, tz="UTC")] if start else panel
        close, elig, ret30, d50, vol = panel_matrices(p)
        rows = []

        # --- the benchmark: the real engine, as configured today
        r = backtest.run(p, s, flags=Flags(abs_momentum=True, volume_min=0.8,
                                           funding_max_pct=0.8, trend_core=0.25,
                                           trend_core_symbols="BTCUSDT,ETHUSDT",
                                           core_dual_ma=True))
        eq = r.equity / r.equity.iloc[0]
        rows.append(stats(eq, r.exposure, "cash_when_unsure (today)"))

        # --- always rotate into the strongest, never cash
        w = weights_from_rank(ret30, elig, TOP_N)
        rows.append(stats(*run_weights(close, w, cost), "always_rotate top5"))

        # --- the dumbest always-invested rule: equal weight, monthly rebalance
        ew = elig.astype(float)
        ew = ew.div(ew.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        month = ew.groupby(ew.index.to_period("M")).transform("first").shift(1).fillna(0.0)
        rows.append(stats(*run_weights(close, month, cost), "always_eq basket"))

        # --- buy weakness instead of strength
        wr = weights_from_rank(ret30, elig, TOP_N, ascending=True)
        rows.append(stats(*run_weights(close, wr, cost), "reversion (buy weakness)"))

        # --- always long BTC+ETH, sized by volatility, no trend gate
        pair = [c for c in ("BTCUSDT", "ETHUSDT") if c in close.columns]
        inv = (0.40 / vol[pair]).clip(0.0, 1.0).fillna(0.0)
        inv = inv.div(max(len(pair), 1)).shift(1).fillna(0.0)
        rows.append(stats(*run_weights(close[pair], inv, cost), "vol_scaled BTC+ETH"))

        # --- the current ranking, but hold BTC instead of cash when nothing qualifies
        wt = weights_from_rank(ret30.where(d50 > 0), elig, TOP_N)
        idle = wt.sum(axis=1) < 0.5
        if "BTCUSDT" in wt.columns:
            wt.loc[idle, :] = 0.0
            wt.loc[idle, "BTCUSDT"] = 1.0
        rows.append(stats(*run_weights(close, wt, cost), "trend_no_cash"))

        df = pd.DataFrame(rows)
        out_all[era] = df.to_dict("records")
        print(f"=== {era} ===")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print()

    (ROOT / "reports" / "different_shapes.json").write_text(
        json.dumps(out_all, indent=2, default=str))


if __name__ == "__main__":
    main()
