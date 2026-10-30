"""The bot already computes the best signal it has, and then trades something else with it.

The strategy's regime filter is "BTC above its 200-day average". It uses that as a GATE: when BTC
is above, go and pick altcoin breakouts; when below, sit in cash. A first look at the modern era
said the gate on its own, applied to BTC itself, beat the whole machine:

    2022-2026, survivorship-free     CAGR     Sharpe   maxDD     invested
    the bot (shipped, 302 trades)    + 2.24%   0.263   -14.8%      6.8%
    BTC buy and hold                 +13.39%   0.256   -67.2%    100.0%
    BTC above its 200d SMA, cash     +33.72%   0.889   -36.1%     51.8%

That first look had no fees, no slippage and no rolling-window check, which is exactly the kind of
number that evaporates. This script does it properly:

  COSTS      fee_rate and slippage from config, charged on every switch, plus 1.5x and 2.0x stress.
             A 200-day channel flips rarely, so the cost should be small - but "should be" is how
             the P6 control arm got away with being a no-op, so it is measured and the round trips
             are counted.
  CAUSAL     the signal is the previous bar's close against an SMA of closes up to that bar, so no
             bar uses its own close to decide its own position. Asserted, not assumed.
  BOTH ERAS  2019-2021 and 2022-2026 separately. The interesting claim is not that this wins today
             but that it wins in BOTH, which is what "stands in the future" has to mean. Note in
             advance that it LOST to buy-and-hold in 2019-2021 (+60.50% against +134.62%) - a rule
             that gives up upside in a mania and keeps capital in a bear is the opposite profile to
             the bot's, and it is the profile a $100 book wants.
  ROLLING    every 12-month window, stepped a quarter, from 2019. Share positive and worst window.
  EXPOSURE   reported everywhere. "Stands in the market and does not stop" was the requirement, and
             6.8% of the time is not standing in the market.

VARIANTS, in increasing order of how much they let the rule stay invested:
    hold            BTC, always
    sma200          BTC above its 200d SMA, else cash                      (the bot's own gate)
    sma200_vol      the same, sized to a volatility target, so exposure moves continuously
    sma100 / 50     shorter channels, because the modern era's trends are shorter
    btc_eth         BTC and ETH equally, each gated by its own 200d SMA
    ensemble        average the 50/100/200 signals - position is 0, 1/3, 2/3 or 1, never on/off

This is deliberately not a model. Six research programs and ~655 rejected variants in this
repository have established that added complexity does not pay here; the only honest next question
is whether the simplest thing that stays invested beats the complicated thing that does not.

    python research/btc_core.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402

BPD = 6                      # 4h bars per day
BPY = BPD * 365
ERAS = {"2019-2021": ("2019-01-01", "2022-01-01"),
        "2022-2026 (modern)": ("2022-01-01", None),
        "2025-2026 (holdout)": ("2025-01-01", None),
        "full sample": ("2018-01-01", None)}


def load_close(sym):
    d = pd.read_parquet(ROOT / "data" / "pit" / "klines" / f"{sym}_4h.parquet")
    d["t"] = pd.to_datetime(d["open_time"], unit="ms", utc=True)
    return d.set_index("t").sort_index()["close"].astype(float)


def signals(c):
    """Target weight in [0, 1] per bar, each one causal: decided by bars already closed.

    `.shift(1)` is applied to the comparison, not to the price, so the weight held during a bar was
    knowable before that bar opened. The assertion below is the real guard.
    """
    out = {}
    out["hold"] = pd.Series(1.0, index=c.index)
    for n in (50, 100, 200):
        sma = c.rolling(n * BPD, min_periods=n * BPD).mean()
        out[f"sma{n}"] = (c > sma).shift(1).astype(float)
    out["ensemble"] = (out["sma50"] + out["sma100"] + out["sma200"]) / 3.0
    # volatility-targeted version of the 200d rule: same direction, continuous size
    vol = np.log(c).diff().rolling(30 * BPD).std() * np.sqrt(BPY)
    out["sma200_vol"] = (out["sma200"] * (0.50 / vol).clip(0.25, 1.0)).shift(1)
    return {k: v.reindex(c.index).fillna(0.0) for k, v in out.items()}


def equity(c, w, fee, slip):
    """Compound the weighted return, charging a round-trip cost on every change in weight."""
    r = np.log(c).diff().fillna(0.0)
    turn = w.diff().abs().fillna(w.abs())
    cost = turn * (fee + slip)
    eq = np.exp((r * w - cost).cumsum())
    return eq, float(turn.sum() / 2.0)


def stats(eq, w, label, trips):
    eq = eq.dropna()
    r = np.log(eq).diff().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    sh = r.mean() / r.std() * np.sqrt(BPY) if r.std() > 0 else 0.0
    return {"run": label, "cagr": cagr, "sharpe": sh,
            "maxdd": float((eq / eq.cummax() - 1).min()),
            "mult": float(eq.iloc[-1] / eq.iloc[0]), "exposure": float(w.mean()),
            "round_trips": trips, "years": round(yrs, 2)}


def rolling_windows(c, w, fee, slip, lo="2019-01-01", months=12, step=3):
    idx = c[lo:].index
    rows, start = [], idx[0]
    while start + pd.DateOffset(months=months) <= idx[-1]:
        end = start + pd.DateOffset(months=months)
        cc, ww = c[start:end], w[start:end]
        if len(cc) > 10:
            eq, t = equity(cc, ww, fee, slip)
            rows.append({"from": start.date(), "ret": eq.iloc[-1] / eq.iloc[0] - 1,
                         "exposure": float(ww.mean())})
        start += pd.DateOffset(months=step)
    return pd.DataFrame(rows)


def main():
    s = config.load()
    fee, slip = s.fee_rate, s.slippage
    btc = load_close("BTCUSDT")
    sig = signals(btc)

    # the lookahead guard: the 200d weight must be reproducible from data strictly before the bar
    sma = btc.rolling(200 * BPD, min_periods=200 * BPD).mean()
    manual = (btc.shift(1) > sma.shift(1)).astype(float)
    assert sig["sma200"].fillna(0).equals(manual.fillna(0)), "sma200 weight is not causal"
    print(f"causality check passed; costs {fee:.4%} fee + {slip:.4%} slippage per side\n")

    try:
        eth = load_close("ETHUSDT")
        esig = signals(eth)
    except Exception:
        eth = None

    all_rows = {}
    for era, (lo, hi) in ERAS.items():
        rows = []
        c = btc[lo:hi] if hi else btc[lo:]
        for name, w in sig.items():
            ww = w.reindex(c.index).fillna(0.0)
            eq, trips = equity(c, ww, fee, slip)
            rows.append(stats(eq, ww, name, trips))
        if eth is not None:
            ce = eth[lo:hi] if hi else eth[lo:]
            common = c.index.intersection(ce.index)
            if len(common) > 100:
                # Two 0.5 sleeves summing to a full book. Averaging two separately-compounded
                # half-weight curves - the first version here - silently builds a HALF-INVESTED
                # portfolio (0.5 x 0.5 per leg) and then reports its small drawdown as
                # diversification. The returns are combined per bar instead.
                wb = sig["sma200"].reindex(common).fillna(0.0) * 0.5
                we = esig["sma200"].reindex(common).fillna(0.0) * 0.5
                rb = np.log(c.loc[common]).diff().fillna(0.0)
                re_ = np.log(ce.loc[common]).diff().fillna(0.0)
                turn = wb.diff().abs().fillna(wb.abs()) + we.diff().abs().fillna(we.abs())
                port = rb * wb + re_ * we - turn * (fee + slip)
                eqp = np.exp(port.cumsum())
                rows.append(stats(eqp, wb + we, "btc_eth sma200", float(turn.sum() / 2.0)))
        df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
        all_rows[era] = df.to_dict("records")
        print(f"=== {era} ===")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print()

    print("=== cost stress, modern era ===")
    cs = []
    c = btc["2022-01-01":]
    for mult in (1.0, 1.5, 2.0, 3.0):
        for name in ("sma200", "sma200_vol", "ensemble"):
            ww = sig[name].reindex(c.index).fillna(0.0)
            eq, trips = equity(c, ww, fee * mult, slip * mult)
            cs.append({"costs": f"{mult:g}x", **stats(eq, ww, name, trips)})
    print(pd.DataFrame(cs).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== rolling 12-month windows from 2019 (stepped a quarter) ===")
    rr = []
    for name in ("hold", "sma200", "sma200_vol", "ensemble"):
        w = sig[name]
        d = rolling_windows(btc, w, fee, slip)
        rr.append({"run": name, "windows": len(d),
                   "share_positive": float((d["ret"] > 0).mean()),
                   "median": float(d["ret"].median()), "worst": float(d["ret"].min()),
                   "best": float(d["ret"].max()), "mean_exposure": float(d["exposure"].mean())})
    print(pd.DataFrame(rr).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    (ROOT / "reports" / "btc_core.json").write_text(json.dumps(
        {"eras": all_rows, "cost_stress": cs, "rolling": rr,
         "costs": {"fee": fee, "slippage": slip}}, indent=2, default=str))


if __name__ == "__main__":
    main()
