"""Relationship features: how a coin sits against BTC, the market, and its own last few bars.

The shipped feature set (features.FEATURE_COLS) describes each coin mostly in isolation - its own
momentum, volatility and volume, plus two BTC numbers. Every ML rejection in RESEARCH_LOG.md was
scored on that set. These columns add the connections between series, all derived from the same
Binance klines and nothing else:

  own microstructure   where the bar closed inside its range, the last bars' returns (short-term
                       reversal or continuation), volatility expanding or compressing, distance
                       from the 30-day high and low, turnover against its own norm, change in
                       taker-buy share
  BTC lead-lag         BTC's last bar and last day, its volatility and taker flow - on the bar
                       BEFORE the one being predicted, so a coin that follows BTC with a lag has
                       something to follow
  market state         breadth (share of coins above their 50-day average, share up on the day),
                       cross-sectional dispersion, the median coin's return, and each coin's
                       return relative to that median
  co-movement          rolling 30-day beta and correlation to BTC
  calendar             4-hour slot of the day and day of the week (UTC)

CAUSALITY. Every column at bar t is a function of rows at or before bar t: rolling windows and
shifts run within a symbol, cross-sectional statistics run within a single bar_time, and BTC
context is joined on the same bar_time. `assert_causal` proves it the only way that counts -
rebuild on a panel truncated at T and require every row at or before T to come out identical.
A feature that used the future would change when the future is removed.
"""

import numpy as np
import pandas as pd

BARS_PER_DAY = 6
W30 = 30 * BARS_PER_DAY

DOTS_COLS = [
    # own microstructure
    "ret_4h", "ret_4h_l1", "ret_4h_l2", "bar_pos", "vol_ratio_7_30", "dd_30d", "up_30d",
    "vol_surge", "taker_chg_1d",
    # BTC lead-lag
    "btc_ret_4h", "btc_ret_4h_l1", "btc_ret_1d", "btc_ret_30d", "btc_vol_7d", "btc_bar_pos",
    "btc_taker_ratio",
    # market state
    "breadth_sma50", "breadth_up_1d", "disp_1d", "mkt_ret_1d", "mkt_ret_7d", "rel_mkt_1d",
    # co-movement
    "beta_30d", "corr_30d",
    # calendar
    "slot", "dow",
]

XRANK = ["ret_4h", "bar_pos", "vol_surge", "dd_30d", "beta_30d", "taker_chg_1d"]


def add_dots(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.sort_values(["symbol", "bar_time"]).reset_index(drop=True).copy()
    g = p.groupby("symbol", sort=False)

    # ---- own microstructure ---------------------------------------------------------------
    if "ret_4h" not in p.columns:
        p["ret_4h"] = g["close"].pct_change()
    p["ret_4h_l1"] = g["ret_4h"].shift(1)
    p["ret_4h_l2"] = g["ret_4h"].shift(2)
    rng = (p["high"] - p["low"])
    p["bar_pos"] = ((p["close"] - p["low"]) / rng.where(rng > 0)).clip(0, 1)
    p["vol_ratio_7_30"] = p["vol_7d"] / p["vol_30d"].where(p["vol_30d"] > 0)
    hi30 = g["high"].transform(lambda x: x.rolling(W30, min_periods=W30 // 2).max())
    lo30 = g["low"].transform(lambda x: x.rolling(W30, min_periods=W30 // 2).min())
    p["dd_30d"] = p["close"] / hi30 - 1
    p["up_30d"] = p["close"] / lo30.where(lo30 > 0) - 1
    qv_mean = g["quote_volume"].transform(lambda x: x.rolling(W30, min_periods=W30 // 2).mean())
    p["vol_surge"] = np.log((p["quote_volume"] / qv_mean.where(qv_mean > 0)).clip(lower=1e-6))
    p["taker_chg_1d"] = p["taker_ratio"] - g["taker_ratio"].shift(BARS_PER_DAY)

    # ---- BTC context, joined on the same bar ----------------------------------------------
    btc = p.loc[p["symbol"] == "BTCUSDT"].set_index("bar_time")
    ctx = pd.DataFrame({
        "btc_ret_4h": btc["ret_4h"],
        "btc_ret_4h_l1": btc["ret_4h"].shift(1),
        "btc_vol_7d": btc["vol_7d"],
        "btc_bar_pos": btc["bar_pos"],
        "btc_taker_ratio": btc["taker_ratio"],
    })
    for col in ("btc_ret_1d", "btc_ret_30d"):
        if col not in p.columns:
            src = col.replace("btc_", "")
            ctx[col] = btc[src] if src in btc.columns else np.nan
    p = p.drop(columns=[c for c in ctx.columns if c in p.columns]).join(ctx, on="bar_time")

    # ---- market state: statistics WITHIN one bar_time, which contains no future -------------
    gb = p.groupby("bar_time", sort=False)
    p["breadth_sma50"] = gb["dist_sma50"].transform(lambda x: (x > 0).mean())
    p["breadth_up_1d"] = gb["ret_1d"].transform(lambda x: (x > 0).mean())
    p["disp_1d"] = gb["ret_1d"].transform("std")
    p["mkt_ret_1d"] = gb["ret_1d"].transform("median")
    p["mkt_ret_7d"] = gb["ret_7d"].transform("median")
    p["rel_mkt_1d"] = p["ret_1d"] - p["mkt_ret_1d"]

    # ---- co-movement with BTC over a rolling 30 days ----------------------------------------
    p["_r"] = p["ret_4h"]
    p["_br"] = p["bar_time"].map(btc["ret_4h"])
    beta = pd.Series(np.nan, index=p.index)
    corr = pd.Series(np.nan, index=p.index)
    for _, idx in p.groupby("symbol", sort=False).groups.items():
        d = p.loc[idx]
        cov = d["_r"].rolling(W30, min_periods=W30 // 2).cov(d["_br"])
        var = d["_br"].rolling(W30, min_periods=W30 // 2).var()
        beta.loc[idx] = (cov / var.where(var > 0)).to_numpy()
        corr.loc[idx] = d["_r"].rolling(W30, min_periods=W30 // 2).corr(d["_br"]).to_numpy()
    p["beta_30d"], p["corr_30d"] = beta.clip(-5, 5), corr
    p = p.drop(columns=["_r", "_br"])

    # ---- calendar ---------------------------------------------------------------------------
    p["slot"] = (p["bar_time"].dt.hour // 4).astype(float)
    p["dow"] = p["bar_time"].dt.dayofweek.astype(float)

    return p.sort_values(["bar_time", "symbol"]).reset_index(drop=True)


def design_dots(p: pd.DataFrame, base_design) -> pd.DataFrame:
    """The base design matrix plus the relationship columns and their cross-sectional ranks."""
    X = base_design(p).copy()
    for col in DOTS_COLS:
        if col not in X.columns:
            X[col] = p[col].to_numpy()
    g = p.groupby("bar_time")
    for col in XRANK:
        X[col + "_xrank"] = g[col].rank(pct=True).to_numpy()
    return X.replace([np.inf, -np.inf], np.nan)


def assert_causal(panel: pd.DataFrame, cut_quantile=0.6, tol=1e-12):
    """Rebuild on a panel truncated at T and require every row at or before T to be unchanged.

    Returns the number of (row, column) cells compared. Raises AssertionError naming the first
    column that differs - which would mean that column at time t depends on data after t.
    """
    full = add_dots(panel)
    T = full["bar_time"].quantile(cut_quantile)
    trunc = add_dots(panel[panel["bar_time"] <= T])
    a = full[full["bar_time"] <= T].set_index(["bar_time", "symbol"]).sort_index()
    b = trunc.set_index(["bar_time", "symbol"]).sort_index()
    assert len(a) == len(b), f"row count changed under truncation: {len(a)} vs {len(b)}"
    cells = 0
    for col in DOTS_COLS:
        x, y = a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float)
        same = (np.isnan(x) & np.isnan(y)) | (np.abs(x - y) <= tol)
        assert same.all(), (f"{col} is not causal: {int((~same).sum())} rows at or before {T} "
                            "change when later data is removed")
        cells += len(x)
    return cells
