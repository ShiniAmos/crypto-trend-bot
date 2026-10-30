"""Phase 4: features on every closed 4h bar (section 4.3). Uses only bars closed at or before t."""

import numpy as np
import pandas as pd

from . import data

BARS_PER_DAY = 6
ANNUAL = np.sqrt(BARS_PER_DAY * 365)


def symbol_features(bars4h: pd.DataFrame, bars1d: pd.DataFrame, donchian=20,
                    bpd=BARS_PER_DAY, tf_ms=4 * 3_600_000, exit_donchian=0) -> pd.DataFrame:
    """Own-past features for one symbol. Every value at row t uses rows <= t only.

    `bpd` is bars per day and `tf_ms` the bar length: with the defaults this is the 4h strategy,
    byte for byte. Passing bpd=24 and tf_ms=1h evaluates the SAME wall-clock horizons every hour
    instead of every four, which is the faster-decision variant (the caller must scale `donchian`
    the same way, e.g. 20 four-hour bars = 80 one-hour bars).
    """
    df = bars4h.copy()
    annual = np.sqrt(bpd * 365)
    # Bar time = its scheduled close (open + 4h). Using close_time would give a few outage-era
    # candles odd timestamps that no longer line up with the other coins' bars.
    df["bar_time"] = pd.to_datetime(df["open_time"] + tf_ms, unit="ms", utc=True)
    c, h, l = df["close"], df["high"], df["low"]
    logret = np.log(c).diff()

    df["ret_4h"] = c.pct_change()
    df["ret_1d"] = c.pct_change(bpd)
    df["ret_7d"] = c.pct_change(7 * bpd)
    df["ret_30d"] = c.pct_change(30 * bpd)
    # Donchian channel of the *previous* N bars, so a breakout compares close to the past only.
    df["don_high"] = h.rolling(donchian).max().shift(1)
    df["don_low"] = l.rolling(donchian).min().shift(1)
    df["stop_level"] = l.rolling(donchian).min()  # the 20-bar low including this bar
    df["breakout"] = c > df["don_high"]
    df["breakdown"] = c < df["don_low"]
    df["don_pos"] = (c - df["don_low"]) / (df["don_high"] - df["don_low"])
    if exit_donchian:
        # A separate, wider low channel used only for exits (P4). Entry is unaffected.
        df["don_low_exit"] = l.rolling(exit_donchian).min().shift(1)
        df["breakdown_exit"] = c < df["don_low_exit"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr_pct"] = tr.rolling(max(2, round(14 * bpd / 6))).mean() / c
    df["vol_30d"] = logret.rolling(30 * bpd).std() * annual
    df["vol_7d"] = logret.rolling(7 * bpd).std() * annual
    qv = df["quote_volume"]
    df["volume_ratio"] = qv.rolling(bpd).sum() / (qv.rolling(20 * bpd).mean() * bpd)
    df["taker_ratio"] = (df["taker_buy_volume"].rolling(bpd).sum()
                         / df["volume"].rolling(bpd).sum())
    df["dollar_vol_30d"] = qv.rolling(30 * bpd).sum() / 30
    df["age_bars"] = np.arange(1, len(df) + 1)

    # Daily SMAs, joined as-of: a 4h bar sees the last daily bar that had closed by its close.
    if len(bars1d):
        d = bars1d[["close_time", "close"]].copy()
        d["sma50_d"] = d["close"].rolling(50).mean()
        d["sma200_d"] = d["close"].rolling(200).mean()
        d = d.drop(columns="close").sort_values("close_time")
        df = pd.merge_asof(df.sort_values("close_time"), d, on="close_time", direction="backward")
    else:
        df["sma50_d"] = df["sma200_d"] = np.nan
    df["dist_sma50"] = df["close"] / df["sma50_d"] - 1
    df["dist_sma200"] = df["close"] / df["sma200_d"] - 1
    return df


FEATURE_COLS = ["ret_1d", "ret_7d", "ret_30d", "dist_sma50", "dist_sma200", "don_pos",
                "atr_pct", "vol_30d", "vol_7d", "volume_ratio", "taker_ratio",
                "rs_1d", "rs_7d", "rs_30d", "btc_ret_7d", "btc_dist_sma200"]


def build_panel(klines_dir, symbols, donchian=20, min_age_bars=None, tf="4h",
                exit_donchian=0) -> pd.DataFrame:
    """Long table (bar_time, symbol) with own-past, cross-section and regime columns.

    `tf` selects the decision timeframe. "4h" is the shipped strategy; "1h" evaluates the same
    horizons four times as often (the caller passes the scaled donchian window).
    """
    bpd = {"1h": 24, "4h": 6}[tf]
    tf_ms = {"1h": 3_600_000, "4h": 4 * 3_600_000}[tf]
    min_age_bars = 90 * bpd if min_age_bars is None else min_age_bars
    frames = []
    for sym in symbols:
        b4 = data.load(klines_dir, sym, tf)
        if b4.empty:
            continue
        f = symbol_features(b4, data.load(klines_dir, sym, "1d"), donchian, bpd, tf_ms,
                            exit_donchian)
        f["symbol"] = sym
        frames.append(f)
    panel = pd.concat(frames, ignore_index=True)
    return add_cross_section(panel, min_age_bars)


def add_cross_section(panel: pd.DataFrame, min_age_bars=540) -> pd.DataFrame:
    btc = panel[panel["symbol"] == "BTCUSDT"].set_index("bar_time")
    if btc.empty:
        raise RuntimeError("BTCUSDT history is required for the regime and relative strength")
    panel = panel.join(btc[["ret_1d", "ret_7d", "ret_30d", "dist_sma200"]]
                       .add_prefix("btc_"), on="bar_time")
    panel["rs_1d"] = panel["ret_1d"] - panel["btc_ret_1d"]
    panel["rs_7d"] = panel["ret_7d"] - panel["btc_ret_7d"]
    panel["rs_30d"] = panel["ret_30d"] - panel["btc_ret_30d"]
    # Regime: BTC above its 200-day SMA = risk-on.
    panel["regime_on"] = panel["btc_dist_sma200"] > 0

    panel["eligible"] = (panel["age_bars"] >= min_age_bars) & panel["rs_30d"].notna() \
        & panel["dist_sma50"].notna() & panel["vol_30d"].notna()
    return score(panel)


def score(panel: pd.DataFrame) -> pd.DataFrame:
    """v1 ranker: momentum vs BTC and trend, as cross-sectional percentiles inside the universe."""
    el = panel["eligible"]
    g = panel[el].groupby("bar_time")
    parts = pd.DataFrame(index=panel.index)
    parts.loc[el, "a"] = g["rs_7d"].rank(pct=True)
    parts.loc[el, "b"] = g["rs_30d"].rank(pct=True)
    parts.loc[el, "c"] = g["dist_sma50"].rank(pct=True)
    # Risk-adjusted: 30-day return per unit of volatility.
    ra = panel["ret_30d"] / panel["vol_30d"]
    parts.loc[el, "d"] = ra[el].groupby(panel.loc[el, "bar_time"]).rank(pct=True)
    panel["score"] = 100 * (0.35 * parts["a"] + 0.25 * parts["b"] + 0.2 * parts["c"] + 0.2 * parts["d"])
    panel["rank"] = panel[el].groupby("bar_time")["score"].rank(ascending=False, method="first")
    panel["n_eligible"] = panel.groupby("bar_time")["eligible"].transform("sum")
    return panel.sort_values(["bar_time", "rank"]).reset_index(drop=True)


def latest_table(panel: pd.DataFrame) -> pd.DataFrame:
    t = panel["bar_time"].max()
    cols = ["symbol", "rank", "score", "rs_7d", "ret_7d", "breakout", "breakdown", "dist_sma50",
            "vol_30d", "volume_ratio", "regime_on", "close"]
    return panel[(panel["bar_time"] == t) & panel["eligible"]][cols].sort_values("rank")
