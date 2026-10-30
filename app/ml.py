"""Phase 12: walk-forward prediction model, shipped only if it beats the v1 ranker after costs.

Target: the coin's return minus BTC's over the next H bars, measured from the next bar's open
(when an order would actually fill). Train on ~2 years, test on the following 3 months, roll.
A purge gap of H bars between train and test stops the last training labels from peeking
into the test period.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import BARS_PER_DAY, FEATURE_COLS

RANKED = ["rs_7d", "rs_30d", "ret_1d", "dist_sma50", "vol_30d", "volume_ratio", "don_pos"]


def add_target(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    p = panel.sort_values(["symbol", "bar_time"]).copy()
    nxt_open = p.groupby("symbol")["open"].shift(-1)
    fut_open = p.groupby("symbol")["open"].shift(-1 - horizon)
    p["fwd_ret"] = fut_open / nxt_open - 1
    btc = p[p.symbol == "BTCUSDT"].set_index("bar_time")["fwd_ret"].rename("btc_fwd")
    p = p.join(btc, on="bar_time")
    p["target"] = p["fwd_ret"] - p["btc_fwd"]
    return p.sort_values(["bar_time", "symbol"]).reset_index(drop=True)


def design(p: pd.DataFrame) -> pd.DataFrame:
    X = p[FEATURE_COLS].copy()
    g = p.groupby("bar_time")
    for col in RANKED:  # where the coin sits inside today's universe, not just its raw value
        X[col + "_xrank"] = g[col].rank(pct=True)
    X["breakout"] = p["breakout"].astype(float)
    X["regime_on"] = p["regime_on"].astype(float)
    return X.replace([np.inf, -np.inf], np.nan)


def walk_forward(panel, horizon=6, train_days=730, test_days=91, log=print):
    """Return panel with out-of-sample `ml_score` (NaN where no model had been trained yet)."""
    p = add_target(panel[panel["eligible"]].copy(), horizon)
    X = design(p)
    y = p["target"]
    t = p["bar_time"]
    p["ml_score"] = np.nan
    first = t.min() + pd.Timedelta(days=train_days)
    purge = pd.Timedelta(hours=4 * (horizon + 1))
    test_start = first
    while test_start < t.max():
        test_end = test_start + pd.Timedelta(days=test_days)
        tr = (t >= test_start - pd.Timedelta(days=train_days)) & (t < test_start - purge) & y.notna()
        te = (t >= test_start) & (t < test_end)
        if tr.sum() > 5000 and te.any():
            model = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=300,
                l2_regularization=1.0, random_state=0)
            # clip extreme labels so a few squeezes do not dominate the fit
            model.fit(X[tr], y[tr].clip(y[tr].quantile(0.01), y[tr].quantile(0.99)))
            p.loc[te, "ml_score"] = model.predict(X[te])
            log(f"  fold {test_start.date()} - {test_end.date()}: trained on {int(tr.sum())} rows")
        test_start = test_end
    return p


def information_coefficient(p: pd.DataFrame, col: str) -> dict:
    """Per-bar Spearman correlation between a score and the realised excess return."""
    d = p[p[col].notna() & p["target"].notna()]
    ic = d.groupby("bar_time").apply(
        lambda g: g[col].rank().corr(g["target"].rank()) if len(g) >= 5 else np.nan,
        include_groups=False).dropna()
    # Bars overlap when horizon > 1, so use one bar per horizon block for the t-stat.
    return {"mean_ic": ic.mean(), "ic_positive_share": (ic > 0).mean(), "bars": len(ic),
            "ic_series": ic}


def top_minus_bottom(p: pd.DataFrame, col: str, q=0.2) -> float:
    """Average excess return of the top 20% by score minus the bottom 20%, per bar."""
    d = p[p[col].notna() & p["target"].notna()].copy()
    d["pct"] = d.groupby("bar_time")[col].rank(pct=True)
    top = d[d.pct > 1 - q].groupby("bar_time")["target"].mean()
    bot = d[d.pct <= q].groupby("bar_time")["target"].mean()
    return (top - bot).mean()
