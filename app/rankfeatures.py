"""One feature pipeline for the rank model, shared by training and live scoring.

The model was found and validated in research (RESEARCH_LOG.md P22). Deploying it means computing
its 57 inputs a second time, live, from a short tail of recent candles on a 1 GB server. Every
serious defect in this project has been one rule implemented in two places, so the steps that turn
klines into the model's input matrix live here once, and both research/train_rank_model.py and
app/rankscore.py call them:

    klines --features.build_panel--> per-coin features (4h + daily)
           --relfeatures.add_dots--> relationship features
           --attach_members-------> that day's top coins, minus fresh relistings
           --add_market_state-----> breadth, dispersion, market return AMONG MEMBERS
           --design---------------> the model's matrix, in the model's column order

TAIL PARITY. The live scorer keeps TAIL_4H bars of 4h history and TAIL_1D days of daily history per
coin instead of nine years. Nothing in the feature code is cumulative, and the longest windows are
180 4h bars (30 days) and 200 daily bars (the 200-day average), so the newest bar's features are
identical whether computed from a tail or from the full history. tests/test_rank_shadow.py holds
the pipeline to that on real candles: a disagreement is a test failure, not a judgement call.
"""

import numpy as np
import pandas as pd

from . import features, ml, relfeatures

HORIZON = 6                  # bars: the label is the 24-hour return, measured from the next open
TAIL_4H = 260                # > 180-bar windows + one-day shifts + the 180-bar relisting lookback
TAIL_1D = 230                # > 200 daily bars for the 200-day average
RELIST_GAP_DAYS = 7
RELIST_QUIET_BARS = 180


def build(kdir, symbols, donchian):
    """Per-coin and relationship features for every bar of every symbol (BTC is always included).

    `kdir` is any directory of klines in app.data's format: the full research cache for training,
    the live scorer's tail store for scoring. Same code either way.
    """
    syms = sorted(set(symbols) | {"BTCUSDT"})
    return relfeatures.add_dots(features.build_panel(kdir, syms, donchian))


def fresh_relist(full: pd.DataFrame) -> pd.Series:
    """True on rows within RELIST_QUIET_BARS bars after a trading stop longer than a week.

    After a long stop every rolling feature straddles the gap, and a reused ticker can be a
    different coin altogether: LUNAUSDT closed 13 May 2022 at 0.00005 and the same ticker resumed
    18 days later at 1.0. Such rows are neither trained on nor scored.
    """
    f = full.sort_values(["symbol", "bar_time"])
    step = f["bar_time"] - f.groupby("symbol", sort=False)["bar_time"].shift(1)
    gap = step.dt.total_seconds().div(86400) > RELIST_GAP_DAYS
    relist = gap.astype(int).groupby(f["symbol"], sort=False).cumsum()
    since = f.groupby([f["symbol"], relist], sort=False).cumcount()
    return ((relist > 0) & (since < RELIST_QUIET_BARS)).reindex(full.index)


def attach_members(full: pd.DataFrame, member: pd.DataFrame) -> pd.DataFrame:
    """Rows of that day's members, excluding fresh relistings.

    `member` is date x symbol from app.rankuniverse.membership. A 4h bar belongs to the UTC day it
    falls in; that day's membership was decided from days already closed.
    """
    day = (full["bar_time"] - pd.Timedelta(milliseconds=1)).dt.normalize()
    m = member.rename_axis(index="day", columns="symbol").stack().rename("member")
    m = m[m].reset_index()[["day", "symbol"]]
    keyed = full.assign(day=day).merge(m.assign(_m=True), on=["day", "symbol"], how="left")
    keep = keyed["_m"].notna().to_numpy() & ~fresh_relist(full).to_numpy()
    return full.loc[keep].copy()


add_market_state = relfeatures.add_market_state


def design(p: pd.DataFrame, columns=None) -> np.ndarray:
    """The model's input matrix. With `columns`, it must match the model's order exactly."""
    X = relfeatures.design_dots(p, ml.design)
    if columns is not None:
        got = list(X.columns)
        if got != list(columns):
            missing = [c for c in columns if c not in got]
            extra = [c for c in got if c not in columns]
            raise ValueError(f"feature columns differ from the model's: missing {missing}, "
                             f"extra {extra}, or in a different order")
    return X.to_numpy(dtype=float)


def feature_names(p: pd.DataFrame) -> list:
    return list(relfeatures.design_dots(p, ml.design).columns)


def forward_returns(full: pd.DataFrame, k: int = HORIZON) -> pd.Series:
    """k-bar return entered at the next bar's open, computed within runs of consecutive bars.

    A row shift does not know that trading stopped, and across a stop it can read a reused ticker
    as a 20,000-fold gain. A position still open when its run ends is closed at the last price that
    traded before the stop, which is the most a holder could have had. Training labels only.
    """
    f = full.sort_values(["symbol", "bar_time"])
    step = f["bar_time"] - f.groupby("symbol", sort=False)["bar_time"].shift(1)
    run = (step != pd.Timedelta(hours=4)).groupby(f["symbol"], sort=False).cumsum()
    gr = f.groupby([f["symbol"], run], sort=False)
    entry = gr["open"].shift(-1)
    exit_ = gr["open"].shift(-1 - k)
    run_end = gr["bar_time"].transform("max")
    stopped = run_end < (f["bar_time"].max() - pd.Timedelta(days=2))
    exit_ = exit_.where(exit_.notna() | ~stopped, gr["close"].transform("last"))
    return (exit_ / entry - 1).reindex(full.index)
