"""The prediction research must not be able to fool itself.

Three properties, each of which a silent failure would turn into a false discovery:

  1. the relationship features use nothing from the future. Proven by truncation: rebuild on a
     panel cut at T and every row at or before T must be unchanged.
  2. the pick edge - the money yardstick - is zero for a model with no selection skill and
     positive for one that has it. A yardstick that reads positive for noise would promote noise.
  3. the walk-forward keeps nothing between folds or between runs: the same configuration scored
     twice, with a different target fitted in between, must produce identical numbers.

WHY THE YARDSTICK IS MONEY. The first prediction grid (20 Nov 2026) flagged "SIGNAL" on 17 of 20
cells, and three of its best results were artefacts: BTC's own rows inflating a "beats BTC" target
(BTC cannot beat itself, so its label is constant), "gains >10%" scoring the same as "falls >10%"
(volatility, not direction), and "is up" at the level bull/bear drift alone gives. AUC rewarded all
three. The spread between the model's top picks and the average coin rewards none of them.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research"))
import dots_features as dots                                     # noqa: E402
import predict_search as ps                                      # noqa: E402


def _panel(n_bars=700, symbols=("BTCUSDT", "AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT"),
           seed=3):
    """A synthetic 4h panel with every column the relationship features read."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2024-01-01", periods=n_bars, freq="4h", tz="UTC")
    frames = []
    for k, sym in enumerate(symbols):
        r = rng.normal(0, 0.01 + 0.003 * k, n_bars)
        close = 100 * np.exp(np.cumsum(r))
        high = close * (1 + np.abs(rng.normal(0, 0.004, n_bars)))
        low = close * (1 - np.abs(rng.normal(0, 0.004, n_bars)))
        d = pd.DataFrame({"bar_time": times, "symbol": sym, "open": close * (1 + rng.normal(0, 0.001, n_bars)),
                          "high": high, "low": low, "close": close,
                          "quote_volume": rng.lognormal(10, 0.5, n_bars),
                          "taker_ratio": rng.uniform(0.3, 0.7, n_bars)})
        s = pd.Series(close)
        d["ret_4h"] = s.pct_change().to_numpy()
        d["ret_1d"] = s.pct_change(6).to_numpy()
        d["ret_7d"] = s.pct_change(42).to_numpy()
        lr = np.log(s).diff()
        d["vol_7d"] = lr.rolling(42).std().to_numpy()
        d["vol_30d"] = lr.rolling(180).std().to_numpy()
        d["dist_sma50"] = (s / s.rolling(300).mean() - 1).to_numpy()
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def test_relationship_features_do_not_see_the_future():
    """Remove the future and nothing at or before the cut may change."""
    cells = dots.assert_causal(_panel())
    assert cells > 0


def test_the_truncation_check_can_actually_fail(monkeypatch):
    """The check is only worth having if a leaking feature trips it.

    A column built from the NEXT bar's return is planted and the check must name it. Without this,
    a truncation test that could never fail would be indistinguishable from one that works.
    """
    real = dots.add_dots

    def leaky(panel):
        p = real(panel)
        p["ret_4h_l1"] = p.groupby("symbol")["ret_4h"].shift(-1)     # the future, mislabelled
        return p

    monkeypatch.setattr(dots, "add_dots", leaky)
    with pytest.raises(AssertionError, match="ret_4h_l1 is not causal"):
        dots.assert_causal(_panel())


def test_the_pick_edge_is_zero_for_noise_and_positive_for_skill():
    rng = np.random.default_rng(0)
    bars, names = 400, 12
    t = np.repeat(np.arange(bars), names)
    fwd = rng.normal(0, 0.03, bars * names)

    noise = ps.fold_edge(t, rng.normal(size=bars * names), fwd)
    oracle = ps.fold_edge(t, fwd, fwd)                       # scores equal to the outcome
    inverse = ps.fold_edge(t, -fwd, fwd)

    assert abs(noise) < 0.004, f"random picks showed an edge of {noise:+.4f}"
    assert oracle > 0.02, "a perfect ranker showed no edge"
    assert inverse < -0.02, "a perfectly wrong ranker showed no negative edge"


def test_a_volatility_predictor_has_no_pick_edge():
    """The artefact AUC rewards and the edge must not.

    Scores proportional to each coin's volatility predict "moves more than X%" very well and say
    nothing about direction. Their top picks must not beat the average coin.
    """
    rng = np.random.default_rng(1)
    bars, names = 600, 12
    vol = np.tile(np.linspace(0.01, 0.08, names), bars)
    fwd = rng.normal(0, 1, bars * names) * vol               # symmetric: wilder, not better
    t = np.repeat(np.arange(bars), names)

    from sklearn.metrics import roc_auc_score
    big = (fwd > 0.05).astype(float)
    assert roc_auc_score(big, vol) > 0.75, "the planted volatility signal is not there"
    assert abs(ps.fold_edge(t, vol, fwd)) < 0.004, (
        "picking the most volatile coins showed an edge - the yardstick is rewarding volatility")


def test_thin_cross_sections_are_not_scored():
    """Top 3 of 3 is not a selection. Bars with too few names must be left out."""
    t = np.repeat(np.arange(50), 3)
    fwd = np.random.default_rng(2).normal(0, 0.02, 150)
    assert np.isnan(ps.fold_edge(t, fwd, fwd, k=3, min_names=5)), (
        "an oracle picking 3 of 3 coins was given an edge; that is the whole market, not a pick")


def test_nothing_is_kept_between_runs():
    """The same configuration, scored twice with another target fitted in between, is identical."""
    rng = np.random.default_rng(4)
    n_bars, names = 1400, 8
    t = np.repeat(np.arange(n_bars) * ps.BAR_NS, names).astype("int64")
    X = rng.normal(size=(n_bars * names, 6))
    y1 = (X[:, 0] + rng.normal(0, 2, len(X)) > 0).astype(float)
    y2 = (X[:, 1] + rng.normal(0, 2, len(X)) > 0).astype(float)
    fwd = rng.normal(0, 0.02, len(X))
    kw = dict(train_days=100, test_days=30, fwd=fwd, threads=1)

    a = ps.walk(X, y1, t, "auc", "linear", 6, 0, int(t.max()) + 1, **kw)
    ps.walk(X, y2, t, "auc", "linear", 6, 0, int(t.max()) + 1, **kw)
    b = ps.walk(X, y1, t, "auc", "linear", 6, 0, int(t.max()) + 1, **kw)

    assert len(a["scores"]) >= 3, "the walk produced too few folds to mean anything"
    assert a["scores"] == b["scores"] and a["edges"] == b["edges"], (
        "the second run differs from the first: something carried over between runs")
    assert np.mean(a["scores"]) > 0.55, "the planted signal was not found, so the walk is not learning"


def test_training_rows_stop_short_of_the_test_window(monkeypatch):
    """The purge gap, checked on what the model is actually handed.

    A label at bar t is the outcome of a window ending at t+H, so the last H bars before a test
    window carry labels computed from test-period prices. A spy model records the newest training
    row and the oldest test row of every fold; the gap between them must exceed the horizon.
    """
    seen = []

    class Spy:
        def fit(self, X, y):
            self.newest_train = X[:, 0].max()
            return self

        def predict_proba(self, X):
            seen.append((self.newest_train, X[:, 0].min()))
            return np.column_stack([np.full(len(X), 0.5), np.linspace(0, 1, len(X))])

    monkeypatch.setattr(ps, "make_model", lambda kind, task: Spy())
    n_bars, names, h = 1500, 6, 6
    bar = np.repeat(np.arange(n_bars), names)
    t = (bar * ps.BAR_NS).astype("int64")
    X = np.column_stack([bar.astype(float), np.zeros(len(bar))])     # column 0 is the bar number
    y = (np.random.default_rng(6).normal(size=len(bar)) > 0).astype(float)

    # 120 days x 6 bars x 6 names = 4,320 training rows, over walk()'s 3,000-row floor
    ps.walk(X, y, t, "auc", "gbm", h, 0, int(t.max()) + 1, train_days=120, test_days=30, threads=1)

    assert len(seen) >= 3, "the walk produced too few folds to check"
    for newest_train, oldest_test in seen:
        gap = oldest_test - newest_train
        assert gap > h, (
            f"training reached bar {newest_train:.0f} and testing began at {oldest_test:.0f}: a "
            f"gap of {gap:.0f} bars with a {h}-bar horizon, so training labels were computed from "
            "test-period prices")


def test_noise_labels_score_at_chance():
    """Nothing in, nothing out: pure-noise labels must come back at 0.50 out of sample."""
    rng = np.random.default_rng(7)
    n_bars, names = 1500, 6
    t = np.repeat(np.arange(n_bars) * ps.BAR_NS, names).astype("int64")
    X = rng.normal(size=(n_bars * names, 3))
    y = (rng.normal(size=len(X)) > 0).astype(float)
    r = ps.walk(X, y, t, "auc", "linear", 6, 0, int(t.max()) + 1,
                train_days=120, test_days=30, threads=1)
    assert len(r["scores"]) >= 3
    assert abs(np.mean(r["scores"]) - 0.5) < 0.03, (
        "noise labels scored away from 0.50 out of sample: information is leaking into the folds")
