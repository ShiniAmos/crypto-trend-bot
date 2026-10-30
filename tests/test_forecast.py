"""app/forecast.py: causal, cached, and completely inert as far as trading is concerned.

Four claims are worth a test here, and they are the four that would do damage if they broke:
  1. the forecast for a bar is identical whether or not later bars exist (no lookahead);
  2. the disk cache is reused for a day and refitted after that;
  3. a missing, truncated or stale cache leaves the website working;
  4. nothing in the trading path can read the forecast, so the audited baseline is untouched.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app import backtest, features, forecast
from app.flags import Flags
from conftest import daily_from_4h, make_bars

APP = Path(__file__).resolve().parent.parent / "app"
SYMS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
NBARS = 2200          # ~1 year of 4h bars: enough for a 200-day SMA plus 500 resolved labels


@pytest.fixture(scope="module")
def panel():
    """A synthetic three-coin panel, long enough that both models can be fitted."""
    frames = []
    for i, sym in enumerate(SYMS):
        b4 = make_bars(NBARS, drift=0.0004, seed=i)
        f = features.symbol_features(b4, daily_from_4h(b4), 20)
        f["symbol"] = sym
        frames.append(f)
    return features.add_cross_section(pd.concat(frames, ignore_index=True), min_age_bars=200)


@pytest.fixture
def s(settings):
    return settings


# ------------------------------------------------------------------ 1. causality

def test_forecast_for_a_bar_ignores_every_later_bar(s, panel):
    """The whole point. Forecast bar T from a panel that stops at T, then from the full panel with
    as_of=T, then from a panel whose bars after T have been replaced with nonsense. All equal."""
    t = panel["bar_time"].drop_duplicates().sort_values().iloc[-60]
    truncated = forecast.compute(s, panel[panel["bar_time"] <= t])

    wrecked = panel.copy()
    future = wrecked["bar_time"] > t
    for col in ("open", "high", "low", "close"):
        wrecked.loc[future, col] *= 7.0          # a violent, impossible future

    for variant in (forecast.compute(s, panel, as_of=t),
                    forecast.compute(s, wrecked, as_of=t)):
        assert variant["as_of"] == truncated["as_of"] == str(t)
        assert variant["vol_fit"] == truncated["vol_fit"]
        assert variant["regime_fit"] == truncated["regime_fit"]
        assert variant["p_regime_on_7d"] == truncated["p_regime_on_7d"]
        assert variant["per_coin"] == truncated["per_coin"]


def test_training_labels_stop_at_least_a_horizon_before_the_forecast_bar(s, panel):
    """The purge. The newest label either model trained on must have resolved before the bar being
    forecast, or the fit has seen the answer."""
    out = forecast.compute(s, panel)
    as_of = pd.Timestamp(out["as_of"])
    for key in ("vol_train_end", "regime_train_end"):
        gap = as_of - pd.Timestamp(out[key])
        assert gap >= pd.Timedelta(days=forecast.HORIZON / forecast.BARS_PER_DAY), \
            f"{key} is only {gap} before the forecast bar"


def test_the_label_is_the_next_42_bars_and_nothing_earlier(panel):
    """Recompute fwd_vol from the raw closes: it is the std of the 42 returns realised AFTER t."""
    v = forecast.vol_frame(panel)
    g = panel[panel["symbol"] == "ETHUSDT"].sort_values("bar_time").reset_index(drop=True)
    sub = v[(v["symbol"] == "ETHUSDT") & v["fwd_vol"].notna()]
    row = sub.iloc[len(sub) // 3]
    i = int(g.index[g["bar_time"] == row["bar_time"]][0])
    c = g["close"].to_numpy()[i:i + forecast.HORIZON + 1]
    want = float(np.std(np.diff(np.log(c)), ddof=1) * forecast.ANNUAL)
    assert want == pytest.approx(float(row["fwd_vol"]), rel=1e-9)


def test_the_features_at_a_bar_do_not_move_when_later_bars_change(panel):
    """vol_1d/7d/30d and the regime feature at bar T come from bars <= T only."""
    t = panel["bar_time"].drop_duplicates().sort_values().iloc[-60]
    wrecked = panel.copy()
    wrecked.loc[wrecked["bar_time"] > t, "close"] *= 0.1

    a = forecast.vol_frame(panel)
    b = forecast.vol_frame(wrecked)
    cols = forecast.HAR_FEATS
    pd.testing.assert_frame_equal(
        a.loc[a["bar_time"] == t, ["symbol"] + cols].reset_index(drop=True),
        b.loc[b["bar_time"] == t, ["symbol"] + cols].reset_index(drop=True))
    assert forecast.btc_frame(panel).loc[t, "x"] == forecast.btc_frame(wrecked).loc[t, "x"]


# ------------------------------------------------------------------ 2. the cache

def test_cache_is_written_then_reused_then_refreshed(s, panel):
    first = forecast.outlook(s, panel)
    assert forecast.cache_path(s).exists()
    assert first["age_hours"] == 0.0 and first["stale"] is False and first["from_cache"] is False

    # A second call the same hour must not refit, and must work with no panel at all - which is
    # the whole reason the cache exists: a page load costs one small JSON read.
    again = forecast.outlook(s, panel=None)
    assert again["from_cache"] is True
    assert again["as_of"] == first["as_of"] and again["generated"] == first["generated"]

    # Still inside the day, from a tz-aware clock: a hit. (Without this the next assertion passes
    # for the wrong reason, because a stamp that cannot be compared reads as infinitely old.)
    gen = pd.Timestamp(first["generated"])
    assert forecast.outlook(s, panel, now=gen + pd.Timedelta(hours=5))["from_cache"] is True
    assert forecast.outlook(s, panel, now=gen.tz_convert("Asia/Kolkata"))["from_cache"] is True

    # A day later it is stale, so it refits.
    assert forecast.outlook(s, panel, now=gen + pd.Timedelta(hours=25))["from_cache"] is False

    # Still inside the day, but force ignores the age entirely.
    assert forecast.outlook(s, panel, force=True)["from_cache"] is False
    # ...and a plain call right after that is a cache hit again.
    assert forecast.outlook(s, panel)["from_cache"] is True


def test_a_cache_read_back_from_disk_matches_what_was_computed(s, panel):
    out = forecast.outlook(s, panel, force=True)
    on_disk = json.loads(forecast.cache_path(s).read_text())
    assert on_disk["as_of"] == out["as_of"]
    assert on_disk["p_regime_on_7d"] == pytest.approx(out["p_regime_on_7d"])
    assert on_disk["per_coin"].keys() == out["per_coin"].keys()


def test_cache_age_is_measured_from_the_stamp_not_the_file(s, panel):
    forecast.outlook(s, panel, force=True)
    d = json.loads(forecast.cache_path(s).read_text())
    d["generated"] = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=3)).isoformat()
    forecast.cache_path(s).write_text(json.dumps(d))
    got = forecast.read_cache(s)
    assert got["stale"] is True and got["age_hours"] > 70


# ------------------------------------------------------------------ 3. degrading gracefully

def test_no_cache_and_no_usable_data_returns_none_rather_than_raising(s):
    assert forecast.read_cache(s) is None
    assert forecast.outlook(s, pd.DataFrame(columns=["bar_time", "symbol", "close"])) is None


@pytest.mark.parametrize("junk", ["", "not json at all", "[1, 2, 3]", '{"as_of": "x"}',
                                  '{"regime_on": true}'])
def test_a_corrupt_cache_reads_as_no_cache(s, junk):
    forecast.cache_path(s).parent.mkdir(parents=True, exist_ok=True)
    forecast.cache_path(s).write_text(junk)
    assert forecast.read_cache(s) is None


def test_a_bad_timestamp_counts_as_infinitely_old(s, panel):
    forecast.outlook(s, panel, force=True)
    d = json.loads(forecast.cache_path(s).read_text())
    d["generated"] = "whenever"
    forecast.cache_path(s).write_text(json.dumps(d))
    got = forecast.read_cache(s)
    assert got is not None and got["stale"] is True and got["age_hours"] == float("inf")


def test_a_stale_cache_is_served_when_a_refit_fails(s, panel):
    """The website case that matters: yesterday's forecast, clearly labelled, beats a blank panel."""
    forecast.outlook(s, panel, force=True)
    d = json.loads(forecast.cache_path(s).read_text())
    d["generated"] = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=5)).isoformat()
    forecast.cache_path(s).write_text(json.dumps(d))

    got = forecast.outlook(s, pd.DataFrame(columns=["bar_time", "symbol", "close"]))
    assert got is not None, "a failed refit must not wipe the outlook"
    assert got["stale"] is True and got["as_of"] == d["as_of"]


def test_too_little_history_raises_in_compute_and_is_swallowed_by_outlook(s, panel):
    short = panel[panel["bar_time"] <= panel["bar_time"].drop_duplicates().iloc[300]]
    with pytest.raises(Exception):
        forecast.compute(s, short)
    assert forecast.outlook(s, short) is None       # no cache to fall back on either


# ------------------------------------------------------------------ 4. it changes no trade

def test_no_flag_was_added_and_the_baseline_is_still_the_baseline():
    """The integration study found no trading use that passed the ship rule, so there is nothing
    to put behind a flag. If a flag ever appears for this, the study has to be redone first."""
    assert Flags().is_baseline
    assert not any("forecast" in f for f in Flags().__dataclass_fields__)


def test_nothing_in_the_trading_path_imports_the_forecaster():
    """Structural, not behavioural: the forecast cannot change a decision if no decision-making
    module can see it. Cheaper to enforce than to re-audit the equity curve."""
    for name in ("backtest", "signals", "risk", "portfolio", "broker", "main", "features",
                 "regime", "universe", "flags", "experiment", "supervisor"):
        src = (APP / f"{name}.py").read_text()
        assert "forecast" not in src, f"app/{name}.py mentions the forecaster"


def test_running_the_forecast_leaves_the_backtest_bit_identical(s, panel):
    before = backtest.run(panel, s, flags=Flags()).equity
    forecast.outlook(s, panel, force=True)
    after = backtest.run(panel, s, flags=Flags()).equity
    pd.testing.assert_series_equal(before, after)
    assert forecast.compute(s, panel)["affects_trading"] is False
