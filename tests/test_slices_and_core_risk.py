"""Steps 3 and 4 do what they say, regardless of whether they earn their place.

All of these flags were REJECTED by the ship rule and ship OFF. They are tested anyway: a flag that
is off but wrong is a trap for the next person who switches it on, and "it was rejected" is a
statement about its returns, not about whether the code is correct.
"""

import numpy as np
import pandas as pd
import pytest

from app import backtest, signals
from app.flags import Flags


def _panel(n=300, vol=0.5, stretch=0.10):
    t = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    rows = []
    for i, ts in enumerate(t):
        for sym, px in (("BTCUSDT", 100.0 + 0.02 * i), ("AAAUSDT", 50.0 + 0.3 * i)):
            rows.append({"bar_time": ts, "symbol": sym, "open": px, "high": px * 1.01,
                         "low": px * 0.99, "close": px})
    p = pd.DataFrame(rows)
    p["dist_sma200"] = np.where(p["symbol"] == "BTCUSDT", stretch, 0.05)
    p["dist_sma50"] = 0.05
    p["btc_dist_sma200"] = 0.10
    p["regime_on"] = True
    p["eligible"] = True
    p["breakout"] = (p["symbol"] == "AAAUSDT") & (p.groupby("symbol").cumcount() > 2)
    p["breakdown"] = False
    p["stop_level"] = p["close"] * 0.9
    p["don_high"] = p["close"] * 1.01
    p["don_low"] = p["close"] * 0.9
    p["don_pos"] = 0.5
    for c, v in (("rs_7d", 0.1), ("rs_30d", 0.1), ("ret_7d", 0.1), ("ret_30d", 0.2),
                 ("vol_7d", 0.5), ("atr_pct", 0.02), ("volume_ratio", 1.5),
                 ("taker_ratio", 0.6), ("quote_volume", 1e7), ("n_eligible", 2)):
        p[c] = v
    p["vol_30d"] = vol
    p["ret_4h"] = p.groupby("symbol")["close"].pct_change().fillna(0.0)
    p["score"] = np.where(p["symbol"] == "AAAUSDT", 0.9, 0.1)
    p["rank"] = np.where(p["symbol"] == "AAAUSDT", 1.0, 2.0)
    return p


CORE = dict(trend_core=0.5, trend_core_symbols="BTCUSDT")


# ---------------------------------------------------------------- step 4: the weight formula

def test_the_vol_cap_halves_the_weight_when_vol_doubles():
    """w = target / vol, so doubling the volatility halves the size - shown with w_max out of the
    way, because at the shipped w_max=0.25 both of these clip to 0.25 and the relationship is
    invisible. That clipping is itself the finding behind 4a being nearly inert on recent data:
    with trend_core=0.25 and w_max=0.25, the cap only bites once vol exceeds 0.72."""
    f = Flags(trend_core=0.5, core_target_vol=0.18, core_w_max=1.0)
    assert signals.core_weight(f, 0.18) == pytest.approx(1.00)
    assert signals.core_weight(f, 0.36) == pytest.approx(0.50)
    assert signals.core_weight(f, 0.72) == pytest.approx(0.25)


def test_w_max_clips_the_cap():
    f = Flags(trend_core=0.5, core_target_vol=0.18, core_w_max=0.25)
    assert signals.core_weight(f, 0.18) == pytest.approx(0.25)
    assert signals.core_weight(f, 0.36) == pytest.approx(0.25)
    assert signals.core_weight(f, 0.90) == pytest.approx(0.20)     # 0.18/0.90, under w_max


def test_the_vol_cap_only_reduces_never_levers():
    f = Flags(trend_core=0.25, core_target_vol=0.18, core_w_max=0.25)
    for v in (0.05, 0.18, 0.40, 1.20, 3.0):
        assert signals.core_weight(f, v) <= 0.25 + 1e-12, f"vol {v} produced leverage"


def test_without_the_cap_the_weight_is_the_flag():
    f = Flags(trend_core=0.25)
    assert signals.core_weight(f, 0.9) == pytest.approx(0.25)
    assert signals.core_weight(f, None) == pytest.approx(0.25)


def test_a_missing_or_silly_volatility_falls_back_to_the_flag():
    f = Flags(trend_core=0.25, core_target_vol=0.18)
    for bad in (None, 0.0, -1.0, float("nan"), float("inf")):
        assert signals.core_weight(f, bad) == pytest.approx(0.25)


# ---------------------------------------------------------------- step 4: the stretch gate

def test_the_stretch_gate_blocks_an_extended_entry():
    f = Flags(trend_core=0.25, core_max_stretch=0.15)
    assert signals.core_entry_allowed(f, 0.10) is True
    assert signals.core_entry_allowed(f, 0.20) is False


def test_the_stretch_gate_is_off_by_default():
    assert signals.core_entry_allowed(Flags(trend_core=0.25), 5.0) is True


def test_the_stretch_gate_does_not_close_a_held_position(settings):
    """It gates ENTRIES only. Closing a winner for running too far is profit-taking, which this
    repository has rejected every time it has been measured."""
    p = _panel(stretch=0.05)                       # enters while not extended
    half = p["bar_time"].quantile(0.5)
    p.loc[(p.symbol == "BTCUSDT") & (p.bar_time > half), "dist_sma200"] = 0.40   # now extended
    r = backtest.run(p, settings, flags=Flags(**CORE, core_max_stretch=0.15))
    sells = r.log[(r.log.side == "SELL") & (r.log.symbol == "BTCUSDT")]
    assert sells.empty, f"the stretch gate closed a held core position: {list(sells.reason)}"


def test_the_stretch_gate_blocks_the_entry_in_the_engine(settings):
    p = _panel(stretch=0.40)                       # extended from the start
    r = backtest.run(p, settings, flags=Flags(**CORE, core_max_stretch=0.15))
    buys = r.log[(r.log.side == "BUY") & (r.log.symbol == "BTCUSDT")]
    assert buys.empty, "an entry 40% above the line passed a 15% stretch cap"


# ---------------------------------------------------------------- step 3: capital slices

def test_slices_cap_the_breakout_names(settings):
    """With slices on, the breakout engine may hold at most slice_max_names."""
    p = _panel()
    r = backtest.run(p, settings, flags=Flags(**CORE, capital_slices=True, slice_max_names=1))
    log = r.log[r.log.side.isin(["BUY", "SELL"])].copy()
    held, peak = set(), 0
    for _, row in log.iterrows():
        if row["symbol"] == "BTCUSDT":
            continue                                # the core is not a breakout slot
        held.add(row["symbol"]) if row["side"] == "BUY" else held.discard(row["symbol"])
        peak = max(peak, len(held))
    assert peak <= 1, f"the breakout engine held {peak} names with slice_max_names=1"


def test_slices_size_the_breakout_inside_its_own_share(settings):
    """A breakout position must be smaller with slices on, because it is sized against the
    remainder after the core's allocation rather than against the whole book."""
    p = _panel()
    a = backtest.run(p, settings, flags=Flags(**CORE))
    b = backtest.run(p, settings, flags=Flags(**CORE, capital_slices=True, slice_max_names=2))
    qa = a.log[(a.log.side == "BUY") & (a.log.symbol == "AAAUSDT")]["qty"].iloc[0]
    qb = b.log[(b.log.side == "BUY") & (b.log.symbol == "AAAUSDT")]["qty"].iloc[0]
    assert qb < qa, f"slices did not shrink the breakout size ({qb} vs {qa})"


def test_slices_off_changes_nothing(settings):
    p = _panel()
    a = backtest.run(p, settings, flags=Flags(**CORE))
    b = backtest.run(p, settings, flags=Flags(**CORE, capital_slices=False, slice_max_names=2))
    pd.testing.assert_series_equal(a.equity, b.equity)
