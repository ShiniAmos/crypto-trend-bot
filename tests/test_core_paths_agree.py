"""Every path that decides whether the core is on must agree, for every flag combination.

THE BUG THIS EXISTS FOR, 30 Oct - 1 Nov 2026.

`core_dual_ma` added the 50-day average to the core's rule. It was wired into two of the three
places that decide:

    signals.core_on()      backtest           both lines   ->  "out"
    Bot.core_tick()        live price tick    both lines   ->  "out"
    Bot.core_rebalance()   live bar close     200-day ONLY ->  "in"

So every hour the bar-close path bought and the live tick sold, at the same price, for three days:
**203 fills and 118.32 USDT of fees** on the demo book before it was noticed. The market happened
to end it by falling below the 200-day line, which made both paths agree again.

Ten tests shipped with that feature, including ones explicitly checking backtest/live agreement.
They all tested the two paths that agreed. That is the hole these tests close.

The approach is deliberately not "test core_rebalance with dual_ma" - that fixes this bug and not
the next one. Instead: enumerate flag combinations, and for each one assert that every path
returns the SAME answer. A new core flag wired into only some paths fails here without anyone
having to think of it.
"""

import itertools
import json

import numpy as np
import pandas as pd
import pytest

from app import backtest, broker, risk, signals, universe
from app.flags import Flags
from app.main import Bot

# every combination of the flags that affect whether the core is on
FLAG_GRID = [
    dict(trend_core=0.5, trend_core_symbols="BTCUSDT"),
    dict(trend_core=0.5, trend_core_symbols="BTCUSDT", core_dual_ma=True),
    dict(trend_core=0.5, trend_core_symbols="BTCUSDT", core_max_stretch=0.15),
    dict(trend_core=0.5, trend_core_symbols="BTCUSDT", core_dual_ma=True,
         core_max_stretch=0.15),
]

# price states: (distance above the 200-day, distance above the 50-day)
STATES = [
    (0.10, 0.05),     # above both
    (0.10, -0.03),    # above the slow line, BELOW the fast one  <- the churn case
    (-0.02, 0.05),    # below the slow line
    (-0.02, -0.03),   # below both
    (0.40, 0.20),     # far above both (the stretch gate's case)
]


def _bar(d200, d50):
    return pd.DataFrame({"symbol": ["BTCUSDT"], "dist_sma200": [d200], "dist_sma50": [d50],
                         "close": [100.0], "vol_30d": [0.5]})


@pytest.fixture
def bot(settings, monkeypatch):
    uni = {"date": "2026-11-05", "symbols": ["BTCUSDT"],
           "coins": [{"symbol": "BTCUSDT", "cg_rank": 1, "live_eligible": True}], "dropped": []}
    settings.universe_path.write_text(json.dumps(uni))
    monkeypatch.setattr(universe, "tradable_status", lambda s: True)
    monkeypatch.setattr(broker.PaperBroker, "filters", lambda self, s: risk.SymbolFilters())
    return Bot(settings)


@pytest.mark.parametrize("flags_kw", FLAG_GRID, ids=lambda d: "+".join(sorted(d)) or "base")
@pytest.mark.parametrize("d200,d50", STATES, ids=lambda v: f"{v:+.2f}")
def test_the_bar_close_path_agrees_with_the_shared_predicate(bot, flags_kw, d200, d50):
    """core_rebalance must BEHAVE as signals.core_on() says, in every state.

    This drives the real method and looks at whether a position exists afterwards. The first
    version of this test called core_on() on both sides of the assertion and compared it with
    itself - it passed against the buggy code, which is the only way to find out that a
    regression test is tautological. Behaviour is the thing to assert; re-deriving the logic in
    the test just tests the test.
    """
    bot.flags = Flags(**flags_kw)
    bot.queued_fills = False
    bar = _bar(d200, d50)
    expected_in = bool(signals.core_on(bar, bot.flags).iloc[0])

    latest = bar.set_index("symbol")
    latest["open"] = latest["high"] = latest["low"] = 100.0
    state = risk.RiskState(peak_equity=10_000.0, day="2026-11-05", day_start_equity=10_000.0)
    bot.core_rebalance(latest, pd.Timestamp("2026-11-05 12:00", tz="UTC"), state, 10_000.0)
    actually_in = "BTCUSDT" in bot.store.positions()

    # the stretch gate blocks new entries, so a far-extended coin is correctly left alone
    if flags_kw.get("core_max_stretch") and d200 > flags_kw["core_max_stretch"]:
        assert not actually_in, "the stretch gate let an over-extended entry through"
        return

    assert actually_in == expected_in, (
        f"core_rebalance {'BOUGHT' if actually_in else 'did not buy'} while the shared predicate "
        f"says {'IN' if expected_in else 'OUT'} - flags={flags_kw} d200={d200} d50={d50}")


@pytest.mark.parametrize("flags_kw", FLAG_GRID, ids=lambda d: "+".join(sorted(d)) or "base")
@pytest.mark.parametrize("d200,d50", STATES, ids=lambda v: f"{v:+.2f}")
def test_the_live_tick_agrees_with_the_shared_predicate(bot, flags_kw, d200, d50):
    """core_tick works from frozen PRICE levels rather than distances; same verdict required."""
    bot.flags = Flags(**flags_kw)
    bar = _bar(d200, d50)
    expected = bool(signals.core_on(bar, bot.flags).iloc[0])

    latest = bar.set_index("symbol")
    lv = bot._core_levels(latest, "BTCUSDT")
    px = 100.0
    above_200 = px > lv["sma200"]
    above_50 = (not lv.get("dual_ma")) or (lv.get("sma50") and px > lv["sma50"])
    actual = bool(above_200 and above_50)

    # the stretch gate only blocks NEW entries, so it is excluded from this comparison
    if flags_kw.get("core_max_stretch") and d200 > flags_kw["core_max_stretch"]:
        pytest.skip("core_max_stretch gates entries only, not the held/not-held state")

    assert actual == expected, (
        f"the live tick says {'IN' if actual else 'OUT'} while the shared predicate says "
        f"{'IN' if expected else 'OUT'} for flags={flags_kw} d200={d200} d50={d50}")


def test_the_exact_state_that_caused_the_churn(bot):
    """Above the 200-day, below the 50-day, dual_ma on. Every path must say OUT."""
    bot.flags = Flags(trend_core=0.5, trend_core_symbols="BTCUSDT", core_dual_ma=True)
    bot.queued_fills = False
    bar = _bar(0.10, -0.03)
    assert not bool(signals.core_on(bar, bot.flags).iloc[0]), "the shared predicate says IN"

    latest = bar.set_index("symbol")
    latest["open"] = latest["high"] = latest["low"] = 100.0
    state = risk.RiskState(peak_equity=10_000.0, day="2026-11-05", day_start_equity=10_000.0)
    bot.core_rebalance(latest, pd.Timestamp("2026-11-05 12:00", tz="UTC"), state, 10_000.0)
    assert "BTCUSDT" not in bot.store.positions(),         "core_rebalance BOUGHT below the 50-day line - this is the 203-fill churn"

    lv = bot._core_levels(latest, "BTCUSDT")
    assert lv["on"] is False, "the snapshot says IN"
    assert lv["dual_ma"] is True and lv.get("sma50"), "the snapshot lost the 50-day level"


def test_the_bar_close_path_does_not_trade_a_symbol_twice_in_one_bar(bot, monkeypatch):
    """The backstop. Even if the paths disagreed again, one bar can produce one decision."""
    bot.flags = Flags(trend_core=0.5, trend_core_symbols="BTCUSDT")
    bot.queued_fills = False
    latest = _bar(0.10, 0.05).set_index("symbol")
    latest["open"] = latest["high"] = latest["low"] = 100.0
    t = pd.Timestamp("2026-11-05 12:00", tz="UTC")
    state = risk.RiskState(peak_equity=10_000.0, day="2026-11-05", day_start_equity=10_000.0)

    bot.core_rebalance(latest, t, state, 10_000.0)
    first = len(bot.store.fills())
    bot.core_rebalance(latest, t, state, 10_000.0)      # same bar, run again
    assert len(bot.store.fills()) == first, "the core traded twice on one bar"


def test_the_guard_releases_on_the_next_bar(bot):
    """It must expire, or the core would be blocked forever after its first action.

    The expiry is Bot.expire_bar_guard, called by on_bar_close before any decision - it is NOT in
    core_rebalance, which only reads the guard. It was in core_rebalance when this guard shipped
    on 5 Nov, and that is precisely why it was inert: on_bar_close wiped the key again at the end
    of every bar in live-cycle mode, so a bar-close entry never survived for the live tick to see.
    See tests/test_churn_guard_survives_the_bar.py.

    This test drives the two calls the way one bar does: expire, then decide.
    """
    bot.flags = Flags(trend_core=0.5, trend_core_symbols="BTCUSDT")
    bot.queued_fills = False
    latest = _bar(0.10, 0.05).set_index("symbol")
    latest["open"] = latest["high"] = latest["low"] = 100.0
    state = risk.RiskState(peak_equity=10_000.0, day="2026-11-05", day_start_equity=10_000.0)

    t1 = pd.Timestamp("2026-11-05 12:00", tz="UTC")
    bot.expire_bar_guard(t1)
    bot.core_rebalance(latest, t1, state, 10_000.0)
    assert "BTCUSDT" in bot.store.positions()
    assert "BTCUSDT" in set(bot.store.get("acted_this_bar") or []), (
        "the entry was not recorded, so the live tick could trade against it on this same bar")

    # next bar, now below its line: the expiry must let the exit through
    out = _bar(-0.05, -0.05).set_index("symbol")
    out["open"] = out["high"] = out["low"] = 100.0
    t2 = pd.Timestamp("2026-11-05 13:00", tz="UTC")
    bot.expire_bar_guard(t2)
    bot.core_rebalance(out, t2, state, 10_000.0)
    assert "BTCUSDT" not in bot.store.positions(), "the guard blocked the next bar's exit"
