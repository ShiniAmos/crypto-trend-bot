"""The trend core must act on a live price, not wait for the hourly bar.

live_tick runs every LIVE_POLL_SECONDS (10 on the server) and used to handle only the breakout
engine. The core acted solely in on_bar_close, so after the core became the main engine on the demo
book the account could sit up to an hour past a genuine crossing of its 200-day average.

Two things are pinned here, and the second is a bug that existed the moment the core shipped:

  1. the core buys and sells on a live price crossing its own 200-day line, within one poll
  2. a core position is NEVER exited on the Donchian low. `levels["exit"]` carries a low for every
     symbol in the panel, so without an explicit exclusion the live tick would sell the core on a
     breakdown - exactly the reactive exit the core was built to avoid.
"""

import json

import pandas as pd
import pytest

from app import broker, data, reconcile, risk, universe
from app.flags import Flags
from app.main import Bot

CORE = dict(trend_core=0.5, trend_core_symbols="BTCUSDT")


@pytest.fixture
def bot(settings, monkeypatch):
    uni = {"date": "2026-10-13", "symbols": ["BTCUSDT", "AAAUSDT"],
           "coins": [{"symbol": s, "cg_rank": i, "live_eligible": True}
                     for i, s in enumerate(["BTCUSDT", "AAAUSDT"], 1)], "dropped": []}
    settings.universe_path.write_text(json.dumps(uni))
    monkeypatch.setattr(universe, "tradable_status", lambda s: True)
    monkeypatch.setattr(broker.PaperBroker, "filters", lambda self, s: risk.SymbolFilters())
    b = Bot(settings)
    b.flags = Flags(**CORE)
    b.queued_fills = False          # demo/live send immediately; this is that path
    return b


def _levels(sma200=100.0, don_low=95.0):
    return {"bar_time": "2026-10-13 03:00:00+00:00", "regime": True,
            "entry": {}, "stop": {"BTCUSDT": 90.0}, "exit": {"BTCUSDT": don_low},
            "rank": {"BTCUSDT": 1.0},
            "core": {"BTCUSDT": {"dist_sma200": 0.05, "sma200": sma200, "on": True}},
            "core_weight": 0.5, "max_names": 3}


def test_the_core_buys_the_moment_price_crosses_its_line(bot):
    bot.store.put("levels", _levels(sma200=100.0))
    bot.levels = None
    bot.live_tick(shared_prices={"BTCUSDT": 101.0})       # just above the 200-day average
    assert "BTCUSDT" in bot.store.positions(), "the core waited for a bar instead of acting live"
    assert "live core" in bot.store.positions()["BTCUSDT"]["reason"]


def test_the_core_does_not_buy_below_its_line(bot):
    bot.store.put("levels", _levels(sma200=100.0))
    bot.levels = None
    bot.live_tick(shared_prices={"BTCUSDT": 99.0})
    assert "BTCUSDT" not in bot.store.positions()


def test_the_core_sells_the_moment_price_loses_its_line(bot):
    bot.store.put("levels", _levels(sma200=100.0))
    bot.levels = None
    bot.live_tick(shared_prices={"BTCUSDT": 101.0})
    assert "BTCUSDT" in bot.store.positions()
    bot.store.put("acted_this_bar", [])                   # a new bar: the guard resets
    bot.live_tick(shared_prices={"BTCUSDT": 99.0})
    assert "BTCUSDT" not in bot.store.positions(), "the core held on below its 200-day average"


def test_a_core_position_is_not_sold_on_the_donchian_low(bot):
    """The bug. levels['exit'] has a low for every symbol; the core must ignore it."""
    bot.store.put("levels", _levels(sma200=100.0, don_low=102.0))
    bot.levels = None
    bot.live_tick(shared_prices={"BTCUSDT": 103.0})       # above both the line and the low
    assert "BTCUSDT" in bot.store.positions()
    bot.store.put("acted_this_bar", [])
    # now under the Donchian low (102) but still above its 200-day average (100): HOLD
    bot.live_tick(shared_prices={"BTCUSDT": 101.0})
    assert "BTCUSDT" in bot.store.positions(), \
        "the live tick exited the core on a Donchian breakdown"


def test_one_core_decision_per_bar_even_if_price_oscillates(bot):
    """Anti-churn: a price flapping across the line must not trade repeatedly on one bar."""
    bot.store.put("levels", _levels(sma200=100.0))
    bot.levels = None
    bot.live_tick(shared_prices={"BTCUSDT": 101.0})
    assert "BTCUSDT" in bot.store.positions()
    bot.live_tick(shared_prices={"BTCUSDT": 99.0})        # same bar, no reset
    assert "BTCUSDT" in bot.store.positions(), "the core churned within a single bar"


def test_the_kill_switch_closes_the_core_live(bot):
    bot.store.put("levels", _levels(sma200=100.0))
    bot.levels = None
    bot.live_tick(shared_prices={"BTCUSDT": 101.0})
    assert "BTCUSDT" in bot.store.positions()
    bot.s.kill_switch_path.write_text("stop")
    bot.store.put("acted_this_bar", [])
    bot.live_tick(shared_prices={"BTCUSDT": 101.0})       # still above its line
    assert "BTCUSDT" not in bot.store.positions(), "the kill switch did not close the core"


def test_safe_mode_blocks_a_live_core_entry(bot, monkeypatch):
    monkeypatch.setattr(reconcile, "safe_mode", lambda store: True)
    bot.store.put("levels", _levels(sma200=100.0))
    bot.levels = None
    bot.live_tick(shared_prices={"BTCUSDT": 101.0})
    assert "BTCUSDT" not in bot.store.positions()


def test_no_core_means_live_tick_is_unchanged(settings, monkeypatch):
    """Flag off: the core branch must do nothing at all."""
    monkeypatch.setattr(universe, "tradable_status", lambda s: True)
    monkeypatch.setattr(broker.PaperBroker, "filters", lambda self, s: risk.SymbolFilters())
    b = Bot(settings)
    b.flags = Flags()
    b.queued_fills = False
    lv = _levels()
    lv["core"], lv["core_weight"] = {}, 0.0
    b.store.put("levels", lv)
    b.live_tick(shared_prices={"BTCUSDT": 101.0})
    assert b.store.positions() == {}
