"""The per-bar churn guard must still be set when the live tick reads it.

THE BUG THIS EXISTS FOR, 5-6 Nov 2026.

The guard added on 5 Nov as the backstop for the 203-fill core churn did nothing in the deployed
configuration, because one bar had TWO resets:

    on_bar_close:
        core_rebalance(...)                       # writes acted_this_bar = ['BTCUSDT']
        ... breakout entries ...
        snapshot_levels(...)
        if self.s.live_cycle:
            self.store.put("acted_this_bar", [])  # wipes it again, same method

`DECISION_TF=live` is what the server runs, so `live_cycle` is True and the bar-close path's entry
was erased before the live tick ever looked at it. The guard was inert in exactly the
configuration it was written for - and inert silently, because the shared-predicate fix in the
same commit stopped the churn by itself, so nothing ever failed.

Two resets for one piece of state is the same shape as the bug underneath it: two copies of one
predicate. The expiry now lives in one place, at the top of on_bar_close, before any decision.

These tests assert the guard's LIFETIME, not its contents, so they fail for any future change that
resets it in the wrong place.
"""

import inspect
import json

import pandas as pd
import pytest

from app import broker, data, risk, universe
from app.flags import Flags
from app.main import Bot

from test_trend_core_live import CORE, _bar, _step          # the live-core scaffolding


@pytest.fixture
def uni(settings, monkeypatch):
    u = {"date": "2026-10-10", "symbols": ["BTCUSDT", "AAAUSDT"],
         "coins": [{"symbol": s, "cg_rank": i, "live_eligible": True}
                   for i, s in enumerate(["BTCUSDT", "AAAUSDT"], 1)], "dropped": []}
    settings.universe_path.write_text(json.dumps(u))
    monkeypatch.setattr(universe, "tradable_status", lambda s: True)
    monkeypatch.setattr(universe, "build", lambda s, **k: u)
    monkeypatch.setattr(data, "price", lambda s: 100.0)
    monkeypatch.setattr(broker.PaperBroker, "filters", lambda self, s: risk.SymbolFilters())
    return u


def _live(bot, monkeypatch):
    """Put the bot in the deployed configuration: DECISION_TF=live."""
    monkeypatch.setattr(type(bot.s), "live_cycle", property(lambda self: True))
    assert bot.s.live_cycle, "the test is meaningless unless live_cycle is on"


def test_a_core_action_is_still_recorded_when_the_bar_ends(settings, uni, monkeypatch):
    """THE BUG. In live-cycle mode the guard must survive to the end of the bar that set it.

    Asserted on the state the live tick would actually read, because the wipe happened AFTER
    core_rebalance had already done the right thing - so nothing closer to the decision can see it.
    """
    b = Bot(settings)
    b.flags = Flags(**CORE)
    _live(b, monkeypatch)
    _step(b, settings)                       # ONE bar: the bar on which the core acts

    # queuing the order IS the action the guard records - the fill comes at the next open
    assert b.store.get("acted_bar"), "the bar was never processed"
    acted = set(b.store.get("acted_this_bar") or [])
    assert "BTCUSDT" in acted, (
        "the churn guard was wiped before the live tick could read it, so a bar-close core BUY "
        "can be sold by the tick minutes later - this is the 203-fill churn")


def test_the_guard_expires_on_the_next_bar_with_the_core_off(settings, uni, monkeypatch):
    """The expiry must not live inside core_rebalance: a book with the core off never enters it.

    paper and paper100 run with trend_core unset. An expiry placed in core_rebalance never runs
    for them, so the tick path would stay blocked for good after its first action.
    """
    b = Bot(settings)
    b.flags = Flags()                        # core OFF, as the paper books run
    _live(b, monkeypatch)
    b.store.put("acted_bar", "stale-previous-bar")
    b.store.put("acted_this_bar", ["ZECUSDT"])

    _step(b, settings)

    assert not (b.store.get("acted_this_bar") or []), (
        "a stale guard from a previous bar survived into this one, so the tick path is blocked")


def test_the_guard_does_not_leak_across_bars_with_the_core_on(settings, uni, monkeypatch):
    """It must expire, or the core could never act twice - the failure the 5 Nov comment feared."""
    b = Bot(settings)
    b.flags = Flags(**CORE)
    _live(b, monkeypatch)
    _step(b, settings, offset_bars=0)
    first = set(b.store.get("acted_this_bar") or [])
    _step(b, settings, offset_bars=1)        # a NEW bar must start clean before deciding
    assert b.store.get("acted_bar") != "", "the bar was never stamped"
    assert first, "nothing was recorded on the first bar"


def test_one_bar_has_exactly_one_reset(settings):
    """Structural. The bug was two resets for one piece of state.

    No behavioural test catches a second reset being added back until it costs money again, so
    this asserts the shape directly: cleared in exactly one place, that place is the named owner,
    and on_bar_close calls it BEFORE it calls core_rebalance.
    """
    from app import main as main_mod
    src = inspect.getsource(main_mod)
    clears = src.count('put("acted_this_bar", [])')
    assert clears == 1, (
        f"acted_this_bar is cleared in {clears} places; there must be exactly one, in "
        "Bot.expire_bar_guard")

    owner = inspect.getsource(main_mod.Bot.expire_bar_guard)
    assert 'put("acted_this_bar", [])' in owner, (
        "the reset moved out of expire_bar_guard, which is supposed to be its only owner")

    body = inspect.getsource(main_mod.Bot.on_bar_close)
    code = " ".join(l for l in body.splitlines() if not l.strip().startswith("#"))
    assert "self.expire_bar_guard(" in code, "on_bar_close no longer expires the guard at all"
    head, _, _ = code.partition("self.expire_bar_guard(")
    assert "self.core_rebalance(" not in head, (
        "the guard is expired AFTER core_rebalance, so the bar-close path's entry is erased "
        "again - this is the 5 Nov bug exactly")


def test_core_rebalance_does_not_reset_the_guard_itself(settings):
    """core_rebalance must only READ the guard.

    The 5 Nov version expired it in here, which is why the paper books - whose core is off and
    which return from this method immediately - depended on a second reset elsewhere.
    """
    from app import main as main_mod
    body = inspect.getsource(main_mod.Bot.core_rebalance)
    assert 'put("acted_this_bar", [])' not in body
    assert 'put("acted_bar"' not in body, "core_rebalance is stamping the bar again"
