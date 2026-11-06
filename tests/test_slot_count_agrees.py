"""Every path must agree how many of the rotation's slots are in use.

THE DIVERGENCE THIS CLOSES, found 6 Nov 2026 in the audit.

The slot count existed in three places with three different answers:

    backtest.run        the engine tag, minus positions being sold this bar      correct
    Bot.on_bar_close    `sym not in core_syms` - symbol membership               wrong
    Bot.live_tick       len(positions) - the core counted against the rotation   wrong

With the core holding BTC and ETH and MAX_NAMES=5, the bar close saw five free slots and the live
tick saw three. Nothing had gone wrong with it yet - which is exactly what was true of
core_dual_ma the week before it cost 118 USDT in fees. One rule, several copies, the live one
holding the wrong answer.

The symbol-membership version has a second bug the engine tag does not: the BREAKOUT engine can
hold BTCUSDT on its own account, and `sym not in core_syms` excludes it from the count, handing
the rotation a free extra slot every time that happens.
"""

import inspect

import pytest

from app import backtest, main as main_mod, signals


def test_a_core_holding_does_not_use_a_rotation_slot():
    pos = {"BTCUSDT": {"engine": "core"}, "ZECUSDT": {"engine": "breakout"}}
    assert signals.breakout_slots_used(pos) == 1


def test_a_breakout_holding_of_a_core_symbol_DOES_use_a_slot():
    """The bug symbol-membership had and the engine tag does not.

    BTCUSDT is in the core's list, but this position was opened by the rotation on a breakout, so
    it occupies one of the rotation's slots. Counting by symbol would have called it free.
    """
    pos = {"BTCUSDT": {"engine": "breakout"}, "ZECUSDT": {"engine": "breakout"}}
    assert signals.breakout_slots_used(pos) == 2, (
        "a breakout-owned BTCUSDT was treated as the core's, giving the rotation a free slot")


def test_a_position_predating_the_engine_column_counts_as_breakout():
    """Rows created before `engine` existed belong to the rotation, as the DB DEFAULT says."""
    assert signals.breakout_slots_used({"ADAUSDT": {}}) == 1
    assert signals.breakout_slots_used({"ADAUSDT": {"engine": None}}) == 1


def test_positions_being_sold_this_bar_can_be_excluded():
    """The backtest decides entries in the pass that queues exits."""
    pos = {"ADAUSDT": {"engine": "breakout"}, "ZECUSDT": {"engine": "breakout"}}
    assert signals.breakout_slots_used(pos, excluding={"ADAUSDT"}) == 1
    assert signals.breakout_slots_used(pos, excluding=["ADAUSDT", "ZECUSDT"]) == 0


def test_the_empty_book_uses_no_slots():
    assert signals.breakout_slots_used({}) == 0


@pytest.mark.parametrize("where", ["on_bar_close", "live_tick"])
def test_neither_live_path_counts_slots_for_itself(where):
    """Structural: the live paths must call the shared function, not re-derive the count.

    No behavioural test catches a fourth copy being added - the paths only disagree once the core
    is on AND a book is near its slot limit - so the shape is asserted directly.
    """
    src = inspect.getsource(getattr(main_mod.Bot, where))
    code = " ".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    assert "signals.breakout_slots_used(" in code, f"{where} does not use the shared slot count"
    assert "not in core_syms]" not in code, (
        f"{where} is counting slots by symbol membership again, which miscounts a "
        "breakout-owned core symbol")


def test_the_backtest_does_not_keep_its_own_copy():
    src = inspect.getsource(backtest)
    assert "signals.breakout_slots_used(" in src
    assert 'not p0.get("engine") == "core" and s2 not in selling' not in src, (
        "backtest.run is counting slots itself again; it must come from signals or the paths "
        "will drift apart exactly as they did before")
