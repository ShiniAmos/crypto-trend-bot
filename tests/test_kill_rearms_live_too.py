"""The 8% drawdown kill must re-arm on the live path, exactly as the backtest's does.

THE BUG THIS EXISTS FOR, found 6 Nov 2026 while auditing the bot.

`risk.update_limits` only ever set `killed = True`. Nothing in app/ ever set it back, so the first
8% drawdown ended a live book permanently - no alert, green heartbeat, `can_enter` refusing every
entry for ever.

backtest.run has always re-armed after a 7-day pause, resuming from a fresh peak. That is not a
cosmetic difference. On the standard panel the flag-off baseline fires this kill **28 times**, and
the published pit figures fire it 9 times over the full sample and 4 in the held-out period. Every
one of those is a pause-and-continue. So the published +21.55% and +12.45% describe a book that
survives dozens of kills, while a live book got exactly one: the headline numbers were unreachable
live by construction, whatever edge the strategy has.

It had already cost a book before anyone looked. The demo account's peak was 5,006.59 and it closed
1 Nov at 4,603.98 - about two USDT past the 8% line, which the 118 USDT of churn fees is precisely
what paid for - and it then sat dead for five days.

The rule now lives in risk.update_limits, so both paths get it from one place. Flag-off parity is
unchanged at 7.3e-12 across 19,935 bars with 28 kills in the curve, which is what proves moving it
changed no backtested number.
"""

import pandas as pd
import pytest

from app import backtest, config, risk


def _settings(cooldown=7):
    s = config.load()
    object.__setattr__(s, "kill_cooldown_days", cooldown) if hasattr(s, "__setattr__") else None
    s.kill_cooldown_days = cooldown
    s.kill_drawdown = 0.08
    s.daily_loss_halt = 0.03
    return s


def _killed_state(peak=5006.59):
    """The demo book as it actually stood on 1 Nov 2026."""
    st = risk.RiskState(peak_equity=peak, day="2026-11-01", day_start_equity=peak)
    risk.update_limits(st, peak * 0.92 - 1, "2026-11-01", _settings())
    assert st.killed, "the 8% kill did not fire, so this test is not testing anything"
    return st


def test_the_live_kill_re_arms_after_the_cooldown():
    """THE BUG: a killed live book must come back, as the backtested one does."""
    st = _killed_state()
    s = _settings(7)

    # six days later: still paused
    risk.update_limits(st, st.peak_equity * 0.93, "2026-11-07", s)
    assert st.killed, "the kill released early; the pause is supposed to be 7 days"

    # seven days later: re-armed, from a fresh peak
    events = risk.update_limits(st, 4_600.0, "2026-11-08", s)
    assert not st.killed, (
        "the drawdown kill never re-armed, so this book can never open another position - the "
        "failure that killed the demo account on 1 Nov")
    assert any("re-armed" in e for e in events), "the re-arm happened silently"
    assert st.peak_equity == pytest.approx(4_600.0), (
        "the peak was not reset, so the book re-arms already 8% underwater and dies again at once")


def test_a_re_armed_book_can_actually_enter_again():
    """The state change has to reach the gate that refuses entries."""
    st = _killed_state()
    assert risk.can_enter(st, 0, 3)[0] is False
    risk.update_limits(st, 4_600.0, "2026-11-08", _settings(7))
    assert risk.can_enter(st, 0, 3)[0] is True, "re-armed but still refusing entries"


def test_cooldown_zero_keeps_the_old_terminal_behaviour():
    """KILL_COOLDOWN_DAYS=0 must disable the re-arm, so the change is reversible by config."""
    st = _killed_state()
    risk.update_limits(st, 4_600.0, "2026-12-25", _settings(0))
    assert st.killed, "cooldown 0 is supposed to mean the kill is permanent"


def test_the_backtest_no_longer_keeps_its_own_copy_of_the_rule():
    """One rule, one place. The whole reason the live path missed it was a second copy."""
    import inspect
    src = inspect.getsource(backtest)
    assert "state.killed, state.peak_equity = False, equity" not in src, (
        "backtest.run is re-arming the kill itself again; it must come from risk.update_limits "
        "or the two paths will drift apart exactly as they did before")


def test_run_argument_still_overrides_the_env_cooldown():
    """research scripts vary kill_cooldown_days per run; the argument must still win."""
    s = _settings(7)
    shimmed = backtest._with_cooldown(s, 30)
    assert shimmed.kill_cooldown_days == 30
    assert shimmed.kill_drawdown == s.kill_drawdown, "the shim stopped forwarding other settings"
    assert s.kill_cooldown_days == 7, "the shim mutated the caller's settings"
