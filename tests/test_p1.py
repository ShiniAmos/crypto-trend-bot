"""P1 entry filters. Each one is a separate hypothesis and must be independently switchable."""

import pandas as pd
import pytest

from app import backtest, signals
from app.flags import Flags
from test_p0 import _panel


def _bar():
    """One bar, four coins, all of them passing the baseline entry rule."""
    return pd.DataFrame([
        # symbol,   ret_30d, btc_ret_30d, rs_7d
        ("AAAUSDT", 0.20, 0.15, 0.05),
        ("BBBUSDT", -0.10, 0.15, 0.05),     # falling on its own: abs_momentum drops it
        ("CCCUSDT", 0.20, -0.05, 0.05),     # BTC falling over 30d: btc_momentum drops it
        ("DDDUSDT", 0.20, 0.15, 0.50),      # already up 50% vs BTC in a week: rs_cap drops it
    ], columns=["symbol", "ret_30d", "btc_ret_30d", "rs_7d"]).assign(
        eligible=True, regime_on=True, breakout=True, score=[90, 89, 88, 87])


class S:
    candidates = 5


def _syms(flags):
    return set(signals.entry_candidates(_bar(), S, flags=flags)["symbol"])


def test_no_flags_keeps_every_candidate():
    assert _syms(None) == {"AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT"}
    assert _syms(Flags()) == {"AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT"}


def test_each_filter_removes_only_its_own_target():
    assert _syms(Flags(abs_momentum=True)) == {"AAAUSDT", "CCCUSDT", "DDDUSDT"}
    assert _syms(Flags(btc_momentum=True)) == {"AAAUSDT", "BBBUSDT", "DDDUSDT"}
    assert _syms(Flags(rs_cap=0.35)) == {"AAAUSDT", "BBBUSDT", "CCCUSDT"}


def test_filters_combine_without_interfering():
    assert _syms(Flags(abs_momentum=True, btc_momentum=True, rs_cap=0.35)) == {"AAAUSDT"}


def test_rs_cap_of_zero_is_off_not_a_zero_threshold():
    """0.0 must mean 'disabled', otherwise it would silently reject every candidate."""
    assert _syms(Flags(rs_cap=0.0)) == _syms(None)


@pytest.mark.parametrize("flags", [Flags(abs_momentum=True), Flags(btc_momentum=True),
                                   Flags(rs_cap=0.35)])
def test_a_filter_can_only_reduce_the_trade_count(settings, flags):
    panel = _panel()
    base = backtest.run(panel, settings)
    filtered = backtest.run(panel, settings, flags=flags)
    assert len(filtered.trades) <= len(base.trades)
    assert filtered.equity.index.equals(base.equity.index)
