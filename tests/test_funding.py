"""The crowding filter: the first signal here that comes from outside the price.

What these pin down: the join cannot see the future, a coin without a perpetual contract is never
filtered, the flag is inert when off, and the filter only ever removes candidates.
"""

import numpy as np
import pandas as pd
import pytest

from app import backtest, funding, signals
from app.flags import Flags
from test_p0 import _panel


def _write_funding(settings, symbol, start_ms, n, rate=0.0001):
    """n funding payments, eight hours apart, starting at start_ms."""
    df = pd.DataFrame({"funding_time": [start_ms + i * 8 * 3_600_000 for i in range(n)],
                       "funding_rate": [rate] * n})
    df.to_parquet(funding.path_for(settings, symbol), index=False)
    return df


def _bar_panel(times, symbols, eligible=True):
    rows = []
    for t in times:
        for s in symbols:
            rows.append({"bar_time": t, "symbol": s, "eligible": eligible})
    return pd.DataFrame(rows)


def test_attach_never_reads_a_future_payment(settings):
    """A payment stamped 08:00 may inform the 08:00 bar, but nothing earlier."""
    t0 = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    start_ms = int(t0.timestamp() * 1000)
    _write_funding(settings, "AAAUSDT", start_ms, 30)
    # a bar before any payment, one at the first payment, and one past the 21 payments the
    # 7-day average needs (7 days x 3 per day) so the trailing mean is actually defined
    times = [t0 - pd.Timedelta(hours=8), t0, t0 + pd.Timedelta(hours=8 * 25)]
    p = funding.attach(_bar_panel(times, ["AAAUSDT"]), settings)
    before = p[p["bar_time"] == times[0]].iloc[0]
    assert pd.isna(before["funding_7d"]), "a bar before any payment must know nothing"
    at_first = p[p["bar_time"] == times[1]].iloc[0]
    assert pd.isna(at_first["funding_7d"]), "one payment is not a seven-day average"
    after = p[p["bar_time"] == times[2]].iloc[0]
    assert not pd.isna(after["funding_7d"]), "a later bar sees the trailing average"


def test_coin_without_a_perpetual_contract_is_never_filtered(settings):
    t0 = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    _write_funding(settings, "AAAUSDT", int(t0.timestamp() * 1000), 60, rate=0.001)
    times = [t0 + pd.Timedelta(hours=8 * i) for i in range(30, 40)]
    p = funding.attach(_bar_panel(times, ["AAAUSDT", "NOPERPUSDT"]), settings)
    noperp = p[p["symbol"] == "NOPERPUSDT"]
    assert noperp["funding_7d"].isna().all() and noperp["funding_pct"].isna().all()

    bar = p[p["bar_time"] == times[-1]].copy()
    bar = bar.assign(regime_on=True, breakout=True, rs_7d=0.1, score=[90.0, 89.0])

    class S:
        candidates = 5
    kept = set(signals.entry_candidates(bar, S, flags=Flags(funding_max_pct=0.8))["symbol"])
    assert "NOPERPUSDT" in kept, "no funding data must mean no opinion, not a rejection"


def test_crowded_coin_is_skipped_and_calm_one_is_not(settings):
    t0 = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    ms = int(t0.timestamp() * 1000)
    _write_funding(settings, "HOTUSDT", ms, 60, rate=0.0030)     # expensive to be long
    _write_funding(settings, "MIDUSDT", ms, 60, rate=0.0005)
    _write_funding(settings, "CALMUSDT", ms, 60, rate=0.0001)
    _write_funding(settings, "COLDUSDT", ms, 60, rate=-0.0005)
    times = [t0 + pd.Timedelta(hours=8 * i) for i in range(30, 40)]
    syms = ["HOTUSDT", "MIDUSDT", "CALMUSDT", "COLDUSDT"]
    p = funding.attach(_bar_panel(times, syms), settings)
    bar = p[p["bar_time"] == times[-1]].copy()
    bar = bar.assign(regime_on=True, breakout=True, rs_7d=0.1,
                     score=[90.0, 89.0, 88.0, 87.0][: len(bar)])

    class S:
        candidates = 5
    kept = set(signals.entry_candidates(bar, S, flags=Flags(funding_max_pct=0.8))["symbol"])
    assert "HOTUSDT" not in kept, "the most crowded trade is skipped"
    assert {"MIDUSDT", "CALMUSDT", "COLDUSDT"} <= kept


def test_flag_off_is_inert(settings):
    panel = _panel()
    a = backtest.run(panel, settings, flags=Flags(abs_momentum=True))
    b = backtest.run(panel, settings, flags=Flags(abs_momentum=True, funding_max_pct=0.0))
    assert a.equity.equals(b.equity)


def test_filter_can_only_remove_candidates(settings):
    """Without funding columns present the filter must not crash or invent candidates."""
    panel = _panel()
    base = backtest.run(panel, settings, flags=Flags(abs_momentum=True))
    filt = backtest.run(panel, settings, flags=Flags(abs_momentum=True, funding_max_pct=0.8))
    assert len(filt.trades) <= len(base.trades)
    assert np.isfinite(filt.equity).all()


def test_age_days_flags_a_stale_download(settings):
    t0 = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    _write_funding(settings, "AAAUSDT", int(t0.timestamp() * 1000), 3)
    now = t0 + pd.Timedelta(days=5)
    assert funding.age_days(settings, "AAAUSDT", now=now.to_pydatetime()) > 4
    assert funding.age_days(settings, "MISSINGUSDT") == float("inf")
