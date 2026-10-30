"""The live loop must ACT on the core, not merely compute it.

tests/test_trend_core.py proves the backtest holds the core. That is not the same claim: the live
path is a different code route, and the whole complaint behind this feature was a bot that "sees
everything and does nothing". These tests drive Bot.on_bar_close directly against a paper broker
and assert that orders actually happen.

Covered here and nowhere else:
  1. a core buy is placed with no breakout anywhere in the data
  2. it happens in APPROVAL mode too - the core is the book's default state, not a trade idea
     waiting to be confirmed, and approval mode is where the old bot did nothing
  3. the core sells itself when the coin loses its 200-day average
  4. the core does not consume the breakout slots
  5. kill switch, safe mode and a non-TRADING status each stop it
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
    uni = {"date": "2026-10-10", "symbols": ["BTCUSDT", "AAAUSDT"],
           "coins": [{"symbol": s, "cg_rank": i, "live_eligible": True}
                     for i, s in enumerate(["BTCUSDT", "AAAUSDT"], 1)], "dropped": []}
    settings.universe_path.write_text(json.dumps(uni))
    monkeypatch.setattr(universe, "tradable_status", lambda s: True)
    monkeypatch.setattr(universe, "build", lambda s, **k: uni)
    monkeypatch.setattr(data, "price", lambda s: 100.0)
    monkeypatch.setattr(broker.PaperBroker, "filters", lambda self, s: risk.SymbolFilters())
    return uni


def _bar(dist200_btc=0.10, alt_breakout=False):
    """One closed bar for two coins, with every column the live loop reads."""
    t = pd.Timestamp("2026-10-10 12:00", tz="UTC")
    rows = []
    for sym, px, d200 in (("BTCUSDT", 100.0, dist200_btc), ("AAAUSDT", 50.0, 0.05)):
        rows.append({
            "bar_time": t, "symbol": sym, "open": px, "high": px * 1.01, "low": px * 0.99,
            "close": px, "dist_sma200": d200, "dist_sma50": 0.03,
            "btc_dist_sma200": dist200_btc, "regime_on": dist200_btc > 0, "eligible": True,
            "breakout": (sym == "AAAUSDT" and alt_breakout), "breakdown": False,
            "breakdown_exit": False, "stop_level": px * 0.9, "don_high": px * 1.02,
            "don_low": px * 0.9, "don_pos": 0.5, "rs_7d": 0.1, "rs_30d": 0.1,
            "ret_7d": 0.1, "ret_30d": 0.2, "ret_4h": 0.0, "vol_30d": 0.5, "vol_7d": 0.5,
            "atr_pct": 0.02, "volume_ratio": 1.5, "taker_ratio": 0.6, "quote_volume": 1e7,
            "score": 90.0 if sym == "AAAUSDT" else 10.0,
            "rank": 1.0 if sym == "AAAUSDT" else 2.0, "n_eligible": 2})
    p = pd.DataFrame(rows)
    return p, t


def _step(b, settings, dist200_btc=0.10, alt_breakout=False, offset_bars=0):
    """Feed the bot one closed bar.

    Orders are DECIDED on a close and FILLED at the next bar's open - the same convention the
    backtest uses, so that a live fill price can never be one the bot saw before deciding. A single
    bar therefore only ever produces a queued order, which is why every assertion below runs at
    least two bars.
    """
    from app import signals
    panel, t = _bar(dist200_btc, alt_breakout)
    t = t + pd.Timedelta("4h") * offset_bars
    panel["bar_time"] = t
    u = json.loads(settings.universe_path.read_text())
    cands = signals.entry_candidates(panel, settings, flags=b.flags)
    b.on_bar_close(force=True, shared=(u, panel, t, panel, cands))
    return b


def _run(settings, flags, dist200_btc=0.10, alt_breakout=False, approval=False, bars=2):
    b = Bot(settings)
    b.flags = flags
    b.approval = approval
    for i in range(bars):
        _step(b, settings, dist200_btc, alt_breakout, offset_bars=i)
    return b


def test_the_core_buys_live_with_no_breakout_anywhere(bot, settings):
    """The headline: nothing breaks out, and the bot still opens a position."""
    b = _run(settings, Flags(**CORE), alt_breakout=False)
    pos = b.store.positions()
    assert "BTCUSDT" in pos, "the live loop computed the core and did nothing"
    assert "200-day" in pos["BTCUSDT"]["reason"]


def test_the_core_acts_even_in_approval_mode(bot, settings):
    """Approval mode is exactly where the old bot watched and waited. The core must not."""
    b = _run(settings, Flags(**CORE), alt_breakout=True, approval=True)
    assert "BTCUSDT" in b.store.positions(), "approval mode blocked the core"


def test_the_core_sells_itself_when_the_average_is_lost(bot, settings):
    b = _run(settings, Flags(**CORE), dist200_btc=0.10, bars=2)
    assert "BTCUSDT" in b.store.positions(), "the core never opened, so the exit is untested"
    # the same bot, now below its own 200-day average: one bar to decide, one to fill
    _step(b, settings, dist200_btc=-0.05, offset_bars=2)
    _step(b, settings, dist200_btc=-0.05, offset_bars=3)
    assert "BTCUSDT" not in b.store.positions(), "the core held on below its 200-day average"


def test_the_core_leaves_the_breakout_slots_alone(bot, settings):
    """Both engines should be able to hold something on the same bar."""
    b = _run(settings, Flags(**CORE), alt_breakout=True)
    pos = set(b.store.positions())
    assert "BTCUSDT" in pos, "the core did not buy"
    assert "AAAUSDT" in pos, "the core consumed the breakout slot"


def test_the_kill_switch_stops_the_core(bot, settings):
    settings.kill_switch_path.write_text("stop")
    b = _run(settings, Flags(**CORE))
    assert "BTCUSDT" not in b.store.positions(), "the kill switch did not stop the core"


def test_safe_mode_stops_the_core(bot, settings, monkeypatch):
    monkeypatch.setattr(reconcile, "safe_mode", lambda store: True)
    b = _run(settings, Flags(**CORE))
    assert "BTCUSDT" not in b.store.positions(), "safe mode did not stop the core"


def test_a_suspended_symbol_stops_the_core(bot, settings, monkeypatch):
    monkeypatch.setattr(universe, "tradable_status", lambda s: s != "BTCUSDT")
    b = _run(settings, Flags(**CORE))
    assert "BTCUSDT" not in b.store.positions(), "the core bought a non-TRADING symbol"


def test_core_off_changes_nothing_live(bot, settings):
    """With the flag off the live loop must behave exactly as before: no core position."""
    b = _run(settings, Flags(), alt_breakout=True)
    assert "BTCUSDT" not in b.store.positions()
