"""The "Next move" page reads a snapshot; this pins what the snapshot must contain.

ui/next_actions.py deliberately re-derives nothing - it only reads `levels` and compares the saved
trigger prices to a live price. That keeps the page from ever disagreeing with the bot, and it
makes the snapshot a contract: if a field goes missing, the page silently shows less rather than
failing, which is the worst outcome for a page whose whole job is to say what will happen next.
"""

import json

import pandas as pd
import pytest

from app import broker, data, risk, signals, universe
from app.flags import Flags
from app.main import Bot


def _rows():
    t = pd.Timestamp("2026-10-10 12:00", tz="UTC")
    out = []
    for sym, px, d200, brk in (("BTCUSDT", 100.0, 0.10, False),
                               ("AAAUSDT", 50.0, 0.04, True)):
        out.append({
            "bar_time": t, "symbol": sym, "open": px, "high": px * 1.01, "low": px * 0.99,
            "close": px, "dist_sma200": d200, "dist_sma50": 0.03, "btc_dist_sma200": 0.10,
            "regime_on": True, "eligible": True, "breakout": brk, "breakdown": False,
            "breakdown_exit": False, "stop_level": px * 0.9, "don_high": px * 1.02,
            "don_low": px * 0.9, "don_pos": 0.5, "rs_7d": 0.1, "rs_30d": 0.1, "ret_7d": 0.1,
            "ret_30d": 0.2, "ret_4h": 0.0, "vol_30d": 0.5, "vol_7d": 0.5, "atr_pct": 0.02,
            "volume_ratio": 1.5, "taker_ratio": 0.6, "quote_volume": 1e7,
            "score": 90.0 if sym == "AAAUSDT" else 80.0,
            "rank": 1.0 if sym == "AAAUSDT" else 2.0, "n_eligible": 2})
    return pd.DataFrame(out).set_index("symbol"), t


@pytest.fixture
def bot(settings, monkeypatch):
    monkeypatch.setattr(universe, "tradable_status", lambda s: True)
    monkeypatch.setattr(data, "price", lambda s: 100.0)
    monkeypatch.setattr(broker.PaperBroker, "filters", lambda self, s: risk.SymbolFilters())
    b = Bot(settings)
    b.flags = Flags(trend_core=0.25, trend_core_symbols="BTCUSDT")
    return b


def test_the_snapshot_carries_the_cores_own_trigger(bot):
    """Not BTC's regime - the coin's own distance from its own 200-day average."""
    latest, t = _rows()
    lv = bot.snapshot_levels(latest, t, {"BTCUSDT", "AAAUSDT"}, regime=True)
    assert "core" in lv, "the page cannot describe the core without this"
    assert lv["core"]["BTCUSDT"]["dist_sma200"] == pytest.approx(0.10)
    assert lv["core"]["BTCUSDT"]["on"] is True


def test_the_snapshot_says_how_big_the_core_is_and_how_many_slots(bot):
    latest, t = _rows()
    lv = bot.snapshot_levels(latest, t, {"BTCUSDT", "AAAUSDT"}, regime=True)
    assert lv["core_weight"] == pytest.approx(0.25)
    assert lv["max_names"] == bot.max_names


def test_a_core_below_its_average_is_reported_as_off(bot):
    latest, t = _rows()
    latest.loc["BTCUSDT", "dist_sma200"] = -0.03
    lv = bot.snapshot_levels(latest, t, {"BTCUSDT", "AAAUSDT"}, regime=True)
    assert lv["core"]["BTCUSDT"]["on"] is False


def test_no_core_section_when_the_core_is_off(settings, monkeypatch):
    monkeypatch.setattr(universe, "tradable_status", lambda s: True)
    monkeypatch.setattr(broker.PaperBroker, "filters", lambda self, s: risk.SymbolFilters())
    b = Bot(settings)
    b.flags = Flags()
    latest, t = _rows()
    lv = b.snapshot_levels(latest, t, {"BTCUSDT", "AAAUSDT"}, regime=True)
    assert lv["core"] == {}
    assert lv["core_weight"] == 0.0


def test_the_snapshot_round_trips_through_the_database(bot):
    """The page reads it back as JSON from sqlite, so it has to survive that trip."""
    latest, t = _rows()
    bot.snapshot_levels(latest, t, {"BTCUSDT", "AAAUSDT"}, regime=True)
    back = bot.store.get("levels")
    assert back["core"]["BTCUSDT"]["on"] is True
    assert isinstance(back["entry"], dict) and isinstance(back["exit"], dict)
    json.dumps(back)      # must stay plain JSON: no numpy types leaking in


def test_entry_and_exit_levels_are_the_prices_the_page_shows(bot):
    """The page prints these as "it will buy above X" - they must be the real trigger prices."""
    latest, t = _rows()
    lv = bot.snapshot_levels(latest, t, {"BTCUSDT", "AAAUSDT"}, regime=True)
    # AAAUSDT passes the filters, so it is on the watch list at its 20-bar high
    assert lv["entry"]["AAAUSDT"] == pytest.approx(float(latest.loc["AAAUSDT", "don_high"]))
    assert lv["exit"]["AAAUSDT"] == pytest.approx(float(latest.loc["AAAUSDT", "don_low"]))
