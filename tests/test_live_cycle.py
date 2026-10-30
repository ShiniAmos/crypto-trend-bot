"""The live cycle: levels from closed bars, triggers checked against the live price any second.

The safety property under test is that nothing here can look at an unfinished candle. The channel,
the ranking and every filter are frozen at a bar close; the tick only asks whether price has since
crossed a frozen number.
"""

import json

import pandas as pd
import pytest

from app import broker, data, main, risk, universe
from app.flags import Flags


@pytest.fixture
def live_bot(settings, monkeypatch):
    monkeypatch.setenv("DECISION_TF", "live")
    from app import config as cfg
    s = cfg.Settings()
    uni = {"date": "2026-10-04", "symbols": ["BTCUSDT", "AAAUSDT"],
           "coins": [{"symbol": x, "cg_rank": i, "live_eligible": True}
                     for i, x in enumerate(["BTCUSDT", "AAAUSDT"], 1)], "dropped": []}
    s.universe_path.write_text(json.dumps(uni))
    monkeypatch.setattr(universe, "tradable_status", lambda x: True)
    monkeypatch.setattr(broker.PaperBroker, "filters", lambda self, x: risk.SymbolFilters())
    bot = main.Bot(s)
    bot.store.put("risk", {"peak_equity": 10_000, "day": "2026-10-04",
                           "day_start_equity": 10_000, "killed": False,
                           "halted_today": False, "killed_day": ""})
    bot.levels = {"bar_time": "2026-10-04 12:00:00+00:00", "regime": True,
                  "entry": {"AAAUSDT": 100.0}, "stop": {"AAAUSDT": 90.0},
                  "exit": {"AAAUSDT": 92.0}, "rank": {"AAAUSDT": 1.0}}
    return bot


def _prices(monkeypatch, mapping):
    monkeypatch.setattr(data, "prices", lambda syms: {s: mapping[s] for s in syms if s in mapping})
    monkeypatch.setattr(data, "price", lambda s: mapping.get(s, 0.0))


def test_live_entry_fires_the_moment_price_crosses(live_bot, monkeypatch):
    _prices(monkeypatch, {"AAAUSDT": 100.5})
    live_bot.live_tick()
    pos = live_bot.store.positions()
    assert "AAAUSDT" in pos
    assert pos["AAAUSDT"]["entry_price"] == pytest.approx(100.5 * (1 + live_bot.s.slippage))
    assert pos["AAAUSDT"]["stop"] == 90.0


def test_live_entry_does_not_fire_below_the_level(live_bot, monkeypatch):
    _prices(monkeypatch, {"AAAUSDT": 99.9})
    live_bot.live_tick()
    assert live_bot.store.positions() == {}


def test_a_symbol_is_acted_on_once_per_bar(live_bot, monkeypatch):
    """Price oscillating around a level must not churn the account."""
    _prices(monkeypatch, {"AAAUSDT": 100.5})
    live_bot.live_tick()
    live_bot.handle(__import__("app.notify", fromlist=["parse"]).parse("FLAT AAA"))
    assert live_bot.store.positions() == {}
    live_bot.live_tick()                       # same bar, price still above the trigger
    assert live_bot.store.positions() == {}, "no re-entry on the same bar"


def test_live_exit_fires_on_a_break(live_bot, monkeypatch):
    _prices(monkeypatch, {"AAAUSDT": 100.5})
    live_bot.live_tick()
    assert "AAAUSDT" in live_bot.store.positions()
    live_bot.store.put("acted_this_bar", [])   # a later bar
    _prices(monkeypatch, {"AAAUSDT": 91.0})    # below the 92 exit level
    live_bot.live_tick()
    assert live_bot.store.positions() == {}
    sells = [f for f in live_bot.store.fills() if f["side"] == "SELL"]
    assert "broke" in sells[0]["reason"] and "live" in sells[0]["reason"]


def test_live_respects_the_slot_limit_and_risk_state(live_bot, monkeypatch):
    live_bot.max_names = 0
    _prices(monkeypatch, {"AAAUSDT": 100.5})
    live_bot.live_tick()
    assert live_bot.store.positions() == {}, "slot limit still applies in the live cycle"


def test_live_blocks_entries_in_safe_mode_but_still_exits(live_bot, monkeypatch):
    from app import reconcile
    _prices(monkeypatch, {"AAAUSDT": 100.5})
    live_bot.live_tick()
    assert "AAAUSDT" in live_bot.store.positions()
    live_bot.store.put(reconcile.SAFE_MODE_KEY, {"since": "now", "differences": ["x"]})
    live_bot.store.put("acted_this_bar", [])
    _prices(monkeypatch, {"AAAUSDT": 91.0})
    live_bot.live_tick()
    assert live_bot.store.positions() == {}, "exits must still work in safe mode"


def test_snapshot_uses_only_closed_bar_columns(live_bot):
    """The frozen levels come from the bar's own columns, never from a live price."""
    latest = pd.DataFrame([{"symbol": "AAAUSDT", "don_high": 100.0, "don_low": 92.0,
                            "stop_level": 90.0, "rank": 1.0, "eligible": True, "rs_7d": 0.1,
                            "ret_30d": 0.2, "volume_ratio": 1.0, "taker_ratio": 0.6,
                            "bar_time": pd.Timestamp("2026-10-04 12:00", tz="UTC")}]
                          ).set_index("symbol")
    lv = live_bot.snapshot_levels(latest, pd.Timestamp("2026-10-04 12:00", tz="UTC"),
                                  {"AAAUSDT"}, True)
    assert lv["entry"]["AAAUSDT"] == 100.0 and lv["exit"]["AAAUSDT"] == 92.0
    assert lv["stop"]["AAAUSDT"] == 90.0


def test_live_mode_still_reads_4h_candles(live_bot):
    assert live_bot.s.decision_tf == "live"
    assert live_bot.s.bar_tf == "4h", "levels are built from finished 4h candles"
    assert live_bot.s.live_cycle and live_bot.s.donchian_for_tf == 20


def test_live_tick_is_a_no_op_without_levels(live_bot, monkeypatch):
    """A restart must not trade on stale or missing levels; it waits for a prime."""
    live_bot.levels = None
    live_bot.store.put("levels", None)
    _prices(monkeypatch, {"AAAUSDT": 999.0})
    live_bot.live_tick()
    assert live_bot.store.positions() == {}


def test_level_tf_overrides_the_candle_the_levels_come_from(monkeypatch, tmp_path):
    from app import config as cfg
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DECISION_TF", "live")
    monkeypatch.setenv("LEVEL_TF", "1h")
    s = cfg.Settings()
    assert s.live_cycle and s.bar_tf == "1h"
    assert s.donchian_for_tf == 80, "the same 3.3-day channel, in hourly bars"

    monkeypatch.setenv("LEVEL_TF", "")
    assert cfg.Settings().bar_tf == "4h", "empty follows DECISION_TF"

    monkeypatch.setenv("LEVEL_TF", "5m")
    with pytest.raises(ValueError):
        cfg.Settings()
