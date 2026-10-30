"""Step 1: the two engines must not reach into each other's positions.

The rule, stated once:

    a CORE position exits only on close < its own SMA200
    a BREAKOUT position exits only on the Donchian low, rank > 10, or regime off

What makes that reliable is that the owning engine is STORED on the position, not inferred. Two
earlier attempts to infer it both broke:

    reason.startswith("core")   the live path writes "live core: ...", which does not match, so a
                                live core buy could never sell itself
    symbol in core_symbols()    wrong for a coin the breakout engine bought before the core was
                                switched on - that position still has a Donchian stop and would
                                silently stop being exited

So the tag travels: buy(engine=) -> record_buy(engine=) -> positions.engine, and through
queued_orders.engine for the paper path where the fill happens a bar later.
"""

import json

import pandas as pd
import pytest

from app import broker, risk, universe
from app.flags import Flags
from app.main import Bot
from app.portfolio import Store

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
    b.queued_fills = False
    return b


def _levels(sma200=100.0, don_low=95.0):
    return {"bar_time": "2026-10-13 03:00:00+00:00", "regime": True,
            "entry": {}, "stop": {"BTCUSDT": 90.0}, "exit": {"BTCUSDT": don_low},
            "rank": {"BTCUSDT": 1.0},
            "core": {"BTCUSDT": {"dist_sma200": 0.05, "sma200": sma200, "on": True}},
            "core_weight": 0.5, "max_names": 3}


# ---------------------------------------------------------------- the tag itself

def test_the_engine_is_stored_on_the_position(settings):
    st = Store(settings.data_dir / "tag.sqlite")
    st.record_buy({"symbol": "BTCUSDT", "qty": 1.0, "price": 100.0, "fee": 0.1,
                   "time": "t", "client_order_id": "c1"}, stop=None, reason="core: x",
                  engine="core")
    st.record_buy({"symbol": "AAAUSDT", "qty": 1.0, "price": 50.0, "fee": 0.1,
                   "time": "t", "client_order_id": "c2"}, stop=45.0, reason="breakout")
    pos = st.positions()
    assert pos["BTCUSDT"]["engine"] == "core"
    assert pos["AAAUSDT"]["engine"] == "breakout"


def test_an_old_book_defaults_to_breakout(settings):
    """The live books predate the column; every row already in them is a breakout position."""
    import sqlite3
    path = settings.data_dir / "old.sqlite"
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE positions (symbol TEXT PRIMARY KEY, qty REAL, entry_price REAL, "
        "entry_time TEXT, stop REAL, cost REAL, reason TEXT, client_order_id TEXT);"
        "INSERT INTO positions VALUES ('AAAUSDT',1.0,50.0,'t',45.0,50.0,'breakout','c1');")
    con.commit()
    con.close()
    st = Store(path)                       # migration runs here
    assert st.positions()["AAAUSDT"]["engine"] == "breakout"


def test_the_tag_survives_the_queued_fill(settings, monkeypatch):
    """Paper queues the order and fills a bar later; the tag must not be lost in between."""
    st = Store(settings.data_dir / "q.sqlite")
    st.queue_order("bar1", "BUY", "BTCUSDT", 1.0, None, "core: above its line", engine="core")
    due = st.due_orders()
    assert due and due[0]["engine"] == "core"


# ---------------------------------------------------------------- isolation

def test_a_core_position_is_never_sold_by_a_donchian_level(bot):
    """The headline requirement of step 1."""
    bot.store.put("levels", _levels(sma200=100.0, don_low=102.0))
    bot.levels = None
    bot.live_tick(shared_prices={"BTCUSDT": 103.0})
    assert bot.store.positions()["BTCUSDT"]["engine"] == "core"
    bot.store.put("acted_this_bar", [])
    # below the Donchian low (102), still above its 200-day average (100)
    bot.live_tick(shared_prices={"BTCUSDT": 101.0})
    assert "BTCUSDT" in bot.store.positions(), "a Donchian level closed a core position"


def test_a_breakout_position_in_a_core_symbol_keeps_its_donchian_exit(bot):
    """The case symbol-based identification got wrong.

    A coin bought by the breakout engine BEFORE the core was switched on is still a breakout
    position. It must keep the exit it was opened with, not inherit the core's.
    """
    bot.store.record_buy({"symbol": "BTCUSDT", "qty": 1.0, "price": 103.0, "fee": 0.1,
                          "time": "t", "client_order_id": "old"}, stop=90.0,
                         reason="breakout, rank 1", engine="breakout")
    bot.store.put("levels", _levels(sma200=100.0, don_low=102.0))
    bot.levels = None
    bot.live_tick(shared_prices={"BTCUSDT": 101.0})     # under the low, over the 200-day line
    assert "BTCUSDT" not in bot.store.positions(), \
        "a breakout position in a core symbol was given the core's exit"


def test_the_core_still_exits_on_its_own_average(bot):
    bot.store.put("levels", _levels(sma200=100.0))
    bot.levels = None
    bot.live_tick(shared_prices={"BTCUSDT": 101.0})
    assert "BTCUSDT" in bot.store.positions()
    bot.store.put("acted_this_bar", [])
    bot.live_tick(shared_prices={"BTCUSDT": 99.0})
    assert "BTCUSDT" not in bot.store.positions()
