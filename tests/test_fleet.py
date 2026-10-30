"""One engine, several accounts.

The point of the fleet is that every account sees the same signal at the same instant, so any
difference between the paper book and the demo book is a difference in EXECUTION, not in thinking.
These tests pin that down, and pin down the separation that must survive it: separate databases,
separate money, separate risk state.
"""

import pandas as pd
import pytest

from app import broker, data, fleet, risk, universe
from app.flags import Flags


@pytest.fixture
def three(settings, monkeypatch):
    uni = {"date": "2026-10-05", "symbols": ["BTCUSDT", "AAAUSDT"],
           "coins": [{"symbol": s, "cg_rank": i, "live_eligible": True}
                     for i, s in enumerate(["BTCUSDT", "AAAUSDT"], 1)], "dropped": []}
    settings.universe_path.write_text(__import__("json").dumps(uni))
    monkeypatch.setattr(universe, "tradable_status", lambda s: True)
    monkeypatch.setattr(universe, "build", lambda s, **k: uni)
    monkeypatch.setattr(data, "price", lambda s: 100.0)
    monkeypatch.setattr(broker.PaperBroker, "filters", lambda self, s: risk.SymbolFilters())
    members = [{"account": "paper", "mode": "paper", "start_equity": 10_000.0, "telegram": False},
               {"account": "paper100", "mode": "paper", "start_equity": 100.0, "telegram": False}]
    return fleet.Fleet(settings, members, log=lambda *_: None)


def test_each_account_keeps_its_own_book(three):
    paths = {b.s.db_path for b in three.bots}
    assert len(paths) == 2, "two accounts must not share one database"
    assert {b.name for b in three.bots} == {"paper", "paper100"}
    sizes = {b.name: b.s.start_equity for b in three.bots}
    assert sizes == {"paper": 10_000.0, "paper100": 100.0}


def test_members_never_download_for_themselves(three):
    """The fleet owns the data; a member fetching its own would be the old waste."""
    assert all(b.s.shared_data for b in three.bots)


def test_only_one_account_speaks_on_telegram(settings):
    settings.telegram_token = "x"
    settings.telegram_chat_id = "1"
    members = fleet.default_members(settings)
    assert sum(m["telegram"] for m in members) == 1, "one voice per signal, not three"


def test_demo_joins_only_when_keys_exist(settings):
    assert all(m["mode"] != "demo" for m in fleet.default_members(settings)), "no keys, no demo"
    settings.binance_key = "k"
    settings.binance_secret = "s"
    assert any(m["mode"] == "demo" for m in fleet.default_members(settings))


def test_one_scan_feeds_every_account(three, monkeypatch):
    """The whole point: the scan runs once, not once per account."""
    calls = []

    def fake_scan(settings, symbols, **kw):
        calls.append(symbols)
        t = pd.Timestamp("2026-10-05 12:00", tz="UTC")
        rows = pd.DataFrame([{"symbol": "BTCUSDT", "bar_time": t, "open": 99.0, "low": 98.0,
                              "close": 100.0, "stop_level": 90.0, "don_high": 101.0,
                              "don_low": 92.0, "rank": 1.0, "score": 99.0, "eligible": True,
                              "regime_on": True, "breakout": False, "breakdown": False,
                              "rs_7d": 0.1, "ret_30d": 0.2, "volume_ratio": 1.0,
                              "taker_ratio": 0.6}])
        return rows, t, rows.set_index("symbol"), rows.iloc[0:0]

    monkeypatch.setattr(fleet.ranker, "scan", fake_scan)
    u, panel, bar_time, rows, cands = three.scan_once()
    assert len(calls) == 1, "one scan for the whole fleet"
    assert str(bar_time).startswith("2026-10-05")


def test_the_universe_hears_about_every_accounts_positions(three, monkeypatch):
    """A coin one account holds must not be dropped from the list because another is flat."""
    seen = {}

    def fake_build(s, held_symbols=None, **k):
        seen["held"] = set(held_symbols or ())
        return {"date": "2026-10-05", "symbols": ["BTCUSDT"], "coins": [], "dropped": []}

    monkeypatch.setattr(fleet.universe, "build", fake_build)
    monkeypatch.setattr(fleet.ranker, "scan",
                        lambda s, syms, **kw: (pd.DataFrame(), None, pd.DataFrame(), pd.DataFrame()))
    three.bots[1].store.record_buy({"time": "t", "symbol": "ZZZUSDT", "qty": 1.0, "price": 10.0,
                                    "fee": 0.0, "client_order_id": "c1"}, stop=9.0, reason="t")
    three.scan_once()
    assert "ZZZUSDT" in seen["held"], "the fleet must pool every account's holdings"


def test_one_price_fetch_covers_the_fleet(three, monkeypatch):
    calls = []
    monkeypatch.setattr(fleet.data, "prices", lambda syms: (calls.append(list(syms)) or
                                                            {s: 100.0 for s in syms}))
    for b in three.bots:
        b.levels = {"bar_time": "t", "regime": True, "entry": {"AAAUSDT": 100.0},
                    "stop": {}, "exit": {}, "rank": {}}
    three.prices_once()
    assert len(calls) == 1 and "AAAUSDT" in calls[0]


def test_a_members_risk_state_is_its_own(three):
    a, b = three.bots
    a.store.put("risk", {"peak_equity": 1.0, "day": "x", "day_start_equity": 1.0,
                         "killed": True, "halted_today": False, "killed_day": "x"})
    assert (b.store.get("risk") or {}).get("killed") is not True, \
        "one account hitting its kill switch must not stop the others"
