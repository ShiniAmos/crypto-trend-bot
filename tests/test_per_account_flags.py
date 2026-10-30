"""One config on one book, another on the rest - and no leakage between them.

Rolling the trend core out to the demo book alone needs per-account flags, and the dangerous part
is not reading the variable. It is that `Fleet.scan_once` computes ONE candidate list from
`bots[0].flags` and hands it to every account. The moment two accounts differ, that list is wrong
for all but the first, and the demo book would quietly trade the paper book's signals - a bug that
would look exactly like "the strategy works" or "the strategy doesn't", at random.

These tests pin the override and that leak.
"""

import json

import pandas as pd
import pytest

from app import broker, data, risk, signals, universe
from app.flags import Flags
from app.main import Bot


def test_an_account_variable_beats_the_shared_one(monkeypatch, settings):
    monkeypatch.setenv("ENABLE_TREND_CORE", "0")
    monkeypatch.setenv("DEMO_ENABLE_TREND_CORE", "0.25")
    assert Flags.from_env(settings, account="demo").trend_core == 0.25
    assert Flags.from_env(settings, account="paper").trend_core == 0.0
    assert Flags.from_env(settings).trend_core == 0.0


def test_the_shared_variable_still_applies_without_an_override(monkeypatch, settings):
    monkeypatch.setenv("ENABLE_TREND_CORE", "0.5")
    assert Flags.from_env(settings, account="demo").trend_core == 0.5
    assert Flags.from_env(settings, account="paper").trend_core == 0.5


def test_max_names_can_be_overridden_per_account(monkeypatch, settings):
    """Config 4 is core + 5 slots. If only the flag were per-account, half of it would apply."""
    monkeypatch.setenv("DEMO_MAX_NAMES", "5")
    demo = Bot(__import__("dataclasses").replace(settings, account_name="demo"))
    paper = Bot(__import__("dataclasses").replace(settings, account_name="paper"))
    assert demo.max_names == 5
    assert paper.max_names == settings.max_names


def _panel():
    """One bar where BTC is above its 200-day average AND has broken out, as has an alt.

    BTC needs the breakout for this test to mean anything: without the core it is a legitimate
    candidate, and with the core it is reserved. That difference is the leak being tested, and the
    first version of this fixture gave BTC no breakout, so BTC was absent from both lists and the
    test "failed" on its own premise rather than on the code.
    """
    t = pd.Timestamp("2026-10-10 12:00", tz="UTC")
    rows = []
    for sym, px, brk in (("BTCUSDT", 100.0, True), ("AAAUSDT", 50.0, True)):
        rows.append({
            "bar_time": t, "symbol": sym, "open": px, "high": px * 1.01, "low": px * 0.99,
            "close": px, "dist_sma200": 0.10, "dist_sma50": 0.03, "btc_dist_sma200": 0.10,
            "regime_on": True, "eligible": True, "breakout": brk, "breakdown": False,
            "breakdown_exit": False, "stop_level": px * 0.9, "don_high": px * 1.02,
            "don_low": px * 0.9, "don_pos": 0.5, "rs_7d": 0.1, "rs_30d": 0.1, "ret_7d": 0.1,
            "ret_30d": 0.2, "ret_4h": 0.0, "vol_30d": 0.5, "vol_7d": 0.5, "atr_pct": 0.02,
            "volume_ratio": 1.5, "taker_ratio": 0.6, "quote_volume": 1e7,
            "score": 90.0 if sym == "AAAUSDT" else 80.0,
            "rank": 1.0 if sym == "AAAUSDT" else 2.0, "n_eligible": 2})
    return pd.DataFrame(rows), t


def test_a_book_never_trades_another_books_candidate_list(monkeypatch, settings):
    """The leak. The shared `cands` is built with the CORE book's flags; the control book is
    handed it anyway and must still re-derive its own, or it inherits the treatment."""
    uni = {"date": "2026-10-10", "symbols": ["BTCUSDT", "AAAUSDT"],
           "coins": [{"symbol": s, "cg_rank": i, "live_eligible": True}
                     for i, s in enumerate(["BTCUSDT", "AAAUSDT"], 1)], "dropped": []}
    settings.universe_path.write_text(json.dumps(uni))
    monkeypatch.setattr(universe, "tradable_status", lambda s: True)
    monkeypatch.setattr(universe, "build", lambda s, **k: uni)
    monkeypatch.setattr(data, "price", lambda s: 100.0)
    monkeypatch.setattr(broker.PaperBroker, "filters", lambda self, s: risk.SymbolFilters())

    panel, t = _panel()
    core_flags = Flags(trend_core=0.25, trend_core_symbols="BTCUSDT")
    # the list the fleet would compute from the TREATMENT book: BTC is reserved by its core, so
    # BTC is absent and only the alt is a candidate
    shared_cands = signals.entry_candidates(panel, settings, flags=core_flags)

    control = Bot(__import__("dataclasses").replace(settings, account_name="paper"))
    control.flags = Flags()          # no core: BTC is NOT reserved for this book
    control.on_bar_close(force=True, shared=(uni, panel, t, panel, shared_cands))

    # The control book re-derived its own candidates, so it is allowed to consider BTC again.
    # If it had used the shared list, BTC could never appear in its signals.
    assert "BTCUSDT" in set(control.store.get("last_signals") or []), \
        "the control book inherited the treatment book's candidate list"
