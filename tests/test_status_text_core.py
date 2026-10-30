"""A core position must not break the status line or the daily summary.

Found on the server on 17 Oct, failing every day since the core shipped:

    File "/opt/bot/app/main.py", line 796, in status_text
      f"({prices[s] / p['entry_price'] - 1:+.1%}) stop {p['stop']:.6g}")
    TypeError: unsupported format string passed to NoneType.__format__

A core position has no stop by design - its exit is its own 200-day average - so `stop` is None and
"{None:.6g}" raises. The consequence was not cosmetic: status_text() feeds daily_summary(), so the
DEMO book, the only one holding core positions and the one being trialled, had never once sent a
daily summary.

The same formatting bug was fixed in buy() when the core shipped and missed here, which is why
these tests cover every place a position is rendered rather than just the one that crashed.
"""

import json

import pandas as pd
import pytest

from app import broker, data, risk, universe
from app.flags import Flags
from app.main import Bot


@pytest.fixture
def bot(settings, monkeypatch):
    uni = {"date": "2026-10-17", "symbols": ["BTCUSDT", "AAAUSDT"],
           "coins": [{"symbol": s, "cg_rank": i, "live_eligible": True}
                     for i, s in enumerate(["BTCUSDT", "AAAUSDT"], 1)], "dropped": []}
    settings.universe_path.write_text(json.dumps(uni))
    monkeypatch.setattr(universe, "tradable_status", lambda s: True)
    monkeypatch.setattr(data, "price", lambda s: {"BTCUSDT": 82000.0}.get(s, 55.0))
    monkeypatch.setattr(broker.PaperBroker, "filters", lambda self, s: risk.SymbolFilters())
    b = Bot(settings)
    b.flags = Flags(trend_core=0.25, trend_core_symbols="BTCUSDT")
    return b


def _hold_core(b):
    b.store.record_buy({"symbol": "BTCUSDT", "qty": 0.00744255, "price": 83324.65,
                        "fee": 0.6, "time": "2026-10-13T04:08:09+00:00",
                        "client_order_id": "c-core"},
                       stop=None, reason="core: +15.6% above its 200-day average",
                       engine="core")


def _hold_breakout(b):
    b.store.record_buy({"symbol": "AAAUSDT", "qty": 10.0, "price": 50.0, "fee": 0.05,
                        "time": "2026-10-13T04:08:09+00:00", "client_order_id": "c-brk"},
                       stop=45.0, reason="breakout, rank 1", engine="breakout")


def test_status_text_survives_a_core_position(bot):
    """The crash itself."""
    _hold_core(bot)
    text = bot.status_text()
    assert "BTCUSDT" in text
    assert "200-day average" in text, text


def test_daily_summary_survives_a_core_position(bot):
    """What the crash actually cost: the demo book's daily report."""
    _hold_core(bot)
    text = bot.daily_summary()
    assert "Daily summary" in text and "BTCUSDT" in text


def test_a_breakout_position_still_shows_its_stop(bot):
    _hold_breakout(bot)
    text = bot.status_text()
    assert "stop 45" in text, text


def test_both_engines_render_together(bot):
    _hold_core(bot)
    _hold_breakout(bot)
    text = bot.status_text()
    assert "200-day average" in text and "stop 45" in text


def test_a_breakout_position_with_no_stop_does_not_crash(bot):
    """Defensive: a stop could be missing for reasons other than the core - a recovered book, a
    partial write. It must read as 'no stop recorded', not raise."""
    bot.store.record_buy({"symbol": "AAAUSDT", "qty": 10.0, "price": 50.0, "fee": 0.05,
                          "time": "t", "client_order_id": "c-nostop"},
                         stop=None, reason="breakout, rank 1", engine="breakout")
    text = bot.status_text()
    assert "no stop recorded" in text, text


def test_one_books_summary_failure_does_not_silence_the_others(settings, monkeypatch):
    """The fleet must keep reporting on the other books when one of them raises.

    This drives the real Fleet.daily(), with only its download work stubbed out - an earlier
    version of this test reimplemented the loop, which would have passed whatever fleet.py did.
    """
    from app import backfill, fleet as fleet_mod
    monkeypatch.setattr(universe, "tradable_status", lambda s: True)
    monkeypatch.setattr(broker.PaperBroker, "filters", lambda self, s: risk.SymbolFilters())
    monkeypatch.setattr(fleet_mod.universe, "build",
                        lambda s, **k: {"date": "2026-10-17", "symbols": ["BTCUSDT"],
                                        "coins": [], "dropped": []})
    monkeypatch.setattr(backfill, "run", lambda *a, **k: None)

    members = [{"account": "paper", "mode": "paper", "start_equity": 10_000.0, "telegram": False},
               {"account": "paper100", "mode": "paper", "start_equity": 100.0, "telegram": False}]
    f = fleet_mod.Fleet(settings, members, log=lambda *_: None)
    sent = []
    for b in f.bots:
        b.tg.send = lambda m, _s=sent: _s.append(m)
    # the FIRST book raises; the second must still get its summary out
    f.bots[0].daily_summary = lambda: (_ for _ in ()).throw(TypeError("boom"))
    f.bots[1].daily_summary = lambda: "ok"

    # daily() returns early before 00:05 UTC, so pretend it is later and that today is new
    import datetime as _dt

    class _Now(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return _dt.datetime(2026, 10, 17, 12, 0, tzinfo=_dt.timezone.utc)

    monkeypatch.setattr(fleet_mod, "datetime", _Now)
    f._last_daily = None
    f.daily()

    assert any("paper100" in m for m in sent), \
        "the second book was silenced by the first book's failure"
    assert not any("[paper]" in m for m in sent), "the failing book should not have reported"
