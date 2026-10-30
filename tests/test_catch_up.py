import pandas as pd

from app import main


def _bar(t, sym, breakdown=False, rank=1, close=100.0):
    return {"bar_time": pd.Timestamp(t, tz="UTC"), "symbol": sym, "regime_on": True,
            "breakdown": breakdown, "rank": rank, "close": close}


def test_missed_sell_signal_is_caught_up(settings):
    bot = main.Bot(settings)
    bot.store.put("last_bar", "2026-09-22 00:00:00+00:00")
    held = {"ETHUSDT": {"qty": 1.0, "entry_price": 100, "cost": 100}}
    panel = pd.DataFrame([
        _bar("2026-09-22 00:00", "ETHUSDT"),                       # already processed
        _bar("2026-09-22 04:00", "ETHUSDT", breakdown=True, close=90),  # missed: sell signal
        _bar("2026-09-22 08:00", "ETHUSDT", close=95),             # missed
        _bar("2026-09-22 12:00", "ETHUSDT", close=97),             # the bar being processed now
    ])
    out = bot.catch_up(panel, pd.Timestamp("2026-09-22 12:00", tz="UTC"), held)
    assert "broke the 20-bar low" in out["ETHUSDT"] and "missed 22 Sep 04:00" in out["ETHUSDT"]
    with bot.store.tx() as db:
        times = [r["time"] for r in db.execute("SELECT time FROM equity ORDER BY time")]
    assert len(times) == 2  # equity filled in for both missed checks


def test_no_missed_checks_means_nothing_to_do(settings):
    bot = main.Bot(settings)
    bot.store.put("last_bar", "2026-09-22 08:00:00+00:00")
    panel = pd.DataFrame([_bar("2026-09-22 08:00", "ETHUSDT"), _bar("2026-09-22 12:00", "ETHUSDT")])
    assert bot.catch_up(panel, pd.Timestamp("2026-09-22 12:00", tz="UTC"), {"ETHUSDT": {"qty": 1}}) == {}


def test_daily_summary_text(settings, monkeypatch):
    bot = main.Bot(settings)
    text = bot.daily_summary()
    assert text.startswith("Daily summary") and "Since start: +0.00%" in text
