"""`core_dual_ma`: hold only while above BOTH the 50-day and the 200-day average.

From P14: the 200-day line alone is slow to let go, and on single-asset runs it whipsawed XRP and
BNB 80 and 70 times at 5.0% and 8.6% win rates. Requiring the faster line too is a slower filter in
effect - fewer, longer holds.

The thing most likely to go wrong is not the rule, it is the two code paths disagreeing about it:
the backtest decides from `dist_sma50`/`dist_sma200` columns, the live tick from price levels
frozen in a snapshot. If one checks two lines and the other checks one, the live book holds what
the backtest sold and every comparison between them becomes meaningless. Half these tests exist
for that.
"""

import json

import numpy as np
import pandas as pd
import pytest

from app import backtest, broker, data, risk, signals, universe
from app.flags import Flags
from app.main import Bot

CORE = dict(trend_core=0.5, trend_core_symbols="BTCUSDT")


# ---------------------------------------------------------------- the predicate

def _bar(d200, d50):
    return pd.DataFrame({"symbol": ["BTCUSDT"], "dist_sma200": [d200],
                         "dist_sma50": [d50], "close": [100.0]})


def test_dual_ma_needs_both_lines():
    f = Flags(**CORE, core_dual_ma=True)
    assert signals.core_on(_bar(0.10, 0.05), f).iloc[0] is np.True_ or \
        bool(signals.core_on(_bar(0.10, 0.05), f).iloc[0])
    assert not bool(signals.core_on(_bar(0.10, -0.02), f).iloc[0]), "below the 50-day: must be out"
    assert not bool(signals.core_on(_bar(-0.02, 0.05), f).iloc[0]), "below the 200-day: must be out"


def test_without_the_flag_only_the_200_day_matters():
    f = Flags(**CORE)
    assert bool(signals.core_on(_bar(0.10, -0.02), f).iloc[0]), \
        "the 50-day must be ignored unless core_dual_ma is on"


def test_missing_50_day_history_fails_rather_than_passes():
    f = Flags(**CORE, core_dual_ma=True)
    assert not bool(signals.core_on(_bar(0.10, np.nan), f).iloc[0])


# ---------------------------------------------------------------- the backtest

def _panel(n=300, d50_after=None):
    t = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    rows = []
    for i, ts in enumerate(t):
        for sym, px in (("BTCUSDT", 100.0 + 0.02 * i), ("AAAUSDT", 50.0 + 0.3 * i)):
            rows.append({"bar_time": ts, "symbol": sym, "open": px, "high": px * 1.01,
                         "low": px * 0.99, "close": px})
    p = pd.DataFrame(rows)
    p["dist_sma200"] = np.where(p["symbol"] == "BTCUSDT", 0.10, 0.05)
    p["dist_sma50"] = 0.05
    if d50_after is not None:
        cut = p["bar_time"].quantile(0.5)
        p.loc[(p.symbol == "BTCUSDT") & (p.bar_time > cut), "dist_sma50"] = d50_after
    p["btc_dist_sma200"] = 0.10
    p["regime_on"] = True
    p["eligible"] = True
    p["breakout"] = (p["symbol"] == "AAAUSDT") & (p.groupby("symbol").cumcount() > 2)
    p["breakdown"] = False
    p["stop_level"] = p["close"] * 0.9
    p["don_high"] = p["close"] * 1.01
    p["don_low"] = p["close"] * 0.9
    p["don_pos"] = 0.5
    for c, v in (("rs_7d", 0.1), ("rs_30d", 0.1), ("ret_7d", 0.1), ("ret_30d", 0.2),
                 ("vol_30d", 0.5), ("vol_7d", 0.5), ("atr_pct", 0.02), ("volume_ratio", 1.5),
                 ("taker_ratio", 0.6), ("quote_volume", 1e7), ("n_eligible", 2)):
        p[c] = v
    p["ret_4h"] = p.groupby("symbol")["close"].pct_change().fillna(0.0)
    p["score"] = np.where(p["symbol"] == "AAAUSDT", 0.9, 0.1)
    p["rank"] = np.where(p["symbol"] == "AAAUSDT", 1.0, 2.0)
    return p


def test_the_backtest_exits_when_the_50_day_is_lost(settings):
    """Price stays above the 200-day throughout; only the 50-day breaks."""
    p = _panel(d50_after=-0.03)
    assert (p.loc[p.symbol == "BTCUSDT", "dist_sma200"] > 0).all()
    r = backtest.run(p, settings, flags=Flags(**CORE, core_dual_ma=True))
    sells = r.log[(r.log.side == "SELL") & (r.log.symbol == "BTCUSDT")]
    assert not sells.empty, "losing the 50-day average did not close the core"


def test_without_the_flag_the_same_panel_holds(settings):
    p = _panel(d50_after=-0.03)
    r = backtest.run(p, settings, flags=Flags(**CORE))
    sells = r.log[(r.log.side == "SELL") & (r.log.symbol == "BTCUSDT")]
    assert sells.empty, "the 50-day closed a position while the flag was off"


def test_flag_off_is_byte_identical(settings):
    p = _panel()
    a = backtest.run(p, settings, flags=Flags(**CORE))
    b = backtest.run(p, settings, flags=Flags(**CORE, core_dual_ma=False))
    pd.testing.assert_series_equal(a.equity, b.equity)


# ---------------------------------------------------------------- live must agree

@pytest.fixture
def bot(settings, monkeypatch):
    uni = {"date": "2026-10-18", "symbols": ["BTCUSDT"],
           "coins": [{"symbol": "BTCUSDT", "cg_rank": 1, "live_eligible": True}], "dropped": []}
    settings.universe_path.write_text(json.dumps(uni))
    monkeypatch.setattr(universe, "tradable_status", lambda s: True)
    monkeypatch.setattr(broker.PaperBroker, "filters", lambda self, s: risk.SymbolFilters())
    b = Bot(settings)
    b.flags = Flags(**CORE, core_dual_ma=True)
    b.queued_fills = False
    return b


def _levels(sma200=100.0, sma50=105.0, dual=True):
    return {"bar_time": "2026-10-18 03:00:00+00:00", "regime": True, "entry": {},
            "stop": {}, "exit": {"BTCUSDT": 95.0}, "rank": {"BTCUSDT": 1.0},
            "core": {"BTCUSDT": {"dist_sma200": 0.05, "sma200": sma200,
                                 "dist_sma50": 0.01, "sma50": sma50,
                                 "dual_ma": dual, "on": True}},
            "core_weight": 0.5, "max_names": 3}


def test_live_does_not_buy_between_the_two_lines(bot):
    """Above the 200-day (100) but below the 50-day (105): the backtest would not hold it,
    so the live tick must not either."""
    bot.store.put("levels", _levels())
    bot.levels = None
    bot.live_tick(shared_prices={"BTCUSDT": 102.0})
    assert "BTCUSDT" not in bot.store.positions()


def test_live_buys_above_both(bot):
    bot.store.put("levels", _levels())
    bot.levels = None
    bot.live_tick(shared_prices={"BTCUSDT": 106.0})
    assert "BTCUSDT" in bot.store.positions()


def test_live_sells_when_the_50_day_is_lost(bot):
    bot.store.put("levels", _levels())
    bot.levels = None
    bot.live_tick(shared_prices={"BTCUSDT": 106.0})
    assert "BTCUSDT" in bot.store.positions()
    bot.store.put("acted_this_bar", [])
    bot.live_tick(shared_prices={"BTCUSDT": 102.0})     # under the 50-day, over the 200-day
    assert "BTCUSDT" not in bot.store.positions(), "the live core ignored the 50-day average"


def test_the_snapshot_carries_both_levels(bot):
    latest = pd.DataFrame({"symbol": ["BTCUSDT"], "dist_sma200": [0.10], "dist_sma50": [0.04],
                           "close": [104.0], "stop_level": [90.0], "don_high": [105.0],
                           "don_low": [95.0], "rank": [1.0], "eligible": [True],
                           "regime_on": [True], "rs_7d": [0.1]}).set_index("symbol")
    lv = bot.snapshot_levels(latest, pd.Timestamp("2026-10-18", tz="UTC"), {"BTCUSDT"}, True)
    c = lv["core"]["BTCUSDT"]
    assert c["dual_ma"] is True
    assert c["sma200"] == pytest.approx(104.0 / 1.10)
    assert c["sma50"] == pytest.approx(104.0 / 1.04)
