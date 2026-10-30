"""The trend core: a position the bot HOLDS, rather than a trade it waits for.

Everything else in this engine is reactive - it needs a Donchian breakout to act. The core does
not: it is long a coin for exactly as long as that coin's close is above its own 200-day average.
These tests pin down the four things that separate it from the breakout engine, because each one
is a way the two could quietly interfere:

  1. it buys WITHOUT a breakout, and holds through bars the breakout rule would have exited
  2. it exits when the coin loses its 200-day average, and only then
  3. it does not consume the breakout slots
  4. with trend_core=0 it does not exist - byte-identical equity, the parity rule for every flag

and two that protect the book:

  5. the drawdown kill still closes it, because the kill belongs to the account, not the strategy
  6. a core coin is never also bought by the breakout engine
"""

import numpy as np
import pandas as pd
import pytest

from app import backtest, signals
from app.flags import Flags


def _panel(n=420, core_above=True, alt_breakout=True):
    """Two coins: BTC (the core candidate) and one alt, on 4h bars.

    BTC is built to sit clearly above or below a rising 200-day average without ever printing a
    Donchian breakout, so anything the core does here cannot be the breakout engine in disguise.
    """
    t = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    rows = []
    for i, ts in enumerate(t):
        # BTC: a slow drift, no new 20-bar highs after the first stretch
        btc_c = 100.0 + (0.05 * i if core_above else -0.02 * i)
        rows.append({"bar_time": ts, "symbol": "BTCUSDT", "open": btc_c, "high": btc_c,
                     "low": btc_c * 0.999, "close": btc_c})
        alt_c = 50.0 + (0.3 * i if alt_breakout else 0.0)
        rows.append({"bar_time": ts, "symbol": "AAAUSDT", "open": alt_c, "high": alt_c,
                     "low": alt_c * 0.999, "close": alt_c})
    p = pd.DataFrame(rows)
    # the columns the engine reads, set directly so the test states its own conditions
    p["dist_sma200"] = np.where(p["symbol"] == "BTCUSDT",
                                0.10 if core_above else -0.10, 0.05)
    p["dist_sma50"] = 0.05
    p["btc_dist_sma200"] = 0.10 if core_above else -0.10
    p["regime_on"] = p["btc_dist_sma200"] > 0
    p["eligible"] = True
    p["breakout"] = (p["symbol"] == "AAAUSDT") & alt_breakout & (p.groupby("symbol").cumcount() > 2)
    p["breakdown"] = False
    p["stop_level"] = p["close"] * 0.9
    p["don_high"] = p["close"] * 1.01
    p["don_low"] = p["close"] * 0.9
    p["don_pos"] = 0.5
    p["rs_7d"] = 0.1
    p["rs_30d"] = 0.1
    p["ret_30d"] = 0.2
    p["ret_7d"] = 0.1
    p["ret_4h"] = p.groupby("symbol")["close"].pct_change().fillna(0.0)
    p["vol_30d"] = 0.5
    p["vol_7d"] = 0.5
    p["atr_pct"] = 0.02
    p["volume_ratio"] = 1.5
    p["taker_ratio"] = 0.6
    p["quote_volume"] = 1e7
    p["score"] = np.where(p["symbol"] == "AAAUSDT", 0.9, 0.1)
    p["rank"] = np.where(p["symbol"] == "AAAUSDT", 1.0, 2.0)
    p["n_eligible"] = 2
    return p


CORE = dict(trend_core=0.5, trend_core_symbols="BTCUSDT")


def test_the_core_buys_without_waiting_for_a_breakout(settings):
    """The headline difference. BTC never breaks out here, and the core is long it anyway."""
    p = _panel(core_above=True, alt_breakout=False)
    assert not p.loc[p.symbol == "BTCUSDT", "breakout"].any(), "the fixture must offer no breakout"
    r = backtest.run(p, settings, flags=Flags(**CORE))
    buys = r.log[(r.log.side == "BUY") & (r.log.symbol == "BTCUSDT")]
    assert len(buys) >= 1, "the core never entered"
    assert "200-day" in buys.iloc[0]["reason"]


def test_the_core_holds_while_the_coin_is_above_its_average(settings):
    p = _panel(core_above=True, alt_breakout=False)
    r = backtest.run(p, settings, flags=Flags(**CORE))
    sells = r.log[(r.log.side == "SELL") & (r.log.symbol == "BTCUSDT")]
    assert sells.empty, f"the core sold while BTC was above its average: {list(sells.reason)}"
    assert r.exposure.iloc[-1] > 0.2, "the core should be holding at the end of the run"


def test_the_core_exits_when_the_average_is_lost(settings):
    """Above for the first half, below for the second: one entry, one exit, in that order."""
    p = _panel(core_above=True, alt_breakout=False)
    half = p["bar_time"].quantile(0.5)
    flip = p["bar_time"] > half
    p.loc[flip & (p.symbol == "BTCUSDT"), "dist_sma200"] = -0.05
    r = backtest.run(p, settings, flags=Flags(**CORE))
    acts = r.log[(r.log.symbol == "BTCUSDT") & r.log.side.isin(["BUY", "SELL"])]
    assert list(acts.side) == ["BUY", "SELL"], f"expected one round trip, got {list(acts.side)}"
    assert "below" in acts.iloc[-1]["reason"]


def test_the_core_does_not_eat_the_breakout_slots(settings):
    """A core holding plus max_names breakout positions, not max_names in total."""
    p = _panel(core_above=True, alt_breakout=True)
    with_core = backtest.run(p, settings, flags=Flags(**CORE))
    held = with_core.log[with_core.log.side == "BUY"]["symbol"]
    assert "BTCUSDT" in set(held) and "AAAUSDT" in set(held), \
        "the core and the breakout engine should both have been able to buy"


def test_a_core_coin_is_never_bought_by_the_breakout_engine(settings):
    """Both engines wanting one coin would fight over its exit. BTC is reserved."""
    p = _panel(core_above=True, alt_breakout=True)
    p.loc[p.symbol == "BTCUSDT", "breakout"] = True      # tempt the breakout engine
    p.loc[p.symbol == "BTCUSDT", "rank"] = 1.0
    r = backtest.run(p, settings, flags=Flags(**CORE))
    btc_buys = r.log[(r.log.side == "BUY") & (r.log.symbol == "BTCUSDT")]
    assert len(btc_buys) == 1, "BTC was bought twice - both engines took it"
    assert "core" in btc_buys.iloc[0]["reason"]


def test_the_drawdown_kill_still_closes_the_core(settings):
    """The kill is a property of the account. A core position must not be exempt from it.

    The price crashes 35% while `dist_sma200` is pinned positive, so the core's own rule still says
    HOLD. The only thing that can close this position is the account-level kill, which makes the
    test unambiguous - and it is the realistic case, since a 200-day average takes months to catch
    a fast crash.
    """
    p = _panel(core_above=True, alt_breakout=False)
    is_btc = p["symbol"] == "BTCUSDT"
    late = p["bar_time"] > p["bar_time"].quantile(0.6)
    for col in ("open", "high", "low", "close"):
        p.loc[is_btc & late, col] = p.loc[is_btc & late, col] * 0.65
    assert (p.loc[is_btc, "dist_sma200"] > 0).all(), "the core's own rule must still say hold"

    s = __import__("dataclasses").replace(settings, kill_drawdown=0.05)
    r = backtest.run(p, s, flags=Flags(**CORE))
    assert r.kills >= 1, "the fixture failed to trigger the drawdown kill"
    sells = r.log[(r.log.side == "SELL") & (r.log.symbol == "BTCUSDT")]
    assert not sells.empty, "the kill fired and the core was still held"
    assert sells.iloc[0]["reason"] == "drawdown kill"


def test_core_off_is_byte_identical(settings):
    """The parity rule every flag in this repo obeys: off must change nothing at all."""
    p = _panel(core_above=True, alt_breakout=True)
    a = backtest.run(p, settings, flags=Flags())
    b = backtest.run(p, settings, flags=Flags(trend_core=0.0,
                                              trend_core_symbols="BTCUSDT,ETHUSDT"))
    pd.testing.assert_series_equal(a.equity, b.equity)
    assert len(a.trades) == len(b.trades)


def test_core_symbols_is_inert_until_the_core_is_on():
    assert signals.core_symbols(Flags(trend_core_symbols="BTCUSDT")) == []
    assert signals.core_symbols(Flags(trend_core=0.5, trend_core_symbols="btcusdt, ethusdt")) \
        == ["BTCUSDT", "ETHUSDT"]


def test_core_on_asks_each_coin_about_its_own_average():
    """Not BTC's average - the coin's own. This is what lets ETH be held while BTC is below."""
    b = pd.DataFrame({"symbol": ["BTCUSDT", "ETHUSDT"], "dist_sma200": [-0.1, 0.1],
                      "close": [1.0, 1.0]})
    on = signals.core_on(b, Flags(trend_core=0.5, trend_core_symbols="BTCUSDT,ETHUSDT"))
    assert list(on) == [False, True]
