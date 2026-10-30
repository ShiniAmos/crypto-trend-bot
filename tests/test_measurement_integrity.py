"""Step 2: the measurement must mean what it says.

Three properties that, if they silently broke, would make every number in RESEARCH_LOG.md an
overstatement rather than a result. None of them is visible in a CAGR, which is exactly why they
are asserted rather than assumed.

  1. BOTH engines fill at the NEXT bar's open, never at the price that triggered the decision.
     Filling at the decision price is the classic backtest lie: it buys the breakout bar's close
     when nobody could have.
  2. The 8% peak-to-trough kill fires in the backtest, not only in the live path.
  3. The daily loss halt fires in the backtest too.
"""

import numpy as np
import pandas as pd
import pytest

from app import backtest, risk
from app.flags import Flags


def _panel(n=240, core=False, crash_at=None, crash_to=0.5):
    """Two coins on 4h bars. BTC is the core candidate, AAA breaks out on a known bar."""
    t = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    rows = []
    for i, ts in enumerate(t):
        f = crash_to if (crash_at is not None and i >= crash_at) else 1.0
        btc = 100.0 * f
        alt = (50.0 + 0.4 * i) * f
        for sym, px in (("BTCUSDT", btc), ("AAAUSDT", alt)):
            rows.append({"bar_time": ts, "symbol": sym,
                         # a distinct open, so a fill at the open is distinguishable from the close
                         "open": px * 0.97, "high": px * 1.01, "low": px * 0.99, "close": px})
    p = pd.DataFrame(rows)
    p["dist_sma200"] = np.where(p["symbol"] == "BTCUSDT", 0.10, 0.05)
    p["dist_sma50"] = 0.05
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


def _fill_is_at_an_open(res, panel, symbol):
    """Every BUY price must match that bar's OPEN (plus slippage), not its close."""
    log = res.log
    buys = log[(log.side == "BUY") & (log.symbol == symbol)]
    assert len(buys), f"no {symbol} buy to check"
    opens = panel[panel.symbol == symbol].set_index("bar_time")["open"]
    closes = panel[panel.symbol == symbol].set_index("bar_time")["close"]
    for _, b in buys.iterrows():
        o, c = opens.loc[b["time"]], closes.loc[b["time"]]
        assert abs(b["price"] / o - 1) < 0.01, \
            f"{symbol} filled at {b['price']:.4f}; the bar open was {o:.4f}"
        assert abs(b["price"] - c) > abs(b["price"] - o), \
            f"{symbol} filled nearer the close ({c:.4f}) than the open ({o:.4f})"


def test_the_breakout_engine_fills_at_the_next_open(settings):
    p = _panel()
    res = backtest.run(p, settings, flags=Flags())
    _fill_is_at_an_open(res, p, "AAAUSDT")


def test_the_core_fills_at_the_next_open(settings):
    """The core queues into the same pending list, so it must obey the same rule."""
    p = _panel()
    res = backtest.run(p, settings, flags=Flags(trend_core=0.5, trend_core_symbols="BTCUSDT"))
    _fill_is_at_an_open(res, p, "BTCUSDT")


def test_a_decision_never_fills_on_its_own_bar(settings):
    """The order must be logged on a LATER bar than the one that decided it."""
    p = _panel()
    res = backtest.run(p, settings, flags=Flags())
    buys = res.log[res.log.side == "BUY"]
    first_breakout = p[(p.symbol == "AAAUSDT") & p.breakout]["bar_time"].min()
    assert buys.iloc[0]["time"] > first_breakout, \
        "filled on the same bar that produced the signal"


def test_the_drawdown_kill_fires_in_the_backtest(settings):
    """A 50% crash must trip the 8% kill and flatten the book."""
    import dataclasses
    p = _panel(crash_at=120, crash_to=0.5)
    s = dataclasses.replace(settings, kill_drawdown=0.08)
    res = backtest.run(p, s, flags=Flags())
    assert res.kills >= 1, "the 8% drawdown kill never fired in the backtest"
    events = res.log[(res.log.side == "EVENT")]
    assert any("drawdown kill" in str(r) for r in events["reason"]), events["reason"].tolist()


def test_the_daily_loss_halt_fires_in_the_backtest(settings):
    import dataclasses
    p = _panel(crash_at=120, crash_to=0.5)
    s = dataclasses.replace(settings, daily_loss_halt=0.02, kill_drawdown=0.95)
    res = backtest.run(p, s, flags=Flags())
    assert res.halts >= 1, "the daily loss halt never fired in the backtest"


def test_the_kill_also_flattens_a_core_position(settings):
    """The brakes are account-level: the core is not exempt from them."""
    import dataclasses
    p = _panel(crash_at=120, crash_to=0.5)
    s = dataclasses.replace(settings, kill_drawdown=0.08)
    res = backtest.run(p, s, flags=Flags(trend_core=0.5, trend_core_symbols="BTCUSDT"))
    assert res.kills >= 1
    sells = res.log[(res.log.side == "SELL") & (res.log.symbol == "BTCUSDT")]
    assert len(sells), "the core was never closed by the drawdown kill"
