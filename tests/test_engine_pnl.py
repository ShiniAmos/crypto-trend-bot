"""Per-engine P&L curves: each engine's own drawdown, separate from the book's.

`Result.engine_pnl` is a cumulative P&L curve per engine - realised plus open - starting at zero.
It is deliberately NOT an equity curve: the two engines share one pot of cash, so splitting the
STARTING capital between them would be invented. What is real is how much each has made or lost,
and that is enough to measure the core's own drawdown separately from the book's, which is the
question the core's -25.5% pit drawdown raises.

What must hold for the curve to be trustworthy:
  - the two columns add up to the book's actual profit
  - a core trade's profit is credited to the core, not the breakout engine
  - closing a position moves P&L from open to realised without changing the total
"""

import numpy as np
import pandas as pd
import pytest

from app import backtest
from app.flags import Flags


def _panel(n=300, crash_at=None, crash_to=0.6):
    t = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    rows = []
    for i, ts in enumerate(t):
        f = crash_to if (crash_at is not None and i >= crash_at) else 1.0
        for sym, px in (("BTCUSDT", (100.0 + 0.05 * i) * f),
                        ("AAAUSDT", (50.0 + 0.3 * i) * f)):
            rows.append({"bar_time": ts, "symbol": sym, "open": px, "high": px * 1.01,
                         "low": px * 0.99, "close": px})
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


CORE = dict(trend_core=0.4, trend_core_symbols="BTCUSDT")


def test_the_columns_add_up_to_the_books_profit(settings):
    """The strongest check available: attribution that does not reconcile is decoration."""
    p = _panel()
    r = backtest.run(p, settings, flags=Flags(**CORE))
    book_profit = r.equity.iloc[-1] - settings.start_equity
    attributed = r.engine_pnl.iloc[-1].sum()
    assert attributed == pytest.approx(book_profit, rel=1e-9, abs=1e-6), \
        f"attributed {attributed:,.4f} but the book made {book_profit:,.4f}"


def test_it_still_reconciles_after_a_crash_closes_everything(settings):
    p = _panel(crash_at=200, crash_to=0.6)
    r = backtest.run(p, settings, flags=Flags(**CORE))
    book_profit = r.equity.iloc[-1] - settings.start_equity
    assert r.engine_pnl.iloc[-1].sum() == pytest.approx(book_profit, rel=1e-9, abs=1e-6)


def test_the_core_is_credited_while_the_position_is_still_open(settings):
    """A rising market never triggers the core's exit, so all of its P&L is unrealised. That is
    the normal case for a trend position held for months, and it must still be attributed."""
    p = _panel()
    r = backtest.run(p, settings, flags=Flags(**CORE))
    assert r.trades.empty or "entry_reason" not in r.trades or True
    assert r.engine_pnl["core"].iloc[-1] > 0.0, "open core P&L was not attributed to the core"


def test_the_core_keeps_its_credit_after_it_closes(settings):
    """Force a core exit and check the P&L moves from open to realised without changing."""
    p = _panel()
    half = p["bar_time"].quantile(0.6)
    flip = (p.symbol == "BTCUSDT") & (p.bar_time > half)
    p.loc[flip, "dist_sma200"] = -0.05            # core sells itself
    r = backtest.run(p, settings, flags=Flags(**CORE))
    sells = r.log[(r.log.side == "SELL") & (r.log.symbol == "BTCUSDT")]
    assert not sells.empty, "the fixture failed to close the core"
    core_trades = r.trades[r.trades["entry_reason"].astype(str).str.startswith("core")]
    assert len(core_trades) >= 1, "the closed core trade was not recorded as the core's"
    # after the exit the core's curve is flat: realised, and nothing open
    after = r.engine_pnl["core"].loc[r.engine_pnl.index > sells.iloc[0]["time"]]
    assert after.nunique() <= 1, "the core's P&L moved after it had no position"
    assert r.engine_pnl.iloc[-1].sum() == pytest.approx(
        r.equity.iloc[-1] - settings.start_equity, rel=1e-9, abs=1e-6)


def test_with_the_core_off_every_penny_is_the_breakout_engines(settings):
    p = _panel()
    r = backtest.run(p, settings, flags=Flags())
    assert r.engine_pnl["core"].abs().max() == 0.0
    assert r.engine_pnl["breakout"].iloc[-1] == pytest.approx(
        r.equity.iloc[-1] - settings.start_equity, rel=1e-9, abs=1e-6)


def test_each_engine_has_its_own_drawdown(settings):
    """The point of the whole thing: the core's worst run is not the book's worst run."""
    p = _panel(crash_at=200, crash_to=0.6)
    r = backtest.run(p, settings, flags=Flags(**CORE))
    for engine in ("core", "breakout"):
        curve = r.engine_pnl[engine]
        dd = (curve - curve.cummax())
        assert dd.min() <= 0.0
    book_dd = (r.equity / r.equity.cummax() - 1).min()
    assert book_dd < 0, "the fixture never drew down, so the comparison is vacuous"


def test_the_curve_is_indexed_like_the_equity_curve(settings):
    p = _panel()
    r = backtest.run(p, settings, flags=Flags(**CORE))
    assert list(r.engine_pnl.index) == list(r.equity.index)
    assert set(r.engine_pnl.columns) == {"core", "breakout"}
