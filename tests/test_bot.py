import json

import numpy as np
import pandas as pd
import pytest
import requests

from app import backtest, broker, data, features, main, notify, risk, universe
from app.portfolio import Store
from conftest import daily_from_4h, make_bars


# ---------------------------------------------------------------- commands (section 5.1)

@pytest.mark.parametrize("text,expected", [
    ("TRADE ETH", ("TRADE", "ETHUSDT", None)),
    ("trade eth 200 usdt", ("TRADE", "ETHUSDT", 200.0)),
    ("TRADE ETHUSDT 150", ("TRADE", "ETHUSDT", 150.0)),
    ("SKIP", ("SKIP", None, None)),
    ("FLAT ETH", ("FLAT", "ETHUSDT", None)),
    ("FLAT ALL", ("FLAT", "ALL", None)),
    ("FORCE SOL", ("FORCE", "SOLUSDT", None)),
    ("STATUS", ("STATUS", None, None)),
])
def test_parse_valid(text, expected):
    c = notify.parse(text)
    assert (c.verb, c.symbol, c.amount_usdt) == expected


@pytest.mark.parametrize("text", ["", "BUY ETH", "TRADE", "FLAT", "SKIP ETH", "TRADE ETH lots", "FLAT ETH 5"])
def test_parse_invalid(text):
    assert notify.parse(text) is None


# ---------------------------------------------------------------- risk (phase 6)

def test_position_size_formula_and_cap():
    # 1% of 10,000 = 100 risk; stop 5 below entry -> 20 units, value 2,000 = 20% (under the 30% cap)
    assert risk.position_size(10_000, 0.01, 100, 95, 0.30) == pytest.approx(20)
    # a tight stop would ask for 100 units (10,000) -> capped at 30% = 30 units
    assert risk.position_size(10_000, 0.01, 100, 99, 0.30) == pytest.approx(30)
    assert risk.position_size(10_000, 0.01, 100, 101, 0.30) == 0  # stop above entry


def test_exchange_filters_round_down_and_min_notional():
    f = risk.SymbolFilters(step_size=0.001, min_qty=0.001, tick_size=0.01, min_notional=5)
    assert f.round_qty(1.23456) == 1.234
    assert f.round_price(123.456) == 123.45
    q, why = f.check(0.01, 100)  # 1 USDT < 5
    assert q == 0 and "MIN_NOTIONAL" in why
    q, why = f.check(0.0555, 100)
    assert q == 0.055 and why is None


def test_daily_halt_and_drawdown_kill(settings):
    st = risk.RiskState(peak_equity=10_000, day="d1", day_start_equity=10_000)
    assert risk.update_limits(st, 9_750, "d1", settings) == []
    ev = risk.update_limits(st, 9_690, "d1", settings)
    assert st.halted_today and any("daily" in e for e in ev)
    risk.update_limits(st, 9_690, "d2", settings)
    assert not st.halted_today  # new day resets the daily cap
    risk.update_limits(st, 9_150, "d2", settings)
    assert st.killed
    assert risk.can_enter(st, 0, 3) == (False, "drawdown kill is active")


# ---------------------------------------------------------------- features: no lookahead

def test_features_do_not_change_when_future_bars_arrive():
    b4 = make_bars(3000, seed=1)
    d1 = daily_from_4h(b4)
    full = features.symbol_features(b4, d1)
    cut = 2000
    b4c = b4.iloc[:cut]
    part = features.symbol_features(b4c, d1[d1["close_time"] <= b4c["close_time"].iloc[-1]])
    cols = ["ret_7d", "don_high", "don_low", "breakout", "atr_pct", "vol_30d", "sma200_d", "dist_sma50"]
    pd.testing.assert_frame_equal(full[cols].iloc[:cut].reset_index(drop=True),
                                  part[cols].reset_index(drop=True))


def test_unfinished_candle_is_dropped():
    b = make_bars(5)
    last_close = int(b["close_time"].iloc[-1])
    assert len(data.closed_only(b, at_ms=last_close)) == 4


# ---------------------------------------------------------------- backtest (phase 5)

def _panel(n=4000, drifts=(0.0004, 0.0008, -0.0003, 0.0002)):
    frames = []
    for i, (sym, dr) in enumerate(zip(["BTCUSDT", "AAAUSDT", "BBBUSDT", "CCCUSDT"], drifts)):
        b4 = make_bars(n, drift=dr, seed=10 + i)
        f = features.symbol_features(b4, daily_from_4h(b4))
        f["symbol"] = sym
        frames.append(f)
    return features.add_cross_section(pd.concat(frames, ignore_index=True), min_age_bars=540)


def test_backtest_log_reconstructs_equity(settings):
    panel = _panel()
    res = backtest.run(panel, settings)
    assert len(res.trades) > 0
    fills = res.log[res.log.side.isin(["BUY", "SELL"])]
    cash = settings.start_equity
    for r in fills.itertuples():
        cash += (-1 if r.side == "BUY" else 1) * r.qty * r.price - r.fee
    open_value = res.equity.iloc[-1] - cash
    # anything left must be the open positions' value; with none open, cash == equity
    buys = fills[fills.side == "BUY"].groupby("symbol").size()
    sells = fills[fills.side == "SELL"].groupby("symbol").size()
    still_open = (buys.sub(sells, fill_value=0) > 0).sum()
    if still_open == 0:
        assert open_value == pytest.approx(0, abs=1e-6)
    assert (res.trades.exit_time > res.trades.entry_time).all()


def test_backtest_fills_on_next_bar_open(settings):
    panel = _panel()
    res = backtest.run(panel, settings)
    buys = res.log[res.log.side == "BUY"].iloc[0]
    row = panel[(panel.symbol == buys.symbol) & (panel.bar_time == buys.time)].iloc[0]
    assert buys.price == pytest.approx(row.open * (1 + settings.slippage))
    prev = panel[(panel.symbol == buys.symbol) & (panel.bar_time < buys.time)].iloc[-1]
    assert prev.breakout and prev.regime_on  # the signal was on the previous, closed bar


def test_delisted_mid_hold_is_closed_at_last_price(settings):
    panel = _panel()
    res = backtest.run(panel, settings)
    first = res.trades.iloc[0]
    # remove every bar of that coin after its entry: it vanished while held
    cut = panel[~((panel.symbol == first.symbol) & (panel.bar_time > first.entry_time))]
    res2 = backtest.run(cut, settings)
    t = res2.trades[res2.trades.symbol == first.symbol].iloc[0]
    assert "no data" in t.exit_reason
    assert np.isfinite(res2.equity).all()


def test_gap_through_stop_exits_at_next_open_not_at_stop(settings):
    panel = _panel()
    res = backtest.run(panel, settings)
    tr = res.trades[res.trades.exit_reason == "broke the 20-bar low"]
    if tr.empty:
        pytest.skip("no stop exits in synthetic data")
    t = tr.iloc[0]
    row = panel[(panel.symbol == t.symbol) & (panel.bar_time == t.exit_time)].iloc[0]
    assert t.exit_price == pytest.approx(row.open * (1 - settings.slippage))


# ---------------------------------------------------------------- state store

def test_fill_recorded_once_and_partial_sell(settings):
    st = Store(settings.db_path)
    fill = {"time": "t", "symbol": "ETHUSDT", "qty": 2.0, "price": 100.0, "fee": 0.2, "client_order_id": "x1"}
    assert st.record_buy(fill, 90, "test")
    assert not st.record_buy(fill, 90, "test")  # a retry with the same id is ignored
    assert st.positions()["ETHUSDT"]["qty"] == 2.0
    st.record_sell({**fill, "qty": 0.5, "price": 110, "client_order_id": "x2"}, "partial")
    assert st.positions()["ETHUSDT"]["qty"] == pytest.approx(1.5)
    st.record_sell({**fill, "qty": 1.5, "price": 110, "client_order_id": "x3"}, "rest")
    assert "ETHUSDT" not in st.positions()


# ---------------------------------------------------------------- approval gate (phase 8)

@pytest.fixture
def gated_bot(settings, monkeypatch):
    uni = {"date": "2026-09-21", "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
           "coins": [{"symbol": s, "cg_rank": i, "live_eligible": True}
                     for i, s in enumerate(["BTCUSDT", "ETHUSDT", "SOLUSDT"], 1)], "dropped": []}
    settings.universe_path.write_text(json.dumps(uni))
    monkeypatch.setattr(universe, "tradable_status", lambda s: True)
    monkeypatch.setattr(data, "price", lambda s: 100.0)
    monkeypatch.setattr(broker.PaperBroker, "filters", lambda self, s: risk.SymbolFilters())
    bot = main.Bot(settings)
    bot.approval = True
    row = {"symbol": "ETHUSDT", "rank": 1, "score": 90, "reason": "breakout", "stop_level": 95.0, "close": 100.0}
    bot.store.add_signal("2026-09-21 12:00:00+00:00", row, expires_at=4_000_000_000)
    bot.store.put("last_bar", "2026-09-21 12:00:00+00:00")
    return bot


def test_trade_on_candidate_places_one_order(gated_bot):
    reply = gated_bot.handle(notify.parse("TRADE ETH"))
    assert reply.startswith("OK")
    assert len([f for f in gated_bot.store.fills() if f["side"] == "BUY"]) == 1
    # the same command again must not buy twice
    assert gated_bot.handle(notify.parse("TRADE ETH")).startswith("REJECTED")
    assert len(gated_bot.store.fills()) == 1


def test_trade_on_non_candidate_or_unlisted_is_rejected(gated_bot):
    assert gated_bot.handle(notify.parse("TRADE SOL")).startswith("REJECTED")
    assert "authenticated" in gated_bot.handle(notify.parse("TRADE DOGE"))
    assert gated_bot.store.fills() == []


def test_silence_and_expiry_produce_no_orders(gated_bot):
    gated_bot.handle_commands()  # nothing queued
    assert gated_bot.store.fills() == []
    assert gated_bot.store.open_signals(now=5_000_000_000) == {}  # expired
    assert gated_bot.handle(notify.parse("TRADE ETH")).startswith("REJECTED")


def test_kill_switch_blocks_trade(gated_bot):
    gated_bot.s.kill_switch_path.write_text("stop")
    assert "kill switch" in gated_bot.handle(notify.parse("TRADE ETH"))


def test_trade_with_amount_and_flat(gated_bot):
    assert gated_bot.handle(notify.parse("TRADE ETH 200 USDT")).startswith("OK")
    pos = gated_bot.store.positions()["ETHUSDT"]
    assert pos["qty"] * pos["entry_price"] == pytest.approx(200, rel=0.01)
    assert gated_bot.handle(notify.parse("FLAT ALL")).startswith("OK")
    assert gated_bot.store.positions() == {}


# ---------------------------------------------------------------- Binance client (phase 7)

class FakeResp:
    def __init__(self, code, body, headers=None):
        self.status_code, self._body, self.headers = code, body, headers or {}

    def json(self):
        return self._body


def _binance(settings, monkeypatch):
    settings.mode, settings.binance_key, settings.binance_secret = "demo", "k", "s"
    monkeypatch.setattr(broker.BinanceBroker, "sync_time", lambda self: None)
    b = broker.BinanceBroker(settings)
    b._filters["ETHUSDT"] = risk.SymbolFilters(0.0001, 0.0001, 0.01, 5)
    monkeypatch.setattr(broker.time, "sleep", lambda s: None)
    return b


def test_429_is_retried(settings, monkeypatch):
    b = _binance(settings, monkeypatch)
    calls = iter([FakeResp(429, {}, {"Retry-After": "0"}), FakeResp(200, {"balances": []})])
    monkeypatch.setattr(b.http, "request", lambda *a, **k: next(calls))
    assert b.get_account() == {"balances": []}


def test_lost_response_does_not_double_buy(settings, monkeypatch):
    b = _binance(settings, monkeypatch)
    placed = []
    order = {"executedQty": "1", "cummulativeQuoteQty": "100", "status": "FILLED", "fills": []}

    def req(method, url, timeout):
        if method == "GET":
            return FakeResp(200, order) if placed else FakeResp(400, {"code": -2013, "msg": "no order"})
        placed.append(url)
        raise requests.ConnectionError("response lost")

    monkeypatch.setattr(b.http, "request", req)
    monkeypatch.setattr(data, "get", lambda *a, **k: {"askPrice": "100", "bidPrice": "100"})
    fill = b.buy("ETHUSDT", 1, "cid1")
    assert len(placed) == 1 and fill["qty"] == 1


# ---------------------------------------------------------------- small wallets

def test_small_wallet_raises_tiny_orders_to_the_exchange_minimum():
    # $100 account, 1% risk = $1; a stop 12% below a $1,600 coin asks for $8.33 -> above the $5 floor
    qty, note = risk.size_with_minimum(100, 0.01, 1600, 1408, 0.30, 5)
    assert qty * 1600 == pytest.approx(8.33, abs=0.01) and note is None
    # a wider stop asks for $2.50: below the floor, but buying $5 risks only 2% -> allowed, with a note
    qty, note = risk.size_with_minimum(100, 0.01, 100, 60, 0.30, 5)
    assert qty * 100 == pytest.approx(5) and "minimum order" in note
    # a very wide stop would need far more risk than allowed -> skipped, with the reason
    qty, note = risk.size_with_minimum(20, 0.01, 100, 20, 0.30, 5)
    assert qty == 0 and "would risk" in note


def test_normal_wallet_sizing_is_unchanged():
    qty, note = risk.size_with_minimum(10_000, 0.01, 100, 95, 0.30, 5)
    assert qty == pytest.approx(risk.position_size(10_000, 0.01, 100, 95, 0.30)) and note is None
