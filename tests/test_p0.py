"""P0 correctness work: stop execution, paper/backtest alignment, reconciliation, skip logging.

The first test is the important one: with every flag off, the bot must behave exactly as the
audited baseline did.
"""

import json

import pandas as pd
import pytest

from app import backtest, broker, data, features, main, reconcile, risk, signals, universe
from app.flags import Flags
from conftest import daily_from_4h, make_bars


def _panel(n=4000, drifts=(0.0004, 0.0008, -0.0003, 0.0002)):
    frames = []
    for i, (sym, dr) in enumerate(zip(["BTCUSDT", "AAAUSDT", "BBBUSDT", "CCCUSDT"], drifts)):
        b4 = make_bars(n, drift=dr, seed=10 + i)
        f = features.symbol_features(b4, daily_from_4h(b4))
        f["symbol"] = sym
        frames.append(f)
    return features.add_cross_section(pd.concat(frames, ignore_index=True), min_age_bars=540)


# ---------------------------------------------------------------- flag-off parity

def test_flags_off_reproduce_the_baseline(settings):
    panel = _panel()
    base = backtest.run(panel, settings)
    flagged = backtest.run(panel, settings, flags=Flags())
    assert base.equity.equals(flagged.equity)
    assert len(base.trades) == len(flagged.trades)
    assert base.kills == flagged.kills and base.halts == flagged.halts


def test_flag_label_and_digest_are_stable():
    assert Flags().is_baseline and Flags().label == "baseline"
    assert Flags(stop_exec=True).enabled == {"stop_exec": True}
    assert Flags().digest() != Flags(stop_exec=True).digest()
    assert Flags().digest() == Flags().digest()


# ---------------------------------------------------------------- P0-A stop execution

def test_stop_exec_changes_exits_and_is_conservative(settings):
    panel = _panel()
    on = backtest.run(panel, settings, flags=Flags(stop_exec=True))
    stops = on.trades[on.trades.exit_reason == "stop hit intrabar"]
    assert len(stops) > 0, "the stop should fire somewhere in 4000 synthetic bars"
    for t in stops.head(10).itertuples():
        bar = panel[(panel.symbol == t.symbol) & (panel.bar_time == t.exit_time)].iloc[0]
        # filled at min(open, stop) minus slippage: never better than the worse of the two
        assert t.exit_price <= max(bar.open, bar.low) * (1 + 1e-9)
        assert bar.low <= t.exit_price / (1 - settings.slippage) * (1 + 1e-9)


def test_stop_never_moves_down(settings):
    from app.portfolio import Store
    st = Store(settings.data_dir / "s.sqlite")
    fill = {"time": "t", "symbol": "AAAUSDT", "qty": 1.0, "price": 100.0, "fee": 0.1,
            "client_order_id": "cid1"}
    st.record_buy(fill, stop=90.0, reason="test")
    st.update_stop("AAAUSDT", 95.0)
    assert st.positions()["AAAUSDT"]["stop"] == 95.0
    st.update_stop("AAAUSDT", 80.0)                      # a lower stop must be ignored
    assert st.positions()["AAAUSDT"]["stop"] == 95.0


def test_gap_below_the_stop_fills_at_the_open(settings):
    """A bar that opens below the stop must fill at the open, not at the (better) stop price."""
    panel = _panel()
    res = backtest.run(panel, settings, flags=Flags(stop_exec=True))
    gaps = []
    for t in res.trades[res.trades.exit_reason == "stop hit intrabar"].itertuples():
        bar = panel[(panel.symbol == t.symbol) & (panel.bar_time == t.exit_time)].iloc[0]
        prev = panel[(panel.symbol == t.symbol) & (panel.bar_time < t.exit_time)].iloc[-1]
        if bar.open < prev.stop_level:
            gaps.append(t.exit_price <= bar.open * (1 - settings.slippage) * (1 + 1e-9))
    if gaps:
        assert all(gaps)


# ---------------------------------------------------------------- P0-B paper == backtest

def test_paper_fill_matches_the_backtest_fill(settings):
    """Same signal, same candle, same price model: the paper broker and the backtest agree."""
    from app.portfolio import Store
    st = Store(settings.db_path)
    pb = broker.PaperBroker(settings, st)
    open_px = 123.456
    fill = pb.buy("AAAUSDT", 1.0, "cid-paper", ref_price=open_px)
    backtest_price = open_px * (1 + settings.slippage)
    assert fill["price"] == pytest.approx(backtest_price)
    assert fill["fee"] == pytest.approx(fill["qty"] * fill["price"] * settings.fee_rate)
    sell = pb.sell("AAAUSDT", 1.0, "cid-paper-s", ref_price=open_px)
    assert sell["price"] == pytest.approx(open_px * (1 - settings.slippage))


def _bot_with_bar(settings, monkeypatch, close=100.0, open_=99.0, low=98.0):
    uni = {"date": "2026-09-25", "symbols": ["BTCUSDT", "AAAUSDT"],
           "coins": [{"symbol": s, "cg_rank": i, "live_eligible": True}
                     for i, s in enumerate(["BTCUSDT", "AAAUSDT"], 1)], "dropped": []}
    settings.universe_path.write_text(json.dumps(uni))
    monkeypatch.setattr(universe, "tradable_status", lambda s: True)
    monkeypatch.setattr(data, "price", lambda s: close)
    monkeypatch.setattr(broker.PaperBroker, "filters", lambda self, s: risk.SymbolFilters())
    bot = main.Bot(settings)
    latest = pd.DataFrame([{"symbol": "AAAUSDT", "open": open_, "low": low, "close": close,
                            "stop_level": 95.0}]).set_index("symbol")
    return bot, latest


def test_queued_order_fills_at_the_next_open(settings, monkeypatch):
    bot, latest = _bot_with_bar(settings, monkeypatch)
    bar = pd.Timestamp("2026-09-25 12:00", tz="UTC")
    bot.store.queue_order(bar, "BUY", "AAAUSDT", 2.0, 95.0, "breakout, rank 1")
    assert len(bot.store.due_orders()) == 1
    bot.fill_due_orders(latest, bar + pd.Timedelta(hours=4))
    pos = bot.store.positions()["AAAUSDT"]
    assert pos["entry_price"] == pytest.approx(99.0 * (1 + settings.slippage))
    assert bot.store.due_orders() == []                 # not filled twice


def test_queued_sell_runs_before_queued_buy(settings, monkeypatch):
    bot, latest = _bot_with_bar(settings, monkeypatch)
    bar = pd.Timestamp("2026-09-25 12:00", tz="UTC")
    bot.store.queue_order(bar, "BUY", "AAAUSDT", 1.0, 95.0, "buy")
    bot.store.queue_order(bar, "SELL", "AAAUSDT", 1.0, 95.0, "sell")
    assert [o["side"] for o in bot.store.due_orders()] == ["SELL", "BUY"]


def test_stale_queued_buy_is_dropped_but_a_stale_sell_still_runs(settings, monkeypatch):
    """Downtime must not resurrect an old entry at today's price; an exit still has to happen."""
    bot, latest = _bot_with_bar(settings, monkeypatch)
    old_bar = pd.Timestamp("2026-09-20 12:00", tz="UTC")     # long before the bar being processed
    now_bar = pd.Timestamp("2026-09-25 12:00", tz="UTC")
    bot.store.queue_order(old_bar, "BUY", "AAAUSDT", 1.0, 95.0, "breakout, rank 1")
    bot.fill_due_orders(latest, now_bar)
    assert bot.store.positions() == {}
    skipped = bot.store.skipped()
    assert len(skipped) == 1 and "stale" in skipped[0]["reason"]

    bot.store.record_buy({"time": "t", "symbol": "AAAUSDT", "qty": 1.0, "price": 100.0,
                          "fee": 0.1, "client_order_id": "cid-held"}, stop=95.0, reason="test")
    bot.store.queue_order(old_bar, "SELL", "AAAUSDT", 1.0, 95.0, "broke the 20-bar low")
    bot.fill_due_orders(latest, now_bar)
    assert bot.store.positions() == {}, "a late exit must still be taken"


def test_flat_cancels_a_queued_buy(settings, monkeypatch):
    bot, _ = _bot_with_bar(settings, monkeypatch)
    bar = pd.Timestamp("2026-09-25 12:00", tz="UTC")
    bot.store.queue_order(bar, "BUY", "AAAUSDT", 1.0, 95.0, "buy")
    from app import notify
    bot.handle(notify.parse("FLAT ALL"))
    assert bot.store.due_orders() == []


# ---------------------------------------------------------------- P0-C reconciliation

class FakeBroker:
    """Stands in for the Binance client: whatever the exchange is said to hold."""

    def __init__(self, balances=None, orders=None):
        self._b = balances or {}
        self._o = orders or []

    def balances(self):
        return self._b

    def open_orders(self, symbol=None):
        return self._o

    def filters(self, symbol):
        return risk.SymbolFilters(step_size=0.001, min_notional=5.0)

    def cash(self):
        return self._b.get("USDT", 0.0)


def _live_store(settings, qty=1.0):
    from app.portfolio import Store
    st = Store(settings.db_path)
    st.record_buy({"time": "t", "symbol": "AAAUSDT", "qty": qty, "price": 100.0, "fee": 0.1,
                   "client_order_id": "cid-live"}, stop=90.0, reason="test")
    return st


def test_reconcile_flags_a_missing_position(settings, monkeypatch):
    monkeypatch.setattr(settings, "mode", "demo")
    st = _live_store(settings)
    rep = reconcile.check(settings, FakeBroker({"USDT": 500.0}), st)
    assert not rep.ok and "short by" in rep.differences[0]


def test_reconcile_flags_an_unknown_balance_and_untracked_order(settings, monkeypatch):
    monkeypatch.setattr(settings, "mode", "demo")
    st = _live_store(settings)
    brk = FakeBroker({"USDT": 500.0, "AAA": 1.0, "ZZZ": 42.0},
                     orders=[{"symbol": "ZZZUSDT", "type": "LIMIT", "clientOrderId": "ghost"}])
    rep = reconcile.check(settings, brk, st)
    assert not rep.ok
    joined = " ".join(rep.differences)
    assert "ZZZ" in joined and "untracked open" in joined


def test_reconcile_passes_when_the_exchange_agrees(settings, monkeypatch):
    monkeypatch.setattr(settings, "mode", "demo")
    st = _live_store(settings)
    rep = reconcile.check(settings, FakeBroker({"USDT": 500.0, "AAA": 1.0}), st)
    assert rep.ok, rep.differences


def test_safe_mode_blocks_entries_but_not_exits(settings, monkeypatch):
    bot, _ = _bot_with_bar(settings, monkeypatch)
    from app import notify
    bot.store.put(reconcile.SAFE_MODE_KEY, {"since": "now", "differences": ["test"]})
    assert bot.handle(notify.parse("TRADE AAA")).startswith("REJECTED")
    assert "safe mode" in bot.handle(notify.parse("TRADE AAA"))
    # exits still work: FLAT on an empty book is rejected for the right reason, not for safe mode
    assert "safe mode" not in bot.handle(notify.parse("FLAT AAA"))


# ---------------------------------------------------------------- P0-D skipped entries

def test_skipped_entries_are_recorded_with_opportunity_cost(settings, monkeypatch):
    bot, _ = _bot_with_bar(settings, monkeypatch)
    panel = _panel(n=1200)
    bars = sorted(panel["bar_time"].unique())
    missed, now = bars[-3], bars[-1]
    now_rows = panel[panel["bar_time"] == now].set_index("symbol")

    class AlwaysCandidate:
        """One candidate on the missed bar, so the recording path is exercised."""

        @staticmethod
        def entry_candidates(bar, s):
            return bar[bar["symbol"] == "AAAUSDT"].assign(rank=1, score=99.0).head(1)

    monkeypatch.setattr(signals, "entry_candidates", AlwaysCandidate.entry_candidates)
    bot.record_skipped_entries(panel, missed, now, {}, now_rows, downtime_min=480.0)
    rows = bot.store.skipped()
    assert len(rows) == 1
    r = rows[0]
    assert r["symbol"] == "AAAUSDT" and r["downtime_minutes"] == 480.0
    assert r["expected_entry"] > 0 and r["opportunity_cost"] is not None
    assert "stale" in r["reason"]


# ---------------------------------------------------------------- autonomous live trading

def test_auto_trade_removes_the_approval_gate(settings, monkeypatch):
    """AUTO_TRADE makes live mode enter by itself; without it, live still waits for TRADE."""
    monkeypatch.setenv("MODE", "live")
    monkeypatch.setenv("LIVE_TRADING_CONFIRMED", "true")
    monkeypatch.setenv("AUTO_TRADE", "false")
    from app import config as cfg
    gated = cfg.Settings()
    assert gated.mode == "live" and not gated.auto_trade

    monkeypatch.setenv("AUTO_TRADE", "true")
    auto = cfg.Settings()
    assert auto.auto_trade
    # the gate is derived from both, and exits are never gated either way
    assert (auto.mode == "live" and not auto.auto_trade) is False


def test_paper_was_never_gated(settings):
    assert settings.mode == "paper" and not settings.auto_trade


def test_demo_endpoint_is_configurable(settings, monkeypatch):
    """A key from the Spot Test Network is rejected by Demo Trading and vice versa."""
    from app import config as cfg
    assert settings.demo_api_url == "https://demo-api.binance.com"
    monkeypatch.setenv("DEMO_API_URL", "https://testnet.binance.vision")
    assert cfg.Settings().demo_api_url == "https://testnet.binance.vision"


def test_reconcile_ignores_cash_assets(settings, monkeypatch):
    """A demo account funded in USDC and holding BNB for fees is not a reconciliation failure.

    First contact with the real Binance demo environment flagged a 5,000 USDC balance as a phantom
    position and blocked trading, which is a false positive: cash is not a position.
    """
    monkeypatch.setattr(settings, "mode", "demo")
    st = _live_store(settings)
    brk = FakeBroker({"USDT": 5000.0, "USDC": 5000.0, "BNB": 0.5, "AAA": 1.0})
    rep = reconcile.check(settings, brk, st)
    assert rep.ok, rep.differences


def test_reconcile_ignores_unsellable_dust(settings, monkeypatch):
    """Two cents of leftover ADA must not block the account forever.

    On 12 Oct 2026 the demo book was blocked from every entry - including the trend core, on the
    day it was switched on - by 0.0877 ADA worth 0.02 USDT, left behind when Binance took its fee
    in the asset bought. The balance is below the 5 USDT minimum order, so it can never be sold,
    which means a safe mode raised over it never clears by itself: the account is bricked until a
    human intervenes. Dust is reported as a note and ignored.
    """
    from app import data as app_data
    monkeypatch.setattr(settings, "mode", "demo")
    monkeypatch.setattr(app_data, "price", lambda s: {"ADAUSDT": 0.25}.get(s, 1000.0))
    st = _live_store(settings)
    brk = FakeBroker({"USDT": 5000.0, "ADA": 0.0877, "AAA": 1.0})
    rep = reconcile.check(settings, brk, st)
    assert rep.ok, f"dust blocked the account: {rep.differences}"
    assert any("ADA" in n for n in rep.notes), "dust should still be reported, just not fatal"


def test_reconcile_still_catches_a_phantom_worth_real_money(settings, monkeypatch):
    """The dust exemption must not become a hole: 40 ADA is a position, not a rounding error."""
    from app import data as app_data
    monkeypatch.setattr(settings, "mode", "demo")
    monkeypatch.setattr(app_data, "price", lambda s: {"ADAUSDT": 0.25}.get(s, 1000.0))
    st = _live_store(settings)
    brk = FakeBroker({"USDT": 5000.0, "ADA": 40.0, "AAA": 1.0})
    rep = reconcile.check(settings, brk, st)
    assert not rep.ok and any("ADA" in d for d in rep.differences)


def test_reconcile_still_catches_a_real_phantom_coin(settings, monkeypatch):
    monkeypatch.setattr(settings, "mode", "demo")
    st = _live_store(settings)
    brk = FakeBroker({"USDT": 5000.0, "USDC": 5000.0, "AAA": 1.0, "ZZZ": 42.0})
    rep = reconcile.check(settings, brk, st)
    assert not rep.ok and any("ZZZ" in d for d in rep.differences)
    assert not any("USDC" in d for d in rep.differences)


def test_fee_taken_in_the_coin_does_not_leave_a_phantom_position(settings):
    """Binance may take its fee in the base asset, so a full exit sells slightly less than was
    bought. On 8 Oct 2026 that left 0.0877 ADA - two cents, unsellable under the 5 USDT minimum -
    and the book kept believing it held a position, blocking one of three slots forever.
    """
    from app.portfolio import Store
    st = Store(settings.data_dir / "dust.sqlite")
    st.record_buy({"time": "t", "symbol": "ADAUSDT", "qty": 1310.9877, "price": 0.2739,
                   "fee": 0.3594, "client_order_id": "buy1"}, stop=0.24, reason="test")
    assert "ADAUSDT" in st.positions()
    st.record_sell({"time": "t2", "symbol": "ADAUSDT", "qty": 1310.9000, "price": 0.2491,
                    "fee": 0.3265, "client_order_id": "sell1"}, reason="exit")
    assert st.positions() == {}, "a remainder too small to trade is not a position"


def test_a_real_partial_sell_still_keeps_the_rest(settings):
    """The dust rule must not swallow a genuine partial exit."""
    from app.portfolio import Store
    st = Store(settings.data_dir / "partial.sqlite")
    st.record_buy({"time": "t", "symbol": "AAAUSDT", "qty": 10.0, "price": 100.0, "fee": 1.0,
                   "client_order_id": "b"}, stop=90.0, reason="test")
    st.record_sell({"time": "t2", "symbol": "AAAUSDT", "qty": 4.0, "price": 100.0, "fee": 0.4,
                    "client_order_id": "s"}, reason="scale out")
    pos = st.positions()
    assert "AAAUSDT" in pos and pos["AAAUSDT"]["qty"] == pytest.approx(6.0)
