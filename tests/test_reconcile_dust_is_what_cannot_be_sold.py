"""Dust is a balance that cannot be sold, not a balance that is small.

THE BUG THIS EXISTS FOR, 5 Nov 2026.

Restarting the fleet after the core churn fix put the demo book straight into safe mode:

    BTC: exchange holds 0.000126 that the database does not know about (about 10.15 USDT)
    ETH: exchange holds 0.0020608 that the database does not know about (about 5.00 USDT)

Both are residue from the churn - 51 buy/sell round trips per symbol, each sell rounded down to
the lot step, leaving a crumb behind every time. Reconciliation's dust line was a flat 1.00 USDT,
so both balances were over it and both blocked every new entry.

But only one of them CAN be cleared. Rounded down to its lot step the ETH is 4.85 USDT, under
Binance's 5 USDT minimum notional: no sell order for it will ever be accepted, so safe mode over
it never clears by itself and the account is bricked until a human intervenes - the same failure
as the 0.0877 ADA of 12 Oct, which is what the 1.00 USDT line was added to fix. It did not
generalise, because value was never the thing that mattered.

The BTC, at 9.67 USDT after rounding, is genuinely sellable and genuinely ought to block.

So the test of dust is the exchange's own order filters, and these tests pin that distinction with
the real numbers from the incident.
"""

import pytest

from app import data as app_data, reconcile, risk

PRICES = {"BTCUSDT": 80_555.0, "ETHUSDT": 2_426.0, "ADAUSDT": 0.25}

# the real Binance spot filters for these pairs, as of this incident
FILTERS = {
    "BTCUSDT": risk.SymbolFilters(step_size=1e-05, min_qty=1e-05, min_notional=5.0),
    "ETHUSDT": risk.SymbolFilters(step_size=1e-04, min_qty=1e-04, min_notional=5.0),
    "ADAUSDT": risk.SymbolFilters(step_size=0.1, min_qty=0.1, min_notional=5.0),
}


class Exchange:
    """Holds whatever it is told to hold, and applies the real filters for the pair."""

    def __init__(self, balances):
        self._b = balances

    def balances(self):
        return self._b

    def open_orders(self, symbol=None):
        return []

    def filters(self, symbol):
        return FILTERS.get(symbol, risk.SymbolFilters())

    def cash(self):
        return self._b.get("USDT", 0.0)


class _Demo:
    """reconcile.check reads only `mode` off settings for the live path."""
    mode = "demo"


DEMO = _Demo()


@pytest.fixture
def live(settings, monkeypatch):
    from app.portfolio import Store
    monkeypatch.setattr(settings, "mode", "demo")
    monkeypatch.setattr(app_data, "price", lambda s: PRICES[s])
    return Store(settings.db_path)


def test_the_eth_residue_that_bricked_the_demo_book_is_dust(live):
    """0.0020608 ETH, 5.00 USDT, 4.85 after rounding: unsellable, so it must not block."""
    rep = reconcile.check(DEMO, Exchange({"USDT": 4603.0, "ETH": 0.0020608}), live)
    assert rep.ok, f"unsellable residue blocked the account again: {rep.differences}"
    assert any("ETH" in n for n in rep.notes), "it must still be reported, just not fatal"


def test_the_btc_residue_alongside_it_is_not_dust(live):
    """0.000126 BTC, 10.15 USDT, 9.67 after rounding: sellable, so it is a real difference."""
    rep = reconcile.check(DEMO, Exchange({"USDT": 4603.0, "BTC": 0.000126}), live)
    assert not rep.ok, "a sellable 10 USDT phantom was waved through as dust"
    assert any("BTC" in d for d in rep.differences)


def test_both_together_is_the_morning_of_5_nov(live):
    """The two must be judged separately: the ETH noted, the BTC blocking."""
    rep = reconcile.check(
        DEMO, Exchange({"USDT": 4603.0, "BTC": 0.000126, "ETH": 0.0020608}), live)
    assert not rep.ok
    assert any("BTC" in d for d in rep.differences), "the sellable BTC must block"
    assert not any("ETH" in d for d in rep.differences), "the stranded ETH must not block"
    assert any("ETH" in n for n in rep.notes), "the stranded ETH must still be reported"


def test_the_ada_of_12_oct_is_still_dust(live):
    """The case the old flat line was written for must keep working under the new rule."""
    rep = reconcile.check(DEMO, Exchange({"USDT": 5000.0, "ADA": 0.0877}), live)
    assert rep.ok, f"the original dust case regressed: {rep.differences}"


def test_a_real_phantom_position_still_blocks(live):
    """The exemption must not become a hole: 400 ADA is a lost trade, not a rounding crumb."""
    rep = reconcile.check(DEMO, Exchange({"USDT": 5000.0, "ADA": 400.0}), live)
    assert not rep.ok and any("ADA" in d for d in rep.differences)


def test_an_unpriceable_balance_is_treated_as_real(live, monkeypatch):
    """When it cannot be valued the bot must NOT guess in its own favour."""
    monkeypatch.setattr(app_data, "price", lambda s: (_ for _ in ()).throw(RuntimeError("no pair")))
    rep = reconcile.check(DEMO, Exchange({"USDT": 5000.0, "WAT": 123.0}), live)
    assert not rep.ok, "an unvaluable untracked balance was silently ignored"
    assert any("WAT" in d for d in rep.differences)


def test_dust_just_over_the_minimum_blocks_and_just_under_does_not(live):
    """The line sits exactly at the exchange's minimum notional, not at a round dollar figure."""
    under = 4.90 / PRICES["ETHUSDT"]          # 4.90 USDT of ETH
    over = 5.10 / PRICES["ETHUSDT"]           # 5.10 USDT of ETH
    assert reconcile.check(DEMO, Exchange({"ETH": under}), live).ok
    assert not reconcile.check(DEMO, Exchange({"ETH": over}), live).ok


