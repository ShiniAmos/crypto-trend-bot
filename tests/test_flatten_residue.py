"""`flatten-residue` sells the untracked balances that block safe mode, and nothing else.

WHY IT EXISTS, 5-6 Nov 2026.

The core churn left rounding residue on the demo account: 51 buy/sell round trips per symbol, each
sell floored to the lot step, leaving a crumb behind every time.

    BTC  0.000126    about 10.16 USDT   ->  9.67 after rounding, over the 5 USDT minimum: SELLABLE
    ETH  0.0020608   about  5.00 USDT   ->  4.85 after rounding, under it: STRANDED FOR EVER

Reconciliation correctly ignores the ETH and correctly blocks on the BTC - a sellable 10 USDT
balance the database knows nothing about is a lost trade until proven otherwise. But there was no
way to clear it: `FLAT` only closes positions the database knows about, and this is by definition
one it does not. The demo book sat blocked.

The two properties worth pinning:
  1. it sells only what reconcile ITSELF marked sellable, so there is one sellability rule
  2. it clears safe mode only if the whole reconciliation then passes - not just the part it
     touched. A command that clears safe mode because it fixed one of two problems is worse than
     one that does nothing.
"""

import pytest

from app import data as app_data, reconcile, risk

PRICES = {"BTCUSDT": 80_555.0, "ETHUSDT": 2_426.0, "ZZZUSDT": 10.0}
FILTERS = {
    "BTCUSDT": risk.SymbolFilters(step_size=1e-05, min_qty=1e-05, min_notional=5.0),
    "ETHUSDT": risk.SymbolFilters(step_size=1e-04, min_qty=1e-04, min_notional=5.0),
}


class Exchange:
    """Holds what it is told to, applies the real filters, and records the sells it is asked for."""

    def __init__(self, balances):
        self._b = dict(balances)
        self.sells = []

    def balances(self):
        return self._b

    def open_orders(self, symbol=None):
        return []

    def filters(self, symbol):
        return FILTERS.get(symbol, risk.SymbolFilters())

    def cash(self):
        return self._b.get("USDT", 0.0)

    def sell(self, symbol, qty, cid, ref_price=None):
        self.sells.append((symbol, qty))
        self._b.pop(symbol[:-4], None)          # the exchange no longer holds it
        return {"price": PRICES[symbol], "qty": qty, "fee": 0.0, "client_order_id": cid}


class Demo:
    mode = "demo"


DEMO = Demo()


@pytest.fixture
def store(settings, monkeypatch):
    from app.portfolio import Store
    monkeypatch.setattr(settings, "mode", "demo")
    monkeypatch.setattr(app_data, "price", lambda s: PRICES[s])
    return Store(settings.db_path)


def test_the_report_marks_only_the_sellable_balance(store):
    """The morning of 5 Nov: the BTC is actionable, the ETH is not."""
    brk = Exchange({"USDT": 4603.0, "BTC": 0.000126, "ETH": 0.0020608})
    rep = reconcile.check(DEMO, brk, store)

    assert [x["asset"] for x in rep.sellable] == ["BTC"], (
        "the stranded ETH must not be offered for sale - no order for it can ever be accepted")
    assert rep.sellable[0]["value"] == pytest.approx(0.000126 * PRICES["BTCUSDT"])
    assert any("ETH" in n for n in rep.notes), "the ETH must still be reported as dust"


def test_selling_the_residue_makes_reconciliation_pass(store):
    """The sequence the command runs: check, sell what was marked, re-check."""
    brk = Exchange({"USDT": 4603.0, "BTC": 0.000126, "ETH": 0.0020608})
    rep = reconcile.check(DEMO, brk, store)
    assert not rep.ok

    for x in rep.sellable:
        sym = x["asset"] + "USDT"
        brk.sell(sym, brk.filters(sym).round_qty(x["qty"]), "flat-test")

    assert brk.sells == [("BTCUSDT", pytest.approx(0.00012))], (
        "the quantity must be floored to the lot step or the exchange rejects the order")
    assert reconcile.check(DEMO, brk, store).ok, "clearing the BTC did not clear the block"


def test_safe_mode_is_not_cleared_while_anything_else_disagrees(store):
    """A second problem this command CANNOT fix must keep safe mode on.

    The other difference here is a position the database believes in that the exchange does not
    hold - a missing coin, not a spare one. Selling residue cannot resolve that, so the command
    must leave safe mode in place rather than report success for the half it did fix.
    """
    store.record_buy({"time": "t", "symbol": "AAAUSDT", "qty": 3.0, "price": 100.0, "fee": 0.1,
                      "client_order_id": "cid-missing"}, stop=90.0, reason="test")
    brk = Exchange({"USDT": 4603.0, "BTC": 0.000126})          # no AAA on the exchange at all
    rep = reconcile.check(DEMO, brk, store)
    reconcile.enter_safe_mode(store, rep, lambda *a: None)
    assert reconcile.safe_mode(store)
    assert any("AAA" in d for d in rep.differences), "the missing position was not detected"

    for x in rep.sellable:
        sym = x["asset"] + "USDT"
        brk.sell(sym, brk.filters(sym).round_qty(x["qty"]), "flat-test")
    assert brk.sells == [("BTCUSDT", pytest.approx(0.00012))], "it sold something else as well"

    again = reconcile.check(DEMO, brk, store)
    assert not again.ok, "a position missing from the exchange was waved through"
    # the command only clears on `again.ok`, so safe mode must still be set
    assert reconcile.safe_mode(store), (
        "safe mode would have been cleared while a 300 USDT position was still unaccounted for")


def test_nothing_to_do_when_the_books_agree(store):
    """With no untracked balance the command must find nothing to sell."""
    rep = reconcile.check(DEMO, Exchange({"USDT": 4603.0}), store)
    assert rep.ok and rep.sellable == []


def test_accepting_residue_clears_the_block_without_an_order(store):
    """The route that does not place a trade.

    The ETH half of the 5 Nov residue could never be sold - 4.85 USDT after rounding, under the
    5 USDT minimum - so selling is not always available, and when it is not, accepting is the only
    way a blocked book ever trades again.
    """
    brk = Exchange({"USDT": 4603.0, "BTC": 0.000126})
    rep = reconcile.check(DEMO, brk, store)
    assert not rep.ok

    reconcile.accept_residue(store, [{"asset": "BTC", "qty": 0.000126}])
    again = reconcile.check(DEMO, brk, store)
    assert again.ok, f"accepting the residue did not clear the block: {again.differences}"
    assert any("accepted" in n for n in again.notes), "the acceptance was applied silently"
    assert brk.sells == [], "accepting must not place an order"


def test_acceptance_subtracts_rather_than_whitelisting(store):
    """A genuine discrepancy ON TOP of accepted residue must still block.

    This is what separates accepting from hiding. If acceptance whitelisted the asset, a real lost
    trade in the same coin would go unreported for ever.
    """
    reconcile.accept_residue(store, [{"asset": "BTC", "qty": 0.000126}])
    rep = reconcile.check(DEMO, Exchange({"USDT": 4603.0, "BTC": 0.5 + 0.000126}), store)
    assert not rep.ok, "a 0.5 BTC phantom was hidden by a 0.000126 BTC acceptance"
    assert any("BTC" in d for d in rep.differences)
    assert any("0.5" in d for d in rep.differences), (
        "the difference should report the 0.5 on top, not the gross balance")


def test_accepting_twice_is_idempotent(store):
    """Running it again must not double the accepted quantity."""
    reconcile.accept_residue(store, [{"asset": "BTC", "qty": 0.000126}])
    second = reconcile.accept_residue(store, [{"asset": "BTC", "qty": 0.000126}])
    assert second == {"BTC": 0.000126}
    assert reconcile.residue_baseline(store) == {"BTC": 0.000126}
