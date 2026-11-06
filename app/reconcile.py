"""P0-C: compare what the database believes with what the exchange actually holds.

Run at every startup and on demand. The rule is simple and deliberately unhelpful to itself: when
the two disagree, the bot does not guess which one is right. It logs, alerts, enters safe mode and
refuses new entries until a human clears it. Exits keep working, because being unable to close a
position is worse than being unable to open one.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone

from .universe import STABLES

SAFE_MODE_KEY = "safe_mode"
DUST = 1e-8


@dataclass
class Report:
    ok: bool = True
    mode: str = "paper"
    differences: list = field(default_factory=list)
    notes: list = field(default_factory=list)      # observed, deliberately not a failure
    checked: dict = field(default_factory=dict)
    # The untracked balances the exchange would actually accept an order for, recorded by the
    # same walk that decides they are differences. `flatten-residue` consumes this rather than
    # asking the sellability question a second time: one rule, one place, which is the lesson of
    # the core churn.
    sellable: list = field(default_factory=list)   # [{"asset", "qty", "value"}]

    def text(self):
        if self.ok:
            msg = (f"Reconciliation OK ({self.mode}): {self.checked.get('positions', 0)} positions, "
                   f"{self.checked.get('stops', 0)} stops, {self.checked.get('open_orders', 0)} "
                   "open orders match the database.")
            return msg + "".join(f"\n  note: {n}" for n in self.notes)
        lines = [f"RECONCILIATION FAILED ({self.mode}). New entries are blocked until this is cleared."]
        lines += [f"  - {d}" for d in self.differences]
        lines.append("Inspect with: python -m app.main reconcile")
        return "\n".join(lines)


def _sellable(brk, asset, qty):
    """Could this untracked balance actually be sold? That is what separates dust from a lost trade.

    The question is not "is it worth much" but "can it leave the account". A balance below the
    exchange's lot step or minimum notional is stranded: no order for it will ever be accepted, so
    a safe mode raised over it can never clear by itself and blocks every entry forever.

    That is not hypothetical, it has happened twice:

        12 Oct 2026   0.0877 ADA, 0.02 USDT   blocked the demo book for days
        05 Nov 2026   0.0020608 ETH, 5.00 USDT   blocked it again

    and a flat one-dollar dust line cannot tell the second case from the real thing. Rounded down
    to ETH's 0.0001 lot step that balance is 4.85 USDT, under the 5 USDT minimum, and just as
    stranded as the ADA was - while 10.15 USDT of BTC sitting next to it WAS sellable and did
    deserve to block. Value does not decide it; the exchange's filters do.

    So this asks the exchange's own filters - the same SymbolFilters.check() every order goes
    through. One predicate, asked by everyone who needs the answer, which is the lesson of the
    core churn of 30 Oct: a rule copied into a second place is a rule that will disagree with
    itself eventually.

    Returns (sellable, value_usdt, reason):
        False  stranded, and therefore dust whatever it is worth
        True   a real discrepancy, and one that can be acted on
        None   undecidable (no USDT pair, or the exchange could not be reached), treated as a real
               difference: guessing in the bot's own favour is how a lost trade goes unnoticed.
    """
    symbol = asset + "USDT"
    try:
        from . import data as _data
        price = _data.price(symbol)
        value = qty * price
    except Exception:
        return None, None, "no USDT price is available for it"
    try:
        _q, why = brk.filters(symbol).check(qty, price)
    except Exception:
        return None, value, "the exchange filters could not be read"
    return (False, value, why) if why else (True, value, "")


def _qty_tolerance(filters, qty):
    """A position matches if it agrees to within one exchange lot step (plus a dust epsilon)."""
    step = getattr(filters, "step_size", 0) or 0
    return max(step, abs(qty) * 1e-6, DUST)


def check(settings, brk, store) -> Report:
    """Paper has no exchange, so it only checks that its own books are internally consistent."""
    positions = store.positions()
    if settings.mode == "paper":
        rep = Report(mode="paper", checked={"positions": len(positions)})
        cash = brk.cash()
        if cash is None or cash < 0:
            rep.ok = False
            rep.differences.append(f"paper cash is {cash}, which cannot be right")
        for sym, p in positions.items():
            if p["qty"] <= 0:
                rep.ok = False
                rep.differences.append(f"{sym}: stored quantity is {p['qty']}")
        return rep

    rep = Report(mode=settings.mode)
    balances = brk.balances()
    open_orders = brk.open_orders() or []
    rep.checked = {"positions": len(positions), "open_orders": len(open_orders),
                   "stops": len(store.stops())}

    for sym, p in positions.items():
        base = sym[:-4]
        held = balances.get(base, 0.0)
        tol = _qty_tolerance(brk.filters(sym), p["qty"])
        if held + tol < p["qty"]:
            rep.ok = False
            rep.differences.append(
                f"{sym}: database says {p['qty']:.8g}, exchange holds {held:.8g} "
                f"(short by {p['qty'] - held:.8g})")

    tracked = {s[:-4] for s in positions}
    for asset, qty in balances.items():
        # Cash is not a position. A demo account is funded in several stablecoins, and BNB is kept
        # for fees, so none of those are evidence that the database has lost track of a trade.
        if asset in STABLES or asset == "BNB" or qty <= DUST or asset in tracked:
            continue
        # Dust is not a lost trade, and treating it as one is worse than useless - it bricks the
        # account. What makes a balance dust is that it cannot be sold, not that it is small, so
        # the exchange's own order filters decide it rather than a dollar figure. See _sellable().
        sellable, value, why = _sellable(brk, asset, qty)
        worth = f" (about {value:,.2f} USDT)" if value is not None else ""
        if sellable is False:
            rep.notes.append(
                f"{asset}: {qty:.8g} of untracked dust{worth} cannot be sold ({why}), ignored")
            continue
        rep.ok = False
        tail = " and is large enough to sell" if sellable else f" ({why})"
        rep.differences.append(
            f"{asset}: exchange holds {qty:.8g} that the database does not know about{worth}{tail}")
        if sellable:
            rep.sellable.append({"asset": asset, "qty": qty, "value": value})

    known_cids = {s["client_order_id"] for s in store.stops().values()}
    for o in open_orders:
        if o.get("clientOrderId") not in known_cids:
            rep.ok = False
            rep.differences.append(
                f"{o.get('symbol')}: untracked open {o.get('type')} order {o.get('clientOrderId')}")

    for sym, s in store.stops().items():
        if sym not in positions:
            rep.ok = False
            rep.differences.append(f"{sym}: a stop order is recorded but no position is open")
        elif not any(o.get("clientOrderId") == s["client_order_id"] for o in open_orders):
            rep.ok = False
            rep.differences.append(
                f"{sym}: stop {s['client_order_id']} is not resting at the exchange "
                "(filled, cancelled or never placed)")
    return rep


def enter_safe_mode(store, report, log, tg=None):
    store.put(SAFE_MODE_KEY, {"since": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              "differences": report.differences})
    log("SAFE MODE: " + report.text().replace("\n", "\n  "))
    if tg is not None:
        tg.send(report.text())


def safe_mode(store):
    return store.get(SAFE_MODE_KEY)


def clear_safe_mode(store):
    store.put(SAFE_MODE_KEY, None)
