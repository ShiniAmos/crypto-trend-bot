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
# An untracked balance worth less than this is dust, not a lost trade. It matches portfolio.DUST_USDT
# and, more importantly, sits below Binance's 5 USDT minimum order: such a balance cannot be sold,
# so a safe mode raised over it would never clear and would block the account permanently.
DUST_USDT = 1.0


@dataclass
class Report:
    ok: bool = True
    mode: str = "paper"
    differences: list = field(default_factory=list)
    notes: list = field(default_factory=list)      # observed, deliberately not a failure
    checked: dict = field(default_factory=dict)

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
        # Value it at the real price when there is a pair for it, rather than guessing. An
        # earlier version multiplied the quantity by the minimum order size, which is meaningless.
        value = None
        try:
            from . import data as _data
            value = qty * _data.price(asset + "USDT")
            worth = f" (about {value:,.2f} USDT)"
        except Exception:
            worth = ""
        # Dust is not a lost trade, and treating it as one is worse than useless: on 12 Oct the
        # demo book was blocked from every entry for days by 0.0877 ADA worth two cents, left
        # behind when Binance took its fee in the asset bought. Such a balance can never be sold -
        # it is below the 5 USDT minimum order - so safe mode over it never clears by itself and
        # blocks the account forever. Below the dust line it is reported and ignored.
        if value is not None and value < DUST_USDT:
            rep.notes.append(f"{asset}: {qty:.8g} of untracked dust{worth}, ignored")
            continue
        rep.ok = False
        rep.differences.append(
            f"{asset}: exchange holds {qty:.8g} that the database does not know about{worth}")

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
