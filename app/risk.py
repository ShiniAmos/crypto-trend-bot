"""Phase 6: position sizing, exchange filters and portfolio-level limits."""

import math
from dataclasses import dataclass


def position_size(equity, risk_pct, entry, stop, max_weight, scale=1.0):
    """Units to buy: (equity x risk_pct) / stop_distance, capped at max_weight of equity.

    `scale` is the optional volatility-target multiplier (P2-J). It is applied to the risk-based
    size only; the max_weight cap is enforced after it, so the multiplier can never be used to
    exceed a portfolio limit.
    """
    dist = entry - stop
    if entry <= 0 or dist <= 0:
        return 0.0
    qty = equity * risk_pct / dist * scale
    return min(qty, equity * max_weight / entry)


def size_with_minimum(equity, risk_pct, entry, stop, max_weight, min_notional,
                      max_risk_multiple=2.0, scale=1.0):
    """Position size for small wallets.

    On a small account the 1%-risk size can land below the exchange's minimum order. Rather than
    silently skipping every trade, buy the minimum when doing so still keeps the risk within
    `max_risk_multiple` times the intended risk; otherwise skip and say why.
    Returns (qty, note): note is None for a normal size, a string when adjusted or skipped.
    """
    qty = position_size(equity, risk_pct, entry, stop, max_weight, scale)
    if qty <= 0:
        return 0.0, "price is already at or below the stop"
    if qty * entry >= min_notional:
        return qty, None
    need = min_notional / entry
    risk_then = need * (entry - stop)
    if risk_then <= equity * risk_pct * max_risk_multiple:
        return need, (f"raised to the {min_notional:g} USDT minimum order "
                      f"(risks {risk_then / equity:.1%} of the account, not {risk_pct:.1%})")
    return 0.0, (f"skipped: the {min_notional:g} USDT minimum order would risk "
                 f"{risk_then / equity:.1%} of the account")


@dataclass
class SymbolFilters:
    step_size: float = 0.0
    min_qty: float = 0.0
    tick_size: float = 0.0
    min_notional: float = 5.0

    @classmethod
    def from_exchange(cls, symbol_info):
        f = {x["filterType"]: x for x in symbol_info["filters"]}
        lot = f.get("LOT_SIZE", {})
        price = f.get("PRICE_FILTER", {})
        notional = f.get("NOTIONAL") or f.get("MIN_NOTIONAL") or {}
        return cls(step_size=float(lot.get("stepSize", 0)), min_qty=float(lot.get("minQty", 0)),
                   tick_size=float(price.get("tickSize", 0)),
                   min_notional=float(notional.get("minNotional", 5)))

    def round_qty(self, qty):
        return _floor_to(qty, self.step_size)

    def round_price(self, price):
        return _floor_to(price, self.tick_size)

    def check(self, qty, price):
        """Return (qty, None) if the order is valid after rounding, else (0, reason)."""
        q = self.round_qty(qty)
        if q <= 0 or q < self.min_qty:
            return 0.0, f"quantity below LOT_SIZE minimum {self.min_qty}"
        if q * price < self.min_notional:
            return 0.0, f"order value {q * price:.2f} below MIN_NOTIONAL {self.min_notional}"
        return q, None


def _floor_to(x, step):
    if not step:
        return x
    decimals = max(0, -int(math.floor(math.log10(step))))
    return round(math.floor(x / step + 1e-9) * step, decimals)


@dataclass
class RiskState:
    peak_equity: float
    day: str
    day_start_equity: float
    killed: bool = False
    halted_today: bool = False
    killed_day: str = ""


def update_limits(state: RiskState, equity, day, settings):
    """Advance the daily and drawdown limits. Returns a list of events to log.

    THE RE-ARM IS FIRST, and it is here rather than in the caller.

    backtest.run has re-armed the 8% kill after a 7-day pause since it was written, resuming from
    a fresh peak. The live path never did: `update_limits` only ever set `killed = True` and
    nothing in app/ set it back, so the first kill was permanent. That is not a small divergence.
    The baseline backtest fires this kill **9 times** over the full sample and **4 times** in the
    held-out period, each time pausing and carrying on - so the published +21.55% and +12.45%
    describe a book that survives nine kills, while a live book got one. The numbers were
    unreachable live by construction, whatever the edge.

    It has already cost a book. The demo account's peak was 5,006.59, it closed at 4,603.98 on
    1 Nov - two USDT past the 8% line, which the 118 USDT of churn fees is what paid for - and it
    has been unable to open a position since, with a green heartbeat and no alert.

    Re-arming before the day roll and before the peak is raised is what reproduces backtest.run's
    ordering exactly, so moving the rule in here changes no backtested number.
    """
    events = []
    cooldown = getattr(settings, "kill_cooldown_days", 0) or 0
    if state.killed and cooldown and state.killed_day:
        import pandas as _pd
        if (_pd.Timestamp(day) - _pd.Timestamp(state.killed_day)).days >= cooldown:
            state.killed, state.peak_equity = False, equity
            events.append(f"kill re-armed after {cooldown}-day pause")
    if day != state.day:
        state.day, state.day_start_equity, state.halted_today = day, equity, False
    state.peak_equity = max(state.peak_equity, equity)
    if not state.halted_today and equity <= state.day_start_equity * (1 - settings.daily_loss_halt):
        state.halted_today = True
        events.append("daily loss cap hit: no new entries until tomorrow (UTC)")
    if not state.killed and equity <= state.peak_equity * (1 - settings.kill_drawdown):
        state.killed, state.killed_day = True, day
        events.append(f"{settings.kill_drawdown:.0%} drawdown kill: flatten everything and stop")
    return events


def can_enter(state: RiskState, n_open, max_names, kill_file_exists=False):
    if kill_file_exists:
        return False, "kill switch file is present"
    if state.killed:
        return False, "drawdown kill is active"
    if state.halted_today:
        return False, "daily loss cap hit"
    if n_open >= max_names:
        return False, f"already holding the maximum of {max_names} names"
    return True, None
