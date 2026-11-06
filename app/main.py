"""Entry point: the 24/7 loop and the command-line tools.

    python -m app.main universe [--refresh]   build today's authenticated universe
    python -m app.main backfill               download/append full kline history
    python -m app.main scan                   score all names on the last closed 4h bar
    python -m app.main backtest [--start ..]  paper-trade the whole history (Phase 5)
    python -m app.main ml                     walk-forward model vs the ranker (Phase 12)
    python -m app.main run                    the bot loop (MODE from .env: paper/demo/live)
    python -m app.main once                   process the last closed bar once, then exit
    python -m app.main cmd "TRADE ETH"        queue a command (same as sending it on Telegram)
    python -m app.main status                 equity, positions, pending signals
    python -m app.main watchdog               alert if the loop's heartbeat is stale (cron)
"""

import argparse
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from . import (backfill, broker, config, data, funding, notify, ranker, reconcile, risk, signals,
               universe)
from .flags import Flags
from .portfolio import Store

REPORTS = config.ROOT / "reports"
NETWORK_ERRORS = (requests.RequestException, data.RateLimited, ConnectionError, TimeoutError)


def log(msg):
    print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} {msg}", flush=True)


class Bot:
    def __init__(self, settings):
        self.s = settings
        self.store = Store(settings.db_path)
        self.broker = broker.make(settings, self.store)
        self.tg = notify.Telegram(settings, self.store, log)
        # Live entries wait for a TRADE command unless AUTO_TRADE is set; paper and demo always
        # trade by themselves. Exits are never gated in any mode.
        self.approval = settings.mode == "live" and not settings.auto_trade
        self.max_names = settings.live_max_names if settings.mode == "live" else settings.max_names
        # Per-account flags: DEMO_ENABLE_TREND_CORE beats ENABLE_TREND_CORE for the demo book, so
        # the fleet can run a treatment book against a control book on identical bars.
        self.flags = Flags.from_env(settings, account=getattr(settings, "account_name", None))
        # MAX_NAMES is a setting rather than a flag, so it needs the same per-account override or
        # a config that changes both (core + slots) could only ever be half-applied to one book.
        import os
        _mn = os.environ.get(f"{(getattr(settings, 'account_name', '') or '').upper()}_MAX_NAMES", "")
        if _mn.strip():
            self.max_names = int(_mn)
        self.levels = None
        # Paper queues its orders and fills them at the next bar's open, exactly as the backtest
        # does (P0-B). Demo and live send real orders when the decision is taken.
        self.queued_fills = settings.mode == "paper"

    # state ------------------------------------------------------------------------------
    def risk_state(self, equity, day):
        d = self.store.get("risk")
        if d is None:
            return risk.RiskState(peak_equity=equity, day=day, day_start_equity=equity)
        return risk.RiskState(**d)

    def equity(self, prices):
        pos = self.store.positions()
        held = sum(p["qty"] * prices.get(s, p["entry_price"]) for s, p in pos.items())
        cash = self.broker.cash()
        if self.s.mode == "live":  # the bot only ever sizes against its own budget
            cash = min(cash, max(self.s.live_budget_usdt - sum(p["cost"] for p in pos.values()), 0))
        return cash + held, cash

    # orders -----------------------------------------------------------------------------
    def buy(self, symbol, qty, stop, reason, bar_time, ref_price=None, engine="breakout"):
        cid = broker.client_order_id(bar_time, symbol, "BUY", self.s.mode[0])
        fill = self.broker.buy(symbol, qty, cid, ref_price=ref_price)
        self.store.record_buy(fill, stop, reason, engine=engine)
        # A trend-core order carries no stop - its exit is the 200-day average - so the stop is
        # only formatted when there is one. "stop {None:.6g}" is a TypeError.
        stop_txt = f"stop {stop:.6g}  " if stop else ""
        self.tg.send(f"BOUGHT {symbol} qty {fill['qty']:.6g} @ {fill['price']:.6g}  "
                     f"{stop_txt}fee {fill['fee']:.2f}  id {cid}\nwhy: {reason}")
        return fill

    def sell(self, symbol, qty, reason, bar_time, ref_price=None):
        cid = broker.client_order_id(bar_time, symbol, "SELL", self.s.mode[0])
        fill = self.broker.sell(symbol, qty, cid, ref_price=ref_price)
        pnl = self.store.record_sell(fill, reason)
        self.tg.send(f"SOLD {symbol} qty {fill['qty']:.6g} @ {fill['price']:.6g}  "
                     f"P/L {pnl or 0:+.2f} USDT  id {cid}\nwhy: {reason}")
        return fill

    # execution helpers --------------------------------------------------------------------
    def fill_due_orders(self, latest, bar_time):
        """P0-B: orders decided on the previous close fill at THIS bar's open, plus slippage.

        The backtest makes exactly this assumption, so paper results and research results can be
        compared. Sells go first, freeing cash for the buys, again as in the backtest.
        """
        previous_bar = str(bar_time - pd.Timedelta(self.s.bar_tf))
        for o in self.store.due_orders():
            sym = o["symbol"]
            if sym not in latest.index:
                self.store.finish_order(o["id"], "dropped", "no data for this bar", bar_time)
                continue
            # A buy queued before downtime is stale: it would fill hours late at a price the signal
            # never saw. Drop it and record it, the same way catch-up drops missed entries (P0-D).
            # A sell is different: getting out late still beats not getting out.
            if o["side"] == "BUY" and o["decided_bar"] != previous_bar:
                self.store.finish_order(o["id"], "dropped", "stale: the bot missed the next open",
                                        bar_time)
                self.store.record_skipped({
                    "observed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "bar_time": o["decided_bar"], "symbol": sym, "rank": None, "score": None,
                    "stop": o["stop"], "expected_entry": None,
                    "price_when_seen": float(latest.loc[sym, "open"]),
                    "reason": "stale: queued before downtime, its fill bar never arrived",
                    "downtime_minutes": None, "opportunity_cost": None})
                log(f"dropped stale queued BUY {sym} from {o['decided_bar']}")
                continue
            open_px = float(latest.loc[sym, "open"])
            try:
                if o["side"] == "SELL":
                    pos = self.store.positions().get(sym)
                    if not pos:
                        self.store.finish_order(o["id"], "dropped", "position already closed", bar_time)
                        continue
                    self.sell(sym, min(o["qty"], pos["qty"]), o["reason"], bar_time, ref_price=open_px)
                else:
                    if sym in self.store.positions():
                        self.store.finish_order(o["id"], "dropped", "already holding", bar_time)
                        continue
                    self.buy(sym, o["qty"], o["stop"], o["reason"], bar_time,
                             ref_price=open_px, engine=o["engine"] or "breakout")
                self.store.finish_order(o["id"], "filled", f"at the {bar_time:%d %b %H:%M} open",
                                        bar_time)
            except Exception as e:
                self.store.finish_order(o["id"], "failed", str(e), bar_time)
                log(f"queued {o['side']} {sym} failed: {e}")

    def stop_check(self, latest, bar_time):
        """P0-A, paper: the stop is an intrabar event. If this bar's low pierced the position's
        stop we are out at min(open, stop), the conservative reading of the bar. Demo and live rest
        a real stop order at the exchange instead, so this only simulates what that order would do.
        """
        if not self.flags.stop_exec:
            return set()
        hit = set()
        for sym, p in self.store.positions().items():
            if sym not in latest.index or not p["stop"]:
                continue
            row = latest.loc[sym]
            if float(row["low"]) > p["stop"]:
                continue
            px = min(float(row["open"]), p["stop"])
            try:
                self.sell(sym, p["qty"], "stop hit intrabar", bar_time, ref_price=px)
                hit.add(sym)
            except Exception as e:
                self.tg.send(f"STOP SELL {sym} FAILED: {e}")
        return hit

    def ratchet_stops(self, latest, bar_time):
        """Move each position's stop up to the new 20-bar low. It never moves down (P0-A).

        In demo and live the resting exchange order is replaced to match; in paper the stored value
        is what `stop_check` uses on the next bar.
        """
        if not self.flags.stop_exec:
            return
        for sym, p in self.store.positions().items():
            if sym not in latest.index:
                continue
            new_stop = float(latest.loc[sym, "stop_level"])
            if not new_stop or new_stop <= (p["stop"] or 0):
                continue
            self.store.update_stop(sym, new_stop)
            if self.s.mode in ("demo", "live"):
                self.place_exchange_stop(sym, p["qty"], new_stop, bar_time, replace=True)

    def place_exchange_stop(self, symbol, qty, stop_price, bar_time, replace=False):
        """Rest a STOP_LOSS_LIMIT at the exchange so the position is protected between checks."""
        if self.s.mode not in ("demo", "live"):
            return None
        old = self.store.stops().get(symbol)
        if replace and old:
            try:
                self.broker.cancel_order(symbol, old["client_order_id"])
            except Exception as e:
                log(f"could not cancel the old stop for {symbol}: {e}")
                return None
        cid = broker.client_order_id(f"{bar_time}", symbol, "STOP", self.s.mode[0] + "s")
        try:
            order = self.broker.place_stop(symbol, qty, stop_price, cid)
        except Exception as e:
            self.tg.send(f"STOP ORDER for {symbol} FAILED: {e}\nThe position is unprotected "
                         "between checks until this is fixed.")
            return None
        self.store.put_stop(symbol, client_order_id=cid, exchange_order_id=order.get("orderId"),
                            stop_price=stop_price, limit_price=order.get("price"), qty=qty,
                            status=order.get("status", "NEW"),
                            updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        return order

    # the live cycle -----------------------------------------------------------------------
    def snapshot_levels(self, latest, bar_time, universe_symbols, regime):
        """Freeze what the last closed bar decided, so the live tick can act on it instantly.

        Nothing here looks at a live price: the channel, the ranking and every filter come from a
        bar that has finished. The live tick only asks whether price has since crossed a level.
        """
        cands = set()
        for sym in latest.index:
            row = latest.loc[sym]
            if sym not in universe_symbols:
                continue
            ok = bool(row.get("eligible")) and regime and bool(row.get("rank", 999) <= self.s.candidates)
            ok = ok and (bool(row.get("rs_7d", 0) > 0) or sym == "BTCUSDT")
            ok = ok and bool(signals.extra_filters(latest.loc[[sym]], self.flags).iloc[0])
            if ok:
                cands.add(sym)
        self.levels = {
            "bar_time": str(bar_time), "regime": regime,
            "entry": {s: float(latest.loc[s, "don_high"]) for s in cands
                      if pd.notna(latest.loc[s, "don_high"])},
            "stop": {s: float(latest.loc[s, "stop_level"]) for s in latest.index
                     if pd.notna(latest.loc[s, "stop_level"])},
            "exit": {s: float(latest.loc[s, "don_low"]) for s in latest.index
                     if pd.notna(latest.loc[s, "don_low"])},
            "rank": {s: float(latest.loc[s, "rank"]) for s in latest.index
                     if pd.notna(latest.loc[s, "rank"])},
            # P7: the core's own state, so the website can say what the core will do next without
            # recomputing anything. Its trigger is each coin's distance from its OWN 200-day
            # average, which is not `regime` (that is BTC's, applied to the rotation).
            # `sma200` is the PRICE of the line, not the distance to it, because the live tick
            # needs something to compare a live price against. A 200-day average barely moves
            # between hourly bars, so a level frozen on the last bar is accurate for the next one
            # - which is what lets the core act within seconds instead of waiting for a bar.
            "core": {s: self._core_levels(latest, s)
                     for s in signals.core_symbols(self.flags)
                     if s in latest.index and pd.notna(latest.loc[s, "dist_sma200"])
                     and float(latest.loc[s, "dist_sma200"]) > -1.0},
            "core_weight": float(getattr(self.flags, "trend_core", 0.0) or 0.0),
            "max_names": int(self.max_names),
        }
        self.store.put("levels", self.levels)
        return self.levels

    def live_tick(self, shared_prices=None):
        """Check live prices against the frozen levels. Entries and exits can fire at any moment.

        Exits are checked before entries, and a symbol already acted on in this bar is not acted on
        again, so a price oscillating around a level cannot churn the account. `shared_prices` lets
        the fleet fetch one quote per symbol for every account instead of one each.
        """
        lv = self.levels or self.store.get("levels")
        if not lv:
            return
        core_lv = lv.get("core") or {}
        held = self.store.positions()
        watch = sorted(set(lv["entry"]) | set(held) | set(core_lv))
        if not watch:
            return
        prices = ({k: v for k, v in shared_prices.items() if k in watch}
                  if shared_prices is not None else data.prices(watch))
        acted = set(self.store.get("acted_this_bar") or [])

        for sym, p in held.items():
            px = prices.get(sym)
            level = lv["exit"].get(sym)
            # A core holding is NOT exited on the Donchian low. `lv["exit"]` carries a low for
            # every symbol in the panel, so without this the live tick would sell the core on a
            # breakdown - the reactive exit the core exists to avoid. Its only exit is its own
            # 200-day average, handled in core_tick(). The engine tag stored on the position is
            # what decides, so a coin the breakout engine bought before the core was switched on
            # keeps its Donchian exit.
            if p.get("engine") == "core":
                continue
            if px is None or level is None or sym in acted:
                continue
            if px < level:
                try:
                    self.sell(sym, p["qty"], f"live: price {px:.6g} broke the {level:.6g} exit level",
                              lv["bar_time"], ref_price=px)
                    acted.add(sym)
                except Exception as e:
                    self.tg.send(f"LIVE SELL {sym} FAILED: {e}")

        # The core acts here too, not only on a closed bar. Its signal is a 200-day average, which
        # does not move between hourly bars - but the PRICE does, and crossing that line is the
        # whole rule. Checking it every poll is the difference between acting in ten seconds and
        # waiting up to an hour.
        acted |= self.core_tick(core_lv, prices, lv.get("bar_time"), acted)

        held = self.store.positions()
        state = risk.RiskState(**(self.store.get("risk") or
                                  {"peak_equity": 0, "day": "", "day_start_equity": 0}))
        # The core does not consume the rotation's slots. This counted len(held), so every core
        # holding silently cost the rotation a slot that the bar-close path and the backtest both
        # considered free.
        ok, why = risk.can_enter(state, signals.breakout_slots_used(held), self.max_names,
                                 self.s.kill_switch_path.exists())
        if ok and reconcile.safe_mode(self.store):
            ok = False
        if not ok or self.approval:
            self.store.put("acted_this_bar", sorted(acted))
            return
        equity, _ = self.equity(prices)
        for sym, trigger in sorted(lv["entry"].items(), key=lambda kv: lv["rank"].get(kv[0], 999)):
            if signals.breakout_slots_used(self.store.positions()) >= self.max_names:
                break
            px = prices.get(sym)
            if px is None or sym in held or sym in acted or px <= trigger:
                continue
            stop = lv["stop"].get(sym)
            if not stop or stop >= px:
                continue
            qty, note = risk.size_with_minimum(equity, self.s.risk_pct, px, stop,
                                               self.s.max_weight,
                                               self.broker.filters(sym).min_notional)
            if qty <= 0:
                continue
            try:
                self.buy(sym, qty, stop, f"live: price {px:.6g} crossed the {trigger:.6g} "
                         f"breakout level (rank {int(lv['rank'].get(sym, 0))})", lv["bar_time"],
                         ref_price=px)
                acted.add(sym)
            except Exception as e:
                log(f"live buy {sym} skipped: {e}")
        self.store.put("acted_this_bar", sorted(acted))

    # the 4h cycle -------------------------------------------------------------------------
    def on_bar_close(self, force=False, shared=None):
        """One account's cycle. `shared` is a (universe, panel, bar_time, rows, cands) tuple
        computed once by the fleet: scoring is identical for every account, so doing it per
        account only burns memory - which is what took the 1 GB server down on 4 Oct."""
        held = self.store.positions()
        if shared is not None:
            u, panel, bar_time, rows, cands = shared
        else:
            # The list is told what we hold, so a position is never ejected for a thin volume day.
            u = universe.build(self.s, held_symbols=set(held))
            symbols = sorted(set(u["symbols"]) | set(held))
            panel, bar_time, rows, cands = ranker.scan(self.s, symbols, flags=self.flags)
        if not force and self.store.get("last_bar") == str(bar_time):
            return
        expected = pd.Timestamp(datetime.now(timezone.utc)).floor(self.s.bar_tf)
        if not force and self.s.shared_data and bar_time < expected:
            return      # sharing another bot's files: wait until it has appended the new bar
        latest = rows.set_index("symbol")
        # Orders decided on the previous close fill at THIS bar's open (P0-B); the stop is then
        # checked against this bar's low (P0-A). Both happen before anything else on the bar.
        if self.queued_fills:
            self.fill_due_orders(latest, bar_time)
        stopped = self.stop_check(latest, bar_time)
        held = self.store.positions()
        prices = latest["close"].to_dict()
        equity, cash = self.equity(prices)
        day = str((bar_time - pd.Timedelta(milliseconds=1)).date())
        state = self.risk_state(equity, day)
        for ev in risk.update_limits(state, equity, day, self.s):
            self.tg.send("RISK: " + ev)
        self.store.put("risk", state.__dict__)
        regime = bool(latest.loc["BTCUSDT", "regime_on"])
        missed_exits = self.catch_up(panel, bar_time, held)
        log(f"bar {bar_time} | equity {equity:,.2f} | cash {cash:,.2f} | regime {'ON' if regime else 'OFF'} "
            f"| holding {list(held) or 'nothing'}")

        core_syms = signals.core_symbols(self.flags)

        self.expire_bar_guard(bar_time)

        # exits run automatically in every mode
        for sym, p in held.items():
            if sym in stopped:
                continue        # already closed by its stop on this bar
            if p.get("engine") == "core":
                continue        # P7: the core decides its own exit, in core_rebalance() below
            reason = None
            if sym not in u["symbols"]:
                reason = "dropped off the authenticated list"
            elif not universe.tradable_status(sym):
                reason = "Binance status is not TRADING"
            elif state.killed or self.s.kill_switch_path.exists():
                reason = "kill switch"
            elif sym in missed_exits:
                reason = missed_exits[sym]
            elif sym in latest.index:
                r = latest.loc[sym]
                broke = bool(r["breakdown_exit"]) if (self.flags.exit_donchian
                                                      and "breakdown_exit" in latest.columns)                     else bool(r["breakdown"])
                entry = p.get("entry_price") or 0
                reason = signals.exit_reason(
                    sym, bool(r["regime_on"]), broke, r["rank"], self.s, self.flags,
                    bars_held=None, in_profit=float(r["close"]) > entry if entry else None)
            if reason:
                self.exit_position(sym, p, reason, bar_time)

        # P7 the trend core, before the breakout entries so the cash an exit frees is available.
        # This is the part of the bot that ACTS without being asked: it does not wait for a
        # breakout, a rank or an approval - while BTC is above its own 200-day average it is long,
        # and it stays long until that stops being true.
        self.core_rebalance(latest, bar_time, state, equity)

        # entries
        # The fleet computes one candidate list, from bots[0]'s flags. The moment two accounts run
        # different flags that list is wrong for every account but the first - this book would
        # silently trade another book's signals. The panel and the scores are flag-independent, so
        # each account re-derives its own candidates from the shared rows; it is one cheap call on
        # a single bar.
        if shared is not None:
            cands = signals.entry_candidates(rows, self.s, flags=self.flags)
        cands = cands[cands["symbol"].isin(u["symbols"])]
        self.store.put("last_signals", list(cands["symbol"]))
        held = self.store.positions()
        # The core is not one of the breakout slots: counting it would silently cost the rotation a
        # third of its capacity. This used to test `s not in core_syms` - symbol membership, which
        # miscounts a BTCUSDT the BREAKOUT engine owns - and the backtest tested the engine tag.
        n_alt = signals.breakout_slots_used(held)
        ok, why = risk.can_enter(state, n_alt, self.max_names, self.s.kill_switch_path.exists())
        if ok and reconcile.safe_mode(self.store):
            ok, why = False, "safe mode: the database and the exchange disagree"
        if self.approval:
            self.store.set_signal_status("expired")  # older bars' signals die with the new bar
            expires = min(time.time() + self.s.signal_expiry_hours * 3600,
                          (bar_time + pd.Timedelta(self.s.bar_tf)).timestamp())
            eligible = {c["symbol"] for c in u["coins"] if c["live_eligible"]}
            listed = []
            for r in cands.itertuples():
                if r.symbol in held or r.symbol not in eligible:
                    continue
                row = r._asdict() | {"reason": f"breakout, rank {int(r.rank)}"}
                self.store.add_signal(str(bar_time), row, expires)
                listed.append(row)
            msg = notify.candidates_message(bar_time, regime, listed[: self.s.candidates],
                                            self.s.signal_expiry_hours)
            if not ok:
                msg += f"\n(New entries blocked: {why})"
            if listed or held:
                self.tg.send(msg)
        elif ok:
            slots = self.max_names - n_alt
            for r in cands[~cands["symbol"].isin(held)].head(slots).itertuples():
                qty, note = risk.size_with_minimum(equity, self.s.risk_pct, r.close, r.stop_level,
                                                   self.s.max_weight,
                                                   self.broker.filters(r.symbol).min_notional)
                if qty <= 0:
                    log(f"{r.symbol} not bought: {note}")
                    continue
                if note:
                    log(f"{r.symbol}: {note}")
                reason = f"breakout, rank {int(r.rank)}, 7d vs BTC {r.rs_7d * 100:+.1f}%"
                if self.queued_fills:
                    # Decided now, filled at the next bar's open, exactly as the backtest does.
                    self.store.queue_order(bar_time, "BUY", r.symbol, qty, r.stop_level, reason)
                    log(f"queued BUY {r.symbol} for the next open: {reason}")
                    continue
                try:
                    fill = self.buy(r.symbol, qty, r.stop_level, reason, bar_time)
                    if self.flags.stop_exec:
                        self.place_exchange_stop(r.symbol, fill["qty"], r.stop_level, bar_time)
                except Exception as e:
                    log(f"buy {r.symbol} skipped: {e}")

        # The snapshot is written on every bar, not only in live-cycle mode. The live tick is one
        # consumer of it; the "Next move" page is the other, and on a 4-hour cycle that page would
        # otherwise have nothing to show at all. It is a single key written once per bar.
        self.snapshot_levels(latest, bar_time, set(u["symbols"]), regime)
        self.ratchet_stops(latest, bar_time)
        equity, cash = self.equity(prices)
        self.store.record_equity(str(bar_time), equity, cash)
        self.store.put("last_bar", str(bar_time))

    def record_skipped_entries(self, panel, t, bar_time, held, now_rows, downtime_min):
        """P0-D: an entry that fired while the bot was down is stale and is not taken, but it is
        written down: which coin, when, at what rank, what it would have cost, and what the price
        did afterwards. That turns downtime into something measurable instead of invisible.
        """
        bar = panel[panel["bar_time"] == t]
        try:
            cands = signals.entry_candidates(bar, self.s)
        except Exception:
            return
        for r in cands.head(self.s.candidates).itertuples():
            if r.symbol in held:
                continue
            later = panel[(panel["symbol"] == r.symbol) & (panel["bar_time"] > t)]
            expected = float(later["open"].iloc[0]) if len(later) else float(r.close)
            seen = float(now_rows.loc[r.symbol, "close"]) if r.symbol in now_rows.index else None
            self.store.record_skipped({
                "observed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "bar_time": str(t), "symbol": r.symbol, "rank": int(r.rank), "score": float(r.score),
                "stop": float(r.stop_level), "expected_entry": expected, "price_when_seen": seen,
                "reason": "stale: the bot was not running when this bar closed",
                "downtime_minutes": round(downtime_min, 1),
                "opportunity_cost": (seen / expected - 1) if seen and expected else None})

    def _core_levels(self, latest, s):
        """The price levels the live tick compares against, for one core coin.

        Stored as PRICES rather than distances because the live tick has a live price and needs
        something to compare it to. `sma50` is carried whenever the column exists so that turning
        `core_dual_ma` on later does not silently leave the live path checking only one line while
        the backtest checks two - the two must never disagree about what the rule is.
        """
        close = float(latest.loc[s, "close"])
        d200 = float(latest.loc[s, "dist_sma200"])
        row = {"dist_sma200": d200, "sma200": close / (1.0 + d200)}
        d50 = latest.loc[s].get("dist_sma50")
        if d50 is not None and pd.notna(d50) and float(d50) > -1.0:
            row["dist_sma50"] = float(d50)
            row["sma50"] = close / (1.0 + float(d50))
        dual = bool(getattr(self.flags, "core_dual_ma", False))
        row["dual_ma"] = dual
        row["on"] = bool(d200 > 0 and (not dual or (d50 is not None and pd.notna(d50)
                                                    and float(d50) > 0)))
        return row

    def core_tick(self, core_lv, prices, bar_time, acted):
        """The trend core, checked against a live price instead of a bar close.

        Same rule as core_rebalance - long while the coin is above its own 200-day average - but
        compared to the price right now against the line frozen on the last bar. A 200-day average
        moves by a fraction of a percent per hour, so that level is accurate; the price is not, and
        the price is the half that matters.

        `acted` is the existing anti-churn guard: a symbol already traded on this bar is left
        alone, so a price oscillating either side of the line cannot churn the account. That makes
        at most one core decision per symbol per bar, which is the right cadence for a signal this
        slow while still reacting within one poll of a genuine crossing.
        """
        if not core_lv:
            return set()
        done = set()
        held = self.store.positions()
        blocked = (self.s.kill_switch_path.exists() or reconcile.safe_mode(self.store))
        state = risk.RiskState(**(self.store.get("risk") or
                                  {"peak_equity": 0, "day": "", "day_start_equity": 0}))
        blocked = blocked or state.killed

        def _above(s, d):
            px = prices.get(s)
            if px is None or not d.get("sma200") or px <= d["sma200"]:
                return False
            if d.get("dual_ma"):
                # P14: both lines, or the live tick would hold what the backtest would have sold
                return bool(d.get("sma50")) and px > d["sma50"]
            return True

        on = [s for s, d in core_lv.items() if _above(s, d) and not blocked]
        equity, _ = self.equity(prices)

        for sym, d in core_lv.items():
            if sym in acted:
                continue
            px, line = prices.get(sym), d.get("sma200")
            # Whether a position is the core's is decided by the SYMBOL, not by parsing its
            # reason text. The first version matched reason.startswith("core"), and the live buy
            # writes "live core: ...", so the core bought and could then never sell itself. A core
            # symbol is reserved from the breakout engine, so anything held in one is the core's.
            p = held.get(sym)
            is_core = p is not None and p.get("engine") == "core"
            if px is None or not line:
                continue
            if is_core and (not _above(sym, d) or blocked):
                lost = ("its 50-day average" if d.get("dual_ma") and d.get("sma50")
                        and px <= d["sma50"] else "its 200-day average")
                why = ("kill switch" if blocked
                       else f"live core: {px:.6g} fell below {lost}")
                try:
                    self.sell(sym, p["qty"], why, bar_time, ref_price=px)
                    done.add(sym)
                except Exception as e:
                    self.tg.send(f"LIVE CORE SELL {sym} FAILED: {e}")
            elif p is None and sym in on:
                target = equity * self.flags.trend_core / len(on)
                f = self.broker.filters(sym)
                qty = f.round_qty(target / px)
                if qty * px < f.min_notional:
                    continue
                why = f"live core: {px:.6g} above its 200-day average {line:.6g}"
                try:
                    self.buy(sym, qty, None, why, bar_time, ref_price=px, engine="core")
                    log(f"live core BUY {sym}: {why}")
                    done.add(sym)
                except Exception as e:
                    self.tg.send(f"LIVE CORE BUY {sym} FAILED: {e}")
        return done

    def expire_bar_guard(self, bar_time):
        """Start the per-bar churn guard fresh. The ONLY place `acted_this_bar` is cleared.

        on_bar_close calls this once, before any decision on the bar. Both decision paths - the
        bar close and the live tick - then read and append to one list for the rest of the bar.

        It is a method rather than three lines inline because the 5 Nov version of this guard was
        inert for want of a single owner. The expiry lived inside core_rebalance, and a SECOND
        reset sat at the end of on_bar_close behind `if self.s.live_cycle`:

            core_rebalance -> acted_this_bar = ['BTCUSDT']   ...   end of bar -> wiped

        `DECISION_TF=live` is what the server runs, so the bar-close path's entry was erased
        before the live tick could ever read it and the backstop protected nothing in exactly the
        configuration it was added for. Two resets for one piece of state is the same shape as the
        bug it was guarding against: two copies of one rule.

        Expiry is by comparing the stored bar, not by clearing on a schedule, because a book whose
        core is off returns from core_rebalance immediately - so an expiry living in there would
        never run for the paper books at all.
        """
        if self.store.get("acted_bar") != str(bar_time):
            self.store.put("acted_bar", str(bar_time))
            self.store.put("acted_this_bar", [])

    def core_rebalance(self, latest, bar_time, state, equity):
        """P7: hold each core coin while it is above its own 200-day average. Act, don't wait.

        This is deliberately the simplest decision in the bot, and the only one that is not
        conditional on a breakout, a rank, a volume reading or an approval message:

            above its own 200-day average  ->  be long it
            below                          ->  be flat it

        It runs in every mode including approval mode, because the core is not a trade idea to be
        confirmed - it is the book's default state. The things that still override it are the ones
        that belong to the account rather than to a strategy: the drawdown kill, the kill-switch
        file, safe mode after a reconciliation mismatch, and a coin Binance has stopped trading.

        No Donchian stop is placed. The core's risk control IS the 200-day average; a stop under a
        long-term trend position would convert it into the reactive trade this flag exists to stop
        being. The drawdown kill remains the backstop.
        """
        core_syms = signals.core_symbols(self.flags)
        if not core_syms:
            return
        held = self.store.positions()
        blocked = (state.killed or self.s.kill_switch_path.exists()
                   or reconcile.safe_mode(self.store))

        # The predicate comes from signals.core_on(), not from a copy of it written here.
        #
        # This used to test `dist_sma200 > 0` inline, which was the same rule until core_dual_ma
        # shipped and added the 50-day line to core_on() and to core_tick() but not to this copy.
        # The result was a bar-close path that BOUGHT because price was above the 200-day and a
        # live path that SOLD because it was below the 50-day, once an hour, for three days:
        # 203 fills and 118.32 USDT of fees on the demo book between 30 Oct and 1 Nov.
        #
        # Three paths decide whether the core is on. They now all call one function, so they
        # cannot disagree about the rule no matter what is added to it later.
        on_flags = signals.core_on(latest.assign(symbol=latest.index), self.flags)
        on = [sym for sym in core_syms
              if sym in latest.index and bool(on_flags.get(sym, False))
              and not blocked and universe.tradable_status(sym)]

        # A symbol already traded on this bar is left alone until the next one. The live tick has
        # had this guard since the core shipped; the bar-close path did not, which is what let the
        # two engines trade the same coin against each other within one bar. It is a backstop, not
        # the fix - the shared predicate above is the fix - but a backstop is what stops a future
        # divergence costing money before anyone notices.
        #
        # This only READS the guard. on_bar_close owns the expiry, so that one bar has exactly one
        # reset and it happens before any decision. See the comment there.
        acted = set(self.store.get("acted_this_bar") or [])

        for sym in core_syms:
            if sym in acted:
                continue
            p = held.get(sym)
            if p and sym not in on:
                why = ("kill switch" if blocked else "core: closed below its 200-day average")
                log(f"core exit {sym}: {why}")
                self.exit_position(sym, p, why, bar_time)
                acted.add(sym)
                self.store.put("acted_this_bar", sorted(acted))
            elif not p and sym in on:
                r = latest.loc[sym]
                # P12 step 4: refuse an entry already stretched far above the line it would exit
                # on. backtest.run has applied this since the flag shipped and the live bar-close
                # path did not - the same class of divergence as the dual_ma churn, found by the
                # path-agreement tests rather than by anyone noticing.
                if not signals.core_entry_allowed(self.flags, r.get("dist_sma200")):
                    continue
                # Equal split across the coins currently above their line. `trend_core` is itself
                # the cap; settings.max_weight governs the rotation's picks, not this deliberate
                # allocation. See the matching comment in backtest.run.
                target = equity * self.flags.trend_core / len(on)
                px = float(r["close"])
                f = self.broker.filters(sym)
                qty = f.round_qty(target / px)
                if qty * px < f.min_notional:
                    log(f"core {sym} not bought: {qty * px:.2f} USDT is below the minimum order")
                    continue
                reason = (f"core: {float(r['dist_sma200']) * 100:+.1f}% above its "
                          f"200-day average")
                if self.queued_fills:
                    self.store.queue_order(bar_time, "BUY", sym, qty, None, reason,
                                           engine="core")
                    log(f"queued core BUY {sym} for the next open: {reason}")
                    acted.add(sym)
                    self.store.put("acted_this_bar", sorted(acted))
                    continue
                try:
                    self.buy(sym, qty, None, reason, bar_time, engine="core")
                    log(f"core BUY {sym}: {reason}")
                    acted.add(sym)
                    self.store.put("acted_this_bar", sorted(acted))
                except Exception as e:
                    log(f"core buy {sym} skipped: {e}")

    def exit_position(self, sym, p, reason, bar_time):
        """Close a position: queued to the next open in paper, straight away in demo and live.

        A resting exchange stop is cancelled first, so the sell cannot race its own protection.
        """
        if self.s.mode in ("demo", "live") and sym in self.store.stops():
            try:
                self.broker.cancel_order(sym, self.store.stops()[sym]["client_order_id"])
            except Exception as e:
                log(f"could not cancel the stop for {sym} before selling: {e}")
            self.store.clear_stop(sym)
        if self.queued_fills:
            self.store.queue_order(bar_time, "SELL", sym, p["qty"], p.get("stop"), reason)
            log(f"queued SELL {sym} for the next open: {reason}")
            return
        try:
            self.sell(sym, p["qty"], reason, bar_time)
        except Exception as e:
            self.tg.send(f"SELL {sym} FAILED: {e}")

    def catch_up(self, panel, bar_time, held):
        """Checks missed while the bot was off (PC asleep, restart). Returns {symbol: sell reason}.

        A sell signal on a missed check is acted on now, at today's price. Buy signals from missed
        checks are dropped: they are stale, the same way a live signal dies with the next bar.
        Equity is also recorded for each missed check so the chart has no hole.
        """
        last = self.store.get("last_bar")
        if not last:
            return {}
        missed = sorted(t for t in panel["bar_time"].unique() if pd.Timestamp(last) < t < bar_time)
        if not missed:
            return {}
        out = {}
        cash = self.broker.cash() if self.s.mode == "paper" else None
        now_rows = panel[panel["bar_time"] == bar_time].set_index("symbol")
        downtime_min = (bar_time - pd.Timestamp(last)).total_seconds() / 60
        for t in missed:
            self.record_skipped_entries(panel, t, bar_time, held, now_rows, downtime_min)
            rows = panel[panel["bar_time"] == t].set_index("symbol")
            for sym in held:
                if sym in out or sym not in rows.index:
                    continue
                r = rows.loc[sym]
                broke = bool(r["breakdown_exit"]) if (self.flags.exit_donchian
                                                      and "breakdown_exit" in rows.columns)                     else bool(r["breakdown"])
                why = signals.exit_reason(sym, bool(r["regime_on"]), broke, r["rank"], self.s,
                                          self.flags)
                if why:
                    out[sym] = f"{why} (at the missed {t:%d %b %H:%M} UTC check; sold on catch-up)"
            if cash is not None:
                value = sum(p["qty"] * float(rows.loc[s, "close"]) for s, p in held.items() if s in rows.index)
                self.store.record_equity(str(t), cash + value, cash)
        log(f"caught up {len(missed)} missed check(s) from {missed[0]} to {missed[-1]}"
            + (f"; sell signals: {', '.join(out)}" if out else "; no sell signals in them"))
        return out

    # commands (section 5.1) ------------------------------------------------------------------
    def handle_commands(self):
        for c in self.store.new_commands():
            try:
                reply = self.handle(notify.parse(c["text"]), c["text"])
                status = "rejected" if reply.startswith("REJECTED") else "done"
            except Exception as e:
                reply, status = f"ERROR: {e}", "error"
            self.store.finish_command(c["id"], status, reply)
            self.tg.send(reply)

    def handle(self, cmd, raw=""):
        if cmd is None:
            return f"REJECTED: could not read {raw!r}. Use TRADE ETH, TRADE ETH 200 USDT, SKIP, FLAT ETH, FLAT ALL, FORCE ETH, STATUS."
        now_bar = self.store.get("last_bar")
        if cmd.verb == "STATUS":
            return self.status_text()
        if cmd.verb == "SKIP":
            self.store.set_signal_status("skipped")
            return "OK: skipped this bar's candidates. No order sent."
        if cmd.verb == "FLAT":
            pos = self.store.positions()
            targets = list(pos) if cmd.symbol == "ALL" else [cmd.symbol]
            # A queued buy must not resurrect a position you just flattened.
            self.store.drop_queued(None if cmd.symbol == "ALL" else cmd.symbol)
            done = []
            for sym in targets:
                if sym not in pos:
                    return f"REJECTED: no open position in {sym}."
                if self.s.mode in ("demo", "live") and sym in self.store.stops():
                    try:
                        self.broker.cancel_order(sym, self.store.stops()[sym]["client_order_id"])
                    except Exception as e:
                        log(f"could not cancel the stop for {sym}: {e}")
                    self.store.clear_stop(sym)
                f = self.sell(sym, pos[sym]["qty"], "FLAT command", f"{datetime.now(timezone.utc).isoformat()}")
                done.append(f"{sym} @ {f['price']:.6g}")
            return "OK: flattened " + (", ".join(done) or "nothing (no positions)")

        # TRADE / FORCE
        if reconcile.safe_mode(self.store):
            return ("REJECTED: safe mode is on because the database and the exchange disagree. "
                    "Run: python -m app.main reconcile")
        sym = cmd.symbol
        u = universe.load(self.s)
        coin = next((c for c in u["coins"] if c["symbol"] == sym), None)
        if coin is None:
            return f"REJECTED: {sym} is not on today's authenticated list."
        if self.s.mode == "live" and not coin["live_eligible"]:
            return f"REJECTED: {sym} has under {self.s.min_history_days} days of history."
        pend = self.store.open_signals()
        if cmd.verb == "TRADE" and sym not in pend:
            return f"REJECTED: {sym} is not a current candidate (or its signal expired)."
        held = self.store.positions()
        if sym in held:
            return f"REJECTED: already holding {sym} (no adding to positions)."
        state = self.store.get("risk") or {}
        ok, why = risk.can_enter(risk.RiskState(**state) if state else risk.RiskState(0, "", 0),
                                 len(held), self.max_names, self.s.kill_switch_path.exists())
        if not ok:
            return f"REJECTED: {why}."
        if not universe.tradable_status(sym):
            return f"REJECTED: Binance status for {sym} is not TRADING."

        price = data.price(sym)
        if sym in pend:
            stop = pend[sym]["stop"]
        else:  # FORCE on a non-candidate: use the current 20-bar low
            _, _, rows, _ = ranker.scan(self.s, sorted(set(u["symbols"])), refresh=False)
            stop = float(rows.set_index("symbol").loc[sym, "stop_level"])
        equity, cash = self.equity({sym: price})
        if cmd.amount_usdt:
            qty = min(cmd.amount_usdt, equity * self.s.max_weight, cash) / price
        else:
            qty, note = risk.size_with_minimum(equity, self.s.risk_pct, price, stop,
                                               self.s.max_weight,
                                               self.broker.filters(sym).min_notional)
            if qty <= 0:
                return f"REJECTED: {note}"
        try:
            fill = self.buy(sym, qty, stop, f"{cmd.verb} command", now_bar or datetime.now(timezone.utc).isoformat())
            if self.flags.stop_exec:
                self.place_exchange_stop(sym, fill["qty"], stop, now_bar or datetime.now(timezone.utc))
        except broker.OrderRejected as e:
            return f"REJECTED: {e}"
        self.store.set_signal_status("traded", sym)
        return (f"OK: bought {fill['qty']:.6g} {sym} @ {fill['price']:.6g}, stop {stop:.6g}, "
                f"id {fill['client_order_id']}")

    def daily_summary(self):
        """One message a day: account, what is held, and yesterday's buys and sells."""
        since = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        recent = [f for f in self.store.fills(limit=200) if f["time"] >= since]
        sold = [f for f in self.store.fills(limit=10_000) if f["side"] == "SELL" and f["pnl"] is not None]
        lines = ["Daily summary", self.status_text()]
        if self.s.mode == "paper":
            eq = self.equity({s: data.price(s) for s in self.store.positions()})[0]
            lines.append(f"Since start: {eq / self.s.start_equity - 1:+.2%}")
        lines.append(f"Last 24h: {sum(f['side'] == 'BUY' for f in recent)} buys, "
                     f"{sum(f['side'] == 'SELL' for f in recent)} sells")
        if sold:
            wins = sum(f["pnl"] > 0 for f in sold)
            lines.append(f"Sold so far: {len(sold)} trades, {wins} won, "
                         f"total {sum(f['pnl'] for f in sold):+,.2f} USDT")
        return "\n".join(lines)

    def status_text(self):
        pos = self.store.positions()
        prices = {s: data.price(s) for s in pos}
        equity, cash = self.equity(prices)
        lines = [f"[{self.s.mode.upper()}] equity {equity:,.2f} USDT, cash {cash:,.2f}"]
        for s, p in pos.items():
            # A core position has no stop by design - its exit is its own 200-day average - so
            # `stop` is None and "{None:.6g}" is a TypeError. That crashed status_text(), which
            # crashed the daily summary, which meant the DEMO book (the only one holding core
            # positions, and the one being trialled) had never sent one. The same formatting bug
            # was fixed in buy() when the core shipped and missed here; the engine tag is now the
            # thing that decides what to print, rather than hoping a stop exists.
            tail = (f"exits below its 200-day average" if p.get("engine") == "core"
                    else f"stop {p['stop']:.6g}" if p.get("stop") else "no stop recorded")
            lines.append(f"  {s}: {p['qty']:.6g} @ {p['entry_price']:.6g} now {prices[s]:.6g} "
                         f"({prices[s] / p['entry_price'] - 1:+.1%}) {tail}")
        for s, sig in self.store.open_signals().items():
            lines.append(f"  pending: {s} rank {sig['rank']} expires "
                         f"{datetime.fromtimestamp(sig['expires_at'], timezone.utc):%H:%M} UTC")
        return "\n".join(lines)

    def boot_reconcile(self):
        """P0-C: before trading anything, check the database against the exchange."""
        try:
            rep = reconcile.check(self.s, self.broker, self.store)
        except Exception as e:                       # an unreachable exchange is not a mismatch
            log(f"reconciliation could not run ({e}); entries stay blocked until it does")
            return None
        if rep.ok:
            log(rep.text())
            if reconcile.safe_mode(self.store):
                reconcile.clear_safe_mode(self.store)
                self.tg.send("Reconciled: the database and the exchange agree again. "
                             "Safe mode is off and entries are allowed.")
        else:
            reconcile.enter_safe_mode(self.store, rep, log, self.tg)
        return rep

    # loop -------------------------------------------------------------------------------------
    def _daily(self):
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")
        if today == self._last_daily or (now.hour, now.minute) < (0, 5):
            return
        u = universe.build(self.s, refresh=not self.s.shared_data,
                           held_symbols=set(self.store.positions()))
        if not self.s.shared_data:
            backfill.run(self.s, u["symbols"], log=lambda *_: None)
            if self.flags.funding_max_pct:
                funding.backfill(self.s, u["symbols"], log=lambda *_: None)
        self._last_daily = today
        log(f"daily: universe {len(u['symbols'])} names, history appended")
        if self.store.get("summary_day") != today:  # once a day, even across restarts
            self.tg.send(self.daily_summary())
            self.store.put("summary_day", today)

    def _bar(self):
        now = datetime.now(timezone.utc)
        last_closed = pd.Timestamp(now).floor(self.s.bar_tf)
        if self.store.get("last_bar") != str(last_closed) and now - last_closed > timedelta(seconds=20):
            self.on_bar_close()  # also catches up any checks missed while offline

    def _live(self):
        if self.s.live_cycle:
            self.live_tick()

    def _telegram(self):
        self.tg.poll()
        self.handle_commands()

    def run_forever(self):
        """Never exits on its own. Each job is tried separately, so a network failure in one does
        not block the others; while offline it retries every 5 seconds and says when it is back."""
        log(f"bot started in {self.s.mode.upper()} mode; approval gate {'ON' if self.approval else 'OFF'}"
            f"; flags {self.flags.label}")
        self.boot_reconcile()
        # In the live cycle the triggers are useless without levels, and waiting up to four hours
        # for the next close would leave the bot blind after every restart.
        if self.s.live_cycle and not self.store.get("levels"):
            try:
                self.on_bar_close(force=True)
                log("live cycle: levels primed from the last closed bar")
            except Exception:
                log("could not prime live levels:\n" + traceback.format_exc())
        self.tg.send(f"Bot started ({self.s.mode}).")
        self._last_daily = None
        offline_since = None
        while True:
            # Heartbeat first: the bot is alive even when the internet is not.
            self.s.heartbeat_path.write_text(str(time.time()))
            failed = None
            for job in (self._daily, self._bar, self._live, self._telegram):
                try:
                    job()
                except NETWORK_ERRORS as e:
                    failed = e
                except Exception:
                    log(f"{job.__name__} error:\n" + traceback.format_exc())
            if failed is not None and offline_since is None:
                offline_since = time.time()
                self.store.put("offline_since", offline_since)
                log(f"OFFLINE: cannot reach Binance/Telegram ({type(failed).__name__}). "
                    "Retrying every 5 seconds; positions are saved locally.")
            elif failed is None and offline_since is not None:
                mins = (time.time() - offline_since) / 60
                offline_since = None
                self.store.put("offline_since", None)
                log(f"ONLINE again after {mins:.1f} minutes; missed checks were caught up.")
                self.tg.send(f"Back online after {mins:.0f} min without internet. "
                             "Any missed 4-hour checks were caught up.\n" + self.status_text())
            pause = self.s.live_poll_seconds if self.s.live_cycle else 15
            time.sleep(5 if offline_since else pause)


# ---------------------------------------------------------------------------------------- CLI

def cmd_backtest(args, s):
    from . import backtest, experiment, features
    u = universe.load(s)
    flags = Flags.from_args(args)
    panel = features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars)
    res = backtest.run(panel, s, start=args.start, end=args.end, flags=flags)
    name = getattr(args, "label", None) or f"standard_{flags.label}"
    report(res, panel, u, s, "ranker")
    rec = experiment.summarise(res, panel, s, flags, name, "standard")
    path = experiment.save(rec, REPORTS)
    print(f"\n  flags: {flags.enabled or 'baseline'}   experiment record: {path}")
    print_regimes(rec)
    return res, panel


def print_regimes(rec):
    if not isinstance(rec.get("regimes"), list) or not rec["regimes"]:
        return
    df = pd.DataFrame(rec["regimes"])
    show = [c for c in ("regime", "years", "cagr", "sharpe", "max_drawdown", "trades",
                        "win_rate", "profit_factor", "exposure") if c in df]
    print("\n  By market regime:\n" + df[show].to_string(index=False,
                                                         float_format=lambda x: f"{x:.3f}"))


def report(res, panel, u, s, name):
    from . import backtest
    REPORTS.mkdir(exist_ok=True)
    m = backtest.metrics(res.equity, res.trades, res.exposure)
    top10 = [c["symbol"] for c in sorted(u["coins"], key=lambda c: c["cg_rank"])[:10]]
    bench = backtest.benchmarks(panel, res.equity.index, top10)
    print(f"\n=== {name}: {m['start']} to {m['end']} ===")
    for k, v in m.items():
        if k not in ("start", "end"):
            print(f"  {k:<16} {v:.4f}" if isinstance(v, float) else f"  {k:<16} {v}")
    print(f"  kills fired      {res.kills}   daily halts {res.halts}")
    for bname, curve in bench.items():
        bm = backtest.metrics(curve)
        print(f"  vs {bname:<22} return {bm['total_return']:+.1%}  CAGR {bm['cagr']:+.1%}  "
              f"max DD {bm['max_drawdown']:.1%}  Sharpe {bm['sharpe']:.2f}")
    yr = pd.DataFrame({"strategy": backtest.yearly(res.equity),
                       **{k: backtest.yearly(v) for k, v in bench.items()}})
    print("\n  Year by year:\n" + (yr * 100).round(1).to_string())
    rb = backtest.random_baselines(panel, res.trades, s)
    if rb:
        print(f"\n  Avg trade {rb['strategy_avg_trade']:+.2%} | random coin, same timing "
              f"{rb['random_coin_avg_trade']:+.2%} (p={rb['p_value_vs_random_coin']:.3f}) | "
              f"same coin, random timing {rb['random_time_avg_trade']:+.2%} (p={rb['p_value_vs_random_time']:.3f})")
    res.trades.to_csv(REPORTS / f"{name}_trades.csv", index=False)
    res.log.to_csv(REPORTS / f"{name}_log.csv", index=False)
    res.equity.to_csv(REPORTS / f"{name}_equity.csv")
    return m, yr, rb


def cmd_ml(args, s):
    from . import backtest, features, ml
    u = universe.load(s)
    panel = features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars)
    p = ml.walk_forward(panel, horizon=args.horizon, log=log)
    oos = p[p["ml_score"].notna()]
    start = oos["bar_time"].min()
    print(f"\nOut-of-sample period: {start.date()} to {oos['bar_time'].max().date()}")
    for col in ("score", "ml_score"):
        ic = ml.information_coefficient(oos, col)
        spread = ml.top_minus_bottom(oos, col)
        print(f"  {col:<9} mean IC {ic['mean_ic']:+.4f}  IC>0 on {ic['ic_positive_share']:.1%} of bars  "
              f"top-minus-bottom 20% {spread:+.3%} per {args.horizon} bars")
    merged = panel.merge(p[["bar_time", "symbol", "ml_score"]], on=["bar_time", "symbol"], how="left")
    r_rank = backtest.run(merged, s, start=start)
    r_ml = backtest.run(merged, s, score_col="ml_score", start=start)
    m1, _, _ = report(r_rank, merged, u, s, "ranker_oos")
    m2, _, _ = report(r_ml, merged, u, s, "ml_oos")
    better = m2["total_return"] > m1["total_return"] and m2["sharpe"] > m1["sharpe"]
    print(f"\nShip the model? {'YES' if better else 'NO'} - it must beat the ranker after fees "
          f"and slippage on return and Sharpe over the same out-of-sample period.")


def set_env(key, value):
    """Write KEY=value into .env (created from .env.example if missing), replacing any old line."""
    env = config.ROOT / ".env"
    if not env.exists():
        env.write_text((config.ROOT / ".env.example").read_text())
    lines = [ln for ln in env.read_text().splitlines() if not ln.startswith(key + "=")]
    lines.append(f"{key}={value}")
    env.write_text("\n".join(lines) + "\n")


def cmd_pit(args, s):
    from . import backtest, experiment, pit
    flags = Flags.from_args(args)
    res, panel, member = pit.run(s, top_n=args.top, log=log, flags=flags)
    m = backtest.metrics(res.equity, res.trades, res.exposure)
    btc = panel[panel.symbol == "BTCUSDT"].set_index("bar_time")["close"].reindex(res.equity.index).ffill()
    curves = {"This bot": res.equity / res.equity.iloc[0], "BTC buy & hold": btc / btc.iloc[0],
              f"Equal-weight top {args.top}": pit.equal_weight_universe(panel, res.equity.index)}
    print()
    print(f"=== Survivorship-free re-test: top {args.top} by volume each day, {m['start']} to {m['end']} ===")
    for k in ("cagr", "max_drawdown", "sharpe", "trades", "win_rate", "avg_trade", "avg_exposure"):
        print(f"  {k:<14} {m[k]:.4f}" if isinstance(m[k], float) else f"  {k:<14} {m[k]}")
    for name, c in list(curves.items())[1:]:
        bm = backtest.metrics(c)
        print(f"  vs {name:<22} CAGR {bm['cagr']:+.1%}  max DD {bm['max_drawdown']:.1%}  Sharpe {bm['sharpe']:.2f}")
    yr = pd.DataFrame({k: backtest.yearly(v) for k, v in curves.items()})
    print("\n  Year by year (%):\n" + (yr * 100).round(1).to_string())
    rb = backtest.random_baselines(panel, res.trades, s)
    print(f"\n  Avg trade {rb['strategy_avg_trade']:+.2%} | random coin, same timing {rb['random_coin_avg_trade']:+.2%} "
          f"(p={rb['p_value_vs_random_coin']:.3f}) | same coin, random timing {rb['random_time_avg_trade']:+.2%} "
          f"(p={rb['p_value_vs_random_time']:.3f})")
    names = res.trades["symbol"].value_counts()
    print(f"\n  Coins traded: {len(names)}; most traded: {', '.join(names.head(12).index.str.replace('USDT', ''))}")
    REPORTS.mkdir(exist_ok=True)
    res.trades.to_csv(REPORTS / f"pit{args.top}_trades.csv", index=False)
    res.equity.to_csv(REPORTS / f"pit{args.top}_equity.csv")
    name = getattr(args, "label", None) or f"pit{args.top}_{flags.label}"
    rec = experiment.summarise(res, panel, s, flags, name, f"pit{args.top}")
    print(f"\n  flags: {flags.enabled or 'baseline'}   experiment record: "
          f"{experiment.save(rec, REPORTS)}")
    print_regimes(rec)


def telegram_setup(s):
    """Connect Telegram: asks for the BotFather token (hidden), then finds your chat by itself."""
    import getpass

    import requests
    if not s.telegram_token:
        print("1) In Telegram, message @BotFather, send /newbot and pick a name for your bot.")
        print("2) BotFather replies with a token. Copy it.")
        token = getpass.getpass("3) Paste the token here (it stays hidden) and press Enter: ").strip()
        if not token:
            sys.exit("No token entered; nothing changed.")
        s.telegram_token = token
    me = requests.get(f"https://api.telegram.org/bot{s.telegram_token}/getMe", timeout=20).json()
    if not me.get("ok"):
        sys.exit(f"Telegram refused the token: {me.get('description')}. Copy it again from BotFather.")
    set_env("TELEGRAM_BOT_TOKEN", s.telegram_token)
    name = me["result"]["username"]
    print(f"Token OK. Now open https://t.me/{name} in Telegram, press Start (or send 'hi').")
    print("Waiting up to 3 minutes for your message...")
    chat = None
    for _ in range(36):
        r = requests.get(f"https://api.telegram.org/bot{s.telegram_token}/getUpdates", timeout=20).json()
        chats = [u["message"]["chat"] for u in r.get("result", []) if "message" in u]
        if chats:
            chat = chats[-1]
            break
        time.sleep(5)
    if chat is None:
        sys.exit(f"No message arrived. Send 'hi' to @{name}, then run this again.")
    set_env("TELEGRAM_CHAT_ID", chat["id"])
    s.telegram_chat_id = str(chat["id"])
    notify.Telegram(s, Store(s.db_path), log).send(
        "Connected. You will get a message on every buy and sell, and a daily summary. "
        "Send STATUS any time to see the account.")
    print(f"Connected to {chat.get('first_name') or chat.get('title')}. A test message was sent to your Telegram.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("universe").add_argument("--refresh", action="store_true")
    b = sub.add_parser("backfill")
    b.add_argument("--tf", nargs="*", default=list(backfill.TIMEFRAMES))
    sub.add_parser("scan")
    bt = sub.add_parser("backtest")
    bt.add_argument("--start")
    bt.add_argument("--end")
    bt.add_argument("--label", help="name for the experiment record in reports/experiments")
    Flags.add_cli(bt)
    sub.add_parser("ml").add_argument("--horizon", type=int, default=6)
    sub.add_parser("run")
    sub.add_parser("once")
    sub.add_parser("cmd").add_argument("text")
    sub.add_parser("status")
    sub.add_parser("watchdog")
    pt = sub.add_parser("pit", help="survivorship-free re-test")
    pt.add_argument("--top", type=int, default=30)
    pt.add_argument("--label", help="name for the experiment record in reports/experiments")
    Flags.add_cli(pt)
    sub.add_parser("reconcile", help="compare the database with the exchange (P0-C)")
    fr = sub.add_parser("flatten-residue",
                        help="sell untracked exchange balances blocking safe mode")
    fr.add_argument("--yes", action="store_true",
                    help="actually place the sell orders; without it this only reports")
    sub.add_parser("fleet", help="run every account from one engine (one scan, several books)")
    sub.add_parser("telegram", help="connect Telegram: finds your chat id and sends a test message")
    args = ap.parse_args(argv)
    s = config.load()

    if args.cmd == "universe":
        u = universe.build(s, refresh=args.refresh)
        print(f"{u['date']}: {len(u['symbols'])} names")
        for c in u["coins"]:
            print(f"  #{c['cg_rank']:<3} {c['symbol']:<10} vol ${c['volume_24h_usdt'] / 1e6:,.0f}M  "
                  f"history {c['history_days']} days{'' if c['live_eligible'] else '  (not live-eligible yet)'}")
        for d in u["dropped"]:
            print(f"  dropped #{d['cg_rank']} {d['coin']}: {d['reason']}")
    elif args.cmd == "backfill":
        u = universe.load(s)
        backfill.run(s, u["symbols"], args.tf, log=log)
        print(backfill.check(s, u["symbols"]).to_string(index=False))
    elif args.cmd == "scan":
        u = universe.load(s)
        _, bar_time, rows, cands = ranker.scan(s, u["symbols"])
        regime = bool(rows.set_index("symbol").loc["BTCUSDT", "regime_on"])
        print(f"Bar closed {bar_time} | BTC regime {'ON (risk-on)' if regime else 'OFF (no new alt entries)'}")
        from .features import latest_table
        print(latest_table(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print("\nTop 5 explained:\n  " + "\n  ".join(ranker.explain(rows)))
        print("\n" + notify.candidates_message(bar_time, regime, [r._asdict() for r in cands.itertuples()],
                                               s.signal_expiry_hours))
    elif args.cmd == "backtest":
        cmd_backtest(args, s)
    elif args.cmd == "ml":
        cmd_ml(args, s)
    elif args.cmd == "run":
        Bot(s).run_forever()
    elif args.cmd == "fleet":
        from .fleet import Fleet, default_members
        Fleet(s, default_members(s), log=log).run_forever()
    elif args.cmd == "once":
        bot = Bot(s)
        bot.on_bar_close(force=True)
        bot.handle_commands()
        print(bot.status_text())
    elif args.cmd == "cmd":
        store = Store(s.db_path)
        if notify.parse(args.text) is None:
            sys.exit(f"Not a valid command: {args.text!r}")
        store.add_command(args.text, "cli")
        print("queued; the running bot handles it within ~15 seconds")
    elif args.cmd == "status":
        print(Bot(s).status_text())
    elif args.cmd == "telegram":
        telegram_setup(s)
    elif args.cmd == "pit":
        cmd_pit(args, s)
    elif args.cmd == "reconcile":
        bot = Bot(s)
        rep = bot.boot_reconcile()
        if rep is None:
            sys.exit("reconciliation could not run; see the log above")
        print(rep.text())
        sys.exit(0 if rep.ok else 1)
    elif args.cmd == "flatten-residue":
        # Rounding residue accumulates: every full exit rounds the quantity down to the exchange's
        # lot step and leaves a crumb behind. 51 core round trips on 30 Oct left 0.000126 BTC and
        # 0.0020608 ETH. The ETH is stranded under the 5 USDT minimum notional, and reconciliation
        # now treats a balance that cannot be sold as dust - but the BTC WAS sellable, so it was a
        # real difference and it blocked every demo entry until a human cleared it.
        #
        # Which balances are sellable comes from reconcile's own walk (Report.sellable), not from
        # a second copy of the rule here. That is the lesson of the core churn.
        if s.mode == "paper":
            sys.exit("paper has no exchange, so there is nothing to flatten")
        bot = Bot(s)
        rep = reconcile.check(s, bot.broker, bot.store)
        if not rep.sellable:
            print(rep.text())
            print()
            print("No sellable untracked balance. Nothing to flatten.")
            sys.exit(0)
        total = sum(x["value"] or 0 for x in rep.sellable)
        print(f"{len(rep.sellable)} untracked balance(s) the exchange will accept an order for, "
              f"worth about {total:,.2f} USDT:")
        for x in rep.sellable:
            print(f"  SELL {x['qty']:.8g} {x['asset']}  (about {x['value']:,.2f} USDT)")
        if not args.yes:
            print()
            print(f"This places real sell orders on the {s.mode} account.")
            print("Re-run with --yes to do it.")
            sys.exit(0)
        sold = []
        for x in rep.sellable:
            sym = x["asset"] + "USDT"
            try:
                qty = bot.broker.filters(sym).round_qty(x["qty"])
                fill = bot.broker.sell(sym, qty, f"flat{int(time.time())}{x['asset']}")
                sold.append(f"{qty:.8g} {x['asset']} at {fill.get('price')}")
                log(f"flattened residue: SELL {qty:.8g} {x['asset']}")
            except Exception as e:
                log(f"could not flatten {x['asset']}: {e}")
        # Re-check rather than assume. Safe mode clears only if the exchange and the database now
        # agree about EVERYTHING, not merely about the balances this command touched.
        again = reconcile.check(s, bot.broker, bot.store)
        print()
        print(again.text())
        print()
        if again.ok:
            reconcile.clear_safe_mode(bot.store)
            print("Safe mode cleared. New entries are allowed again.")
        else:
            print("Safe mode left in place: something else still disagrees.")
        if sold:
            bot.tg.send("Flattened untracked residue: " + "; ".join(sold) + ". "
                        + ("Safe mode cleared." if again.ok else "Safe mode still active."))
        sys.exit(0 if again.ok else 1)
    elif args.cmd == "watchdog":
        hb = s.heartbeat_path
        age = time.time() - float(hb.read_text()) if hb.exists() else float("inf")
        if age > 45 * 60:
            notify.Telegram(s, Store(s.db_path), log).send(
                f"WATCHDOG: bot heartbeat is {age / 60:.0f} minutes old - the process may be down.")
            sys.exit(1)
        print(f"heartbeat {age / 60:.1f} minutes old: OK")


if __name__ == "__main__":
    main()
