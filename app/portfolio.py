"""Durable state in SQLite so a reboot still knows positions, pending signals and risk limits."""

import json
import sqlite3
import time
from contextlib import contextmanager

SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
    symbol TEXT PRIMARY KEY, qty REAL, entry_price REAL, entry_time TEXT, stop REAL,
    cost REAL, reason TEXT, client_order_id TEXT, engine TEXT DEFAULT 'breakout');
CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT, symbol TEXT, side TEXT, qty REAL,
    price REAL, fee REAL, reason TEXT, equity REAL, client_order_id TEXT UNIQUE, pnl REAL);
CREATE TABLE IF NOT EXISTS pending_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT, bar_time TEXT, symbol TEXT, rank INTEGER, score REAL,
    reason TEXT, stop REAL, price REAL, expires_at REAL, status TEXT DEFAULT 'pending',
    UNIQUE(bar_time, symbol));
CREATE TABLE IF NOT EXISTS commands (
    id INTEGER PRIMARY KEY AUTOINCREMENT, received REAL, text TEXT, source TEXT,
    status TEXT DEFAULT 'new', reply TEXT);
CREATE TABLE IF NOT EXISTS equity (time TEXT PRIMARY KEY, equity REAL, cash REAL);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
-- P0-B: decisions taken on a bar close fill at the NEXT bar's open, exactly as the backtest does.
CREATE TABLE IF NOT EXISTS queued_orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT, decided_bar TEXT, side TEXT, symbol TEXT, qty REAL,
    stop REAL, reason TEXT, status TEXT DEFAULT 'queued', note TEXT, filled_bar TEXT,
    engine TEXT DEFAULT 'breakout',
    UNIQUE(decided_bar, side, symbol));
-- P0-D: entries that were dropped because the bot was not running when they fired.
CREATE TABLE IF NOT EXISTS skipped_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT, observed_at TEXT, bar_time TEXT, symbol TEXT,
    rank INTEGER, score REAL, stop REAL, expected_entry REAL, price_when_seen REAL,
    reason TEXT, downtime_minutes REAL, opportunity_cost REAL);
-- P0-A: the exchange-side stop that protects a live position between 4h checks.
CREATE TABLE IF NOT EXISTS stop_orders (
    symbol TEXT PRIMARY KEY, client_order_id TEXT, exchange_order_id TEXT, stop_price REAL,
    limit_price REAL, qty REAL, status TEXT, updated_at TEXT);
"""


# A holding worth less than this cannot be sold (exchange minimums start around 5 USDT), so it is
# treated as closed rather than left blocking a slot.
DUST_USDT = 1.0


class Store:
    def __init__(self, path):
        self.path = str(path)
        with self.tx() as db:
            db.executescript(SCHEMA)
            self._migrate(db)

    @staticmethod
    def _migrate(db):
        """Columns added after a book was already live.

        CREATE TABLE IF NOT EXISTS does nothing to an existing table, and the paper and demo books
        on the server were created before `engine` existed. Every row already there belongs to the
        breakout engine, which is what the DEFAULT gives them.
        """
        for table in ("positions", "queued_orders"):
            have = {r["name"] for r in db.execute(f"PRAGMA table_info({table})")}
            if "engine" not in have:
                db.execute(f"ALTER TABLE {table} ADD COLUMN engine TEXT DEFAULT 'breakout'")

    @contextmanager
    def tx(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    # key/value ----------------------------------------------------------------
    def get(self, key, default=None):
        with self.tx() as db:
            row = db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def put(self, key, value):
        with self.tx() as db:
            db.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (key, json.dumps(value)))

    # positions and fills ---------------------------------------------------------
    def positions(self):
        with self.tx() as db:
            return {r["symbol"]: dict(r) for r in db.execute("SELECT * FROM positions")}

    def record_buy(self, fill, stop, reason, engine="breakout"):
        """Open (or add to) a position and log the fill in one transaction.

        `engine` tags which strategy owns the position - "breakout" or "core". Exits are applied
        per engine, so this tag is what stops a Donchian low from closing a core holding. It is
        stored rather than inferred: deriving it from the reason text broke once the live path
        wrote a different prefix, and deriving it from the symbol is wrong for a coin the breakout
        engine bought before the core was switched on.
        """
        with self.tx() as db:
            if db.execute("SELECT 1 FROM fills WHERE client_order_id=?",
                          (fill["client_order_id"],)).fetchone():
                return False  # already recorded: retries must not double-count
            cost = fill["qty"] * fill["price"] + fill["fee"]
            old = db.execute("SELECT * FROM positions WHERE symbol=?", (fill["symbol"],)).fetchone()
            if old:
                qty = old["qty"] + fill["qty"]
                db.execute("UPDATE positions SET qty=?, cost=?, entry_price=? WHERE symbol=?",
                           (qty, old["cost"] + cost, (old["cost"] + cost) / qty, fill["symbol"]))
            else:
                db.execute(
                    "INSERT INTO positions (symbol, qty, entry_price, entry_time, stop, cost, "
                    "reason, client_order_id, engine) VALUES (?,?,?,?,?,?,?,?,?)",
                    (fill["symbol"], fill["qty"], fill["price"], fill["time"], stop, cost,
                     reason, fill["client_order_id"], engine))
            self._log_fill(db, fill, "BUY", reason, None)
        return True

    def update_stop(self, symbol, stop):
        """Ratchet a position's stop. Never moves it down (P0-A)."""
        with self.tx() as db:
            db.execute("UPDATE positions SET stop=? WHERE symbol=? AND ?>stop",
                       (float(stop), symbol, float(stop)))

    def record_sell(self, fill, reason):
        with self.tx() as db:
            if db.execute("SELECT 1 FROM fills WHERE client_order_id=?",
                          (fill["client_order_id"],)).fetchone():
                return None
            pos = db.execute("SELECT * FROM positions WHERE symbol=?", (fill["symbol"],)).fetchone()
            if not pos:
                return None
            qty = min(fill["qty"], pos["qty"])
            share = qty / pos["qty"]
            proceeds = qty * fill["price"] - fill["fee"]
            pnl = proceeds - pos["cost"] * share
            # Binance can take its fee in the coin itself, so a full exit sells slightly less
            # than was bought and leaves a sliver behind. On 8 Oct 2026 a real demo trade left
            # 0.0877 ADA - about 2 cents, far below the 5 USDT minimum order, so it could never
            # be sold again - and the book went on believing it held a position, occupying one of
            # only three slots indefinitely. A remainder too small to trade is not a position.
            left = pos["qty"] - qty
            if left <= pos["qty"] * 1e-6 or left * fill["price"] < DUST_USDT:
                db.execute("DELETE FROM positions WHERE symbol=?", (fill["symbol"],))
            else:  # partial fill: keep the rest open
                db.execute("UPDATE positions SET qty=?, cost=? WHERE symbol=?",
                           (pos["qty"] - qty, pos["cost"] * (1 - share), fill["symbol"]))
            self._log_fill(db, fill, "SELL", reason, pnl)
        return pnl

    @staticmethod
    def _log_fill(db, fill, side, reason, pnl):
        db.execute("INSERT INTO fills (time, symbol, side, qty, price, fee, reason, equity, "
                   "client_order_id, pnl) VALUES (?,?,?,?,?,?,?,?,?,?)",
                   (fill["time"], fill["symbol"], side, fill["qty"], fill["price"], fill["fee"],
                    reason, fill.get("equity"), fill["client_order_id"], pnl))

    def fills(self, limit=50):
        with self.tx() as db:
            return [dict(r) for r in db.execute(
                "SELECT * FROM fills ORDER BY id DESC LIMIT ?", (limit,))]

    def record_equity(self, when, equity, cash):
        with self.tx() as db:
            db.execute("INSERT OR REPLACE INTO equity VALUES (?,?,?)", (when, equity, cash))

    # approval gate -----------------------------------------------------------------
    def add_signal(self, bar_time, row, expires_at):
        with self.tx() as db:
            db.execute("INSERT OR IGNORE INTO pending_signals (bar_time, symbol, rank, score, "
                       "reason, stop, price, expires_at) VALUES (?,?,?,?,?,?,?,?)",
                       (bar_time, row["symbol"], int(row["rank"]), float(row["score"]),
                        row["reason"], float(row["stop_level"]), float(row["close"]), expires_at))

    def open_signals(self, now=None):
        now = now or time.time()
        with self.tx() as db:
            db.execute("UPDATE pending_signals SET status='expired' "
                       "WHERE status='pending' AND expires_at <= ?", (now,))
            return {r["symbol"]: dict(r) for r in db.execute(
                "SELECT * FROM pending_signals WHERE status='pending' ORDER BY rank")}

    def set_signal_status(self, status, symbol=None):
        with self.tx() as db:
            if symbol:
                db.execute("UPDATE pending_signals SET status=? WHERE status='pending' AND symbol=?",
                           (status, symbol))
            else:
                db.execute("UPDATE pending_signals SET status=? WHERE status='pending'", (status,))

    # queued orders: decided on one bar close, filled at the next bar's open (P0-B) --------
    def queue_order(self, decided_bar, side, symbol, qty, stop, reason, engine="breakout"):
        """Queue an order for the next open. `engine` rides along so a core buy queued in paper
        mode still fills tagged as the core's; without it the tag was assigned at fill time and
        every queued core entry became a breakout position with a Donchian exit."""
        with self.tx() as db:
            db.execute("INSERT OR IGNORE INTO queued_orders (decided_bar, side, symbol, qty, "
                       "stop, reason, engine) VALUES (?,?,?,?,?,?,?)",
                       (str(decided_bar), side, symbol, float(qty),
                        None if stop is None else float(stop), reason, engine))

    def due_orders(self):
        """Queued orders waiting for the next open. Sells first, so cash is free for buys."""
        with self.tx() as db:
            return [dict(r) for r in db.execute(
                "SELECT * FROM queued_orders WHERE status='queued' "
                "ORDER BY CASE side WHEN 'SELL' THEN 0 ELSE 1 END, id")]

    def finish_order(self, oid, status, note=None, filled_bar=None):
        with self.tx() as db:
            db.execute("UPDATE queued_orders SET status=?, note=?, filled_bar=? WHERE id=?",
                       (status, note, str(filled_bar) if filled_bar else None, oid))

    def drop_queued(self, symbol=None, side=None):
        """Cancel still-queued orders, e.g. when a manual command overtakes them."""
        q = "UPDATE queued_orders SET status='cancelled' WHERE status='queued'"
        args = []
        if symbol:
            q, args = q + " AND symbol=?", args + [symbol]
        if side:
            q, args = q + " AND side=?", args + [side]
        with self.tx() as db:
            return db.execute(q, args).rowcount

    # skipped entries (P0-D) ---------------------------------------------------------------
    def record_skipped(self, row):
        with self.tx() as db:
            db.execute("INSERT INTO skipped_signals (observed_at, bar_time, symbol, rank, score, "
                       "stop, expected_entry, price_when_seen, reason, downtime_minutes, "
                       "opportunity_cost) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                       (row["observed_at"], row["bar_time"], row["symbol"], row.get("rank"),
                        row.get("score"), row.get("stop"), row.get("expected_entry"),
                        row.get("price_when_seen"), row["reason"], row.get("downtime_minutes"),
                        row.get("opportunity_cost")))

    def skipped(self, limit=100):
        with self.tx() as db:
            return [dict(r) for r in db.execute(
                "SELECT * FROM skipped_signals ORDER BY id DESC LIMIT ?", (limit,))]

    # exchange stop orders (P0-A) ----------------------------------------------------------
    def put_stop(self, symbol, **row):
        with self.tx() as db:
            db.execute("INSERT OR REPLACE INTO stop_orders (symbol, client_order_id, "
                       "exchange_order_id, stop_price, limit_price, qty, status, updated_at) "
                       "VALUES (?,?,?,?,?,?,?,?)",
                       (symbol, row.get("client_order_id"), str(row.get("exchange_order_id") or ""),
                        row.get("stop_price"), row.get("limit_price"), row.get("qty"),
                        row.get("status"), row.get("updated_at")))

    def stops(self):
        with self.tx() as db:
            return {r["symbol"]: dict(r) for r in db.execute("SELECT * FROM stop_orders")}

    def clear_stop(self, symbol):
        with self.tx() as db:
            db.execute("DELETE FROM stop_orders WHERE symbol=?", (symbol,))

    def add_command(self, text, source):
        with self.tx() as db:
            db.execute("INSERT INTO commands (received, text, source) VALUES (?,?,?)",
                       (time.time(), text, source))

    def new_commands(self):
        with self.tx() as db:
            return [dict(r) for r in db.execute("SELECT * FROM commands WHERE status='new' ORDER BY id")]

    def finish_command(self, cid, status, reply):
        with self.tx() as db:
            db.execute("UPDATE commands SET status=?, reply=? WHERE id=?", (status, reply, cid))
