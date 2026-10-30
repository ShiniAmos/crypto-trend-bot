"""One engine, several accounts.

The three accounts used to be three processes, each downloading the same candles, building the same
panel and scoring the same coins. On a 1 GB server that is three times the memory for one piece of
thinking, and on 4 Oct it took the machine down.

The fleet does the work once per cycle and hands the result to each account:

    universe -> panel -> scores -> candidates        once
    live prices                                      once
    sizing, orders, risk limits, state               per account, independently

That is not only cheaper, it is more honest: the paper and demo accounts now provably see the same
signal at the same instant, so a difference between them is a difference in EXECUTION - minimum
order sizes, available cash, fills - which is exactly what the demo run is meant to measure. Three
separate processes could drift apart by a bar and nobody would know.

Each account keeps its own database, its own risk state and its own broker. One account hitting its
drawdown kill does not touch the others.

    python -m app.main fleet
"""

import time
import traceback
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import data, ranker, universe
from .config import Settings


def member_settings(base: Settings, account: str, mode: str, start_equity: float,
                    telegram: bool) -> Settings:
    """Settings for one account: its own name, mode and size, everything else inherited.

    Built through dataclasses.replace so the usual validation runs - a member asking for live mode
    without LIVE_TRADING_CONFIRMED still raises, exactly as a standalone bot would.
    """
    return replace(base, mode=mode, start_equity=start_equity, account_name=account,
                   shared_data=True,            # the fleet owns the download; members never fetch
                   telegram_token=base.telegram_token if telegram else "")


def default_members(settings):
    """The three accounts this project runs, and why each exists.

    paper   - the $10,000 record, the one the backtest is compared against
    small   - $100, where Binance minimum orders and rounding actually bite
    demo    - real signed orders against Binance Demo Trading, the dress rehearsal for live
    The demo account is the one that speaks on Telegram; two more voices for the same signal would
    only be noise.
    """
    members = [{"account": "paper", "mode": "paper", "start_equity": 10_000.0, "telegram": False},
               {"account": "paper100", "mode": "paper", "start_equity": 100.0, "telegram": False}]
    if settings.binance_key and settings.binance_secret:
        members.append({"account": "demo", "mode": "demo", "start_equity": 0.0, "telegram": True})
    else:
        members[0]["telegram"] = True     # no keys yet: the paper account reports
    return members


class Fleet:
    """Runs several Bots from one scan. Members are independent in money, shared in thinking."""

    def __init__(self, settings, members, log=print):
        from .main import Bot
        self.s = settings
        self.log = log
        self.bots = []
        for m in members:
            ms = member_settings(settings, m["account"], m["mode"], m.get("start_equity", 10_000.0),
                                 m.get("telegram", False))
            bot = Bot(ms)
            bot.name = m["account"]
            self.bots.append(bot)
        self.last_bar = None
        self._last_daily = None

    # ---------------------------------------------------------------- shared work
    def scan_once(self):
        """Universe, panel and candidates, computed once for every account.

        The universe is told about every position any account holds, so a coin one account is in is
        never dropped from the list underneath it.
        """
        held = set()
        for b in self.bots:
            held |= set(b.store.positions())
        u = universe.build(self.s, held_symbols=held)
        symbols = sorted(set(u["symbols"]) | held)
        flags = self.bots[0].flags if self.bots else None
        panel, bar_time, rows, cands = ranker.scan(self.s, symbols, flags=flags)
        return u, panel, bar_time, rows, cands

    def prices_once(self):
        """One quote per symbol for the whole fleet, instead of one request per account."""
        watch = set()
        for b in self.bots:
            lv = b.levels or b.store.get("levels") or {}
            watch |= set(lv.get("entry", {})) | set(b.store.positions())
        return data.prices(sorted(watch)) if watch else {}

    # ---------------------------------------------------------------- the cycle
    def on_bar(self, force=False):
        shared = self.scan_once()
        bar_time = shared[2]
        for b in self.bots:
            try:
                b.on_bar_close(force=force, shared=shared)
            except Exception:
                self.log(f"{b.name}: bar failed\n" + traceback.format_exc())
        self.last_bar = str(bar_time)

    def live(self):
        if not self.s.live_cycle:
            return
        prices = self.prices_once()
        if not prices:
            return
        for b in self.bots:
            try:
                b.live_tick(shared_prices=prices)
            except Exception:
                self.log(f"{b.name}: live tick failed\n" + traceback.format_exc())

    def daily(self):
        """Housekeeping once a day: the download the members no longer do for themselves."""
        from . import backfill, funding
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")
        if today == self._last_daily or (now.hour, now.minute) < (0, 5):
            return
        held = set()
        for b in self.bots:
            held |= set(b.store.positions())
        u = universe.build(self.s, refresh=True, held_symbols=held)
        backfill.run(self.s, u["symbols"], log=lambda *_: None)
        flags = self.bots[0].flags if self.bots else None
        if getattr(flags, "funding_max_pct", 0):
            funding.backfill(self.s, u["symbols"], log=lambda *_: None)
        self._last_daily = today
        self.log(f"daily: universe {len(u['symbols'])} names, history appended")
        for b in self.bots:
            # One book's summary must not cost the others theirs. Until 17 Oct a formatting error
            # in the demo book's summary raised out of daily() entirely; demo happened to be last
            # in the list so the other two had already sent, but in any other order they would
            # have been silently skipped every day. A report failing is never a reason to stop
            # reporting on everything else.
            if b.store.get("summary_day") != today:
                try:
                    b.tg.send(f"[{b.name}] " + b.daily_summary())
                    b.store.put("summary_day", today)
                except Exception:
                    self.log(f"{b.name}: daily summary failed\n" + traceback.format_exc())

    def telegram(self):
        for b in self.bots:
            b.tg.poll()
            b.handle_commands()

    def run_forever(self):
        names = ", ".join(f"{b.name}({b.s.mode})" for b in self.bots)
        # Each account's own flags are logged, not just the first one's. With per-account overrides
        # a single "flags" line would be actively misleading about what the other books are doing.
        self.log(f"fleet started: {names}")
        for b in self.bots:
            self.log(f"  {b.name}: {b.s.mode}, flags {b.flags.label}, "
                     f"max_names {b.max_names}")
        for b in self.bots:
            b.boot_reconcile()
            if self.s.live_cycle and not b.store.get("levels"):
                try:
                    b.on_bar_close(force=True)
                except Exception:
                    self.log(f"{b.name}: could not prime levels\n" + traceback.format_exc())
        for b in self.bots:
            b.tg.send(f"[{b.name}] started ({b.s.mode}).")

        offline_since = None
        while True:
            self.s.heartbeat_path.write_text(str(time.time()))
            failed = None
            for job in (self.daily, self.bar_if_due, self.live, self.telegram):
                try:
                    job()
                except Exception as e:
                    from .main import NETWORK_ERRORS
                    if isinstance(e, NETWORK_ERRORS):
                        failed = e
                    else:
                        self.log(f"{job.__name__} error:\n" + traceback.format_exc())
            if failed is not None and offline_since is None:
                offline_since = time.time()
                self.log(f"OFFLINE: cannot reach Binance/Telegram ({type(failed).__name__}). "
                         "Retrying every 5 seconds; positions are saved locally.")
            elif failed is None and offline_since is not None:
                mins = (time.time() - offline_since) / 60
                offline_since = None
                self.log(f"ONLINE again after {mins:.1f} minutes; missed checks were caught up.")
            pause = self.s.live_poll_seconds if self.s.live_cycle else 15
            time.sleep(5 if offline_since else pause)

    def bar_if_due(self):
        now = datetime.now(timezone.utc)
        last_closed = pd.Timestamp(now).floor(self.s.bar_tf)
        if self.last_bar != str(last_closed) and now - last_closed > timedelta(seconds=20):
            self.on_bar()

    def status(self):
        return "\n".join(f"[{b.name}] " + b.status_text() for b in self.bots)
