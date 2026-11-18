"""What the forward test CAN measure: execution, not returns.

    python research/execution_quality.py
    python research/execution_quality.py --book demo --db backups/2026-11-06/state_demo...sqlite

WHY THIS EXISTS. The returns question is unanswerable on this timescale: 32.2% signal precision
against a 30.3% breakeven needs about 2,248 trades to separate at 95% confidence, which at the
observed 0.17 trades a day is roughly 35 years. P20 then showed that out of sample the entry timing
is a coin flip anyway (p_time 0.496 and 0.512) and the coin selection does not clear 0.05.

Execution is the opposite kind of question. Every fill is an observation, the demo book has 212 of
them against a real exchange, and the answers change decisions NOW:

  - is 0.10% per side the real fee, or does the exchange charge differently?
  - is 7 bps the real slippage, or is it 25?
  - the backtest assumes an order decided on a close fills at the NEXT BAR'S OPEN. What does the
    live book actually pay against that reference? That error is the gap between the backtest and
    reality, and nothing else in this repository measures it.
  - what does lot-step rounding cost per round trip, and is that material at 100 USD?

Each of those is answerable by February rather than by 2061, and each of them is an input the
backtest currently assumes rather than knows.

ONLY THE DEMO BOOK COUNTS. paper and paper100 fills are synthetic: PaperBroker applies
settings.slippage itself, so measuring slippage there would return the assumption back. Demo places
real signed orders against Binance Demo Trading and its fills come back with the exchange's own
price and commission, which is the whole reason that book exists.

A CAVEAT ON WHAT DEMO TRADING PROVES. Binance's demo venue is not the live order book: its fills
are generous and its liquidity is not real. Treat these numbers as a floor on cost rather than an
estimate of it. What they can establish with confidence is the fee model, the rounding loss, and
whether the live code paths price and size orders the way the backtest believes - and if any of
those is wrong here, it is wrong live too.
"""

import argparse
import json
import sqlite3
import statistics
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL_FEE = 0.0010          # settings.fee_rate
MODEL_SLIP_BPS = 7.0        # settings.slippage_bps
API = "https://api.binance.com/api/v3/klines"


def klines(symbol, interval, start_ms, end_ms):
    """Every candle in the window, paged. Public endpoint, no key."""
    out, cursor = [], start_ms
    while cursor < end_ms:
        url = f"{API}?symbol={symbol}&interval={interval}&startTime={cursor}&endTime={end_ms}&limit=1000"
        batch = json.load(urllib.request.urlopen(url, timeout=30))
        if not batch:
            break
        out.extend(batch)
        cursor = batch[-1][6] + 1
        if len(batch) < 1000:
            break
    return out


def fills(db):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = list(c.execute(
        "select time, side, symbol, qty, price, fee, reason, client_order_id from fills order by id"))
    return [dict(zip(("time", "side", "symbol", "qty", "price", "fee", "reason", "cid"), r))
            for r in rows]


def ms(iso):
    return int(datetime.fromisoformat(iso).timestamp() * 1000)


def section(title):
    print("\n" + "=" * 86)
    print(title)
    print()


def report_fees(rows):
    section("1. THE FEE MODEL. Is 0.10% per side what the exchange actually charges?")
    rates = []
    for f in rows:
        notional = f["qty"] * f["price"]
        if notional > 0 and f["fee"] is not None:
            rates.append(f["fee"] / notional)
    if not rates:
        print("  no priced fills")
        return
    med = statistics.median(rates)
    print(f"  {len(rates)} fills   modelled {MODEL_FEE:.4%}   "
          f"median {med:.4%}   mean {statistics.fmean(rates):.4%}   "
          f"min {min(rates):.4%}   max {max(rates):.4%}")
    off = (med - MODEL_FEE) / MODEL_FEE
    print(f"  the model is off by {off:+.2%} of itself -> "
          f"{'ACCURATE' if abs(off) < 0.02 else 'WRONG, and every backtest inherits it'}")
    zero = sum(1 for f in rows if not f["fee"])
    if zero:
        print(f"  WARNING: {zero} fills recorded a zero fee. A commission charged in BNB is "
              "recorded as zero (app/broker.py), so real costs would be understated.")


def _path_of(reason):
    r = (reason or "").lower()
    if r.startswith("live core"):
        return "live tick, core"
    if r.startswith("core"):
        return "bar close, core"
    if r.startswith("live:"):
        return "live tick, breakout"
    return "bar close, breakout"


def _is_churn(f):
    """The 30 Oct - 1 Nov core churn: 203 fills buying and selling the same coin each hour.

    They must be excluded from any price statistic. They bought and sold within the same bar at
    essentially the same price, so they sit on top of the bar's open by construction and would
    drag any median to zero - making a broken cost model look accurate. They are kept for the fee
    and rounding sections, where 203 real exchange commissions are exactly what is wanted.
    """
    return _path_of(f["reason"]) in ("bar close, core", "live tick, core") \
        and "2026-10-30T06:05" < f["time"] < "2026-11-02"


def report_slippage(rows, interval="4h"):
    section("2. FILL PRICE vs THE BACKTEST'S REFERENCE.\n"
            "   backtest.run fills an order decided on a close at the NEXT BAR'S OPEN.\n"
            "   This compares what the live book actually paid against that same reference,\n"
            "   split by which code path fired, because they are not the same question.")
    live = [f for f in rows if not _is_churn(f)]
    print(f"  {len(rows) - len(live)} churn fills excluded from this section "
          f"(same-bar round trips sit on the open by construction and would flatter the model)")
    print(f"  {len(live)} fills left\n")

    by_sym = defaultdict(list)
    for f in live:
        by_sym[f["symbol"]].append(f)

    per_path = defaultdict(list)
    detail = []
    for sym, fs in sorted(by_sym.items()):
        lo = min(ms(f["time"]) for f in fs) - 6 * 3600 * 1000
        hi = max(ms(f["time"]) for f in fs) + 6 * 3600 * 1000
        try:
            ks = klines(sym, interval, lo, hi)
        except Exception as e:                                   # noqa: BLE001
            print(f"  {sym}: could not fetch candles ({e})")
            continue
        bars = [(int(k[0]), int(k[6]), float(k[1]), float(k[4])) for k in ks]
        for f in fs:
            t = ms(f["time"])
            bar = next((b for b in bars if b[0] <= t <= b[1]), None)
            if bar is None or bar[2] <= 0:
                continue
            sign = 1 if f["side"] == "BUY" else -1
            bps = (f["price"] / bar[2] - 1.0) * sign * 1e4
            per_path[_path_of(f["reason"])].append(bps)
            detail.append((f["time"][:16], sym, f["side"], _path_of(f["reason"]),
                           f["price"], bar[2], bps))

    print(f"  {'when':17s} {'symbol':9s} {'side':5s} {'path':21s} {'paid':>11s} "
          f"{'bar open':>11s} {'vs ref':>9s}")
    for when, sym, side, path, px, ref, bps in detail:
        print(f"  {when:17s} {sym:9s} {side:5s} {path:21s} {px:11.4f} {ref:11.4f} {bps:+8.1f}b")

    print()
    print(f"  {'path':24s} {'n':>4s} {'median':>10s} {'mean':>10s} {'worst':>10s}")
    for path in sorted(per_path, key=lambda p: -len(per_path[p])):
        v = per_path[path]
        print(f"  {path:24s} {len(v):4d} {statistics.median(v):+9.1f}b "
              f"{statistics.fmean(v):+9.1f}b {max(v):+9.1f}b")

    bar_close = per_path["bar close, breakout"] + per_path["bar close, core"]
    tick = per_path["live tick, breakout"] + per_path["live tick, core"]
    print()
    if bar_close:
        m = statistics.median(bar_close)
        print(f"  BAR CLOSE fills, which is the convention the backtest models: "
              f"median {m:+.1f} bps against a modelled {MODEL_SLIP_BPS:+.1f}.")
        print(f"  -> the cost model is {'ACCURATE or conservative' if m <= MODEL_SLIP_BPS else 'OPTIMISTIC'} "
              "on the path it claims to describe.")
    if tick:
        m = statistics.median(tick)
        print(f"  LIVE TICK fills: median {m:+.1f} bps. This is not slippage, it is the price")
        print("  having already moved before the tick fired - the measured cost of deciding inside")
        print("  the bar, which the research log rejected on 4 Oct and the server ran anyway until")
        print("  6 Nov. DECISION_TF=4h since then, so this number should stop growing.")
    if bar_close and tick:
        print(f"\n  tick costs {statistics.median(tick) - statistics.median(bar_close):+.1f} bps "
              "more per fill than the bar close, measured on this book.")


def report_rounding(rows):
    section("3. LOT-STEP ROUNDING. What each round trip leaves stranded, and what that costs.")
    pos = defaultdict(float)
    bought = defaultdict(float)
    trips = defaultdict(int)
    for f in rows:
        if f["side"] == "BUY":
            pos[f["symbol"]] += f["qty"]
            bought[f["symbol"]] += f["qty"]
        else:
            pos[f["symbol"]] -= f["qty"]
            trips[f["symbol"]] += 1
    any_res = False
    for sym in sorted(pos):
        residue, n = pos[sym], trips[sym]
        if residue <= 1e-12 or not n:
            continue
        any_res = True
        per = residue / n
        print(f"  {sym:9s} {n:4d} round trips   left {residue:.10g} stranded   "
              f"{per:.3g} per trip   {residue / bought[sym]:.4%} of everything bought")
    if not any_res:
        print("  no residue: every position closed to zero")
        return
    print("\n  Residue is unsellable once it falls under the exchange's minimum notional, so it is")
    print("  a permanent loss, not a rounding wobble. At 100 USD with positions near the 5 USDT")
    print("  minimum this is the cost that scales worst, because the step is fixed and the")
    print("  position is small.")


def report_reasons(rows):
    section("4. WHAT ACTUALLY TRIGGERED EACH FILL. The live paths, counted.")
    kinds = defaultdict(int)
    for f in rows:
        kinds[_path_of(f["reason"])] += 1
    total = sum(kinds.values()) or 1
    for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]):
        print(f"  {k:24s} {v:5d}   {v / total:5.1%}")
    print("\n  A book on DECISION_TF=4h should show no 'live tick' rows for bars after the switch.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="", help="path to a book; default is the newest demo backup")
    # THE REFERENCE MUST MATCH THE CADENCE THAT WAS RUNNING, or this measures price drift
    # instead of execution. The demo book's fills all predate the 6 Nov switch back to 4h, when
    # the server ran LEVEL_TF=1h - so 1h is the correct reference for them and 4h is not. Scored
    # against 4h, a fill at 23:06 is compared with the 20:00 open and picks up three hours of
    # drift: it reported the cost model as OPTIMISTIC by +40 bps when on the right reference it is
    # conservative by 5. Use 4h for fills after 6 Nov 2026.
    ap.add_argument("--interval", default="1h",
                    help="candle interval for the price reference; must match the DECISION_TF "
                         "cadence in force when the fills happened (1h before 6 Nov 2026, 4h after)")
    ap.add_argument("--no-network", action="store_true", help="skip the price comparison")
    args = ap.parse_args(argv)

    db = args.db
    if not db:
        cands = sorted(ROOT.glob("backups/*/state_demo*.sqlite"))
        if not cands:
            sys.exit("no demo book found; pass --db")
        db = str(cands[-1])
    print(f"book: {db}")
    rows = fills(db)
    print(f"{len(rows)} fills")
    if not rows:
        sys.exit("nothing to measure")
    print(f"from {rows[0]['time']} to {rows[-1]['time']}")

    report_fees(rows)
    if not args.no_network:
        report_slippage(rows, args.interval)
    report_rounding(rows)
    report_reasons(rows)

    section("WHAT THIS DOES AND DOES NOT SETTLE")
    print("  Settles: the fee model, the rounding loss, which code path fired, and the size of the")
    print("  gap between live fills and the price the backtest assumes.")
    print()
    print("  Does not settle: whether the strategy makes money. That needs ~2,248 trades and P20")
    print("  found the entry timing is a coin flip out of sample. Nothing here changes that, and")
    print("  no amount of execution measurement will.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
