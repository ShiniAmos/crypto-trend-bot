# Crypto-bot: code audit

First audited 25 September 2026 against the working tree. **Updated the same day** after the P0
correctness work and the first P1 strategy change. Every claim cites a file and line.
Companions: `ALGORITHMS.md` (how each rule works), `METRIC_MATRIX.md` (every computed number),
`RESEARCH_LOG.md` (what was measured, and what was rejected).

Size now: 4,006 lines of Python across 21 modules, 64 tests, 4 commits, 10 experiment records.

---

## 1. Executive snapshot

A **single-strategy, long-only spot momentum rotation bot** on Binance, driven entirely by closed
4-hour candles fetched over REST. It ranks a daily-rebuilt universe by a four-part percentile
score, buys Donchian breakouts among the top 5 while Bitcoin is above its 200-day average and the
coin is rising on its own, and exits on a 20-bar low, a rank drop past 10, or a regime flip.
Position size comes from risk, not conviction. State lives in SQLite; alerts and commands run over
Telegram.

**Modes it can safely run today:** backtest and paper — both exercised continuously and covered by
tests. `demo` and `live` are implemented but **have still never touched Binance**: every
Binance-client test is monkeypatched against fake responses (`tests/test_bot.py:220-259`). Live
additionally needs `LIVE_TRADING_CONFIRMED=true` as a second opt-in (`app/config.py:77-81`).

**Honesty score (docs vs code): 9.5/10**, up from 8.5. The three discrepancies found in the first
audit are fixed: the paper fill model now matches the backtest, the website no longer mislabels the
entry-time stop as the sell trigger, and the orphan metrics are documented as such. Remaining
deduction: the Dockerfile still ships only `app/`, so the container cannot run the website.

---

## 2. What changed on 25 September 2026

Four commits, from `f6ffdaf` (frozen baseline) to `70ed81e`. The repository had **no commits at
all** before that day, so there was no rollback point; there is now.

| Item | Status | Evidence |
|---|---|---|
| **P0-B** paper fills at the next bar's open, like the backtest | **SHIPPED** | `app/main.py:fill_due_orders`; verified live: fill 66.6166 = open 66.5700 × (1+slippage) |
| **P0-C** boot reconciliation and safe mode | **SHIPPED** | `app/reconcile.py`; passes on the server at every start |
| **P0-D** skipped entries recorded, not discarded | **SHIPPED** | `skipped_signals` table + dashboard panel |
| **P0-A** intrabar stop, ratchet-up, exchange STOP_LOSS_LIMIT | **built, flag OFF** | failed the ship rule on both panels |
| **P1-E** absolute momentum (`ret_30d > 0`) | **SHIPPED, live in paper** | passes standard, pit30 and the held-out half |
| **P1-H** volume filter at 0.8 | **SHIPPED, live in paper** | pit30 7.08% → 7.84%, p_coin 0.022 → 0.012 |
| **P3-N** taker ratio ≥ 0.5 | **SHIPPED, live in paper** | pit30 7.84% → 10.18%, Sharpe 0.697 → 0.880 |
| P1-G breadth, P1-F BTC sleeve | **rejected** | breadth −8 CAGR pts; sleeve DD −15% → −32% |
| P2 scale-outs, vol target, correlation cap | **rejected** | win rate up, return down; see the log |
| P3 walk-forward weights, ML veto, ATR band, late breakout | **rejected** | aggregate out-of-sample worse |
| `rs_cap`, `btc_momentum`, `abs+btc` combined | **rejected** | `RESEARCH_LOG.md` decision table |

**The one strategy change that shipped**, with its evidence:

| Panel | CAGR | Sharpe | Max drawdown |
|---|---|---|---|
| Standard, 9.1 years | 20.24% → **20.67%** | 1.291 → **1.367** | −20.43% → **−16.33%** |
| Survivorship-free (pit30) | 5.73% → **7.08%** | 0.502 → **0.607** | −29.33% → **−22.58%** |
| Held-out 2022-2026 | 12.09% → **15.07%** | 0.891 → **1.100** | −24.30% → **−16.33%** |

**Flag-off parity is exact**: the equity curve produced with every flag off is byte-identical to
the curve saved from `f6ffdaf` before any edit (`reports/_parity_baseline_equity.csv`).

---

## 3. What the bot can perform today

### Trading

- Rebuild a tradable universe daily from CoinGecko's top 40, filtered to Binance spot pairs with
  real volume and history (`app/universe.py:53`). 19 coins pass today.
- Score every eligible coin on each closed 4-hour bar and rank them cross-sectionally
  (`app/features.py:93`).
- Buy breakouts that pass four conditions plus the shipped absolute-momentum filter, sized so a
  normal exit costs 1% of the account, capped at 30% per name and 3 names
  (`app/signals.py:23`, `app/risk.py:7`).
- Exit automatically on five grounds, in every mode including live (`app/main.py:102-119`).
- Execute paper fills at the next bar's open ± 7 bps, charging 0.1% each way — the same model the
  research uses (`app/broker.py:52`, `app/backtest.py:113`).
- Queue, cancel and expire orders safely: a buy queued before downtime is dropped and recorded; a
  late sell still executes (`app/main.py:fill_due_orders`).
- Run two accounts side by side on one price store: the $10,000 paper account and a $100 wallet
  that exposes minimum-order effects (`app/config.py:64-66`).

### Protection

- Daily 3% loss halt, 8% drawdown kill, `KILL` file, `FLAT ALL` (`app/risk.py:89-110`).
- Reconcile the database against exchange balances, open orders and stops at every start; on a
  mismatch it logs, alerts, enters safe mode and blocks entries while still allowing exits
  (`app/reconcile.py:check`).
- Idempotent orders: deterministic `clientOrderId`, GET-before-POST, no blind POST retry, unique
  constraint on fills (`app/broker.py:21,160`, `app/portfolio.py:14`).
- Rest a `STOP_LOSS_LIMIT` at the exchange and ratchet it upward — built and unit-tested, currently
  switched off (`app/broker.py:place_stop`, `app/main.py:ratchet_stops`).
- Survive an internet outage: retry every 5 seconds, switch to Binance's backup host, catch up
  missed checks, and deliver queued Telegram messages afterwards.

### Research

- `backtest` — 9 years, next-open fills, full metrics, benchmarks, random baselines.
- `pit` — rebuilds the universe as it was on each historical day from all 641 ever-listed USDT
  pairs, including delisted ones. This is the honest panel.
- `--enable-*` flags with experiment records: every run writes JSON + a CSV row carrying flags, git
  commit, seed, costs, 16 metrics and a regime breakdown (`app/experiment.py`).
- Regime attribution from causal BTC data only (`app/regime.py`).
- `research/trade_study.py` — what separates winning trades from losing ones, discovering on
  2017-2021 and validating on 2022-2026.
- `ml` — walk-forward gradient boosting with a purge gap, measured and rejected.
- 64 tests, including lookahead, execution timing, reconciliation, idempotency and flag parity.

### Watching it

- Password-protected website on port 8601: account, positions with the **live** exit trigger,
  trades, missed-entry log, history test, live Binance stream.
- Telegram on every buy and sell, a daily summary, and the `TRADE / FORCE / SKIP / FLAT / STATUS`
  command set.
- Heartbeat file plus a 30-minute cron watchdog that alerts if the loop dies.

### What it cannot do

Predict a price. Trade futures, margin or leverage. Move or withdraw money. Short. React between
4-hour checks (unless the stop flag is on, and even then only in demo/live). Place a real order —
because no key has ever been configured.

---

## 4. Feature matrix

| Area | Status | Files | Trigger | Tests |
|---|---|---|---|---|
| config + flags | IMPLEMENTED | `app/config.py`, `app/flags.py` | import | parity, digest |
| universe | IMPLEMENTED | `app/universe.py` | daily 00:05 UTC | offline fallback |
| backfill / klines | IMPLEMENTED | `app/backfill.py`, `app/data.py` | daily + each scan | host failover, closed-candle |
| websocket ingest | MISSING (server) / browser only | `ui/live_market.py` | page open | none |
| features + score | IMPLEMENTED | `app/features.py` | each scan | lookahead (2) |
| signals + P1 filters | IMPLEMENTED | `app/signals.py` | each bar | 7 filter tests |
| risk / sizing | IMPLEMENTED | `app/risk.py` | each entry | sizing, filters, limits |
| broker (paper) | IMPLEMENTED | `app/broker.py:31` | each order | fill model matches backtest |
| broker (demo/live) | IMPLEMENTED, NEVER RUN | `app/broker.py:75` | live/demo only | fakes only |
| exchange stop | IMPLEMENTED, FLAG OFF | `app/broker.py:place_stop` | after entry | intrabar, ratchet, gap |
| queued execution | IMPLEMENTED | `app/main.py:fill_due_orders` | each bar | lifecycle, stale, ordering |
| reconciliation | IMPLEMENTED | `app/reconcile.py` | boot + CLI | 4 cases |
| portfolio/state | IMPLEMENTED | `app/portfolio.py` | every write | dedup, partial sell |
| notify/telegram | IMPLEMENTED | `app/notify.py` | every ~15 s | parsing, outbox |
| approval gate | IMPLEMENTED | `app/main.py:126` | live mode | 5 tests |
| catch-up + skip log | IMPLEMENTED | `app/main.py:165` | on restart | 3 tests |
| backtest / pit / ml | IMPLEMENTED | `app/backtest.py`, `pit.py`, `ml.py` | CLI | mechanics (4) |
| experiments / regimes | IMPLEMENTED | `app/experiment.py`, `app/regime.py` | each run | — |
| dashboard | IMPLEMENTED | `dashboard.py`, `ui/` | systemd | none |

---

## 5. Gaps that still exist

1. **No order has ever reached Binance**, not even the demo environment. Every broker test uses
   fakes (`tests/test_bot.py:220-259`).
2. **No exchange reconciliation has ever run against a real account** — the code is tested against a
   fake broker only.
3. **The stop is off**, so between 4-hour checks a position is unprotected. Measured cost of
   turning it on: −1.1 points of CAGR on the standard panel, +4.0 points of drawdown on pit30.
4. **Entry timing is not skill.** On the survivorship-free panel the strategy still fails to beat
   random entry timing (p = 0.344 with the new filter, 0.506 before). The edge is in coin
   selection (p = 0.022), not in the moment.
5. **Dockerfile ships only `app/`** (`Dockerfile:5`), so the container cannot run the website, and
   no volume is declared for `data/`.
6. **Recent years are weaker.** 2017-2021 returned 26.7% a year against 19.9% over the last three,
   and the survivorship-free panel showed 2024 and 2025 flat to negative.
7. **One server, one region.** Oracle Mumbai, free tier, no backup host, no alerting if the VM
   itself disappears.
8. **`don_pos` can divide by zero** on a flat channel; ML path only (`app/features.py:31`).
9. **`dollar_vol_30d` is still dead code** (`app/features.py:40`).
10. **Add-to-position code remains unreachable** (`app/portfolio.py:69-73`).

---

## 6. Risk of going live

| Check | Status | Evidence |
|---|---|---|
| Keys never in source | GREEN | `app/config.py:21-27`; keys empty on both machines |
| Live needs a second opt-in | GREEN | `LIVE_TRADING_CONFIRMED` (`app/config.py:77-81`) |
| Human gate on entries | GREEN | `app/main.py:42`; silence means no order |
| Order idempotency | GREEN | `app/broker.py:21,160`; lost-response test |
| Durable state across reboot | GREEN | SQLite + `Restart=always` |
| **State reconciled against the exchange** | **AMBER** (was RED) | `app/reconcile.py` exists and runs; never tested against a real account |
| Kill switch | GREEN | `KILL` file, `FLAT ALL`, drawdown kill |
| Closed-candle rule | GREEN | `app/data.py:85-88` + lookahead tests |
| Paper matches research execution | **GREEN** (was RED) | verified to six decimal places |
| Exchange filter compliance | GREEN | `app/risk.py:61-68` |
| Rate limits, host failover, time drift | GREEN | tested |
| **Stop enforced between bars** | **RED** | built, switched off; see gap 3 |
| **Demo week completed** | **RED** | never run |

**Verdict: still not ready for real money.** Two reds remain, one structural and one procedural.
Paper mode is sound and now measures the same thing the research does.

**Before any real money, in order:** run the demo environment for two to four weeks with no code
changes; decide the stop question explicitly, accepting its measured cost; reconcile against that
demo account and watch safe mode trigger at least once deliberately; then reconsider, with a
withdrawal-disabled key, one coin, and an amount that can be lost entirely.

---

## 7. File index

| Path | Responsibility |
|---|---|
| `app/config.py` | Settings from `.env`, mode validation, per-account paths |
| `app/flags.py` | Research feature flags, default off, CLI and `.env` wiring |
| `app/universe.py` | Daily authenticated coin list and its filters |
| `app/data.py` | Binance REST, retries, host failover, parquet kline store |
| `app/backfill.py` | Resumable history download and integrity check |
| `app/features.py` | Per-bar metrics, cross-section, regime, score |
| `app/signals.py` | Entry rule, optional P1 filters, exit reason |
| `app/ranker.py` | Latest-bar scan and plain-words explanation |
| `app/risk.py` | Sizing, exchange filters, daily and drawdown limits |
| `app/broker.py` | Paper fills; signed Binance client; resting stop orders |
| `app/portfolio.py` | SQLite: positions, fills, signals, commands, equity, queued orders, skips, stops |
| `app/reconcile.py` | Database versus exchange, safe mode |
| `app/main.py` | 4h cycle, queued execution, catch-up, commands, loop, CLI |
| `app/backtest.py` | Historical simulation, metrics, benchmarks, random baselines |
| `app/pit.py` | Survivorship-free universe reconstruction and re-test |
| `app/regime.py` | Causal market-regime labels and per-regime metrics |
| `app/experiment.py` | Experiment records and the ship-rule decision table |
| `app/ml.py` | Walk-forward model (research only, not wired) |
| `app/supervisor.py` | Windows process babysitter |
| `research/trade_study.py` | Winner-versus-loser study, discovery and validation halves |
| `dashboard.py`, `ui/` | Password-gated read-only website |
| `tests/` | 64 tests: baseline behaviour, P0 correctness, P1 filters |
| `deploy/` | systemd units, watchdog cron, logrotate, VPS installer, packer |

---

## 8. Data stores

| Store | Path | Holds |
|---|---|---|
| SQLite per account | `data/state_paper.sqlite`, `state_paper100.sqlite` | positions, fills, pending signals, commands, equity, **queued_orders**, **skipped_signals**, **stop_orders**, kv |
| Parquet | `data/klines/SYMBOL_TF.parquet` (85 MB) | 1d/4h/1h OHLCV, quote and taker-buy volume |
| Parquet (research) | `data/pit/klines/` (239 MB) | every ever-listed USDT pair |
| JSON | `data/universe.json` | today's list, metadata, drop reasons |
| JSON + CSV | `reports/experiments/` | one record per backtest, plus `index.csv` |
| CSV | `reports/*.csv` | trades, fill logs, equity curves, trade study |

Secrets live only in `.env`: `MODE`, Binance key and secret (both empty), Telegram token and chat
id, `DASHBOARD_PASSWORD`, `LIVE_TRADING_CONFIRMED`, the risk overrides and the `ENABLE_*` flags.

---

## 9. Where it runs

Oracle Cloud Always Free, Ubuntu 24.04, Mumbai, 1 GB plus 2 GB swap, at `<YOUR_SERVER_IP>`.
Three systemd services — the $10,000 paper account, the $100 account and the website on port 8601 —
all `Restart=always`, plus a 30-minute watchdog cron. Confirmed running with
`flags abs_momentum` and reconciliation passing.
