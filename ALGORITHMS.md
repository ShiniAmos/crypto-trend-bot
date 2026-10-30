# Algorithms

Every algorithm in this repository, as implemented on 25 September 2026. Pseudocode follows the
real control flow; line references are exact. Companion documents: `METRIC_MATRIX.md` (every
computed number) and `AUDIT.md` (what is implemented vs tested vs live-used).

Status key: **LIVE** = runs in the trading loop. **RESEARCH** = reachable only from the CLI.
**DEAD** = present but nothing calls it.

**What is switched on right now** (server, 4 Oct 2026): `MODE=paper`, `DECISION_TF=4h`,
`AUTO_TRADE=false`, and three entry filters — `ENABLE_ABS_MOMENTUM`, `ENABLE_VOLUME_MIN=0.8`,
`ENABLE_TAKER_MIN=0.5`. Everything else is off. Two switches exist but are not engaged:
`DECISION_TF=1h` (decide hourly; measured as worse) and `AUTO_TRADE=true` (live entries without
the Telegram approval step).

| # | Algorithm | Status | Entry point |
|---|---|---|---|
| 1 | Universe construction | LIVE | `app/universe.py:53` |
| 2 | Kline backfill and integrity | LIVE | `app/backfill.py:10` |
| 3 | Feature computation | LIVE | `app/features.py:12` |
| 4 | Cross-sectional scoring (the ranker) | LIVE | `app/features.py:93` |
| 5 | Entry rule | LIVE | `app/signals.py:6` |
| 6 | Exit rule | LIVE | `app/signals.py:20` |
| 7 | Position sizing | LIVE | `app/risk.py:7`, `app/risk.py:16` |
| 8 | Exchange filter rounding | LIVE | `app/risk.py:40` |
| 9 | Portfolio circuit breakers | LIVE | `app/risk.py:89`, `app/risk.py:104` |
| 10 | The 4-hour cycle | LIVE | `app/main.py:78` |
| 11 | Downtime catch-up | LIVE | `app/main.py:165` |
| 12 | Approval gate (live mode) | LIVE | `app/main.py:126`, `app/main.py:207` |
| 13 | Order idempotency | LIVE | `app/broker.py:21`, `app/broker.py:160` |
| 14 | Paper fill model | LIVE | `app/broker.py:52` |
| 15 | Backtest engine | RESEARCH | `app/backtest.py:43` |
| 16 | Random baselines | RESEARCH | `app/backtest.py:235` |
| 17 | Survivorship-free universe | RESEARCH | `app/pit.py:73` |
| 18 | Walk-forward ML model | RESEARCH | `app/ml.py:39` |
| 19 | Exit variants (two, both rejected) | RESEARCH | `app/backtest.py:43` params |
| 20 | Add-to-position merge | DEAD | `app/portfolio.py:69` |

---

## 1. Universe construction

**Where:** `app/universe.py:53` (`build`). **When:** daily at 00:05 UTC (`app/main.py:303`), and on
demand via `python -m app.main universe --refresh`.

```
if universe.json exists and its date == today and not refresh:
    return it                                     # two runs on one day agree
fetch Binance exchangeInfo (SPOT), 24h tickers, CoinGecko top CG_TOP(40)
if any fetch fails:
    return yesterday's file if it exists, else raise
for coin in CoinGecko order (market cap descending):
    drop if stablecoin        # name in STABLES, or id contains "usd" and 0.97 < price < 1.03
    drop if wrapped/staked    # name in WRAPPED, or id starts with wrapped-/staked-
    drop if meme              # name in MEMES, when EXCLUDE_MEMES
    drop if leveraged         # regex ^[A-Z]{3,}(UP|DOWN|BULL|BEAR)$ or [0-9][LS]$
    drop if no Binance USDT spot pair with status TRADING and isSpotTradingAllowed
    drop if 24h quoteVolume < MIN_VOLUME_USDT (20M)
    keep with: symbol, cg_rank, volume, first_candle, history_days,
               live_eligible = history_days >= MIN_HISTORY_DAYS (90)
    stop at UNIVERSE_SIZE (30)
if BTCUSDT was dropped: raise          # the regime and all RS depend on it
write data/universe.json {date, built_at, symbols, coins, dropped}
```

Filter constants: `STABLES` 28 names, `WRAPPED` 24 names, `MEMES` 19 names
(`app/universe.py:16-23`). `live_eligible` is advisory metadata — the 90-day rule is enforced at
trade time in live mode only (`app/main.py:233`), while the panel enforces its own 540-bar age gate
for every mode (`app/features.py:88`).

A held coin that falls off the list is sold on the next cycle (`app/main.py:104`), and
`tradable_status` re-checks Binance status per held name every bar (`app/universe.py:116`).

---

## 2. Kline backfill and integrity

**Where:** `app/backfill.py:10` (`backfill_symbol`), `app/backfill.py:35` (`check`).

```
start = last stored open_time + one interval, or 0 for a fresh symbol
loop:
    page = GET /api/v3/klines(symbol, tf, startTime=start, limit=1000)
    page = page with close_time < now_ms          # unfinished candle discarded
    stop if empty
    start = last open_time in page + one interval
    stop if len(page) < 1000
save = concat(old, pages) -> drop_duplicates(open_time, keep=last) -> sort -> parquet
```

Writes go to a per-process temp file then `replace()`, retried 5 times on Windows lock errors
(`app/data.py:111-121`), so two accounts can never corrupt one file. Timeframes: `1d`, `4h`, `1h`
(`app/backfill.py:7`); only `1d` and `4h` feed decisions (`app/ranker.py:9`).

`check` reports per symbol: bar count, first date, last close, duplicate count, and gaps, where a
gap is any `open_time.diff() != interval` (`app/backfill.py:43`). Gaps are **reported, never
filled** — missing exchange outages stay missing.

**Transport** (`app/data.py:29`): two public hosts (`api.binance.com`,
`data-api.binance.vision`) alternated per attempt, 6 tries, exponential backoff from 1 s, with
`Retry-After` honoured on 429/418, retries on 5xx, and host rotation on 451/403 region blocks.

---

## 3. Feature computation

**Where:** `app/features.py:12` (`symbol_features`), `app/features.py:62` (`build_panel`),
`app/features.py:76` (`add_cross_section`).

Per symbol, on 4h bars, with the daily frame joined as-of:

```
bar_time = open_time + 4h                       # scheduled close, not close_time
own-past metrics: returns, Donchian, ATR, realised vol, volume and taker ratios
daily SMA50 and SMA200 merged backward by close_time   # a 4h bar sees only closed daily bars
```

Then across symbols at each `bar_time`:

```
join BTC's ret_1d, ret_7d, ret_30d, dist_sma200 onto every row
rs_h = ret_h - btc_ret_h            for h in 1d, 7d, 30d
regime_on = btc_dist_sma200 > 0
eligible  = age_bars >= 540 and rs_30d, dist_sma50, vol_30d all present
```

**Lookahead control.** Donchian bounds are `.shift(1)`ed so a breakout compares this close with the
*previous* 20 bars (`app/features.py:26-27`); the daily join is `direction="backward"`
(`app/features.py:49`); unfinished candles never enter storage (`app/data.py:85`). Asserted by
`tests/test_bot.py:69` (features for old bars do not change when future bars arrive) and
`tests/test_bot.py:81`.

Full formulas: `METRIC_MATRIX.md`.

---

## 4. Cross-sectional scoring — the ranker

**Where:** `app/features.py:93` (`score`). Recomputed on every closed bar for the whole panel.

```
within each bar_time, over eligible rows only:
    a = percentile_rank(rs_7d)                  weight 0.35
    b = percentile_rank(rs_30d)                 weight 0.25
    c = percentile_rank(dist_sma50)             weight 0.20
    d = percentile_rank(ret_30d / vol_30d)      weight 0.20
score = 100 * (0.35a + 0.25b + 0.20c + 0.20d)
rank  = descending score, ties broken by index order (method="first")
n_eligible = count of eligible rows on that bar
```

Every component is a **percentile inside that bar's eligible set**, so the score answers "how does
this coin compare with the others today", not "how good is this number". Consequences worth
knowing:

- Weights are hardcoded in one expression (`app/features.py:104`); they are not in `.env`.
- With N eligible coins the best possible percentile is 1.0, so the top score is 100 only when one
  coin leads all four components.
- Part *d* divides by `vol_30d`; a coin with near-zero realised volatility would dominate, but the
  540-bar eligibility gate and real crypto volatility make that unreachable in practice.
- `n_eligible` varies by day; on days when fewer than 5 coins are eligible the candidate list is
  correspondingly shorter.

**Explainability.** `app/ranker.py:17` (`explain`) renders the top names as sentences, and the same
numbers are stored per trade in the `reason` column (`app/main.py:156`), so any past decision can be
reconstructed from the fills table plus the parquet history.

---

## 5. Entry rule

**Where:** `app/signals.py` (`entry_candidates` and `extra_filters`). Applied to one bar at a time;
if a whole panel is passed, the ranking is computed inside each bar, never across the frame.

**As it runs today — seven conditions, all on the same closed bar:**

```
candidates = rows where
        eligible                  # 90 days of history, all inputs present
    and regime_on                 # BTC above its 200-day SMA
    and breakout                  # close > prior 20-bar high
    and rs_7d > 0                 # beat BTC over 7 days
    and rank <= CANDIDATES (5)
   OR (symbol == BTCUSDT and regime_on and breakout and rank <= CANDIDATES)
                                  # BTC is exempt from the RS test: its own RS is zero
  AND the shipped filters:
        ret_30d > 0               # ENABLE_ABS_MOMENTUM   up on its own, not just vs a falling BTC
    and volume_ratio >= 0.8       # ENABLE_VOLUME_MIN     not a breakout on a dead day
    and taker_ratio >= 0.5        # ENABLE_TAKER_MIN      buyers lifting the offer, not sellers
sorted by rank ascending
```

The three named filters are feature flags, off by default in code and switched on in `.env`. Each
was measured on the standard panel, the survivorship-free panel and a held-out period before being
enabled; twenty other candidates were rejected. See `RESEARCH_LOG.md`.

BTC is exempted from the RS test because its own relative strength is identically zero
(`app/signals.py:16`).

The loop then applies portfolio-level gates before any order (`app/main.py:122-159`):

```
drop candidates no longer in today's universe
if can_enter fails (kill file / drawdown kill / daily halt / at MAX_NAMES): no entries
slots = MAX_NAMES - open positions
for each candidate in rank order, excluding symbols already held, up to slots:
    size it, then buy
```

The same rule runs vectorised in the backtest (`app/backtest.py:51-52`), which is what keeps the
simulation and the live loop honest with each other.

---

## 6. Exit rule

**Where:** `app/signals.py:20` (`exit_reason`), called from `app/main.py:114` and
`app/backtest.py:171`.

```
exit_reason(regime_on, breakdown, rank):
    if not regime_on:            return "BTC regime flipped off"
    if breakdown:                return "broke the 20-bar low"      # close < prior 20-bar low
    if rank is missing or > EXIT_RANK (10):
                                 return "fell out of the top 10"
    return None                                                      # keep holding
```

`on_bar_close` evaluates a wider ladder, first match wins (`app/main.py:102-119`):

```
1  symbol no longer in today's universe        -> "dropped off the authenticated list"
2  Binance status != TRADING                   -> "Binance status is not TRADING"
3  drawdown kill active, or data/KILL exists    -> "kill switch"
4  a sell signal from a bar missed while offline-> that reason, marked as caught up
5  exit_reason on this bar                      -> regime / breakdown / rank
```

There is no profit target and no time stop. The exit level is the rolling 20-bar low, so it **moves
in both directions**: it rises as old lows leave the window and falls when a new lower low enters
it (measured on live data: TAO's fell on 5 of 60 consecutive bars). An earlier version of this
document claimed it never moves down; that was wrong.

Baseline behaviour evaluates that level **only at bar closes** and nothing rests at the exchange, so
a gap through it exits at the next open (`tests/test_bot.py:139`). With `--enable-stop-exec`
(P0-A) each position instead carries its own stop, which ratchets up with the 20-bar low and never
moves down, is checked against every bar's low, and in demo/live rests at the exchange as a
STOP_LOSS_LIMIT.

Exits are never gated by the approval mechanism: they run automatically in paper, demo and live
(`app/main.py:101`).

---

## 7. Position sizing

**Where:** `app/risk.py:7` (`position_size`), `app/risk.py:16` (`size_with_minimum`).

```
position_size(equity, risk_pct, entry, stop, max_weight):
    dist = entry - stop
    if entry <= 0 or dist <= 0: return 0
    qty = equity * risk_pct / dist                 # 1% of equity at the stop
    return min(qty, equity * max_weight / entry)   # never more than 30% of equity in one name
```

Small-wallet handling, which is what makes a $100 account tradable at all:

```
size_with_minimum(..., min_notional, max_risk_multiple=2.0):
    qty = position_size(...)
    if qty <= 0:                       return 0, "price is already at or below the stop"
    if qty * entry >= min_notional:    return qty, None            # normal case
    need = min_notional / entry                                    # the smallest legal order
    if need * (entry - stop) <= equity * risk_pct * 2:
        return need, "raised to the minimum order (risks X%, not 1%)"
    return 0, "skipped: the minimum order would risk X% of the account"
```

So on a small wallet the risk ceiling is **2% per trade, not 1%**, and below that the trade is
skipped with a logged reason. Covered by `tests/test_bot.py:263` and `tests/test_bot.py:275`.

The manual `TRADE SYM 200 USDT` path bypasses risk sizing entirely and uses
`min(amount, equity * MAX_WEIGHT, cash) / price` (`app/main.py:257`).

---

## 8. Exchange filter rounding

**Where:** `app/risk.py:40` (`SymbolFilters`), applied inside both brokers before every order.

```
step_size, min_qty   from LOT_SIZE
tick_size            from PRICE_FILTER
min_notional         from NOTIONAL or MIN_NOTIONAL, default 5 USDT
round_qty / round_price = floor to the step, decimals derived from log10(step), epsilon 1e-9
check(qty, price):
    q = round_qty(qty)
    reject if q <= 0 or q < min_qty            -> "quantity below LOT_SIZE minimum"
    reject if q * price < min_notional          -> "order value below MIN_NOTIONAL"
```

Always floors, never rounds up, so an order can only get smaller than intended — never larger than
cash allows. Asserted by `tests/test_bot.py:45`.

---

## 9. Portfolio circuit breakers

**Where:** `app/risk.py:80` (`RiskState`), `app/risk.py:89` (`update_limits`), `app/risk.py:104`
(`can_enter`). State is persisted to SQLite each bar (`app/main.py:95`).

```
update_limits(state, equity, day):
    if day != state.day:                        # UTC day boundary
        state.day, state.day_start_equity, state.halted_today = day, equity, False
    state.peak_equity = max(state.peak_equity, equity)
    if equity <= day_start_equity * (1 - DAILY_LOSS_HALT 3%):   halted_today = True
    if equity <= peak_equity      * (1 - KILL_DRAWDOWN   8%):   killed = True

can_enter(state, n_open, max_names, kill_file_exists):
    False if data/KILL exists          -> "kill switch file is present"
    False if state.killed              -> "drawdown kill is active"
    False if state.halted_today        -> "daily loss cap hit"
    False if n_open >= max_names       -> "already holding the maximum"
    True otherwise
```

`killed` is sticky in live operation: nothing re-arms it, and while it is set every held position is
sold with reason "kill switch" (`app/main.py:108`). The backtest re-arms after
`kill_cooldown_days=7` purely so the simulation can continue measuring (`app/backtest.py:150`).

`max_names` is `LIVE_MAX_NAMES` (1) in live mode and `MAX_NAMES` (3) otherwise
(`app/main.py:43`).

---

## 10. The 4-hour cycle

**Where:** `app/main.py:78` (`on_bar_close`), scheduled by `app/main.py:317` (`_bar`).

```
tick every 15 s:
    write heartbeat
    _daily()     : after 00:05 UTC, rebuild universe, append history, send daily summary once
    _bar()       : if floor(now, 4h) != last_bar and now - bar > 20 s -> on_bar_close()
    _telegram()  : poll updates, execute queued commands

on_bar_close():
    build universe (cached per day)
    symbols = universe + anything still held
    panel, bar_time, rows, candidates = ranker.scan(...)    # appends history, scores all
    skip if bar_time already processed (unless force)
    if SHARED_DATA and bar_time < floor(now,4h): return     # wait for the main bot to append
    equity, cash = mark to market at this bar's closes
    day = (bar_time - 1 ms).date()                          # a 00:00 bar belongs to the previous day
    update circuit breakers, persist state, announce events
    missed = catch_up(...)
    EXITS   (the five-rung ladder in section 6)
    ENTRIES (approval gate in live mode, automatic otherwise)
    record equity, set last_bar
```

Each of the three jobs is wrapped separately (`app/main.py:49-55`), so a Binance outage cannot stop
Telegram command handling, and vice versa. Network errors flip the loop into a 5 s retry cadence and
produce one "OFFLINE" log line, then one "ONLINE again" message with the elapsed minutes
(`app/main.py:56-67`).

---

## 11. Downtime catch-up

**Where:** `app/main.py:165` (`catch_up`). Covered by `tests/test_catch_up.py`.

```
missed = bars strictly between last_bar and the current bar_time
for each missed bar, oldest first:
    for each held symbol not already flagged:
        if exit_reason fires on that bar -> remember the first such reason
    in paper mode, record equity for that bar so the chart has no hole
return {symbol: reason}          # acted on now, at today's price, in on_bar_close
```

**Asymmetric by design:** sell signals from missed bars are honoured, buy signals are discarded as
stale, mirroring how a live signal expires with the next bar. The reason string records which
missed check produced it, so the trade log stays reconstructible.

---

## 12. Approval gate (live mode only)

**Where:** `app/main.py:126` (publishing), `app/main.py:207` (`handle`). Active when
`mode == "live"` (`app/main.py:42`).

```
on each bar:
    expire all older pending signals
    expires_at = min(now + SIGNAL_EXPIRY_HOURS (2h), bar_time + 4h)
    for each candidate not held and with >= 90 days history:
        store a pending signal (bar_time, symbol, rank, score, stop, price, expires_at)
    send the candidates message; append the blocking reason if entries are barred

TRADE <SYM> [amount USDT] validates, in order:
    symbol on today's authenticated list                -> else REJECTED
    live_eligible (>= MIN_HISTORY_DAYS)                 -> else REJECTED
    a pending, unexpired signal exists for it           -> else REJECTED   (FORCE skips this one)
    not already held                                     -> else REJECTED (no adding)
    can_enter passes                                     -> else REJECTED
    Binance status is TRADING                            -> else REJECTED
then: stop = pending signal's stop, or for FORCE the current 20-bar low
      qty  = explicit amount, or size_with_minimum
      buy, mark the signal traded
```

Other commands: `SKIP` marks this bar's signals skipped; `FLAT <SYM>` / `FLAT ALL` sells at the
current price in any mode; `STATUS` prints equity, positions and pending signals. Grammar and
rejection cases: `app/notify.py:13-43`, tested at `tests/test_bot.py:25-33`.

**Silence produces no order** (`tests/test_bot.py:198`), and a non-candidate cannot be traded except
through `FORCE`, which still honours every other check (`tests/test_bot.py:192`).

---

## 13. Order idempotency

**Where:** `app/broker.py:21` (`client_order_id`), `app/broker.py:160` (`create_order`),
`app/portfolio.py:64` and `:82` (fill deduplication).

```
cid = "cb" + mode_initial + unix(bar_time) + symbol[:10] + side[0]      # deterministic
create_order(...):
    existing = get_order(symbol, cid)        # ask Binance first
    if existing: return it                   # never place twice
    place LIMIT IOC at ask*1.002 (buy) or bid*0.998 (sell), rounded to tick
    on network error or rejection: look the order up by cid again before giving up
POST is never blindly retried inside _signed (app/broker.py:110)
5xx on a POST raises "order status unknown" rather than retrying (app/broker.py:115)
record_buy / record_sell ignore a client_order_id already in the fills table
```

The order type is **LIMIT IOC at a marketable price**, not MARKET: it fills immediately or cancels
the remainder, and never rests on the book. Asserted by `tests/test_bot.py:244` (a lost response
places exactly one order) and `tests/test_bot.py:237` (429 is retried).

---

## 14. Paper fill model

**Where:** `app/broker.py:52` (`buy`), `app/broker.py:63` (`sell`).

```
buy:  px = current ticker price * (1 + SLIPPAGE_BPS/10000)
      qty = min(requested, cash / (px * (1 + FEE_RATE)))
      apply exchange filters; reject if below minimum
      cash -= qty*px + fee            where fee = qty*px*FEE_RATE (0.1%)
sell: px = current ticker price * (1 - slippage); cash += qty*px - fee
```

**This differs from the backtest**, which fills at the *next* bar's open (`app/backtest.py:98` and
`:113`). The live paper account therefore executes at the decision moment, while the published
return figures assume a one-bar delay. Noted as gap 2 in `AUDIT.md`.

---

## 15. Backtest engine

**Where:** `app/backtest.py:43` (`run`). Vectorised setup, then a single pass over bars.

```
precompute entry_ok = eligible & regime_on & breakout & (rs_7d > 0 or BTC) & rank <= CANDIDATES
for each bar t, in order:
    1  fill pending orders at this bar's open: SELLS first, then BUYS
         sell price = open * (1 - slip); buy price = open * (1 + slip); fee both ways
         a buy below min_notional (5 USDT) is logged as SKIP
         a sell with no data for 6 consecutive bars fills at the last known close
    2  mark to market at this bar's close; track peak per position; record equity and exposure
    3  risk limits: re-arm the kill after 7 days, then update_limits
    4  decide on this close, queue for the next open:
         exits via exit_reason (plus the optional variants)
         entries: slots = max_names - (open - already selling), best ranks first, sized normally
```

Timing is the core guarantee: **decide on the close of bar t, fill at the open of bar t+1**, proven
by `tests/test_bot.py:117`. Equity reconstructed from the fill log matches the reported curve
(`tests/test_bot.py:99`).

**Metrics** (`app/backtest.py:199`): total return, CAGR, max drawdown, Sharpe annualised by
`sqrt(6*365)`, Calmar, average exposure, trade count, win rate, mean and median trade, profit
factor, average bars held. **Benchmarks** (`app/backtest.py:226`): BTC buy-and-hold and an
equal-weight top-10 basket rebalanced every bar without fees.

---

## 16. Random baselines

**Where:** `app/backtest.py:235` (`random_baselines`), n = 500 draws, seed 7.

```
for each actual trade (entry bar e, exit bar x, coin k):
    random coin : same e and x, but a coin drawn from those eligible at bar e-1
    random time : same coin and holding length, entered at a random risk-on bar
each draw is charged the same slippage and fee as the real trade
p-value = share of the 500 random runs whose mean trade beat the strategy's mean trade
```

This is the honest test of whether the ranker adds anything: it separates "the coins went up" from
"we picked the right coins" (random coin) and from "we picked the right moments" (random time).

---

## 17. Survivorship-free universe

**Where:** `app/pit.py:73` (`run`). The most important research algorithm in the repo, because it is
what turned the flattering +20%/yr into the honest +6-13%/yr.

```
list every USDT pair in Binance's public archive, delisted ones included (app/pit.py:24)
drop stablecoins, fiat pairs, wrapped, leveraged and meme names (app/pit.py:39)
download 1d history for all of them
membership: for each day, rank coins by trailing 30-day quote volume, shifted one day,
            keep the top N                                   (app/pit.py:60)
download 4h history for every coin that was ever in the top N
build the panel, attach membership by day, then
    eligible = eligible and (member that day or BTC)
rescore, and run the ordinary backtest on that panel
```

`trailing = rolling(30).sum().shift(1)` is the anti-lookahead step: a day's universe uses only
volume from days already closed (`app/pit.py:150`). Benchmark for it is an equal-weight hold of that
day's whole universe (`app/pit.py:95`).

---

## 18. Walk-forward ML model

**Where:** `app/ml.py:39` (`walk_forward`). RESEARCH only — no live code path reads `ml_score`.

```
target  = (coin open[t+1+H] / open[t+1] - 1) - (same for BTC)          H = 6 bars (1 day)
features= FEATURE_COLS (16) + cross-sectional percentile ranks of 7 of them + breakout + regime
model   = HistGradientBoostingRegressor(max_iter=300, lr=0.03, leaves=15,
                                        min_samples_leaf=300, l2=1.0, seed 0)
roll:   train on the previous 730 days, purge the last H+1 bars, test the next 91 days, repeat
        labels clipped to the 1st / 99th percentile of the training fold
evaluate: per-bar Spearman IC, share of bars with IC > 0, top-20% minus bottom-20% spread
ship if and only if the model beats the ranker on BOTH total return and Sharpe out of sample
```

Measuring the target from the *next* open is what keeps the label honest: it is the price an order
would actually get. The purge gap stops the last training labels from overlapping the test window.
Verdict recorded in `AUDIT.md` and the README: rejected.

---

## 19. Exit variants (both rejected)

Both are keyword arguments on the backtest, never reachable from the live loop
(`app/backtest.py:43-46`):

- `never_sell_at_loss=True` — suppresses every exit except the drawdown kill while the position is
  underwater (`app/backtest.py:174`). Produces a high win rate and one catastrophic trade.
- `trail_pct=0.03` — adds "sold after giving back X% from its best close" (`app/backtest.py:172`),
  using the per-position `peak` tracked at `app/backtest.py:132`.

---

## 20. Add-to-position merge (dead code)

`app/portfolio.py:69-73` implements weighted-average merging when a buy arrives for a symbol already
held. Nothing can reach it: automatic entries exclude held symbols (`app/main.py:146`) and the
command path rejects them (`app/main.py:239`).

---

## Parameters, and where they live

| Parameter | Default | Where | Used by |
|---|---|---|---|
| `DONCHIAN_BARS` | 20 | `.env` / `app/config.py:44` | 3, 5, 6, 7 |
| `CANDIDATES` | 5 | `.env` / `app/config.py:46` | 5 |
| `EXIT_RANK` | 10 | `.env` / `app/config.py:45` | 6 |
| `RISK_PCT` | 0.01 | `.env` / `app/config.py:50` | 7 |
| `MAX_NAMES` | 3 | `.env` / `app/config.py:51` | 5, 9 |
| `MAX_WEIGHT` | 0.30 | `.env` / `app/config.py:52` | 7 |
| `DAILY_LOSS_HALT` | 0.03 | `.env` / `app/config.py:53` | 9 |
| `KILL_DRAWDOWN` | 0.08 | `.env` / `app/config.py:54` | 9 |
| `FEE_RATE` | 0.001 | `.env` / `app/config.py:55` | 14, 15, 16 |
| `SLIPPAGE_BPS` | 7 | `.env` / `app/config.py:56` | 14, 15, 16 |
| `SIGNAL_EXPIRY_HOURS` | 2 | `.env` / `app/config.py:59` | 12 |
| `LIVE_MAX_NAMES` | 1 | `.env` / `app/config.py:60` | 9, 12 |
| `LIVE_BUDGET_USDT` | 100 | `.env` / `app/config.py:61` | 10 |
| `UNIVERSE_SIZE`, `CG_TOP` | 30, 40 | `.env` / `app/config.py:36-37` | 1 |
| `MIN_VOLUME_USDT` | 20M | `.env` / `app/config.py:38` | 1 |
| `MIN_HISTORY_DAYS` | 90 | `.env` / `app/config.py:39` | 1, 12 |
| Score weights 0.35/0.25/0.20/0.20 | — | **hardcoded** `app/features.py:104` | 4 |
| Eligibility age 540 bars | — | **hardcoded** `app/features.py:62` | 3, 4 |
| `max_risk_multiple` 2.0 | — | **hardcoded** `app/risk.py:17` | 7 |
| Backtest `min_notional` 5.0 | — | **hardcoded** `app/backtest.py:69` | 15 |
| Kill cooldown 7 days | — | backtest default `app/backtest.py:43` | 15 |
| ML horizon / train / test | 6 / 730d / 91d | CLI + defaults `app/ml.py:39` | 18 |
