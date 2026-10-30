# Metric matrix

Every number this bot computes, where it comes from, and whether anything reads it.
Audited 25 September 2026 against the working tree. Companions: `ALGORITHMS.md` (how the rules use
them) and `AUDIT.md` (implemented vs tested vs live-used).

**Consumer key**

| Tag | Meaning |
|---|---|
| **DECISION** | a buy, sell or sizing rule reads it |
| **RANK** | it feeds the score that orders the universe |
| **GATE** | it decides whether a row may be scored at all |
| **DISPLAY** | shown on Telegram, the website or the CLI only |
| **ML** | read only by the rejected walk-forward model (`app/ml.py`) |
| **DEAD** | nothing reads it |

Common inputs: 4h candles (`data/klines/SYM_4h.parquet`), daily candles for the two SMAs,
`BARS_PER_DAY = 6`. Warm-up is in 4h bars unless stated.

---

## 1. Time and identity

| Metric | Formula | Location | Warm-up | Consumer | Notes |
|---|---|---|---|---|---|
| `bar_time` | `open_time + 4h`, UTC | `features.py:17` | 0 | **DECISION** | The scheduled close, deliberately not `close_time`: outage-era candles carry odd close times that would stop coins lining up across the panel |
| `close_time` | from Binance | `data.py:16` | 0 | **GATE** | Drives the closed-candle rule and the as-of daily join |
| `age_bars` | `1..len(df)` | `features.py:41` | 0 | **GATE** | Bars of own history; 540 needed (90 days) |
| `symbol` | universe list | `features.py:70` | — | **DECISION** | |
| `day` | `(bar_time - 1 ms).date()` | `main.py:91`, `backtest.py:85` | — | **DECISION** | A 00:00 bar belongs to the previous UTC day, so the daily loss cap resets correctly |

---

## 2. Returns

| Metric | Formula | Location | Warm-up | Consumer | Notes |
|---|---|---|---|---|---|
| `ret_4h` | `close.pct_change()` | `features.py:21` | 1 | **DISPLAY** | Only the pit equal-weight benchmark reads it (`pit.py:98`) |
| `ret_1d` | `close.pct_change(6)` | `features.py:22` | 6 | **ML** | Also the base of `rs_1d` |
| `ret_7d` | `close.pct_change(42)` | `features.py:23` | 42 | **RANK**, **DISPLAY** | Simple return, not log |
| `ret_30d` | `close.pct_change(180)` | `features.py:24` | 180 | **RANK** | Feeds score parts *b* and *d* |
| `logret` | `log(close).diff()` | `features.py:19` | 1 | intermediate | Only used for the volatility terms |

All returns are **simple** (`pct_change`) except the volatility inputs, which are log returns. No
metric mixes the two.

---

## 3. Trend

| Metric | Formula | Location | Warm-up | Consumer | Notes |
|---|---|---|---|---|---|
| `sma50_d` | daily `close.rolling(50).mean()` | `features.py:46` | 50 days | intermediate | Joined `merge_asof(direction="backward")` so a 4h bar sees only daily bars already closed |
| `sma200_d` | daily `close.rolling(200).mean()` | `features.py:47` | 200 days | intermediate | |
| `dist_sma50` | `close / sma50_d - 1` | `features.py:52` | 50 days | **RANK** | Score part *c*, weight 0.20 |
| `dist_sma200` | `close / sma200_d - 1` | `features.py:53` | 200 days | **ML** per coin; **DECISION** for BTC | BTC's value becomes `btc_dist_sma200` and defines the regime |
| `regime_on` | `btc_dist_sma200 > 0` | `features.py:86` | 200 days | **DECISION** | Entry rule 1 and exit rule 1; the single largest on/off switch in the system |

No EMA, no golden/death cross, no MACD anywhere in the repo.

---

## 4. Breakout channel

| Metric | Formula | Location | Warm-up | Consumer | Notes |
|---|---|---|---|---|---|
| `don_high` | `high.rolling(20).max().shift(1)` | `features.py:26` | 21 | **DECISION** | `.shift(1)` is the anti-lookahead step: the close is compared with the *previous* 20 bars |
| `don_low` | `low.rolling(20).min().shift(1)` | `features.py:27` | 21 | **DECISION** | |
| `stop_level` | `low.rolling(20).min()` **including this bar** | `features.py:28` | 20 | **DECISION** | The stop price and the sizing denominator. Deliberately different from `don_low` |
| `breakout` | `close > don_high` | `features.py:29` | 21 | **DECISION** | Entry rule 2. True on about 5% of bars |
| `breakdown` | `close < don_low` | `features.py:30` | 21 | **DECISION** | Exit rule 2 |
| `don_pos` | `(close - don_low) / (don_high - don_low)` | `features.py:31` | 21 | **ML** | **Divides by zero on a perfectly flat channel** — harmless today because only the ML path reads it |

The two-definition split matters: `don_low` is shifted so a breakdown is judged against the past,
while `stop_level` includes the current bar so the stop a position is given never sits above the
low that has just printed.

---

## 5. Volatility

| Metric | Formula | Location | Warm-up | Consumer | Notes |
|---|---|---|---|---|---|
| `tr` | `max(h-l, |h - prev close|, |l - prev close|)` | `features.py:32` | 2 | intermediate | Standard true range |
| `atr_pct` | `tr.rolling(14).mean() / close` | `features.py:33` | 15 | **ML** | **Not used by the stop.** The stop is the Donchian low, not an ATR multiple |
| `vol_30d` | `logret.rolling(180).std() * sqrt(6*365)` | `features.py:34` | 180 | **RANK**, **GATE** | Annualised; denominator of score part *d* |
| `vol_7d` | `logret.rolling(42).std() * sqrt(6*365)` | `features.py:35` | 42 | **ML** | |
| `ANNUAL` | `sqrt(6 * 365)` = 46.8 | `features.py:9` | — | constant | Assumes 6 bars a day, every day — correct for crypto |

No Parkinson, Garman-Klass or Yang-Zhang estimator; no Bollinger bands.

---

## 6. Volume

| Metric | Formula | Location | Warm-up | Consumer | Notes |
|---|---|---|---|---|---|
| `volume_ratio` | 1-day quote volume ÷ (20-day mean daily quote volume) | `features.py:37` | 120 | **DISPLAY**, **ML** | Shown as `vol OK/LOW` at a 0.7 threshold in the Telegram template (`notify.py:52`) — **display only, it never blocks a trade** |
| `taker_ratio` | 1-day taker-buy volume ÷ 1-day total volume | `features.py:38` | 6 | **ML** | Buy-side pressure proxy |
| `dollar_vol_30d` | 30-day quote volume ÷ 30 | `features.py:40` | 180 | **DEAD** | Referenced nowhere in the repo |
| `quoteVolume` (24h) | Binance ticker | `universe.py:88` | — | **GATE** | The 20M USDT universe floor — the only volume filter that actually rejects anything |

No OBV, no volume-spike flag, no accumulation/distribution.

---

## 7. Cross-section and relative strength

Computed after all symbols are stacked (`features.py:76`).

| Metric | Formula | Location | Warm-up | Consumer | Notes |
|---|---|---|---|---|---|
| `btc_ret_1d/7d/30d` | BTC's returns joined onto every row by `bar_time` | `features.py:80` | as source | intermediate | |
| `btc_dist_sma200` | BTC's distance from its 200-day SMA | `features.py:80` | 200 days | **DECISION** | |
| `rs_1d` | `ret_1d - btc_ret_1d` | `features.py:82` | 6 | **ML** | |
| `rs_7d` | `ret_7d - btc_ret_7d` | `features.py:83` | 42 | **DECISION**, **RANK** | Entry rule 3 (`> 0`) and score part *a*, weight 0.35 — the heaviest single input |
| `rs_30d` | `ret_30d - btc_ret_30d` | `features.py:84` | 180 | **RANK**, **GATE** | Score part *b*, weight 0.25 |
| `eligible` | `age_bars >= 540` and `rs_30d`, `dist_sma50`, `vol_30d` all present | `features.py:88` | 180 | **GATE** | Nothing ineligible can be scored, ranked, bought or counted |
| `n_eligible` | count of eligible rows per bar | `features.py:106` | — | **DISPLAY** | |

Relative strength is an arithmetic **difference of returns**, not a ratio and not a beta-adjusted
residual.

---

## 8. The score

| Metric | Formula | Location | Consumer | Notes |
|---|---|---|---|---|
| part *a* | `percentile_rank(rs_7d)` within the bar | `features.py:98` | **RANK** | weight 0.35 |
| part *b* | `percentile_rank(rs_30d)` | `features.py:99` | **RANK** | weight 0.25 |
| part *c* | `percentile_rank(dist_sma50)` | `features.py:100` | **RANK** | weight 0.20 |
| part *d* | `percentile_rank(ret_30d / vol_30d)` | `features.py:102-103` | **RANK** | weight 0.20; return per unit of risk |
| `score` | `100 * (0.35a + 0.25b + 0.20c + 0.20d)` | `features.py:104` | **DECISION** | Weights are **hardcoded**, not in `.env` |
| `rank` | descending `score`, ties by index order | `features.py:105` | **DECISION** | Entry rule 4 (`<= 5`), exit rule 3 (`> 10`) |

Each part is a percentile **within that bar's eligible set**, so the score is always relative to the
day's field. A score of 100 means the coin led all four components that bar.

---

## 9. Risk and sizing

| Metric | Formula | Location | Consumer | Notes |
|---|---|---|---|---|
| `dist` | `entry - stop_level` | `risk.py:9` | **DECISION** | Zero or negative returns size 0 |
| `qty` (risk-based) | `equity * RISK_PCT / dist` | `risk.py:12` | **DECISION** | 1% of equity at the stop |
| weight cap | `equity * MAX_WEIGHT / entry` | `risk.py:13` | **DECISION** | 30% of equity per name |
| minimum-order lift | `min_notional / entry`, allowed while risk <= `2 x RISK_PCT` | `risk.py:29-34` | **DECISION** | Small wallets: effective ceiling is **2% per trade**, else the trade is skipped |
| `peak_equity` | running max of equity | `risk.py:91` | **DECISION** | Drawdown kill reference |
| `day_start_equity` | equity at the first bar of a UTC day | `risk.py:92` | **DECISION** | Daily halt reference |
| daily halt trigger | `equity <= day_start * (1 - 0.03)` | `risk.py:93` | **DECISION** | Blocks entries for the rest of the day |
| kill trigger | `equity <= peak * (1 - 0.08)` | `risk.py:96` | **DECISION** | Flattens and stays killed |
| `step_size`, `min_qty`, `tick_size`, `min_notional` | Binance LOT_SIZE / PRICE_FILTER / NOTIONAL | `risk.py:47-54` | **DECISION** | Always floored, never rounded up |

---

## 10. Account and reporting

| Metric | Formula | Location | Consumer | Notes |
|---|---|---|---|---|
| `equity` | `cash + Σ qty * price` | `main.py:52-58` | **DECISION** | Live mode caps cash at `LIVE_BUDGET_USDT` minus open cost |
| `cash` | paper: kv store; demo/live: USDT balance | `broker.py:40`, `broker.py:138` | **DECISION** | |
| `pnl` per trade | `proceeds - cost * share` | `portfolio.py:91` | **DISPLAY** | Fees included on both legs |
| `exposure` | `held / equity` | `backtest.py:141` | **DISPLAY** | Averaged about 11% historically |
| `total_return`, `cagr` | standard | `backtest.py:204-205` | **DISPLAY** | Years from calendar days / 365.25 |
| `max_drawdown` | `min(equity / cummax - 1)` | `backtest.py:206` | **DISPLAY** | |
| `sharpe` | `mean(r) / std(r) * sqrt(6*365)` | `backtest.py:207` | **DISPLAY** | Per-bar returns, zero risk-free rate |
| `calmar` | `cagr / |max_drawdown|` | `backtest.py:208` | **DISPLAY** | |
| `win_rate`, `avg_trade`, `median_trade`, `profit_factor`, `avg_bars_held` | standard | `backtest.py:211-216` | **DISPLAY** | |
| `p_value_vs_random_coin` / `_time` | share of 500 random runs beating the strategy | `backtest.py:277-279` | **DISPLAY** | The honest skill test |
| `mean_ic`, `ic_positive_share` | per-bar Spearman of score vs realised excess return | `ml.py:65-73` | **ML** | |
| `top_minus_bottom` | mean target of top 20% minus bottom 20% by score | `ml.py:76-82` | **ML** | |

---

## Summary counts

| Category | Count | Names |
|---|---|---|
| Drive a buy, sell or size | **14** | `regime_on`, `breakout`, `breakdown`, `don_high`, `don_low`, `stop_level`, `rs_7d`, `rank`, `score`, `eligible`, `age_bars`, `equity`, `dist`, exchange filters |
| Feed the score only | **4** | `rs_7d`, `rs_30d`, `dist_sma50`, `ret_30d / vol_30d` |
| Display only | **3** | `volume_ratio` (as vol OK/LOW), `ret_4h`, `n_eligible` |
| ML-only | **6** | `atr_pct`, `vol_7d`, `taker_ratio`, `don_pos`, `rs_1d`, per-coin `dist_sma200` |
| Dead | **1** | `dollar_vol_30d` |

**Indicators a typical momentum bot has that this one does not compute at all:** RSI, MACD,
Stochastic, ADX, Bollinger bands, OBV, EMA of any length, funding rates, open interest, order-book
imbalance, correlation or beta to BTC (only the return difference), and any sentiment or news input.

**Known numerical issues**

1. `don_pos` divides by `don_high - don_low`, which can be zero (`features.py:31`). ML path only.
2. Score part *d* divides by `vol_30d` (`features.py:102`); the eligibility gate requires
   `vol_30d` to be present but not non-zero.
3. `volume_ratio` needs 120 bars of warm-up, longer than the 540-bar eligibility gate implies for a
   newly listed coin — in practice the age gate dominates.
4. `taker_ratio` assumes `volume` is non-zero over a whole day (`features.py:38`); a full day of
   zero volume would produce NaN, which is not currently filtered.
5. Warm-up is enforced indirectly through `eligible` (540 bars), which is longer than every metric's
   own warm-up except `sma200_d` at 200 **days** — a coin with 540 four-hour bars (90 days) still has
   no 200-day SMA, so `dist_sma200` stays NaN for its own row. This only matters for the ML feature;
   the regime uses BTC's value, which always exists.
