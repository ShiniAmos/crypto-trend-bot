# Research log

One entry per phase. Every number here comes from a saved experiment record in
`reports/experiments/`, reproducible with the command printed beside it.

Costs are the same in every run: 7 bps slippage per side, 0.10% fee per side (0.20% round trip),
applied once per leg — verified in `app/backtest.py:98,113,119` and `app/broker.py:52-68`.

---

## Phase 0 — harness and checkpoint (25 Sep 2026)

Commit `f6ffdaf` is the frozen baseline: the bot exactly as audited, 42 tests passing. The
repository had **no commits at all** before this, so there was no rollback point; there is now.

Added, none of which change trading behaviour:

| File | Purpose |
|---|---|
| `app/flags.py` | Feature flags, all default off. CLI `--enable-*`, live `.env` `ENABLE_*` |
| `app/regime.py` | Causal regime labels from BTC only, and per-regime metrics |
| `app/experiment.py` | Machine-readable experiment records and the ship-rule decision table |
| `.gitattributes` | Forces LF on deploy scripts — a CRLF here is what broke the first VPS install |

**Baseline by market regime** (standard panel, `python -m app.main backtest --label standard_baseline`):

| Regime | Years | CAGR | Sharpe | Max DD | Trades | Win rate | Profit factor | Exposure |
|---|---|---|---|---|---|---|---|---|
| strong bull | 2.45 | +79.5% | 2.69 | −13.7% | 231 | 41.6% | 1.93 | 22.2% |
| weak bull | 1.24 | +43.6% | 1.80 | −14.9% | 123 | 32.5% | 1.12 | 25.0% |
| sideways/chop | 1.03 | −6.5% | −0.36 | −12.7% | 114 | 38.6% | 2.39 | 16.0% |
| recovery | 0.80 | −6.6% | −2.68 | −5.4% | 6 | 50.0% | 1.66 | 0.1% |
| bear | 3.58 | −2.4% | −1.73 | −8.2% | 5 | 20.0% | 0.32 | 0.0% |

The strategy earns everything in the two bull regimes and bleeds slowly everywhere else. Exposure
in bear and recovery is effectively zero — the BTC regime filter does what it claims, and 3.58 of
the 9.1 years were spent almost entirely in cash. **Nothing here says the strategy works in a bear
market; it says it mostly stays out of one.**

---

## Phase 1 — P0 correctness work

### P0-B Paper execution now matches the backtest — SHIPPED

Paper decisions are queued and filled at the **next bar's open ± slippage**, sells before buys,
exactly as `app/backtest.py` does. Verified end to end against live data: a queued LTC buy filled
at 66.6166, and that bar's open × (1 + slippage) = 66.6166.

Before this, paper filled at the ticker price at decision time, so paper P&L and backtest P&L were
never measuring the same thing. They are now.

Side effect worth knowing: entries and exits in paper are one bar (up to 4 hours) later than they
used to be. Both running paper accounts restart their comparison from 25 Sep 2026.

### P0-C Boot reconciliation — SHIPPED

`app/reconcile.py` compares positions, balances, open orders and recorded stops against the
exchange at every start and on `python -m app.main reconcile`. A mismatch logs a high-severity
event, sends Telegram, writes `safe_mode` and blocks **entries only** — exits keep working, because
being unable to close a position is worse than being unable to open one.

Paper mode has no exchange, so it checks its own books for internal contradictions instead.

### P0-D Skipped entries are recorded — SHIPPED

Entries dropped for staleness now land in `skipped_signals` with the bar, coin, rank, score, stop,
what it would have paid, the price when the bot noticed, downtime in minutes and the move since.
Two paths feed it: bars missed while the bot was down, and queued buys whose fill bar never
arrived. Visible on the website under "Buys missed while the bot was not running".

### P0-A Stop execution — IMPLEMENTED, **NOT SHIPPED** (flag stays off)

Behind `--enable-stop-exec` / `ENABLE_STOP_EXEC`:

- each position carries its own stop, which **ratchets up** with the 20-bar low and never moves down
  (`Store.update_stop` refuses a lower value);
- the backtest and paper check every bar's **low** against it and fill at **min(open, stop)**, so a
  gap below the stop fills at the open, never at the better price;
- **stop first**: the stop is resolved before any other exit on that bar, including a position
  bought at that same open;
- demo and live rest a real `STOP_LOSS_LIMIT` at the exchange, replace it as it ratchets, cancel it
  before selling, and record it in `stop_orders` so reconciliation can see it.

**Decision table**

| Feature | Panel | CAGR base → new | Sharpe base → new | Max DD base → new | Trades | Random coin p | Random time p | Decision |
|---|---|---|---|---|---|---|---|---|
| stop_exec | standard | 20.24% → **19.15%** | 1.291 → **1.325** | −20.43% → **−19.37%** | 479 → 589 | 0.002 | 0.000 | **KEEP BASELINE** |
| stop_exec | pit30 | 5.73% → **6.74%** | 0.502 → **0.588** | −29.33% → **−33.34%** | 669 → 741 | 0.000 | 0.000 | **KEEP BASELINE** |

Each panel improves two of the three required measures and damages the third, and it is not the
same third: the standard panel loses 1.1 points of CAGR, the survivorship-free panel loses 4.0
points of drawdown. By the master ship rule — improve or hold on CAGR, Sharpe **and** max drawdown,
on both panels — this fails, and ambiguity means keep the old strategy.

Two honest observations that do **not** override that:

- The stopped version beats the random baselines far more decisively (random-time p 0.506 → 0.000
  on pit30). Its trades look less like luck. The ship rule does not weigh this, and the drawdown
  damage on the honest panel is real.
- A 4-hour backtest cannot price the risk this feature exists for: a collapse between two checks,
  or a crash while the bot is down. The measurement says it costs return; it does not say it fails
  to protect.

**Where that leaves it:** the flag stays off, so the shipped strategy is unchanged. The exchange
stop remains available as an operational decision for any future demo or live run, to be taken at
the deployment review and recorded as insurance, not as a performance improvement.

### Regression check

`reports/_parity_baseline_equity.csv` was generated from commit `f6ffdaf` before any edit. Re-run
after all P0 work with flags off: **the equity curve is byte-identical**. `tests/test_p0.py` also
asserts parity directly.

### Tests

57 passing (42 original, 15 new). New coverage: flag-off parity, flag identity, intrabar stop
conservatism, gap-below-stop fills, stop ratchet direction, paper fill equals backtest fill,
queued-order lifecycle, sell-before-buy ordering, stale queued buy dropped while a stale sell still
executes, FLAT cancelling queued orders, reconciliation (missing position, unknown balance,
untracked order, clean case), safe mode blocking entries but not exits, skipped-entry recording.

### Reproduce

```bash
python -m pytest tests -q
python -m app.main backtest --label standard_baseline
python -m app.main backtest --enable-stop-exec --label standard_stopexec
python -m app.main pit --top 30 --label pit30_baseline
python -m app.main pit --top 30 --enable-stop-exec --label pit30_stopexec
```

---

## Phase 2 — P1 entry filters

### The study that produced the hypotheses

`research/trade_study.py` takes all 482 historical trades, attaches every feature available at the
entry decision, and splits them: **discover on 2017-2021, validate on 2022-2026**. Twenty-two
relationships were tested. Two survived both halves with |t| > 1, which is roughly what chance
would produce, so the survivors were treated as hypotheses to test, not as findings.

| Condition at entry | Trades | Average result | 2017-2021 | 2022-2026 |
|---|---|---|---|---|
| Coin's own 30-day return < 0 | 101 | **−0.61%**, 26% win | +1.04% | −1.73% |
| Coin's own 30-day return ≥ 0 | 381 | **+4.37%**, 40% win | +6.75% | +2.76% |
| BTC's 30-day return < 0 | 156 | **+0.41%**, 29% win | +1.51% | −0.18% |
| BTC's 30-day return ≥ 0 | 326 | +4.2% to +5.6% | | |
| rs_7d above +35% versus BTC | 26 | **−4.02%** | −5.40% | −2.65% |

In plain terms: the bot was buying coins that merely fell more slowly than Bitcoin, and buying into
months when Bitcoin itself was sinking. A third of all its trades were of the second kind and they
earned nothing.

Everything else — including `score` itself, the ranker's own output — flipped sign between the two
halves. That is the single most useful line in the study: the ranking predicts which coin to
prefer, but among the trades actually taken, its level says nothing reliable about the outcome.

### Decision table

| Feature | Panel | CAGR base → new | Sharpe base → new | Max DD base → new | Trades | Decision |
|---|---|---|---|---|---|---|
| **abs_momentum** | standard | 20.24% → **20.67%** | 1.291 → **1.367** | −20.43% → **−16.33%** | 479 → 423 | **CANDIDATE FOR SHIP** |
| **abs_momentum** | pit30 | 5.73% → **7.08%** | 0.502 → **0.607** | −29.33% → **−22.58%** | 669 → 631 | **CANDIDATE FOR SHIP** |
| btc_momentum | standard | 20.24% → 18.90% | 1.291 → 1.324 | −20.43% → −16.13% | 479 → 361 | KEEP BASELINE (CAGR) |
| btc_momentum | pit30 | 5.73% → 7.72% | 0.502 → **0.696** | −29.33% → −19.88% | 669 → 516 | passes here only |
| rs_cap 0.35 | standard | 20.24% → 17.89% | 1.291 → 1.191 | −20.43% → −22.75% | 479 → 463 | KEEP BASELINE (all three) |
| abs + btc combined | standard | 20.24% → 18.89% | 1.291 → 1.333 | −20.43% → −15.67% | 479 → 352 | worse than abs alone |

### Held-out check (2022-2026 only, the half not used for discovery)

| Run | CAGR | Sharpe | Max DD | Trades | Win |
|---|---|---|---|---|---|
| baseline | 12.09% | 0.891 | −24.30% | 287 | 37.3% |
| **abs_momentum** | **15.07%** | **1.100** | **−16.33%** | 253 | 40.7% |
| btc_momentum | 13.38% | 1.063 | −16.13% | 210 | 41.9% |
| abs + btc | 13.64% | 1.078 | −15.67% | 205 | 41.0% |

### P1-E abs_momentum — SHIPPED, enabled in paper

**Rule added:** a candidate must also have a positive 30-day return of its own. One line, one
condition, no new parameter to tune.

It improves CAGR, Sharpe **and** maximum drawdown on the standard panel, on the survivorship-free
panel, and on the held-out period. Expectancy per trade rose from +0.67% to +0.90% on pit30.
It still beats the random-coin baseline (p = 0.022) and, like the baseline, still does **not** beat
random timing on pit30 (p = 0.344): the coin choice carries the edge, the entry moment does not.

**Rejected:** `rs_cap` failed on every measure despite the study showing those 26 trades lose money
— removing them changed which coins filled the slots and the replacements were worse.
`btc_momentum` passes on pit30 and fails on the standard panel, so the evidence is mixed and the
baseline stands. The combination of the two is worse than `abs_momentum` alone, so the simpler rule
wins.

### Reproduce

```bash
python research/trade_study.py
python -m app.main backtest --enable-abs-momentum --label standard_absmom
python -m app.main pit --top 30 --enable-abs-momentum --label pit30_absmom
```

---

## Phase 3 — remaining P1-P3 flags (27 Sep 2026)

Baseline for everything below is the **shipped** state: `abs_momentum` on, 19-coin pinned universe
(`reports/_parity_universe.json`), identical dates, costs and risk settings. Standard and held-out
are measured first with `research/compare.py`; the survivorship-free panel is only run for a
candidate that already passed both, because it costs ten minutes and a failure elsewhere already
decides the question.

Shipped baseline numbers: standard CAGR 20.67% / Sharpe 1.367 / DD −16.33% / p_coin 0.028;
held-out CAGR 15.07% / Sharpe 1.100 / DD −16.33% / p_coin 0.112; pit30 CAGR 7.08% / Sharpe 0.607 /
DD −22.58% / p_coin 0.022.

### P1-G ENABLE_BREADTH — **REJECT**, flag stays off

Rule tested: when fewer than 35% of eligible non-BTC coins are beating BTC over 30 days, take no
alt entry; BTC may still qualify on its own rules. Above 55%, and in the middle band, the ordinary
rules are unchanged.

| Panel | CAGR | Sharpe | Max DD | Trades | p_coin | p_time |
|---|---|---|---|---|---|---|
| standard baseline | 20.67% | 1.367 | −16.33% | 423 | 0.028 | 0.004 |
| standard + breadth | **12.81%** | **1.079** | **−20.20%** | 336 | **0.138** | 0.112 |
| held-out baseline | 15.07% | 1.100 | −16.33% | 253 | 0.112 | 0.192 |
| held-out + breadth | **5.55%** | **0.577** | **−20.20%** | 191 | **0.600** | 0.728 |

Fails all four criteria on both panels: it costs eight points of CAGR, lowers Sharpe, deepens the
drawdown **and** destroys the coin-selection significance (p_coin 0.028 → 0.138, and 0.600 on the
held-out half). The rule removes 87 trades and the ones it removes were disproportionately good.

Read together with the earlier study, this is consistent: breadth already flipped sign between the
two halves there (+0.049 then −0.052), which was the warning. Low breadth is not a warning sign in
this universe — it is what a young rotation looks like before the rest of the market follows.

### P1-H ENABLE_VOLUME — three thresholds, one survivor to the pit stage

Rule tested: a breakout only counts if the coin's last day of turnover was at least T times its own
20-day average. A missing ratio fails closed.

| T | Panel | CAGR | Sharpe | Max DD | Trades | p_coin | Verdict |
|---|---|---|---|---|---|---|---|
| — | standard baseline | 20.67% | 1.367 | −16.33% | 423 | 0.028 | |
| **0.8** | standard | **20.71%** | **1.371** | **−15.09%** | 398 | 0.040 | **PASS** |
| 1.0 | standard | 16.64% | 1.241 | −18.73% | 371 | 0.074 | REJECT |
| 1.2 | standard | 17.22% | 1.330 | −12.78% | 343 | 0.072 | REJECT (CAGR) |
| — | held-out baseline | 15.07% | 1.100 | −16.33% | 253 | 0.112 | |
| **0.8** | held-out | 14.73% | **1.100** | **−15.09%** | 233 | 0.196 | **PASS** (CAGR flat inside epsilon) |
| 1.0 | held-out | 12.97% | 1.016 | −18.73% | 218 | 0.180 | REJECT |
| 1.2 | held-out | 14.55% | 1.181 | −12.78% | 202 | 0.186 | REJECT (CAGR) |

Only **T = 0.8** clears both. Note what that threshold actually is: it does not demand a busy day,
it rejects a breakout printed on a *quiet* one — under 80% of the coin's own normal turnover. The
stricter thresholds reject too much, and 1.2 shows the shape of the trade-off clearly: the best
drawdown of all (−12.78%) bought with 3.5 points of CAGR.

Per the rule, at most one T may ship, and the survivorship-free panel decides it.

**Survivorship-free panel, T = 0.8:**

| Run | CAGR | Sharpe | Max DD | Trades | Win | p_coin | p_time |
|---|---|---|---|---|---|---|---|
| baseline (abs_momentum) | 7.08% | 0.607 | −22.58% | 631 | 37.4% | 0.022 | 0.344 |
| **+ volume 0.8** | **7.84%** | **0.697** | **−19.35%** | 591 | 38.2% | **0.012** | **0.112** |

### P1-H ENABLE_VOLUME at 0.8 — **SHIP**, enabled in paper

Passes all six conditions: better or equal on CAGR, Sharpe and drawdown on the standard panel and
the survivorship-free panel, held-out clean with one metric flat inside the declared epsilon,
p_coin better than baseline everywhere, flag-off parity unchanged, identical universe, dates,
costs and risk settings.

The strongest evidence is not the return: it is that **both significance measures improved on the
honest panel** (p_coin 0.022 → 0.012, p_time 0.344 → 0.112). Filtering breakouts by turnover makes
the remaining trades look less like luck, which is the opposite of what a curve-fitted filter does.

What it actually does: it rejects a breakout printed on a day quieter than 80% of that coin's own
normal turnover — 40 such trades on the honest panel. It does not ask for a volume surge; 1.0 and
1.2 both fail.

### P1-F ENABLE_BTC_SLEEVE — **REJECT**, flag stays off

Rule tested: on a risk-on bar where no alt qualifies after every shipped filter, buy BTC instead of
holding cash. Regime off still means USDT only. BTC sized by the ordinary 1% formula.

| Panel | CAGR | Sharpe | Max DD | Trades | Win | p_coin |
|---|---|---|---|---|---|---|
| standard baseline | 20.71% | 1.371 | −15.09% | 398 | 39.7% | 0.040 |
| standard + sleeve | **11.29%** | **0.732** | **−32.38%** | 823 | 32.4% | 0.024 |
| held-out baseline | 14.73% | 1.100 | −15.09% | 233 | 40.8% | — |
| held-out + sleeve | **3.03%** | **0.274** | **−32.38%** | 500 | 29.6% | — |

It more than doubles the trade count (398 → 823) and halves the win rate, because "no alt
qualifies" is the normal state of a chop market: the sleeve buys BTC there, gets stopped on the
20-bar low, and buys it again on the next empty bar. The drawdown doubles.

Idle capital is not the disease. On this strategy, being in cash when nothing qualifies is where
the drawdown control comes from — the regime table in Phase 0 already showed near-zero exposure in
bear and recovery is what keeps those regimes to −2.4%.

### P1 combination (breadth + volume) — not run

The rule permits a combination only when both parts are individually non-harmful. Breadth was
harmful on every measure, so the combination was not tested.

## P2 — exits and sizing

Baseline throughout: `abs_momentum` + `volume_min=0.8` (standard CAGR 20.71% / Sharpe 1.371 /
DD −15.09%; held-out 14.73% / 1.100 / −15.09%).

### P2-I ENABLE_SCALE_OUT_R and ENABLE_SCALE_OUT_PCT — **REJECT**, both flags stay off

Half the position leaves at +1.5R (or +8%), the remainder is protected at breakeven plus the round
trip's fees. Both trigger on a bar close and fill at the next open, like every other exit here, so
no intrabar assumption and no OHLC ambiguity is introduced.

| Run | CAGR | Sharpe | Max DD | Trades | **Win rate** |
|---|---|---|---|---|---|
| baseline | 20.71% | 1.371 | −15.09% | 398 | 39.7% |
| +1.5R | **14.26%** | 1.342 | −13.31% | 516 | **52.9%** |
| +8% | **11.28%** | 1.155 | −13.56% | 619 | **55.4%** |
| held-out baseline | 14.73% | 1.100 | −15.09% | 233 | 40.8% |
| held-out +1.5R | **10.53%** | 1.041 | −13.31% | 293 | 51.9% |
| held-out +8% | **9.06%** | 0.957 | −11.14% | 348 | 55.5% |

This is the cleanest illustration in the whole log of why win rate is the wrong objective. Both
variants do exactly what they promise — the win rate jumps from 40% to 53-55% and the drawdown
improves — and both destroy a third of the return, because the winners they cut in half are what
pays for the 60% of trades that lose. The random baselines even improve (p_coin 0.000), which is a
reminder that statistical significance and profitability are different questions.

### P2-I ENABLE_TRAIL_REST — not run

The rule permits a profit-giveback trail on the remainder only if a scale-out shipped. Neither did.

### P2-J ENABLE_VOL_TARGET — **REJECT**, flag stays off

`qty *= clip(median(vol_30d that bar) / vol_30d, 0.5, 1.5)`, applied before every existing cap.

| Panel | CAGR | Sharpe | Max DD | p_coin |
|---|---|---|---|---|
| standard baseline | 20.71% | 1.371 | −15.09% | 0.040 |
| standard + vol target | **21.06%** | **1.398** | **−14.01%** | **0.018** |
| held-out baseline | 14.73% | 1.100 | −15.09% | — |
| held-out + vol target | **13.63%** | 1.096 | −14.01% | — |

Passes the standard panel outright and improves the significance, then loses 1.10 CAGR points on
the held-out half — twice the declared 0.5-point epsilon — with Sharpe also a hair below. Two
metrics miss, and the rule allows one. This is the closest call so far, and it is exactly the case
the epsilon was pre-declared for: the answer is no, and it is not re-run with a different epsilon.

### P2-K ENABLE_CORR_CAP — **REJECT**, flag stays off

Refuse a candidate correlating above 0.75 with anything held, on 4h returns over 180 bars, measured
against holdings **and** anything already queued on the same bar. An unmeasurable correlation
allows the trade, which is the deterministic choice for newly listed coins.

| Panel | CAGR | Sharpe | Max DD | Trades |
|---|---|---|---|---|
| standard baseline | 20.71% | 1.371 | −15.09% | 398 |
| standard + corr cap | **19.19%** | 1.384 | −14.82% | 380 |
| held-out baseline | 14.73% | 1.100 | −15.09% | 233 |
| held-out + corr cap | 14.35% | 1.113 | **−11.65%** | 224 |

The mirror image of vol targeting: it passes the held-out half cleanly and fails the standard panel
on CAGR. Worth recording that it does what it claims — the held-out drawdown improves by 3.4 points
— but the standard panel has no epsilon, so it stays off.

## P3 — ranker weights, the model, and the orphan metrics

### P3-N ENABLE_TAKER_MIN at 0.5 — **SHIP**, enabled in paper

Rule: on the breakout bar, at least half of the day's volume must be buyer-initiated
(`taker_ratio >= 0.5`). The metric already existed and fed nothing but the rejected model.

| Panel | CAGR | Sharpe | Max DD | Trades | Win | p_coin | p_time |
|---|---|---|---|---|---|---|---|
| standard baseline | 20.71% | 1.371 | −15.09% | 398 | 39.7% | 0.040 | 0.004 |
| standard + taker | **20.73%** | **1.430** | **−13.24%** | 366 | 41.3% | **0.006** | 0.002 |
| held-out baseline | 14.73% | 1.100 | −15.09% | 233 | 40.8% | 0.196 | 0.322 |
| held-out + taker | **15.48%** | **1.223** | **−13.24%** | 210 | 42.4% | **0.026** | 0.126 |
| pit30 baseline | 7.84% | 0.697 | −19.35% | 591 | 38.2% | 0.012 | 0.112 |
| pit30 + taker | **10.18%** | **0.880** | **−17.13%** | 543 | 40.7% | **0.006** | **0.052** |

Passes every criterion on every panel, and it is the only candidate in the whole program that
improves **all three metrics plus both significance measures simultaneously on all three panels**.
On the survivorship-free panel it adds 2.3 points of CAGR and 0.18 of Sharpe while taking 48 fewer
trades, and it drags the random-timing p-value to 0.052 — the first time this strategy has come
close to showing entry-timing skill.

Interpretation: a breakout where sellers are hitting the bid is a different event from one where
buyers are lifting the offer, even when the price prints the same number.

### P3-L ENABLE_WF_WEIGHTS — **REJECT**, shipped weights stand

29 folds, 730-day train / 91-day test with a purge gap, 19 weight sets in a small simplex around
0.35 / 0.25 / 0.20 / 0.20, chosen by training-window information coefficient and frozen for the
next fold only.

| Run | CAGR | Sharpe | Max DD | Trades |
|---|---|---|---|---|
| shipped weights | 23.68% | 1.454 | −15.09% | 373 |
| walk-forward weights | **22.30%** | **1.413** | **−16.29%** | 367 |

The fold criterion was met — 16 of 29 folds beat the baseline on all three metrics, longest run 5 —
but the aggregate out-of-sample result is worse on all three, and the master ship rule governs.
16/29 is a coin flip, and a run of 5 in 29 draws at that rate is unremarkable.

The sharpest detail: **the shipped weights were never chosen in a single fold**, and they still
won out of sample. Training-window IC does not predict what makes money here, which agrees with
the trade study finding that the score's own level had no reliable relation to trade outcome.

**P3-L variant: component C replaced by acceleration, percentile(rs_7d − rs_30d)** — **REJECT**

| Run | CAGR | Sharpe | Max DD | Trades |
|---|---|---|---|---|
| shipped weights | 23.68% | 1.454 | −15.09% | 373 |
| walk-forward, acceleration | **21.25%** | **1.340** | **−15.45%** | 413 |

Worse than both the shipped weights and the dist_sma50 version of the search: 9 of 29 folds beat
the baseline against 16, longest run 3 against 5. Acceleration is not a better third component
here, and re-fitting the blend still does not transfer out of sample. As before, the shipped
weights were chosen in zero folds and still won.

### P3-M ENABLE_ML_VETO — **REJECT**, model stays disconnected

`entry_ok &= ml_score > 0`, out-of-sample from 15 Nov 2019, 227,089 scored rows, both sides
measured over identical bars.

| Run | CAGR | Sharpe | Max DD | Trades | Win | p_coin |
|---|---|---|---|---|---|---|
| baseline | 24.71% | 1.503 | −15.09% | 364 | 39.8% | 0.060 |
| + ml veto | **16.74%** | **1.310** | −13.06% | 254 | 41.7% | 0.020 |

It behaves exactly like every other "be more selective" idea here: fewer trades, higher win rate,
better drawdown, a third less money. The model remains disconnected, which is an acceptable result.

### P3-N ENABLE_ATR_BAND — **REJECT**

ATR% between the universe's 20th and 80th percentile that bar. Standard CAGR 20.71% → 12.04%,
Sharpe 1.371 → 0.975, DD −15.09% → −26.30%, p_coin 0.040 → 0.172. Trimming the volatile tail
removes the trades that pay.

### P3-N ENABLE_LATE_BREAKOUT — **REJECT, and the rule is degenerate**

Proposed: reject when `don_pos > 0.95 AND volume_ratio < 1`. Measured on the panel:
**100% of 13,189 breakout bars have don_pos ≥ 1.0, minimum exactly 1.000.** A close above the
prior 20-bar high is the top of the channel by construction, so the first clause is always true
and the rule reduces to `volume_ratio >= 1.0` — already tested and rejected. Its numbers are
identical to that run, which is what prompted the check rather than a shrug.

## Faster decisions, autonomy, and a model trained on the bot's own trades (4 Oct 2026)

Requested: decide continuously rather than every four hours, trade without the Telegram approval
step, and research an ML method that makes money.

### Track A — deciding more often: **REJECT** on the evidence, offered as a switch

Two designs, both over the same nine years, same shipped filters, same costs:

| Run | CAGR | Sharpe | Max DD | Trades/yr | Avg trade | p_coin |
|---|---|---|---|---|---|---|
| 4h close (shipped) | **20.71%** | **1.428** | **−13.24%** | 40.2 | **+4.8%** | **0.006** |
| 1h decisions (same horizons) | 17.91% | 1.271 | −16.03% | 46.9 | +3.6% | 0.038 |
| 1h trigger on the 4h channel | 16.57% | 1.197 | −14.87% | 46.7 | +3.2% | 0.098 |

Acting sooner is worse in every column, and the reason is visible in the average trade: it falls
from +4.8% to +3.2%. **The four-hour close is not a delay, it is a filter** — many breakouts that
look alive mid-bar are dead by the close, and buying them earlier buys more failures. The second
design also loses coin-selection significance entirely (p_coin 0.006 → 0.098).

`DECISION_TF` now accepts `1h` or `4h`, defaulting to the validated `4h`, with the cost documented
in `.env.example`. The horizons are identical in wall-clock terms either way: a 20-bar channel on
4h bars becomes an 80-bar channel on 1h bars.

### Track B — autonomous live entries: **IMPLEMENTED**

`AUTO_TRADE=true` removes the human approval step in live mode, so the bot enters by itself.
Telegram still reports every fill, and `FLAT`, `STATUS` and the kill switch still work. Paper and
demo were never gated. Exits were never gated in any mode. Default is off, and it has no effect
while `MODE=paper`.

### Track C — a model trained on the bot's own trades: **REJECT**

Earlier ML attempts predicted next-day excess return for every coin. This one asks the narrower
question that matters: *given a candidate the rules already like, will this trade make money?*

2,255 candidate trades labelled with their real outcome — entered at the next open, exited by the
live exit rule, costs on both legs. 44.0% win rate, mean +5.30%, median −2.09%. Walk-forward with
a 30-day purge, 22 folds.

**The model works, and that is the problem.** Out of sample, by predicted win probability:

| Quartile | Trades | Win rate | **Mean return** |
|---|---|---|---|
| lowest | 514 | 43.6% | **+8.79%** |
| low | 513 | 41.1% | +3.79% |
| high | 513 | 44.3% | +5.87% |
| highest | 513 | **46.0%** | **+3.65%** |

It predicts win rate — and win rate is **inversely** related to return. The trades most likely to
win are the small ones. Retraining on return instead of probability sharpened the correlation
(+0.026 → +0.069) and left the inversion intact: its top quartile wins 48.0% of the time and earns
+4.77%, its bottom quartile wins 37.6% and earns +5.83%.

Six formulations, all measured from 29 Apr 2020:

| Run | CAGR | Sharpe | Max DD | Trades | Win |
|---|---|---|---|---|---|
| baseline | **26.51%** | **1.622** | **−13.24%** | 327 | 41.3% |
| veto p_win > 0.40 / 0.45 / 0.50 | 16.16 / 14.98 / 13.46% | 1.29 / 1.24 / 1.22 | ≈ −14% | 230 / 219 / 197 | up to 46.2% |
| veto e_ret > 0 / 0.02 | 18.48% | 1.35 / 1.38 | −13.2 / −13.9% | 283 / 258 | up to 45.7% |
| rank by e_ret | 25.25% | 1.595 | −16.98% | 532 | 39.3% |
| rank by p_win | 24.02% | 1.539 | −16.39% | 531 | 39.4% |
| **size by e_ret** | **30.56%** | 1.460 | **−20.67%** | 335 | 41.2% |

Every filtering variant raises the win rate and destroys the return. Sizing — the one formulation
that deletes no trades — is the only one to beat the baseline on CAGR, and it pays for it with
Sharpe 1.62 → 1.46 and drawdown −13.2% → −20.7%. It fails the ship rule, and it fails it in the
most informative way: the model's edge is real but far too weak to justify the variance it adds.

**Conclusion after six ML formulations and eleven rule-based filters.** This strategy's return
lives in a small number of very large winners that are not identifiable from entry-time features.
Anything that removes candidates removes some of those winners, and the arithmetic never recovers.
The three things that did ship — up on its own, real volume, real buying — do not predict which
trade wins; they remove trades that were never real breakouts.

## P4 — how long to hold (4 Oct 2026)

The untested dimension. Everything before this changed which trade is taken; these change its
length. Baseline is the shipped state; one change each.

### Standard panel

| Run | CAGR | Sharpe | Max DD | Trades | Win | Avg bars | Best trade | p_coin |
|---|---|---|---|---|---|---|---|---|
| baseline | 20.73% | 1.430 | −13.24% | 366 | 41.3% | 47 | +155% | 0.006 |
| **hold through regime** | **21.93%** | **1.485** | **−12.62%** | 349 | 43.0% | 51 | +113% | 0.026 |
| exit channel 30 | 20.83% | 1.394 | −12.99% | 297 | 43.8% | 65 | **+253%** | 0.050 |
| exit channel 40 | 16.59% | 1.155 | −17.08% | 280 | 42.1% | 70 | +217% | 0.120 |
| exit channel 55 | 15.25% | 1.051 | −17.75% | 246 | 43.9% | 83 | +217% | 0.252 |
| exit rank 15 / 20 | 16.73% | 1.238 | −15.58% | 348 | 39.9% | 50 | +155% | 0.080 |
| time stop 12 bars | 19.02% | 1.371 | **−10.82%** | 487 | 27.9% | 29 | **+208%** | 0.004 |
| time stop 30 bars | 20.51% | 1.429 | −12.31% | 404 | 34.9% | 41 | +155% | 0.044 |

### Held-out 2022-2026

| Run | CAGR | Sharpe | Max DD |
|---|---|---|---|
| baseline | 15.48% | 1.223 | −13.24% |
| **hold through regime** | **15.84%** | **1.239** | **−12.62%** |
| time stop 12 bars | **17.98%** | **1.368** | **−10.82%** |
| time stop 30 bars | **16.31%** | **1.283** | **−11.78%** |
| exit channel 30 | 17.54% | 1.276 | −12.99% |

### P4 ENABLE_HOLD_THROUGH_REGIME — **REJECT** on the survivorship-free panel

A BTC regime flip stops new entries but no longer force-sells what is already held. It improved
all three metrics on the standard panel and the held-out half, then failed where it counts:

| Panel | CAGR | Sharpe | Max DD | p_coin |
|---|---|---|---|---|
| standard baseline | 20.73% | 1.430 | −13.24% | 0.006 |
| standard + hold | **21.93%** | **1.485** | **−12.62%** | 0.026 |
| held-out baseline | 15.48% | 1.223 | −13.24% | — |
| held-out + hold | **15.84%** | **1.239** | **−12.62%** | — |
| **pit30 baseline** | **10.18%** | **0.880** | **−17.13%** | 0.006 |
| **pit30 + hold** | 10.49% | **0.878** | **−18.88%** | 0.002 |

The honest panel pays the bill: drawdown worsens by 1.75 points for 0.3 points of CAGR, and
Sharpe is flat. The reason is exactly why the pit panel exists — holding alt positions through a
Bitcoin downtrend survives nicely on *today's* coins, which are the ones that lived. Across the
full universe, including everything later delisted, carrying alts into a bear market is what it
sounds like.

That is the third feature in this programme to pass both fast panels and die on the
survivorship-free one, after the resting stop and the volatility target.

### P4 time stops — **REJECT**, and the most interesting rejection in the log

Both close a position still underwater after N bars. Both **fail the standard panel on CAGR**
(19.02% and 20.51% against 20.73%) and both **beat the held-out half clearly** — time stop 12
takes held-out CAGR from 15.48% to 17.98%, Sharpe 1.223 to 1.368, and drawdown from −13.24% to
−10.82%, the best drawdown of any variant tested.

Note what it does to the shape: the win rate collapses to 27.9% while the best trade grows from
+155% to +208%. It is not picking better trades — it is killing dead ones early and freeing the
slot, which is the first thing in this entire programme to make the right tail bigger rather than
smaller. Time stop 30 is a rounding error from passing (CAGR −0.22 points, Sharpe −0.0002).

By the rule both stay off, and they are not re-run with a looser epsilon. If one dimension deserves
a second look later with fresh data, it is this one.

### P4 exit channels and rank thresholds — **REJECT**

Widening the exit channel to 30 bars nearly passes (Sharpe 1.394 against 1.430 is the only miss)
and produces the single best trade of any run at +253%, confirming that room helps winners. Wider
still (40, 55) is clearly worse, and both rank-threshold variants fail on three criteria.

## Live cycle and universe v2 (4 Oct 2026)

Both requested directly. One is a robustness fix with a measurable cause; the other is a change of
execution style whose closest measured proxy says it costs return. Recording which is which.

### Universe v2 — **SHIPPED**, a fix for an observed failure

On 4 Oct the list fell from 19 names to 14: ADA, LINK, TRX and XLM were each ejected on a single
thin 24-hour reading against a $20M floor. Nothing about those businesses changed overnight; the
measurement was simply a snapshot taken on a quiet day in a market-wide liquidity drought.

Three rules replace it:

| Rule | Effect |
|---|---|
| Judge on the **median of the last 7 readings**, not today's | one dead day cannot eject a coin |
| **Hysteresis**: an incumbent leaves only below 0.7x the floor | staying is easier than joining |
| **Never drop a held coin** for volume alone | the bot stops selling positions over a thin day |
| **Floor the list** at 12 names, topping up by volume | a drought cannot starve the strategy |

Re-run against the same live market: **17 coins instead of 14** — ADA, LINK and TRX return, while
HYPE still goes on a genuine $11.8M median. Recalls are labelled in `universe.json`, not silent.

This is not a return claim and was not measured as one: it is a fix for an identified fault in how
the list was being measured. The backtest universe (`pit.py`) already used trailing 30-day volume,
which is why this fault never showed up there.

### Live cycle — **IMPLEMENTED**, and the proxy measurement says it costs return

`DECISION_TF=live` checks the live price against frozen levels every `LIVE_POLL_SECONDS` (15 by
default), so an entry or exit can happen at any second rather than on a 4-hour boundary.

What is frozen and what is live, precisely:

- **frozen at each 4h close**: the breakout level, the exit level, the stop, the ranking, the
  regime and all three shipped filters. Nothing is computed from an unfinished candle.
- **live**: only the price, and only the question "has it crossed a number decided on a closed bar".

Safeguards: exits are checked before entries; a symbol is acted on **once per bar**, so a price
oscillating around a level cannot churn the account; slot limits, daily halt, drawdown kill, safe
mode and the approval gate all still apply; fills are at the live price plus slippage.

**The honest expectation.** The nearest thing to this that can be backtested is hourly triggering
on the same 4h channel, measured on 4 Oct: CAGR 16.57% against 20.71%, Sharpe 1.197 against 1.428,
and coin-selection significance lost entirely (p_coin 0.006 → 0.098). Tick-level triggering cannot
be backtested on the data held here, so it is not validated — it is a deliberate choice to act
sooner, with the measured cost of acting sooner already on the record.

8 new tests cover it, including that the levels come only from closed-bar columns.

## P5 — adaptive sizing, and the minute question (4 Oct 2026)

### Can it trade on the minute? **Measured on real 1-minute candles**

Five liquid coins, 120 days of genuine 1m data, costs on both legs.

| Design | Return (120d) | Sharpe | Max DD | Trades | Fees paid |
|---|---|---|---|---|---|
| Same 3.3-day channel, decided every minute | **+4.56%** | 1.375 | −4.8% | 18 | $50 (0.5%) |
| True minute-scale: 80-minute channel | **−16.58%** | −3.68 | −19.4% | 210 | **$1,142 (11.4%)** |

The minute-scale strategy loses 16.6% in four months and **pays 11.4% of the account in fees**.
Its average trade is −0.28% against a round-trip cost of 0.27%: it is not a strategy, it is a
transfer to the exchange. The first row is the shipped strategy checked more often, which is what
the live cycle already does — and better, at ten-second polling.

### P5-A ENABLE_REGIME_RISK — **REJECT**

Half the risk in the regimes the report says bleed (chop, recovery, bear).

| Panel | CAGR | Sharpe | Max DD |
|---|---|---|---|
| standard baseline | 20.71% | 1.371 | −15.09% |
| standard + regime risk | 19.33% | 1.373 | **−14.06%** |
| held-out baseline | 14.73% | 1.100 | −15.09% |
| held-out + regime risk | 11.69% | 1.008 | **−14.06%** |

### P5-B ENABLE_EQUITY_CURVE_RISK — **REJECT**

Half the risk while the last ten closed trades average negative.

| Panel | CAGR | Sharpe | Max DD |
|---|---|---|---|
| standard baseline | 20.71% | 1.371 | −15.09% |
| standard + equity curve | 16.97% | 1.315 | **−11.85%** |
| held-out baseline | 14.73% | 1.100 | −15.09% |
| held-out + equity curve | 12.67% | 1.090 | **−11.85%** |

Both do exactly what they promise — the drawdown improves by 1 to 3.2 points — and both cost more
return than the comfort is worth. This is now the most repeated result in the log: **every attempt
to be more cautious has cost more return than the drawdown it saved**, whether it works by removing
candidates or by shrinking them.

The one exception across the whole programme runs the other way: sizing **up** on the model's
predicted return lifted CAGR to 30.56% and took drawdown to −20.67%. More risk bought more return;
less risk bought less. The strategy is already close to the efficient frontier of its own family,
and the remaining choice is how much of it to hold, not how to be cleverer inside it.

## The nine-family profit search (4 Oct 2026)

Nine strategy families explored in parallel by independent agents, ~265 variants, every claimed win
attacked by three adversarial verifiers (lookahead, independent reproduction, robustness).

**Result: nothing beat the shipped bot. Two variants passed the ship rule mechanically; both died
under verification.**

Correct baseline for all of it: standard **CAGR 0.2073 / Sharpe 1.4295 / maxDD -0.1324 / 366
trades / p_coin 0.006**; held-out **0.1548 / 1.2229 / -0.1324 / 210**; pit30 **0.1018 / 0.8796 /
-0.1713 / p_coin 0.006**.

| Family | Best variant | Standard CAGR/Sharpe/DD | Held-out | Verdict |
|---|---|---|---|---|
| Universe size | top-15 by trailing volume | 0.2121 / 1.4507 / -0.1115 | 0.1636 / 1.2688 / -0.1115 | **passed, then refuted** |
| Tighter rank exit (lead) | exit_rank=5 | 0.2080 / 1.5050 / -0.1176 | 0.1570 / 1.3230 / -0.1176 | **passed, then refuted** |
| Vol-scaled channel | Donchian AND ATR band k=1 N=20 | 0.2041 / 1.5800 / -0.1064 | 0.1928 / 1.5404 / -0.1010 | reject, CAGR short 0.32pp |
| ML right tail | +20% classifier as size scaler | 0.2539 / 1.5004 / -0.1664 | 0.2370 / 1.5120 / -0.1664 | reject, drawdown; **tail AUC 0.4914** |
| Portfolio | 6 slots, conviction tilt | 0.2718 / 1.4168 / -0.1811 | 0.2527 / 1.3598 / -0.1811 | reject, Sharpe and DD |
| Pure rotation | no breakout trigger | 0.2650 / 1.3847 / -0.2385 | 0.2196 / 1.2112 / -0.2162 | reject, DD 6.5-8.8pp worse |
| Regime switching | 1.15x size in strong bull | 0.2102 / 1.3648 / -0.1472 | 0.1584 / 1.2018 / -0.1472 | reject, and the premise was false |
| Ensemble voting | k=2 of 3 signals | 0.2053 / 1.4111 / -0.1419 | 0.1452 / 1.1586 / -0.1419 | reject on all three |
| Longer horizons | shipped rank + rs_90d>0 | 0.1641 / 1.3983 / -0.1063 | 0.1150 / 1.1547 / -0.0963 | reject, -4.3pp CAGR |
| Reversal | bottom-5 by score | 0.0768 / 0.8537 / -0.1783 | 0.0598 / 0.7667 / -0.1279 | reject; p_coin 0.395 |

### The five things the failures establish

1. **The right tail is not predictable from OHLCV.** 16.3% of trades beat +20% and carry **155% of
   all return**; 5.6% beat +50% and carry 91%. A classifier trained directly on "will this beat
   +20%" scored **AUC 0.4914 out of sample, worse than a coin flip**, with the share of big winners
   flat across its predicted quartiles. Third independent confirmation. Stop trying.
2. **Every "more return" knob is leverage.** Across 43 portfolio runs exposure explains CAGR
   (r=+0.90) and drawdown (r=+0.93) nearly equally, but drawdown rises **1.29x faster** per unit of
   exposure, because the 8% kill switch fires more often (9 firings at 1% risk, 48 at 3%).
3. **The breakout trigger earns its keep.** Removing it raises CAGR 3.15pp but cuts the average
   trade from +4.80% to +3.19% and the win rate 41.3% to 38.7%; the gain is time-at-risk, and
   exposure-matched it loses with 9.4pp more drawdown.
4. **The old "+13.4% on top-10" universe finding does not transfer.** It reproduces on the
   point-in-time panel (top-10 0.1693 vs top-30 0.0990) but not on the pinned list, because the
   shipped builder already excludes exactly what the pit top-30 is stuffed with. That finding was
   measuring a filter we already have.
5. **"Passes both panels" is weak evidence, now quantified.** Placebo nulls, information-free
   universe cuts with matched persistence, **pass the same ship rule 7.5% to 22% of the time**.
   And the two panels are not independent: the worst drawdown in nine years falls after 2022, so
   standard and held-out max drawdown are the same episode for nearly every variant.

### Two corrections to this log

**The regime "chop bleed" was a misreading, and it sent me down a dead end.** The P5 regime-risk
hypothesis rested on the regime table showing -6.5%/yr in sideways/chop. That figure is
**mark-to-market attribution by bar**: positions opened in a bull regime marking down while the
market chops. Measured by the regime a trade was **entered** in:

| Regime at entry | Trades | Total P/L | Avg trade |
|---|---|---|---|
| strong bull | 192 | +$24,687 | +7.21% |
| weak bull | 88 | +$9,969 | +3.18% |
| **sideways/chop** | **80** | **+$6,436** | **+1.09%** |
| recovery | 3 | +$76 | +1.76% |
| bear | 3 | +$15 | -0.09% |

Chop entries **make money**. Refusing them costs 2.8 CAGR points and makes drawdown 6 points worse.
Where this log says the strategy bleeds in chop, it means its open positions mark down during chop
bars, not that trading in chop loses money.

**exit_rank=5 was never tested in P4.** That study tested only looser exits (15, 20) on the
assumption that holding longer helps. Tighter was the untested direction, and it clears the ship
rule on both panels. It still fails: the CAGR margin is **7 basis points over 9.1 years**, the
entire edge is November-December 2024, neighbouring grid cells (4, 6, 7, 8, 9) fail, and it dies
at **1.5x costs** while trading 548 times against 366. It is a genuine risk-shape improvement,
Sharpe +0.076/+0.100 and drawdown +1.48pp with p_coin 0.000, and zero return improvement.

### New rules adopted from this search

- **Placebo null required.** Any future claim must be scored against 40+ information-free variants
  of the same shape, with its percentile reported.
- **Sub-period attribution required.** If the differing trades concentrate in one quarter, it is noise.
- **Cost stress added to the ship rule.** A variant that raises turnover must also pass at 1.5x and
  2x costs. exit_rank=5 would have been rejected on this in one run.

---

## FIXLOG

**27 Sep 2026 — parity gate reported DRIFT (max abs diff 6,959).**
Classification: **neither a code bug nor a flag leak — an invalidated baseline artifact.**
`reports/_parity_baseline_equity.csv` was frozen against an 18-coin universe; `data/universe.json`
was rebuilt to 19 coins (HYPE and ONDO in, HBAR out) by a smoke test later the same day, so the
comparison was between two different coin lists.

Proof it was not a regression: the baseline commit `f6ffdaf` was checked out into a temporary
worktree and run against today's 19-coin universe. Current code with every flag off reproduces it
to **7.3e-12** over 19,935 bars — floating-point noise, not behaviour.

Fix: the universe is now pinned in `reports/_parity_universe.json` and `research/parity_check.py`
reads that file rather than `data/universe.json`, so a daily rebuild can no longer masquerade as a
code change. The baseline equity file was re-pinned to the same universe, verified identical under
both codebases before replacing it.

---

## Still open

- **P1** absolute momentum, BTC sleeve, breadth gate, volume hard filter
- **P2** two-stage exit, volatility targeting, correlation cap
- **P3** walk-forward ranker weights, ML veto, orphan-metric filters
- **Stress tests** beyond the existing offline/delist/429/lost-response set
- The live/demo stop path is covered by unit tests against fakes only. **No order of any kind has
  ever been sent to Binance**, including the demo environment.

**27 Sep 2026 — ship-rule implementation was stricter than the rule.**
Classification: **ship-rule miss (my bug, not a strategy result).** `research/compare.py:verdict`
applied the p_coin criterion to the held-out panel as well, and treated the held-out epsilon as a
blanket tolerance rather than "at most one metric may be flat".

Effect: `volume_min=0.8` was reported REJECT when the stated rule makes it PASS — its held-out CAGR
is 14.73% against a 15.07% baseline, a 0.34-point gap inside the pre-declared 0.5-point epsilon,
with Sharpe and drawdown both better.

Fix: `verdict()` now counts how many metrics lean on the epsilon and refuses more than one, and
checks p_coin on the standard and survivorship-free panels only. Re-ran all three thresholds under
the corrected rule; 1.0 and 1.2 remain rejected on their own numbers, which the fix did not touch.

**27 Sep 2026 — the research baseline could not express a shipped float flag.**
Classification: **implementation bug in the harness, not in the bot.** `research/compare.py`
accepted `--baseline-flags` as bare names only, so once `volume_min=0.8` shipped, the "baseline"
it compared against was still `abs_momentum` alone, and every later candidate would have been
measured against a state the bot no longer runs.

Caught on the first candidate after the ship (btc_sleeve). Fix: `--baseline-flags` now parses
`name=value` and defaults to `abs_momentum,volume_min=0.8`. The sleeve was re-measured against the
true shipped baseline; it failed either way, and the recorded numbers are the corrected ones.

**4 Oct 2026 — the time stop never fired.**
Classification: **implementation bug, mine.** `exit_reason` tested `in_profit is False`, but the
backtest passes a numpy bool, which is not the Python `False` singleton, so the condition was
never true and the rule was silently dead.

Caught by a smell rather than a crash: the time-stop runs returned numbers **identical** to the
baseline in every column, which a real change never does. Fixed to `not in_profit` with an explicit
None check, regression test added with numpy bools, and both variants re-measured. The corrected
results are the ones recorded above, and they changed the verdict from "no effect" to a genuine
near-miss.

**4 Oct 2026 — the volume median counted today twice.**
Classification: **implementation bug, mine.** `volume_history()` records the current reading, and
`build()` then passed that same reading to `typical_volume()` as `latest`, so the day being
smoothed got double weight in its own median.

Caught while writing the tests for universe v2 rather than by a failure. Fixed by passing `None`;
the history already contains it.

**4 Oct 2026 - the research harness compared against a baseline the bot no longer ran.**
Classification: **implementation bug, mine, with consequences for reported numbers.**
research/compare.py defaulted --baseline-flags to abs_momentum,volume_min=0.8. When taker_min=0.5
shipped on 27 Sep that default was never updated, so every comparison run through that script
afterwards was scored against **CAGR 0.2071 / Sharpe 1.3712 / maxDD -0.1509** instead of the real
**0.2073 / 1.4295 / -0.1324**. The wrong row is 0.058 Sharpe and 1.85 drawdown points easier to beat.

Caught by eight of the nine search agents independently, each checking the brief against the raw
experiment files rather than trusting it.

Affected: regime_risk and equity_curve_risk (both 4 Oct). **No verdict changes**, both were
rejections and fail the true baseline by more. One reported detail was wrong: regime_risk drawdown
was described as an improvement (-0.1406 against -0.1509); against the real baseline of -0.1324 it
is **worse**, so that variant fails all three metrics rather than two.

Fixed: the default now reads abs_momentum,volume_min=0.8,taker_min=0.5.

---

## P6 — predicting the LOSSES instead of the returns (10 Oct 2026)

Asked for "a model which precisely picks the coins and loses less". Every model in this repo until
now predicted something good and tried to buy more of it; none of them worked. This program asked
the mirror question, which had never been tested: can the model predict which entries are about to
**lose**, so that less money goes into them?

### The finding, and it is the first real one from any model here

**Losses are forecastable. Returns are not.** Out of sample, 1,587 candidates from 2020-04-29,
walk-forward 730d train / 91d test with a 30-day purge gap, same features and hyperparameters as
`research/candidate_model.py`:

| target | what it predicts | out-of-sample skill |
|---|---|---|
| `loss20` | P(net < −20%) | **AUC 0.7819**, Brier 0.0473 (base rate 3.5%) |
| `loss10` | P(net < −10%) | **AUC 0.6625**, Brier 0.1756 (base rate 18.5%) |
| `mae` | maximum adverse excursion | **Spearman +0.3976** vs realised |
| `downside` | min(net, 0) | Spearman +0.2927 vs realised MAE |

For contrast, every return-prediction model in this log scored AUC 0.4688–0.5219, i.e. nothing.
The MAE model at ρ = +0.40 is the strongest out-of-sample signal any model in this repository has
produced.

### Why it cannot be turned into money, stated as plainly as the numbers allow

The model is reading **volatility**, and volatility is where the payoff is. The quartile tables are
the whole story — the trades it correctly flags as dangerous are the trades that pay:

| quartile by predicted `loss10` | trades | mean net | mean MAE | share losing >10% |
|---|---|---|---|---|
| safest | 397 | **+6.80%** | −5.54% | 5.5% |
| safe | 397 | +2.28% | −8.67% | 18.4% |
| risky | 396 | +8.16% | −8.82% | 19.4% |
| riskiest | 397 | **+9.60%** | −12.42% | **30.5%** |

The danger ranking is real and monotone — 5.5% → 30.5% of trades losing more than 10%. The return
ranking runs the other way. Ranking by predicted MAE says the same thing from the other side: the
shallowest-expected quartile earned +9.75%, the deepest-expected +6.73%.

This is the cleanest statement yet of a result this log has circled three times ("win rate runs
inversely to profit", P4 time stops, P5 regime risk): **in this system, avoiding losses and keeping
returns are the same choice, and you cannot have both.**

### What was tested on top of that, and rejected

55 configurations across three scripts. Baseline (shipped flags, from 2020-04-29):
**CAGR 0.2731 / Sharpe 1.6851 / maxDD −0.1017 / 298 trades / win 40.9%.**

**P6-A `loss_model.py` — size DOWN by predicted loss, multiplier capped at [0.5, 1.0] — REJECT.**
Deliberately unable to lever up, so any gain had to come from removing risk where the losses were.

| run | CAGR | Sharpe | maxDD |
|---|---|---|---|
| size-down `loss10` | 0.2203 | 1.6808 | −0.0887 |
| size-down `mae` | 0.2279 | 1.6968 | −0.0934 |
| size-down `downside` | 0.2211 | 1.6911 | −0.0929 |
| *null:* const 0.75× | 0.1960 | 1.6101 | −0.1004 |
| *null:* const 0.50× | 0.1304 | 1.5969 | −0.0652 |

Every size-down beat its shuffled control and the constant nulls, so the model's ordering is doing
real work — and every one still costs 4.5–5.3 points of CAGR to buy 1–1.3 points of drawdown. The
same trade this log has refused five times.

**P6-B the one apparent pass, `veto worst 10% loss10` — REJECT after verification.**
It cleared the ship rule on the standard panel: CAGR 0.2816 ≥ 0.2731, Sharpe 1.7228 ≥ 1.6851,
maxDD −0.1017 (identical). It was one cell out of 16 and it removed **2 net trades out of 298**, so
`research/loss_model_verify.py` perturbed the two things that must not matter:

| veto fraction | 0.050 | 0.075 | **0.100** | 0.125 | 0.150 | 0.200 |
|---|---|---|---|---|---|---|
| ΔCAGR | +0.0002 | −0.0036 | **+0.0085** | −0.0179 | −0.0195 | −0.0277 |
| ΔSharpe | +0.0051 | −0.0031 | **+0.0377** | −0.0413 | −0.0471 | −0.0722 |

A single-cell spike with negative neighbours on both sides, degrading monotonically as the veto
widens. Survived 10 of 30 threshold × seed cells (33%); the pre-declared bar was 60%. The seed arm
was uninformative — `HistGradientBoosting` returned AUC 0.6625 for all five seeds, i.e. it is
deterministic on this data — so the threshold curve carried the whole verdict, and it rejects.

**P6-C `loss_risk_sizing.py` — normalise risk per trade with the MAE forecast — REJECT.**
The one formulation that selects nothing: size each entry so every trade carries the same forecast
drawdown, renormalised to mean 1.0 so average exposure is unchanged.

| run | CAGR | Sharpe | maxDD |
|---|---|---|---|
| forecast MAE | 0.2724 | 1.5770 | **−0.1503** |
| *arithmetic:* `vol_30d` | 0.2556 | 1.6288 | −0.1078 |
| *null:* shuffled | 0.2061–0.2624 | 1.5061–1.6150 | −0.0964…−0.1170 |

Fails the ship rule on all three metrics, and loses to the plain volatility ratio on Sharpe and
drawdown — **arithmetic beats the model again, now 5 of 6 studies.** Sizing by forecast depth
*raises* maxDD by half, because equalising forecast risk means putting more into the
shallow-expected names, which are the ones with no payoff. The three risk targets (0.08/0.10/0.12)
produce identical rows by construction: normalising to mean 1.0 removes the target constant.

### What this changes in the bot

Nothing. No flag was added; `app/` is untouched by this program. The loss model stays out of the
trading path.

What it does change is what can honestly be claimed about it. There now exists a validated
out-of-sample forecast of trade drawdown (AUC 0.78 on the −20% tail) that is **good enough to set
expectations and useless for selection**. The only defensible uses are reporting ones — telling the
operator that the current candidate sits in the 30%-chance-of-a-10%-loss bucket — and those do not
need a flag because they do not change a decision.

**10 Oct 2026 — the arithmetic control in P6-C was a no-op for a whole run.**
Classification: **implementation bug, mine, caught before the verdict was reported.**
`loss_risk_sizing.scaled()` clipped to [0.5, 1.5] **before** renormalising to mean 1.0. `vol_30d`
is annualised (median 0.826), so `target / vol` lands near 0.04 and the clip pinned **98.8%** of
rows to the 0.5 floor; dividing a near-constant by its own mean then produced exactly 1.0 on every
row. The "realised volatility" arm therefore reproduced the baseline bit for bit —
0.2731 / 1.6851 / −0.1017, 298 trades — and `beats_realised` was silently just a second copy of the
ship-rule test.

Caught by the tell that should always be checked: a control arm that matches the baseline in every
column to four decimals is not a result, it is a disconnected wire. The same tell caught the dead
time stop in P4.

Fixed by normalising before clipping, which also makes each arm scale-free — correct for a control
whose only job is to carry the *shape* of an ordering whose units are not comparable. The verdict
did not change (the forecast arm fails the ship rule outright, maxDD −0.1503 against −0.1017), but
one of the three reported columns was meaningless until the re-run.

---

## Calendar-year attribution — where the profit actually comes from (10 Oct 2026)

`research/year_attribution.py`, shipped flags, both panels rebuilt offline from the local cache.
This was never broken out before, and it is the most important table in this log.

### Survivorship-free (pit, 383 coins ever in the top 30) — the honest panel

| year | return | worst DD in year | trades | win rate |
|---|---|---|---|---|
| 2018 | −0.03% | −0.40% | 1 | 0% |
| 2019 | **+19.90%** | −9.93% | 48 | 33.3% |
| 2020 | **+37.10%** | −8.43% | 89 | 41.6% |
| 2021 | **+29.68%** | −8.87% | 106 | 44.3% |
| 2022 | 0.00% | 0.00% | **0** | — |
| 2023 | +5.80% | −13.40% | 100 | 40.0% |
| 2024 | **−2.90%** | −13.12% | 101 | 38.6% |
| 2025 | **−0.74%** | −9.83% | 85 | 41.2% |
| 2026 YTD | +8.96% | −3.72% | 16 | 50.0% |

Overall CAGR 9.90%, Sharpe 0.8565, maxDD −17.13%, 546 trades, ×2.37 over the span.
**Profitable years: 5 of 10.**

### The era split, which is the whole answer

| era | pit (survivorship-free) | standard panel |
|---|---|---|
| 2019–2021 | ×2.13 = **+28.70% a year** | ×2.82 = +41.20% a year |
| 2022–2026 | ×1.11 = **+2.13% a year** | ×1.98 = +14.58% a year |
| 2023–2025 only | ×1.02 = **+0.65% a year** | ×1.79 = +21.47% a year |
| drop the best year (2020) | 6.26% a year over 9 | 14.08% a year over 9 |

**The strategy's edge is 2019–2021.** Survivorship-free, the three most recent complete years
compound to +0.65% a year — flat, after costs, with 286 trades taken. 2020 alone is 36.6% of the
entire compounded result; removing it drops the CAGR from 9.90% to 6.26%.

The gap between the two columns is the measure of how much survivorship bias flatters the recent
era: on today's 19-coin list 2023–2025 looks like +21.47% a year, and on the real universe of the
time it was +0.65%. Almost the whole apparent recent performance is the coin list, not the rules.

### What 2022 says in the strategy's favour

Zero trades, zero drawdown, through the worst crypto bear year in the sample. The BTC 200-day
regime filter did exactly what it exists to do and kept the book in cash for twelve months. That is
a real property, and it is why maxDD is −17.13% rather than the −60%+ a long-only basket took.

### What this does not say

It does not say the strategy is broken: 2023–2025 is flat, not negative, and a trend-following
system is expected to go flat in a range. It does say that **no claim about this bot should be made
from its multi-year CAGR**, because that number is an average over one bull market and three
ordinary years, and the user will be trading in an ordinary year.

Pre-registered before any further work: the recent-era numbers above (2022–2026 pit, +2.13% a year)
are the bar any future change is judged against, in addition to the full-sample ship rule. A change
that improves the 2019–2021 era and not the recent one is fitting a market that has already gone.

**Discrepancy noted:** this run reports pit CAGR 9.90% where earlier entries quote 10.71% for
pit30. The cached pit universe has grown to 383 ever-top-30 coins and now extends into 2026, so the
two numbers are not the same panel. The 9.90% figure is the current one; the difference does not
affect any verdict, every one of which was a rejection by a wider margin.

---

## P7 - the trend core: the bot stops waiting (10 Oct 2026)

The brief: *"I want this entire bot to think and take action automatically, I dont want this bot to
see everything and do nothing"*, and before that, *"think about standing in the market which grows
with time and does not stop"* and *"consider latest technique, not 2019-2021 tech"*.

That is not a request for a better filter. Measured, the complaint is exact: over 2022-2026,
survivorship-free, the alt-rotation engine is **invested 5.8% of the time** and earns **+0.54% a
year**, and on the 2025-2026 half it earns **-2.06% a year** with an average trade of **-0.88%**.

### First: every full-sample result was scored on an objective dominated by 2019-2021

`research/modern_era.py` re-scores the shipped flags and the old rejections on 2022-2026 only,
split discover (2022-2024) / validate (2025-2026). The leave-one-out table is the headline:

| flag removed | modern CAGR | modern Sharpe | better WITHOUT it? |
|---|---|---|---|
| abs_momentum | +1.26% | 0.1777 | **yes** |
| volume_min | +2.30% | 0.2717 | **yes** |
| taker_min | **+6.00%** | **0.6010** | **yes** |
| funding_max_pct | +2.24% | 0.2631 | **yes** |

**All four shipped filters hurt in the modern era.** Each passed the full-sample ship rule on three
panels. They are 2019-2021 artifacts, and ~655 rejections were judged by the same contaminated
objective - a change that helped 2023-2025 but diluted 2020 was rejected for the wrong reason.
Re-scored here, `breadth` (+0.3329 Sharpe), `stop_exec` (+0.2717) and `hold_through_regime`
(+0.2421) all clear both modern halves having been rejected on the full sample.

### Then: the bot already computes the best signal it has, and trades something else with it

`research/btc_core.py`, BTC only, real costs, causality asserted:

| 2022-2026 | CAGR | Sharpe | maxDD | invested |
|---|---|---|---|---|
| the alt engine | +0.54% | 0.105 | -14.7% | **5.8%** |
| BTC buy and hold | +13.39% | 0.256 | -67.2% | 100% |
| **BTC above its own 200d SMA, else cash** | **+31.12%** | **0.829** | -37.9% | **51.8%** |

Still +26.06% at 3x costs; 27 round trips in 4.76 years. That rule **is** the bot regime filter.
It was being used as a permission slip to go hunting altcoin breakouts.

### What shipped: `ENABLE_TREND_CORE` (default OFF)

A **persistent position**, not a candidate. No breakout, no Donchian stop, no rank, no approval:
long each core coin while it closes above its own 200-day average, flat when it does not. It does
not consume the breakout slots, and the account-level controls still override it (drawdown kill,
kill-switch file, safe mode, non-TRADING status). In the real engine, on the pit panel:

| | modern CAGR | modern Sharpe | Calmar | HOLDOUT CAGR | HOLDOUT Sharpe | invested |
|---|---|---|---|---|---|---|
| baseline | +0.54% | 0.105 | 0.037 | **-2.06%** | **-0.206** | 5.8% |
| **core 0.25 BTC** | **+8.28%** | **0.604** | **0.415** | **+2.77%** | **+0.269** | **19.0%** |
| core 0.50 BTC | +8.44% | 0.577 | 0.393 | -0.55% | +0.034 | 21.1% |
| core 0.75 BTC+ETH | +8.86% | 0.487 | 0.293 | -0.88% | +0.058 | 29.9% |

0.50 and 0.75 are identical because `max_weight` caps them. Four of six variants improve Sharpe on
the modern era **and** the holdout. `core 0.25 BTC` is the only one that turns the holdout
profitable, and it is the smallest allocation - the opposite of a curve fit.

### Forward-robustness: 4 of 7, and it is reported as a FAIL

`research/forward_check.py` against the pre-declared bar:

| check | result |
|---|---|
| positive in >=70% of rolling 12-month windows | **FAIL** - 62.5% (baseline 50.0%) |
| worst window no worse than baseline | **FAIL** - -9.08% vs -8.97% |
| survives 1.5x costs | PASS - +7.41% CAGR |
| at 1.5x costs no worse than baseline | PASS - Sharpe 0.549 vs 0.256 |
| profitable on 2025-2026 | PASS - +2.77% |
| 2025-2026 no worse than baseline | PASS - Sharpe +0.269 vs -0.206 |
| beats random-coin null at p<=0.05 | **FAIL** - p=0.148 |

By the pre-declared standard this is **NOT forward-robust**, and it is not being claimed as such.
Two of the three failures are marginal (0.11 drawdown points; 62.5% against a 70% bar on only 16
overlapping windows). The placebo failure deserves the context that the **baseline fails it far
worse - p=0.662 against 0.148 - and the baseline average modern-era trade is -0.88% against the
core +0.17%.** The bar is one neither configuration meets; the core misses it by less.

So: a clear improvement on the thing that was asked about, short of the bar this log requires for a
default-on change. It ships **OFF**, with 174 tests passing and parity exact at 7.3e-12.

### What it does not fix

Exposure goes from 5.8% to 19.0%, not to the ~52% of the standalone rule. The slot limits, the
minimum order size and the alt engine own cash use all bite. "Stands in the market" is improved
threefold and is still not satisfied; closing the rest means letting the core run at a larger
weight than `max_weight` currently permits, which is a separate change and a separate test.

---

## P8 - the idle cash (10 Oct 2026)

*"Our paper money sits idle when 3 slots are full - suggest something so it can be utilised."*

Correct, and the measurement is worse than the complaint. On the pit panel, 2022-2026, shipped
flags, with `max_names=3  risk_pct=1%  max_weight=30%`:

| | value |
|---|---|
| mean position weight at entry | **6.90%** |
| so three full slots | ~21% of the book |
| fraction of bars holding anything at all | **39.5%** |
| mean exposure | **5.84%** |
| highest exposure ever reached | 52.81% |

Two problems wearing one coat, needing different answers:
  **(a)** on 60.5% of bars the engine holds *nothing*, because it is waiting for a breakout;
  **(b)** when it is as full as it gets, a 1%-risk position with a stop ~14% below entry is a 7%
  position, so three of them is a fifth of the account.

### The trap, stated before the numbers

Scaling a position up multiplies return **and** drawdown by the same factor. That is leverage, not
edge, and it only pays if the thing being scaled makes money. This engine earns +0.54% a year over
the modern era and **-2.06% on the 2025-2026 holdout**. Tripling a loss gives a bigger loss.

### B - more risk per slot: REJECT, and it is the worst option on the board

| | modern CAGR | modern maxDD | HOLDOUT CAGR | HOLDOUT maxDD |
|---|---|---|---|---|
| baseline 1% | +0.54% | -14.70% | -2.06% | -9.59% |
| risk 2% | -0.97% | -27.76% | **-7.46%** | -19.60% |
| risk 3% | -0.79% | -49.59% | **-15.32%** | -41.01% |

Exactly the predicted failure, and it is the intuitive answer, which is why it was worth measuring.
Deploying more cash into a losing engine is the single most expensive thing on this list.

### C - more slots: ACCEPT, the cheapest win in the log

| | modern CAGR | modern Calmar | HOLDOUT CAGR | HOLDOUT Sharpe | maxDD |
|---|---|---|---|---|---|
| baseline (3) | +0.54% | 0.037 | -2.06% | -0.206 | -14.70% |
| **5 slots** | **+3.21%** | 0.225 | **+3.67%** | **+0.376** | **-14.27%** |
| 8 slots | +2.21% | 0.156 | +2.55% | +0.277 | -14.17% |

Five slots gives the **best holdout return of anything tested** and its drawdown is *slightly
better* than the baseline's. It barely uses more cash (5.84% -> 7.24% exposure), which is the
point: it is not a cash-deployment fix at all, it is **diversification**. Same size per name, more
names, so no single loser dominates. This also reverses a decision made on 24 Sep, when raising
slots to 5 was waved off against a $100 book - on the modern era it is free.

### D - the trend core: ACCEPT at 0.25, REJECT above it

| | modern CAGR | modern Calmar | HOLDOUT CAGR | exposure | bars invested |
|---|---|---|---|---|---|
| baseline | +0.54% | 0.037 | -2.06% | 5.84% | 39.5% |
| **core 0.25** | **+8.28%** | **0.415** | **+2.77%** | **18.96%** | **48.9%** |
| core 0.50 | +4.13% | 0.116 | -0.92% | 28.49% | 44.8% |
| core 0.75 | +14.91% | 0.455 | **-3.15%** | 37.43% | 43.4% |
| core 0.50 BTC+ETH | +2.18% | 0.049 | **-10.84%** | 29.67% | 45.3% |

The core is the only option that genuinely deploys the idle cash: exposure triples and the book
holds something on 48.9% of bars instead of 39.5%.

**And a negative result on my own change.** P7 capped the core at `settings.max_weight`, which
silently clipped 0.50 and 0.75 to 0.30 and made them identical rows. That cap was removed here so
the weights mean what they say - and with it removed, **every core above 0.25 fails the holdout**,
0.75 by -3.15% with a -27.66% drawdown. The clip had been hiding a bad idea rather than a good one.
0.25 remains the only weight that works, and it is the smallest on the grid.

### E - both together: ACCEPT, and it is the answer to the question asked

| | modern CAGR | modern Sharpe | HOLDOUT CAGR | HOLDOUT Sharpe | exposure |
|---|---|---|---|---|---|
| baseline | +0.54% | 0.105 | -2.06% | -0.206 | 5.84% |
| core 0.25 alone | +8.28% | 0.604 | +2.77% | +0.269 | 18.96% |
| 5 slots alone | +3.21% | 0.331 | +3.67% | +0.376 | 7.24% |
| **core 0.25 + 5 slots** | **+7.37%** | **0.507** | **+2.92%** | **+0.264** | **20.01%** |
| core 0.35 + 5 slots | +8.14% | 0.507 | -0.33% | +0.065 | 23.80% |

Five variants raise both the modern era and the holdout with the holdout actually profitable:
`5 slots`, `8 slots`, `core 0.25`, `core 0.25 + 5 slots`, `core 0.25 + 8 slots`. Of those,
**core 0.25 + 5 slots puts the most money to work - 20.0% mean exposure against 5.84%** - while
keeping the holdout positive. Note the cliff immediately above it: 0.35 + 5 slots looks better on
the modern era and goes negative on the holdout.

### The honest summary

The idle cash is mostly **not a sizing problem**. 60.5% of the idleness is the engine having nothing
it is willing to hold, and the only fix for that is giving it something worth holding - which is
the core. The remaining fix is diversification, not leverage: five slots rather than bigger slots.

Everything here is measured on four years of discovery and under two of validation, and the drawdown
roughly doubles (-14.70% to -23.65%) for the recommended pair. Both flags stay **OFF** by default;
174 tests pass and parity is exact at 7.3e-12.

---

## P9 - the decision table, and a correction (10 Oct 2026)

*"What do you suggest?"* The two P8 recommendations had never been tested alongside dropping the
filters, so recommending them together would have been a guess. Pit panel, all three windows:

| config | modern CAGR | modern Sharpe | HOLDOUT CAGR | HOLDOUT Sharpe | full Sharpe | exposure |
|---|---|---|---|---|---|---|
| 1 today: 3 slots, 4 filters | +0.54% | 0.105 | **-2.06%** | -0.206 | 0.976 | 5.8% |
| 2 + core .25 + 5 slots | +7.37% | 0.507 | +2.92% | +0.264 | 0.848 | 20.0% |
| 3 drop `taker_min` only | +6.00% | 0.601 | +1.47% | +0.202 | 0.944 | 6.6% |
| **4 drop `taker_min` + core .25 + 5 slots** | **+10.76%** | **0.650** | **+4.27%** | **+0.335** | 0.910 | **20.8%** |
| 5 NO filters, 3 slots | -2.77% | -0.176 | **-10.09%** | -0.937 | 0.504 | 8.4% |
| 6 NO filters + core + 5 slots | +1.05% | 0.153 | **-17.33%** | -0.938 | 0.516 | 22.2% |

**Config 4 is the best configuration found in this project** on the modern era and on the holdout,
and it gives up only 0.066 of full-sample Sharpe to get there.

### The correction

P7 reported, from the leave-one-out table, that *"all four shipped filters hurt in the modern era"*.
That statement is true of each filter's **marginal** contribution and false as a conclusion about
the set. Removing all four is a catastrophe - holdout **-10.09%** at 3 slots and **-17.33%** with the
core - while removing `taker_min` alone is the single best one-line improvement available
(Sharpe 0.105 -> 0.601).

So `abs_momentum`, `volume_min` and `funding_max_pct` are doing real work; they are substitutes for
one another, which is why each looks removable while none of them is. Only `taker_min` should go.
Leave-one-out measures the margin, not the group, and the group was never tested until now.

### The holdout is now spent - declared, not discovered later

2025-2026 has been consulted across P7, P8 and P9: six variants, then fifteen, then six. It is no
longer a holdout, it is a validation set, and a further round of tuning against it would be
fitting it. **No more configurations are to be scored on 2025-2026.** The next honest test of
config 4 is forward paper and demo trading on bars that do not exist yet.

### What config 4 actually promises

+4.27% a year with a -18.15% drawdown on the most recent data, invested ~20% of the time. On a $100
book that is about $4 a year and a $18 dip along the way. It is a real improvement over a
configuration that currently loses money, and it is not a money machine.

---

## P10 - trade more often, without leverage (12 Oct 2026)

*"If we wait for all the levels to be met it can miss a lot ... we can also do futures trading and
set the bot so it does not sit idle, search pattern."*

Three proposals in one. They have very different evidence behind them.

### Futures / leverage - REJECT, and it is already measured

This is the same lever as `risk_pct`, with liquidation added. From P8, on the pit panel:

| | 2025-26 holdout CAGR | holdout maxDD |
|---|---|---|
| risk 1% (today) | -2.06% | -9.59% |
| risk 2% | **-7.46%** | -19.60% |
| risk 3% | **-15.32%** | **-41.01%** |

Leverage multiplies return and drawdown by the same factor, and this engine's recent-era return is
negative. Futures adds two costs on top that spot does not have - funding paid continuously by the
long side, and liquidation, which converts a drawdown the strategy would have recovered from into a
permanent loss. The repository is spot-only; adding margin, liquidation prices, funding accrual and
position mode is a large build in the live path, for a strategy whose best validated edge is about
+4% a year. **Not built, not recommended.**

### Sitting idle - already solved, never switched on

Config 4 raises exposure 5.8% -> 20.8% and the holdout from -2.06% to +4.27%. On 12 Oct the server
still had **zero** `DEMO_` lines in `.env`: the code is deployed and running, the flag was never
set. The books have held nothing since 8 Oct, which is config 1 behaving exactly as measured.

### "Search pattern" - measured, and the answer is NOT simply "lower the bar"

`research/more_signals.py`, every lever that produces more trades without touching position size.
**Scored on 2022-2024 only**; the 2025-2026 holdout was declared spent in P9 and is not consulted.

| variant | CAGR | Calmar | trades | exposure |
|---|---|---|---|---|
| baseline (d20, c5, 3 slots) | +2.35% | 0.210 | 183 | 6.2% |
| **donchian 10** | +0.68% | **0.036** | 235 | 6.9% |
| donchian 15 | +3.11% | 0.272 | 198 | 6.6% |
| candidates 12 | +5.31% | 0.412 | 245 | 7.8% |
| max_names 8 | +2.02% | **0.143** | 246 | 7.7% |
| no RS gate | +3.27% | 0.294 | 186 | 6.5% |
| core 0.25 + 5 slots | +9.97% | 0.422 | 238 | 20.9% |
| donchian 15 + core + 5 slots | +11.59% | 0.479 | 255 | 20.7% |
| **no RS gate + core + 5 slots** | **+11.87%** | **0.518** | 246 | **21.3%** |

**Lowering the bar too far makes it worse, not better.** A 10-bar breakout channel produces 52 more
trades and cuts Calmar from 0.210 to 0.036 - the extra signals are junk. Eight slots does the same
thing more mildly. So "more activity" is not the objective; the two variants that traded more and
lost ground prove the point better than the ones that worked.

Moderate loosening does look better - a 15-bar channel, a deeper ranking, or dropping the
"must beat BTC over 7 days" gate - and all of them improve most when combined with the core, which
is the piece that actually fixes idleness.

### Why none of this is a recommendation yet

Every row above is a **discover-window** number. There is no holdout left to check them against: it
was spent in P9. These are exactly the conditions that produced the four shipped filters, each of
which passed three panels and then turned out to be a 2019-2021 artifact.

So the only honest sequence is: ship the one configuration that **has** been validated forward
(config 4), let the demo book generate bars that nothing has been fitted to, and test these
candidates against that new data when there is enough of it. Any other order is fitting.


---

## P11 - what to do when the 200-day rule says no (13 Oct 2026)

*"Describe a rule for the long idle period when the 200-day rule is not fulfilled, to top that up
and get a good catch."*

The premise is right: BTC is below its own 200-day average on **44.7% of all bars** - about 1,488
days of the 3,335 in the sample. That is the time the book sits in cash.

### The idle stretches, and the one that matters most

| window | days | BTC over the window |
|---|---|---|
| 2018-03-14 .. 2019-04-02 | 384 | **-45.7%** |
| 2019-11-08 .. 2020-01-19 | 71 | +0.6% |
| 2020-03-08 .. 2020-04-29 | 51 | +0.9% |
| 2021-05-21 .. 2021-08-09 | 79 | +27.2% |
| 2022-01-01 .. 2023-01-14 | 378 | **-54.6%** |
| 2023-08-30 .. 2023-10-02 | 33 | +3.2% |
| **2025-11-03 .. 2026-08-20** | **289** | **-35.6%** |

The last row is the live answer to "why has this thing done nothing". For 289 days ending seven
weeks ago, BTC fell 35.6% and the book held nothing and lost nothing. The idle period was not
wasted capital; it was the strategy working.

### Four fallback rules, all REJECT

Each may act only while idle. Returns are across all idle time, with real costs.

| rule | total | positive in | at 2x costs |
|---|---|---|---|
| **cash** (baseline) | 0.00% | - | 0.00% |
| sma50 - hold BTC above its 50-day average | **-9.82%** | 4 of 7 windows | **-36.79%** |
| sma50 at half size | -5.04% | 4 of 7 | -20.49% |
| bounce - buy 15% below the 50-day average | -3.77% | 2 of 7 | -26.19% |
| alt_rs - hold the strongest alt by 30-day RS | **-100.00%** | 2 of 7 | **-100.00%** |

**Every one of them loses money, and `cash` beats all four.**

The per-window table shows exactly how. `sma50` is positive in four of seven windows - it does
catch the early recoveries - and then gives it all back and more in the two that matter:
**-23.23% in 2018 and -33.19% in 2022**. Small wins, huge losses: the shape of a rule that trades
against a downtrend. That is precisely the behaviour the 200-day filter exists to prevent, so this
is a direct measurement of whether its caution costs more than it saves. It does not.

`alt_rs` is the starkest. Holding the strongest altcoin through bear markets **loses the entire
account** - -99.98% in 2022 alone. In a bear market the strongest alt is still falling; relative
strength is not absolute strength.

### The conclusion, which is not the one that was hoped for

**Cash IS the rule for the idle period.** There is nothing to top it up with. The reason this
strategy makes anything at all over nine years is that it refuses to trade in the 44.7% of bars
where trading loses money, and every attempt here to "catch something" in that window converted a
flat line into a loss.

What legitimately reduces idleness is the opposite end: being invested MORE when the 200-day rule
says yes. That is the trend core, which raises time-in-market from 5.8% to ~20% and holds BTC on
48.9% of bars rather than waiting for a breakout. The idle half is not the part with money left in
it.

### Live cycle - already running

`DECISION_TF=live`, `LIVE_POLL_SECONDS=10`, `LEVEL_TF=1h` on the server: prices are checked every
ten seconds against levels frozen on the last closed hourly bar, and an entry or exit fires the
moment a level is crossed. It has not been on a 4-hour cycle since 4 Oct.


---

## P12 - engine isolation and an honest re-measurement (13 Oct 2026)

Work order: isolate the two engines, lock the measurement, then test risk changes. Steps 1 and 2.

### Step 1 - engine isolation

The owning engine is now **stored** on the position (`positions.engine`, and `queued_orders.engine`
so the tag survives the paper path's next-bar fill), rather than inferred. Both earlier inference
attempts were wrong, and each had already caused a bug:

| inferred by | why it broke |
|---|---|
| `reason.startswith("core")` | the live path writes `"live core: ..."`, which does not match, so a live core buy could never sell itself |
| `symbol in core_symbols()` | wrong for a coin the breakout engine bought BEFORE the core was switched on - that position silently lost its Donchian exit |

Exits are now strictly per engine:

    core      exits only on close < its own SMA200
    breakout  exits only on the Donchian low, rank > 10, or regime off

A migration adds the column to the books already live on the server; every row already there
defaults to `breakout`, which is what it is. `backtest.py` stores the same word, so a backtest
position and a paper position can no longer be described differently.

Six tests, including the two the brief asked for and the one that catches the symbol-based error:
a core position is never closed by a Donchian level, and a **breakout** position in a core symbol
keeps its Donchian exit.

### Step 2 - the measurement, re-run on the flags actually in force

`research/report_panels.py` prints both panels side by side for one flag set, reading the flags the
way the live bot reads them (`ENABLE_*`, with per-account overrides).

**Shipped configuration** - `abs_momentum + volume_min=0.8 + taker_min=0.5 + funding_max_pct=0.8`,
risk 1%, 3 names, 8% kill:

| panel | CAGR | Sharpe | maxDD | Calmar | exposure | trades |
|---|---|---|---|---|---|---|
| standard, full | 20.73% | 1.4295 | -13.24% | 1.5665 | 9.5% | 366 |
| standard, held-out 2022-2026 | 15.48% | 1.2229 | -13.24% | 1.1695 | 10.3% | 210 |
| **PIT, full** | **10.71%** | **0.9556** | **-14.70%** | 0.7289 | 6.3% | 509 |
| **PIT, held-out 2022-2026** | **0.54%** | **0.1048** | **-14.70%** | 0.0369 | 5.8% | 276 |

**The old "~7% CAGR, Sharpe ~0.6, maxDD -23%" is stale and is hereby retired.** With the volume and
funding filters on, the pit figure is **10.71% / 0.9556 / -14.70%** - better on all three. The
drawdown improvement is the largest part of it.

**Demo configuration** - the same plus `trend_core=0.25` on BTC, `taker_min` off:

| panel | CAGR | Sharpe | maxDD | Calmar | exposure | trades |
|---|---|---|---|---|---|---|
| standard, full | 19.58% | 1.0156 | -29.19% | 0.6708 | 19.3% | 448 |
| **PIT, full** | **16.96%** | **0.9437** | **-25.54%** | 0.6640 | 18.0% | 585 |
| **PIT, held-out 2022-2026** | **9.06%** | **0.6265** | **-21.63%** | 0.4190 | 19.1% | 310 |

Per-engine attribution on PIT full: **core 13,353 USDT (45%), breakout 16,330 USDT (55%)**, from 70
core trades out of 585. The core produces nearly half the profit from 12% of the trades.

The held-out comparison is the one that matters: **0.54% -> 9.06% CAGR and 0.1048 -> 0.6265 Sharpe,
for a drawdown of -21.63% against -14.70%.** More return per unit of drawdown (Calmar 0.419 against
0.037), and a deeper worst case.

### Measurement integrity, asserted rather than assumed

Six further tests, because none of these is visible in a CAGR:
- the breakout engine fills at the next bar's **open**, not the close that triggered it
- the **core** fills at the next open too (it queues into the same pending list)
- no order fills on the bar that decided it
- the 8% peak-to-trough kill fires **in the backtest**, not only live
- the daily loss halt fires in the backtest
- the kill flattens a **core** position too: the brakes are account-level

200 tests pass; flag-off parity exact at 7.3e-12.

### Steps 3 and 4 - capital slices and core risk: ALL REJECT

Baseline is the demo book's live configuration (`abs_momentum + volume_min=0.8 +
funding_max_pct=0.8 + trend_core=0.25` on BTC). Ship rule applied as written, not loosened.

**PIT panel - the one to quote:**

| variant | CAGR | Sharpe | maxDD | Calmar | exposure | trades |
|---|---|---|---|---|---|---|
| baseline (demo live) | **16.96%** | **0.9437** | -25.54% | 0.6640 | 18.0% | 585 |
| 3 slices | 11.15% | 0.7753 | -25.99% | 0.4290 | 16.2% | 454 |
| 4a vol cap 0.18 | 16.75% | 0.9491 | **-23.53%** | **0.7120** | 17.6% | 587 |
| 4b stretch cap 15% | 15.30% | 0.9455 | -22.90% | 0.6682 | 14.6% | 587 |
| 3+4a slices + vol cap | 12.04% | 0.8495 | -20.48% | 0.5877 | 16.0% | 454 |
| 3+4b slices + stretch | 11.91% | 0.9048 | **-19.29%** | 0.6176 | 13.4% | 449 |

**PIT held-out 2022-2026:**

| variant | CAGR | Sharpe | maxDD | Calmar |
|---|---|---|---|---|
| baseline | **9.06%** | 0.6265 | -21.63% | 0.4190 |
| 3 slices | 6.87% | 0.5853 | -20.48% | 0.3355 |
| 4a vol cap | 9.06% | 0.6265 | -21.63% | 0.4190 |
| 4b stretch cap | 7.90% | 0.6110 | -19.75% | 0.4002 |
| 3+4b slices + stretch | 8.45% | **0.7449** | **-18.68%** | **0.4526** |

| variant | standard | pit | held-out | pit DD < 30% | SHIPS |
|---|---|---|---|---|---|
| 3 slices | ✗ | ✗ | ✗ | ✓ | **no** |
| 4a vol cap 0.18 | ✗ | ✗ | ✓ | ✓ | **no** |
| 4b stretch cap 15% | ✗ | ✗ | ✗ | ✓ | **no** |
| 3+4a | ✗ | ✗ | ✗ | ✓ | **no** |
| 3+4b | ✗ | ✗ | ✗ | ✓ | **no** |

**Nothing ships. All five flags stay OFF.**

Three things worth recording rather than burying:

**4a came within 0.21 of a percentage point.** On pit it improves Sharpe (0.9491 vs 0.9437),
drawdown (-23.53% vs -25.54%) and Calmar (0.7120 vs 0.6640), and fails only on CAGR, by 0.0021.
The rule says all three must be no worse, so it is a REJECT and stays off. It is not re-run with a
different target, because tuning until it passes on the same sample is the failure mode this log
exists to prevent.

**4a is nearly inert on recent data, and the reason is arithmetic.** `w = min(0.18/vol, 0.25)` with
`trend_core` also 0.25 means the cap only reduces anything once vol exceeds 0.72. On the held-out
window its numbers are IDENTICAL to the baseline to four decimals - same CAGR, same Sharpe, same
310 trades - because it never bound. A flag that cannot act on the period that matters is not a
risk control, and that is a better reason to reject it than its CAGR.

**The brief's drawdown condition is not triggered.** Core pit drawdown is -25.54%, inside the 30%
limit, so no cap is forced on and no core change is rejected on that ground.

Every variant improves drawdown and most improve Sharpe; all cost CAGR. That is the same trade this
log has now refused eight times, and it is refused again here.

12 further tests cover the mechanics regardless of the verdict - a flag that is off but wrong is a
trap for whoever switches it on. They pin that the cap can only reduce and never lever, that a
missing or absurd volatility falls back to the flat weight, that the stretch gate blocks entries
and never closes a held position, that slices cap the breakout names and shrink its size, and that
every one of them off reproduces the baseline equity curve exactly.

218 tests pass; flag-off parity exact at 7.3e-12.

---

## P13 - per-engine P&L curves, and the core is not the risky one (14 Oct 2026)

The core's drawdown was the open question: adding it took the book's pit drawdown from -14.70% to
-25.54%, so the core looked like the dangerous half. That number does not say so - it is the BOOK's
drawdown while the core happened to be running, and it includes whatever the breakout engine was
doing at the same time.

`Result.engine_pnl` now carries a cumulative P&L curve per engine, realised plus open, starting at
zero. It is deliberately a P&L curve and not an equity curve: the engines share one pot of cash, so
any split of the STARTING capital between them would be invented. Each engine's own drawdown is
then measured in money against its own high-water mark, divided by starting equity so it is
comparable with the book's percentage.

### Each engine's own worst run

| panel | book maxDD | **core** own maxDD | **breakout** own maxDD |
|---|---|---|---|
| standard full | -29.19% | **-61.02%** | **-78.78%** |
| pit full | -25.54% | **-51.62%** | **-77.51%** |
| pit held-out 2022-26 | -21.63% | **-18.66%** | **-28.01%** |

**The breakout engine's own drawdown is worse than the core's, on every panel.** The book's
drawdown deepened when the core was switched on because the core puts more money to work - exposure
5.8% -> 18.0% - not because the core is the more dangerous engine. The suspicion was the wrong way
round.

Read those figures correctly: they are the worst peak-to-trough fall in that engine's own
cumulative P&L, against starting equity. The breakout engine's P&L curve reaches +163% of starting
equity on pit, so a -77.51% fall is **giving back accumulated profit**, not losing 77% of the
account. They do not sum to the book's drawdown, because the two engines' worst runs happen at
different times - which is the whole reason for measuring them apart.

### Who actually made the money

| panel | core P&L | breakout P&L | core share | reconciles |
|---|---|---|---|---|
| standard full | +99.3% | +310.0% | 24.3% | yes |
| pit full | +154.7% | +163.3% | 48.7% | yes |
| **pit held-out 2022-26** | **+30.8%** | **+20.3%** | **60.3%** | yes |

(P&L as a fraction of starting equity.)

On the full pit sample the two engines are near-equal partners. **On the held-out period the core
makes more than the breakout engine** - 60.3% of the profit - from a twelfth of the trades, while
having the shallower drawdown of the two.

Every row reconciles: the two columns add up to the book's actual profit to within 1e-6. Attribution
that does not add up is decoration, so that check is a test, not a note.

### What this does and does not license

It does not say the core should be made larger. P12 step 4 tested exactly that and every variant
above `trend_core=0.25` failed the ship rule, and the holdout is spent. It does say the core is not
the part to cap first if drawdown becomes the binding constraint - the breakout engine is.

7 tests, the strongest being that the columns reconcile with the book's profit after a crash has
closed everything.

225 tests pass; flag-off parity exact at 7.3e-12.

---

## P14 - SOL, XRP, BNB: medium holds and the cost of never selling at a loss (18 Oct 2026)

Asked for after two weeks of live losses: change technique, hold for a MEDIUM period, never sell
below the buying price, test on SOL/XRP/BNB, show a positive backtest. All four measured on
single-asset long-only runs with the live costs, filled at the next bar's open.

### 1. "Never sell at a loss" - REJECT, and the win rate is how it hides

It cannot lose a trade: every closed trade is a winner by construction. What it does instead is
refuse to close losers, so they stop being trades and start being a frozen portfolio. The money is
still gone, it is simply no longer counted.

**2025-2026, the market actually being traded:**

| coin | rule | win rate | CAGR | maxDD | bars underwater |
|---|---|---|---|---|---|
| SOL | donchian_med | 38.3% | -20.71% | -66.89% | 20.9% |
| SOL | **donchian_med + never sell at a loss** | **100.0%** | **-24.82%** | **-75.38%** | **91.7%** |
| XRP | donchian_med | 34.1% | -30.55% | -68.99% | 23.3% |
| XRP | **+ never sell at a loss** | **100.0%** | -26.13% | -72.72% | **94.0%** |
| BTC | donchian_med | 40.0% | -9.11% | -38.09% | 22.3% |
| BTC | **+ never sell at a loss** | **90.0%** | **-12.88%** | **-53.50%** | **86.9%** |

A **100% win rate alongside a -24.82% annual return**, with the position underwater on 92% of all
bars. That is the whole illusion in one row.

The starkest case is XRP `sma200_nl` in 2025-2026: **zero trades closed**, -37.60% return, 68% of
bars underwater. Never being allowed to sell at a loss meant never selling, so it simply held a
falling asset for the entire period.

Where it appears to help - XRP and BNB on the full modern era - it is not the no-loss logic doing
the work. Plain `sma200` whipsawed those two badly (80 and 70 trades at 5.0% and 8.6% win rates),
and refusing to sell accidentally suppressed the overtrading. The fix for whipsaw is a slower
filter, not a refusal to realise losses, and the next section is what that looks like.

### 2. Medium holds - measured, and they are the worst performer

`donchian_med` (buy a 20-bar high, sell a 30-bar low, 60-bar cap) holds for a **median of 6.75 to
10 days** - exactly the medium interval asked for. In 2025-2026 it is negative on **all five**
coins: SOL -20.71%, XRP -30.55%, BNB -22.17%, BTC -9.11%, ETH -15.80%.

### 3. What DID work: `sma50_200` - long only while above BOTH its 50-day and 200-day average

**2025-2026**, against buy-and-hold in the same window:

| coin | buy & hold | **sma50_200** | Sharpe | maxDD | trades | win rate |
|---|---|---|---|---|---|---|
| **BNB** | +6.38% | **+37.67%** | **1.1348** | -23.86% | 7 | 28.6% |
| **SOL** | -22.79% | **+30.62%** | 0.9337 | -22.58% | 6 | 16.7% |
| **ETH** | -11.63% | **+26.92%** | 1.0142 | **-14.63%** | 2 | 50.0% |
| BTC | -5.50% | +3.72% | 0.2784 | -17.03% | 10 | **0.0%** |
| XRP | -17.71% | -11.07% | -0.4486 | -33.05% | 12 | 0.0% |

Buy-and-hold was negative on four of the five coins; `sma50_200` was positive on three and beat
buy-and-hold on all five. It also holds up over the longer modern era - SOL +39.39%, ETH +21.63%,
BTC +30.08% with the best Calmar on the board at 1.0277.

### The finding that answers the complaint directly

**The winning strategies have terrible win rates.** `sma50_200` on BTC in 2025-2026 won **0 of 10
trades** and still returned +3.72%, beating buy-and-hold by 9 points. On SOL it won 1 of 6 and
returned +30.62%. Meanwhile the 100%-win-rate rule lost a quarter of the account.

The live paper book has won **1 trade in 5**. That is not evidence the strategy is broken; it is
the normal shape of trend following, and it is the same shape every profitable column above has.
Profit here comes from the size of the wins, not their frequency - which is also why "never sell at
a loss" destroys it: it caps the winners' only advantage while letting the losers run.

### Status

Exploratory. Five coins x six strategies x three eras is ninety cells, and picking the best of them
is selection. `sma50_200` being strongest in BOTH eras on BOTH SOL and BNB is more than a single
lucky cell, but it is not a validated result.

**It also used 2025-2026, which P9 declared spent.** Nothing here can ship on this evidence. What
it is good for is deciding what to forward-test next, on bars that do not exist yet.

---

## P15 - `core_dual_ma`: the first change to clear the ship rule since the core (18 Oct 2026)

P14 found, on single-asset runs, that requiring a coin to be above BOTH its 50-day and its 200-day
average beat buy-and-hold on all five of SOL, XRP, BNB, BTC and ETH in 2025-2026. This is that idea
put through the real engine and the full ship rule, nothing loosened.

### The result

Baseline is the demo book's live configuration (`abs_momentum + volume_min=0.8 +
funding_max_pct=0.8 + trend_core=0.25` on BTC).

| panel | | CAGR | Sharpe | maxDD | Calmar |
|---|---|---|---|---|---|
| **standard** | baseline | 19.58% | 1.0156 | -29.19% | 0.6708 |
| | **dual MA, BTC+ETH** | **27.03%** | **1.2480** | **-26.09%** | **1.0361** |
| **pit** | baseline | 16.96% | 0.9437 | -25.54% | 0.6640 |
| | **dual MA, BTC+ETH** | **21.55%** | **1.0971** | **-22.02%** | **0.9787** |
| **pit held-out 2022-26** | baseline | 9.06% | 0.6265 | -21.63% | 0.4190 |
| | **dual MA, BTC+ETH** | **12.45%** | **0.7758** | **-22.02%** | **0.5655** |

Better on CAGR, Sharpe **and** drawdown, on both panels, with the held-out improving too. That is
the whole rule, met without an epsilon.

### The decomposition, because it changes two things at once

| variant | pit CAGR | pit Sharpe | pit maxDD | held-out dCAGR | held-out dSharpe |
|---|---|---|---|---|---|
| A baseline: BTC only, 200-day | 16.96% | 0.9437 | -25.54% | - | - |
| B BTC only + dual MA | 17.63% | 1.0227 | **-21.23%** | **-0.0071** | **-0.0132** |
| C BTC+ETH, 200-day only | 17.04% | **0.8672** | **-28.34%** | +0.0220 | +0.0538 |
| **D BTC+ETH + dual MA** | **21.55%** | **1.0971** | -22.02% | **+0.0339** | **+0.1493** |

**Neither change passes on its own.** The dual MA alone fixes the drawdown (-25.54% -> -21.23%) and
slightly *worsens* the held-out. Adding ETH alone improves the held-out and *worsens* full-sample
Sharpe and drawdown. Together they complement: ETH supplies return and volatility, the second
moving average takes the volatility back out. The interaction is the result, not either part.

### Robustness

| | baseline | dual MA, BTC+ETH |
|---|---|---|
| pit CAGR at **2x costs** | 11.52% | **18.92%** |
| pit Sharpe at 2x costs | 0.6961 | **0.9728** |
| placebo, p vs random coin | 0.0020 | **0.0000** |

At double the fees and slippage it still returns more than the baseline does at normal costs. The
random-coin null is the strongest in this log: **p = 0.000**.

### The caveat that matters, stated plainly

**The idea came from looking at 2025-2026, and the held-out window includes 2025-2026.** P14 found
`sma50_200` by scanning ninety single-asset cells, and P9 had already declared 2025-2026 spent. So
the held-out column above is not independent evidence for this particular change - it cannot be,
because that data helped choose it.

What is independent: the full-sample pit and standard panels are dominated by 2018-2024, which had
no part in the choice, and it passes on both. The 2x-cost and placebo results do not depend on the
sample selection at all. That is enough to ship the flag and not enough to call it proven.

It ships **ON for the demo book only**, where the paper books remain the control, and the verdict
is the forward test on bars that do not exist yet.

10 tests, half of them on the thing most likely to break: that the backtest and the live tick
agree about the rule. If one checks two lines and the other checks one, the live book holds what
the backtest sold and every comparison between them becomes meaningless.

241 tests pass; flag-off parity exact at 7.3e-12.

---

## P16 - what a 90%+ win rate actually costs (18 Oct 2026)

*"Not 100% profit, but can we get the win percentage above 90 or 95?"*

Yes. And the answer is more interesting than the usual warning, so here is the measurement rather
than the lecture: a 9 x 5 take-profit / stop-loss grid on BTC, ETH, SOL, BNB and XRP, 2022-2026,
live costs, exits checked against the bar's high and low so a touch counts.

### The win rate is reachable, and it is not even unprofitable

**13 of 45 cells clear a 90% win rate.** Several of them make money:

| take profit | stop loss | **win rate** | CAGR | maxDD | avg win | avg loss |
|---|---|---|---|---|---|---|
| +8% | -60% | **96.64%** | **+19.88%** | **-65.45%** | +7.8% | -60.1% |
| +12% | -60% | **95.02%** | **+19.45%** | -65.78% | +11.8% | -60.1% |
| +5% | -60% | **97.57%** | +15.99% | -66.42% | +4.8% | -60.1% |
| +3% | -60% | **97.98%** | +10.85% | -66.54% | +2.8% | -60.1% |
| +1% | -60% | **99.01%** | **-2.96%** | -67.87% | +0.8% | -60.1% |

A **96.6% win rate at +19.88% a year** is a real configuration, not a trick. So the honest answer
is not "impossible" - it is "available, at a price", and the price is in a column most people do
not look at.

### The price is drawdown, not return

| | CAGR | maxDD | **Calmar** | win rate |
|---|---|---|---|---|
| best 95%+ win-rate cell (+8% / -60%) | +19.88% | **-65.45%** | **0.30** | 96.64% |
| the config that just shipped (`core_dual_ma`, pit) | +21.55% | **-22.02%** | **0.98** | ~13% |

**Almost identical return. Three times the drawdown.** On a $100 book that is watching it fall to
$35 instead of $78, for the same money at the end.

The mechanism is visible in the table: every 90%+ cell has a **-60% stop**. The win rate is high
because almost nothing falls 60% - and when something does, one trade erases sixty winners at +1%.
The grid's own arithmetic says so: avg win +0.8% against avg loss -60.1% needs a 98.7% win rate
merely to break even, which is why the 99.01% cell still loses money.

### The number that settles it

Across all 225 coin-cells:

    correlation between win rate and CAGR:  +0.0422

**Essentially zero.** The win rate carries almost no information about whether a strategy makes
money. The six most profitable cells have win rates of 88.8%, 71.4%, 65.1%, 88.1%, 96.6% and 95.0%
- spread across the whole range, because what pays is

    expectancy = P(win) x average win - P(loss) x average loss

and raising P(win) necessarily shrinks the average win and grows the average loss. The two always
move against each other.

### So what IS worth doing

Not chasing the percentage. The live paper book's 1-win-in-5 is not the problem to solve; P14
showed `sma50_200` winning 0 of 10 trades on BTC in 2025-26 and still beating buy-and-hold by nine
points.

If the goal is psychological - seeing more green than red - the +8%/-60% cell genuinely delivers it
at a comparable return, and the cost is a -65% drawdown that must be sat through. That is a real
choice, and it is the operator's to make, not a modelling question. It is **not** recommended here,
because a drawdown of that size on a small book is the one thing most likely to end the experiment
before the strategy gets a chance to work.

---

## P17 - five drastically different shapes, and why the cash is not the problem (30 Oct 2026)

*"This strategy is designed to do nothing most of the time... we need some drastic change."*

The complaint is accurate: median 18-day window +0.00%, 26% of months green, ~18% exposure. Sixteen
programs have tuned that design and none changed its SHAPE. So this tests shapes - different
answers to "what should the book hold right now" - rather than more parameters.

### Held-out 2022-2026, pit panel, real costs

| shape | CAGR | Sharpe | maxDD | exposure | months green | median 18d |
|---|---|---|---|---|---|---|
| **cash_when_unsure (today)** | **+12.45%** | **0.6907** | **-22.02%** | 18.2% | 26.3% | +0.00% |
| always_rotate top5 | **-85.06%** | -1.8522 | **-99.99%** | 100% | 28.1% | -9.16% |
| always_eq basket | -37.74% | -0.6162 | -93.92% | 100% | 38.6% | -2.25% |
| reversion (buy weakness) | **-80.66%** | -1.7126 | **-99.97%** | 100% | 24.6% | -6.11% |
| trend_no_cash | **-87.81%** | -1.9844 | **-100.00%** | 100% | 24.6% | -10.14% |
| **vol_scaled BTC+ETH** | +9.72% | 0.2407 | -55.59% | **76.0%** | **56.1%** | -0.02% |

### The finding: being always invested does not underperform, it **destroys the account**

Three of the five always-on shapes lose **99.97% to 100%** of the capital on the survivorship-free
panel. Not a bad stretch - a wipeout. `trend_no_cash`, which is the current ranking with *only one
change* - hold BTC instead of cash when nothing qualifies - goes to **-100%**.

The reason is the panel. The pit universe contains the coins that actually existed at the time,
including the ones that went to zero. Holding the top five by momentum through 2022 means holding
whatever was strongest on the way into a 90% drawdown, repeatedly, with no exit.

**The cash is not the bug. It is the only thing keeping the account alive.** That reframes the
entire complaint: the bot does nothing most of the time because most of the time, doing something
loses money - and this is the measurement that shows how much.

Buying weakness fails the same way (-80.66%). Every shape that stays invested in altcoins dies.

### The one always-invested shape that survives

`vol_scaled BTC+ETH` - always long BTC and ETH, sized inversely to volatility, **no trend gate at
all** - is the only always-on shape that makes money. And it delivers exactly what was asked for:

| | today's bot | vol_scaled BTC+ETH |
|---|---|---|
| months green | 26.3% | **56.1%** |
| exposure | 18.2% | **76.0%** |
| CAGR | **+12.45%** | +9.72% |
| maxDD | **-22.02%** | **-55.59%** |
| Sharpe | **0.6907** | 0.2407 |

More than twice as many green months and four times the time invested - for about a fifth less
return and **two and a half times the drawdown**. On $100 that is watching it fall to $44 instead
of $78.

That is a real choice and an operator's to make. It is the honest answer to "I want to see it
doing something": there is a version that does, it holds only the two largest coins, it never sits
in cash, and the price is the drawdown.

### What this closes

The idea that idleness is the defect. It is not; it is the mechanism. Any future proposal to
"trade more" now has to explain why it will not land where `always_rotate` and `trend_no_cash` did,
and those two are not marginal failures - they are total losses of capital.
