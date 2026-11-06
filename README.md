# crypto-trend-bot

A spot crypto trading bot that holds strength when the market is risk-on and **sits in cash when it
is not** — which, on this data, is most of the time.

Two engines run side by side:

- **Trend core** — hold BTC and ETH for as long as each closes above *both* its 50-day and 200-day
  average. A persistent position, not a trade: no breakout, no stop, no rank.
- **Breakout rotation** — buy the strongest altcoins by relative strength when they clear a 20-bar
  high, size each so a stop at the 20-bar low risks 1% of equity, exit on a breakdown or a rank drop.

Above both sits an account-level brake: an 8% peak-to-trough kill and a daily loss halt that close
everything regardless of what either engine thinks.

## Results

Survivorship-free backtest — each day's universe rebuilt from the volume known that morning,
**including coins that were later delisted**. Fees 0.10% per side, 7 bps slippage, orders filled at
the next bar's open.

| | full sample 2018–2026 | held-out 2022–2026 |
|---|---|---|
| CAGR | +21.55% | **+12.45%** |
| Sharpe | 1.0971 | 0.6907 |
| max drawdown | −22.02% | −22.02% |
| average exposure | 17.3% | 18.2% |

For scale: holding BTC over 2022–2026 returned +13.39% a year with a **−67%** drawdown. This earns
about the same with a third of the pain. That is the product — it is insurance, not income.

```bash
pip install -r requirements.txt
python -m app.main backfill          # download candles
python -m app.main backtest          # run the backtest
python research/report_panels.py     # both panels side by side
pytest -q                            # 316 tests
```

## Read this before you take the numbers seriously

**The median 18-day window returns +0.00%.** Only 26% of months are green. The longest losing
streak in the backtest is 11 months, and 94% of all time is spent below a previous high. This
design is idle by construction.

**That idleness is the mechanism, not a defect.** Five always-invested variants were tested on the
same survivorship-free data: holding the top 5 by momentum lost **−99.99%**, buying weakness lost
**−99.97%**, and the current ranking with *one change* — hold BTC instead of cash when nothing
qualifies — lost **−100%**. The cash is what keeps the account alive.

**The standard backtest lies and is included anyway.** Applying today's coin list to all of history
reports +27% CAGR for the same rules. The survivorship-free figure is +21.55%. Always quote the
second one.

**It has been run forward for 46 days and lost 5.3%**, against BTC −6.3%, ETH −15.4% and SOL
−19.7% over the same window, holding 3.4% of its capital and sitting fully in cash on 71% of bars.
Losing less than the market while barely invested is what this design is for, so judge it against
BTC rather than against zero.

**But the sample says almost nothing, and that is the most important thing on this page.** Thirteen
distinct trade decisions, three winners. The 95% confidence interval on that win rate is **8.2% to
50.3%** — it contains both the 30.3% breakeven and the 32.2% the backtest claims. Showing that
32.2% really beats breakeven at 95% confidence needs **2,248 trades**, which at the observed rate
of 0.17 trades a day is **35 years**. This strategy cannot generate enough evidence to confirm or
deny its own edge within a working lifetime. No amount of forward testing fixes that; only a
higher trade rate would, and the one measurement here of deciding more often says acting sooner
costs return.

## What is actually in here

The research is the larger half of the repository.

- `RESEARCH_LOG.md` — every program run, including **~660 rejected variants**. The rejections are
  the useful part: ML as a ranker, ML as a veto, nine strategy families, five trend-prediction
  approaches, volatility targeting, correlation caps, time stops, faster timeframes.
- `app/` — the engine. Backtest and live are *meant* to share every predicate. They did not: on
  30 Oct 2026 a flag reached two of the three code paths that decide whether the trend core is on,
  the bar close bought on one rule while the live tick sold on another, and it cost 203 fills and
  118 USDT of fees before anyone noticed. A 6 Nov audit found the same shape in the slot count and
  found that the 8% drawdown kill re-armed in the backtest and never did live — so the published
  figures described a book surviving dozens of kills while a live book died on its first. Both are
  fixed, both have path-agreement tests, and the lesson is in the code: one predicate, one place.
- `research/` — the measurement tools: survivorship-free panel builder, the ship rule, placebo
  nulls, cost stress, per-engine P&L attribution.
- `tests/` — 316 tests. Several exist because a bug reached live money first, and each of those
  names the incident, the date and the cost in its docstring.

**The ship rule**, applied throughout: a change ships only if CAGR, Sharpe *and* max drawdown are
no worse on the standard panel **and** the survivorship-free panel, with the held-out period not
failing. Flag off must reproduce the baseline equity curve byte-for-byte. Of ~660 variants, four
filters and two structural changes passed.

Some findings worth more than the bot:

- Returns are unpredictable (AUC 0.47–0.52) but **losses are** (AUC 0.78) — and the trades a model
  correctly flags as dangerous are the profitable ones, so it cannot be traded.
- Win rate carries almost no information about profit: across 225 configurations the correlation
  between win rate and CAGR is **+0.04**. A 96.6% win rate at +19.88% a year exists; it comes with
  a −65% drawdown.
- The strategy's edge is concentrated in 2019–2021. Survivorship-free, 2023–2025 compounds to
  **+0.65% a year**.

## Setup

Copy `.env.example` to `.env` and fill it in. Runs in `paper` mode by default and will not place a
real order unless explicitly configured otherwise. `deploy/` contains the VPS scripts; replace
`<YOUR_SERVER_IP>` with your own.

## What this is not

Not advice, not a product, and not managing anyone's money. Published so the method and the
rejections can be checked.

## Licence

MIT.
