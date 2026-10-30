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
pytest -q                            # 241 tests
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

**It has been run forward on paper and demo for 18 days and lost ~4.8%**, against BTC −6.7% and ETH
−11.5% over the same window. That is the 4th percentile of backtested 18-day windows: a bad run,
inside the expected range, and far too short to conclude anything.

## What is actually in here

The research is the larger half of the repository.

- `RESEARCH_LOG.md` — every program run, including **~660 rejected variants**. The rejections are
  the useful part: ML as a ranker, ML as a veto, nine strategy families, five trend-prediction
  approaches, volatility targeting, correlation caps, time stops, faster timeframes.
- `app/` — the engine. Backtest and live share every predicate, so the two cannot drift apart.
- `research/` — the measurement tools: survivorship-free panel builder, the ship rule, placebo
  nulls, cost stress, per-engine P&L attribution.
- `tests/` — 241 tests. Several exist because a bug reached live money first.

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
