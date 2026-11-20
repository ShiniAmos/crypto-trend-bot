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
pytest -q                            # 327 tests
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
- `tests/` — 327 tests. Several exist because a bug reached live money first, and each of those
  names the incident, the date and the cost in its docstring.

**The ship rule**, applied throughout: a change ships only if CAGR, Sharpe *and* max drawdown are
no worse on the standard panel **and** the survivorship-free panel, with the held-out period not
failing. Flag off must reproduce the baseline equity curve byte-for-byte. Of ~660 variants, four
filters and two structural changes passed.

Some findings worth more than the bot:

- **A model can tell which coins are about to do badly. It cannot tell which will do well.**
  Gradient boosting on relationship features (BTC lead-lag, market breadth, rolling beta, bar
  microstructure), trained walk-forward on the survivorship-free universe, ranks coins by their
  next-24-hour return correctly in **29 of 29 quarters** out of sample. All of that skill is at the
  bottom: the lowest-scored tenth of coins loses 0.68% a day against the average coin in
  development and 0.87% on the holdout, while the highest-scored tenth has no edge left on the
  holdout at all. Four different long-only rules built on its top picks were tried and all four
  failed out of sample. One of them made +80% on 17 coins that are alive today and −70% on the 383
  that actually traded, having bought LUNA six times on the way to zero — the cleanest
  demonstration of survivorship bias in this repository.
- **Used the other way round, it is worth something.** As a veto on the breakout entries the bot
  already takes — skip a coin the model ranks in its bottom third — it lifts the survivorship-free
  held-out backtest from +11.96% to **+14.57%** a year, Sharpe 0.69 to **0.88**, with a shallower
  drawdown. Twelve random vetoes of the same size averaged +9.90%, worse than no veto at all, and
  none matched it. It still **fails this project's ship rule**, because the 19-coin survivor panel
  disagrees by under a point, so it has not shipped. A filter whose value is avoiding coins about to
  do badly is expected to look weakest on a panel assembled from coins that did not.
- **Out of sample, the entry timing is a coin flip.** Against the repository's own placebo null,
  the same coin held for the same number of bars entered at a *random* risk-on moment does as well
  as the strategy's own entry: **p = 0.496** for the trend core and **0.512** for the breakout
  rotation on 2022–2026 data. Coin selection does not clear 0.05 either (0.064 and 0.086). Both
  are p = 0.000 on the full sample, so the significance lives in 2019–2021 — the era the year
  attribution already identified, and one this engine will not trade again. Whatever this bot does
  after 2022, it is not picking coins or moments better than chance.
- **What it does provably do is lose less in a fall.** Run forward through a 20% BTC decline it lost
  5.9%, at 3.4% average exposure, with a −7.0% worst drawdown. That claim verifies in weeks rather
  than decades and does not require the entry timing to have an edge, which is why it is the one
  worth judging this on.
- Deciding inside the bar instead of on the 4-hour close cost **+113.7 bps per breakout fill**,
  measured on real exchange fills — sixteen times the modelled slippage, about 2.3% per round trip
  against an average winning trade of +4.8%. The cost model itself is accurate: 212 real
  commissions against a modelled 0.10% came in at a median 0.1001%.
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
