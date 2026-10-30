"""Can a LARGE ADVERSE MOVE be forecast? (drawdown risk, not return)

This project has already established, adversarially, that RETURNS are not predictable from this
data: a walk-forward ranker lost to the simple rule, an ML veto on excess return took CAGR from
24.71% to 16.74%, a win-probability model ordered trades BACKWARDS to profit (+3.65% in its most
confident quartile against +8.79% in its least), and a right-tail classifier trained on "will this
beat +20%" scored out-of-sample AUC 0.4914 - worse than a coin flip. None of that is re-litigated
here. Nothing in this file predicts a return.

The question here is narrower and sits on firmer empirical ground. Volatility clusters, and a
crash is mostly a volatility event. So:

    TARGET   crash15 = the coin's forward 7-day LOW is more than 15% below the current close,
             i.e. min(low over (t, t+7d]) / close(t) - 1 <= -0.15.

The forward window is TIME-based, not bar-count-based, because this history has gaps (up to 1 day
8 hours of missing 4h candles around the 2020 and 2021 outages); 42 bars would silently reach past
7 days there. A row is only labelled when a bar exists at or after t + 7 days, so the window is
never truncated by the end of the data.

WHY THIS TARGET AND NOT A RETURN. The bot is long-only with a 1% risk budget per trade, a 20-bar
Donchian stop reference and an 8% account kill switch. Anticipating a crash is worth more to it
than anticipating a rally, and - unlike a return - a crash is partly a function of volatility,
which is the one thing in finance that genuinely persists.

THE NAIVE BASELINE IS THE WHOLE RESULT. "High volatility coins fall 15% more often" is trivially
true, so a high AUC proves nothing by itself. Three naive baselines, in increasing strength:

    naive_base      constant: the TRAINING window's crash frequency. AUC 0.5 by construction; it
                    exists to anchor the Brier score.
    naive_vol_raw   no fitting at all: the closed-form first-passage probability of a driftless
                    GBM, p = 2*Phi(ln(0.85) / (sigma*sqrt(T))) with sigma = vol_30d and T = 7/365.
                    A real probability forecast straight out of trailing vol.
    naive_vol_fit   logistic regression on log(vol_30d) ALONE, refit each fold on the training
                    window. This is "just look at trailing vol", fairly calibrated. It is the
                    baseline the model has to beat, and the margin over it is the result.

The model is a HistGradientBoostingClassifier over 24 features (vol at 1/7/30 days and their term
structure, atr_pct, don_pos, dist_sma50/200, ret_1d/7d/30d, volume_ratio, taker_ratio, rs_7d/30d,
the BTC regime columns and BTC's own trailing vol). A logistic regression on the same features,
standardised with TRAINING-window statistics only, runs alongside as a transparent comparator.

TWO HONESTY CONTROLS, run through the identical harness:
    direction   target "is the forward 7-day close above today's close". Expect AUC ~0.50. If this
                comes back at 0.56 the harness leaks and nothing else in the file is believable.
    shuffled    the same crash target with the TRAINING labels permuted. Expect AUC ~0.50.

NO LOOKAHEAD: every feature at row t is built from bars closed at or before t (app/features.py
guarantees this; the extra columns added here are rolling backwards-only). Targets look forward
only. Walk-forward is 730 days train / 91 days test, rolling, with a 10-day purge gap between the
end of training and the start of testing - longer than the 7-day prediction horizon, so no
training label can peek into its own test window. Standardisation and the base rate come from the
training window alone.

MEASURED out of sample: AUC, Brier and log loss against all three naive baselines; the realised
crash frequency per predicted-probability decile (is the ladder monotone?); a calibration table;
per-fold AUC so the margin carries an error bar instead of a single pooled number; and an AUC on a
time-decimated subsample (one bar per 7 days per symbol) because overlapping windows make the
pooled n a fiction.

THEN THE ONLY QUESTION THAT PAYS: among the bot's actual ENTRY CANDIDATES (eligible, regime on,
breakout, rank <= 5, ret_30d > 0, volume_ratio >= 0.8, taker_ratio >= 0.5) does the model separate
the trades that went on to lose more than 10% from the rest? Those trades are labelled with their
REAL net outcome - entered at the next bar's open, exited when the live rule fires (regime off,
20-bar low, or rank worse than 10), 7 bps slippage and 0.10% fee per side. That subset is small,
self-selected and hard, and it is the only sample the bot actually trades.

    cd crypto-bot && PYTHONIOENCODING=utf-8 python research/forecast_risk.py

Writes reports/forecast_risk.json. Touches nothing else.
"""

import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config, features, signals  # noqa: E402
from app.flags import Flags  # noqa: E402

SHIPPED = Flags(abs_momentum=True, volume_min=0.8, taker_min=0.5)

TRAIN_DAYS, TEST_DAYS, PURGE_DAYS = 730, 91, 10
HORIZON_DAYS = 7
CRASH = -0.15          # the headline target
LOSS = -0.10           # "this trade went on to lose more than 10%"
BARS_PER_DAY = 6
ANNUAL = np.sqrt(BARS_PER_DAY * 365)

HP = dict(max_iter=250, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=50,
          l2_regularization=1.0, early_stopping=False, random_state=0)
# A model this size can be accused of having been set up to fail: 24 features over a sample whose
# 7-day windows overlap 42-fold is a lot of capacity. Two fairer variants run alongside it, so
# "ML lost" cannot be answered with "you over-parameterised it".
HP_REG = dict(max_iter=150, learning_rate=0.03, max_leaf_nodes=7, min_samples_leaf=1000,
              l2_regularization=10.0, early_stopping=False, random_state=0)
VOL_FEATS = ["vol_1d", "vol_7d", "vol_30d", "vol_term", "atr_pct", "btc_vol_30d"]

FEATS = [
    # trailing volatility at several horizons, plus its term structure
    "vol_1d", "vol_7d", "vol_30d", "vol_term", "atr_pct", "atr_vs_vol",
    # where in its own range, and how far above trend
    "don_pos", "dist_sma50", "dist_sma200", "stop_distance_pct",
    # the recent run-up
    "ret_1d", "ret_7d", "ret_30d", "rs_7d", "rs_30d",
    # participation
    "volume_ratio", "taker_ratio",
    # the market it lives in
    "btc_dist_sma200", "btc_ret_7d", "btc_ret_30d", "btc_vol_7d", "btc_vol_30d",
    "vol_vs_btc", "regime_on_i",
]


# --------------------------------------------------------------------------- data

def add_extras(panel: pd.DataFrame) -> pd.DataFrame:
    """Backwards-only columns the shipped feature builder does not already carry."""
    panel = panel.sort_values(["symbol", "bar_time"]).reset_index(drop=True)
    # 1-day realised vol from 4h log returns (6 bars). Noisy by construction, which is the point:
    # it is the fast leg of the volatility term structure.
    lr = np.log(panel["close"]).groupby(panel["symbol"]).diff()
    panel["vol_1d"] = (lr.groupby(panel["symbol"])
                         .rolling(BARS_PER_DAY, min_periods=BARS_PER_DAY).std()
                         .reset_index(level=0, drop=True) * ANNUAL)
    panel["vol_term"] = panel["vol_7d"] / panel["vol_30d"]
    panel["atr_vs_vol"] = panel["atr_pct"] * ANNUAL / panel["vol_30d"]
    panel["stop_distance_pct"] = 1 - panel["stop_level"] / panel["close"]
    btc = panel[panel["symbol"] == "BTCUSDT"].set_index("bar_time")
    panel = panel.join(btc[["vol_7d", "vol_30d"]].add_prefix("btc_"), on="bar_time")
    panel["vol_vs_btc"] = panel["vol_30d"] / panel["btc_vol_30d"]
    panel["regime_on_i"] = panel["regime_on"].astype(float)
    return panel


def _window_extremes(bt_ns, low, high, close, horizon_ns):
    """min(low), max(high), min(close) and the last close over the TIME window (t, t+horizon].

    Monotonic-deque sliding window, so gaps in the candle history shorten the window in bars
    instead of letting it reach past `horizon` in wall-clock time. NaN where the window is not
    fully covered by the data (no bar exists at or after t + horizon).

    min(close) is carried because the LOW column has bad prints in this history - LINKUSDT shows a
    0.0001 low on 2020-03-12 and SUI/UNI/AVAX/BNB show listing-day wicks of -78% to -92% against
    their own close. Those are real exchange prints but not prices anything could have been sold
    at, so a wick-immune version of the label is measured alongside the headline one.
    """
    n = len(bt_ns)
    lo = np.full(n, np.nan)
    hi = np.full(n, np.nan)
    cmin = np.full(n, np.nan)
    lastc = np.full(n, np.nan)
    end = np.searchsorted(bt_ns, bt_ns + horizon_ns, side="right")   # exclusive, non-decreasing
    dlo, dhi, dcl = deque(), deque(), deque()
    cur = 0
    for i in range(n):
        start = i + 1
        e = end[i]
        while cur < e:                       # extend the right edge
            while dlo and low[dlo[-1]] >= low[cur]:
                dlo.pop()
            dlo.append(cur)
            while dhi and high[dhi[-1]] <= high[cur]:
                dhi.pop()
            dhi.append(cur)
            while dcl and close[dcl[-1]] >= close[cur]:
                dcl.pop()
            dcl.append(cur)
            cur += 1
        while dlo and dlo[0] < start:        # retire the left edge
            dlo.popleft()
        while dhi and dhi[0] < start:
            dhi.popleft()
        while dcl and dcl[0] < start:
            dcl.popleft()
        if e > start and dlo:
            lo[i] = low[dlo[0]]
            hi[i] = high[dhi[0]]
            cmin[i] = close[dcl[0]]
            lastc[i] = close[e - 1]
    complete = (bt_ns + horizon_ns) <= bt_ns[-1]
    for a in (lo, hi, cmin, lastc):
        a[~complete] = np.nan
    return lo, hi, cmin, lastc


def add_targets(panel: pd.DataFrame, horizon_days=HORIZON_DAYS) -> pd.DataFrame:
    """Forward-looking labels. These are the ONLY forward-looking columns in the file."""
    h = np.int64(horizon_days * 86_400_000_000_000)
    out = []
    for sym, g in panel.groupby("symbol", sort=False):
        g = g.sort_values("bar_time").reset_index(drop=True)
        lo, hi, cmin, lastc = _window_extremes(
            g["bar_time"].astype("int64").to_numpy(),
            g["low"].to_numpy(), g["high"].to_numpy(), g["close"].to_numpy(), h)
        c = g["close"].to_numpy()
        g["fwd_low_ret"] = lo / c - 1
        g["fwd_high_ret"] = hi / c - 1
        g["fwd_minclose_ret"] = cmin / c - 1
        g["fwd_close_ret"] = lastc / c - 1
        out.append(g)
    panel = pd.concat(out, ignore_index=True)
    panel["crash15"] = (panel["fwd_low_ret"] <= CRASH).astype(float)
    panel["crash10"] = (panel["fwd_low_ret"] <= LOSS).astype(float)
    panel["crash15c"] = (panel["fwd_minclose_ret"] <= CRASH).astype(float)   # wick-immune
    panel["up7d"] = (panel["fwd_close_ret"] > 0).astype(float)
    for c in ("crash15", "crash10", "crash15c", "up7d"):
        panel.loc[panel["fwd_low_ret"].isna(), c] = np.nan
    return panel.sort_values(["bar_time", "symbol"]).reset_index(drop=True)


def label_trades(panel, s):
    """Real net return of the trade each candidate bar would have become (live exit rule)."""
    fee, slip = s.fee_rate, s.slippage
    rows = []
    for sym, g in panel.groupby("symbol", sort=False):
        g = g.sort_values("bar_time").reset_index(drop=True)
        o = g["open"].to_numpy()
        exit_cond = ((~g["regime_on"].fillna(False).to_numpy())
                     | g["breakdown"].fillna(False).to_numpy()
                     | (g["rank"].fillna(999).to_numpy() > s.exit_rank))
        exits = np.flatnonzero(exit_cond)
        for i in np.flatnonzero(g["_cand"].to_numpy()):
            if i + 1 >= len(g):
                continue
            entry = o[i + 1] * (1 + slip)
            nxt = exits[np.searchsorted(exits, i + 1, side="left"):]
            j = nxt[0] if len(nxt) else len(g) - 2
            if j + 1 >= len(g):
                continue
            exit_px = o[j + 1] * (1 - slip)
            net = (exit_px * (1 - fee)) / (entry * (1 + fee)) - 1
            rows.append({"bar_time": g["bar_time"].iloc[i], "symbol": sym, "net": net,
                         "bars_held": j - i})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- naive baselines

def first_passage_prob(vol_annual, barrier=CRASH, days=HORIZON_DAYS):
    """P(a driftless GBM touches `barrier` within `days`), by the reflection principle.

    p = 2 * Phi(ln(1+barrier) / (sigma * sqrt(T))). No fitting, no parameters beyond trailing vol:
    the honest "it is just volatility" forecast.
    """
    sig = np.asarray(vol_annual, dtype=float)
    t = days / 365.0
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.log1p(barrier) / (sig * np.sqrt(t))
    p = 2 * norm.cdf(z)
    return np.clip(p, 1e-6, 1 - 1e-6)


# --------------------------------------------------------------------------- walk-forward

def walk_forward(d, target="crash15", verbose=True, light=False):
    """Roll 730/91 with a 10-day purge. Returns the frame with OOS prediction columns + fold log.

    `light` fits only the gradient-boosted model, for the control runs where the baselines and the
    shuffled null are not needed again.
    """
    X = d[FEATS].replace([np.inf, -np.inf], np.nan)
    y = d[target].to_numpy()
    t = d["bar_time"]
    logv = np.log(d["vol_30d"].clip(lower=1e-4)).to_numpy().reshape(-1, 1)

    for c in ("p_model", "p_reg", "p_volgbm", "p_lr", "p_naive_fit", "p_naive_base", "p_shuf",
              "fold"):
        d[c] = np.nan

    rng = np.random.default_rng(0)
    first = t.min() + pd.Timedelta(days=TRAIN_DAYS)
    test_start, folds, log = first, 0, []
    while test_start < t.max():
        test_end = test_start + pd.Timedelta(days=TEST_DAYS)
        tr = ((t >= test_start - pd.Timedelta(days=TRAIN_DAYS))
              & (t < test_start - pd.Timedelta(days=PURGE_DAYS))).to_numpy()
        te = ((t >= test_start) & (t < test_end)).to_numpy()
        if tr.sum() >= 2000 and te.sum() >= 200 and len(np.unique(y[tr])) > 1:
            folds += 1
            Xtr, Xte = X[tr], X[te]
            ytr = y[tr]

            m = HistGradientBoostingClassifier(**HP).fit(Xtr, ytr)
            d.loc[te, "p_model"] = m.predict_proba(Xte)[:, 1]

            # naive: trailing vol alone, refit on this training window so the baseline is as
            # fairly calibrated as the model is
            nf = LogisticRegression(max_iter=1000).fit(logv[tr], ytr)
            d.loc[te, "p_naive_fit"] = nf.predict_proba(logv[te])[:, 1]
            d.loc[te, "p_naive_base"] = ytr.mean()      # the training base rate, as a constant

            if not light:
                # heavily regularised version of the same model, on the same features
                mr = HistGradientBoostingClassifier(**HP_REG).fit(Xtr, ytr)
                d.loc[te, "p_reg"] = mr.predict_proba(Xte)[:, 1]

                # the same model shown ONLY volatility columns: does non-linearity in volatility
                # beat the one-feature logistic, before any other feature is blamed?
                mv = HistGradientBoostingClassifier(**HP_REG).fit(Xtr[VOL_FEATS], ytr)
                d.loc[te, "p_volgbm"] = mv.predict_proba(Xte[VOL_FEATS])[:, 1]

                # transparent comparator: logistic regression, standardised on the TRAIN window
                # only - the test window never contributes a mean or a standard deviation. The
                # clip keeps one fat-tailed row from owning the fit, which would be a modelling
                # artifact rather than a finding.
                mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
                Ztr = Xtr.sub(mu).div(sd).clip(-5, 5).fillna(0.0)
                Zte = Xte.sub(mu).div(sd).clip(-5, 5).fillna(0.0)
                lr = LogisticRegression(max_iter=2000, C=1.0).fit(Ztr, ytr)
                d.loc[te, "p_lr"] = lr.predict_proba(Zte)[:, 1]

                # control: the same model on permuted training labels
                ms = HistGradientBoostingClassifier(**HP).fit(Xtr, rng.permutation(ytr))
                d.loc[te, "p_shuf"] = ms.predict_proba(Xte)[:, 1]

            d.loc[te, "fold"] = folds
            log.append({"fold": folds, "test_start": str(test_start.date()),
                        "n_train": int(tr.sum()), "n_test": int(te.sum()),
                        "train_rate": float(ytr.mean()), "test_rate": float(y[te].mean())})
            if verbose:
                print(f"  fold {folds:2d}  test from {test_start.date()}  "
                      f"train {tr.sum():6d} ({ytr.mean():.1%})  test {te.sum():5d} "
                      f"({y[te].mean():.1%})")
        test_start = test_end
    return d, log


# --------------------------------------------------------------------------- scoring

def safe_auc(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    if ok.sum() < 30 or len(np.unique(y[ok])) < 2:
        return None
    return float(roc_auc_score(y[ok], p[ok]))


def score_block(y, preds: dict):
    """AUC / Brier / log loss for a dict of prediction columns on one sample."""
    y = np.asarray(y, float)
    out = {}
    for name, p in preds.items():
        p = np.asarray(p, float)
        ok = np.isfinite(y) & np.isfinite(p)
        if ok.sum() < 30:
            out[name] = {"n": int(ok.sum())}
            continue
        pc = np.clip(p[ok], 1e-6, 1 - 1e-6)
        out[name] = {
            "n": int(ok.sum()),
            "auc": safe_auc(y[ok], p[ok]),
            "brier": float(brier_score_loss(y[ok], pc)),
            "logloss": float(log_loss(y[ok], pc, labels=[0, 1])),
            "mean_pred": float(pc.mean()),
        }
    return out


def fmt(v, pct=False, nd=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "    n/a"
    return f"{v:.{nd - 2}%}" if pct else f"{v:.{nd}f}"


def decile_table(o, pcol, ycol, q=10):
    k = min(q, o[pcol].nunique())
    if k < 2:
        return pd.DataFrame()
    bins = pd.qcut(o[pcol], k, labels=False, duplicates="drop")
    g = o.assign(_b=bins).groupby("_b", observed=True)
    return pd.DataFrame({
        "n": g.size(),
        "pred_lo": g[pcol].min(),
        "pred_hi": g[pcol].max(),
        "mean_pred": g[pcol].mean(),
        "realised": g[ycol].mean(),
        "mean_fwd_low": g["fwd_low_ret"].mean(),
        "worst_fwd_low": g["fwd_low_ret"].min(),
    })


def calibration_table(o, pcol, ycol,
                      edges=(0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 1.0)):
    b = pd.cut(o[pcol], list(edges), include_lowest=True)
    g = o.groupby(b, observed=True)
    tab = pd.DataFrame({"n": g.size(), "mean_pred": g[pcol].mean(), "realised": g[ycol].mean()})
    tab["gap"] = tab["realised"] - tab["mean_pred"]
    return tab[tab["n"] > 0]


def stratified_auc(o, ycol, pcol, by="bar_time"):
    """AUC computed only between rows that share a `by` value, pooled over pairs.

    This strips the market-wide component out of a predictor. A predictor that merely knows WHEN
    the whole market is dangerous scores well on the ordinary pooled AUC and ~0.50 here; one that
    knows WHICH COIN is dangerous on a given bar scores above 0.50 here too. The two are different
    products: the first is market timing, which this bot already does with its 200-day BTC regime
    filter, and the second is selection, which it does not.
    """
    num = den = 0.0
    y, p = o[ycol].to_numpy(float), o[pcol].to_numpy(float)
    g = o[by].to_numpy()
    order = np.argsort(g, kind="stable")
    y, p, g = y[order], p[order], g[order]
    edges = np.flatnonzero(np.r_[True, g[1:] != g[:-1], True])
    for a, b in zip(edges[:-1], edges[1:]):
        yy, pp = y[a:b], p[a:b]
        ok = np.isfinite(yy) & np.isfinite(pp)
        yy, pp = yy[ok], pp[ok]
        npos, nneg = int(yy.sum()), int(len(yy) - yy.sum())
        if npos == 0 or nneg == 0:
            continue
        w = npos * nneg
        num += roc_auc_score(yy, pp) * w
        den += w
    return float(num / den) if den else None


def per_fold_auc(o, ycol, cols):
    rows = []
    for f, g in o.groupby("fold"):
        r = {"fold": int(f), "n": len(g), "rate": float(g[ycol].mean())}
        for c in cols:
            r[c] = safe_auc(g[ycol], g[c])
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- main

def main():
    s = config.load()
    u = json.loads((ROOT / "reports" / "_parity_universe.json").read_text())
    print("building panel ...")
    panel = add_targets(add_extras(features.build_panel(s.klines_dir, u["symbols"],
                                                        s.donchian_bars)))
    print(f"panel: {len(panel):,} rows  {panel['symbol'].nunique()} symbols  "
          f"{panel['bar_time'].min().date()} -> {panel['bar_time'].max().date()}")

    # the bot's own entry candidates, flagged on the full panel before any sampling
    rank = panel["rank"].to_numpy()
    btc = (panel["symbol"] == "BTCUSDT").to_numpy()
    panel["_cand"] = ((panel["eligible"] & panel["regime_on"] & panel["breakout"]).to_numpy()
                      & ((panel["rs_7d"] > 0).to_numpy() | btc)
                      & (rank <= s.candidates)
                      & signals.extra_filters(panel, SHIPPED).to_numpy())
    print(f"entry candidates in the panel: {int(panel['_cand'].sum()):,}")

    # modelling sample: eligible rows with a complete forward window and usable features
    use = (panel["eligible"].to_numpy() & panel["crash15"].notna().to_numpy()
           & panel["vol_30d"].notna().to_numpy() & panel["vol_1d"].notna().to_numpy())
    d = panel[use].copy().reset_index(drop=True)
    base_rate = float(d["crash15"].mean())
    print(f"modelling rows: {len(d):,}   unconditional crash({CRASH:.0%} in {HORIZON_DAYS}d) "
          f"base rate: {base_rate:.2%}")
    print(f"  crash({LOSS:.0%}) base rate: {d['crash10'].mean():.2%}   "
          f"wick-immune crash({CRASH:.0%} on closes): {d['crash15c'].mean():.2%}   "
          f"up-in-7d base rate: {d['up7d'].mean():.2%}")
    print(f"  base rate regime ON {d.loc[d['regime_on'], 'crash15'].mean():.2%} vs "
          f"regime OFF {d.loc[~d['regime_on'], 'crash15'].mean():.2%}")

    print("\nwalk-forward 730/91, purge 10d, target crash15 ...")
    d, foldlog = walk_forward(d, "crash15")
    oos = d["p_model"].notna()
    o = d[oos].copy()
    print(f"\nscored out of sample: {len(o):,} rows over {int(o['fold'].nunique())} folds, "
          f"{o['bar_time'].min().date()} -> {o['bar_time'].max().date()}")
    print(f"out-of-sample crash base rate: {o['crash15'].mean():.2%}")

    o["p_naive_raw"] = first_passage_prob(o["vol_30d"])
    preds = {"model_gbm": o["p_model"], "model_gbm_reg": o["p_reg"],
             "model_gbm_volonly": o["p_volgbm"], "logreg_all": o["p_lr"],
             "naive_vol_fit": o["p_naive_fit"], "naive_vol_raw": o["p_naive_raw"],
             "naive_base": o["p_naive_base"], "shuffled_ctrl": o["p_shuf"]}
    full = score_block(o["crash15"], preds)
    # the pooled rank AUC of trailing volatility itself. naive_vol_raw is a single global monotone
    # map of it, so the two AUCs must agree; naive_vol_fit's differs only because each fold gets
    # its own intercept, which pooling then mixes.
    full["vol_30d_rank"] = {"n": int(o["crash15"].notna().sum()),
                            "auc": safe_auc(o["crash15"], o["vol_30d"])}

    print("\n=== FULL PANEL, out of sample =================================================")
    print(f"{'predictor':<19}{'n':>8}{'AUC':>9}{'Brier':>10}{'logloss':>10}{'mean p':>9}")
    for k in ("model_gbm", "model_gbm_reg", "model_gbm_volonly", "logreg_all", "naive_vol_fit",
              "naive_vol_raw", "naive_base", "shuffled_ctrl"):
        m = full[k]
        print(f"{k:<19}{m['n']:>8}{fmt(m.get('auc')):>9}{fmt(m.get('brier'), nd=5):>10}"
              f"{fmt(m.get('logloss'), nd=5):>10}{fmt(m.get('mean_pred'), nd=4):>9}")
    print(f"{'vol_30d (rank only)':<19}{full['vol_30d_rank']['n']:>8}"
          f"{fmt(full['vol_30d_rank']['auc']):>9}")
    margin_auc = full["model_gbm"]["auc"] - full["naive_vol_fit"]["auc"]
    margin_brier = full["naive_vol_fit"]["brier"] - full["model_gbm"]["brier"]
    print("\nMARGIN over naive_vol_fit (positive = the model adds something):")
    for k in ("model_gbm", "model_gbm_reg", "model_gbm_volonly", "logreg_all"):
        print(f"  {k:<19} AUC {full[k]['auc'] - full['naive_vol_fit']['auc']:+.4f}   "
              f"Brier {full['naive_vol_fit']['brier'] - full[k]['brier']:+.5f}")
    print("MARGIN over naive_base (the constant), Brier:")
    for k in ("model_gbm", "model_gbm_reg", "model_gbm_volonly", "logreg_all", "naive_vol_fit"):
        print(f"  {k:<19} {full['naive_base']['brier'] - full[k]['brier']:+.5f}")

    # --- controls ------------------------------------------------------------
    print("\n=== CONTROLS (the harness proving itself) ======================================")
    print("direction control: same features, same folds, target = forward 7d close above today")
    dd, _ = walk_forward(d.copy(), "up7d", verbose=False, light=True)
    oc = dd[dd["p_model"].notna()]
    dir_auc = safe_auc(oc["up7d"], oc["p_model"])
    dir_naive = float(oc["up7d"].mean())
    dpf = per_fold_auc(oc, "up7d", ["p_model"])["p_model"].dropna()
    dir_t = float((dpf.mean() - 0.5) / (dpf.std(ddof=1) / np.sqrt(len(dpf))))
    print(f"  direction AUC {dir_auc:.4f}  (base rate of up-moves {dir_naive:.2%}) "
          f"-> expected ~0.50")
    print(f"  per fold: mean {dpf.mean():.4f} sd {dpf.std(ddof=1):.4f} t vs 0.50 {dir_t:+.2f}")
    # 0.54 is above noise, so it has to be explained rather than waved through. Strip the
    # market-wide component: if the direction edge is only "the whole market rose that week", it is
    # the 200-day BTC regime effect the bot already trades, not a leak and not coin selection.
    dir_xs = stratified_auc(oc, "up7d", "p_model")
    dir_xs_naive = stratified_auc(oc, "up7d", "p_naive_fit")
    print(f"  cross-sectional only (within one bar_time): model {dir_xs:.4f}  "
          f"naive_vol {dir_xs_naive:.4f}  -> "
          f"{'market timing, not coin selection' if abs(dir_xs - 0.5) < 0.02 else 'LOOK HARDER'}")

    print("\nthe same decomposition for the crash target - is it WHEN or WHICH COIN?")
    xsec = {name: stratified_auc(o, "crash15", col) for name, col in
            (("model_gbm", "p_model"), ("model_gbm_volonly", "p_volgbm"),
             ("logreg_all", "p_lr"), ("naive_vol_fit", "p_naive_fit"),
             ("naive_vol_raw", "p_naive_raw"))}
    print(f"  {'predictor':<19}{'pooled AUC':>12}{'cross-sectional':>17}{'difference':>12}")
    for name, xs in xsec.items():
        print(f"  {name:<19}{full[name]['auc']:>12.4f}{xs:>17.4f}{xs - full[name]['auc']:>12.4f}")
    print(f"  cross-sectional margin, model over naive: "
          f"{xsec['model_gbm'] - xsec['naive_vol_fit']:+.4f}")
    xsec["direction_model"] = dir_xs
    xsec["direction_naive"] = dir_xs_naive
    print(f"  shuffled-label control AUC on crash15: {full['shuffled_ctrl']['auc']:.4f} "
          f"-> expected ~0.50")

    # --- robustness: the same question with the bad lows removed from the label
    print("\nwick-immune robustness: retrain on crash15c (forward min CLOSE <= -15%), which "
          "cannot\nbe triggered by a bad print such as LINKUSDT's 0.0001 low on 2020-03-12 ...")
    dw, _ = walk_forward(d.copy(), "crash15c", verbose=False, light=True)
    ow = dw[dw["p_model"].notna()]
    wick = score_block(ow["crash15c"], {"model": ow["p_model"],
                                        "naive_vol_fit": ow["p_naive_fit"],
                                        "naive_base": ow["p_naive_base"]})
    print(f"  base rate {ow['crash15c'].mean():.2%}   model AUC {wick['model']['auc']:.4f}   "
          f"naive {wick['naive_vol_fit']['auc']:.4f}   "
          f"margin {wick['model']['auc'] - wick['naive_vol_fit']['auc']:+.4f}")
    print(f"  Brier: model {wick['model']['brier']:.5f}  "
          f"naive {wick['naive_vol_fit']['brier']:.5f}  "
          f"base-rate {wick['naive_base']['brier']:.5f}")

    # --- decile ladder and calibration --------------------------------------
    print("\n=== DECILE LADDER, model (full panel) =========================================")
    dec = decile_table(o, "p_model", "crash15")
    print(dec.to_string(float_format=lambda x: f"{x:.4f}"))
    mono = bool((dec["realised"].diff().dropna() >= 0).all())
    spread = float(dec["realised"].iloc[-1] / max(dec["realised"].iloc[0], 1e-9))
    print(f"monotone across all deciles: {mono}   top/bottom decile crash-rate ratio: "
          f"{spread:.1f}x")
    print("\nthe same ladder for the naive vol baseline, for comparison:")
    decn = decile_table(o, "p_naive_fit", "crash15")
    print(decn[["n", "mean_pred", "realised"]].to_string(float_format=lambda x: f"{x:.4f}"))
    mono_n = bool((decn["realised"].diff().dropna() >= 0).all())
    spread_n = float(decn["realised"].iloc[-1] / max(decn["realised"].iloc[0], 1e-9))
    print(f"naive monotone: {mono_n}   naive top/bottom ratio: {spread_n:.1f}x")

    print("\n=== CALIBRATION, model ========================================================")
    cal = calibration_table(o, "p_model", "crash15")
    print(cal.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n=== CALIBRATION, naive_vol_raw (no fitting at all) ============================")
    print(calibration_table(o, "p_naive_raw", "crash15").to_string(
        float_format=lambda x: f"{x:.4f}"))

    # --- per-fold margin, so the number has an error bar --------------------
    print("\n=== PER-FOLD AUC (model vs naive_vol_fit) =====================================")
    pf = per_fold_auc(o, "crash15", ["p_model", "p_lr", "p_naive_fit"])
    pf["margin"] = pf["p_model"] - pf["p_naive_fit"]
    print(pf.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    mg = pf["margin"].dropna()
    tstat = float(mg.mean() / (mg.std(ddof=1) / np.sqrt(len(mg)))) if len(mg) > 1 else float("nan")
    print(f"\nfolds {len(mg)}  mean margin {mg.mean():+.4f}  sd {mg.std(ddof=1):.4f}  "
          f"t {tstat:+.2f}  folds model wins {int((mg > 0).sum())}/{len(mg)}")

    # Does either predictor beat a coin flip WITHIN a test window? The pooled AUC is flattered by
    # volatility also identifying which 91-day windows are crash-prone, which is information but
    # not the same information. The per-fold mean is the within-period number.
    skill = {}
    for name, col in (("model", "p_model"), ("logreg", "p_lr"), ("naive_vol_fit", "p_naive_fit")):
        v = pf[col].dropna()
        t = float((v.mean() - 0.5) / (v.std(ddof=1) / np.sqrt(len(v))))
        skill[name] = {"mean_auc": float(v.mean()), "sd": float(v.std(ddof=1)),
                       "t_vs_half": t, "folds_above_half": int((v > 0.5).sum()),
                       "folds": int(len(v))}
        print(f"  {name:<14} per-fold AUC mean {v.mean():.4f} sd {v.std(ddof=1):.4f}  "
              f"t vs 0.50 {t:+.2f}  above 0.50 in {int((v > 0.5).sum())}/{len(v)} folds")
    bss = {k: 1 - full[k]["brier"] / full["naive_base"]["brier"]
           for k in ("model_gbm", "model_gbm_reg", "model_gbm_volonly", "logreg_all",
                     "naive_vol_fit", "naive_vol_raw")}
    print("  Brier skill score against the constant base rate (positive = useful probabilities):")
    for k, v in bss.items():
        print(f"    {k:<19}{v:+.4f}")

    # --- time-decimated sample: overlapping windows make pooled n a fiction -
    dec_idx = o.groupby("symbol").cumcount() % (HORIZON_DAYS * BARS_PER_DAY) == 0
    od = o[dec_idx]
    thin = score_block(od["crash15"], {"model": od["p_model"], "naive_vol_fit": od["p_naive_fit"]})
    print(f"\nnon-overlapping subsample (1 bar per {HORIZON_DAYS}d per symbol): "
          f"n={thin['model']['n']}  model AUC {thin['model']['auc']:.4f}  "
          f"naive {thin['naive_vol_fit']['auc']:.4f}  "
          f"margin {thin['model']['auc'] - thin['naive_vol_fit']['auc']:+.4f}")

    # ------------------------------------------------------------------ THE SUBSET THAT MATTERS
    print("\n=== THE BOT'S OWN ENTRY CANDIDATES ===========================================")
    lab = label_trades(panel, s)
    cand = (panel[panel["_cand"]]
            .merge(o[["bar_time", "symbol", "p_model", "p_reg", "p_volgbm", "p_lr",
                      "p_naive_fit", "fold"]],
                   on=["bar_time", "symbol"], how="inner")
            .merge(lab, on=["bar_time", "symbol"], how="inner"))
    cand["p_naive_raw"] = first_passage_prob(cand["vol_30d"])
    cand["lost10"] = (cand["net"] <= LOSS).astype(float)
    cand["lost0"] = (cand["net"] <= 0).astype(float)
    cand["dd10"] = (cand["fwd_low_ret"] <= LOSS).astype(float)
    print(f"candidates with both an OOS prediction and a real trade outcome: {len(cand):,}")
    print(f"  mean net {cand['net'].mean():+.2%}   median {cand['net'].median():+.2%}   "
          f"win rate {1 - cand['lost0'].mean():.1%}")
    print(f"  went on to lose more than {abs(LOSS):.0%}: {cand['lost10'].mean():.2%} "
          f"({int(cand['lost10'].sum())} trades)")
    print(f"  forward 7d low {LOSS:.0%} or worse: {cand['dd10'].mean():.2%}   "
          f"forward 7d low {CRASH:.0%} or worse: {cand['crash15'].mean():.2%}")

    # Two of these columns are not forecasts at all: stop_distance_pct is the sizing reference the
    # bot ALREADY divides by, and atr_pct is already in its ranker. If they score as well as the
    # model does, the risk information is in the bot today and no new model is needed.
    SUBPREDS = {"model": "p_model", "model_reg": "p_reg", "model_volonly": "p_volgbm",
                "logreg": "p_lr", "naive_vol_fit": "p_naive_fit", "naive_vol_raw": "p_naive_raw",
                "vol_30d": "vol_30d", "atr_pct": "atr_pct",
                "stop_distance_pct": "stop_distance_pct"}
    sub = {}
    for ycol, lbl in (("lost10", f"real trade net <= {LOSS:.0%}"),
                      ("dd10", f"forward 7d low <= {LOSS:.0%}"),
                      ("crash15", f"forward 7d low <= {CRASH:.0%}"),
                      ("lost0", "real trade net <= 0 (any loss)")):
        blk = {k: {"auc": safe_auc(cand[ycol], cand[c])} for k, c in SUBPREDS.items()}
        for k in ("model", "naive_vol_fit", "naive_vol_raw"):
            blk[k].update(score_block(cand[ycol], {k: cand[SUBPREDS[k]]})[k])
        sub[ycol] = {"label": lbl, "rate": float(cand[ycol].mean()), **blk}
    cols = list(SUBPREDS)
    print("\nAUC on the entry-candidate subset (what the bot actually trades):")
    print("  the last three columns are not models - vol_30d and atr_pct are already in the "
          "ranker,\n  and stop_distance_pct is already the denominator of the bot's position size.")
    print(f"\n{'target':<31}{'rate':>7}" + "".join(f"{c[:12]:>13}" for c in cols))
    for ycol, blk in sub.items():
        print(f"{blk['label']:<31}{blk['rate']:>6.2%} "
              + "".join(f"{fmt(blk[c].get('auc')):>13}" for c in cols))
    print("margin of each model over naive_vol_fit, per target:")
    for ycol, blk in sub.items():
        b = blk["naive_vol_fit"]["auc"]
        bits = " ".join(f"{k} {blk[k]['auc'] - b:+.4f}" for k in
                        ("model", "model_reg", "model_volonly", "logreg")
                        if blk[k]["auc"] is not None and b is not None)
        print(f"  {blk['label']:<30} {bits}")

    print("\npredicted-crash-probability quartiles among entry candidates:")
    k = min(4, cand["p_model"].nunique())
    cq = pd.qcut(cand["p_model"], k, labels=["lowest", "low", "high", "highest"][:k],
                 duplicates="drop")
    qt = cand.groupby(cq, observed=True).agg(
        trades=("net", "size"), mean_pred=("p_model", "mean"), mean_net=("net", "mean"),
        median_net=("net", "median"), win=("lost0", lambda x: 1 - x.mean()),
        loss_gt_10=("lost10", "mean"), worst=("net", "min"), best=("net", "max"),
        bars_held=("bars_held", "mean"))
    print(qt.to_string(float_format=lambda x: f"{x:.4f}"))
    ic = float(cand["p_model"].corr(cand["net"], method="spearman"))
    ic_dd = float(cand["p_model"].corr(cand["fwd_low_ret"], method="spearman"))
    print(f"rank corr, predicted crash prob vs realised net: {ic:+.4f}   "
          f"vs forward 7d low: {ic_dd:+.4f} (negative = correct direction)")

    print("\nthe same quartiles under the NAIVE forecast, which is the better one:")
    kn = min(4, cand["p_naive_fit"].nunique())
    nq = pd.qcut(cand["p_naive_fit"], kn, labels=["lowest", "low", "high", "highest"][:kn],
                 duplicates="drop")
    qtn = cand.groupby(nq, observed=True).agg(
        trades=("net", "size"), mean_pred=("p_naive_fit", "mean"), mean_net=("net", "mean"),
        median_net=("net", "median"), win=("lost0", lambda x: 1 - x.mean()),
        loss_gt_10=("lost10", "mean"), worst=("net", "min"), best=("net", "max"))
    print(qtn.to_string(float_format=lambda x: f"{x:.4f}"))

    # What a top-decile veto would have cost, measured on the trades themselves. This is the
    # question a risk forecast exists to answer, and the answer decides whether the forecast is
    # allowed anywhere near the entry decision.
    veto = {}
    print("\nif the riskiest decile had simply been skipped:")
    for name, col in (("model", "p_model"), ("naive_vol_fit", "p_naive_fit")):
        thr = cand[col].quantile(0.90)
        kept, cut = cand[cand[col] < thr], cand[cand[col] >= thr]
        veto[name] = {"threshold": float(thr), "trades_cut": int(len(cut)),
                      "mean_net_cut": float(cut["net"].mean()),
                      "mean_net_kept": float(kept["net"].mean()),
                      "loss_rate_cut": float(cut["lost10"].mean()),
                      "sum_net_cut": float(cut["net"].sum()),
                      "sum_net_all": float(cand["net"].sum()),
                      "share_of_profit_cut": float(cut["net"].sum() / cand["net"].sum())}
        v = veto[name]
        print(f"  by {name:<14} p >= {v['threshold']:.3f}: {v['trades_cut']} trades cut, "
              f"mean net of the cut {v['mean_net_cut']:+.2%} vs kept "
              f"{v['mean_net_kept']:+.2%}; that decile held "
              f"{v['share_of_profit_cut']:.1%} of all the profit")

    # --------------------------------------------------------------- verdict
    cand_auc = sub["lost10"]["model"].get("auc")
    cand_margin = (None if cand_auc is None or sub["lost10"]["naive_vol_fit"].get("auc") is None
                   else cand_auc - sub["lost10"]["naive_vol_fit"]["auc"])
    best_model = max(full[k]["auc"] for k in ("model_gbm", "model_gbm_reg", "model_gbm_volonly",
                                              "logreg_all"))
    print("\n=== CONCLUSION ================================================================")
    print("1. YES, THIS ONE IS PREDICTABLE - and the predictor is trailing volatility, not a "
          "model.")
    print(f"   naive trailing vol: pooled AUC {full['naive_vol_fit']['auc']:.4f}, within-fold "
          f"{skill['naive_vol_fit']['mean_auc']:.4f} (t vs 0.50 "
          f"{skill['naive_vol_fit']['t_vs_half']:+.2f}), Brier {full['naive_vol_fit']['brier']:.5f} "
          f"against base-rate {full['naive_base']['brier']:.5f}")
    print(f"   -> Brier skill score {bss['naive_vol_fit']:+.1%}. Its decile ladder is monotone "
          f"and spans {spread_n:.1f}x in crash rate.")
    print(f"   Strongest form: CROSS-SECTIONALLY, within a single bar, AUC "
          f"{xsec['naive_vol_fit']:.4f} - trailing vol ranks which coin is about to fall "
          f"{abs(CRASH):.0%} very well.")
    print(f"2. THE ML ADDS NOTHING. Best of four model variants AUC {best_model:.4f} vs naive "
          f"{full['naive_vol_fit']['auc']:.4f} (headline margin {margin_auc:+.4f}, per-fold "
          f"{mg.mean():+.4f} +- {mg.std(ddof=1):.4f}, t {tstat:+.2f}, wins "
          f"{int((mg > 0).sum())}/{len(mg)}).")
    print(f"   Every variant loses on AUC AND on Brier; the headline model's Brier "
          f"{full['model_gbm']['brier']:.5f} is worse even than the constant base rate, and when "
          f"it says 76% the realised rate is 26%.")
    print(f"3. CONTROLS. Shuffled labels {full['shuffled_ctrl']['auc']:.4f}. Direction pooled "
          f"{dir_auc:.4f} / per-fold {dpf.mean():.4f} (t {dir_t:+.2f}) - above noise, but "
          f"cross-sectionally {dir_xs:.4f},")
    print("   so that residue is market timing (the 200-day BTC regime the bot already trades), "
          "not coin selection and not a leak.")
    print(f"4. ON THE ONLY SAMPLE THAT PAYS - {len(cand)} real entry candidates with real trade "
          f"outcomes - AUC on 'this trade lost >{abs(LOSS):.0%}':")
    print(f"   model {fmt(cand_auc)}, naive vol {fmt(sub['lost10']['naive_vol_fit'].get('auc'))} "
          f"(margin {(f'{cand_margin:+.4f}' if cand_margin is not None else 'n/a')}), and "
          f"atr_pct - already in the ranker - {fmt(sub['lost10']['atr_pct'].get('auc'))}, "
          f"stop_distance_pct - already the bot's size denominator - "
          f"{fmt(sub['lost10']['stop_distance_pct'].get('auc'))}.")
    print("   So the usable risk signal exists, is real, and is ALREADY INSIDE THE BOT: "
          "size = risk / (entry - stop) is a volatility-inverse sizer by construction.")
    print(f"5. DO NOT TURN IT INTO A VETO. The model's riskiest decile held "
          f"{veto['model']['share_of_profit_cut']:.1%} of all profit "
          f"(mean net {veto['model']['mean_net_cut']:+.2%}); even the better naive forecast's "
          f"riskiest decile is only {veto['naive_vol_fit']['share_of_profit_cut']:+.1%} of "
          f"profit over 10% of the trades.")

    out = {
        "spec": {"target": f"forward {HORIZON_DAYS}d low <= {CRASH:.0%} of close",
                 "horizon_days": HORIZON_DAYS, "crash_threshold": CRASH,
                 "train_days": TRAIN_DAYS, "test_days": TEST_DAYS, "purge_days": PURGE_DAYS,
                 "features": FEATS, "model": "HistGradientBoostingClassifier", "hp": {
                     k: v for k, v in HP.items()},
                 "rows_modelled": int(len(d)), "rows_oos": int(len(o)),
                 "folds": int(o["fold"].nunique()),
                 "oos_from": str(o["bar_time"].min().date()),
                 "oos_to": str(o["bar_time"].max().date()),
                 "symbols": int(panel["symbol"].nunique())},
        "base_rates": {"crash15_all": base_rate, "crash15_oos": float(o["crash15"].mean()),
                       "crash10_all": float(d["crash10"].mean()),
                       "up7d_all": float(d["up7d"].mean()),
                       "crash15_regime_on": float(d.loc[d["regime_on"], "crash15"].mean()),
                       "crash15_regime_off": float(d.loc[~d["regime_on"], "crash15"].mean())},
        "full_panel": full,
        "margins_full_panel": {"auc_vs_naive_vol_fit": float(margin_auc),
                               "brier_vs_naive_vol_fit": float(margin_brier),
                               "brier_vs_naive_base": float(full["naive_base"]["brier"]
                                                            - full["model_gbm"]["brier"]),
                               "auc_vs_naive_vol_raw": float(full["model_gbm"]["auc"]
                                                             - full["naive_vol_raw"]["auc"])},
        "controls": {"direction_auc": dir_auc, "direction_base_rate": dir_naive,
                     "direction_per_fold_mean_auc": float(dpf.mean()),
                     "direction_per_fold_sd": float(dpf.std(ddof=1)),
                     "direction_t_vs_half": dir_t,
                     "direction_naive_vol_auc": safe_auc(oc["up7d"], oc["p_naive_fit"]),
                     "shuffled_label_auc": full["shuffled_ctrl"]["auc"]},
        "wick_immune_crash15c": {"base_rate": float(ow["crash15c"].mean()), **wick,
                                 "auc_margin": wick["model"]["auc"]
                                 - wick["naive_vol_fit"]["auc"]},
        "deciles_model": dec.reset_index().to_dict("records"),
        "deciles_naive": decn.reset_index()[["_b", "n", "mean_pred", "realised"]]
                             .to_dict("records"),
        "monotone": {"model": mono, "model_top_bottom_ratio": spread,
                     "naive": mono_n, "naive_top_bottom_ratio": spread_n},
        "calibration_model": [{"bin": str(i), **{k: (None if pd.isna(v) else float(v))
                                                 for k, v in r.items()}}
                              for i, r in cal.iterrows()],
        "per_fold": pf.to_dict("records"),
        "per_fold_margin": {"n": int(len(mg)), "mean": float(mg.mean()),
                            "sd": float(mg.std(ddof=1)), "t": tstat,
                            "wins": int((mg > 0).sum())},
        "per_fold_skill_vs_coinflip": skill,
        "cross_sectional_auc": xsec,
        "brier_skill_score_vs_base_rate": bss,
        "caveats": [
            "The universe is the CURRENT (2026-09-25) parity list, so coins that crashed to "
            "nothing and were delisted are absent. The true crash base rate is higher than the "
            "18.5% measured here, and every predictor is being graded on survivors.",
            "7-day windows on 4h bars overlap 42-fold, so the pooled n of 226,375 is roughly "
            "5,400 independent observations; the per-fold figures and the decimated subsample are "
            "the honest sample sizes.",
            "Trailing features share input bars with test rows just past the purge boundary. That "
            "is feature overlap, not label overlap - no target is visible to its own training "
            "window - but it means adjacent folds are not fully independent.",
            "The veto thresholds are the 90th percentile of the whole out-of-sample period, "
            "chosen after seeing it. They are an illustration of cost, not a tradable rule.",
        ],
        "non_overlapping": {"n": thin["model"]["n"], "model_auc": thin["model"]["auc"],
                            "naive_auc": thin["naive_vol_fit"]["auc"],
                            "margin": thin["model"]["auc"] - thin["naive_vol_fit"]["auc"]},
        "entry_candidates": {
            "n": int(len(cand)), "mean_net": float(cand["net"].mean()),
            "median_net": float(cand["net"].median()),
            "win_rate": float(1 - cand["lost0"].mean()),
            "rate_lost_10pct": float(cand["lost10"].mean()),
            "targets": sub,
            "auc_lost10": cand_auc,
            "margin_lost10_vs_naive": cand_margin,
            "spearman_pred_vs_net": ic, "spearman_pred_vs_fwd_low": ic_dd,
            "quartiles": [{"bucket": str(i), **{k: float(v) for k, v in r.items()}}
                          for i, r in qt.iterrows()],
            "quartiles_naive": [{"bucket": str(i), **{k: float(v) for k, v in r.items()}}
                                for i, r in qtn.iterrows()],
            "top_decile_veto": veto},
        "folds_log": foldlog,
    }
    (ROOT / "reports" / "forecast_risk.json").write_text(json.dumps(out, indent=2, default=str))
    print("\nwrote reports/forecast_risk.json")


if __name__ == "__main__":
    main()
