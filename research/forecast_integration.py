"""Does a genuinely skilful volatility forecast make the TRADING STRATEGY better?

Three prior studies in this repo established, with adversarial verification, that RETURNS are not
predictable from this data (ranker, ml_veto, win-probability, right-tail classifier - all rejected,
see RESEARCH_LOG.md). Two things in this data ARE predictable out of sample:

  1. 7-day forward realised volatility. A 3-coefficient HAR regression on log(vol_1d), log(vol_7d),
     log(vol_30d) scores pooled out-of-sample log-R2 0.5308 against 0.4270 for the naive
     "next vol = vol_30d" (research/forecast_volatility.py). The margin is +0.1038, 25/28 folds.
  2. 7-day regime persistence, but only mechanically: regime_on is DEFINED as dist_sma200 > 0, so
     a logistic on dist_sma200/vol_30d reaches AUC 0.9779 (research/forecast_regime.py).
  3. P(a 15% adverse move in 7 days), and only through trailing vol: a closed-form driftless
     first-passage probability scores AUC 0.6772 with no fitting (research/forecast_risk.py).

This file asks the only question left: does any of that skill turn into P&L, under the ship rule?
It does NOT re-litigate return prediction and it never deletes a candidate - every variant changes
SIZE, because this project has already established that filtering candidates destroys the return
(16% of trades carry 155% of the profit).

WHAT IS TESTED
  Route 1 - FORECAST-BASED SIZING. Replace the trailing-volatility input to position size with the
    HAR forecast. Cross-sectional form (bar median / own forecast) and absolute form (a trailing
    train-window target / own forecast), both clipped to [0.5, 1.5] by the engine.
  Route 2 - FORECAST-BASED STOP DISTANCE. Set the sizing reference to close*(1 - k*sigma_fcast)
    instead of the 20-bar low, with k calibrated on TRAIN rows only so the median stop distance
    matches the baseline's. Run both with stop_exec off (where stop_level only sets size) and with
    stop_exec on (where it is a real intrabar stop, so the forecast changes the EXIT too).
  Route 3 - FORECAST-BASED RISK GATE. Size x0.5 when the predicted 7-day crash probability is in
    the top train-window quintile, or when P(regime still on in 7 days) < 0.5. Never a refusal.

WHAT MAKES IT HONEST
  * Every forecast is walk-forward: 730d train / 10d purge / 91d test, rolling. The purge exceeds
    the 7-day label horizon, so no training label can see into its own test window. Rows before the
    first fold (2017-08 .. 2019-11) get multiplier 1.0 - the bot genuinely had no forecast then.
  * MATCHED NAIVE CONTROLS. Every forecast variant has a twin driven by vol_30d (and vol_7d),
    applied on exactly the same rows. If the naive twin scores the same, the forecast's measured
    accuracy edge is worth nothing here, whatever its R2.
  * INVERTED CONTROLS. The same shape with the forecast upside down (size UP the wild coin). If
    the inverse does as well, the result is noise, not risk management.
  * CONSTANT-MULTIPLIER CONTROLS. A vol-target multiplier clipped to [0.5, 1.5] has a mean above
    1.0, so part of any gain is simply extra leverage. A flat multiplier equal to the variant's own
    realised mean separates allocation from leverage.
  * PLACEBO NULL. 40 variants of the winning shape driven by randomised forecasts (20 within-bar
    permutations, 20 global permutations). The fraction that pass the ship rule is the noise floor.
  * SUB-PERIOD ATTRIBUTION and COST STRESS at 1.5x and 2x.

SHIP RULE (both panels): CAGR >= baseline AND Sharpe >= baseline AND maxDD not worse.
  standard   CAGR 0.2073  Sharpe 1.4295  maxDD -0.1324  trades 366
  held-out   CAGR 0.1548  Sharpe 1.2229  maxDD -0.1324  (from 2022-01-01)

Nothing outside this file is modified. Sizing goes through backtest.run(size_scale_col=...) and
through the stop_level column on a COPY of the panel; costs go through a COPY of the settings.

    PYTHONIOENCODING=utf-8 python research/forecast_integration.py
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import backtest, config, features, signals  # noqa: E402
from app.flags import Flags  # noqa: E402

SHIPPED = dict(abs_momentum=True, volume_min=0.8, taker_min=0.5)
HELD_OUT = "2022-01-01"
HORIZON = 42                       # 42 four-hour bars = 7 days
ANNUAL = np.sqrt(6 * 365)          # exactly as app/features.py annualises vol_7d / vol_30d
TRAIN_DAYS, TEST_DAYS, PURGE_DAYS = 730, 91, 10
HAR_FEATS = ["lv1", "lv7", "lv30"]
CRASH_DROP = 0.15                  # the adverse move forecast_risk.py found predictable
STOP_BARS = 20                     # the Donchian window the baseline stop uses

BASE = {"standard": dict(cagr=0.2073, sharpe=1.4295, maxdd=-0.1324, trades=366),
        "held_out": dict(cagr=0.1548, sharpe=1.2229, maxdd=-0.1324)}


# ===================================================================== data and labels

def build(s):
    u = json.loads((ROOT / "reports" / "_parity_universe.json").read_text())
    panel = features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars)

    step = pd.Timedelta(hours=4 * HORIZON)
    out = []
    for _, g in panel.sort_values(["symbol", "bar_time"]).groupby("symbol", sort=False):
        r = np.log(g["close"]).diff()
        # forward realised vol over the NEXT 42 bars, annualised like vol_7d. Only kept where the
        # 42 bars are actually contiguous (this history has outage gaps up to 1d8h).
        fwd = r.rolling(HORIZON).std().shift(-HORIZON) * ANNUAL
        contiguous = (g["bar_time"].shift(-HORIZON) - g["bar_time"]) == step
        # the realised forward WINDOW LOW, for the crash label
        fwd_low = g["low"].rolling(HORIZON).min().shift(-HORIZON)
        out.append(g.assign(fwd_vol=fwd.where(contiguous),
                            fwd_low=fwd_low.where(contiguous),
                            vol_1d=r.rolling(6).std() * ANNUAL))
    panel = pd.concat(out).sort_index()        # restore the original row order / index
    panel["crash"] = (panel["fwd_low"] < panel["close"] * (1 - CRASH_DROP)).astype(float)
    panel.loc[panel["fwd_low"].isna(), "crash"] = np.nan

    ok = (panel["eligible"] & panel["vol_1d"].gt(0) & panel["vol_7d"].gt(0)
          & panel["vol_30d"].gt(0)).fillna(False)
    panel["_har_ok"] = ok
    panel.loc[ok, "lv1"] = np.log(panel.loc[ok, "vol_1d"])
    panel.loc[ok, "lv7"] = np.log(panel.loc[ok, "vol_7d"])
    panel.loc[ok, "lv30"] = np.log(panel.loc[ok, "vol_30d"])
    panel["y"] = np.log(panel["fwd_vol"].where(panel["fwd_vol"] > 0))
    return panel


# ===================================================================== walk-forward forecasts

def ols(X, y):
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def folds_of(panel):
    first, last = panel["bar_time"].min(), panel["bar_time"].max()
    purge = pd.Timedelta(days=PURGE_DAYS)
    t0 = first + pd.Timedelta(days=TRAIN_DAYS) + purge
    out = []
    while t0 < last:
        out.append((t0 - purge - pd.Timedelta(days=TRAIN_DAYS), t0 - purge,
                    t0, t0 + pd.Timedelta(days=TEST_DAYS)))
        t0 += pd.Timedelta(days=TEST_DAYS)
    return out


def har_forecast(panel, verbose=True):
    """pred_vol per (bar, coin), walk-forward HAR. Also records the per-fold skill vs the naives."""
    panel["pred_vol"] = np.nan
    panel["train_med_pred"] = np.nan          # causal absolute-vol target, per fold
    panel["train_q80_crash"] = np.nan         # causal top-quintile threshold, per fold
    rows, betas = [], []
    feat = panel["_har_ok"].to_numpy()
    lab = panel["y"].notna().to_numpy()
    bt = panel["bar_time"]

    for k, (tr0, tr1, te0, te1) in enumerate(folds_of(panel)):
        tr = panel[(bt > tr0) & (bt <= tr1) & feat & lab]
        te_i = panel.index[(bt > te0) & (bt <= te1) & feat]
        if len(tr) < 2000 or len(te_i) < 200:
            continue
        mu, sd = tr[HAR_FEATS].mean().to_numpy(), tr[HAR_FEATS].std().to_numpy()
        sd = np.where(sd > 0, sd, 1.0)
        Ztr = (tr[HAR_FEATS].to_numpy() - mu) / sd
        b = ols(Ztr, tr["y"].to_numpy())
        p_tr = b[0] + Ztr @ b[1:]
        # exp(E[log v]) is a median, not a mean: one multiplicative correction, fitted on TRAIN.
        kcal = float(tr["fwd_vol"].mean() / np.exp(p_tr).mean())
        Zte = (panel.loc[te_i, HAR_FEATS].to_numpy() - mu) / sd
        pred = kcal * np.exp(b[0] + Zte @ b[1:])
        panel.loc[te_i, "pred_vol"] = pred
        panel.loc[te_i, "train_med_pred"] = float(np.median(kcal * np.exp(p_tr)))
        # the crash-probability threshold, also from TRAIN rows only
        ptr_crash = crash_prob(pd.Series(kcal * np.exp(p_tr)))
        panel.loc[te_i, "train_q80_crash"] = float(ptr_crash.quantile(0.80))
        betas.append(dict(fold=k, const=float(b[0]),
                          **{f: float(v) for f, v in zip(HAR_FEATS, b[1:])}))

        # per-fold skill, scored only where the label has resolved
        sc = panel.loc[te_i].assign(p=pred)
        sc = sc[sc["y"].notna()]
        if len(sc) > 50:
            rows.append(dict(fold=k, test_start=str(te0.date()), n=len(sc),
                             r2_har=_r2(sc["y"], np.log(sc["p"])),
                             r2_v30=_r2(sc["y"], sc["lv30"]),
                             r2_v7=_r2(sc["y"], sc["lv7"])))
    sk = pd.DataFrame(rows)
    if verbose:
        pooled = panel[panel["pred_vol"].notna() & panel["y"].notna()]
        print(f"  HAR walk-forward: {len(sk)} folds, {len(pooled):,} scored rows, "
              f"{pooled['bar_time'].min().date()} .. {pooled['bar_time'].max().date()}")
        print(f"  pooled OOS log-R2  HAR {_r2(pooled['y'], np.log(pooled['pred_vol'])):.4f}   "
              f"naive vol_30d {_r2(pooled['y'], pooled['lv30']):.4f}   "
              f"naive vol_7d {_r2(pooled['y'], pooled['lv7']):.4f}")
        print(f"  folds where HAR beats vol_30d: {(sk.r2_har > sk.r2_v30).sum()}/{len(sk)}   "
              f"spearman(pred, realised) "
              f"{pooled['pred_vol'].corr(pooled['fwd_vol'], method='spearman'):+.4f}")
    return panel, sk, betas


def _r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def crash_prob(sigma_ann, drop=CRASH_DROP, bars=HORIZON):
    """P(min of a driftless GBM over `bars` falls more than `drop` below spot), closed form.

    Reflection principle: P(min_{0..T} W < -a) = 2*Phi(-a / (sigma*sqrt(T))). No fitting, and
    forecast_risk.py measured it at AUC 0.6772 out of sample straight out of the formula.
    """
    sig = pd.Series(sigma_ann).astype(float) * np.sqrt(bars / (6 * 365))
    a = -np.log(1 - drop)
    return pd.Series(2 * norm.cdf(-a / sig.replace(0, np.nan)), index=sig.index)


def regime_forecast(panel, verbose=True):
    """P(regime_on at t + 42 bars), walk-forward logistic on the single feature dist/vol.

    forecast_regime.py established that the 14-feature GBM LOSES to this one number, so the one
    number is what gets tested here. BTC-only series, joined back onto every row by bar_time.
    """
    btc = panel[panel["symbol"] == "BTCUSDT"].sort_values("bar_time").copy()
    step = pd.Timedelta(hours=4 * HORIZON)
    btc["y_reg"] = btc["regime_on"].astype(float).shift(-HORIZON)
    btc.loc[(btc["bar_time"].shift(-HORIZON) - btc["bar_time"]) != step, "y_reg"] = np.nan
    btc["x"] = btc["dist_sma200"] / btc["vol_30d"]
    btc["p_reg"] = np.nan
    ok = btc["x"].replace([np.inf, -np.inf], np.nan).notna()
    bt = btc["bar_time"]
    n_fold = 0
    for tr0, tr1, te0, te1 in folds_of(btc):
        tr = btc[(bt > tr0) & (bt <= tr1) & ok & btc["y_reg"].notna()]
        te_i = btc.index[(bt > te0) & (bt <= te1) & ok]
        if len(tr) < 500 or len(te_i) < 50 or tr["y_reg"].nunique() < 2:
            continue
        m = LogisticRegression(C=1.0, max_iter=1000).fit(tr[["x"]].to_numpy(), tr["y_reg"])
        btc.loc[te_i, "p_reg"] = m.predict_proba(btc.loc[te_i, ["x"]].to_numpy())[:, 1]
        n_fold += 1
    if verbose:
        sc = btc[btc["p_reg"].notna() & btc["y_reg"].notna()]
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(sc["y_reg"], sc["p_reg"]) if sc["y_reg"].nunique() > 1 else np.nan
        pers = roc_auc_score(sc["y_reg"], sc["regime_on"].astype(float))
        print(f"  regime logistic: {n_fold} folds, {len(sc):,} scored BTC bars, "
              f"OOS AUC {auc:.4f} vs plain persistence {pers:.4f}")
        print(f"    bars it calls at risk (p<0.5) while the regime is ON: "
              f"{int(((sc['p_reg'] < 0.5) & sc['regime_on']).sum()):,} "
              f"({((sc['p_reg'] < 0.5) & sc['regime_on']).mean():.2%} of scored bars)")
    return panel.join(btc.set_index("bar_time")[["p_reg"]], on="bar_time")


# ===================================================================== multiplier columns

def xs_mult(panel, col, invert=False):
    """Cross-sectional vol target: bar median of `col` over its own forecast. The brief's shape."""
    v = panel[col].replace([np.inf, -np.inf], np.nan)
    el = panel["eligible"].fillna(False)
    med = v.where(el).groupby(panel["bar_time"]).transform("median")
    m = (v / med) if invert else (med / v)
    return m.clip(0.5, 1.5).fillna(1.0)


def abs_mult(panel, col, target_col, invert=False):
    """Absolute vol target against a causal (train-window) reference level."""
    v = panel[col].replace([np.inf, -np.inf], np.nan)
    t = panel[target_col]
    m = (v / t) if invert else (t / v)
    return m.clip(0.5, 1.5).fillna(1.0)


def mean_neutral(mult, panel, on):
    """Rescale a multiplier so it averages 1.0 at entry candidates: a pure REALLOCATION of the same
    risk budget, with the leverage change taken out.

    This matters more than it looks. median/x is right-skewed and then clipped at [0.5, 1.5], so the
    raw cross-sectional vol target comes out with a mean BELOW 1 at the bars this strategy actually
    trades - it quietly de-levers the whole book by ~9% as well as tilting between coins. Those two
    effects have to be separated before anything can be attributed to the forecast. The divisor is
    the TRAIN-window mean over candidate rows, so it is causal.
    """
    m = mult.copy()
    bt = panel["bar_time"]
    for tr0, tr1, te0, te1 in folds_of(panel):
        tr = (bt > tr0) & (bt <= tr1) & on
        te = (bt > te0) & (bt <= te1)
        if tr.sum() < 100:
            continue
        mu = float(mult[tr].mean())
        if mu > 0:
            m[te] = mult[te] / mu
    return m.clip(0.5, 1.5).fillna(1.0)


def matched(mult, panel):
    """Apply a multiplier ONLY on rows where the HAR forecast exists, so a naive control is
    compared over exactly the same bars rather than over two extra years the model never saw."""
    return mult.where(panel["pred_vol"].notna(), 1.0)


def gate_mult(cond, lo=0.5):
    """Size x`lo` where `cond` is true. Never a refusal - this project has established that
    deleting candidates destroys the return."""
    return pd.Series(np.where(cond.fillna(False), lo, 1.0), index=cond.index)


def vol_stop(panel, col, train_k):
    """Sizing reference from a forecast instead of the 20-bar low:
         stop = close * (1 - k * sigma_h),  sigma_h = sigma_ann * sqrt(STOP_BARS / (6*365))
    `k` is calibrated per fold on TRAIN rows so the MEDIAN stop distance matches the baseline's;
    without that the variant would silently be a leverage change rather than a stop change."""
    sig = panel[col] * np.sqrt(STOP_BARS / (6 * 365))
    stop = panel["close"] * (1 - (train_k * sig).clip(0.005, 0.60))
    return stop.where(panel[col].notna() & train_k.notna(), panel["stop_level"])


def calibrate_k(panel, col, on):
    """Per-fold k = median(baseline stop distance %) / median(forecast sigma_h), TRAIN rows only.

    `on` must be the ENTRY-CANDIDATE mask, not every eligible bar. A breakout bar has just printed a
    new high, so its 20-bar low sits far below the close (median 16.4%) while a typical bar's sits
    much closer. Calibrating on all bars therefore makes the forecast stop ~2.7x tighter at the only
    bars that matter, which multiplies the risk-based size by 2.7 and turns a test of stop SHAPE
    into a test of leverage. Calibrating on candidates holds the level fixed and isolates the shape.
    """
    base_d = (1 - panel["stop_level"] / panel["close"])
    sig = panel[col] * np.sqrt(STOP_BARS / (6 * 365))
    k = pd.Series(np.nan, index=panel.index)
    bt = panel["bar_time"]
    for tr0, tr1, te0, te1 in folds_of(panel):
        tr = (bt > tr0) & (bt <= tr1) & on & base_d.gt(0) & sig.gt(0)
        te = (bt > te0) & (bt <= te1)
        if tr.sum() < 100:
            continue
        k[te] = float(base_d[tr].median() / sig[tr].median())
    return k


# ===================================================================== running and scoring

def run_variant(panel, s, flags, mult=None, stop_col=None, **kw):
    p = panel
    if mult is not None or stop_col is not None:
        p = panel.copy()
    if stop_col is not None:
        p["stop_level"] = p[stop_col]
    if mult is not None:
        p["_mult"] = mult
        return backtest.run(p, s, flags=flags, size_scale_col="_mult", **kw)
    return backtest.run(p, s, flags=flags, **kw)


def m_of(res):
    m = backtest.metrics(res.equity, res.trades, res.exposure)
    return dict(cagr=float(m["cagr"]), sharpe=float(m["sharpe"]),
                maxdd=float(m["max_drawdown"]), trades=int(m.get("trades", 0)),
                win=float(m.get("win_rate", np.nan)), expo=float(m.get("avg_exposure", np.nan)),
                avg_trade=float(m.get("avg_trade", np.nan)))


def passes(v, b, eps=1e-9):
    return bool(v["cagr"] >= b["cagr"] - eps and v["sharpe"] >= b["sharpe"] - eps
                and v["maxdd"] >= b["maxdd"] - eps)


def entry_rows(panel, s, flags):
    """The engine's own entry predicate, reused so the two cannot drift apart."""
    btc = panel["symbol"] == "BTCUSDT"
    return (panel["eligible"] & panel["regime_on"] & panel["breakout"]
            & ((panel["rs_7d"] > 0) | btc) & (panel["rank"] <= s.candidates)
            & signals.extra_filters(panel, flags)).fillna(False)


def fill_mult(res, panel, mult):
    """Mean multiplier over the entries the engine ACTUALLY took - the number that says whether a
    variant reallocated the same risk budget or just added leverage."""
    log = res.log
    if log.empty or "side" not in log:
        return np.nan
    buys = log[log["side"] == "BUY"][["time", "symbol"]]
    key = pd.MultiIndex.from_frame(panel[["bar_time", "symbol"]])
    ser = pd.Series(mult.to_numpy(), index=key)
    # a BUY fills at the open of the bar AFTER the decision bar
    want = pd.MultiIndex.from_arrays([buys["time"] - pd.Timedelta(hours=4), buys["symbol"]])
    v = ser.reindex(want)
    return float(v.mean())


def trade_diagnostic(res, panel, M):
    """Join the BASELINE's own trades to the multiplier each variant would have given them.

    This is the mechanism check. A vol-target multiplier is a bet that the calm coins pay better per
    unit of risk. If the strategy's profit instead lives in its most volatile entries, then a BETTER
    volatility forecast sizes the winners down more accurately, and accuracy costs money. This
    project already established that 16% of trades carry 155% of the profit, so the question is
    simply which side of the multiplier those trades sit on.
    """
    tr = res.trades.copy()
    if tr.empty:
        return {}
    key = pd.MultiIndex.from_frame(panel[["bar_time", "symbol"]])
    want = pd.MultiIndex.from_arrays([tr["entry_time"] - pd.Timedelta(hours=4), tr["symbol"]])
    cols = {"pred_vol": panel["pred_vol"], "vol_30d": panel["vol_30d"]}
    cols.update({k: v for k, v in M.items() if k in
                 ("size_fcast_xs", "size_v30_xs_matched", "gate_crash_fcast")})
    for name, ser in cols.items():
        tr[name] = pd.Series(ser.to_numpy(), index=key).reindex(want).to_numpy()
    tr = tr[tr["pred_vol"].notna()]
    res_out = {"n_trades_with_forecast": int(len(tr))}
    for name in ("size_fcast_xs", "size_v30_xs_matched"):
        sub = tr[tr[name].notna()]
        q = pd.qcut(sub[name], 3, labels=["sized DOWN", "middle", "sized UP"], duplicates="drop")
        g = sub.groupby(q, observed=True)["ret"].agg(["count", "mean", "median", "sum"])
        g["share_of_total_pnl"] = sub.groupby(q, observed=True)["pnl"].sum() / sub["pnl"].sum()
        ic = float(sub[name].corr(sub["ret"], method="spearman"))
        print(f"\n  {name}  (spearman(multiplier, trade return) = {ic:+.4f}, n={len(sub)})")
        print(g.to_string(float_format=lambda x: f"{x:+.4f}"))
        lo, hi = g["mean"].iloc[0], g["mean"].iloc[-1]
        print(f"    the tercile it sizes DOWN averaged {lo:+.2%} per trade; the one it sizes UP "
              f"{hi:+.2%}  -> sizing down the right coins? {'YES' if hi > lo else 'NO, BACKWARDS'}")
        res_out[name] = {"spearman_mult_vs_return": ic, "n": int(len(sub)),
                         "table": g.reset_index().rename(columns={name: "bucket"})
                                   .astype({"bucket": str}).to_dict("records")}
    # the big winners: which side of the multiplier do they sit on?
    big = tr[tr["ret"] > 0.20]
    print(f"\n  the {len(big)} trades that beat +20% carry "
          f"{big['pnl'].sum() / tr['pnl'].sum():.0%} of all P&L; their mean size_fcast_xs "
          f"multiplier is {big['size_fcast_xs'].mean():.3f} against {tr['size_fcast_xs'].mean():.3f} "
          f"for every trade")
    res_out["big_winners"] = {"n": int(len(big)),
                              "share_of_pnl": float(big["pnl"].sum() / tr["pnl"].sum()),
                              "mean_mult": float(big["size_fcast_xs"].mean()),
                              "mean_mult_all": float(tr["size_fcast_xs"].mean()),
                              "mean_pred_vol": float(big["pred_vol"].mean()),
                              "mean_pred_vol_all": float(tr["pred_vol"].mean())}
    return res_out


# ===================================================================== attribution / stress

def quarterly_attrib(var_eq, base_eq):
    """Where does the outperformance come from? Quarterly log-return differences."""
    a = np.log(var_eq.resample("QE").last()).diff().dropna()
    b = np.log(base_eq.resample("QE").last()).diff().dropna()
    d = (a - b).dropna()
    tot = d.sum()
    top = d.abs().sort_values(ascending=False).head(3)
    return {"total_log_outperf": float(tot),
            "n_quarters": int(len(d)),
            "quarters_ahead": int((d > 0).sum()),
            "top3": [{"quarter": str(q.date()), "log_diff": float(d[q]),
                      "share_of_total": float(d[q] / tot) if tot else np.nan} for q in top.index],
            "share_from_best_quarter": float(d.max() / tot) if tot else np.nan,
            "yearly": {str(k): float(v) for k, v in
                       (np.log(var_eq.resample("YE").last()).diff()
                        - np.log(base_eq.resample("YE").last()).diff()).dropna().items()}}


def cost_stress(panel, s, flags, mult, stop_col, label, out):
    print(f"\n--- cost stress: {label} ---")
    for mu in (1.5, 2.0):
        s2 = copy.deepcopy(s)
        s2.slippage_bps = s.slippage_bps * mu
        s2.fee_rate = s.fee_rate * mu
        b_std = m_of(run_variant(panel, s2, flags))
        b_hld = m_of(run_variant(panel, s2, flags, start=HELD_OUT))
        v_std = m_of(run_variant(panel, s2, flags, mult=mult, stop_col=stop_col))
        v_hld = m_of(run_variant(panel, s2, flags, mult=mult, stop_col=stop_col, start=HELD_OUT))
        ok = passes(v_std, b_std) and passes(v_hld, b_hld)
        out.append({"label": label, "cost_mult": mu, "ships_vs_restressed_baseline": ok,
                    "baseline_standard": b_std, "variant_standard": v_std,
                    "baseline_held": b_hld, "variant_held": v_hld})
        print(f"  {mu:g}x costs  standard  base {b_std['cagr']:+.4f}/{b_std['sharpe']:.3f}  "
              f"var {v_std['cagr']:+.4f}/{v_std['sharpe']:.3f}   "
              f"held base {b_hld['cagr']:+.4f}/{b_hld['sharpe']:.3f}  "
              f"var {v_hld['cagr']:+.4f}/{v_hld['sharpe']:.3f}   ships {ok}")


def placebo(panel, s, flags, shape, n=40, seed=0):
    """40 variants of the winning SHAPE driven by randomised forecasts.

    20 permute pred_vol WITHIN each bar (keeps the cross-sectional distribution, destroys which
    coin is which) and 20 permute it globally. Both carry no information. The share that passes the
    ship rule is the noise floor the real variant has to clear.
    """
    rng = np.random.default_rng(seed)
    have = panel["pred_vol"].notna().to_numpy()
    rows = []
    for i in range(n):
        p = panel.copy()
        v = p["pred_vol"].to_numpy(float).copy()
        idx = np.flatnonzero(have)
        if i < n // 2:
            v[idx] = _perm_within(v[idx], p["bar_time"].to_numpy()[idx], rng)
        else:
            v[idx] = rng.permutation(v[idx])
        p["pred_vol"] = v
        mult = shape(p)
        a = m_of(run_variant(p, s, flags, mult=mult))
        b = m_of(run_variant(p, s, flags, mult=mult, start=HELD_OUT))
        rows.append({"i": i, "kind": "within_bar" if i < n // 2 else "global",
                     "std_cagr": a["cagr"], "std_sharpe": a["sharpe"], "std_maxdd": a["maxdd"],
                     "held_cagr": b["cagr"], "held_sharpe": b["sharpe"], "held_maxdd": b["maxdd"],
                     "ships": passes(a, BASE["standard"]) and passes(b, BASE["held_out"])})
    return pd.DataFrame(rows)


def _perm_within(vals, groups, rng):
    """Permute `vals` inside each contiguous run of equal `groups`."""
    out = vals.copy()
    edges = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1], True])
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a > 1:
            out[a:b] = rng.permutation(vals[a:b])
    return out


# ===================================================================== main

def main():
    s = config.load()
    flags = Flags(**SHIPPED)
    out = {}

    print("building panel, labels and walk-forward forecasts ...")
    panel = build(s)
    panel, skill, betas = har_forecast(panel)
    panel = regime_forecast(panel)
    first_pred = panel.loc[panel["pred_vol"].notna(), "bar_time"].min()
    print(f"  first forecast bar {first_pred.date()}  -> rows before it size normally (mult 1.0)")
    out["forecast_skill"] = {
        "folds": int(len(skill)), "folds_har_beats_vol30": int((skill.r2_har > skill.r2_v30).sum()),
        "pooled_r2_har": _r2(panel.loc[panel.pred_vol.notna(), "y"],
                             np.log(panel.loc[panel.pred_vol.notna(), "pred_vol"])),
        "pooled_r2_vol30": _r2(panel.loc[panel.pred_vol.notna(), "y"],
                               panel.loc[panel.pred_vol.notna(), "lv30"]),
        "per_fold": skill.to_dict("records"), "har_betas": betas,
        "first_forecast_bar": str(first_pred)}

    # crash probabilities, from the forecast and from the naive trailing vol
    panel["cp_fcast"] = crash_prob(panel["pred_vol"])
    panel["cp_v30"] = crash_prob(panel["vol_30d"])

    # ------------------------------------------------------------- how much is even reachable
    cand = entry_rows(panel, s, flags)
    cov = panel.loc[cand, "pred_vol"].notna().mean()
    print(f"\ncandidate bars {int(cand.sum()):,}; a forecast exists on {cov:.1%} of them "
          f"({int((cand & panel['pred_vol'].notna()).sum()):,} rows)")
    out["coverage"] = {"candidate_bars": int(cand.sum()), "with_forecast": float(cov)}

    # ------------------------------------------------------------- multiplier library
    k_f = calibrate_k(panel, "pred_vol", cand)
    k_n = calibrate_k(panel, "vol_30d", cand)
    panel["stop_fcast"] = vol_stop(panel, "pred_vol", k_f)
    panel["stop_v30"] = vol_stop(panel, "vol_30d", k_n)
    # the leverage-only control: the SAME 20-bar-low stop, moved to the forecast stop's median
    # distance. No information at all, so if it matches the forecast stop, the forecast added none.
    lev = float(((1 - panel["stop_level"] / panel["close"])[cand].median())
                / ((1 - panel["stop_fcast"] / panel["close"])[cand & panel["pred_vol"].notna()]
                   .median()))
    panel["stop_const_lev"] = panel["close"] - (panel["close"] - panel["stop_level"]) / lev
    out["stop_leverage_factor_of_forecast_stop"] = lev

    M = {}
    M["size_fcast_xs"] = xs_mult(panel, "pred_vol")
    M["size_v30_xs"] = xs_mult(panel, "vol_30d")
    M["size_v30_xs_matched"] = matched(xs_mult(panel, "vol_30d"), panel)
    M["size_v7_xs_matched"] = matched(xs_mult(panel, "vol_7d"), panel)
    M["size_fcast_xs_INVERTED"] = xs_mult(panel, "pred_vol", invert=True)
    M["size_fcast_abs"] = abs_mult(panel, "pred_vol", "train_med_pred")
    M["size_v30_abs_matched"] = matched(abs_mult(panel, "vol_30d", "train_med_pred"), panel)
    M["gate_crash_fcast"] = gate_mult(panel["cp_fcast"] > panel["train_q80_crash"])
    M["gate_crash_v30_matched"] = matched(
        gate_mult(panel["cp_v30"] > panel["train_q80_crash"]), panel)
    M["gate_regime_p50"] = gate_mult(panel["p_reg"] < 0.5)
    M["gate_regime_p50_INVERTED"] = gate_mult(panel["p_reg"] >= 0.5)
    M["size_fcast_xs+gate_regime"] = (M["size_fcast_xs"] * M["gate_regime_p50"]).clip(0.5, 1.5)
    M["size_fcast_xs+gate_crash"] = (M["size_fcast_xs"] * M["gate_crash_fcast"]).clip(0.5, 1.5)
    # mean-neutral forms: the same tilt with the incidental de-levering removed (see mean_neutral)
    M["size_fcast_xs_meanneutral"] = mean_neutral(M["size_fcast_xs"], panel, cand)
    M["size_v30_xs_mn_matched"] = matched(
        mean_neutral(xs_mult(panel, "vol_30d"), panel, cand), panel)
    M["size_fcast_xs_mn_INVERTED"] = mean_neutral(
        xs_mult(panel, "pred_vol", invert=True), panel, cand)

    # ------------------------------------------------------------- baselines
    print("\n=== BASELINE (recomputed) ===")
    b_std = m_of(run_variant(panel, s, flags))
    b_hld = m_of(run_variant(panel, s, flags, start=HELD_OUT))
    for tag, m, q in (("standard", b_std, BASE["standard"]), ("held_out", b_hld, BASE["held_out"])):
        print(f"  {tag:9s} CAGR {m['cagr']:.4f} (quoted {q['cagr']:.4f})  "
              f"Sharpe {m['sharpe']:.4f} (quoted {q['sharpe']:.4f})  "
              f"maxDD {m['maxdd']:.4f} (quoted {q['maxdd']:.4f})  trades {m['trades']}")
    out["baseline"] = {"standard": b_std, "held_out": b_hld, "quoted": BASE}

    # cross-check: the cross-sectional vol_30d multiplier built here must equal the engine's own
    # vol_target flag, or the plumbing is wrong.
    a = run_variant(panel, s, flags, mult=M["size_v30_xs"])
    b = run_variant(panel, s, Flags(**SHIPPED, vol_target=1.0))
    same = bool(np.allclose(a.equity.to_numpy(), b.equity.to_numpy(), rtol=1e-9, atol=1e-6))
    print(f"\nplumbing check: size_scale_col route == engine's own vol_target flag: {same}")
    out["plumbing_check_vol_target_matches_flag"] = same

    # ------------------------------------------------------------- the variant table
    print("\n=== VARIANTS (shipped flags; only SIZE changes; the engine clips to [0.5,1.5]) ===")
    hdr = (f"{'variant':30s} {'CAGR':>8s} {'Sharpe':>7s} {'maxDD':>8s} {'trd':>4s} "
           f"{'expo':>6s} {'mult':>5s} | {'hCAGR':>7s} {'hShrp':>6s} {'hDD':>7s}  ship")
    print(hdr)
    print("-" * len(hdr))
    table = []
    for name, mult in M.items():
        r_std = run_variant(panel, s, flags, mult=mult)
        r_hld = run_variant(panel, s, flags, mult=mult, start=HELD_OUT)
        v_std, v_hld = m_of(r_std), m_of(r_hld)
        fm = fill_mult(r_std, panel, mult)
        ship = passes(v_std, BASE["standard"]) and passes(v_hld, BASE["held_out"])
        ship_r = passes(v_std, b_std) and passes(v_hld, b_hld)
        table.append(dict(variant=name, standard=v_std, held_out=v_hld, mean_fill_mult=fm,
                          ships_vs_quoted=ship, ships_vs_recomputed=ship_r))
        print(f"{name:30s} {v_std['cagr']:8.4f} {v_std['sharpe']:7.4f} {v_std['maxdd']:8.4f} "
              f"{v_std['trades']:4d} {v_std['expo']:6.3f} {fm:5.3f} | "
              f"{v_hld['cagr']:7.4f} {v_hld['sharpe']:6.3f} {v_hld['maxdd']:7.4f}  "
              f"{'SHIP' if ship else '-'}")

    # constant-multiplier controls: how much of any gain is just leverage?
    print("\n--- constant-multiplier controls (pure leverage, no information at all) ---")
    consts = sorted({round(float(t["mean_fill_mult"]), 3) for t in table
                     if np.isfinite(t["mean_fill_mult"])} | {1.1, 1.2})
    const_by_mult = {}
    for cv in consts:
        mult = pd.Series(float(np.clip(cv, 0.5, 1.5)), index=panel.index)
        v_std = m_of(run_variant(panel, s, flags, mult=mult))
        v_hld = m_of(run_variant(panel, s, flags, mult=mult, start=HELD_OUT))
        ship = passes(v_std, BASE["standard"]) and passes(v_hld, BASE["held_out"])
        table.append(dict(variant=f"const_{cv:g}", standard=v_std, held_out=v_hld,
                          mean_fill_mult=cv, ships_vs_quoted=ship,
                          ships_vs_recomputed=passes(v_std, b_std) and passes(v_hld, b_hld)))
        const_by_mult[cv] = {"standard": v_std, "held_out": v_hld}
        print(f"{'const ' + format(cv, 'g'):30s} {v_std['cagr']:8.4f} {v_std['sharpe']:7.4f} "
              f"{v_std['maxdd']:8.4f} {v_std['trades']:4d} {v_std['expo']:6.3f} {cv:5.3f} | "
              f"{v_hld['cagr']:7.4f} {v_hld['sharpe']:6.3f} {v_hld['maxdd']:7.4f}  "
              f"{'SHIP' if ship else '-'}")

    # ------------------------------------------------------------- leverage-adjusted comparison
    print("\n=== LEVERAGE-ADJUSTED: each variant against the FLAT multiplier of its own mean ===")
    print("(this is the only comparison that measures the forecast's ALLOCATION skill. A raw")
    print(" median/x multiplier also de-levers the book, and that is not information.)")
    hdr2 = (f"{'variant':30s} {'mult':>5s} {'CAGR':>8s} {'flat':>8s} {'tilt':>8s} | "
            f"{'Sharpe':>7s} {'flat':>7s} {'tilt':>7s}")
    print(hdr2)
    print("-" * len(hdr2))
    ladj = []
    for t in table:
        if t["variant"].startswith("const_") or not np.isfinite(t["mean_fill_mult"]):
            continue
        cv = min(const_by_mult, key=lambda c: abs(c - t["mean_fill_mult"]))
        if abs(cv - t["mean_fill_mult"]) > 0.01:
            continue
        f = const_by_mult[cv]
        d_c = t["standard"]["cagr"] - f["standard"]["cagr"]
        d_s = t["standard"]["sharpe"] - f["standard"]["sharpe"]
        ladj.append(dict(variant=t["variant"], mean_mult=t["mean_fill_mult"], flat_mult=cv,
                         cagr=t["standard"]["cagr"], flat_cagr=f["standard"]["cagr"],
                         tilt_cagr=d_c, sharpe=t["standard"]["sharpe"],
                         flat_sharpe=f["standard"]["sharpe"], tilt_sharpe=d_s))
        print(f"{t['variant']:30s} {t['mean_fill_mult']:5.3f} {t['standard']['cagr']:8.4f} "
              f"{f['standard']['cagr']:8.4f} {d_c:+8.4f} | {t['standard']['sharpe']:7.4f} "
              f"{f['standard']['sharpe']:7.4f} {d_s:+7.4f}")
    out["leverage_adjusted"] = ladj

    # ------------------------------------------------------------- WHY: where the profit lives
    print("\n=== WHY: what does a vol forecast size DOWN? (baseline trades, by entry-bar bucket) ===")
    why = trade_diagnostic(run_variant(panel, s, flags), panel, M)
    out["why"] = why

    # ------------------------------------------------------------- Route 2: the stop distance
    print("\n=== ROUTE 2: forecast-driven STOP DISTANCE ===")
    bd = (1 - panel["stop_level"] / panel["close"])[cand]
    fd = (1 - panel["stop_fcast"] / panel["close"])[cand & panel["pred_vol"].notna()]
    print(f"  stop distance at candidate bars: 20-bar low median {bd.median():.3%} "
          f"(p10 {bd.quantile(.1):.3%} p90 {bd.quantile(.9):.3%});  "
          f"forecast median {fd.median():.3%} (p10 {fd.quantile(.1):.3%} p90 {fd.quantile(.9):.3%})")
    print(f"  k calibrated on candidate bars, so the LEVEL is matched and only the SHAPE differs "
          f"(residual leverage factor {lev:.3f}x; 'const leverage' below is that factor alone)")
    print("  NOTE: with stop_exec OFF (the shipped config) stop_level only sets SIZE - the exit is")
    print("  the 20-bar-low CLOSE rule. stop_exec ON is where the forecast changes the exit too;")
    print("  that flag is not in the shipped baseline, so it must clear the same shipped bar.")
    stop_rows = []
    for fl_name, fl in (("stop_exec_off", flags), ("stop_exec_on", Flags(**SHIPPED, stop_exec=True))):
        ref = m_of(run_variant(panel, s, fl))
        ref_h = m_of(run_variant(panel, s, fl, start=HELD_OUT))
        stop_rows.append(dict(variant=f"{fl_name}: 20-bar low (reference)", standard=ref,
                              held_out=ref_h,
                              ships_vs_quoted=passes(ref, BASE["standard"]) and passes(ref_h, BASE["held_out"])))
        for sc, lbl in (("stop_fcast", "HAR sigma stop"), ("stop_v30", "vol_30d sigma stop"),
                        ("stop_const_lev", "const leverage, no forecast")):
            v = m_of(run_variant(panel, s, fl, stop_col=sc))
            vh = m_of(run_variant(panel, s, fl, stop_col=sc, start=HELD_OUT))
            stop_rows.append(dict(variant=f"{fl_name}: {lbl}", standard=v, held_out=vh,
                                  ships_vs_quoted=passes(v, BASE["standard"]) and passes(vh, BASE["held_out"]),
                                  ships_vs_same_flag_reference=passes(v, ref) and passes(vh, ref_h)))
    for r in stop_rows:
        v, h = r["standard"], r["held_out"]
        print(f"  {r['variant']:38s} {v['cagr']:8.4f} {v['sharpe']:7.4f} {v['maxdd']:8.4f} "
              f"{v['trades']:4d} | {h['cagr']:7.4f} {h['sharpe']:6.3f} {h['maxdd']:7.4f}  "
              f"{'SHIP' if r['ships_vs_quoted'] else '-'}")
    out["route2_stop"] = stop_rows

    # ------------------------------------------------------------- verdict and follow-ups
    out["variants"] = table
    shipped_ok = [t for t in table if t["ships_vs_quoted"]]
    print(f"\n=== SHIP RULE: {len(shipped_ok)} of {len(table)} size variants pass on BOTH panels ===")
    for t in shipped_ok:
        print(f"  PASS  {t['variant']}")

    # the best REAL (non-inverted, non-const) variant by standard CAGR, for the placebo and stress
    real = [t for t in table if not t["variant"].startswith("const_")
            and "INVERTED" not in t["variant"]]
    best = max(real, key=lambda t: t["standard"]["cagr"])
    print(f"\nbest real variant by CAGR: {best['variant']}  "
          f"({best['standard']['cagr']:.4f} vs baseline {b_std['cagr']:.4f})")

    # PLACEBO NULL on the best forecast-driven shape (whether or not it shipped)
    shapes = {
        "size_fcast_xs": lambda p: xs_mult(p, "pred_vol"),
        "size_fcast_abs": lambda p: abs_mult(p, "pred_vol", "train_med_pred"),
        "size_fcast_xs_meanneutral": lambda p: mean_neutral(xs_mult(p, "pred_vol"), p, cand),
        "gate_crash_fcast": lambda p: gate_mult(crash_prob(p["pred_vol"]) > p["train_q80_crash"]),
    }
    best_shape = max([t for t in real if t["variant"] in shapes],
                     key=lambda t: t["standard"]["cagr"])
    shape_name = best_shape["variant"]
    out["placebo"] = {}
    # the best-performing shape AND the cleanest construction (mean-neutral), so the null covers
    # both the number a reader would quote and the one that is actually defensible.
    for nm in dict.fromkeys([shape_name, "size_fcast_xs_meanneutral"]):
        print(f"\n=== PLACEBO NULL: 40 randomised forecasts through the shape '{nm}' ===")
        pl = placebo(panel, s, flags, shapes[nm], n=40)
        real_row = [t for t in table if t["variant"] == nm][0]
        pct = float((pl["std_cagr"] <= real_row["standard"]["cagr"]).mean())
        print(f"  placebos passing the ship rule: {int(pl['ships'].sum())}/40 "
              f"({pl['ships'].mean():.0%})")
        print(f"  placebo standard CAGR: mean {pl['std_cagr'].mean():.4f}  "
              f"sd {pl['std_cagr'].std():.4f}  min {pl['std_cagr'].min():.4f}  "
              f"max {pl['std_cagr'].max():.4f}")
        print(f"  the real '{nm}' CAGR {real_row['standard']['cagr']:.4f} sits at the "
              f"{pct:.0%} percentile of the placebo distribution")
        print(f"  placebo Sharpe: mean {pl['std_sharpe'].mean():.4f} "
              f"max {pl['std_sharpe'].max():.4f}  (real {real_row['standard']['sharpe']:.4f})")
        out["placebo"][nm] = {"n": 40, "n_ships": int(pl["ships"].sum()),
                              "real_percentile": pct, "rows": pl.to_dict("records"),
                              "summary": {k: float(pl[k].mean()) for k in
                                          ("std_cagr", "std_sharpe", "held_cagr", "held_sharpe")}}

    # ------------------------------------------------------------- is the null just dilution?
    print(f"\n=== FORECAST ERA ONLY (from {first_pred.date()}): is the null caused by the two "
          f"pre-forecast years? ===")
    era = m_of(run_variant(panel, s, flags, start=str(first_pred.date())))
    print(f"  {'baseline':30s} {era['cagr']:8.4f} {era['sharpe']:7.4f} {era['maxdd']:8.4f} "
          f"{era['trades']:4d}")
    era_rows = {"baseline": era}
    for nm in ("size_fcast_xs", "size_fcast_abs", "size_fcast_xs_meanneutral",
               "gate_regime_p50", "gate_crash_fcast", "size_v30_xs_matched"):
        v = m_of(run_variant(panel, s, flags, mult=M[nm], start=str(first_pred.date())))
        era_rows[nm] = v
        print(f"  {nm:30s} {v['cagr']:8.4f} {v['sharpe']:7.4f} {v['maxdd']:8.4f} {v['trades']:4d} "
              f"  {'beats baseline on all 3' if passes(v, era) else '-'}")
    out["forecast_era_only"] = {"start": str(first_pred.date()), "rows": era_rows}

    # SUB-PERIOD ATTRIBUTION for the best real variant and for the forecast's own shape
    print("\n=== SUB-PERIOD ATTRIBUTION (log-return difference vs baseline, by quarter) ===")
    base_eq = run_variant(panel, s, flags).equity
    attrib = {}
    for name in dict.fromkeys([best["variant"], shape_name]):
        mult = M[name]
        eq = run_variant(panel, s, flags, mult=mult).equity
        a = quarterly_attrib(eq, base_eq)
        attrib[name] = a
        print(f"  {name}: total log outperformance {a['total_log_outperf']:+.4f} over "
              f"{a['n_quarters']} quarters, ahead in {a['quarters_ahead']}")
        share = (f"best quarter alone carries {a['share_from_best_quarter']:+.0%} of it; "
                 if abs(a["total_log_outperf"]) > 0.05 else
                 "(total is ~0, so shares of it are meaningless) ")
        print("    " + share + "top 3 by size: "
              + ", ".join(f"{t['quarter']} {t['log_diff']:+.4f}" for t in a["top3"]))
        print("    yearly log diff: " + "  ".join(f"{k[:4]} {v:+.3f}" for k, v in a["yearly"].items()))
    out["attribution"] = attrib

    # COST STRESS on the best real variant (and on any variant that shipped)
    stress = []
    for name in dict.fromkeys([best["variant"], shape_name, "size_fcast_xs_meanneutral"]
                              + [t["variant"] for t in shipped_ok
                                 if not t["variant"].startswith("const_")]):
        cost_stress(panel, s, flags, M[name], None, name, stress)
    out["cost_stress"] = stress

    p = ROOT / "reports" / "forecast_integration.json"
    p.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
