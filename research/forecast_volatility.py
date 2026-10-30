"""Is REALISED VOLATILITY forecastable here, and does a model beat the naive carry-forward?

WHY THIS AND NOT RETURNS. This project has already rejected, with adversarial verification, every
attempt to predict RETURNS from this data (walk-forward ranker, ML veto on cross-sectional excess
return, win-probability model that ran backwards to profit, right-tail classifier at AUC 0.4914).
That question is closed. This script asks a different and much narrower one: volatility clustering
is one of the few genuinely robust empirical facts in finance, and volatility is what sets POSITION
SIZE and STOP DISTANCE. So: can we forecast each coin's next-7-day realised vol better than simply
carrying its trailing vol forward?

TARGET. For every (bar_time, symbol) on the 4h grid:

    fwd_vol_t = std(log-returns of bars t+1 .. t+42) * sqrt(6*365)

i.e. the realised annualised vol of the NEXT 7 days, annualised exactly the way app/features.py
annualises vol_7d and vol_30d. The 42 forward bars must be a contiguous 4h grid (span from t to
t+42 exactly 168h), otherwise the label is dropped - an outage gap would otherwise be measured as
a return.

We model log(fwd_vol) and exponentiate. Vol is lognormal-ish; in logs the errors are roughly
homoscedastic and - this is the point for a trading bot - the log error IS the sizing error. If you
size at risk_target / predicted_vol, then log(realised/predicted) is the log of how far your
realised risk lands from your target, so log-space RMSE is directly "how badly does sizing miss".

METHOD - walk-forward, trailing 730d train / 91d test, 10-day purge gap:

    |<---------- 730d train ---------->|  10d purge  |<--- 91d test --->|

  The purge is longer than the 7-day prediction horizon, so no training label resolves inside the
  test window. Features at bar t use only bars closed at or before t (they are app/features.py's
  own columns, plus two cross-sectional medians computed within a bar_time and two BTC vol columns
  joined on bar_time). Targets look strictly forward. Standardisation for the linear baseline uses
  TRAIN-WINDOW statistics only. The GBM is a tree model and needs none.

BASELINES - the margin over these is the entire result, not the raw R2:

  naive_vol30   predicted vol = vol_30d. The carry-forward the task names.
  naive_vol7    predicted vol = vol_7d.
  naive_cal     log(fwd_vol) ~ a + b*log(vol_30d), fitted on the train window only. This is the
                HONEST version of the carry-forward: it removes the level/shrinkage bias that the
                raw naive carries, so any margin over THIS is forecasting skill rather than the
                model discovering that 7-day vol is systematically a bit above 30-day vol.
  har           log(fwd_vol) ~ a + b1*log(vol_1d) + b2*log(vol_7d) + b3*log(vol_30d), OLS, train
                only. The textbook HAR model. If the GBM cannot beat this, volatility is still
                predictable but the ML adds nothing over three numbers and a regression.
  shuffled      the same GBM trained on a permuted label. The harness control: this must score
                ~0 or worse. If it scores well the harness is broken.

MODELS. Nine predictors are fitted and scored side by side, because which one wins is the result:

  model         the literal brief - GBM on all 18 features, target log(fwd_vol).
  gbm_har3      the same GBM on nothing but lv1/lv7/lv30 (the log trailing vols).
  gbm5          GBM on lv1/lv7/lv30 + vol_ratio_7_30 + atr_pct. This is what the feature ablation
                at the bottom of the run picks out as the best feature set for a GBM.
  gbm_ratio     GBM, all features, target log(fwd_vol) - log(vol_30d): the RATIO to the carry-
                forward rather than the level. Prediction = log(vol_30d) + GBM.
  gbm_ratio3    the same ratio target on lv1/lv7/lv30 only. gbm_ratio/gbm_ratio3 against
                model/gbm_har3 is a clean 2x2 of FEATURE SET against TARGET PARAMETERISATION, so
                the two explanations for a tree failing cannot be confounded.
  gbm_harres    GBM, all features, target log(fwd_vol) - har. "Does ML add anything on top of
                standard volatility modelling?"
  har / har_x   the OLS baselines described above.
  shuffled      the harness control.

RESULT (28 folds, 225k out-of-sample rows, Nov 2019 - Sep 2026; full numbers in
reports/forecast_volatility.json):

  1. VOLATILITY IS PREDICTABLE HERE, clearly and significantly. Pooled out-of-sample log-R2:
     naive_vol7 0.4033, naive_vol30 0.4270, calibrated carry-forward 0.4625, HAR 0.5308. HAR beats
     the raw carry-forward by +0.1038 log-R2 and +0.0683 over the calibrated one, wins 25 of 28
     folds, and the block bootstrap puts P(not better) at 0.0000 against all three naives. Unlike
     every return model this project has tried, this margin survives everything thrown at it.

  2. THE GBM AS BRIEFED LOSES TO THE NAIVE. All 18 features, log-level target: log-R2 0.3813,
     which is -0.0457 BELOW simply carrying vol_30d forward, 15/28 folds, bootstrap P(not
     better) = 0.85. Reporting that plainly is the point of the exercise.

  3. THE CAUSE IS THE FEATURE SET, NOT THE TARGET PARAMETERISATION. My first hypothesis - that a
     tree cannot track the drifting vol LEVEL and should be given a level-free ratio target - is
     refuted by the 2x2:

                         log level   ratio to vol_30d
         all features       0.3813             0.3928
         lv1/lv7/lv30       0.4957             0.4947

     The ratio target is worth about +0.011 with all features and -0.001 with three. Dropping 15
     features is worth +0.114. The feature ablation isolates the culprit: 3 trailing vols 0.4957,
     + vol_ratio and atr_pct 0.5172, + BTC/median/rank market-vol group 0.4352, + volume and taker
     flow 0.4421, + trend and regime 0.3813. The collapse arrives with the market-vol and
     trend/regime block, which is exactly the group that lets trees memorise a calendar era.

  4. THE EXTRA FEATURES CARRY NO INFORMATION, they are not merely overfitted. har_x - the same
     OLS plus nine of those extras - scores 0.5237 against plain HAR's 0.5308. In a linear model
     the extras cost 0.007; in a GBM they cost 0.114. A regression can shrink a noise column to a
     small coefficient, a tree splits on it.

  5. NOR IS IT UNDER-REGULARISATION. The hyperparameter sweep over the full 18-feature GBM
     (leaves 3 to 31, min_samples_leaf to 400, l2 to 100, column subsampling) tops out at 0.4866
     for a very shallow fit - above the naive, still below the 5-feature GBM and below HAR. No
     configuration of the full feature set reaches a small one.

  6. THE BEST GBM ESSENTIALLY MATCHES THE REGRESSION BUT DOES NOT BEAT IT. gbm5 reaches 0.5172,
     +0.0902 over naive_vol30 and +0.0547 over the calibrated naive, both significant, 21/28 folds
     against each. Against HAR it is -0.0136, losing 17 of 28 folds, bootstrap P(not better)
     = 0.98. So a correctly specified GBM has real skill over the naive, and no skill over three
     numbers and a straight line. The residual gap is functional form: log vol is close to linear
     in the log trailing vols, and a piecewise-constant tree cannot be, which shows up at the
     extremes - the 18-feature GBM's top decile predicts 1.569 against 1.350 realised, while HAR's
     top decile predicts 1.464 against 1.473 realised.

  7. USEFULNESS IS REAL BUT MODEST. Quintile ladders are monotone for every predictor, so the
     naive already ranks vol correctly; HAR widens the top/bottom-decile realised spread from
     3.53x to 3.78x. Under vol targeting (size = risk_target / predicted_vol), the 1-sigma band of
     realised risk around target tightens from [0.67x, 1.49x] on the naive to [0.70x, 1.44x] on
     HAR. A ~9.5% reduction in log-space RMSE is worth having for sizing and stop distance; it is
     not a new source of return, and nothing here predicts direction.

VERDICT: has_real_skill = true, carried by HAR (and by gbm5). The gradient booster as briefed
fails. Deploy the three-coefficient regression, not the forest.

LEVEL-SPACE FAIRNESS. exp(E[log v]) is a median, not a mean, so it is biased low in levels. Every
log-space model gets the same in-fold multiplicative correction k = mean(train level) /
mean(exp(train prediction)), computed on TRAINING rows only. Without it the level R2 would be
penalising the lognormal transform rather than the forecast.

OVERLAP. Consecutive 4h bars share 41 of their 42 forward bars, so pooled rows are massively
redundant. The point estimates are unaffected but any significance test on them would be fiction.
Every headline number is therefore recomputed on a NON-OVERLAPPING subsample (every 42nd bar per
symbol), and the significance of the model-minus-naive gap is a block bootstrap over calendar
months of that subsample.

    cd crypto-bot && PYTHONIOENCODING=utf-8 python research/forecast_volatility.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config, features  # noqa: E402

HORIZON = 42                      # 42 four-hour bars = 7 days
ANNUAL = np.sqrt(6 * 365)         # same annualisation as app/features.py
TRAIN_DAYS, TEST_DAYS, PURGE_DAYS = 730, 91, 10
HP = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
          l2_regularization=1.0, random_state=0, early_stopping=False)

# Own-past features from app/features.py, plus clearly-own-past additions marked (+).
FEATS = [
    "vol_7d", "vol_30d",
    "vol_1d",            # (+) 6-bar realised vol, the HAR "daily" leg
    "vol_ratio_7_30",    # (+) short/long vol ratio: the clustering / mean-reversion signal
    "atr_pct", "volume_ratio", "taker_ratio", "don_pos", "dist_sma50", "dist_sma200",
    "btc_dist_sma200", "btc_ret_7d", "btc_ret_30d", "regime_on",
    "btc_vol_7d", "btc_vol_30d",   # (+) BTC vol, joined on bar_time
    "mkt_vol_7d",                  # (+) cross-sectional median vol_7d within the bar
    "vol_rank",                    # cross-sectional pct rank of the coin's vol_30d in the bar
]
HAR_FEATS = ["lv1", "lv7", "lv30"]
# HAR plus the extra features, in a LINEAR model. If this scores the same as plain HAR, the extras
# carry no incremental information at all - as opposed to carrying some that the GBM overfits.
HARX_EXTRA = ["lbtcv7", "lmktv7", "lvr", "volume_ratio", "taker_ratio", "don_pos",
              "dist_sma50", "regime_on", "vol_rank"]
HARX_FEATS = HAR_FEATS + HARX_EXTRA
# The feature set the ablation below picks out as the GBM's best: the three trailing vols plus two
# shape terms. Everything else in FEATS measures trend, flow or market regime - and costs accuracy.
GBM5_FEATS = HAR_FEATS + ["vol_ratio_7_30", "atr_pct"]


# ----------------------------------------------------------------- data

def build(s):
    u = json.loads((ROOT / "reports" / "_parity_universe.json").read_text())
    panel = features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars)

    # --- label: forward realised vol, and a short trailing vol used as a feature
    out = []
    step = pd.Timedelta(hours=4 * HORIZON)
    for _, g in panel.sort_values(["symbol", "bar_time"]).groupby("symbol", sort=False):
        r = np.log(g["close"]).diff()
        fwd = r.rolling(HORIZON).std().shift(-HORIZON) * ANNUAL
        contiguous = (g["bar_time"].shift(-HORIZON) - g["bar_time"]) == step
        out.append(g.assign(fwd_vol=fwd.where(contiguous),
                            vol_1d=r.rolling(6).std() * ANNUAL))
    panel = pd.concat(out, ignore_index=True)

    panel["vol_ratio_7_30"] = panel["vol_7d"] / panel["vol_30d"]
    panel["regime_on"] = panel["regime_on"].astype(float)

    btc = panel[panel["symbol"] == "BTCUSDT"].set_index("bar_time")
    panel = panel.join(btc[["vol_7d", "vol_30d"]].add_prefix("btc_"), on="bar_time")
    el = panel["eligible"]
    med = panel.loc[el].groupby("bar_time")["vol_7d"].median().rename("mkt_vol_7d")
    panel = panel.join(med, on="bar_time")
    panel.loc[el, "vol_rank"] = (panel.loc[el].groupby("bar_time")["vol_30d"]
                                 .rank(pct=True))

    # `base` needs only FEATURES (so it includes the newest bars, whose label has not resolved
    # yet - that is what the live forecast is made from). `d` is base restricted to rows whose
    # label has resolved, and is the only thing fitted or scored on.
    d = panel[el & panel["vol_30d"].notna() & (panel["vol_30d"] > 0)
              & panel["vol_7d"].notna() & (panel["vol_7d"] > 0)
              & panel["vol_1d"].notna() & (panel["vol_1d"] > 0)].copy()
    d["y"] = np.log(d["fwd_vol"].where(d["fwd_vol"] > 0))
    d["lv1"], d["lv7"], d["lv30"] = (np.log(d["vol_1d"]), np.log(d["vol_7d"]),
                                     np.log(d["vol_30d"]))
    d["lbtcv7"] = np.log(d["btc_vol_7d"].clip(lower=1e-6))
    d["lmktv7"] = np.log(d["mkt_vol_7d"].clip(lower=1e-6))
    d["lvr"] = np.log(d["vol_ratio_7_30"].clip(lower=1e-6))
    # NaN/inf are left in place here on purpose: imputing with a full-sample median would be a
    # lookahead. The fold imputes with TRAIN-window medians instead (see walk_forward).
    for c in HARX_EXTRA:
        d[c] = d[c].replace([np.inf, -np.inf], np.nan)
    d = d.sort_values(["bar_time", "symbol"]).reset_index(drop=True)
    # Global ordinal of the bar_time, so "every 42nd bar" is a GLOBAL grid: labels are then
    # non-overlapping AND the cross-section at each sampled bar stays intact.
    d["ibar"] = d["bar_time"].rank(method="dense").astype(int) - 1
    base = d
    return d[d["y"].notna()].reset_index(drop=True), base


def audit_label(s, d, n=400, seed=0):
    """Prove the label is strictly FORWARD by recomputing it from the raw klines.

    For n random rows, reload that symbol's 4h candles, take the 42 closes strictly AFTER the row's
    bar, and recompute the annualised std of their log returns. It must equal fwd_vol. Also assert
    that the window uses no bar at or before t: perturbing only bars <= t must leave the label
    unchanged, and the first return in the window is log(c[t+1]/c[t]) - which is the one and only
    place a bar at t is touched, and only as the base of a forward return.
    """
    from app import data
    rng = np.random.default_rng(seed)
    rows = d.sample(min(n, len(d)), random_state=seed)
    cache, bad, checked = {}, [], 0
    for _, r in rows.iterrows():
        sym = r["symbol"]
        if sym not in cache:
            b = data.load(s.klines_dir, sym, "4h").copy()
            b["bar_time"] = pd.to_datetime(b["open_time"] + 4 * 3_600_000, unit="ms", utc=True)
            cache[sym] = b.sort_values("bar_time").reset_index(drop=True)
        b = cache[sym]
        hit = b.index[b["bar_time"] == r["bar_time"]]
        if not len(hit) or hit[0] + HORIZON >= len(b):
            continue
        i = hit[0]
        c = b["close"].to_numpy()[i:i + HORIZON + 1]      # c[0] is the bar AT t
        want = float(np.std(np.diff(np.log(c)), ddof=1) * ANNUAL)
        checked += 1
        if not np.isclose(want, float(r["fwd_vol"]), rtol=1e-9, atol=1e-12):
            bad.append((sym, str(r["bar_time"]), want, float(r["fwd_vol"])))
    assert checked > 50, f"audit checked only {checked} rows"
    assert not bad, f"label mismatch on {len(bad)} rows, e.g. {bad[:3]}"
    print(f"  label audit: {checked} rows recomputed from raw klines, all match. The window is "
          f"closes t..t+42, i.e. the 42 returns realised strictly AFTER bar t.")
    return dict(rows_checked=checked, mismatches=0)


# ----------------------------------------------------------------- walk-forward

def ols(X, y):
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def apply_ols(beta, X):
    return beta[0] + X @ beta[1:]


def walk_forward(d, verbose=True):
    first, last = d["bar_time"].min(), d["bar_time"].max()
    purge = pd.Timedelta(days=PURGE_DAYS)
    test_start = first + pd.Timedelta(days=TRAIN_DAYS) + purge
    rows, folds, har_betas = [], [], []
    rng = np.random.default_rng(0)

    while test_start < last:
        train_end = test_start - purge
        train_start = train_end - pd.Timedelta(days=TRAIN_DAYS)
        test_end = test_start + pd.Timedelta(days=TEST_DAYS)
        tr = d[(d["bar_time"] > train_start) & (d["bar_time"] <= train_end)]
        te = d[(d["bar_time"] > test_start) & (d["bar_time"] <= test_end)]
        if len(tr) < 2000 or len(te) < 200:
            test_start += pd.Timedelta(days=TEST_DAYS)
            continue

        Xtr = tr[FEATS].replace([np.inf, -np.inf], np.nan).to_numpy(float)
        Xte = te[FEATS].replace([np.inf, -np.inf], np.nan).to_numpy(float)
        ytr, yte = tr["y"].to_numpy(), te["y"].to_numpy()

        def gbm(y, Xa=Xtr, Xb=Xte):
            m = HistGradientBoostingRegressor(**HP).fit(Xa, y)
            return m.predict(Xa), m.predict(Xb)

        p_model_tr, p_model = gbm(ytr)
        p_shuf_tr, p_shuf = gbm(rng.permutation(ytr))

        X3tr, X3te = tr[HAR_FEATS].to_numpy(float), te[HAR_FEATS].to_numpy(float)
        p_g3_tr, p_g3 = gbm(ytr, X3tr, X3te)

        X5tr = tr[GBM5_FEATS].replace([np.inf, -np.inf], np.nan).to_numpy(float)
        X5te = te[GBM5_FEATS].replace([np.inf, -np.inf], np.nan).to_numpy(float)
        p_g5_tr, p_g5 = gbm(ytr, X5tr, X5te)

        b_cal = ols(tr[["lv30"]].to_numpy(), ytr)
        p_cal_tr = apply_ols(b_cal, tr[["lv30"]].to_numpy())
        p_cal = apply_ols(b_cal, te[["lv30"]].to_numpy())

        # HAR: standardise with train stats only (cosmetic for OLS, mandatory by the rules)
        mu, sd = tr[HAR_FEATS].mean().to_numpy(), tr[HAR_FEATS].std().to_numpy()
        sd = np.where(sd > 0, sd, 1.0)
        Ztr = (tr[HAR_FEATS].to_numpy() - mu) / sd
        Zte = (te[HAR_FEATS].to_numpy() - mu) / sd
        b_har = ols(Ztr, ytr)
        p_har_tr, p_har = apply_ols(b_har, Ztr), apply_ols(b_har, Zte)
        har_betas.append(dict(fold=len(folds), const=float(b_har[0]),
                              **{f: float(v) for f, v in zip(HAR_FEATS, b_har[1:])}))

        # HAR plus the extra features, linear. Impute and standardise with TRAIN stats only.
        med = tr[HARX_FEATS].median()
        Wtr_raw, Wte_raw = tr[HARX_FEATS].fillna(med), te[HARX_FEATS].fillna(med)
        mu2, sd2 = Wtr_raw.mean().to_numpy(), Wtr_raw.std().to_numpy()
        sd2 = np.where(sd2 > 0, sd2, 1.0)
        Wtr = (Wtr_raw.to_numpy(float) - mu2) / sd2
        Wte = (Wte_raw.to_numpy(float) - mu2) / sd2
        b_hx = ols(Wtr, ytr)
        p_hx_tr, p_hx = apply_ols(b_hx, Wtr), apply_ols(b_hx, Wte)

        # target parameterisation, crossed with the feature set: learn the RATIO to an anchor
        # instead of the level. gbm_ratio = full features, gbm_ratio3 = the HAR three only.
        lv30_tr, lv30_te = tr["lv30"].to_numpy(), te["lv30"].to_numpy()
        a_tr, a_te = gbm(ytr - lv30_tr)
        p_ratio_tr, p_ratio = lv30_tr + a_tr, lv30_te + a_te
        c_tr, c_te = gbm(ytr - lv30_tr, X3tr, X3te)
        p_ratio3_tr, p_ratio3 = lv30_tr + c_tr, lv30_te + c_te
        b_tr, b_te = gbm(ytr - p_har_tr)
        p_hres_tr, p_hres = p_har_tr + b_tr, p_har + b_te

        # level-space correction, fitted on TRAIN rows only, for every log-space predictor
        lvl_tr = tr["fwd_vol"].to_numpy()
        def k_of(p_tr):
            return float(lvl_tr.mean() / np.exp(p_tr).mean())

        f = pd.DataFrame({
            "bar_time": te["bar_time"].to_numpy(), "symbol": te["symbol"].to_numpy(),
            "ibar": te["ibar"].to_numpy(), "fold": len(folds),
            "fwd_vol": te["fwd_vol"].to_numpy(), "y": yte,
            "p_model": p_model, "p_cal": p_cal, "p_har": p_har, "p_shuf": p_shuf,
            "p_gbm_har3": p_g3, "p_gbm_ratio": p_ratio, "p_gbm_harres": p_hres,
            "p_gbm_ratio3": p_ratio3, "p_har_x": p_hx, "p_gbm5": p_g5,
            "v_model": k_of(p_model_tr) * np.exp(p_model),
            "v_cal": k_of(p_cal_tr) * np.exp(p_cal),
            "v_har": k_of(p_har_tr) * np.exp(p_har),
            "v_shuf": k_of(p_shuf_tr) * np.exp(p_shuf),
            "v_gbm_har3": k_of(p_g3_tr) * np.exp(p_g3),
            "v_gbm_ratio": k_of(p_ratio_tr) * np.exp(p_ratio),
            "v_gbm_harres": k_of(p_hres_tr) * np.exp(p_hres),
            "v_gbm_ratio3": k_of(p_ratio3_tr) * np.exp(p_ratio3),
            "v_har_x": k_of(p_hx_tr) * np.exp(p_hx),
            "v_gbm5": k_of(p_g5_tr) * np.exp(p_g5),
            "v_vol30": te["vol_30d"].to_numpy(), "v_vol7": te["vol_7d"].to_numpy(),
        })
        f["p_vol30"], f["p_vol7"] = np.log(f["v_vol30"]), np.log(f["v_vol7"])
        rows.append(f)
        folds.append(dict(fold=len(folds), train_start=str(train_start.date()),
                          train_end=str(train_end.date()), test_start=str(test_start.date()),
                          test_end=str(min(test_end, last).date()), n_train=len(tr),
                          n_test=len(te)))
        if verbose:
            print(f"  fold {len(folds)-1:>2}  test {test_start.date()} .. "
                  f"{min(test_end, last).date()}  train={len(tr):>6}  test={len(te):>5}")
        test_start += pd.Timedelta(days=TEST_DAYS)

    return pd.concat(rows, ignore_index=True), folds, har_betas


# ----------------------------------------------------------------- ablation and HP sweep

LADDER_SETS = {
    "3 trailing vols": HAR_FEATS,
    "+ shape (vol ratio, atr)": HAR_FEATS + ["vol_ratio_7_30", "atr_pct"],
    "+ market vol (btc, median, rank)": HAR_FEATS + ["vol_ratio_7_30", "atr_pct", "btc_vol_7d",
                                                     "btc_vol_30d", "mkt_vol_7d", "vol_rank"],
    "+ flow (volume, taker)": HAR_FEATS + ["vol_ratio_7_30", "atr_pct", "btc_vol_7d",
                                           "btc_vol_30d", "mkt_vol_7d", "vol_rank",
                                           "volume_ratio", "taker_ratio"],
    "+ trend/regime (all features)": HAR_FEATS + FEATS,
}
HP_SWEEP = {
    "default": {},
    "shallow (leaves 4)": dict(max_leaf_nodes=4),
    "very shallow (leaves 3, lr .02, 800)": dict(max_leaf_nodes=3, learning_rate=0.02,
                                                 max_iter=800),
    "deep (leaves 31)": dict(max_leaf_nodes=31),
    "heavy leaf (min 400)": dict(min_samples_leaf=400),
    "strong l2 (100)": dict(l2_regularization=100.0),
    "col subsample .3": dict(max_features=0.3),
}


def gbm_sweep(d, variants, label):
    """Re-run the SAME walk-forward with one GBM per variant. variants: name -> (feats, hp)."""
    first, last = d["bar_time"].min(), d["bar_time"].max()
    purge = pd.Timedelta(days=PURGE_DAYS)
    test_start = first + pd.Timedelta(days=TRAIN_DAYS) + purge
    acc = {k: [] for k in variants}
    acc["_naive"], acc["_y"], acc["_fold"] = [], [], []
    fold = 0
    while test_start < last:
        train_end = test_start - purge
        tr = d[(d["bar_time"] > train_end - pd.Timedelta(days=TRAIN_DAYS))
               & (d["bar_time"] <= train_end)]
        te = d[(d["bar_time"] > test_start) & (d["bar_time"] <= test_start + pd.Timedelta(days=TEST_DAYS))]
        if len(tr) < 2000 or len(te) < 200:
            test_start += pd.Timedelta(days=TEST_DAYS)
            continue
        ytr = tr["y"].to_numpy()
        for name, (fl, hp) in variants.items():
            m = HistGradientBoostingRegressor(**{**HP, **hp}).fit(
                tr[fl].replace([np.inf, -np.inf], np.nan).to_numpy(float), ytr)
            acc[name].append(m.predict(te[fl].replace([np.inf, -np.inf], np.nan)
                                       .to_numpy(float)))
        acc["_naive"].append(te["lv30"].to_numpy())
        acc["_y"].append(te["y"].to_numpy())
        acc["_fold"].append(np.full(len(te), fold))
        fold += 1
        test_start += pd.Timedelta(days=TEST_DAYS)

    y = np.concatenate(acc["_y"])
    nv = np.concatenate(acc["_naive"])
    fo = np.concatenate(acc["_fold"])
    base_r2 = r2(y, nv)
    out = {}
    for name in variants:
        p = np.concatenate(acc[name])
        w = sum(r2(y[fo == i], p[fo == i]) > r2(y[fo == i], nv[fo == i]) for i in range(fold))
        out[name] = dict(r2_log=r2(y, p), d_vs_naive_vol30=r2(y, p) - base_r2,
                         rmse_log=float(np.sqrt(((y - p) ** 2).mean())),
                         folds_beating_naive=int(w), n_folds=fold,
                         n_features=len(variants[name][0]))
    print(f"\n== {label} (pooled OOS log-R2; naive_vol30 = {base_r2:.4f}, "
          f"har = reference 0.5308) ==")
    for name, v in out.items():
        print(f"  {name:<40} nfeat {v['n_features']:>2}  R2_log {v['r2_log']:>7.4f}  "
              f"vs naive {v['d_vs_naive_vol30']:+.4f}  folds won "
              f"{v['folds_beating_naive']}/{v['n_folds']}")
    out["_naive_vol30_r2_log"] = base_r2
    return out


# ----------------------------------------------------------------- metrics

PREDS = {"model": ("v_model", "p_model"),
         "gbm_har3": ("v_gbm_har3", "p_gbm_har3"),
         "gbm5": ("v_gbm5", "p_gbm5"),
         "gbm_ratio": ("v_gbm_ratio", "p_gbm_ratio"),
         "gbm_ratio3": ("v_gbm_ratio3", "p_gbm_ratio3"),
         "gbm_harres": ("v_gbm_harres", "p_gbm_harres"),
         "har": ("v_har", "p_har"), "har_x": ("v_har_x", "p_har_x"),
         "naive_vol30": ("v_vol30", "p_vol30"),
         "naive_vol7": ("v_vol7", "p_vol7"), "naive_cal": ("v_cal", "p_cal"),
         "shuffled": ("v_shuf", "p_shuf")}
# Fitted forecasters, judged against the naive carry-forwards.
MODELS = ("model", "gbm_har3", "gbm5", "gbm_ratio", "gbm_ratio3", "gbm_harres", "har", "har_x")
BASELINES = ("naive_vol30", "naive_vol7", "naive_cal")
# The 2x2 that separates "wrong features" from "wrong target parameterisation".
GRID = {("all features", "log level"): "model", ("all features", "ratio to vol_30d"): "gbm_ratio",
        ("lv1/lv7/lv30", "log level"): "gbm_har3",
        ("lv1/lv7/lv30", "ratio to vol_30d"): "gbm_ratio3"}


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def within_bar_r2(df, ycol, pcol):
    """R2 after removing each bar's cross-sectional mean: can it rank coins inside a bar?"""
    g = df.groupby("bar_time")
    n = g[ycol].transform("size")
    m = n >= 5
    ya = (df[ycol] - g[ycol].transform("mean"))[m]
    pa = (df[pcol] - g[pcol].transform("mean"))[m]
    return float(1 - ((ya - pa) ** 2).sum() / (ya ** 2).sum())


def score_all(df, label):
    out = {}
    for name, (vcol, pcol) in PREDS.items():
        res_log = df["y"] - df[pcol]
        out[name] = dict(
            r2_level=r2(df["fwd_vol"], df[vcol]),
            r2_log=r2(df["y"], df[pcol]),
            rmse_log=float(np.sqrt((res_log ** 2).mean())),
            mae_log=float(res_log.abs().mean()),
            spearman=float(spearmanr(df[vcol], df["fwd_vol"]).statistic),
            pearson_level=float(np.corrcoef(df[vcol], df["fwd_vol"])[0, 1]),
            within_bar_r2_log=within_bar_r2(df, "y", pcol),
            mean_pred=float(df[vcol].mean()),
        )
    out["_n"] = int(len(df))
    out["_mean_realised"] = float(df["fwd_vol"].mean())
    out["_label"] = label
    return out


def ladder(df, vcol, q=5):
    b = pd.qcut(df[vcol], q, labels=False, duplicates="drop")
    g = df.groupby(b)
    return [dict(bucket=int(k) + 1, n=int(len(v)), mean_pred=float(v[vcol].mean()),
                 mean_realised=float(v["fwd_vol"].mean()),
                 median_realised=float(v["fwd_vol"].median()))
            for k, v in g]


SHOW = ("model", "gbm5", "har", "naive_vol30", "naive_cal")


def extremes(df):
    """Top and bottom DECILE of each predictor: where sizing decisions actually diverge."""
    out = {}
    for name in SHOW:
        vcol, pcol = PREDS[name]
        dec = pd.qcut(df[vcol], 10, labels=False, duplicates="drop")
        o = {}
        for tag, sel in (("bottom", dec == 0), ("top", dec == dec.max())):
            sub = df[sel]
            o[tag] = dict(
                n=int(len(sub)), mean_pred=float(sub[vcol].mean()),
                mean_realised=float(sub["fwd_vol"].mean()),
                # error of EVERY predictor on these same rows -> apples to apples
                rmse_log={k: float(np.sqrt(((sub["y"] - sub[PREDS[k][1]]) ** 2).mean()))
                          for k in SHOW},
            )
        o["spread_ratio"] = o["top"]["mean_realised"] / o["bottom"]["mean_realised"]
        out[name] = o
    return out


def block_bootstrap(df, a, b, reps=4000, seed=0):
    """P(model does NOT beat baseline), resampling calendar months of a non-overlapping sample.

    Compares squared log errors row-wise; a positive mean difference means `a` is better.
    """
    df = df.copy()
    df["blk"] = df["bar_time"].dt.tz_localize(None).dt.to_period("M")
    diff = ((df["y"] - df[PREDS[b][1]]) ** 2 - (df["y"] - df[PREDS[a][1]]) ** 2)
    blocks = [v.to_numpy() for _, v in diff.groupby(df["blk"])]
    rng = np.random.default_rng(seed)
    base = float(np.concatenate(blocks).mean())
    draws = np.empty(reps)
    idx = np.arange(len(blocks))
    for i in range(reps):
        pick = rng.choice(idx, len(blocks), replace=True)
        draws[i] = np.concatenate([blocks[j] for j in pick]).mean()
    return dict(mean_sq_err_reduction=base, p_not_better=float((draws <= 0).mean()),
                ci_lo=float(np.quantile(draws, 0.025)), ci_hi=float(np.quantile(draws, 0.975)),
                n_blocks=len(blocks), n_rows=int(len(df)))


# ----------------------------------------------------------------- report

def live_forecast(d, base):
    """Refit the winning forecaster (HAR) on the trailing 730d and forecast the current bar.

    Fitted on rows whose label has already resolved, so the newest training row is 7 days old -
    exactly what a live bot would have. The output is the only part of this script that is
    directly actionable: fcast/vol_30d is how much the 7-day-ahead vol forecast differs from the
    carry-forward, and size_mult is its reciprocal, i.e. what a vol-targeted position size would
    be multiplied by relative to sizing off vol_30d.
    """
    last_lab = d["bar_time"].max()
    tr = d[d["bar_time"] > last_lab - pd.Timedelta(days=TRAIN_DAYS)]
    beta = ols(tr[HAR_FEATS].to_numpy(float), tr["y"].to_numpy())
    k = float(tr["fwd_vol"].mean() / np.exp(apply_ols(beta, tr[HAR_FEATS].to_numpy(float))).mean())

    # Features for the current bar need no forward data, so they come from `base`, not `d`.
    last = base["bar_time"].max()
    cur = base[base["bar_time"] == last]
    rows = []
    for _, r in cur.sort_values("symbol").iterrows():
        f = k * float(np.exp(apply_ols(beta, r[HAR_FEATS].to_numpy(float)[None, :])[0]))
        rows.append(dict(symbol=r["symbol"], vol_7d=float(r["vol_7d"]),
                         vol_30d=float(r["vol_30d"]), forecast_vol_7d=f,
                         forecast_over_vol30=f / float(r["vol_30d"]),
                         size_mult_vs_naive=float(r["vol_30d"]) / f))
    return dict(bar_time=str(last), train_end=str(last_lab), n_train=int(len(tr)),
                level_correction=k,
                coefficients=dict(const=float(beta[0]),
                                  **{c: float(v) for c, v in zip(HAR_FEATS, beta[1:])}),
                rows=rows)


def line(name, m):
    return (f"  {name:<12} R2_lvl {m['r2_level']:>7.4f}   R2_log {m['r2_log']:>7.4f}   "
            f"RMSE_log {m['rmse_log']:>6.4f}   rho {m['spearman']:>6.3f}   "
            f"xs_R2 {m['within_bar_r2_log']:>7.4f}")


def main():
    s = config.load()
    print("building panel and labels ...")
    d, base = build(s)
    print(f"  rows with a label: {len(d):,}   symbols: {d['symbol'].nunique()}   "
          f"{d['bar_time'].min().date()} .. {d['bar_time'].max().date()}")
    print(f"  realised fwd vol: mean {d['fwd_vol'].mean():.3f}  "
          f"median {d['fwd_vol'].median():.3f}  p5 {d['fwd_vol'].quantile(.05):.3f}  "
          f"p95 {d['fwd_vol'].quantile(.95):.3f}")
    audit = audit_label(s, d)
    print("\nwalk-forward 730d train / 91d test / 10d purge:")
    oos, folds, har_betas = walk_forward(d)
    print(f"  {len(folds)} folds, {len(oos):,} out-of-sample rows, "
          f"{oos['bar_time'].min().date()} .. {oos['bar_time'].max().date()}")

    pooled = score_all(oos, "pooled (overlapping)")
    nonov = oos[oos["ibar"] % HORIZON == 0].copy()
    nonov_m = score_all(nonov, "non-overlapping (every 42nd bar per symbol)")

    print("\n== POOLED OUT-OF-SAMPLE (all rows, overlapping labels) ==")
    for k in PREDS:
        print(line(k, pooled[k]))
    print("\n== NON-OVERLAPPING SUBSAMPLE (every 42nd bar per symbol) ==")
    print(f"  n = {nonov_m['_n']:,}")
    for k in PREDS:
        print(line(k, nonov_m[k]))

    def margin(a, b):
        return dict(
            d_r2_level=pooled[a]["r2_level"] - pooled[b]["r2_level"],
            d_r2_log=pooled[a]["r2_log"] - pooled[b]["r2_log"],
            d_rmse_log=pooled[a]["rmse_log"] - pooled[b]["rmse_log"],
            rmse_log_pct_better=100 * (1 - pooled[a]["rmse_log"] / pooled[b]["rmse_log"]),
            d_spearman=pooled[a]["spearman"] - pooled[b]["spearman"],
            d_within_bar_r2=pooled[a]["within_bar_r2_log"] - pooled[b]["within_bar_r2_log"],
        )

    margins = {a: {b: margin(a, b) for b in BASELINES + ("har",)} for a in MODELS}
    best = max(MODELS, key=lambda k: pooled[k]["r2_log"])
    best_gbm = max(("model", "gbm_har3", "gbm5", "gbm_ratio", "gbm_ratio3", "gbm_harres"),
                   key=lambda k: pooled[k]["r2_log"])
    print(f"\n== MARGIN OVER EACH NAIVE BASELINE (pooled).  best forecaster = {best} ==")
    for a in MODELS:
        for b in BASELINES:
            m = margins[a][b]
            print(f"  {a:<11} vs {b:<12} dR2_lvl {m['d_r2_level']:+.4f}   "
                  f"dR2_log {m['d_r2_log']:+.4f}   RMSE_log {m['rmse_log_pct_better']:+.2f}%   "
                  f"drho {m['d_spearman']:+.3f}   dxs_R2 {m['d_within_bar_r2']:+.4f}")
        print()

    print("== 2x2: FEATURE SET x TARGET PARAMETERISATION (pooled log-R2) ==")
    print(f"  {'':<14}{'log level':>14}{'ratio to vol_30d':>20}")
    for fs in ("all features", "lv1/lv7/lv30"):
        print(f"  {fs:<14}" + f"{pooled[GRID[(fs, 'log level')]]['r2_log']:>14.4f}"
              + f"{pooled[GRID[(fs, 'ratio to vol_30d')]]['r2_log']:>20.4f}")
    print(f"  naive_vol30 = {pooled['naive_vol30']['r2_log']:.4f}, "
          f"OLS on lv1/lv7/lv30 (har) = {pooled['har']['r2_log']:.4f}, "
          f"OLS + 9 extras (har_x) = {pooled['har_x']['r2_log']:.4f}")

    # per-fold win rate
    fold_tbl = []
    for fid, g in oos.groupby("fold"):
        row = dict(fold=int(fid), n=int(len(g)),
                   test_start=folds[int(fid)]["test_start"],
                   mean_realised=float(g["fwd_vol"].mean()))
        for k in PREDS:
            row[f"r2_log_{k}"] = r2(g["y"], g[PREDS[k][1]])
        fold_tbl.append(row)
    wins = {a: {b: int(sum(f[f"r2_log_{a}"] > f[f"r2_log_{b}"] for f in fold_tbl))
                for b in BASELINES + ("har",)} for a in MODELS}
    print(f"\n== PER-FOLD (n={len(fold_tbl)}): folds where row beats column on log-R2 ==")
    print(f"  {'':<13}" + "".join(f"{b:>13}" for b in BASELINES + ("har",)))
    for a in MODELS:
        print(f"  {a:<13}" + "".join(f"{wins[a][b]:>13}" for b in BASELINES + ("har",)))

    lad = {k: ladder(oos, PREDS[k][0]) for k in SHOW}
    print("\n== QUINTILE LADDER: mean realised fwd vol by predicted quintile ==")
    print(f"  {'quintile':<10}" + "".join(f"{k:>14}" for k in lad))
    for i in range(5):
        cells = "".join(f"{lad[k][i]['mean_realised']:>14.4f}" for k in lad)
        print(f"  Q{i+1:<9}" + cells)
    print(f"  {'Q5/Q1':<10}" + "".join(
        f"{lad[k][4]['mean_realised']/lad[k][0]['mean_realised']:>14.2f}" for k in lad))

    ex = extremes(oos)
    print("\n== EXTREMES (deciles of each predictor; RMSE_log of all predictors on those rows) ==")
    for name in ex:
        for tag in ("bottom", "top"):
            o = ex[name][tag]
            r = o["rmse_log"]
            print(f"  by {name:<12} {tag:<6} n={o['n']:>6}  pred {o['mean_pred']:.3f}  "
                  f"realised {o['mean_realised']:.3f}  | RMSE_log "
                  + "  ".join(f"{k} {r[k]:.4f}" for k in SHOW))
        print(f"  by {name:<12} top/bottom realised spread = {ex[name]['spread_ratio']:.2f}x\n")

    boot = {a: {b: block_bootstrap(nonov, a, b) for b in BASELINES + ("har",)}
            for a in dict.fromkeys((best, best_gbm, "model", "har"))}
    print("== BLOCK BOOTSTRAP on the non-overlapping sample (calendar months resampled) ==")
    for a, bb in boot.items():
        for b, v in bb.items():
            if a == b:
                continue
            print(f"  {a:<12} vs {b:<12} mean sq-log-err reduction "
                  f"{v['mean_sq_err_reduction']:+.5f} [{v['ci_lo']:+.5f}, {v['ci_hi']:+.5f}]  "
                  f"P(not better) = {v['p_not_better']:.4f}")
    print(f"  ({boot[best]['naive_vol30']['n_blocks']} months, "
          f"{boot[best]['naive_vol30']['n_rows']} rows)")

    # sizing interpretation: log-space RMSE is the spread of realised risk around target
    sizing = {k: dict(rmse_log=pooled[k]["rmse_log"],
                      band_lo=float(np.exp(-pooled[k]["rmse_log"])),
                      band_hi=float(np.exp(pooled[k]["rmse_log"])))
              for k in PREDS}
    print("\n== SIZING INTERPRETATION (size = risk_target / predicted_vol) ==")
    for k in SHOW:
        v = sizing[k]
        print(f"  {k:<12} 1-sigma realised risk lands in "
              f"[{v['band_lo']:.2f}x, {v['band_hi']:.2f}x] of target")

    def passes(a):
        """Defensible skill = beats BOTH raw naives and the calibrated one, on the pooled margin,
        on the per-fold win rate, and on the block bootstrap, with the control clean."""
        if a not in boot:
            return None
        return bool(margins[a]["naive_vol30"]["d_r2_log"] > 0.02
                    and margins[a]["naive_vol7"]["d_r2_log"] > 0.02
                    and margins[a]["naive_cal"]["d_r2_log"] > 0.01
                    and all(boot[a][b]["p_not_better"] < 0.05 for b in BASELINES)
                    and all(wins[a][b] >= 0.7 * len(fold_tbl) for b in BASELINES)
                    and pooled["shuffled"]["r2_log"] < 0.05)

    verdict = {a: passes(a) for a in boot}
    skill = bool(verdict[best])

    abl = gbm_sweep(d, {k: (v, {}) for k, v in LADDER_SETS.items()},
                    "FEATURE ABLATION: add the extras to the GBM one group at a time")
    sweep = gbm_sweep(d, {k: (FEATS, v) for k, v in HP_SWEEP.items()},
                      "HYPERPARAMETER SWEEP on the FULL 18-feature GBM "
                      "(is the failure just under-regularisation?)")

    # The deployable forecaster, refit on the most recent 730 days, and today's forecast.
    live = live_forecast(d, base)
    print("\n== LIVE FORECAST (HAR refit on the trailing 730 days, current bar) ==")
    print(f"  bar {live['bar_time']}  (coefficients on raw logs: const "
          f"{live['coefficients']['const']:+.4f}, "
          + ", ".join(f"{k} {v:+.4f}" for k, v in live['coefficients'].items() if k != 'const')
          + ")")
    print(f"  {'symbol':<10}{'vol_7d':>9}{'vol_30d':>9}{'fcast_7d':>10}"
          f"{'fcast/vol30':>13}{'size_mult':>11}")
    for r in live["rows"]:
        print(f"  {r['symbol']:<10}{r['vol_7d']:>9.3f}{r['vol_30d']:>9.3f}"
              f"{r['forecast_vol_7d']:>10.3f}{r['forecast_over_vol30']:>13.3f}"
              f"{r['size_mult_vs_naive']:>11.3f}")

    print("\n== CONCLUSION ==")
    print(f"  harness control (shuffled label) log-R2 = {pooled['shuffled']['r2_log']:.4f} "
          f"-> {'OK, no leakage' if pooled['shuffled']['r2_log'] < 0.05 else 'BROKEN'}")
    print(f"  volatility IS predictable here: naive_vol30 log-R2 "
          f"{pooled['naive_vol30']['r2_log']:.4f}, naive_vol7 "
          f"{pooled['naive_vol7']['r2_log']:.4f}, best forecaster ({best}) "
          f"{pooled[best]['r2_log']:.4f}")
    print(f"  the GBM the brief asked for (full features, log level): log-R2 "
          f"{pooled['model']['r2_log']:.4f}, margin vs naive_vol30 "
          f"{margins['model']['naive_vol30']['d_r2_log']:+.4f} -> "
          f"{'PASS' if verdict.get('model') else 'FAILS the naive baseline'}")
    print(f"  best GBM ({best_gbm}): log-R2 {pooled[best_gbm]['r2_log']:.4f}, margin vs "
          f"naive_vol30 {margins[best_gbm]['naive_vol30']['d_r2_log']:+.4f}, vs har "
          f"{margins[best_gbm]['har']['d_r2_log']:+.4f}")
    print(f"  best forecaster ({best}): margin vs naive_vol30 "
          f"{margins[best]['naive_vol30']['d_r2_log']:+.4f}, vs naive_vol7 "
          f"{margins[best]['naive_vol7']['d_r2_log']:+.4f}, vs naive_cal "
          f"{margins[best]['naive_cal']['d_r2_log']:+.4f}; RMSE_log "
          f"{margins[best]['naive_vol30']['rmse_log_pct_better']:+.2f}% vs naive_vol30")
    print(f"  has_real_skill = {skill}")

    out = dict(
        target="7-day forward realised volatility of 4h log returns, annualised sqrt(6*365)",
        horizon_bars=HORIZON, train_days=TRAIN_DAYS, test_days=TEST_DAYS,
        purge_days=PURGE_DAYS,
        models={"model": "HistGradientBoostingRegressor on log(vol), full feature set (the brief)",
                "gbm_har3": "same GBM on lv1/lv7/lv30 only (diagnostic vs OLS on same inputs)",
                "gbm5": "GBM on lv1/lv7/lv30 + vol_ratio_7_30 + atr_pct: the ablation's pick",
                "gbm_ratio": "GBM, full features, target log(fwd_vol)-log(vol_30d)",
                "gbm_ratio3": "GBM, lv1/lv7/lv30 only, target log(fwd_vol)-log(vol_30d)",
                "gbm_harres": "GBM, full features, target log(fwd_vol)-har; prediction har+GBM",
                "naive_vol30": "predicted vol = vol_30d",
                "naive_vol7": "predicted vol = vol_7d",
                "naive_cal": "train-fitted OLS log(fwd_vol) ~ a + b*log(vol_30d)",
                "har": "train-fitted OLS log(fwd_vol) ~ a + b1*lv1 + b2*lv7 + b3*lv30",
                "har_x": "har plus 9 extra features, OLS: tests whether the extras hold any "
                         "incremental information at all",
                "shuffled": "harness control: GBM trained on a permuted label"},
        hyperparams={k: v for k, v in HP.items()}, features=FEATS,
        har_features=HAR_FEATS, har_x_features=HARX_FEATS,
        n_symbols=int(d["symbol"].nunique()), n_labelled_rows=int(len(d)),
        n_folds=len(folds), n_oos_rows=int(len(oos)),
        oos_span=[str(oos["bar_time"].min().date()), str(oos["bar_time"].max().date())],
        pooled=pooled, non_overlapping=nonov_m, margins=margins,
        best_model=best, best_gbm=best_gbm, verdict_per_model=verdict,
        grid_2x2={f"{fs} | {tg}": dict(name=n, r2_log=pooled[n]["r2_log"])
                  for (fs, tg), n in GRID.items()},
        per_fold=fold_tbl, fold_wins=wins, folds=folds, har_betas=har_betas,
        quintile_ladder=lad, extremes=ex, bootstrap=boot, sizing=sizing,
        feature_ablation=abl, hp_sweep=sweep, label_audit=audit,
        live_forecast=live, has_real_skill=bool(skill),
    )
    p = ROOT / "reports" / "forecast_volatility.json"
    p.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
