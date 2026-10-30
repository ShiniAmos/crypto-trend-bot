"""Market-state model: classify BTC's trend into four states and PREDICT the next one.

WHY THIS EXISTS. The shipped bot is reactive: it waits for a close above the prior 20-bar high.
The owner asked for a bot that predicts the trend and enters on the prediction. Every attempt in
this repo to predict a RETURN has failed (next-7d direction AUC 0.47-0.52, right-tail classifier
0.49, ML ranker/veto/regression all cut CAGR by a third, 265 variants across nine families shipped
nothing). But a return and a trend are not the same object. Returns are close to a martingale;
trend STATE is slow and autocorrelated. "Is the market in a persistent uptrend, and will it still
be in one in seven days" is a different question, and it is the one nobody here has asked.

WHAT IS BUILT, in three parts.

1. STATES, from BTC only, causally. Two readings: `above` (BTC above its 200-day average) and
   `mom` = trailing 90-day return divided by 30-day realised volatility, which is the trailing
   trend measured in units of how fast the thing moves. Four states:
       3 strong uptrend  above and mom above T_hi
       2 weak uptrend    above and mom between T_lo and T_hi
       1 chop            above but mom below T_lo, OR below trend with mom still positive
       0 downtrend       below trend and mom negative
   T_hi and T_lo are quantiles of `mom` fitted ON THE TRAINING WINDOW OF EACH FOLD ONLY and then
   applied to that fold's test bars. They are refitted every fold, so the thresholds drift; the
   drift is recorded in the report rather than hidden by a single global fit (a global fit would be
   lookahead).

   TWO definitions are carried all the way through, because the first one measured badly. `raw`
   reads the thresholds bar by bar, and its median episode is under one day: `mom` sits on a
   threshold and the label chatters between neighbouring states. `dwell` adds hysteresis - the
   state changes only once the raw reading has held for a full day (6 bars) - which is still purely
   causal and makes the episodes behave like regimes instead of like noise. Reporting both is the
   point: it shows the state definition was not tuned until the answer came out nicely.

2. THE TRANSITION MODEL. Target: the state 7 days (42 bars) ahead. Five predictors, same folds,
   same rows:
       prior        - the training window's class frequencies. The floor.
       persistence  - the state in 7 days is the state now (one-hot). The predictor to beat.
       markov       - P(state_{t+7d} | state_t), counted DIRECTLY at the 7-day lag on the training
                      window. The honest baseline the brief asks for, and pure arithmetic.
       markov_dur   - the same, conditioned also on how long the current state has lasted
                      (four duration buckets, falling back to the plain row on thin cells).
                      The direct test of "does duration carry information".
       gbm / logit  - multiclass models, 19 causal BTC features including duration and funding.
   A model that cannot beat `markov` has added nothing, however good its raw number looks.

   A detail that decides how part 3 has to be read: at this horizon the modal next state is always
   the current state, so argmax(markov) IS persistence, bar for bar. Anything traded off
   argmax(markov) is therefore a STATE-NOW gate, not a forecast. Only the gbm/logit columns
   actually call changes, so only those can honour the owner's requirement, and they are traded
   separately and labelled as such.

3. THE STRATEGY. The predicted state replaces BOTH reactive triggers:
       breakout  := the model says an uptrend is ahead   (NEVER don_high - the trigger is gone)
       regime_on := the model says an uptrend is ahead   (NOT dist_sma200 > 0)
   so entry and the market-level exit are both forward-looking. Selection among coins is the
   shipped ranker, untouched. Sizing is tested flat, state-discrete (full in a strong uptrend, 60%
   in a weak one) and conviction-continuous, through size_scale_col. Stops are tested both as the
   shipped 20-bar low and as a 3-ATR stop, because with no breakout trigger an entry can land
   inches above the 20-bar low, which would make the risk-based size explode into the weight cap.

   Three controls run alongside, and they are what makes any result attributable:
       ctrl_notrig_nomodel  - trigger removed, NO model, gate still the reactive 200-day cross.
                              Separates "dropping the breakout trigger" from "the forecast".
       ctrl_shipped_trigger - the 20-bar-high trigger kept, only the regime comes from the model.
                              Separates "the forecast" from "dropping the trigger".
       ctrl_and_now         - the model must agree with the state the market is in right now.
   Without those, a win or a loss could not be assigned to the forecast at all.

CAUSALITY. Panel features are already own-past only (features.symbol_features; daily SMAs joined
as-of backward). Labels look forward a fixed 42 bars. Train and test are separated by a purge gap
of 43 bars, so no training label can overlap a test bar. State thresholds, the dwell filter, the
Markov matrices, the class priors and the logit's standardisation all come from the training window
alone. In the backtest a prediction stamped at bar t is read on bar t's close and fills at bar
t+1's open, which is the engine's own timing, unchanged. The engine, its costs, its fills, its risk
limits and its three slots are untouched, so every number here is comparable with the shipped bot's.

SAMPLE SIZE, honestly. 42-bar labels on 4h bars overlap almost completely; ~14,000 out-of-sample
rows are nowhere near 14,000 observations. The real unit is the STATE EPISODE, and nine years of
BTC holds only a few dozen long ones. Episode counts sit next to every headline number, and the
accuracy margin over Markov gets a moving-block bootstrap with blocks as long as the label.

    cd crypto-bot && PYTHONIOENCODING=utf-8 python research/trend_states.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import backtest, config, features, funding  # noqa: E402
from app.flags import Flags  # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import log_loss, roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "trend_states.json"
BPD = 6                                  # 4h bars per day
HORIZON = 7 * BPD                        # 42 bars = the 7-day forecast horizon
TRAIN_DAYS = 730
TEST_DAYS = 91
MIN_TRAIN_ROWS = 2000
SEED = 0
STATES = ["downtrend", "chop", "weak uptrend", "strong uptrend"]      # index 0..3
UP = 2                                   # state >= UP means "in an uptrend"
Q_LO, Q_HI = 0.25, 0.60                  # training quantiles of `mom` among above-trend bars
DUR_EDGES = [0, 7, 21, 60, np.inf]       # duration buckets in days
DWELL = BPD                              # hysteresis: a state change must hold one full day
HELD_OUT = "2022-01-01"
MODELS = ["gbm", "logit", "markov_dur", "markov", "persistence", "prior"]

SHIPPED = Flags(abs_momentum=True, volume_min=0.8, taker_min=0.5)
FUNDING = Flags(abs_momentum=True, volume_min=0.8, taker_min=0.5, funding_max_pct=0.8)
SHIP_RULE = {                            # the numbers a candidate has to beat, from the brief
    "standard": {"cagr": 0.2073, "sharpe": 1.4295, "max_drawdown": -0.1324, "trades": 366},
    "heldout": {"cagr": 0.1548, "sharpe": 1.2229, "max_drawdown": -0.1324, "trades": 210},
}


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ 1. states and features

def btc_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """One causal row per 4h BTC bar. Every column at t uses bars closed at or before t."""
    b = (panel[panel["symbol"] == "BTCUSDT"].sort_values("bar_time").set_index("bar_time"))
    b = b[b["dist_sma200"].notna()].copy()          # the 200-day line is undefined before that
    d = b["dist_sma200"]
    b["above"] = d > 0
    b["ret_90d"] = b["close"].pct_change(90 * BPD)
    b["mom"] = b["ret_90d"] / b["vol_30d"]          # trailing trend in volatility units
    b["dist_over_vol"] = d / b["vol_30d"]           # "how many sigmas to the 200-day line"
    b["slope_sma200_7d"] = d - d.shift(7 * BPD)
    b["slope_sma200_30d"] = d - d.shift(30 * BPD)
    b["vol_ratio"] = b["vol_7d"] / b["vol_30d"]     # is volatility expanding right now
    b["mom_chg_7d"] = b["mom"] - b["mom"].shift(7 * BPD)
    b["mom_chg_30d"] = b["mom"] - b["mom"].shift(30 * BPD)
    b["dd_1y"] = b["close"] / b["close"].rolling(365 * BPD, min_periods=30 * BPD).max() - 1
    b["ret_14d"] = b["close"].pct_change(14 * BPD)
    return b


def raw_states(b: pd.DataFrame, t_lo: float, t_hi: float) -> pd.Series:
    """The four states, from `above` and `mom` plus two thresholds fitted on training data."""
    m, above = b["mom"], b["above"]
    s = pd.Series(1.0, index=b.index)                                  # chop by default
    s[above & (m > t_hi)] = 3
    s[above & (m > t_lo) & (m <= t_hi)] = 2
    s[above & (m <= t_lo)] = 1
    s[~above & (m > 0)] = 1
    s[~above & (m <= 0)] = 0
    s[m.isna()] = np.nan                                               # no trailing trend yet
    return s


def dwell_filter(raw: pd.Series, k: int) -> pd.Series:
    """Hysteresis: adopt a new state only after the raw reading has held k consecutive bars.

    Causal by construction - the value at bar i is a function of raw readings at bars <= i only.
    Without this the label chatters across a threshold and the "regime" has a median life of under
    a day, which is not a regime.
    """
    v = raw.to_numpy(float)
    out = np.full(len(v), np.nan)
    cur = np.nan
    run_val, run = np.nan, 0
    for i, x in enumerate(v):
        if np.isnan(x):
            out[i] = cur
            continue
        if x == run_val:
            run += 1
        else:
            run_val, run = x, 1
        if np.isnan(cur):
            cur = x                       # bootstrap on the first readable bar
        elif x != cur and run >= k:
            cur = x
        out[i] = cur
    return pd.Series(out, index=raw.index)


def label_states(b, t_lo, t_hi, dwell=0) -> pd.Series:
    s = raw_states(b, t_lo, t_hi)
    return dwell_filter(s, dwell) if dwell else s


def duration_days(state: pd.Series) -> pd.Series:
    """Bars since this state began, in days. Past-only: a run length cannot see the future."""
    s = state.ffill()
    run = s.ne(s.shift(1)).cumsum()
    return s.groupby(run).cumcount() / BPD


FEATS = ["dist_sma200", "dist_over_vol", "slope_sma200_7d", "slope_sma200_30d",
         "ret_7d", "ret_14d", "ret_30d", "ret_90d", "mom", "mom_chg_7d", "mom_chg_30d",
         "vol_7d", "vol_30d", "vol_ratio", "dd_1y", "volume_ratio", "taker_ratio",
         "funding_7d", "dur_days", "state_f"]


def design(b: pd.DataFrame, state: pd.Series, dur: pd.Series) -> pd.DataFrame:
    X = b.reindex(columns=[c for c in FEATS if c not in ("dur_days", "state_f")]).copy()
    X["dur_days"] = dur
    X["state_f"] = state                      # the model is allowed to know where it is now
    return X.replace([np.inf, -np.inf], np.nan)


# ------------------------------------------------------------------ 2. baselines

def markov(state_tr: np.ndarray, next_tr: np.ndarray, alpha=1.0) -> np.ndarray:
    """Direct 7-day-lag transition matrix, row-normalised, Laplace-smoothed."""
    M = np.full((4, 4), alpha, float)
    for a, c in zip(state_tr.astype(int), next_tr.astype(int)):
        M[a, c] += 1
    return M / M.sum(axis=1, keepdims=True)


def markov_duration(state_tr, next_tr, dur_tr, base: np.ndarray, min_cell=60):
    """P(next | state, duration bucket), falling back to the plain Markov row on thin cells."""
    buckets = np.digitize(dur_tr, DUR_EDGES[1:-1])
    out = {}
    for s in range(4):
        for k in range(len(DUR_EDGES) - 1):
            m = (state_tr == s) & (buckets == k)
            out[(s, k)] = markov(state_tr[m], next_tr[m])[s] if m.sum() >= min_cell else base[s]
    return out


def apply_markov_duration(table, state_te, dur_te):
    buckets = np.digitize(dur_te, DUR_EDGES[1:-1])
    return np.array([table[(int(s), int(k))] for s, k in zip(state_te, buckets)])


def onehot(state: np.ndarray) -> np.ndarray:
    P = np.zeros((len(state), 4))
    P[np.arange(len(state)), state.astype(int)] = 1.0
    return P


# ------------------------------------------------------------------ 3. scoring

def macro_auc(y: np.ndarray, P: np.ndarray) -> dict:
    """One-vs-rest AUC per state plus the macro average over the states that are present."""
    per = {}
    for k in range(4):
        yk = (y == k).astype(int)
        per[STATES[k]] = float(roc_auc_score(yk, P[:, k])) if 0 < yk.sum() < len(yk) else None
    vals = [v for v in per.values() if v is not None]
    return {"per_state": per, "macro": float(np.mean(vals)) if vals else None}


def score_block(y: np.ndarray, P: np.ndarray) -> dict:
    P = np.clip(P, 1e-9, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    yh = P.argmax(axis=1)
    up_true = (y >= UP).astype(int)
    up_p = P[:, UP:].sum(axis=1)
    a = macro_auc(y, P)
    return {"accuracy": float((yh == y).mean()),
            "macro_auc": a["macro"], "auc_per_state": a["per_state"],
            "log_loss": float(log_loss(y, P, labels=[0, 1, 2, 3])),
            "brier": float(np.mean((P - onehot(y)) ** 2) * 4),
            "uptrend_auc": (float(roc_auc_score(up_true, up_p))
                            if 0 < up_true.sum() < len(up_true) else None),
            "uptrend_accuracy": float(((up_p >= 0.5).astype(int) == up_true).mean()),
            "uptrend_base_rate": float(up_true.mean())}


def block_bootstrap_margin(y, Pa, Pb, block=HORIZON, n=400, seed=SEED):
    """Moving-block bootstrap on the accuracy margin of Pa over Pb, blocks as long as the label."""
    rng = np.random.default_rng(seed)
    d = (Pa.argmax(axis=1) == y).astype(float) - (Pb.argmax(axis=1) == y).astype(float)
    n_blocks = max(1, len(d) // block)
    starts = len(d) - block
    if starts <= 0:
        return {}
    draws = [float(np.mean([d[s:s + block].mean()
                            for s in rng.integers(0, starts, n_blocks)])) for _ in range(n)]
    return {"mean": float(np.mean(d)), "p05": float(np.percentile(draws, 5)),
            "p95": float(np.percentile(draws, 95)),
            "share_positive": float(np.mean(np.array(draws) > 0))}


# ------------------------------------------------------------------ walk-forward

def fit_logit(Xtr, ytr, Xte):
    med = Xtr.median()
    mu, sd = Xtr.mean(), Xtr.std().replace(0, 1.0)
    A = ((Xtr.fillna(med) - mu) / sd).values
    B = ((Xte.fillna(med) - mu) / sd).values
    m = LogisticRegression(C=1.0, max_iter=3000)
    m.fit(A, ytr)
    P = np.zeros((len(B), 4))
    P[:, m.classes_.astype(int)] = m.predict_proba(B)
    return P


def fit_gbm(Xtr, ytr, Xte):
    g = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=7,
                                       min_samples_leaf=200, l2_regularization=1.0,
                                       random_state=SEED)
    g.fit(Xtr, ytr)
    P = np.zeros((len(Xte), 4))
    P[:, g.classes_.astype(int)] = g.predict_proba(Xte)
    return P


def walk_forward(b: pd.DataFrame, dwell=0, tag="") -> tuple:
    """OOS predictions from every predictor, one row per test bar, plus the per-fold record."""
    t = b.index
    purge = pd.Timedelta(hours=4 * (HORIZON + 1))
    rows, folds = [], []
    test_start = t.min() + pd.Timedelta(days=TRAIN_DAYS)
    while test_start < t.max():
        test_end = test_start + pd.Timedelta(days=TEST_DAYS)
        tr_win = (t >= test_start - pd.Timedelta(days=TRAIN_DAYS)) & (t < test_start - purge)
        te_win = (t >= test_start) & (t < test_end)
        if tr_win.sum() < MIN_TRAIN_ROWS or te_win.sum() == 0:
            test_start = test_end
            continue

        # Thresholds from the TRAINING window only, then applied to the whole index so the state,
        # its duration and the forward label are all one consistent causal function of past bars.
        mom_tr, above_tr = b.loc[tr_win, "mom"], b.loc[tr_win, "above"]
        pool = mom_tr[above_tr & mom_tr.notna()]
        if len(pool) < 200:
            test_start = test_end
            continue
        t_lo, t_hi = float(pool.quantile(Q_LO)), float(pool.quantile(Q_HI))

        state = label_states(b, t_lo, t_hi, dwell)
        dur = duration_days(state)
        nxt = state.shift(-HORIZON)                     # the label: the state 7 days ahead
        X = design(b, state, dur)

        ok = (state.notna() & nxt.notna()).values
        tr, te = tr_win & ok, te_win & ok
        if tr.sum() < MIN_TRAIN_ROWS or te.sum() == 0 or state[tr].nunique() < 2:
            test_start = test_end
            continue

        ytr, yte = nxt[tr].astype(int).values, nxt[te].astype(int).values
        s_tr, s_te = state[tr].astype(int).values, state[te].astype(int).values
        M = markov(s_tr, ytr)
        dur_tab = markov_duration(s_tr, ytr, dur[tr].values, M)
        prior = np.array([(ytr == k).mean() for k in range(4)])

        P = {"gbm": fit_gbm(X[tr], ytr, X[te]),
             "logit": fit_logit(X[tr], ytr, X[te]),
             "markov": M[s_te],
             "markov_dur": apply_markov_duration(dur_tab, s_te, dur[te].values),
             "persistence": onehot(s_te),
             "prior": np.tile(prior, (len(s_te), 1))}

        f = pd.DataFrame({"y": yte, "state": s_te, "dur": dur[te].values}, index=t[te])
        for name, Pm in P.items():
            for k in range(4):
                f[f"{name}_{k}"] = Pm[:, k]
        f["fold"], f["t_lo"], f["t_hi"] = str(test_start.date()), t_lo, t_hi
        rows.append(f)
        folds.append({"fold": str(test_start.date()), "train_rows": int(tr.sum()),
                      "test_rows": int(te.sum()), "t_lo": t_lo, "t_hi": t_hi,
                      "acc_gbm": float((P["gbm"].argmax(1) == yte).mean()),
                      "acc_markov": float((P["markov"].argmax(1) == yte).mean()),
                      "acc_persistence": float((P["persistence"].argmax(1) == yte).mean())})
        log(f"  [{tag}] fold {test_start.date()} train={int(tr.sum())} test={int(te.sum())} "
            f"T=[{t_lo:+.3f},{t_hi:+.3f}] acc gbm={folds[-1]['acc_gbm']:.3f} "
            f"markov={folds[-1]['acc_markov']:.3f} pers={folds[-1]['acc_persistence']:.3f}")
        test_start = test_end
    return (pd.concat(rows).sort_index() if rows else pd.DataFrame()), folds


def probs(oos: pd.DataFrame, name: str) -> np.ndarray:
    return oos[[f"{name}_{k}" for k in range(4)]].to_numpy()


# ------------------------------------------------------------------ descriptive stickiness

def stickiness(b: pd.DataFrame, dwell=0) -> dict:
    """How sticky the states are, described on the FULL sample.

    Descriptive only: not used to produce any traded signal, so the full-sample threshold fit is a
    description rather than lookahead. Nothing in part 3 reads these numbers.
    """
    pool = b["mom"][b["above"] & b["mom"].notna()]
    t_lo, t_hi = float(pool.quantile(Q_LO)), float(pool.quantile(Q_HI))
    s = label_states(b, t_lo, t_hi, dwell).dropna().astype(int)
    out = {"thresholds_full_sample": {"t_lo": t_lo, "t_hi": t_hi},
           "share_of_bars": {STATES[k]: float((s == k).mean()) for k in range(4)}}
    for lag, lab in [(1, "4h"), (BPD, "1d"), (HORIZON, "7d"), (30 * BPD, "30d")]:
        nx = s.shift(-lag).dropna().astype(int)
        a = s.loc[nx.index].values
        M = np.zeros((4, 4))
        for i, j in zip(a, nx.values):
            M[i, j] += 1
        rs = M.sum(axis=1, keepdims=True)
        out[f"transition_{lab}"] = {
            STATES[i]: ({STATES[j]: (float(M[i, j] / rs[i, 0]) if rs[i, 0] else None)
                         for j in range(4)} | {"n": int(rs[i, 0])}) for i in range(4)}
    run = s.ne(s.shift(1)).cumsum()
    eps = s.groupby(run).agg(state="first", bars="size")
    out["episodes"] = {
        "total": int(len(eps)), "longer_than_horizon": int((eps["bars"] > HORIZON).sum()),
        "per_state": {STATES[k]: {
            "episodes": int((eps["state"] == k).sum()),
            "median_days": (float(eps.loc[eps["state"] == k, "bars"].median() / BPD)
                            if (eps["state"] == k).any() else None),
            "mean_days": (float(eps.loc[eps["state"] == k, "bars"].mean() / BPD)
                          if (eps["state"] == k).any() else None),
            "longest_days": (float(eps.loc[eps["state"] == k, "bars"].max() / BPD)
                             if (eps["state"] == k).any() else None)} for k in range(4)}}
    return out


# ------------------------------------------------------------------ 4. the strategy

def pred_table(oos: pd.DataFrame, model: str) -> pd.DataFrame:
    """Per-bar predicted state probabilities, indexed by bar_time. OOS folds only."""
    P = probs(oos, model)
    P = P / P.sum(axis=1, keepdims=True)
    return pd.DataFrame({"p_down": P[:, 0], "p_chop": P[:, 1], "p_weak": P[:, 2],
                         "p_strong": P[:, 3], "p_up": P[:, UP:].sum(axis=1),
                         "pred_state": P.argmax(axis=1).astype(float),
                         "state_now": oos["state"].values}, index=oos.index)


def up_signal(tbl: pd.DataFrame, gate: str) -> pd.Series:
    """Turn a prediction into the boolean "an uptrend is ahead", indexed by bar_time.

    argmax   - the most likely state 7 days out is an uptrend
    pNN      - P(uptrend in 7 days) >= 0.NN
    and_now  - argmax AND the market is in an uptrend right now (the agreement control)
    now      - the state right now, no forecast at all (the arithmetic reference)
    """
    pred_up = tbl["pred_state"] >= UP
    now_up = tbl["state_now"] >= UP
    if gate == "argmax":
        return pred_up
    if gate == "and_now":
        return pred_up & now_up
    if gate == "now":
        return now_up
    return tbl["p_up"] >= float(gate[1:]) / 100


def build(panel: pd.DataFrame, tbl: pd.DataFrame, up: pd.Series, trigger="model",
          regime="model", stop="donchian", size=None, atr_k=3.0):
    """A copy of the panel with the strategy written into the engine's own columns.

    trigger  - "model": breakout IS the forecast, so don_high is never read. "shipped": keep the
               reactive 20-bar-high trigger. "always": trigger removed and no model consulted.
    regime   - "model": regime_on is the forecast, governing entry AND the market-level exit.
               "shipped": keep dist_sma200 > 0.
    stop     - "donchian": the shipped 20-bar low. "atr": close * (1 - k*ATR%), which keeps the
               risk-based size sane when the entry is not a fresh high.
    size     - None, "state" (full in a strong uptrend, 60% in a weak one) or "conviction"
               (0.5 + p_strong), written to a column for size_scale_col.
    """
    p = panel.copy()
    p["_up"] = p["bar_time"].map(up.astype(float)).fillna(0.0) > 0.5     # no forecast = no entry
    p["p_strong"] = p["bar_time"].map(tbl["p_strong"])
    p["pred_state"] = p["bar_time"].map(tbl["pred_state"])

    if regime == "model":
        p["regime_on"] = p["_up"]
    if trigger == "model":
        p["breakout"] = p["regime_on"] if regime == "model" else p["_up"]
    elif trigger == "always":
        p["breakout"] = True

    if stop == "atr":
        alt = p["close"] * (1 - atr_k * p["atr_pct"])
        # Must stay strictly below the close or sizing returns zero; fall back to the shipped level
        # wherever ATR is missing or the arithmetic would put the stop at or above the price.
        p["stop_level"] = np.where(p["atr_pct"].notna() & (alt < p["close"]) & (alt > 0),
                                   alt, p["stop_level"])
    col = None
    if size == "state":
        col = "_size_scale"
        p[col] = np.where(p["pred_state"] >= 3, 1.0, np.where(p["pred_state"] >= UP, 0.6, 0.5))
    elif size == "conviction":
        col = "_size_scale"
        p[col] = (0.5 + p["p_strong"]).fillna(1.0)
    return p, col


def run_one(p, s, flags, col, start):
    r = backtest.run(p, s, flags=flags, start=start, size_scale_col=col)
    if r.trades.empty or len(r.equity) < 10:
        return {"cagr": float("nan"), "sharpe": float("nan"), "max_drawdown": float("nan"),
                "trades": 0}
    m = backtest.metrics(r.equity, r.trades, r.exposure)
    return {k: (float(m[k]) if k in m else None) for k in
            ["cagr", "sharpe", "max_drawdown", "trades", "win_rate", "avg_trade",
             "profit_factor", "avg_bars_held", "avg_exposure", "calmar"]}


def passes(cand: dict, base: dict, tol=1e-9) -> bool:
    if not np.isfinite(cand.get("cagr", np.nan)):
        return False
    return (cand["cagr"] >= base["cagr"] - tol and cand["sharpe"] >= base["sharpe"] - tol
            and cand["max_drawdown"] >= base["max_drawdown"] - tol)


# the grid. forecast-driven entries first, then the controls that make them attributable.
VARIANTS = [
    # name,                       states,  model,     gate,     trigger,  regime,  stop,  size, flags
    ("gbm_argmax_atr",            "dwell", "gbm",     "argmax", "model", "model", "atr", None, "shipped"),
    ("gbm_argmax_donch",          "dwell", "gbm",     "argmax", "model", "model", "donchian", None, "shipped"),
    ("gbm_p50_atr",               "dwell", "gbm",     "p50",    "model", "model", "atr", None, "shipped"),
    ("gbm_p65_atr",               "dwell", "gbm",     "p65",    "model", "model", "atr", None, "shipped"),
    ("gbm_argmax_atr_funding",    "dwell", "gbm",     "argmax", "model", "model", "atr", None, "funding"),
    ("gbm_argmax_atr_size_state", "dwell", "gbm",     "argmax", "model", "model", "atr", "state", "shipped"),
    ("gbm_argmax_atr_size_conv",  "dwell", "gbm",     "argmax", "model", "model", "atr", "conviction", "shipped"),
    ("gbm_and_now_atr",           "dwell", "gbm",     "and_now", "model", "model", "atr", None, "shipped"),
    ("gbm_and_now_atr_funding",   "dwell", "gbm",     "and_now", "model", "model", "atr", None, "funding"),
    ("gbm_argmax_atr_raw",        "raw",   "gbm",     "argmax", "model", "model", "atr", None, "shipped"),
    ("gbm_argmax_atr_nofilters",  "dwell", "gbm",     "argmax", "model", "model", "atr", None, "none"),
    ("logit_argmax_atr",          "dwell", "logit",   "argmax", "model", "model", "atr", None, "shipped"),
    # argmax(markov) == the state right now, so these are STATE-NOW gates, not forecasts
    ("statenow_atr",              "dwell", "markov",  "now",    "model", "model", "atr", None, "shipped"),
    ("statenow_donch",            "dwell", "markov",  "now",    "model", "model", "donchian", None, "shipped"),
    ("statenow_atr_funding",      "dwell", "markov",  "now",    "model", "model", "atr", None, "funding"),
    ("statenow_atr_size_state",   "dwell", "markov",  "now",    "model", "model", "atr", "state", "shipped"),
    ("statenow_p70_atr",          "dwell", "markov",  "p70",    "model", "model", "atr", None, "shipped"),
    ("statenow_atr_raw",          "raw",   "markov",  "now",    "model", "model", "atr", None, "shipped"),
    # controls
    ("ctrl_notrig_nomodel_atr",   "dwell", "gbm",     "argmax", "always", "shipped", "atr", None, "shipped"),
    ("ctrl_notrig_nomodel_donch", "dwell", "gbm",     "argmax", "always", "shipped", "donchian", None, "shipped"),
    ("ctrl_shiptrig_gbm_regime",  "dwell", "gbm",     "argmax", "shipped", "model", "donchian", None, "shipped"),
    ("ctrl_shiptrig_state_regime", "dwell", "markov", "now",    "shipped", "model", "donchian", None, "shipped"),
    ("ctrl_gbmtrig_react_regime", "dwell", "gbm",     "argmax", "model", "shipped", "atr", None, "shipped"),
]
FLAGSETS = {"shipped": SHIPPED, "funding": FUNDING, "none": Flags()}


# ------------------------------------------------------------------ main

def skill_report(oos, folds, label, out):
    y = oos["y"].to_numpy(int)
    sec = {"rows": int(len(oos)), "folds": len(folds),
           "start": str(oos.index.min().date()), "end": str(oos.index.max().date()),
           "label_distribution": {STATES[k]: float((y == k).mean()) for k in range(4)},
           "skill": {n: score_block(y, probs(oos, n)) for n in MODELS}}
    base, pers = sec["skill"]["markov"], sec["skill"]["persistence"]
    sec["margins_vs_markov"] = {
        n: {"accuracy": sec["skill"][n]["accuracy"] - base["accuracy"],
            "macro_auc": (sec["skill"][n]["macro_auc"] or 0) - (base["macro_auc"] or 0),
            "log_loss": sec["skill"][n]["log_loss"] - base["log_loss"],
            "uptrend_auc": (sec["skill"][n]["uptrend_auc"] or 0) - (base["uptrend_auc"] or 0)}
        for n in MODELS}
    sec["margins_vs_persistence"] = {
        n: {"accuracy": sec["skill"][n]["accuracy"] - pers["accuracy"],
            "uptrend_auc": (sec["skill"][n]["uptrend_auc"] or 0) - (pers["uptrend_auc"] or 0)}
        for n in MODELS}
    sec["bootstrap_acc_margin"] = {
        "gbm_over_markov": block_bootstrap_margin(y, probs(oos, "gbm"), probs(oos, "markov")),
        "markov_over_persistence": block_bootstrap_margin(y, probs(oos, "markov"),
                                                          probs(oos, "persistence"))}
    s_now = oos["state"].to_numpy(int)
    changed = s_now != y
    sec["state_changed_share"] = float(changed.mean())
    sec["change_detection"] = {}
    for n in MODELS:
        called = probs(oos, n).argmax(1) != s_now
        tp = int((called & changed).sum())
        sec["change_detection"][n] = {
            "predicted_a_change": int(called.sum()),
            "recall_on_changes": float(tp / changed.sum()) if changed.sum() else None,
            "precision_on_changes": float(tp / called.sum()) if called.sum() else None}
    sec["argmax_markov_equals_persistence"] = bool(
        (probs(oos, "markov").argmax(1) == probs(oos, "persistence").argmax(1)).all())
    out[label] = sec

    log(f"\n  [{label}] pooled OOS {len(oos)} rows / {len(folds)} folds, "
        f"{oos.index.min().date()} .. {oos.index.max().date()}; "
        f"the state differed 7d later on {changed.mean():.1%} of bars")
    log(f"  {'predictor':<13}{'acc':>8}{'macroAUC':>10}{'logloss':>10}{'upAUC':>8}{'upAcc':>8}"
        f"{'dAcc_mkv':>10}{'dUpAUC':>9}{'chg_recall':>12}")
    for n in MODELS:
        k, m = sec["skill"][n], sec["margins_vs_markov"][n]
        c = sec["change_detection"][n]
        log(f"  {n:<13}{k['accuracy']:>8.4f}{(k['macro_auc'] or 0):>10.4f}{k['log_loss']:>10.4f}"
            f"{(k['uptrend_auc'] or 0):>8.4f}{k['uptrend_accuracy']:>8.4f}"
            f"{m['accuracy']:>+10.4f}{m['uptrend_auc']:>+9.4f}"
            f"{(c['recall_on_changes'] or 0):>12.3f}")
    bs = sec["bootstrap_acc_margin"]["gbm_over_markov"]
    if bs:
        log(f"  gbm-over-markov accuracy margin: mean {bs['mean']:+.4f} "
            f"[{bs['p05']:+.4f}, {bs['p95']:+.4f}], positive in {bs['share_positive']:.0%} of draws")
    log(f"  argmax(markov) is identical to persistence: {sec['argmax_markov_equals_persistence']}")
    return sec


def main():
    t0 = time.time()
    s = config.load()
    u = json.loads((ROOT / "reports" / "_parity_universe.json").read_text())
    log("building the panel ...")
    panel = features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars)
    panel = funding.attach(panel, s)
    log(f"  {len(panel)} rows, {panel['symbol'].nunique()} symbols, "
        f"{panel['bar_time'].min().date()} .. {panel['bar_time'].max().date()}")

    report = {"horizon_bars": HORIZON, "train_days": TRAIN_DAYS, "test_days": TEST_DAYS,
              "states": STATES, "quantiles": {"lo": Q_LO, "hi": Q_HI}, "dwell_bars": DWELL,
              "ship_rule": SHIP_RULE}
    b = btc_frame(panel)

    log("\n--- how sticky are the states (full sample, descriptive only) ---")
    for label, dw in [("raw", 0), ("dwell", DWELL)]:
        st = stickiness(b, dw)
        report[f"stickiness_{label}"] = st
        log(f"  [{label}]  episodes={st['episodes']['total']}  "
            f"longer_than_7d={st['episodes']['longer_than_horizon']}")
        for k in STATES:
            row, e = st[f"transition_7d"][k], st["episodes"]["per_state"][k]
            log(f"    {k:<16} share={st['share_of_bars'][k]:.3f} stay_7d={row[k]:.3f} "
                f"episodes={e['episodes']:<4} median={e['median_days']:.1f}d "
                f"mean={e['mean_days']:.1f}d longest={e['longest_days']:.0f}d")

    log("\n--- A. does the state transition PREDICT? walk-forward ---")
    oos_all, folds_all = {}, {}
    for label, dw in [("raw", 0), ("dwell", DWELL)]:
        oos_all[label], folds_all[label] = walk_forward(b, dw, label)
        if oos_all[label].empty:
            raise SystemExit(f"no folds produced for {label}")
        report[f"folds_{label}"] = folds_all[label]
    report["partA"] = {}
    for label in ("raw", "dwell"):
        skill_report(oos_all[label], folds_all[label], label, report["partA"])

    log("\n--- B. does the prediction MAKE MONEY? ---")
    preds = {sd: {m: pred_table(oos_all[sd], m) for m in ("gbm", "logit", "markov")}
             for sd in ("raw", "dwell")}
    first = min(oos_all["raw"].index.min(), oos_all["dwell"].index.min())
    start_std = str(first.date())
    report["candidate_start"] = start_std

    bases = {}
    for tag, st_ in [("standard", None), ("heldout", HELD_OUT),
                     ("standard_same_span", start_std)]:
        bases[tag] = run_one(panel, s, SHIPPED, None, st_)
        log(f"  shipped bot [{tag:<19}] cagr={bases[tag]['cagr']:.4f} "
            f"sharpe={bases[tag]['sharpe']:.4f} maxdd={bases[tag]['max_drawdown']:.4f} "
            f"trades={bases[tag]['trades']:.0f}")
    report["baselines"] = bases

    rows = []
    for name, sd, model, gate, trig, reg, stop, size, fl in VARIANTS:
        tbl = preds[sd][model]
        p2, col = build(panel, tbl, up_signal(tbl, gate), trig, reg, stop, size)
        res = {"variant": name, "states": sd, "model": model, "gate": gate, "trigger": trig,
               "regime": reg, "stop": stop, "size": size or "flat", "flags": fl,
               "forward_looking": gate not in ("now",) and trig != "always"}
        for tag, st_ in [("standard", start_std), ("heldout", HELD_OUT)]:
            res[tag] = run_one(p2, s, FLAGSETS[fl], col, st_)
        res["beats_same_span"] = passes(res["standard"], bases["standard_same_span"])
        res["beats_heldout"] = passes(res["heldout"], bases["heldout"])
        res["beats_ship_rule_standard"] = passes(res["standard"], SHIP_RULE["standard"])
        res["beats_ship_rule_heldout"] = passes(res["heldout"], SHIP_RULE["heldout"])
        res["beats_both"] = res["beats_same_span"] and res["beats_heldout"]
        rows.append(res)
        log(f"  {name:<28} {fl:<8} std cagr={res['standard']['cagr']:>7.4f} "
            f"sh={res['standard']['sharpe']:>6.3f} dd={res['standard']['max_drawdown']:>7.4f} "
            f"n={res['standard']['trades']:<5.0f}| ho cagr={res['heldout']['cagr']:>7.4f} "
            f"sh={res['heldout']['sharpe']:>6.3f} dd={res['heldout']['max_drawdown']:>7.4f} "
            f"n={res['heldout']['trades']:<5.0f}{' PASS' if res['beats_both'] else ''}")
    report["strategies"] = rows
    report["winners"] = [r["variant"] for r in rows if r["beats_both"]]
    report["winners_ship_rule"] = [r["variant"] for r in rows
                                   if r["beats_ship_rule_standard"] and r["beats_ship_rule_heldout"]]

    log("\n--- verdict ---")
    for label in ("raw", "dwell"):
        sec = report["partA"][label]
        beat = [n for n in ("gbm", "logit", "markov_dur")
                if sec["margins_vs_markov"][n]["accuracy"] > 0]
        log(f"  [{label}] models beating the Markov baseline on accuracy: {beat or 'none'}")
        log(f"  [{label}] markov upAUC {sec['skill']['markov']['uptrend_auc']:.4f} vs "
            f"persistence {sec['skill']['persistence']['uptrend_auc']:.4f} vs "
            f"gbm {sec['skill']['gbm']['uptrend_auc']:.4f} vs "
            f"prior {sec['skill']['prior']['uptrend_auc']:.4f}")
    log(f"  strategies beating the shipped bot on BOTH panels: {report['winners'] or 'none'}")
    best = max(rows, key=lambda r: (r["heldout"]["sharpe"] if np.isfinite(r["heldout"]["sharpe"])
                                    else -9))
    log(f"  best held-out Sharpe among candidates: {best['variant']} "
        f"{best['heldout']['sharpe']:.3f} (shipped {bases['heldout']['sharpe']:.3f})")
    OUT.write_text(json.dumps(report, indent=2, default=str))
    log(f"\nwrote {OUT}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
