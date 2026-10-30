"""The two things in this data that ARE forecastable: volatility, and the market state.

WHY THIS MODULE EXISTS, AND WHY IT TOUCHES NO TRADE. Four separate attempts to predict RETURNS
from this data failed (walk-forward ranker, ML veto, win-probability model, right-tail classifier
at AUC 0.4914). Two targets did survive adversarial walk-forward testing:

    7-day realised volatility   pooled out-of-sample log-R2 0.53 against 0.43 for the naive
                                "next vol = vol_30d"; 25 of 28 folds, bootstrap P(not better) 0.
    7-day regime persistence    AUC 0.978 against 0.927 for "the state stays as it is".

Then `research/forecast_integration.py` tried to spend that skill: 34 sizing variants, 6 stop
variants, 3 risk gates. None passed the ship rule; the better volatility forecast made the strategy
WORSE (CAGR 0.2073 -> 0.1884) because this strategy's profit sits on the volatile side, so sizing
down the loud coins sizes down the payers. So this module is deliberately INERT: it informs the
owner and the website, and no trading decision reads it. There is no flag for it, because there is
nothing that earned one.

WHAT THE MODELS ARE. Both are small enough to audit by eye, which is the point - the research found
the small model beating the gradient booster on both targets.

    volatility   HAR: three coefficients, OLS of log(forward 7d vol) on log vol at 1, 7 and 30 days,
                 pooled over coins, plus one level correction so the mean of exp(fit) matches the
                 mean of the realised vol it was fitted on. The 18-feature booster scored BELOW the
                 naive baseline; the extra features carry no information here.
    regime       two coefficients: a logistic on dist_sma200 / vol_30d, i.e. "how many sigmas is
                 BTC from the line that defines the regime". This beat a 16-feature booster by
                 0.034 AUC. It is close to arithmetic the bot already has, and is honest about that.

CAUSALITY. Features at bar t use only bars closed at or before t (app/features.py guarantees it).
Labels look 42 bars forward, so the newest row a fit may train on is 7 days old - exactly what a
live bot has. Both models refit on a trailing 730 days each time the cache refreshes; nothing is
standardised or quantiled with statistics from outside that window.
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .features import ANNUAL, BARS_PER_DAY

HORIZON = 7 * BARS_PER_DAY        # 42 four-hour bars = 7 days, the horizon with measured skill
TRAIN_DAYS = 730
MAX_AGE_HOURS = 24.0              # refresh at most daily; the forecast moves slower than that
RIDGE = 1e-3                      # keeps a separable training window off probability 0 and 1

HAR_FEATS = ["lv1", "lv7", "lv30"]

# Frozen out-of-sample numbers from the research, carried here so the UI can quote skill without
# re-running anything. Changing a model means re-running the research and updating these.
SKILL = {
    "vol_log_r2": 0.5308, "vol_log_r2_naive": 0.4270, "vol_folds": "25/28",
    "vol_rmse_gain": 0.081,           # log-space sizing error, this module against vol_30d
    "regime_auc": 0.9777, "regime_auc_naive": 0.9289,
    "direction_auc": 0.4688,          # the control: price direction is a coin flip, slightly worse
    "source": "research/forecast_module_check.py - 28 walk-forward folds, Nov 2019 to Sep 2026, "
              "225,345 out-of-sample rows",
}


# --------------------------------------------------------------------- tiny fitters

def _ols(X, y):
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def _apply_ols(beta, X):
    return beta[0] + np.asarray(X, float) @ beta[1:]


def _logit(x, y, iters=50):
    """Two-coefficient logistic by Newton steps, ridged. numpy only, so importing this module
    costs the website nothing."""
    X = np.column_stack([np.ones(len(x)), np.asarray(x, float)])
    y = np.asarray(y, float)
    b = np.zeros(2)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ b, -30, 30)))
        g = X.T @ (y - p) - RIDGE * b
        H = X.T @ (X * (p * (1 - p))[:, None]) + RIDGE * np.eye(2)
        step = np.linalg.solve(H, g)
        b = b + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return b


# --------------------------------------------------------------------- volatility

def vol_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """Per (bar_time, symbol): the three HAR legs, and the forward 7-day realised vol label.

    The label is only defined where bars t..t+42 are a contiguous 4h grid - an outage gap would
    otherwise be measured as one enormous return.
    """
    step = pd.Timedelta(hours=4 * HORIZON)
    out = []
    for _, g in panel.sort_values(["symbol", "bar_time"]).groupby("symbol", sort=False):
        r = np.log(g["close"]).diff()
        fwd = r.rolling(HORIZON).std().shift(-HORIZON) * ANNUAL
        contiguous = (g["bar_time"].shift(-HORIZON) - g["bar_time"]) == step
        out.append(g[["bar_time", "symbol", "vol_7d", "vol_30d", "eligible"]].assign(
            vol_1d=r.rolling(BARS_PER_DAY).std() * ANNUAL,
            fwd_vol=fwd.where(contiguous)))
    d = pd.concat(out, ignore_index=True)
    ok = (d[["vol_1d", "vol_7d", "vol_30d"]] > 0).all(axis=1) & d["eligible"].fillna(False)
    d = d[ok].copy()
    d["lv1"], d["lv7"], d["lv30"] = (np.log(d["vol_1d"]), np.log(d["vol_7d"]),
                                     np.log(d["vol_30d"]))
    d["y"] = np.log(d["fwd_vol"].where(d["fwd_vol"] > 0))
    return d.sort_values(["bar_time", "symbol"]).reset_index(drop=True)


def fit_vol(train: pd.DataFrame) -> dict:
    """HAR coefficients plus the level correction, from resolved labels only."""
    tr = train[train["y"].notna()]
    if len(tr) < 500:
        raise ValueError(f"only {len(tr)} resolved volatility labels, need 500")
    X = tr[HAR_FEATS].to_numpy(float)
    beta = _ols(X, tr["y"].to_numpy(float))
    k = float(tr["fwd_vol"].mean() / np.exp(_apply_ols(beta, X)).mean())
    # Terciles of this model's OWN forecasts over the training window, so "calm / normal / wild"
    # means "the bottom / middle / top third of what we have been forecasting for this universe
    # lately" - a like-for-like comparison. Terciles of realised vol would be the wrong yardstick:
    # a forecast is shrunk towards the mean, so it would almost never reach either tail.
    cuts = np.quantile(k * np.exp(_apply_ols(beta, X)), [1 / 3, 2 / 3])
    return {"beta": [float(v) for v in beta], "k": k,
            "calm_below": float(cuts[0]), "wild_above": float(cuts[1]),
            "n_train": int(len(tr))}


def predict_vol(fit: dict, rows: pd.DataFrame) -> np.ndarray:
    return fit["k"] * np.exp(_apply_ols(np.asarray(fit["beta"]), rows[HAR_FEATS].to_numpy(float)))


def vol_words(fit: dict, v: float) -> str:
    return "calm" if v < fit["calm_below"] else "wild" if v > fit["wild_above"] else "normal"


# --------------------------------------------------------------------- regime

def btc_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """BTC, one row per bar: the state, the one feature that forecasts it, and the forward label."""
    b = (panel[panel["symbol"] == "BTCUSDT"].sort_values("bar_time")
         .set_index("bar_time"))
    b = b[b["dist_sma200"].notna() & (b["vol_30d"] > 0)].copy()
    if b.empty:
        raise ValueError("no BTC bars with a 200-day average yet")
    b["state"] = b["dist_sma200"] > 0                    # == features.regime_on
    b["x"] = b["dist_sma200"] / b["vol_30d"]             # "how many sigmas to the flip"
    b["y"] = b["state"].shift(-HORIZON)                  # the state 7 days later
    run = b["state"].ne(b["state"].shift(1)).cumsum()
    b["state_since"] = b.index.to_series().groupby(run).transform("first")
    return b[["close", "state", "x", "y", "state_since", "vol_7d", "vol_30d", "dist_sma200"]]


def fit_regime(train: pd.DataFrame) -> dict:
    tr = train[train["y"].notna() & train["x"].notna()]
    if len(tr) < 500:
        raise ValueError(f"only {len(tr)} resolved regime labels, need 500")
    mu, sd = float(tr["x"].mean()), float(tr["x"].std())
    sd = sd if sd > 0 else 1.0
    beta = _logit((tr["x"].to_numpy(float) - mu) / sd, tr["y"].to_numpy(float))
    return {"beta": [float(v) for v in beta], "mu": mu, "sd": sd,
            "base_rate": float(tr["y"].mean()), "n_train": int(len(tr))}


def predict_regime(fit: dict, x: float) -> float:
    z = (float(x) - fit["mu"]) / fit["sd"]
    b = fit["beta"]
    return float(1.0 / (1.0 + np.exp(-np.clip(b[0] + b[1] * z, -30, 30))))


# --------------------------------------------------------------------- the forecast

def compute(settings, panel: pd.DataFrame, as_of=None) -> dict:
    """Refit both models on the trailing 730 days and forecast the newest closed bar.

    Train windows end at the newest RESOLVED label, which is 42 bars old, so this is exactly the
    information a live bot has. Raises if there is not enough history; the caller decides what a
    website should do about that.

    `as_of` pretends it is that bar and drops everything after it. It exists so the causality test
    can prove the forecast for a bar is identical whether or not later bars are present; the live
    path never passes it.
    """
    if as_of is not None:
        panel = panel[panel["bar_time"] <= pd.Timestamp(as_of)]
    v = vol_frame(panel)
    resolved = v.loc[v["y"].notna(), "bar_time"]
    if resolved.empty:
        raise ValueError("no resolved volatility labels in this panel")
    v_end = resolved.max()
    vfit = fit_vol(v[v["bar_time"].between(v_end - pd.Timedelta(days=TRAIN_DAYS), v_end)])

    last = v["bar_time"].max()
    cur = v[v["bar_time"] == last].sort_values("symbol").copy()
    cur["forecast"] = predict_vol(vfit, cur)

    per_coin = {}
    for r in cur.itertuples():
        per_coin[r.symbol] = {"vol_7d": float(r.vol_7d), "vol_30d": float(r.vol_30d),
                              "forecast_vol_7d": float(r.forecast),
                              "words": vol_words(vfit, float(r.forecast))}

    b = btc_frame(panel)
    r_end = b.loc[b["y"].notna()].index.max()
    rfit = fit_regime(b[(b.index > r_end - pd.Timedelta(days=TRAIN_DAYS)) & (b.index <= r_end)])
    now = b.iloc[-1]

    med = float(np.median([c["forecast_vol_7d"] for c in per_coin.values()])) if per_coin else None
    return {
        "as_of": str(last),
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "regime_on": bool(now["state"]),
        "regime_days_held": float((b.index[-1] - now["state_since"]).total_seconds() / 86400),
        "p_regime_on_7d": predict_regime(rfit, now["x"]),
        "btc_sigmas_from_flip": float(now["x"]),
        "btc_vol_7d_forecast": per_coin.get("BTCUSDT", {}).get("forecast_vol_7d"),
        "universe_vol_forecast": med,
        "universe_vol_words": vol_words(vfit, med) if med is not None else None,
        "per_coin": per_coin,
        "vol_fit": vfit,
        "regime_fit": rfit,
        # The newest label either fit was allowed to see. Must be at least 7 days before `as_of`,
        # which is the purge the research used and what tests/test_forecast.py asserts.
        "vol_train_end": str(v_end),
        "regime_train_end": str(r_end),
        "skill": dict(SKILL),
        "horizon_days": HORIZON / BARS_PER_DAY,
        "affects_trading": False,
    }


# --------------------------------------------------------------------- disk cache

def cache_path(settings):
    return settings.data_dir / "forecast.json"


def _age_hours(d: dict, now=None) -> float:
    """Hours since this forecast was computed. An unparseable stamp counts as infinitely old."""
    def utc(x):
        t = pd.Timestamp(x)
        return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
    try:
        ref = pd.Timestamp.now(tz="UTC") if now is None else utc(now)
        return max(0.0, (ref - utc(d["generated"])).total_seconds() / 3600)
    except Exception:
        return float("inf")


def read_cache(settings, now=None):
    """The saved forecast, with `age_hours` and `stale` filled in, or None if it is unusable.

    Never raises. A truncated or hand-edited file must leave the website working, so anything that
    does not parse into a dict with the fields the page reads is treated as "no cache".
    """
    try:
        d = json.loads(cache_path(settings).read_text())
        if not isinstance(d, dict) or "as_of" not in d or "regime_on" not in d:
            return None
        d["age_hours"] = _age_hours(d, now)
        d["stale"] = d["age_hours"] > MAX_AGE_HOURS
        d["from_cache"] = True
        return d
    except Exception:
        return None


def write_cache(settings, out: dict):
    p = cache_path(settings)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(out, indent=1, default=str))
    tmp.replace(p)                      # atomic, so a reader never sees half a file
    return p


def outlook(settings, panel: pd.DataFrame = None, max_age_hours=MAX_AGE_HOURS, now=None,
            force=False):
    """The cached market outlook, refitted at most once a day. Never raises.

    Returns the forecast dict (with `age_hours` and `stale`), or None if there is neither a usable
    cache nor enough data to build one. `panel` is optional only so a caller that already has one
    does not pay for a second build; omitting it builds one.
    """
    cached = read_cache(settings, now)
    if cached is not None and not force and cached["age_hours"] <= max_age_hours:
        return cached
    try:
        if panel is None:
            from . import features, universe
            u = universe.load(settings)
            panel = features.build_panel(settings.klines_dir, u["symbols"],
                                         settings.donchian_bars)
        out = compute(settings, panel)
        try:
            write_cache(settings, out)
        except Exception:
            pass                        # a read-only data dir is not a reason to show nothing
        out["age_hours"], out["stale"], out["from_cache"] = 0.0, False, False
        return out
    except Exception:
        return cached                   # a stale forecast, clearly labelled, beats a blank panel
