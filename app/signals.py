"""v1 rules (section 6): momentum rotation with a Bitcoin regime filter."""

import numpy as np
import pandas as pd

# P1-G breadth bands. Below LOW no alt may be entered; between LOW and HIGH and above HIGH the
# ordinary rules apply unchanged, so HIGH is documentation rather than a second gate.
BREADTH_LOW = 0.35
BREADTH_HIGH = 0.55


def breadth(df: pd.DataFrame) -> pd.Series:
    """Share of eligible non-BTC coins beating BTC over 30 days, per bar.

    Works on one bar or on a whole panel: the value is computed within each bar_time, never
    across bars, so it cannot see the future.
    """
    el = df["eligible"] & (df["symbol"] != "BTCUSDT")
    pos = el & (df["rs_30d"] > 0)
    if "bar_time" in df.columns:
        num = pos.groupby(df["bar_time"]).transform("sum")
        den = el.groupby(df["bar_time"]).transform("sum")
    else:
        num = pd.Series(pos.sum(), index=df.index)
        den = pd.Series(el.sum(), index=df.index)
    return num / den.replace(0, np.nan)


def extra_filters(b: pd.DataFrame, flags=None):
    """Optional entry filters (P1). With no flags set this is all-True, so the baseline is exact.

    Each one is a separate hypothesis, measured on its own before any two are combined.
    """
    ok = pd.Series(True, index=b.index)
    if flags is None:
        return ok
    if getattr(flags, "abs_momentum", False):
        ok &= b["ret_30d"] > 0                      # up on its own, not just versus a falling BTC
    if getattr(flags, "btc_momentum", False):
        ok &= b["btc_ret_30d"] > 0                  # BTC itself rising over 30 days
    if getattr(flags, "rs_cap", 0):
        ok &= b["rs_7d"] <= flags.rs_cap            # not already blown off
    if getattr(flags, "volume_min", 0):
        # A breakout on a quiet day is the classic false break. NaN (not enough history for the
        # 20-day average) fails the test rather than passing it by default.
        ok &= b["volume_ratio"].fillna(-1) >= flags.volume_min
    if getattr(flags, "funding_max_pct", 0):
        # NaN means no perpetual contract, which must pass: the rule cannot judge what it cannot see.
        crowded = (b["funding_pct"] > flags.funding_max_pct) if "funding_pct" in b.columns             else pd.Series(False, index=b.index)
        ok &= ~crowded.fillna(False)
    if getattr(flags, "taker_min", 0):
        ok &= b["taker_ratio"].fillna(-1) >= flags.taker_min
    if getattr(flags, "atr_band", False):
        # Skip the calmest and the wildest fifth of the universe on that bar.
        a = b["atr_pct"]
        el = b["eligible"] if "eligible" in b.columns else pd.Series(True, index=b.index)
        if "bar_time" in b.columns:
            lo = a.where(el).groupby(b["bar_time"]).transform(lambda x: x.quantile(0.2))
            hi = a.where(el).groupby(b["bar_time"]).transform(lambda x: x.quantile(0.8))
        else:
            lo = pd.Series(a.where(el).quantile(0.2), index=b.index)
            hi = pd.Series(a.where(el).quantile(0.8), index=b.index)
        ok &= a.between(lo, hi)
    if getattr(flags, "late_breakout", False):
        # At the very top of the channel and on a quiet day: the late, thin break.
        late = (b["don_pos"] > 0.95) & (b["volume_ratio"].fillna(0) < 1.0)
        ok &= ~late.fillna(False)
    if getattr(flags, "ml_veto", False):
        ok &= (b["ml_score"] > 0).fillna(False) if "ml_score" in b.columns else False
    if getattr(flags, "breadth", False):
        # Thin market: almost nothing is beating BTC, so take no alt at all. BTC keeps its own
        # right to qualify, which is what stops this from being a flat "stay in cash" rule.
        wide = breadth(b).fillna(0) >= BREADTH_LOW
        ok &= wide | (b["symbol"] == "BTCUSDT")
    return ok


def add_btc_sleeve(b: pd.DataFrame, ok: pd.Series, flags=None) -> pd.Series:
    """P1-F: on a risk-on bar where nothing else qualifies, let BTC itself be the candidate.

    Shared by the live loop and the backtest so both use exactly the same predicate. It only ever
    adds BTC, and only on bars that would otherwise buy nothing; it never removes a candidate and
    never fires while the regime is off.
    """
    if not getattr(flags, "btc_sleeve", False):
        return ok
    is_btc = b["symbol"] == "BTCUSDT"
    alt_ok = ok & ~is_btc
    if "bar_time" in b.columns:
        any_alt = alt_ok.groupby(b["bar_time"]).transform("any")
    else:
        any_alt = pd.Series(bool(alt_ok.any()), index=b.index)
    return ok | (~any_alt & is_btc & b["regime_on"].fillna(False) & b["eligible"].fillna(False))


def core_symbols(flags=None) -> list:
    """The coins the trend core may hold, or [] when the core is off.

    Returns [] whenever trend_core is 0, so every caller can treat "is the core on" and "which
    coins" as the same question and no caller has to check two fields.
    """
    if not getattr(flags, "trend_core", 0):
        return []
    raw = getattr(flags, "trend_core_symbols", "") or ""
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def core_on(b: pd.DataFrame, flags=None) -> pd.Series:
    """P7: True where a core coin closed above its OWN 200-day average.

    Note "its own". The breakout engine gates everything on BTC's 200-day average (`regime_on`);
    the core asks the question separately of each coin it holds, so ETH can be held while BTC is
    below its line and the other way round. The same predicate serves the backtest and the live
    loop, which is the only way the two cannot drift apart.

    This is the whole entry and exit rule for the core. There is no breakout, no Donchian stop and
    no rank: `dist_sma200 > 0` decided on this close, acted on at the next open - the same
    convention as every other decision in this engine.
    """
    syms = core_symbols(flags)
    if not syms:
        return pd.Series(False, index=b.index)
    on = (b["symbol"].isin(syms) & (b["dist_sma200"] > 0)
          & b["dist_sma200"].notna() & b["close"].notna())
    if getattr(flags, "core_dual_ma", False):
        # P14: also above its 50-day average. A coin below the faster line while still above the
        # slow one is the whipsaw zone - drifting sideways under a 200-day average that has not
        # caught up yet. NaN fails rather than passes: not enough history is not a buy signal.
        on &= (b["dist_sma50"] > 0).fillna(False) if "dist_sma50" in b.columns else False
    return on


def core_weight(flags, vol_30d=None):
    """P12 step 4: the per-coin core weight, after any volatility cap.

    Without `core_target_vol` this is simply the flag's own share. With it, a coin's weight is
    inversely proportional to its volatility - target / realised - so a wild coin is held smaller
    for the same risk. Clipped at `core_w_max` so the cap can only ever reduce, never lever.
    """
    base = float(getattr(flags, "trend_core", 0.0) or 0.0)
    tv = float(getattr(flags, "core_target_vol", 0.0) or 0.0)
    if not tv or vol_30d is None or not np.isfinite(vol_30d) or vol_30d <= 0:
        return base
    return float(min(tv / vol_30d, float(getattr(flags, "core_w_max", 0.25) or 0.25)))


def core_entry_allowed(flags, dist_sma200):
    """P12 step 4: refuse a NEW core entry that is already stretched far above its exit line.

    Only gates ENTRIES. A position already held is not closed for becoming extended - that would
    be a profit-taking rule, and taking profits early is the one change this repository has
    rejected most consistently.
    """
    m = float(getattr(flags, "core_max_stretch", 0.0) or 0.0)
    if not m:
        return True
    return bool(dist_sma200 is not None and np.isfinite(dist_sma200) and dist_sma200 <= m)


CORR_BARS = 180          # 30 days of 4h bars
CORR_MIN_OBS = 90        # below this the correlation is not measurable, and the trade is allowed


def vol_scale(b: pd.DataFrame, flags=None) -> pd.Series:
    """P2-J: size multiplier, the bar's median volatility over this coin's, clipped to [0.5, 1.5].

    A quiet coin gets up to 1.5x the risk-based size, a wild one as little as 0.5x. Every
    portfolio cap is applied afterwards, so this can never breach a limit.
    """
    if not getattr(flags, "vol_target", 0):
        return pd.Series(1.0, index=b.index)
    v = b["vol_30d"]
    el = b["eligible"] if "eligible" in b.columns else pd.Series(True, index=b.index)
    if "bar_time" in b.columns:
        med = v.where(el).groupby(b["bar_time"]).transform("median")
    else:
        med = pd.Series(v.where(el).median(), index=b.index)
    return (med / v).clip(0.5, 1.5).fillna(1.0)


REGIME_RISK = {"strong bull": 1.0, "weak bull": 1.0, "sideways/chop": 0.5,
               "recovery": 0.5, "bear": 0.5}


def regime_scale(panel, flags=None) -> pd.Series:
    """P5: size by the market the trade is born into.

    The regime table says this strategy earns its money in the two bull regimes and bleeds slowly
    in chop and recovery. This halves the risk taken in those, using the same causal labels the
    reports use - BTC's own position against its 200-day average and its trailing 90-day return.
    """
    if not getattr(flags, "regime_risk", False):
        return pd.Series(1.0, index=panel.index)
    from . import regime
    labels = regime.label_bars(panel)
    return panel["bar_time"].map(labels).map(REGIME_RISK).fillna(1.0)


def max_corr_with_held(wide: pd.DataFrame, end_label, symbol, held, bars=CORR_BARS,
                       min_obs=CORR_MIN_OBS) -> float:
    """P2-K: strongest correlation between this coin's 4h returns and anything already held.

    Returns 0.0 when it cannot be measured — too little overlapping history, or the coin is not in
    the frame. Allowing an unmeasurable pair is the deterministic choice: refusing it would
    silently bar every newly listed coin.
    """
    if symbol not in wide.columns or not held:
        return 0.0
    try:
        end = wide.index.get_loc(end_label)
    except KeyError:
        return 0.0
    win = wide.iloc[max(0, end - bars + 1):end + 1]
    a = win[symbol]
    best = 0.0
    for h in held:
        if h not in win.columns or h == symbol:
            continue
        pair = pd.concat([a, win[h]], axis=1).dropna()
        if len(pair) < min_obs:
            continue
        c = pair.iloc[:, 0].corr(pair.iloc[:, 1])
        if pd.notna(c):
            best = max(best, float(c))
    return best


def entry_candidates(bar: pd.DataFrame, settings, score_col="score", flags=None) -> pd.DataFrame:
    """Rows of one bar that qualify as entry candidates, best first.

    Entry = risk-on + close above the prior 20-bar high + strong relative strength vs BTC
    (positive 7d RS and ranked in the top CANDIDATES by score).
    """
    b = bar[bar["eligible"]]
    # Rank inside each bar. Passing a whole panel used to rank globally, which silently returned
    # almost nothing: only the highest-scoring rows in the entire frame could reach the top 5.
    if "bar_time" in b.columns and b["bar_time"].nunique() > 1:
        rank = b.groupby("bar_time")[score_col].rank(ascending=False, method="first")
    else:
        rank = b[score_col].rank(ascending=False, method="first")
    ok = b["regime_on"] & b["breakout"] & (b["rs_7d"] > 0) & (rank <= settings.candidates)
    # BTC itself can be held too; its RS is 0, so allow it on breakout when it ranks.
    ok |= (b["symbol"] == "BTCUSDT") & b["regime_on"] & b["breakout"] & (rank <= settings.candidates)
    ok &= extra_filters(b, flags)
    ok = add_btc_sleeve(b, ok, flags)
    # P7: a coin the trend core owns is reserved for it, exactly as in backtest.run. Both paths
    # read the list from core_symbols(), so the live loop and the backtest cannot disagree about
    # which coins the breakout engine is allowed to touch.
    core = core_symbols(flags)
    if core:
        ok &= ~b["symbol"].isin(core)
    return b[ok].assign(_r=rank[ok]).sort_values("_r").drop(columns="_r")


def exit_reason(symbol, regime_on, breakdown, rank, settings, flags=None, bars_held=None,
                in_profit=None):
    """Why a held position should close on this bar, or None to keep holding.

    With no flags this is the shipped ladder exactly. The flags change how long a winner is given:
    `hold_through_regime` stops a BTC flip from forcing an exit, `exit_rank` moves the rank-drop
    threshold, and `time_stop_bars` closes a position that is still underwater after N bars.
    The caller decides which breakdown column to pass, so a wider exit channel needs no change here.
    """
    if not regime_on and not getattr(flags, "hold_through_regime", False):
        return "BTC regime flipped off"
    if breakdown:
        return "broke the 20-bar low"
    limit = getattr(flags, "exit_rank", 0) or settings.exit_rank
    if rank is None or pd.isna(rank) or rank > limit:
        return f"fell out of the top {limit}"
    stop_bars = getattr(flags, "time_stop_bars", 0)
    # `not in_profit`, never `in_profit is False`: the caller passes a numpy bool, which is not the
    # Python False singleton, so the identity check silently disabled this rule entirely.
    if (stop_bars and bars_held is not None and bars_held >= stop_bars
            and in_profit is not None and not in_profit):
        return f"time stop: still negative after {stop_bars} bars"
    return None
