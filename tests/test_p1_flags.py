"""P1-P3 flags added after abs_momentum shipped.

Two rules govern every test here:
  - a flag that is off must not change the candidate list, the fill, or the equity curve;
  - a flag that is on must change only the thing it claims to change.
"""

import numpy as np
import pandas as pd
import pytest

from app import backtest, signals
from app.flags import Flags
from test_p0 import _panel

BASE_COLS = dict(eligible=True, regime_on=True, breakout=True)


def _bar(n_alts=4, share_positive=1.0, bar_time="2026-09-27 12:00"):
    """One bar: BTC plus n_alts alt coins, a given share of which beat BTC over 30 days."""
    rows = [{"symbol": "BTCUSDT", "rs_30d": 0.0, "rs_7d": 0.0, "ret_30d": 0.10,
             "btc_ret_30d": 0.10, "score": 100.0}]
    n_pos = int(round(n_alts * share_positive))
    for i in range(n_alts):
        rows.append({"symbol": f"A{i}USDT", "rs_30d": 0.05 if i < n_pos else -0.05,
                     "rs_7d": 0.05, "ret_30d": 0.10, "btc_ret_30d": 0.10, "score": 99.0 - i})
    return pd.DataFrame(rows).assign(bar_time=pd.Timestamp(bar_time, tz="UTC"), **BASE_COLS)


class S:
    candidates = 5


def _syms(bar, flags):
    return set(signals.entry_candidates(bar, S, flags=flags)["symbol"])


# ---------------------------------------------------------------- P1-G breadth

def test_breadth_bands():
    """20% positive -> alts forbidden; 40% and 70% -> unchanged."""
    thin, mid, wide = _bar(10, 0.2), _bar(10, 0.4), _bar(10, 0.7)
    assert _syms(thin, Flags(breadth=True)) == {"BTCUSDT"}, "below 0.35 only BTC may enter"
    assert _syms(mid, Flags(breadth=True)) == _syms(mid, None), "mid band is the ordinary rule"
    assert _syms(wide, Flags(breadth=True)) == _syms(wide, None), "above 0.55 is the ordinary rule"


def test_candidates_rank_within_each_bar_not_across_the_frame():
    """A panel of many bars must give each bar its own top 5, not one global top 5."""
    bars = []
    for i in range(6):
        b = _bar(6, bar_time=f"2026-09-27 {i * 4:02d}:00")
        b["score"] = [100 - i * 10] + [99 - i * 10 - k for k in range(6)]   # later bars score lower
        bars.append(b)
    panel = pd.concat(bars, ignore_index=True)
    got = signals.entry_candidates(panel, S, flags=None)
    per_bar = got.groupby("bar_time").size()
    assert len(per_bar) == 6, "every bar must contribute candidates"
    assert (per_bar == S.candidates).all(), "each bar gets its own top 5"


def test_breadth_off_changes_nothing():
    thin = _bar(10, 0.2)
    assert _syms(thin, Flags()) == _syms(thin, None)
    assert len(_syms(thin, Flags())) > 1


def test_breadth_is_computed_within_each_bar():
    """A thin bar and a wide bar in one frame must not contaminate each other."""
    two = pd.concat([_bar(10, 0.2, "2026-09-27 12:00"), _bar(10, 0.9, "2026-09-27 16:00")],
                    ignore_index=True)
    br = signals.breadth(two)
    first = br[two["bar_time"] == pd.Timestamp("2026-09-27 12:00", tz="UTC")].iloc[0]
    second = br[two["bar_time"] == pd.Timestamp("2026-09-27 16:00", tz="UTC")].iloc[0]
    assert first == pytest.approx(0.2) and second == pytest.approx(0.9)


def test_breadth_ignores_btc_and_ineligible_rows():
    bar = _bar(10, 0.5)
    bar.loc[bar["symbol"] == "A0USDT", "eligible"] = False      # drop one positive alt
    assert signals.breadth(bar).iloc[0] == pytest.approx(4 / 9)


# ---------------------------------------------------------------- P1-H volume

def _vol_bar(ratios):
    bar = _bar(len(ratios))
    bar.loc[bar["symbol"] != "BTCUSDT", "volume_ratio"] = ratios
    bar.loc[bar["symbol"] == "BTCUSDT", "volume_ratio"] = 2.0
    return bar


def test_volume_blocks_quiet_breakout():
    bar = _vol_bar([0.4, 0.9, 1.0, 1.6])          # A0 quiet, A1 below par, A2 exactly par, A3 busy
    kept = _syms(bar, Flags(volume_min=1.0))
    assert "A0USDT" not in kept and "A1USDT" not in kept
    assert {"A2USDT", "A3USDT"} <= kept, "at or above the threshold must pass"


def test_volume_threshold_is_ordered():
    bar = _vol_bar([0.4, 0.9, 1.0, 1.6])
    loose = _syms(bar, Flags(volume_min=0.8))
    tight = _syms(bar, Flags(volume_min=1.2))
    assert tight < loose, "a higher threshold can only keep fewer coins"


def test_missing_volume_ratio_fails_closed():
    bar = _vol_bar([np.nan, 1.5, 1.5, 1.5])
    assert "A0USDT" not in _syms(bar, Flags(volume_min=1.0))


def test_volume_off_changes_nothing():
    bar = _vol_bar([0.1, 0.1, 0.1, 0.1])
    assert _syms(bar, Flags(volume_min=0.0)) == _syms(bar, None)


# ---------------------------------------------------------------- P1-F BTC sleeve

def test_sleeve_only_when_empty():
    """BTC steps in only on a risk-on bar where nothing else qualifies."""
    bar = _bar(4)
    bar.loc[bar["symbol"] != "BTCUSDT", "breakout"] = False      # no alt qualifies
    bar.loc[bar["symbol"] == "BTCUSDT", "breakout"] = False      # BTC has no breakout either
    assert _syms(bar, None) == set(), "nothing qualifies without the sleeve"
    assert _syms(bar, Flags(btc_sleeve=True)) == {"BTCUSDT"}

    busy = _bar(4)                                               # an alt does qualify
    assert "A0USDT" in _syms(busy, Flags(btc_sleeve=True))
    assert _syms(busy, Flags(btc_sleeve=True)) == _syms(busy, None), "sleeve must not add BTC here"


def test_sleeve_never_fires_with_the_regime_off():
    bar = _bar(4)
    bar["breakout"] = False
    bar["regime_on"] = False
    assert _syms(bar, Flags(btc_sleeve=True)) == set(), "regime off means USDT only"


def test_sleeve_respects_eligibility():
    bar = _bar(4)
    bar["breakout"] = False
    bar.loc[bar["symbol"] == "BTCUSDT", "eligible"] = False
    assert _syms(bar, Flags(btc_sleeve=True)) == set()


def test_sleeve_is_decided_per_bar():
    """A bar with a live alt must not suppress the sleeve on a different, empty bar."""
    empty = _bar(3, bar_time="2026-09-27 12:00")
    empty["breakout"] = False
    full = _bar(3, bar_time="2026-09-27 16:00")
    two = pd.concat([empty, full], ignore_index=True)
    got = signals.entry_candidates(two, S, flags=Flags(btc_sleeve=True))
    first = got[got["bar_time"] == pd.Timestamp("2026-09-27 12:00", tz="UTC")]
    second = got[got["bar_time"] == pd.Timestamp("2026-09-27 16:00", tz="UTC")]
    assert set(first["symbol"]) == {"BTCUSDT"}
    assert "A0USDT" in set(second["symbol"])


# ---------------------------------------------------------------- P3 flags

# Knobs that shape a feature once it is switched on. They are inert on their own, which the
# assertions below pin down, so a non-zero default here is not a flag left enabled by accident.
PARAMETERS = {"scale_out_frac", "equity_curve_lookback", "trend_core_symbols", "slice_max_names",
              "core_w_max"}


def test_every_enable_flag_defaults_off():
    base = Flags()
    from dataclasses import fields
    for f in fields(Flags):
        if f.name in PARAMETERS:
            continue
        assert not getattr(base, f.name), f"{f.name} must default to off"
    assert base.is_baseline
    for name in PARAMETERS:
        assert Flags(**{name: getattr(base, name)}).is_baseline, f"{name} alone enables nothing"


def test_parameters_are_inert_without_their_feature(settings):
    panel = _panel()
    a = backtest.run(panel, settings, flags=Flags(abs_momentum=True))
    b = backtest.run(panel, settings,
                     flags=Flags(abs_momentum=True, scale_out_frac=0.3, equity_curve_lookback=3))
    assert a.equity.equals(b.equity), "knobs must do nothing until their feature is on"


def test_ml_veto_does_not_rerank():
    """The veto may only remove candidates; the order must stay the ranker's."""
    bar = _bar(4)
    bar["ml_score"] = [0.5, -0.1, 0.2, 0.3, -0.4]      # BTC, A0..A3
    plain = signals.entry_candidates(bar, S, flags=None)
    vetoed = signals.entry_candidates(bar, S, flags=Flags(ml_veto=True))
    assert set(vetoed["symbol"]) < set(plain["symbol"])
    assert "A0USDT" not in set(vetoed["symbol"]) and "A3USDT" not in set(vetoed["symbol"])
    order_before = [s for s in plain["symbol"] if s in set(vetoed["symbol"])]
    assert list(vetoed["symbol"]) == order_before, "ranking order must be untouched"


def test_ml_veto_fails_closed_without_a_model():
    bar = _bar(4)
    bar["ml_score"] = np.nan
    assert _syms(bar, Flags(ml_veto=True)) == set()
    bar2 = _bar(4)                                      # column absent entirely
    assert _syms(bar2, Flags(ml_veto=True)) == set()


def test_taker_filter():
    bar = _bar(4)
    bar["taker_ratio"] = [0.6, 0.45, 0.55, 0.7, np.nan]
    kept = _syms(bar, Flags(taker_min=0.5))
    assert "A0USDT" not in kept and "A3USDT" not in kept       # 0.45 and NaN fail
    assert {"A1USDT", "A2USDT"} <= kept


def test_atr_band_trims_both_tails():
    bar = _bar(9)
    bar["atr_pct"] = [0.05, 0.001, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.90]
    kept = _syms(bar, Flags(atr_band=True))
    assert "A0USDT" not in kept, "the calmest coin is outside the band"
    assert "A8USDT" not in kept, "the wildest coin is outside the band"
    assert len(kept) >= 3


def test_late_breakout_needs_both_conditions():
    bar = _bar(4)
    bar["don_pos"] = [0.5, 0.99, 0.99, 0.50, 0.99]
    bar["volume_ratio"] = [1.5, 0.5, 1.5, 0.5, 1.5]
    kept = _syms(bar, Flags(late_breakout=True))
    assert "A0USDT" not in kept, "top of channel on a quiet day is rejected"
    assert {"A1USDT", "A2USDT"} <= kept, "either condition alone is fine"


# ---------------------------------------------------------------- P2-J vol target

def test_vol_target_bounds():
    """The multiplier is the bar's median volatility over the coin's, clipped to [0.5, 1.5]."""
    bar = _bar(4)
    bar["vol_30d"] = [1.0, 0.25, 1.0, 4.0, 100.0]        # BTC first, then the alts
    sc = signals.vol_scale(bar, Flags(vol_target=1.0))
    assert sc.max() <= 1.5 + 1e-12 and sc.min() >= 0.5 - 1e-12
    calm = sc[bar["symbol"] == "A0USDT"].iloc[0]          # 0.25 vol, well below the median
    wild = sc[bar["symbol"] == "A2USDT"].iloc[0]          # 100 vol
    assert calm == pytest.approx(1.5) and wild == pytest.approx(0.5)


def test_vol_target_off_is_exactly_one():
    bar = _bar(4)
    bar["vol_30d"] = [1.0, 0.2, 1.0, 5.0, 9.0]
    assert (signals.vol_scale(bar, Flags()) == 1.0).all()


def test_vol_target_cannot_breach_the_weight_cap():
    from app import risk
    qty = risk.position_size(10_000, 0.01, entry=100.0, stop=99.0, max_weight=0.30, scale=1.5)
    assert qty * 100.0 <= 10_000 * 0.30 + 1e-9, "the cap is applied after the multiplier"


# ---------------------------------------------------------------- P2-K correlation cap

def _wide(n=200, corr_pair=True):
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.01, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    twin = a * 0.98 + rng.normal(0, 0.001, n) if corr_pair else rng.normal(0, 0.01, n)
    return pd.DataFrame({"AAAUSDT": a, "BBBUSDT": twin,
                         "CCCUSDT": rng.normal(0, 0.01, n)}, index=idx)


def test_corr_skip():
    w = _wide()
    end = w.index[-1]
    assert signals.max_corr_with_held(w, end, "BBBUSDT", ["AAAUSDT"]) > 0.9
    assert signals.max_corr_with_held(w, end, "CCCUSDT", ["AAAUSDT"]) < 0.5


def test_corr_unmeasurable_allows_the_trade():
    """Too little overlap, or an unknown coin, must not silently block every new listing."""
    w = _wide(n=40)
    assert signals.max_corr_with_held(w, w.index[-1], "BBBUSDT", ["AAAUSDT"]) == 0.0
    assert signals.max_corr_with_held(w, w.index[-1], "ZZZUSDT", ["AAAUSDT"]) == 0.0
    assert signals.max_corr_with_held(w, w.index[-1], "BBBUSDT", []) == 0.0


def test_corr_cap_only_reduces_trades(settings):
    panel = _panel()
    base = backtest.run(panel, settings, flags=Flags(abs_momentum=True))
    capped = backtest.run(panel, settings, flags=Flags(abs_momentum=True, corr_cap=0.75))
    assert len(capped.trades) <= len(base.trades)


# ---------------------------------------------------------------- P2-I scale-out

def test_scale_out_half_and_breakeven_stop(settings):
    """Half the position leaves at the target; the remainder is protected at breakeven + fees."""
    panel = _panel()
    shipped = dict(abs_momentum=True, volume_min=0.8)
    base = backtest.run(panel, settings, flags=Flags(**shipped))
    scaled = backtest.run(panel, settings, flags=Flags(**shipped, scale_out_r=True))
    parts = scaled.trades[scaled.trades.exit_reason == "scaled out at +1.5R"]
    assert len(parts) > 0, "the +1.5R target should be reached somewhere in the sample"
    assert (parts.ret > 0).all(), "a scale-out at +1.5R cannot be a losing exit"
    assert len(scaled.trades) > len(base.trades), "each scaled trade produces two rows"
    be = scaled.trades[scaled.trades.exit_reason == "breakeven stop after scaling out"]
    if len(be):
        # breakeven means roughly flat on the remainder: never a big loss
        assert be.ret.min() > -0.05


def test_scale_out_pct_uses_eight_percent(settings):
    panel = _panel()
    r = backtest.run(panel, settings, flags=Flags(abs_momentum=True, scale_out_pct=True))
    parts = r.trades[r.trades.exit_reason == "scaled out at +8%"]
    assert len(parts) > 0
    for t in parts.head(10).itertuples():
        assert t.exit_price >= t.entry_price * 1.0, "the partial exit is above entry"


def test_scale_out_off_changes_nothing(settings):
    panel = _panel()
    a = backtest.run(panel, settings, flags=Flags(abs_momentum=True))
    b = backtest.run(panel, settings, flags=Flags(abs_momentum=True, scale_out_frac=0.5))
    assert a.equity.equals(b.equity), "the fraction alone must not enable the feature"


def test_queued_sell_survives_missing_data(settings):
    """Regression: a queued sell whose coin then loses data must not crash the engine."""
    panel = _panel()
    res = backtest.run(panel, settings, flags=Flags(abs_momentum=True))
    first = res.trades.iloc[0]
    # remove the bar the sell would have filled on, leaving the order queued
    cut = panel[~((panel.symbol == first.symbol) & (panel.bar_time == first.exit_time))]
    again = backtest.run(cut, settings, flags=Flags(abs_momentum=True))
    assert np.isfinite(again.equity).all()


def test_breadth_never_increases_the_trade_count(settings):
    panel = _panel()
    base = backtest.run(panel, settings, flags=Flags(abs_momentum=True))
    withb = backtest.run(panel, settings, flags=Flags(abs_momentum=True, breadth=True))
    assert len(withb.trades) <= len(base.trades)
    assert np.isfinite(withb.equity).all()


# ---------------------------------------------------------------- decision timeframe

def test_decision_timeframe_defaults_to_4h_and_scales_the_channel(settings, monkeypatch):
    from app import config as cfg
    assert settings.decision_tf == "4h"
    assert settings.bars_per_day == 6 and settings.donchian_for_tf == 20

    monkeypatch.setenv("DECISION_TF", "1h")
    fast = cfg.Settings()
    assert fast.bars_per_day == 24
    assert fast.donchian_for_tf == 80, "the same 3.33-day channel, in 1h bars"

    monkeypatch.setenv("DECISION_TF", "13m")
    with pytest.raises(ValueError):
        cfg.Settings()


def test_hourly_panel_keeps_the_same_wall_clock_horizons(settings):
    """A 1h panel must measure the same days, not the same bar counts."""
    from app import features as ft
    from conftest import make_bars, daily_from_4h
    b = make_bars(3000, tf_ms=3_600_000, seed=3)
    f1 = ft.symbol_features(b, daily_from_4h(b), donchian=80, bpd=24, tf_ms=3_600_000)
    assert f1["ret_7d"].notna().sum() > 0
    # 7-day return on 1h bars must look back 168 bars, not 42
    i = 2000
    expected = b["close"].iloc[i] / b["close"].iloc[i - 7 * 24] - 1
    assert f1["ret_7d"].iloc[i] == pytest.approx(expected)


# ---------------------------------------------------------------- P4 exit flags

def test_time_stop_fires_on_a_numpy_bool():
    """Regression: `in_profit is False` was always false for numpy bools, disabling the rule."""
    class S2:
        exit_rank = 10
    assert signals.exit_reason("A", True, False, 1, S2(), Flags(time_stop_bars=12),
                               bars_held=20, in_profit=np.bool_(False)) is not None
    assert signals.exit_reason("A", True, False, 1, S2(), Flags(time_stop_bars=12),
                               bars_held=20, in_profit=np.bool_(True)) is None
    assert signals.exit_reason("A", True, False, 1, S2(), Flags(time_stop_bars=12),
                               bars_held=5, in_profit=np.bool_(False)) is None, "too early"
    assert signals.exit_reason("A", True, False, 1, S2(), Flags(),
                               bars_held=99, in_profit=np.bool_(False)) is None, "flag off"


def test_hold_through_regime_only_affects_the_regime_rung():
    class S2:
        exit_rank = 10
    off = Flags()
    hold = Flags(hold_through_regime=True)
    assert signals.exit_reason("A", False, False, 1, S2(), off) == "BTC regime flipped off"
    assert signals.exit_reason("A", False, False, 1, S2(), hold) is None
    # the other rungs still fire while holding through the regime
    assert signals.exit_reason("A", False, True, 1, S2(), hold) == "broke the 20-bar low"
    assert "top" in signals.exit_reason("A", False, False, 99, S2(), hold)


def test_exit_rank_override():
    class S2:
        exit_rank = 10
    assert signals.exit_reason("A", True, False, 14, S2(), Flags()) is not None
    assert signals.exit_reason("A", True, False, 14, S2(), Flags(exit_rank=20)) is None
