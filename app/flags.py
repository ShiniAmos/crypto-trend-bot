"""Feature flags for research.

Every strategy change starts life here, defaulting to OFF. With all flags off the bot must
reproduce the audited baseline bar for bar; `tests/test_flags.py` asserts that. Flags reach the
backtest through the CLI (`--enable-...`) and the live loop through `.env`, never through a
hardcoded change to the strategy.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class Flags:
    # P0-A: exit intrabar when the bar's low pierces the position's stop, instead of only
    # checking the close against the Donchian low. Fills at min(open, stop), stop before anything
    # else on that bar. The stop ratchets up with the 20-bar low and never moves down.
    stop_exec: bool = False

    # P1-E: the coin must be up on its own over 30 days, not merely beating a falling BTC.
    # Study: trades entered with ret_30d < 0 averaged -0.61% (26% win) against +4.37% (40% win),
    # in both halves of the history.
    abs_momentum: bool = False

    # Absolute momentum on BTC itself, on top of the 200-day regime: entries taken while BTC's
    # 30-day return was negative averaged +0.41% (29% win) against +4.2% to +5.6% otherwise.
    btc_momentum: bool = False

    # Skip a candidate that has already run too far: rs_7d above this (e.g. 0.35 = +35% against
    # BTC in a week) averaged -4.02% over 26 trades, negative in both halves. 0 disables it.
    rs_cap: float = 0.0

    # P1-G: market breadth. When fewer than BREADTH_LOW of the eligible alts are beating BTC over
    # 30 days, take no alt entries at all; BTC itself may still qualify on its own rules.
    breadth: bool = False

    # P1-H: the breakout must come with volume. volume_ratio is the last day's quote volume against
    # its own 20-day average, so 1.0 means "a normal day's turnover". 0 disables the filter.
    # A missing ratio counts as a failure, which is the conservative reading.
    volume_min: float = 0.0

    # P1-F: when the regime is risk-on but no alt qualifies after every other filter, hold BTC
    # instead of sitting in cash. Regime off means USDT only, as before. BTC is sized by the same
    # 1% formula and obeys every portfolio limit.
    btc_sleeve: bool = False

    # P2-I: take half off the table once the trade is working, then protect the rest at breakeven
    # plus fees. Two mutually exclusive triggers, measured separately; at most one may ship.
    #   scale_out_r   - close at +1.5R, where R is the distance from entry to the entry-time stop
    #   scale_out_pct - close at +8% from entry
    # Both trigger on a bar CLOSE and fill at the NEXT bar's open, exactly like every other exit in
    # this system, so no intrabar assumption is introduced and there is no OHLC ambiguity.
    scale_out_r: bool = False
    scale_out_pct: bool = False
    scale_out_frac: float = 0.5

    # P2-J: scale the position by how volatile the coin is against the rest of the universe that
    # bar. A 150%-volatility coin should not take the same weight as a calm one. The multiplier is
    # clipped to [0.5, 1.5] and every existing cap still applies afterwards. 0 disables it.
    vol_target: float = 0.0

    # P2-K: refuse a candidate whose 4h returns over the last CORR_BARS bars correlate above this
    # with anything already held. 0 disables it. Unmeasurable correlation allows the trade.
    corr_cap: float = 0.0

    # P3-M: the walk-forward model as a veto only, never as a replacement for the ranker.
    # A candidate must also carry a positive predicted excess return. Rows with no model yet
    # (before the first fold) fail closed.
    ml_veto: bool = False

    # P3-N: orphan metrics, one flag each, tested individually and never stacked.
    #   taker_min      - buying pressure on the breakout day, e.g. 0.5
    #   atr_band       - the coin's ATR% must sit between the universe's 20th and 80th percentile
    #   late_breakout  - reject a breakout at the very top of its channel on below-average volume
    taker_min: float = 0.0

    # P6: crowding. Skip a breakout whose 7-day funding sits above this percentile of the eligible
    # universe that bar - the trade is already crowded with leveraged longs. 0 disables it.
    # The first idea in this project to pass every panel plus a placebo null and 2x cost stress.
    # A coin with no perpetual contract has no funding and is never filtered.
    funding_max_pct: float = 0.0
    atr_band: bool = False
    late_breakout: bool = False

    # P4 (exits): everything measured so far changes WHICH trade is taken. These change HOW LONG
    # it is held, which is the dimension nothing has tested.
    #   exit_donchian      - use a wider low channel to exit than the 20 bars used to enter, so a
    #                        winner is given more room (classic asymmetric Donchian). 0 = off.
    #   exit_rank          - override the rank-drop exit (default 10). Higher holds longer.
    #   hold_through_regime- a BTC regime flip stops new entries but does not force an exit.
    #   time_stop_bars     - close a position still underwater after this many bars. 0 = off.
    # P5 (adaptive): the bot changes its own risk with conditions instead of running one size
    # everywhere. Both act on SIZE, never on which candidate is taken - the evidence is that
    # filtering removes the few huge winners the strategy lives on, while sizing cannot.
    #   regime_risk       - take less risk in the regimes that historically bleed (chop, recovery)
    #   equity_curve_risk - take less risk while the strategy's own recent trades are negative
    regime_risk: bool = False
    equity_curve_risk: bool = False
    equity_curve_lookback: int = 10

    exit_donchian: int = 0
    exit_rank: int = 0
    hold_through_regime: bool = False
    time_stop_bars: int = 0

    # ---------------------------------------------------------------- the trend core (P7)
    # Everything above this line decides WHICH ALTCOIN BREAKOUT to buy. Calendar-year attribution
    # (10 Oct 2026) showed why that is the wrong question now: survivorship-free, the alt-rotation
    # engine earns +2.24% a year over 2022-2026 and is invested 6.8% of the time, while simply
    # holding BTC whenever it is above its own 200-day average earns +31.12% a year, invested 51.8%
    # of the time, through the same costs.
    #
    # The core is a PERSISTENT position, not a candidate. It does not wait for a breakout, it does
    # not use a Donchian stop, and it does not compete for the breakout slots. It holds a coin for
    # as long as that coin is above its own 200-day average, and nothing else.
    #
    #   trend_core          fraction of equity the core may hold, 0.0 = off (the whole flag)
    #   trend_core_symbols  which coins the core may hold, comma separated
    trend_core: float = 0.0
    trend_core_symbols: str = "BTCUSDT,ETHUSDT"

    # P12 step 3: fixed capital slices. Without this the two engines compete for one pot of cash -
    # whichever queues first spends it - so a backtest cannot say how much return came from which,
    # and a good month for one engine quietly starves the other.
    #   capital_slices    on/off. The core is sized inside `trend_core` of equity; the breakout
    #                     engine is sized inside the remainder and capped at `slice_max_names`.
    #   slice_max_names   names the breakout engine may hold when slices are on (the brief: 2)
    capital_slices: bool = False
    slice_max_names: int = 2

    # P12 step 4: core risk. The core's drawdown is its weak point - pit maxDD -25.5% against the
    # breakout engine's -14.7% - because a weight-based size takes no account of how volatile the
    # coin is, nor of how stretched it already is above the line it would exit on.
    #   core_target_vol   w_i = min(target_vol / vol_30d_i, core_w_max). A coin at 36% annualised
    #                     vol gets half the weight of one at 18%.
    #   core_w_max        the per-coin ceiling that cap is clipped to
    #   core_max_stretch  refuse a NEW core entry more than this far above its 200-day average.
    #                     Entering 40% extended means the exit is 40% away; this is a separate
    #                     flag because it is a different idea from volatility.
    core_target_vol: float = 0.0
    core_w_max: float = 0.25
    core_max_stretch: float = 0.0

    # P14: hold a core coin only while it is above BOTH its 50-day and its 200-day average.
    # The 200-day line alone is slow to let go: on the single-asset runs it whipsawed XRP and BNB
    # 80 and 70 times at 5.0% and 8.6% win rates. Requiring the faster line as well is a slower
    # filter in effect - fewer, longer holds - which is the honest fix for whipsaw, as opposed to
    # refusing to realise losses, which merely hides it (see P14).
    core_dual_ma: bool = False

    @classmethod
    def from_args(cls, args):
        """Build from an argparse namespace: --enable-stop-exec -> stop_exec=True."""
        vals = {}
        for f in fields(cls):
            v = getattr(args, f.name, None)
            if v is not None:
                vals[f.name] = v
        return cls(**vals)

    @classmethod
    def from_env(cls, settings, account=None):
        """Live/paper flags, read from .env (ENABLE_STOP_EXEC=true).

        With `account`, an account-specific variable wins over the shared one:

            ENABLE_TREND_CORE=0            every book
            DEMO_ENABLE_TREND_CORE=0.25    the demo book only

        That is what lets the fleet run one configuration against another on the same bars - a
        treatment book and a control book - instead of changing the strategy for all three at once
        and having nothing to compare the result against.
        """
        import os
        vals = {}
        prefix = f"{account.upper()}_" if account else ""
        for f in fields(cls):
            raw = ""
            if prefix:
                raw = os.environ.get(prefix + "ENABLE_" + f.name.upper(), "")
            if raw == "":
                raw = os.environ.get("ENABLE_" + f.name.upper(), "")
            if raw != "":
                vals[f.name] = raw.strip().lower() in ("1", "true", "yes", "on") \
                    if f.type == "bool" or isinstance(f.default, bool) else type(f.default)(raw)
        return cls(**vals)

    @classmethod
    def add_cli(cls, parser):
        for f in fields(cls):
            name = "--enable-" + f.name.replace("_", "-")
            if isinstance(f.default, bool):
                parser.add_argument(name, dest=f.name, action="store_true", default=None)
            else:
                parser.add_argument(name, dest=f.name, type=type(f.default), default=None)

    @property
    def enabled(self) -> dict:
        """Only the flags that differ from the baseline, for logs and experiment records."""
        base = type(self)()
        return {k: v for k, v in asdict(self).items() if v != getattr(base, k)}

    @property
    def is_baseline(self) -> bool:
        return not self.enabled

    @property
    def label(self) -> str:
        return "baseline" if self.is_baseline else "+".join(sorted(self.enabled))

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]
