"""Phase 5: bar-by-bar simulation of the whole basket, with fees, slippage and risk limits.

Timing: a decision is made on the close of bar t using only bars closed by then, and fills at the
open of bar t+1 (plus slippage and fee). No lookahead.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import risk, signals

BARS_PER_YEAR = 6 * 365
ONE_MS = pd.Timedelta(milliseconds=1)


@dataclass
class Result:
    equity: pd.Series
    trades: pd.DataFrame          # closed round trips
    log: pd.DataFrame             # every fill and risk event, in order
    exposure: pd.Series
    kills: int = 0
    halts: int = 0
    # Cumulative P&L per engine, one column each ("core", "breakout"), indexed like `equity`.
    # Deliberately a P&L curve starting at zero rather than an equity curve: the two engines share
    # one pot of cash, so any split of the STARTING capital between them would be invented. What is
    # real is how much each has made or lost, realised plus open, and that is enough to measure an
    # engine's own drawdown separately from the book's.
    engine_pnl: pd.DataFrame = field(default_factory=pd.DataFrame)
    meta: dict = field(default_factory=dict)


def _utc(x):
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _rank(panel, score_col):
    if score_col == "score":
        return panel["rank"].to_numpy()
    el = panel["eligible"] & panel[score_col].notna()
    r = pd.Series(np.nan, index=panel.index)
    r[el] = panel[el].groupby("bar_time")[score_col].rank(ascending=False, method="first")
    return r.to_numpy()


def run(panel, settings, score_col="score", start=None, end=None, kill_cooldown_days=7,
        max_names=None, never_sell_at_loss=False, trail_pct=None, flags=None,
        size_scale_col=None):
    """Variants for testing (both rejected, see the README): `never_sell_at_loss=True` holds a
    losing position; `trail_pct=0.03` also sells when a coin gives back 3% from its best close.

    `flags` carries the research feature flags (app/flags.py). With the default flags this function
    behaves exactly as the audited baseline.
    """
    from .flags import Flags
    flags = flags or Flags()

    p = panel.sort_values(["bar_time", "symbol"]).reset_index(drop=True)
    rank = _rank(p, score_col)
    btc = (p["symbol"] == "BTCUSDT").to_numpy()
    entry_ok = (p["eligible"] & p["regime_on"] & p["breakout"]).to_numpy() \
        & ((p["rs_7d"] > 0).to_numpy() | btc) & (rank <= settings.candidates)
    entry_ok &= signals.extra_filters(p, flags).to_numpy()      # P1 filters, all off by default
    # The sleeve is the same predicate the live loop uses, so the two cannot drift apart.
    entry_ok = signals.add_btc_sleeve(p, pd.Series(entry_ok, index=p.index), flags).to_numpy()
    # P7: a coin the trend core owns is reserved for the core. If the breakout engine could also
    # buy it, the two would fight over one position - the core wanting to hold while the breakout
    # rule wants out on a rank drop - and whichever queued its order last would win.
    core_syms = signals.core_symbols(flags)
    if core_syms:
        entry_ok &= ~p["symbol"].isin(core_syms).to_numpy()
    core_ok = signals.core_on(p, flags).to_numpy()
    if start is not None:
        entry_ok &= (p["bar_time"] >= _utc(start)).to_numpy()
    if end is not None:
        keep = (p["bar_time"] <= _utc(end)).to_numpy()
        p, rank, entry_ok, btc = p[keep].reset_index(drop=True), rank[keep], entry_ok[keep], btc[keep]
        core_ok = core_ok[keep]

    times = p["bar_time"].to_numpy()
    bounds = np.flatnonzero(np.r_[True, times[1:] != times[:-1], True])
    sym = p["symbol"].to_numpy()
    o, c = p["open"].to_numpy(), p["close"].to_numpy()
    lo = p["low"].to_numpy()
    stop_lvl = p["stop_level"].to_numpy()
    regime = p["regime_on"].to_numpy()
    # A wider exit channel only changes which column the exit reads (P4).
    breakdown = (p["breakdown_exit"] if flags.exit_donchian and "breakdown_exit" in p.columns
                 else p["breakdown"]).to_numpy()
    start_ts = _utc(start) if start is not None else None

    # P2-J / P2-K: prepared once, so a per-bar decision costs a slice and not a rebuild.
    scale_all = (signals.vol_scale(p, flags) * signals.regime_scale(p, flags)).to_numpy()
    recent_rets = []            # P5 equity-curve risk: the strategy's own last closed trades
    if size_scale_col:          # research only: size each entry by a supplied multiplier column
        scale_all = scale_all * p[size_scale_col].fillna(1.0).clip(0.5, 1.5).to_numpy()
    wide_rets = (p.pivot_table(index="bar_time", columns="symbol", values="ret_4h")
                 if flags.corr_cap else None)

    max_names = max_names or settings.max_names
    min_notional = 5.0      # Binance's smallest spot order, in USDT
    slip, fee = settings.slippage, settings.fee_rate
    cash = settings.start_equity
    pos = {}            # sym -> dict(qty, entry_price, entry_time, cost, reason, last_close)
    pending = []        # (side, sym, qty, reason)
    missing = {}        # sym -> consecutive bars with no data while held
    trades, log, eq_t, eq_v, expo = [], [], [], [], []
    realised = {"core": 0.0, "breakout": 0.0}     # per-engine, closed trades only
    eng_rows = []                                 # per-bar cumulative P&L, realised + open
    state = None
    kills = halts = 0

    for bi in range(len(bounds) - 1):
        a, b = bounds[bi], bounds[bi + 1]
        t = pd.Timestamp(times[a])
        if start_ts is not None and t < start_ts:
            continue
        here = {sym[j]: j for j in range(a, b)}
        day = str((t - ONE_MS).date())

        # 1) fill yesterday's decisions at this bar's open: sells first, then buys
        still = []
        for side, s, qty, reason, stop_at_entry in sorted(pending, key=lambda x: x[0] != "SELL"):
            j = here.get(s)
            if side == "SELL_PART":
                # P2-I: sell part of the position, leave the rest open and lift its stop to
                # breakeven plus the round trip's fees.
                p0 = pos.get(s)
                if j is None or p0 is None:
                    continue
                px = o[j] * (1 - slip)
                part = min(qty, p0["qty"])
                share = part / p0["qty"]
                cost_part = p0["cost"] * share
                proceeds = part * px * (1 - fee)
                cash += proceeds
                trades.append({"symbol": s, "entry_time": p0["entry_time"], "exit_time": t,
                               "entry_price": p0["entry_price"], "exit_price": px,
                               "qty": part, "pnl": proceeds - cost_part,
                               "ret": proceeds / cost_part - 1,
                               "bars_held": p0["bars"], "entry_reason": p0["reason"],
                               "exit_reason": reason})
                log.append({"time": t, "symbol": s, "side": "SELL", "qty": part, "price": px,
                            "fee": part * px * fee, "reason": reason})
                realised["core" if p0.get("engine") == "core" else "breakout"] += (
                    proceeds - cost_part)
                p0["qty"] -= part
                p0["cost"] -= cost_part
                p0["scaled"] = True
                p0["stop"] = p0["entry_price"] * (1 + 2 * fee)
                if p0["qty"] <= 0:
                    pos.pop(s)
                continue
            if side == "SELL":
                if j is None:
                    if missing.get(s, 0) < 6:
                        still.append((side, s, qty, reason, stop_at_entry))
                        continue
                    px, reason = pos[s]["last_close"], reason + " (no data: closed at last price)"
                else:
                    px = o[j] * (1 - slip)
                p0 = pos.pop(s)
                proceeds = p0["qty"] * px * (1 - fee)
                cash += proceeds
                trades.append({"symbol": s, "entry_time": p0["entry_time"], "exit_time": t,
                               "entry_price": p0["entry_price"], "exit_price": px,
                               "qty": p0["qty"], "pnl": proceeds - p0["cost"],
                               "ret": proceeds / p0["cost"] - 1,
                               "bars_held": p0["bars"], "entry_reason": p0["reason"],
                               "exit_reason": reason})
                realised["core" if p0.get("engine") == "core" else "breakout"] += (
                    proceeds - p0["cost"])
                recent_rets.append(proceeds / p0["cost"] - 1)
                log.append({"time": t, "symbol": s, "side": "SELL", "qty": p0["qty"], "price": px,
                            "fee": p0["qty"] * px * fee, "reason": reason})
            else:
                if j is None or s in pos:
                    continue
                px = o[j] * (1 + slip)
                qty = min(qty, cash / (px * (1 + fee)))
                if qty * px < min_notional:
                    log.append({"time": t, "symbol": s, "side": "SKIP", "qty": 0, "price": px,
                                "fee": 0, "reason": "not enough cash / below minimum order"})
                    continue
                cost = qty * px * (1 + fee)
                cash -= cost
                pos[s] = {"qty": qty, "entry_price": px, "entry_time": t, "cost": cost,
                          "reason": reason, "last_close": c[j], "peak": c[j], "bars": 0,
                          "stop": stop_at_entry, "scaled": False,
                          # The engine that owns this position, in the same vocabulary the live
                          # book stores on its `engine` column, so a backtest position and a paper
                          # position cannot be described differently. A core coin is excluded from
                          # entry_ok, so any fill in one belongs to the core.
                          "engine": "core" if s in core_syms else "breakout",
                          "r_unit": max(px - (stop_at_entry or 0), 0.0)}
                log.append({"time": t, "symbol": s, "side": "BUY", "qty": qty, "price": px,
                            "fee": qty * px * fee, "reason": reason})
        pending = still

        # 1b) P0-A: the stop is an intrabar event, checked before anything else on this bar.
        # If the low pierced the stop we are out at min(open, stop): a gap below the stop fills at
        # the open, never at the (better) stop price. A position bought at this open can be stopped
        # in the same bar, which is the conservative reading of an OHLC bar.
        if flags.stop_exec:
            for s in [s for s, p0 in pos.items() if p0.get("stop")]:
                j = here.get(s)
                if j is None or lo[j] > pos[s]["stop"]:
                    continue
                px = min(o[j], pos[s]["stop"]) * (1 - slip)
                p0 = pos.pop(s)
                proceeds = p0["qty"] * px * (1 - fee)
                cash += proceeds
                reason = "stop hit intrabar"
                trades.append({"symbol": s, "entry_time": p0["entry_time"], "exit_time": t,
                               "entry_price": p0["entry_price"], "exit_price": px,
                               "qty": p0["qty"], "pnl": proceeds - p0["cost"],
                               "ret": proceeds / p0["cost"] - 1,
                               "bars_held": p0["bars"], "entry_reason": p0["reason"],
                               "exit_reason": reason})
                realised["core" if p0.get("engine") == "core" else "breakout"] += (
                    proceeds - p0["cost"])
                recent_rets.append(proceeds / p0["cost"] - 1)
                log.append({"time": t, "symbol": s, "side": "SELL", "qty": p0["qty"], "price": px,
                            "fee": p0["qty"] * px * fee, "reason": reason})
                pending = [x for x in pending if not (x[0] == "SELL" and x[1] == s)]

        # 2) mark to market at the close
        for s, p0 in pos.items():
            j = here.get(s)
            if j is not None:
                p0["last_close"] = c[j]
                p0["peak"] = max(p0.get("peak", 0), c[j])
                # The stop ratchets up with the 20-bar low and never moves down (P0-A).
                if flags.stop_exec and p0.get("stop") is not None and not np.isnan(stop_lvl[j]):
                    p0["stop"] = max(p0["stop"], stop_lvl[j])
                missing.pop(s, None)
            else:
                missing[s] = missing.get(s, 0) + 1
            p0["bars"] += 1
        held = sum(p0["qty"] * p0["last_close"] for p0 in pos.values())
        equity = cash + held
        eq_t.append(t)
        eq_v.append(equity)
        expo.append(held / equity if equity > 0 else 0)
        # open P&L per engine: market value now against what was paid for it
        open_pnl = {"core": 0.0, "breakout": 0.0}
        for p0 in pos.values():
            e = "core" if p0.get("engine") == "core" else "breakout"
            open_pnl[e] += p0["qty"] * p0["last_close"] - p0["cost"]
        eng_rows.append({"core": realised["core"] + open_pnl["core"],
                         "breakout": realised["breakout"] + open_pnl["breakout"]})
        for row in log[::-1]:
            if row["time"] != t:
                break
            row["equity"] = equity

        # 3) risk limits
        if state is None:
            state = risk.RiskState(peak_equity=equity, day=day, day_start_equity=equity)
        if state.killed and (pd.Timestamp(day) - pd.Timestamp(state.killed_day)).days >= kill_cooldown_days:
            state.killed, state.peak_equity = False, equity
            log.append({"time": t, "symbol": "", "side": "EVENT", "qty": 0, "price": 0, "fee": 0,
                        "reason": f"kill re-armed after {kill_cooldown_days}-day pause", "equity": equity})
        for ev in risk.update_limits(state, equity, day, settings):
            kills += "kill" in ev
            halts += "daily" in ev
            log.append({"time": t, "symbol": "", "side": "EVENT", "qty": 0, "price": 0, "fee": 0,
                        "reason": ev, "equity": equity})

        # 4) decide on this close; orders fill on the next bar's open
        # Star-unpack: pending rows carry the stop as well, and a queued sell that could not fill
        # (its data vanished) is still sitting here. A fixed-width unpack crashed on exactly that.
        selling = {s for side, s, *_ in pending if side == "SELL"}
        scaling = {s for side, s, *_ in pending if side == "SELL_PART"}

        # 4a) P7 the trend core, decided before the breakout engine so the cash it frees on an exit
        # is available to the breakout buys queued on the same bar (sells fill first).
        #
        # The core's whole rule: hold the coin while it is above its own 200-day average. There is
        # no stop, no rank test and no breakout - those belong to the other engine. The drawdown
        # kill still applies, because the kill is a property of the book and not of a strategy.
        if core_syms:
            on_now = {s2 for s2 in core_syms
                      if here.get(s2) is not None and core_ok[here[s2]]}
            for s2 in core_syms:
                j2, p0 = here.get(s2), pos.get(s2)
                core_held = p0 is not None and p0.get("engine") == "core"
                want = bool(s2 in on_now) and not state.killed
                if core_held and not want and s2 not in selling:
                    why = "drawdown kill" if state.killed else "core: closed below its 200-day avg"
                    pending.append(("SELL", s2, p0["qty"], why, None))
                    selling.add(s2)
                elif want and p0 is None and s2 not in {x[1] for x in pending if x[0] == "BUY"}:
                    # P12 step 4: refuse an entry that is already stretched far above the line it
                    # would exit on. Entries only - a held position is never closed for this.
                    if not signals.core_entry_allowed(
                            flags, float(p["dist_sma200"].to_numpy()[j2])
                            if "dist_sma200" in p.columns else None):
                        continue
                    # Equal split of the core allocation across the coins currently above the line.
                    #
                    # `trend_core` IS the cap, and settings.max_weight deliberately does not apply.
                    # max_weight is a concentration limit for the rotation, where the engine picks
                    # the name and a 30% bet on whatever broke out this week needs bounding. The
                    # core's holding is not a pick - it is the stated size of a deliberate BTC
                    # allocation - so capping it at max_weight only produced the symptom that
                    # trend_core=0.50 and 0.75 were identical rows, both silently clipped to 0.30.
                    # P12 step 4: the volatility cap replaces the flat weight when it is on.
                    w = signals.core_weight(
                        flags, float(p["vol_30d"].to_numpy()[j2])
                        if "vol_30d" in p.columns else None)
                    target = equity * w / max(len(on_now), 1)
                    qty2 = target / c[j2]
                    if qty2 * c[j2] >= min_notional:
                        pending.append(("BUY", s2, qty2,
                                        "core: closed above its 200-day avg", None))
        # P2-I: decide the partial exit on this close; it fills at the next open like every other
        # exit. Once scaled, the remainder is protected by a close below breakeven plus fees.
        if flags.scale_out_r or flags.scale_out_pct:
            for s, p0 in list(pos.items()):
                j = here.get(s)
                if j is None or s in selling or s in scaling:
                    continue
                if not p0.get("scaled"):
                    if flags.scale_out_r and p0.get("r_unit", 0) <= 0:
                        continue                  # no measurable R, leave the trade alone
                    target = (p0["entry_price"] + 1.5 * p0["r_unit"]) if flags.scale_out_r \
                        else p0["entry_price"] * 1.08
                    if c[j] >= target:
                        why = "scaled out at +1.5R" if flags.scale_out_r else "scaled out at +8%"
                        pending.append(("SELL_PART", s, p0["qty"] * flags.scale_out_frac, why,
                                        p0.get("stop")))
                        scaling.add(s)
                elif c[j] <= p0["entry_price"] * (1 + 2 * fee):
                    pending.append(("SELL", s, p0["qty"], "breakeven stop after scaling out",
                                    p0.get("stop")))
                    selling.add(s)
        for s in pos:
            if s in selling or pos[s].get("engine") == "core":
                continue        # P7: the core has its own exit, decided in 4a above
            j = here.get(s)
            if state.killed:
                reason = "drawdown kill"
            elif j is None:
                reason = "no data / delisted" if missing.get(s, 0) >= 6 else None
            else:
                reason = signals.exit_reason(
                    sym[j], regime[j], breakdown[j], rank[j], settings, flags,
                    bars_held=pos[s]["bars"], in_profit=c[j] > pos[s]["entry_price"])
                if reason is None and trail_pct and c[j] <= pos[s].get("peak", c[j]) * (1 - trail_pct):
                    reason = f"gave back {trail_pct:.0%} from its best price"
            if reason and never_sell_at_loss and reason != "drawdown kill"                     and pos[s]["last_close"] < pos[s]["entry_price"]:
                reason = None      # the variant: sit on the loser and wait for it to recover
            if reason:
                pending.append(("SELL", s, pos[s]["qty"], reason, pos[s].get("stop")))
                selling.add(s)
        # P7: the core is not one of the breakout slots. Counting it would mean a core holding in
        # BTC silently costs the rotation a third of its capacity.
        n_alt = sum(1 for s2, p0 in pos.items() if not p0.get("engine") == "core" and s2 not in selling)
        # P12 step 3: fixed capital slices. The breakout engine is sized against its OWN slice of
        # equity rather than the whole book, so the two engines stop competing for one pot of cash.
        # Without this, whichever queues first spends it: a bar where the core enters leaves the
        # rotation sizing off a smaller number, and the per-engine attribution becomes meaningless
        # because each engine's size depended on what the other happened to do that bar.
        alt_cap = max_names
        slice_equity = equity
        if flags.capital_slices:
            alt_cap = flags.slice_max_names
            # The remainder after the core's reserved allocation. With the core off the breakout
            # engine still trades inside its own slice, which is then the whole book.
            slice_equity = equity * (1.0 - (flags.trend_core or 0.0))
        ok, _ = risk.can_enter(state, n_alt, alt_cap)
        if ok:
            slots = alt_cap - n_alt
            cands = [j for j in range(a, b) if entry_ok[j] and sym[j] not in pos]
            cands.sort(key=lambda j: rank[j])
            taken = []
            for j in cands:
                if len(taken) >= slots:
                    break
                if flags.corr_cap:
                    # Compared against what is held plus what this bar has already queued, so two
                    # correlated names cannot slip in on the same decision.
                    against = list(pos) + taken
                    if signals.max_corr_with_held(wide_rets, t, sym[j], against) > flags.corr_cap:
                        log.append({"time": t, "symbol": sym[j], "side": "SKIP", "qty": 0,
                                    "price": c[j], "fee": 0,
                                    "reason": f"correlation above {flags.corr_cap:g} with a held name"})
                        continue
                scale = scale_all[j]
                if flags.equity_curve_risk and len(recent_rets) >= 5:
                    window = recent_rets[-flags.equity_curve_lookback:]
                    if sum(window) / len(window) < 0:
                        scale *= 0.5       # the bot is cold: stake less until it warms up
                qty, note = risk.size_with_minimum(slice_equity, settings.risk_pct, c[j],
                                                   stop_lvl[j], settings.max_weight,
                                                   min_notional, scale=scale)
                if qty > 0:
                    taken.append(sym[j])
                    pending.append(("BUY", sym[j], qty,
                                    f"breakout, rank {int(rank[j])}, stop {stop_lvl[j]:.6g}",
                                    float(stop_lvl[j])))

    equity = pd.Series(eq_v, index=pd.DatetimeIndex(eq_t), name="equity")
    eng = (pd.DataFrame(eng_rows, index=equity.index) if eng_rows
           else pd.DataFrame(columns=["core", "breakout"]))
    return Result(equity=equity, trades=pd.DataFrame(trades), log=pd.DataFrame(log),
                  exposure=pd.Series(expo, index=equity.index), kills=kills, halts=halts,
                  engine_pnl=eng, meta={"score_col": score_col})


# ---------------------------------------------------------------- metrics and benchmarks

def metrics(equity: pd.Series, trades: pd.DataFrame = None, exposure: pd.Series = None,
            bars_per_year=BARS_PER_YEAR) -> dict:
    """`bars_per_year` must match the decision timeframe, or Sharpe is scaled wrongly:
    6*365 for 4h bars, 24*365 for 1h bars."""
    r = equity.pct_change().dropna()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    dd = equity / equity.cummax() - 1
    out = {"start": equity.index[0].date(), "end": equity.index[-1].date(),
           "total_return": equity.iloc[-1] / equity.iloc[0] - 1,
           "cagr": (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1,
           "max_drawdown": dd.min(),
           "sharpe": r.mean() / r.std() * np.sqrt(bars_per_year) if r.std() > 0 else 0.0}
    out["calmar"] = out["cagr"] / abs(out["max_drawdown"]) if out["max_drawdown"] < 0 else np.nan
    if exposure is not None:
        out["avg_exposure"] = exposure.mean()
    if trades is not None and len(trades):
        wins, losses = trades.loc[trades.pnl > 0, "pnl"].sum(), -trades.loc[trades.pnl <= 0, "pnl"].sum()
        out.update({"trades": len(trades), "win_rate": (trades.pnl > 0).mean(),
                    "avg_trade": trades.ret.mean(), "median_trade": trades.ret.median(),
                    "profit_factor": wins / losses if losses > 0 else np.inf,
                    "avg_bars_held": trades.bars_held.mean()})
    return out


def yearly(equity: pd.Series) -> pd.Series:
    e = equity.resample("YE").last()
    first = equity.iloc[0]
    return e.pct_change().fillna(e.iloc[0] / first - 1).rename(lambda d: d.year)


def benchmarks(panel, index, top_symbols):
    """BTC buy-and-hold and an equal-weight top-10 basket (rebalanced every bar, no fees)."""
    closes = panel.pivot(index="bar_time", columns="symbol", values="close").reindex(index)
    btc = closes["BTCUSDT"].ffill()
    rets = closes[[s for s in top_symbols if s in closes]].pct_change(fill_method=None)
    ew = (1 + rets.mean(axis=1).fillna(0)).cumprod()
    return {"BTC buy & hold": btc / btc.iloc[0], "Equal-weight top 10": ew / ew.iloc[0]}


def random_baselines(panel, trades, settings, n=500, seed=7):
    """Did the ranker choose coins and times better than chance?

    - random coin: same entry and exit bars, but a random coin that was eligible at that moment.
    - random time: same coin and holding length, entered at a random risk-on bar instead.
    Each gives the share of random runs whose average trade beat the strategy (a p-value).
    """
    if trades.empty:
        return {}
    rng = np.random.default_rng(seed)
    opens = panel.pivot(index="bar_time", columns="symbol", values="open")
    elig = panel.pivot(index="bar_time", columns="symbol", values="eligible").reindex(opens.index)
    elig = elig.astype("boolean").fillna(False).astype(bool)
    regime = panel[panel.symbol == "BTCUSDT"].set_index("bar_time")["regime_on"].reindex(opens.index)
    tpos = {t: i for i, t in enumerate(opens.index)}
    O = opens.to_numpy()
    cols = {s: k for k, s in enumerate(opens.columns)}
    cost = (1 - settings.slippage) * (1 - settings.fee_rate) / ((1 + settings.slippage) * (1 + settings.fee_rate))

    def ret(k, e, x):
        a, b = O[e, k], O[x, k]
        return b / a * cost - 1 if a > 0 and b > 0 and not (np.isnan(a) or np.isnan(b)) else np.nan

    coin_draws, time_draws, actual = [], [], []
    risk_on_bars = np.flatnonzero(regime.astype("boolean").fillna(False).to_numpy(dtype=bool))
    for tr in trades.itertuples():
        e, x = tpos.get(tr.entry_time), tpos.get(tr.exit_time)
        if e is None or x is None or e == 0:
            continue
        k = cols[tr.symbol]
        actual.append(ret(k, e, x))
        pool = [cols[s] for s in opens.columns[elig.iloc[e - 1].to_numpy()]]
        coin_draws.append([ret(pk, e, x) for pk in rng.choice(pool, n)] if pool else [np.nan] * n)
        hold = x - e
        ok = risk_on_bars[(risk_on_bars + 1 + hold < len(O)) & elig.iloc[:, k].to_numpy()[risk_on_bars]]
        starts = rng.choice(ok, n) + 1 if len(ok) else np.zeros(n, int)
        time_draws.append([ret(k, s, s + hold) for s in starts])
    actual = np.nanmean(actual)
    coin_means = np.nanmean(np.array(coin_draws), axis=0)
    time_means = np.nanmean(np.array(time_draws), axis=0)
    return {"strategy_avg_trade": actual,
            "random_coin_avg_trade": float(np.mean(coin_means)),
            "p_value_vs_random_coin": float((coin_means >= actual).mean()),
            "random_time_avg_trade": float(np.mean(time_means)),
            "p_value_vs_random_time": float((time_means >= actual).mean())}
