"""The Bot page: account, trades, what the bot sees, and the history test.

Read-only: it shows what the running bot has saved (data/state_<mode>.sqlite) and the history
test. It never places orders, so it is safe to open, close or refresh at any time. The same page
works for MODE=demo and MODE=live later, because it reads whichever state file the mode uses.
"""

import time
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import backtest, config, data, features, forecast, signals, universe
from app.portfolio import Store

S = config.load()
STORE = Store(S.db_path)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # validated categorical slots 1-3


# ------------------------------------------------------------------ data helpers

@st.cache_data(ttl=15, show_spinner=False)
def prices(symbols):
    """Live prices; if Binance cannot be reached, the last saved close from local history."""
    out, offline = {}, []
    for s in symbols:
        try:
            out[s] = data.price(s)
        except Exception:
            local = data.load(S.klines_dir, s, "4h")
            if len(local):
                out[s] = float(local["close"].iloc[-1])
                offline.append(pd.to_datetime(local["close_time"].iloc[-1] + 1, unit="ms", utc=True))
    return out, (max(offline) if offline else None)


@st.cache_data(show_spinner="Scoring the latest bar...")
def latest_scan(last_bar):
    u = universe.load(S)
    panel = features.build_panel(S.klines_dir, u["symbols"], S.donchian_bars)
    t = panel["bar_time"].max()
    rows = panel[panel["bar_time"] == t]
    return t, rows, signals.entry_candidates(rows, S)


@st.cache_data(ttl=24 * 3600, show_spinner="Replaying the rules over every year since 2017 (about 10 seconds)...")
def history_test(day):
    u = universe.load(S)
    panel = features.build_panel(S.klines_dir, u["symbols"], S.donchian_bars)
    res = backtest.run(panel, S)
    top10 = [c["symbol"] for c in sorted(u["coins"], key=lambda c: c["cg_rank"])[:10]]
    bench = backtest.benchmarks(panel, res.equity.index, top10)
    curves = {"This bot": res.equity / res.equity.iloc[0], **bench}
    stats = {k: backtest.metrics(v) for k, v in curves.items()}
    stats["This bot"] = backtest.metrics(res.equity, res.trades, res.exposure)
    years = pd.DataFrame({k: backtest.yearly(v) for k, v in curves.items()})
    rb = backtest.random_baselines(panel, res.trades, S)
    return curves, stats, years, rb, len(u["symbols"])


def bot_status():
    hb = S.heartbeat_path
    age = time.time() - float(hb.read_text()) if hb.exists() else None
    if age is None or age > 3 * 60:
        return "🔴 Not running", "double-click start.bat to start it"
    off = STORE.get("offline_since")
    if off:
        return "🟠 Running, no internet", (f"offline since {datetime.fromtimestamp(off):%H:%M}; "
                                          "retrying every 5 s, positions are safe")
    return "🟢 Running", f"last heartbeat {age:.0f} s ago"


@st.cache_data(ttl=300, show_spinner=False)
def _exit_triggers(symbols):
    """The price each held coin must close below to be sold: the current 20-bar low.

    This moves as the channel moves, so it is not the number stored when the position opened.
    """
    from app import data as _data, features as _features
    out = {}
    for s in symbols:
        try:
            f = _features.symbol_features(_data.load(S.klines_dir, s, "4h"),
                                          _data.load(S.klines_dir, s, "1d"), S.donchian_bars)
            out[s] = float(f["don_low"].iloc[-1])
        except Exception:
            pass
    return out


def money(x):
    return f"${x:,.2f}"


def trades_table(fills, pos, px):
    """Every buy and sell. A buy that is still held shows today's price and its live profit."""
    df = pd.DataFrame(fills)
    if df.empty:
        return df
    open_ids = {p["client_order_id"]: s for s, p in pos.items()}
    now, profit, pct, status = [], [], [], []
    for r in df.itertuples():
        if r.side == "BUY" and r.client_order_id in open_ids and r.symbol in px:
            cost = r.qty * r.price + r.fee
            value = r.qty * px[r.symbol]
            now.append(px[r.symbol]); profit.append(value - cost); pct.append(value / cost - 1)
            status.append("still held (live)")
        elif r.side == "SELL" and r.pnl is not None:
            cost = r.qty * r.price - r.fee - r.pnl
            now.append(None); profit.append(r.pnl); pct.append(r.pnl / cost if cost else None)
            status.append("sold (locked in)")
        else:
            now.append(None); profit.append(None); pct.append(None)
            status.append("bought, later sold" if r.side == "BUY" else "")
    df["time"] = pd.to_datetime(df["time"], utc=True, format="mixed").dt.strftime("%d %b %H:%M")
    df["coin"] = df["symbol"].str.replace("USDT", "")
    df["value"] = df["qty"] * df["price"]
    df["price now"], df["profit $"], df["profit %"], df["status"] = now, profit, pct, status
    return df[["time", "side", "coin", "status", "qty", "price", "price now", "profit $", "profit %",
               "value", "fee", "reason"]]


# ------------------------------------------------------------------ header

status, status_note = bot_status()
h1, h2 = st.columns([3, 1])
h1.title("Crypto Bot")
h1.caption(f"{S.mode.upper()} MODE · "
           + ("pretend money, no real orders" if S.mode == "paper" else
              "Binance Demo account" if S.mode == "demo" else "REAL MONEY, entries need your TRADE command"))
h2.markdown(f"### {status}")
h2.caption(status_note)


# ------------------------------------------------------------------ market outlook

@st.cache_data(ttl=3600, show_spinner="Refitting the weekly outlook (about 15 seconds, once a day)...")
def outlook():
    """The two forecasts that survived testing. Returns None rather than raising: a page that
    cannot show the outlook must still show the account."""
    return forecast.outlook(S)


def market_outlook():
    o = outlook()
    st.subheader("Market outlook")
    if o is None:
        st.info("No outlook yet. It needs about two and a half years of saved price history; "
                "it will appear by itself once the bot has downloaded enough.")
        return

    on = o["regime_on"]
    held = o.get("regime_days_held") or 0
    p = o.get("p_regime_on_7d")
    words = o.get("universe_vol_words") or "unknown"
    counts = pd.Series([c["words"] for c in o.get("per_coin", {}).values()]).value_counts()

    c = st.columns(3)
    c[0].metric("Market state", "Risk-on" if on else "Risk-off",
                f"held for {held:.0f} days", delta_color="off")
    c[0].caption("Bitcoin is above its 200-day average, so the bot is allowed to buy coins."
                 if on else "Bitcoin is below its 200-day average, so the bot stays in cash.")
    if p is not None:
        # Never print 100% or 0%. The model is well calibrated in the middle (Brier 0.054) but it
        # is a two-coefficient logistic, and nothing here is certain.
        shown = "over 99%" if p > 0.99 else "under 1%" if p < 0.01 else f"{p:.0%}"
        c[1].metric("Still risk-on in 7 days" if on else "Risk-on again within 7 days",
                    shown, "chance", delta_color="off")
        c[1].caption("Beats simply assuming nothing changes: 0.98 against 0.93 on the ranking test, "
                     "and a better-calibrated number (Brier 0.054 against 0.069).")
    c[2].metric("Expected swings this week", words.capitalize(),
                ", ".join(f"{n} {k}" for k, n in counts.items()) or None, delta_color="off")
    c[2].caption("Across the coins being watched, against how much they have moved over the last "
                 "two years. Wild weeks mean bigger price gaps, not worse odds.")

    st.warning("**These are forecasts of how much prices will MOVE and of whether the market is "
               "risk-on - not forecasts of WHERE prices are going.** This project measured price "
               f"direction and found it unpredictable: AUC {o['skill']['direction_auc']:.2f}, where "
               "0.50 is a coin toss. There is no price target anywhere on this page, and the bot "
               "does not trade on this outlook - testing showed that using it made results worse, "
               "so it is shown for information only.")

    with st.expander(f"Per coin, expected swing over the next 7 days (as of {o['as_of'][:16]} UTC)"):
        rows = [{"coin": s.replace("USDT", ""), "last 30 days": v["vol_30d"],
                 "next 7 days (forecast)": v["forecast_vol_7d"], "in words": v["words"]}
                for s, v in sorted(o["per_coin"].items(),
                                   key=lambda kv: -kv[1]["forecast_vol_7d"])]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                     column_config={
                         "last 30 days": st.column_config.NumberColumn(format="percent"),
                         "next 7 days (forecast)": st.column_config.NumberColumn(format="percent")})
        st.caption("Annualised volatility: 100% means a typical year's worth of swinging. The "
                   "forecast is three coefficients on the coin's own recent volatility at 1, 7 and "
                   "30 days. Out of sample it explains more of next week's movement than carrying "
                   "the 30-day figure forward does (0.53 against 0.43), over 28 walk-forward "
                   "periods from 2019 to 2026.")
    if o.get("stale"):
        st.caption(f"Outlook is {o['age_hours'] / 24:.1f} days old - the bot could not refresh it. "
                   "The market state shown may have changed.")


market_outlook()


# ------------------------------------------------------------------ live account (refreshes every 30 seconds)

@st.fragment(run_every=30)
def account():
    pos = STORE.positions()
    px, offline_at = prices(tuple(sorted(pos)))
    if offline_at is not None:
        st.warning(f"Can't reach Binance right now. Showing the last saved prices "
                   f"({offline_at:%d %b %H:%M} UTC). This page reconnects by itself every 30 seconds.")
    cash = STORE.get("cash", S.start_equity) if S.mode == "paper" else None
    held = sum(p["qty"] * px.get(s, p["entry_price"]) for s, p in pos.items())
    fills = STORE.fills(limit=500)
    closed = [f for f in fills if f["side"] == "SELL" and f["pnl"] is not None]

    st.subheader("Account")
    cost = sum(p["cost"] for p in pos.values())
    open_pnl = held - cost
    closed_pnl = sum(f["pnl"] for f in closed)
    wins = sum(f["pnl"] > 0 for f in closed)
    c = st.columns(5)
    if cash is not None:
        equity = cash + held
        c[0].metric("Account value", money(equity), f"{equity / S.start_equity - 1:+.2%} since start")
        c[1].metric("Cash", money(cash))
    c[2].metric("Invested", money(held), f"{len(pos)} of {S.max_names} slots used", delta_color="off")
    c[3].metric("Profit not yet sold", f"{'+' if open_pnl >= 0 else '-'}{money(abs(open_pnl))}",
                f"{open_pnl / cost:+.2%} on coins held (live)" if cost else "nothing held", delta_color="normal" if cost else "off")
    c[4].metric("Profit from sold trades", f"{'+' if closed_pnl >= 0 else '-'}{money(abs(closed_pnl))}",
                f"{wins} won / {len(closed) - wins} lost" if closed else "nothing sold yet", delta_color="off")

    st.markdown("**Open positions**")
    if pos:
        rows = []
        triggers = _exit_triggers(list(pos))
        for s, p in pos.items():
            now = px.get(s, p["entry_price"])
            rows.append({"coin": s.replace("USDT", ""), "amount": p["qty"], "bought at": p["entry_price"],
                         "price now": now, "profit $": p["qty"] * now - p["cost"],
                         "profit %": p["qty"] * now / p["cost"] - 1,
                         "value now": p["qty"] * now,
                         # The live exit is a close below the *current* 20-bar low, which moves with
                         # the price. The number stored at entry is only the sizing reference.
                         "sells if it closes below": triggers.get(s, p["stop"]),
                         "stop set at entry": p["stop"],
                         "bought because": p["reason"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, column_config={
            "profit $": st.column_config.NumberColumn(format="$%+.2f"),
            "profit %": st.column_config.NumberColumn(format="percent"),
            "value now": st.column_config.NumberColumn(format="dollar"),
            "amount": st.column_config.NumberColumn(format="%.4f")})
    else:
        st.info("No open positions. The bot is waiting for a coin to break out while Bitcoin is in an uptrend.")

    eq = pd.DataFrame(_equity_rows())
    if len(eq) >= 2:
        fig = go.Figure(go.Scatter(x=eq["time"], y=eq["equity"], mode="lines",
                                   line=dict(color=BLUE, width=2), name="Account value",
                                   hovertemplate="%{x|%d %b %H:%M}<br>$%{y:,.2f}<extra></extra>"))
        fig.update_layout(height=260, margin=dict(l=0, r=0, t=30, b=0),
                          title="Account value at each 4-hour check", yaxis_tickprefix="$", hovermode="x")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Trades**")
    t = trades_table(fills, pos, px)
    if t.empty:
        st.caption("No trades yet.")
    else:
        st.dataframe(t, hide_index=True, use_container_width=True, column_config={
            "value": st.column_config.NumberColumn(format="dollar"),
            "fee": st.column_config.NumberColumn(format="$%.2f"),
            "profit $": st.column_config.NumberColumn(format="$%+.2f"),
            "profit %": st.column_config.NumberColumn(format="percent"),
            "qty": st.column_config.NumberColumn(format="%.4f")})
    skipped = STORE.skipped(limit=25)
    if skipped:
        cost = [r["opportunity_cost"] for r in skipped if r["opportunity_cost"] is not None]
        avg = f", average move since: {sum(cost) / len(cost):+.1%}" if cost else ""
        with st.expander(f"Buys missed while the bot was not running ({len(skipped)}{avg})"):
            st.caption("These signals fired on a 4-hour check the bot was down for. They are not "
                       "taken later because they are stale, but they are recorded so the cost of "
                       "downtime can be measured.")
            sk = pd.DataFrame(skipped)[["bar_time", "symbol", "rank", "expected_entry",
                                        "price_when_seen", "opportunity_cost", "downtime_minutes"]]
            sk["symbol"] = sk["symbol"].str.replace("USDT", "", regex=False)
            st.dataframe(sk.rename(columns={"bar_time": "signal bar", "symbol": "coin",
                                            "expected_entry": "would have paid",
                                            "price_when_seen": "price when seen",
                                            "opportunity_cost": "move since",
                                            "downtime_minutes": "bot was down (min)"}),
                         hide_index=True, use_container_width=True,
                         column_config={"move since": st.column_config.NumberColumn(format="percent")})

    st.caption(f"Updated {datetime.now():%H:%M:%S} · prices refresh every 30 seconds")


def _equity_rows():
    with STORE.tx() as db:
        return [dict(r) for r in db.execute("SELECT time, equity FROM equity ORDER BY time")]


account()


# ------------------------------------------------------------------ the $100 wallet running alongside

@st.fragment(run_every=30)
def small_account():
    """A second paper account with a $100 wallet, trading the same signals at the same time."""
    path = S.data_dir / "state_paper100.sqlite"
    if not path.exists():
        return
    store = Store(path)
    pos, fills = store.positions(), store.fills(limit=200)
    px, _ = prices(tuple(sorted(pos)))
    held = sum(p["qty"] * px.get(s, p["entry_price"]) for s, p in pos.items())
    cash = store.get("cash", 100.0)
    closed = [f for f in fills if f["side"] == "SELL" and f["pnl"] is not None]
    equity = cash + held
    with st.expander(f"Small wallet test: $100 → **{money(equity)}** ({equity / 100 - 1:+.2%}), "
                     f"{len(pos)} coin(s) held, {len(closed)} sold", expanded=False):
        st.caption("The same bot, same signals, same moment, but with a $100 wallet. It shows what a small "
                   "real account would actually do: order sizes, Binance's $5 minimum, and profit in dollars.")
        c = st.columns(4)
        c[0].metric("Wallet value", money(equity), f"{equity / 100 - 1:+.2%}")
        c[1].metric("Cash", money(cash))
        c[2].metric("Profit not yet sold", f"{held - sum(p['cost'] for p in pos.values()):+.2f}")
        c[3].metric("Profit from sold trades", f"{sum(f['pnl'] for f in closed):+.2f}")
        if pos:
            st.dataframe(pd.DataFrame([{
                "coin": s.replace("USDT", ""), "amount": p["qty"], "bought at": p["entry_price"],
                "value now": p["qty"] * px.get(s, p["entry_price"]),
                "profit $": p["qty"] * px.get(s, p["entry_price"]) - p["cost"]} for s, p in pos.items()]),
                hide_index=True, use_container_width=True, column_config={
                    "value now": st.column_config.NumberColumn(format="$%.2f"),
                    "profit $": st.column_config.NumberColumn(format="$%+.2f"),
                    "amount": st.column_config.NumberColumn(format="%.6f")})
        else:
            st.caption("Nothing held yet.")


small_account()


# ------------------------------------------------------------------ paper vs history test

def paper_stats():
    """What the live paper bot has done so far, measured the same way as the history test."""
    with STORE.tx() as db:
        fills = [dict(r) for r in db.execute("SELECT * FROM fills ORDER BY id")]
        eq = [dict(r) for r in db.execute("SELECT * FROM equity ORDER BY time")]
    if not eq:
        return None
    start = pd.Timestamp(min([eq[0]["time"]] + [f["time"] for f in fills[:1]]))
    start = start.tz_localize("UTC") if start.tzinfo is None else start
    days = max((pd.Timestamp.now(tz="UTC") - start).total_seconds() / 86400, 1 / 24)
    opened, trades = {}, []
    for f in fills:
        if f["side"] == "BUY":
            opened[f["symbol"]] = f
        elif f["side"] == "SELL" and f["pnl"] is not None:
            cost = f["qty"] * f["price"] - f["fee"] - f["pnl"]
            b = opened.pop(f["symbol"], None)
            held = (pd.Timestamp(f["time"]) - pd.Timestamp(b["time"])).total_seconds() / 86400 if b else None
            trades.append({"ret": f["pnl"] / cost if cost else 0, "days": held, "win": f["pnl"] > 0})
    invested = [(e["equity"] - e["cash"]) / e["equity"] for e in eq if e["equity"]]
    last = eq[-1]["equity"]
    return {"days": days, "buys": sum(f["side"] == "BUY" for f in fills), "trades": trades,
            "invested": sum(invested) / len(invested), "return": last / S.start_equity - 1}


def history_range(curve, days):
    """Return after `days` in the history test: the middle 80% of all start dates."""
    d = curve.resample("D").last().dropna()
    n = max(int(round(days)), 1)
    r = (d.shift(-n) / d - 1).dropna()
    return r.quantile(0.1), r.quantile(0.9)


st.divider()
st.subheader("Is it behaving like the history test?")
curves, stats, years, rb, n_coins = history_test(datetime.now(timezone.utc).date().isoformat())
ps = paper_stats()
if ps is None:
    st.caption("The bot has not completed a 4-hour check yet.")
else:
    h = stats["This bot"]
    h_years = (pd.Timestamp(h["end"]) - pd.Timestamp(h["start"])).days / 365.25
    h_per_week = h["trades"] / (h_years * 52)
    tr = pd.DataFrame(ps["trades"])
    n = len(tr)
    lo, hi = history_range(curves["This bot"], ps["days"])
    enough_time, enough_trades = ps["days"] >= 14, n >= 10

    def verdict(ok, ready):
        return ("✅ normal" if ok else "⚠️ different") if ready else "⏳ too early"

    per_week = ps["buys"] / (ps["days"] / 7)
    rows = [
        ("Account return so far", f"{ps['return']:+.2%}",
         f"{lo:+.1%} to {hi:+.1%} after the same {ps['days']:.0f} day(s)", verdict(lo <= ps["return"] <= hi, True)),
        ("Buys per week", f"{per_week:.1f}", f"{h_per_week:.1f}",
         verdict(h_per_week / 3 <= per_week <= h_per_week * 3, enough_time)),
        ("Share of money invested", f"{ps['invested']:.0%}", f"{h['avg_exposure']:.0%}",
         verdict(h["avg_exposure"] / 3 <= ps["invested"] <= h["avg_exposure"] * 3, enough_time)),
        ("Trades won", f"{tr['win'].mean():.0%}" if n else "none sold yet", f"{h['win_rate']:.0%}",
         verdict(n and abs(tr["win"].mean() - h["win_rate"]) <= 0.2, enough_trades)),
        ("Average sold trade", f"{tr['ret'].mean():+.2%}" if n else "none sold yet", f"{h['avg_trade']:+.2%}",
         verdict(n and tr["ret"].mean() > -0.02, enough_trades)),
        ("Average days held", f"{tr['days'].mean():.1f}" if n and tr["days"].notna().any() else "none sold yet",
         f"{h['avg_bars_held'] / 6:.1f}", verdict(n and 0.33 <= tr["days"].mean() / (h["avg_bars_held"] / 6) <= 3, enough_trades)),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["", "Your paper bot", "9-year history test", "Check"]),
                 hide_index=True, use_container_width=True)
    st.progress(min(n / 30, 1.0), text=f"{n} of 30 sold trades: about 30 are needed before the numbers mean much "
                                        f"(running {ps['days']:.1f} days; the test averaged {h_per_week:.1f} buys a week)")
    st.caption("⏳ means there is not enough data yet to judge. ⚠️ is a prompt to look closer, not proof that "
               "something is broken: short stretches often differ from a 9-year average.")


# ------------------------------------------------------------------ right now

st.divider()
st.subheader("What the bot sees right now")
try:
    bar_time, rows, cands = latest_scan(STORE.get("last_bar", "none"))
    regime = bool(rows.set_index("symbol").loc["BTCUSDT", "regime_on"])
    next_check = bar_time + pd.Timedelta(hours=4)
    a, b, c = st.columns(3)
    a.metric("Bitcoin trend", "Up: buying allowed" if regime else "Down: no new buys")
    b.metric("Last 4-hour check", f"{bar_time:%d %b %H:%M} UTC")
    c.metric("Next check", f"{next_check:%H:%M} UTC")
    if len(cands):
        names = ", ".join(cands["symbol"].str.replace("USDT", ""))
        st.success(f"Buy signal on the last check: **{names}**")
    else:
        st.caption("No coin met every buy rule on the last check.")
    top = rows[rows["eligible"]].sort_values("rank").head(10)
    view = pd.DataFrame({
        "rank": top["rank"].astype(int), "coin": top["symbol"].str.replace("USDT", ""),
        "score (0-100)": top["score"].round(0), "7 days vs Bitcoin": top["rs_7d"],
        "above 50-day average": top["dist_sma50"], "broke out": top["breakout"].map({True: "yes", False: ""}),
        "price": top["close"]})
    st.dataframe(view, hide_index=True, use_container_width=True, column_config={
        "7 days vs Bitcoin": st.column_config.NumberColumn(format="percent"),
        "above 50-day average": st.column_config.NumberColumn(format="percent")})
    st.caption("Buy rules: Bitcoin above its 200-day average, the coin closes above its last 20 bars' high, "
               "it beats Bitcoin over 7 days, and it ranks in the top 5. It sells when any of those break "
               "down: Bitcoin trend turns, the coin closes below its last 20 bars' low, or it drops out of the top 10.")
except FileNotFoundError:
    st.warning("No data yet. Run: python -m app.main universe, then python -m app.main backfill")


# ------------------------------------------------------------------ history test

st.divider()
st.subheader("How these rules did on past data")
curves, stats, years, rb, n_coins = history_test(datetime.now(timezone.utc).date().isoformat())
me = stats["This bot"]
st.caption(f"The same rules replayed on {n_coins} coins, {me['start']:%b %Y} to {me['end']:%b %Y}, "
           f"with fees and slippage, starting from $10,000.")

k = st.columns(4)
k[0].metric("Return per year", f"{me['cagr']:+.1%}")
k[1].metric("Worst drop from a peak", f"{me['max_drawdown']:.0%}",
            f"Bitcoin: {stats['BTC buy & hold']['max_drawdown']:.0%}", delta_color="off")
k[2].metric("Trades", me.get("trades", 0), f"{me.get('win_rate', 0):.0%} won", delta_color="off")
k[3].metric("Return per unit of risk", f"{me['sharpe']:.2f}",
            f"Bitcoin: {stats['BTC buy & hold']['sharpe']:.2f}", delta_color="off")

fig = go.Figure()
for (name, curve), color in zip(curves.items(), (BLUE, ORANGE, AQUA)):
    daily = curve.resample("D").last().dropna() * S.start_equity
    fig.add_trace(go.Scatter(x=daily.index, y=daily, name=name, mode="lines",
                             line=dict(color=color, width=2.5 if name == "This bot" else 1.5),
                             hovertemplate="%{x|%d %b %Y}  $%{y:,.0f}<extra>" + name + "</extra>"))
fig.update_layout(height=380, margin=dict(l=0, r=0, t=40, b=0), hovermode="x unified",
                  title="Value of $10,000 (log scale: equal steps = equal % change)",
                  yaxis=dict(type="log", tickprefix="$"),
                  legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right", title=None))
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns([3, 2])
compare = pd.DataFrame({name: {"Return per year": s["cagr"], "Worst drop": s["max_drawdown"],
                               "Return per unit of risk (Sharpe)": s["sharpe"]}
                        for name, s in stats.items()})
left.markdown("**Compared with just holding**")
left.dataframe(compare.T.style.format({"Return per year": "{:+.1%}", "Worst drop": "{:.0%}",
                                       "Return per unit of risk (Sharpe)": "{:.2f}"}),
               use_container_width=True)
right.markdown("**Year by year**")
right.dataframe((years.rename(index=lambda y: str(y))).style.format("{:+.0%}"), use_container_width=True)

if rb:
    st.markdown(
        f"**Better than luck?** The bot's average trade made {rb['strategy_avg_trade']:+.1%}. "
        f"Picking a random coin at the same moments made {rb['random_coin_avg_trade']:+.1%}, "
        f"and the same coins bought at random times made {rb['random_time_avg_trade']:+.1%}. "
        f"Chance of doing this well by luck: about {rb['p_value_vs_random_coin']:.0%} and "
        f"{rb['p_value_vs_random_time']:.0%}.")
st.warning("**These numbers flatter the past.** They use today's top coins, which are the survivors. A fairer "
           "re-test (22 Sep 2026) used the top coins as they were at the time, including coins later delisted: "
           "about **+6% to +13% a year** with worst drops of **−17% to −34%**, and flat to negative in 2024–2025. "
           "In that test, picking coins beat random picks only weakly, and the timing of buys was no better "
           "than random. Past results do not guarantee future ones.")

with st.expander("Settings in use"):
    st.write({"mode": S.mode, "risk per trade": f"{S.risk_pct:.0%}", "max coins held": S.max_names,
              "max share of account in one coin": f"{S.max_weight:.0%}",
              "stop new buys after a daily loss of": f"{S.daily_loss_halt:.0%}",
              "sell everything after a drop from peak of": f"{S.kill_drawdown:.0%}",
              "fee per side": f"{S.fee_rate:.2%}", "slippage per side": f"{S.slippage_bps:g} bps"})
    st.caption("Change these in the .env file (see .env.example), then restart the bot.")
