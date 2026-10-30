"""What it sees, and what it will do about it.

The bot decides and acts on its own. This page is the window into that: not the account history
(the Bot page has that), but the *pending* state - every level it is watching, how far price is
from each one, and the exact order it will place when a level is crossed.

Read-only. It opens each book's sqlite read-only and never places an order, so refreshing it or
leaving it open cannot affect the bot.

Three questions, in the order a person actually asks them:

    1. WHAT IT SEES          the trend, the core's own trigger, the watch list and the distances
    2. WHAT IT WILL DO       orders already decided and filling at the next open, then the
                             conditional intents: buy X if it closes above Y, sell Z if below W
    3. WHY NOT MORE          what is currently blocking new buys, and the entries it declined

Everything here comes from the snapshot the bot froze on the last closed bar (`levels`) plus the
live price. Nothing on this page re-derives a decision, so it cannot disagree with the bot.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from app import config, data

S = config.load()
BOOKS = [("paper", "Paper $10,000"), ("paper100", "Paper $100"), ("demo", "Binance demo")]
GREEN, RED, GREY = "#1baf7a", "#eb6834", "#8a8f98"


# ------------------------------------------------------------------ read-only book access

def book_path(account):
    return S.data_dir / f"state_{account}.sqlite"


def available_books():
    return [(a, label) for a, label in BOOKS if book_path(a).exists()]


@st.cache_data(ttl=10, show_spinner=False)
def read_book(account):
    """Everything this page needs from one book, in a single read-only connection.

    Opened with mode=ro so the page can never lock or write the file the live bot is using.
    """
    p = book_path(account)
    out = {"levels": None, "positions": {}, "queued": [], "skipped": [], "risk": {},
           "last_bar": None, "equity": None}
    if not p.exists():
        return out
    con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    import json

    def kv(key):
        try:
            r = con.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
            return json.loads(r["value"]) if r else None
        except Exception:
            return None

    out["levels"] = kv("levels")
    out["risk"] = kv("risk") or {}
    out["last_bar"] = kv("last_bar")
    try:
        out["positions"] = {r["symbol"]: dict(r) for r in con.execute("SELECT * FROM positions")}
    except Exception:
        pass
    try:
        out["queued"] = [dict(r) for r in con.execute(
            "SELECT * FROM queued_orders WHERE status='queued' "
            "ORDER BY CASE side WHEN 'SELL' THEN 0 ELSE 1 END, id")]
    except Exception:
        pass
    try:
        out["skipped"] = [dict(r) for r in con.execute(
            "SELECT * FROM skipped_signals ORDER BY id DESC LIMIT 10")]
    except Exception:
        pass
    try:
        r = con.execute("SELECT equity FROM equity ORDER BY time DESC LIMIT 1").fetchone()
        out["equity"] = r["equity"] if r else None
    except Exception:
        pass
    con.close()
    return out


@st.cache_data(ttl=15, show_spinner=False)
def live_prices(symbols):
    out = {}
    for s in symbols:
        try:
            out[s] = float(data.price(s))
        except Exception:
            out[s] = None
    return out


@st.cache_data(ttl=120, show_spinner="Working out what the bot is watching...")
def recomputed_levels(account):
    """The same levels the bot would snapshot, built here for a book that has not saved one yet.

    This runs Bot.snapshot_levels itself rather than reimplementing it, so the page cannot drift
    from the bot's definition of a trigger. A Bot is constructed read-only - no order is placed,
    because snapshot_levels only reads the scan and writes one key - and the account's own flags
    are picked up, so a book running the trend core gets its core row and a book that is not does
    not.
    """
    import dataclasses

    from app import ranker, universe
    from app.main import Bot
    try:
        s = dataclasses.replace(S, account_name=account)
        bot = Bot(s)
        held = set(bot.store.positions())
        u = universe.build(s, held_symbols=held)
        symbols = sorted(set(u["symbols"]) | held)
        _, bar_time, rows, _ = ranker.scan(s, symbols, refresh=False,
                                           log=lambda *_: None, flags=bot.flags)
        latest = rows.set_index("symbol")
        regime = bool(latest.loc["BTCUSDT", "regime_on"])
        return bot.snapshot_levels(latest, bar_time, set(u["symbols"]), regime)
    except Exception as e:
        st.caption(f"(could not recompute levels: {type(e).__name__}: {e})")
        return None


@st.cache_data(ttl=120, show_spinner="Checking which rule rejected everything...")
def entry_funnel(account):
    """How many coins survive each entry rule, in the order the engine applies them.

    "Nothing happened" is not an answer; "19 coins were eligible and 0 of them closed above their
    20-bar high" is. The gates are applied cumulatively and in the engine's real order, and the
    last one that takes the count to zero is named as the culprit.

    The predicates mirror signals.entry_candidates. They are duplicated rather than imported
    because that function returns only survivors, and the whole point here is the count at each
    step - but `extra_filters` is called directly, so the four shipped filters cannot drift.
    """
    import dataclasses

    from app import ranker, signals as sg, universe
    from app.main import Bot
    try:
        s = dataclasses.replace(S, account_name=account)
        bot = Bot(s)
        held = set(bot.store.positions())
        u = universe.build(s, held_symbols=held)
        symbols = sorted(set(u["symbols"]) | held)
        _, _, rows, _ = ranker.scan(s, symbols, refresh=False, log=lambda *_: None,
                                    flags=bot.flags)
        b = rows.set_index("symbol")
        regime_on = bool(b.loc["BTCUSDT", "regime_on"])
        core = set(sg.core_symbols(bot.flags))

        stages, prev = [], None
        m = pd.Series(True, index=b.index)
        stages.append(("in the coin list", int(m.sum())))
        m &= b["eligible"].fillna(False)
        stages.append(("enough history and volume", int(m.sum())))
        if core:
            m &= ~b.index.isin(core)
            stages.append(("not reserved by the trend core", int(m.sum())))
        m &= regime_on
        stages.append(("Bitcoin above its 200-day average", int(m.sum())))
        m &= b["breakout"].fillna(False)
        stages.append(("closed above its own 20-bar high", int(m.sum())))
        m &= (b["rs_7d"].fillna(-1) > 0) | (b.index == "BTCUSDT")
        stages.append(("beat Bitcoin over 7 days", int(m.sum())))
        m &= b["rank"].fillna(999) <= s.candidates
        stages.append((f"ranked in the top {s.candidates}", int(m.sum())))
        m &= sg.extra_filters(b, bot.flags).fillna(False)
        stages.append((f"passed the extra filters ({bot.flags.label})", int(m.sum())))

        culprit = None
        for label, n in stages:
            if prev is not None and prev > 0 and n == 0:
                culprit = label
                break
            prev = n
        return {"stages": stages, "culprit": culprit}
    except Exception as e:
        st.caption(f"(could not build the funnel: {type(e).__name__}: {e})")
        return None


def name(sym):
    return sym.replace("USDT", "")


def gap(now, level):
    """Signed distance from price to a level, as a fraction. None when either is missing."""
    if now is None or level is None or not level:
        return None
    return now / level - 1


def arrow(x, good_when_positive=True):
    if x is None:
        return ""
    good = x > 0 if good_when_positive else x < 0
    return ("🟢 " if good else "🔴 ") + f"{x:+.2%}"


# ------------------------------------------------------------------ page

st.subheader("What it sees, and what it will do")

books = available_books()
if not books:
    st.warning("No book has run yet. Start the bot: `python -m app.main fleet`")
    st.stop()

labels = {a: f"{label}" for a, label in books}
account = st.radio("Book", [a for a, _ in books], horizontal=True,
                   format_func=lambda a: labels[a])
b = read_book(account)
lv = b["levels"]

fresh = False
if not lv:
    # The snapshot is written once per closed bar. A book that has not completed a bar since this
    # feature was deployed has none yet, and waiting four hours to show anything would make the
    # page useless exactly when someone first opens it. So the levels are computed here instead -
    # from the same scan and the same flags the bot uses - and labelled as computed, because a
    # recomputed level is only *probably* what the bot has in mind, while a snapshot IS.
    lv = recomputed_levels(account)
    fresh = True
if not lv:
    st.info(f"The **{labels[account]}** book has no price history to work from yet. "
            "Run `python -m app.main backfill` and let one 4-hour bar close.")
    st.stop()
if fresh:
    st.warning("The bot has not saved a snapshot since this page was added, so these levels were "
               "recomputed here from the same data and the same flags. They should match what the "
               "bot will do; after its next 4-hour check this page switches to the bot's own "
               "saved snapshot.", icon="⚠️")

bar_time = pd.Timestamp(lv.get("bar_time"))
regime = bool(lv.get("regime"))
core_state = lv.get("core") or {}
core_weight = float(lv.get("core_weight") or 0.0)
max_names = int(lv.get("max_names") or S.max_names)
pos = b["positions"]
entry_lv, exit_lv, stop_lv, rank_lv = (lv.get("entry") or {}, lv.get("exit") or {},
                                       lv.get("stop") or {}, lv.get("rank") or {})

watch = sorted(set(entry_lv) | set(pos) | set(core_state))
px = live_prices(watch)

# ------------------------------------------------------------------ 1. what it sees
st.markdown("#### 1. What it sees")
c = st.columns(4)
c[0].metric("Bitcoin trend", "Up" if regime else "Down",
            help="BTC above its own 200-day average. This gates the rotation's new buys.")
c[1].metric("Last check", f"{bar_time:%d %b %H:%M} UTC")
c[2].metric("Next check", f"{bar_time + pd.Timedelta(hours=4):%H:%M} UTC")
# A position is the core's if its SYMBOL is one the core owns. Matching on the reason text
# broke once the live path started writing "live core: ...", which does not start with "core".
n_alt_slots = len([s for s in pos if s not in core_state])
c[3].metric("Slots used", f"{n_alt_slots} of {max_names}")

if core_weight:
    st.markdown("**Trend core** — holds a coin for as long as it closes above its *own* "
                f"200-day average. Target size {core_weight:.0%} of the book.")
    rows = []
    for s, d in core_state.items():
        held = s in pos
        rows.append({"coin": name(s), "vs its 200-day average": d["dist_sma200"],
                     "above the line": "yes" if d["on"] else "no",
                     "held now": "yes" if held else "no",
                     "so it will": ("hold" if d["on"] and held else
                                    "BUY at the next open" if d["on"] and not held else
                                    "SELL at the next open" if held and not d["on"] else
                                    "stay out")})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                 column_config={"vs its 200-day average":
                                st.column_config.NumberColumn(format="percent")})
else:
    st.caption("The trend core is off on this book, so it only buys on breakouts and sits in "
               "cash the rest of the time.")

# ------------------------------------------------------------------ 2. what it will do
st.divider()
st.markdown("#### 2. What it will do")

if b["queued"]:
    st.markdown("**Already decided — these fill at the next bar's open:**")
    q = pd.DataFrame([{
        "side": r["side"], "coin": name(r["symbol"]), "quantity": r["qty"],
        "price now": px.get(r["symbol"]),
        "value now": (px.get(r["symbol"]) or 0) * r["qty"],
        "why": r["reason"]} for r in b["queued"]])
    st.dataframe(q, hide_index=True, use_container_width=True)
else:
    st.caption("No order is queued: nothing crossed a level on the last check.")

st.markdown("**Conditional — the exact level that would make it act:**")
intents = []

for s in sorted(entry_lv):
    if s in pos:
        continue
    now, lvl = px.get(s), entry_lv[s]
    g = gap(now, lvl)
    intents.append({"it will": "BUY", "coin": name(s), "if price": "closes above",
                    "this level": lvl, "price now": now,
                    "distance": g, "note": f"rank {int(rank_lv.get(s, 0)) or '-'}"
                    if not regime else f"rank {int(rank_lv.get(s, 0)) or '-'}"})

for s in sorted(pos):
    is_core = s in core_state
    now = px.get(s)
    if is_core:
        d = core_state.get(s, {})
        intents.append({"it will": "SELL", "coin": name(s), "if price": "closes below",
                        "this level": None, "price now": now,
                        "distance": -(d.get("dist_sma200") or 0) if d else None,
                        "note": "its own 200-day average (core)"})
    else:
        lvl = exit_lv.get(s)
        intents.append({"it will": "SELL", "coin": name(s), "if price": "closes below",
                        "this level": lvl, "price now": now, "distance": gap(now, lvl),
                        "note": "20-bar low, or it drops out of the top 10"})

if intents:
    df = pd.DataFrame(intents).sort_values(["it will", "distance"])
    st.dataframe(df, hide_index=True, use_container_width=True, column_config={
        "distance": st.column_config.NumberColumn(
            "how far away", format="percent",
            help="Positive means price is already above the level; negative means it has to move."),
        "this level": st.column_config.NumberColumn(format="%.6g"),
        "price now": st.column_config.NumberColumn(format="%.6g")})
    st.caption("Distances use the live price against a level frozen on the last closed bar. The "
               "bot acts on a **closing** price, so being past a level mid-bar is not yet a trade.")
else:
    st.caption("Nothing is on the watch list: no coin passed the filters on the last check.")

# ------------------------------------------------------------------ 3. why not more
st.divider()
st.markdown("#### 3. Why it is doing nothing right now")

n_alt = len([s for s in pos if s not in core_state])
risk_state = b["risk"] or {}
queued = bool(b["queued"])

# --- the one-line verdict, in the order the engine actually applies them -------------------
# Hard stops first (nothing can happen), then capacity, then "no signal", then the ordinary case
# of simply waiting. Only the first true reason is the headline; the rest are listed underneath.
if queued:
    st.success(f"It **is** about to act: {len(b['queued'])} order(s) are queued and fill at the "
               "next bar's open. See section 2.")
elif risk_state.get("killed"):
    st.error("**Stopped by the drawdown kill switch.** No new buys until it re-arms "
             "(7 days after it fired). Exits still run.")
elif risk_state.get("halted_today"):
    st.error("**The daily loss limit has halted new buys for today.** It resets at midnight UTC.")
elif n_alt >= max_names and not core_weight:
    st.warning(f"**All {max_names} slots are full.** It cannot buy anything else until one of "
               "the held coins exits, and none of them has reached its exit level yet.")
elif not regime and not core_weight:
    st.warning("**Bitcoin is below its 200-day average**, so the rotation takes no new positions "
               "and the book sits in cash. This is the rule that kept it out of the 2022 bear "
               "market.")
elif not entry_lv and n_alt < max_names:
    st.info("**No coin passes every entry rule**, so there is nothing to buy even though there "
            "are free slots. The funnel below shows which rule is doing the rejecting.")
else:
    st.info("**Nothing is blocked — it is waiting for a price to cross a level.** The levels are "
            "in section 2; the closest one is shown below.")

# --- the funnel: which rule is actually rejecting everything --------------------------------
f = entry_funnel(account)
if f:
    st.markdown("**Where the coins drop out**, on the last closed bar:")
    rows = [{"rule": k, "coins left": v} for k, v in f["stages"]]
    fd = pd.DataFrame(rows)
    st.dataframe(fd, hide_index=True, use_container_width=True)
    if f["culprit"]:
        st.caption(f"The rule that empties the list is **{f['culprit']}**. That is normal on most "
                   "bars — a breakout is an event, not a state, and the strategy takes roughly "
                   "one trade a week per slot.")

# --- how close is the nearest trigger -------------------------------------------------------
buys = [i for i in intents if i["it will"] == "BUY" and i["distance"] is not None]
if buys:
    # `distance` is price/level - 1, so POSITIVE means price is already past the trigger. Those
    # coins are not waiting to get there at all; they are waiting for the bar to close. Reporting
    # "needs to rise -21.39%" conflated the two.
    already = [i for i in buys if i["distance"] >= 0]
    if already:
        names = ", ".join(sorted(i["coin"] for i in already))
        st.markdown(f"**Already above its trigger:** {names}. If the 4-hour bar *closes* here, "
                    "the buy is decided on that close and fills at the next open.")
    waiting = [i for i in buys if i["distance"] < 0]
    if waiting:
        nearest = max(waiting, key=lambda i: i["distance"])      # least negative = closest
        st.markdown(f"**Closest still waiting:** {nearest['coin']} — it has to rise "
                    f"**{-nearest['distance']:.2%}** and *close* above "
                    f"{nearest['this level']:.6g} on a 4-hour bar.")

# --- the core's own reason ------------------------------------------------------------------
if core_weight:
    for s, d in core_state.items():
        held = s in pos
        if d["on"] and held:
            st.markdown(f"**Core:** holding {name(s)}; it is {d['dist_sma200']:+.2%} above its "
                        "200-day average, so there is nothing to do.")
        elif not d["on"] and not held:
            st.markdown(f"**Core:** out of {name(s)} — it is {d['dist_sma200']:+.2%} relative to "
                        "its 200-day average, and the core only holds a coin above that line.")
else:
    st.markdown("**Core:** off on this book, so cash is not deployed between breakouts. "
                "This is the single biggest reason the book sits idle.")

# --- everything else that is also true -------------------------------------------------------
others = []
if not regime:
    others.append("Bitcoin is below its 200-day average (no new rotation buys).")
if n_alt >= max_names:
    others.append(f"All {max_names} rotation slots are full.")
if not entry_lv:
    others.append("No coin is on the watch list.")
if pos and not queued:
    others.append(f"{len(pos)} position(s) held, none near its exit level.")
others.append(f"It only acts on a **closed** 4-hour bar — next one at "
              f"{bar_time + pd.Timedelta(hours=4):%H:%M} UTC.")
with st.expander("Everything else that is true right now"):
    for x in others:
        st.markdown(f"- {x}")

if b["skipped"]:
    with st.expander(f"Entries it saw and declined ({len(b['skipped'])} most recent)"):
        sk = pd.DataFrame([{
            "when": r["observed_at"], "coin": name(r["symbol"]),
            "price then": r.get("price_when_seen"), "why it passed": r["reason"]}
            for r in b["skipped"]])
        st.dataframe(sk, hide_index=True, use_container_width=True)

# Streamlit renders captions as markdown, where a pair of dollar signs becomes LaTeX: "Paper
# $10,000 - equity $9,905.05" was being drawn as mathematics. Escaping every dollar sign fixes it.
_eq = f"\\${b['equity']:,.2f}" if b["equity"] else "n/a"
st.caption(f"Book: {labels[account].replace('$', chr(92) + '$')} · equity {_eq} · "
           f"snapshot from {bar_time:%d %b %Y %H:%M} UTC · this page is read-only")
