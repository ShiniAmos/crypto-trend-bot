"""The Live market page: Binance prices streamed straight to the browser, tick by tick.

The page talks to Binance's public market-data stream directly from your browser (no server in
between), so prices are as fresh as Binance sends them. Click a coin to open its live chart,
order book and trades. Read-only: nothing here places orders.
"""

import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from app import config, universe
from app.portfolio import Store

S = config.load()
store = Store(S.db_path)

try:
    u = universe.load(S)
    symbols = [c["symbol"] for c in sorted(u["coins"], key=lambda c: c["cg_rank"])]
except FileNotFoundError:
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]

held = {s: p["entry_price"] for s, p in store.positions().items()}
signals = [s for s in (store.get("last_signals") or []) if s in symbols]
cfg = {"symbols": symbols, "held": held, "signals": signals,
       "start": next(iter(held), "BTCUSDT")}

st.markdown("#### Live market")
st.caption("Prices come straight from Binance to your browser on every trade. Click a coin to open its "
           "live chart, order book and trades. HOLDING = the bot owns it; BUY SIGNAL = it met the buy "
           "rules on the last check.")
html = (Path(__file__).with_name("live_market.html").read_text(encoding="utf-8")
        .replace("__CONFIG__", json.dumps(cfg)))
components.html(html, height=1080, scrolling=True)
