"""Website for the bot:  streamlit run dashboard.py --server.port 8601

Three pages:
  Bot          account, trades and the history test
  Next move    what the bot is watching and the exact order it will place when a level is crossed,
               per book - the window into decisions it takes on its own
  Live market  Binance prices streamed straight to your browser

Read-only: nothing here can place an order.
"""

import hmac
import os

import streamlit as st

from app import config  # noqa: F401  (loads .env so DASHBOARD_PASSWORD is available)

st.set_page_config(page_title="Crypto Bot", page_icon="📈", layout="wide")


def _password_ok():
    """If DASHBOARD_PASSWORD is set in .env, ask for it once per browser session."""
    want = os.environ.get("DASHBOARD_PASSWORD", "")
    if not want or st.session_state.get("authed"):
        return True
    st.title("Crypto Bot")
    typed = st.text_input("Password", type="password")
    if typed and hmac.compare_digest(typed, want):
        st.session_state["authed"] = True
        st.rerun()
    elif typed:
        st.error("Wrong password.")
    return False


if not _password_ok():
    st.stop()


st.navigation([st.Page("ui/bot.py", title="Bot", icon="🤖", default=True),
               st.Page("ui/next_actions.py", title="Next move", icon="🎯"),
               st.Page("ui/live_market.py", title="Live market", icon="📈")]).run()
