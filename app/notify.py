"""Phase 8: Telegram alerts and the TRADE / SKIP / FLAT / FORCE command set (section 5.1).

Without a Telegram token the bot prints messages to its log and you can type commands with
`python -m app.main cmd "TRADE ETH"`. Messages from any chat other than TELEGRAM_CHAT_ID
are ignored.
"""

import re
from dataclasses import dataclass

import requests

CMD = re.compile(
    r"^\s*(?P<verb>TRADE|FORCE|FLAT|SKIP|STATUS)"
    r"(?:\s+(?P<sym>[A-Za-z0-9]+))?"
    r"(?:\s+(?P<amt>\d+(?:\.\d+)?)\s*(?:USDT)?)?\s*$", re.I)


@dataclass
class Command:
    verb: str
    symbol: str = None
    amount_usdt: float = None


def parse(text: str):
    """Return a Command, or None if the text is not a valid command."""
    m = CMD.match(text or "")
    if not m:
        return None
    verb = m["verb"].upper()
    sym = m["sym"].upper() if m["sym"] else None
    if sym and verb in ("TRADE", "FORCE", "FLAT") and sym != "ALL" and not sym.endswith("USDT"):
        sym += "USDT"
    if verb in ("TRADE", "FORCE") and not sym:
        return None
    if verb == "FLAT" and not sym:
        return None
    if verb in ("SKIP", "STATUS") and (sym or m["amt"]):
        return None
    if verb == "FLAT" and m["amt"]:
        return None
    return Command(verb, sym, float(m["amt"]) if m["amt"] else None)


def candidates_message(bar_time, regime_on, rows, expiry_hours):
    """The section-9 template."""
    lines = [f"Candidates (BTC regime: {'ON' if regime_on else 'OFF'})  |  bar {bar_time:%Y-%m-%d %H:%M} UTC"]
    for i, r in enumerate(rows, 1):
        lines.append(f"{i} {r['symbol'][:-4]:<6} RS {r['rs_7d'] * 100:+.1f}   "
                     f"breakout {'YES' if r['breakout'] else 'NO'}   "
                     f"vol {'OK' if r['volume_ratio'] >= 0.7 else 'LOW'}   stop {r['stop_level']:.6g}")
    if rows:
        ex = rows[0]["symbol"][:-4]
        lines.append(f"Reply: TRADE {ex}    or    TRADE {ex} 150 USDT    or    SKIP    or    FLAT ALL")
        lines.append(f"No reply in {expiry_hours:g} hours = no live order.")
    else:
        lines.append("No candidates this bar.")
    return "\n".join(lines)


class Telegram:
    def __init__(self, settings, store, log=print):
        self.token, self.chat = settings.telegram_token, str(settings.telegram_chat_id)
        self.store, self.log = store, log

    @property
    def enabled(self):
        return bool(self.token and self.chat)

    def _safe(self, err):
        """Error text with the bot token hidden (request errors include the full URL)."""
        return str(err).replace(self.token, "<token hidden>") if self.token else str(err)

    def _post(self, text):
        requests.post(f"https://api.telegram.org/bot{self.token}/sendMessage",
                      json={"chat_id": self.chat, "text": text}, timeout=10).raise_for_status()

    def _failed(self, what, err):
        """Log a Telegram failure once per outage, not every few seconds."""
        if not self.store.get("tg_down"):
            self.log(f"telegram {what} failed (will keep retrying quietly): {self._safe(err)}")
            self.store.put("tg_down", True)

    def send(self, text):
        """Send now, or keep it in an outbox and deliver it when the connection is back."""
        self.log("[message] " + text.replace("\n", "\n          "))
        if not self.enabled:
            return
        try:
            self.flush()
            self._post(text)
        except requests.RequestException as e:
            self.store.put("tg_outbox", (self.store.get("tg_outbox") or [])[-50:] + [text])
            self._failed("send", e)

    def flush(self):
        """Deliver messages that could not be sent while offline, oldest first."""
        outbox = self.store.get("tg_outbox") or []
        while outbox:
            self._post("(delayed, sent while offline)\n" + outbox[0])
            outbox = outbox[1:]
            self.store.put("tg_outbox", outbox)

    def poll(self):
        """Move new messages from the owner's chat into the commands table."""
        if not self.enabled:
            return
        offset = self.store.get("tg_offset", 0)
        try:
            r = requests.get(f"https://api.telegram.org/bot{self.token}/getUpdates",
                             params={"offset": offset, "timeout": 0}, timeout=10).json()
            self.flush()
        except requests.RequestException as e:
            self._failed("connection", e)
            return
        if self.store.get("tg_down"):
            self.log("telegram reachable again")
            self.store.put("tg_down", False)
        for u in r.get("result", []):
            offset = u["update_id"] + 1
            msg = u.get("message") or {}
            if str(msg.get("chat", {}).get("id")) == self.chat and msg.get("text"):
                self.store.add_command(msg["text"], "telegram")
        self.store.put("tg_offset", offset)
