"""Order execution. PaperBroker fills virtually; BinanceBroker talks to Demo or Live spot.

Both return fills as dicts: time, symbol, qty, price, fee, client_order_id.
Live access is only possible when Settings allowed MODE=live (LIVE_TRADING_CONFIRMED=true);
the approval gate in main.py decides *when* an entry order may be sent.
"""

import hashlib
import hmac
import time
from datetime import datetime, timezone
from urllib.parse import urlencode

import requests

from . import data
from .config import BINANCE_URLS
from .risk import SymbolFilters


def client_order_id(bar_time, symbol, side, tag=""):
    """Deterministic id, so a retry of the same decision can never double-buy."""
    stamp = int(datetime.fromisoformat(str(bar_time)).timestamp()) if bar_time else int(time.time())
    return f"cb{tag}{stamp}{symbol[:10]}{side[0]}"[:36]


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class PaperBroker:
    """Virtual fills at the current price +/- slippage, minus the fee (Phase 5)."""

    def __init__(self, settings, store):
        self.s, self.store = settings, store
        if store.get("cash") is None:
            store.put("cash", settings.start_equity)
        self._filters = {}

    def cash(self):
        return self.store.get("cash")

    def filters(self, symbol):
        if symbol not in self._filters:
            try:
                info = data.get("/api/v3/exchangeInfo", {"symbol": symbol})["symbols"][0]
                self._filters[symbol] = SymbolFilters.from_exchange(info)
            except Exception:
                self._filters[symbol] = SymbolFilters()
        return self._filters[symbol]

    def buy(self, symbol, qty, cid, ref_price=None):
        px = (ref_price or data.price(symbol)) * (1 + self.s.slippage)
        qty = min(qty, self.cash() / (px * (1 + self.s.fee_rate)))
        qty, why = self.filters(symbol).check(qty, px)
        if why:
            raise OrderRejected(why)
        fee = qty * px * self.s.fee_rate
        self.store.put("cash", self.cash() - qty * px - fee)
        return {"time": _now(), "symbol": symbol, "qty": qty, "price": px, "fee": fee,
                "client_order_id": cid}

    def sell(self, symbol, qty, cid, ref_price=None):
        px = (ref_price or data.price(symbol)) * (1 - self.s.slippage)
        fee = qty * px * self.s.fee_rate
        self.store.put("cash", self.cash() + qty * px - fee)
        return {"time": _now(), "symbol": symbol, "qty": qty, "price": px, "fee": fee,
                "client_order_id": cid}

    # Paper has no exchange to rest an order on: the loop simulates the stop against each closed
    # bar's low, the same way the backtest does (P0-A).
    def open_orders(self, symbol=None):
        return []

    def place_stop(self, symbol, qty, stop_price, cid):
        return {"symbol": symbol, "qty": qty, "stopPrice": stop_price, "clientOrderId": cid,
                "orderId": "", "status": "SIMULATED"}

    def cancel_order(self, symbol, cid):
        return {"symbol": symbol, "clientOrderId": cid, "status": "SIMULATED_CANCEL"}


class OrderRejected(Exception):
    pass


class BinanceBroker:
    """Signed spot REST client for demo-api.binance.com (demo) or api.binance.com (live)."""

    def __init__(self, settings, store=None):
        if settings.mode not in ("demo", "live"):
            raise ValueError("BinanceBroker is for MODE=demo or MODE=live")
        if not settings.binance_key or not settings.binance_secret:
            raise RuntimeError("Set BINANCE_API_KEY and BINANCE_API_SECRET in .env")
        self.s, self.store = settings, store
        self.base = settings.demo_api_url if settings.mode == "demo" else BINANCE_URLS["live"]
        self.http = requests.Session()
        self.http.headers["X-MBX-APIKEY"] = settings.binance_key
        self.offset_ms = 0
        self._filters = {}
        self.sync_time()

    # plumbing ---------------------------------------------------------------------
    def sync_time(self):
        server = self.http.get(self.base + "/api/v3/time", timeout=10).json()["serverTime"]
        self.offset_ms = server - int(time.time() * 1000)

    def _signed(self, method, path, params=None, tries=5):
        params = dict(params or {})
        delay = 1.0
        for attempt in range(tries):
            params["timestamp"] = int(time.time() * 1000) + self.offset_ms
            params["recvWindow"] = 5000
            params.pop("signature", None)
            query = urlencode(params)
            sig = hmac.new(self.s.binance_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
            try:
                r = self.http.request(method, f"{self.base}{path}?{query}&signature={sig}", timeout=20)
            except requests.RequestException:
                # A POST may have reached Binance: never resend it blindly, let the caller
                # look the order up by its clientOrderId instead.
                if method == "POST" or attempt == tries - 1:
                    raise
                time.sleep(delay)
                delay *= 2
                continue
            if r.status_code >= 500 and method == "POST":
                raise OrderRejected(f"Binance {r.status_code}: order status unknown")
            if r.status_code in (429, 418) or r.status_code >= 500:
                time.sleep(float(r.headers.get("Retry-After", delay)))
                delay *= 2
                continue
            body = r.json()
            if r.status_code == 400 and body.get("code") == -1021:  # timestamp outside recvWindow
                self.sync_time()
                continue
            if r.status_code >= 400:
                raise OrderRejected(f"Binance {body.get('code')}: {body.get('msg')}")
            return body
        raise OrderRejected(f"{path}: gave up after {tries} tries")

    # account ------------------------------------------------------------------------
    def get_account(self):
        return self._signed("GET", "/api/v3/account", {"omitZeroBalances": "true"})

    def balances(self):
        return {b["asset"]: float(b["free"]) + float(b["locked"])
                for b in self.get_account()["balances"]}

    def cash(self):
        return self.balances().get("USDT", 0.0)

    def filters(self, symbol):
        if symbol not in self._filters:
            info = self.http.get(self.base + "/api/v3/exchangeInfo", params={"symbol": symbol},
                                 timeout=20).json()["symbols"][0]
            self._filters[symbol] = SymbolFilters.from_exchange(info)
        return self._filters[symbol]

    # orders -------------------------------------------------------------------------
    def get_order(self, symbol, cid):
        try:
            return self._signed("GET", "/api/v3/order", {"symbol": symbol, "origClientOrderId": cid})
        except OrderRejected as e:
            if "-2013" in str(e):  # order does not exist
                return None
            raise

    def cancel_order(self, symbol, cid):
        return self._signed("DELETE", "/api/v3/order", {"symbol": symbol, "origClientOrderId": cid})

    def create_order(self, symbol, side, qty, cid, limit_price=None):
        """LIMIT IOC at a marketable price: fills now or cancels the rest; never rests on the book.

        Idempotent: if an order with this clientOrderId already exists, it is returned instead.
        """
        existing = self.get_order(symbol, cid)
        if existing:
            return existing
        f = self.filters(symbol)
        ref = limit_price or data.get("/api/v3/ticker/bookTicker", {"symbol": symbol},
                                      base=self.base)["askPrice" if side == "BUY" else "bidPrice"]
        ref = float(ref)
        px = f.round_price(ref * (1.002 if side == "BUY" else 0.998))
        q, why = f.check(qty, px)
        if why:
            raise OrderRejected(why)
        params = {"symbol": symbol, "side": side, "type": "LIMIT", "timeInForce": "IOC",
                  "quantity": f"{q:.8f}".rstrip("0").rstrip("."),
                  "price": f"{px:.8f}".rstrip("0").rstrip("."),
                  "newClientOrderId": cid, "newOrderRespType": "FULL"}
        try:
            return self._signed("POST", "/api/v3/order", params)
        except (requests.RequestException, OrderRejected):
            # The request may have reached Binance even if we did not see the answer.
            existing = self.get_order(symbol, cid)
            if existing:
                return existing
            raise

    def _fill(self, order, symbol, cid):
        qty = float(order.get("executedQty", 0))
        if qty <= 0:
            raise OrderRejected(f"order {cid} did not fill (status {order.get('status')})")
        price = float(order.get("cummulativeQuoteQty", 0)) / qty
        fills = order.get("fills", [])
        base = symbol[:-4]
        base_fee = sum(float(f["commission"]) for f in fills if f["commissionAsset"] == base)
        usdt_fee = sum(float(f["commission"]) for f in fills if f["commissionAsset"] == "USDT")
        # A fee paid in the coin means we hold less of it; count it in USDT for the log.
        return {"time": _now(), "symbol": symbol, "qty": qty - base_fee, "price": price,
                "fee": base_fee * price + usdt_fee, "client_order_id": cid,
                "status": order.get("status")}

    def open_orders(self, symbol=None):
        return self._signed("GET", "/api/v3/openOrders", {"symbol": symbol} if symbol else {})

    def place_stop(self, symbol, qty, stop_price, cid, limit_offset=0.003):
        """Resting STOP_LOSS_LIMIT that protects the position between 4h checks (P0-A).

        The limit sits a little below the trigger so a fast move still fills; a gap through both
        leaves the order resting, which the next 4h check notices and closes at market.
        Idempotent by clientOrderId, like every other order this bot sends.
        """
        existing = self.get_order(symbol, cid)
        if existing:
            return existing
        f = self.filters(symbol)
        stop_px = f.round_price(stop_price)
        limit_px = f.round_price(stop_price * (1 - limit_offset))
        q, why = f.check(qty, limit_px)
        if why:
            raise OrderRejected(why)
        params = {"symbol": symbol, "side": "SELL", "type": "STOP_LOSS_LIMIT", "timeInForce": "GTC",
                  "quantity": f"{q:.8f}".rstrip("0").rstrip("."),
                  "price": f"{limit_px:.8f}".rstrip("0").rstrip("."),
                  "stopPrice": f"{stop_px:.8f}".rstrip("0").rstrip("."),
                  "newClientOrderId": cid}
        try:
            return self._signed("POST", "/api/v3/order", params)
        except (requests.RequestException, OrderRejected):
            existing = self.get_order(symbol, cid)
            if existing:
                return existing
            raise

    def buy(self, symbol, qty, cid, ref_price=None):
        return self._fill(self.create_order(symbol, "BUY", qty, cid), symbol, cid)

    def sell(self, symbol, qty, cid, ref_price=None):
        free = self.balances().get(symbol[:-4], 0.0)  # fees taken in the coin reduce what we hold
        return self._fill(self.create_order(symbol, "SELL", min(qty, free), cid), symbol, cid)


def make(settings, store):
    return PaperBroker(settings, store) if settings.mode == "paper" else BinanceBroker(settings, store)
