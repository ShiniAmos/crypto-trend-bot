import json

import pytest
import requests

from app import data, notify, universe
from app.portfolio import Store


class Resp:
    def __init__(self, code, body):
        self.status_code, self._body, self.headers = code, body, {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(self.status_code)


def test_public_data_switches_to_backup_host(monkeypatch):
    hosts = []

    def fake_get(url, params=None, timeout=None):
        hosts.append(url[: -len("/api/v3/ticker/price")])
        if url.startswith(data.PUBLIC_BASES[0]):
            raise requests.ConnectionError("main host down")
        return Resp(200, {"price": "1"})

    monkeypatch.setattr(data._session, "get", fake_get)
    monkeypatch.setattr(data.time, "sleep", lambda s: None)
    assert data.get("/api/v3/ticker/price") == {"price": "1"}
    assert hosts == data.PUBLIC_BASES  # tried main, then backup


def test_region_block_uses_backup_host(monkeypatch):
    calls = iter([Resp(451, {}), Resp(200, {"ok": 1})])
    monkeypatch.setattr(data._session, "get", lambda *a, **k: next(calls))
    assert data.get("/api/v3/time") == {"ok": 1}


def test_offline_everywhere_raises_so_the_loop_can_retry(monkeypatch):
    def down(*a, **k):
        raise requests.ConnectionError("no internet")
    monkeypatch.setattr(data._session, "get", down)
    monkeypatch.setattr(data.time, "sleep", lambda s: None)
    with pytest.raises(requests.ConnectionError):
        data.get("/api/v3/time")


def test_telegram_keeps_messages_while_offline_and_sends_them_later(settings, monkeypatch):
    settings.telegram_token, settings.telegram_chat_id = "t", "1"
    store = Store(settings.db_path)
    tg = notify.Telegram(settings, store, log=lambda *_: None)
    sent, online = [], {"up": False}

    def post(url, json=None, timeout=None):
        if not online["up"]:
            raise requests.ConnectionError("offline")
        sent.append(json["text"])
        return Resp(200, {})

    monkeypatch.setattr(notify.requests, "post", post)
    tg.send("SOLD TAO")
    assert sent == [] and store.get("tg_outbox") == ["SOLD TAO"]
    online["up"] = True
    tg.send("BOUGHT ETH")
    assert sent == ["(delayed, sent while offline)\nSOLD TAO", "BOUGHT ETH"]
    assert store.get("tg_outbox") == []


def test_universe_keeps_last_list_when_offline(settings, monkeypatch):
    saved = {"date": "2026-09-21", "symbols": ["BTCUSDT"], "coins": [], "dropped": []}
    settings.universe_path.write_text(json.dumps(saved))

    def down(*a, **k):
        raise requests.ConnectionError("offline")
    monkeypatch.setattr(universe, "binance_usdt_pairs", down)
    assert universe.build(settings, today="2026-09-22", refresh=True) == saved
