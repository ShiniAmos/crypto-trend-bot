"""Universe v2: judge a coin on its habit, make leaving harder than joining, never collapse.

The rule this replaces ejected ADA, LINK, TRX and XLM on 4 Oct 2026 because each printed one thin
day against a $20M snapshot. These tests pin the behaviour that stops that happening again.
"""

import json

import pytest

from app import data, universe

TOP = [{"symbol": "btc", "id": "bitcoin", "market_cap_rank": 1, "current_price": 60000},
       {"symbol": "eth", "id": "ethereum", "market_cap_rank": 2, "current_price": 3000},
       {"symbol": "ada", "id": "cardano", "market_cap_rank": 3, "current_price": 0.4},
       {"symbol": "sol", "id": "solana", "market_cap_rank": 4, "current_price": 150}]


def _small(monkeypatch, min_names="1"):
    """Settings whose drought top-up cannot mask an ejection in a four-coin fixture."""
    monkeypatch.setenv("UNIVERSE_MIN_NAMES", min_names)
    from app import config as cfg
    return cfg.Settings()


def _wire(monkeypatch, volumes):
    pairs = {c["symbol"].upper(): {"symbol": c["symbol"].upper() + "USDT"} for c in TOP}
    monkeypatch.setattr(universe, "binance_usdt_pairs", lambda: pairs)
    monkeypatch.setattr(universe, "coingecko_top", lambda n: TOP)
    monkeypatch.setattr(data, "ticker_24h",
                        lambda: {s: {"quoteVolume": str(v)} for s, v in volumes.items()})
    monkeypatch.setattr(data, "first_candle_ms", lambda s, tf="1d": 1_500_000_000_000)
    monkeypatch.setattr(data, "now_ms", lambda: 1_800_000_000_000)


BUSY = 50e6
THIN = 5e6


def test_one_thin_day_does_not_eject_an_established_coin(settings, monkeypatch):
    """Six healthy readings then one bad one: the median still clears the bar."""
    _wire(monkeypatch, {s: BUSY for s in ("BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT")})
    for _ in range(6):
        universe.build(settings, refresh=True)
    _wire(monkeypatch, {"BTCUSDT": BUSY, "ETHUSDT": BUSY, "ADAUSDT": THIN, "SOLUSDT": BUSY})
    u = universe.build(settings, refresh=True)
    assert "ADAUSDT" in u["symbols"], "one quiet day must not eject a coin with a healthy habit"


def test_a_sustained_drought_does_eject_it(settings, monkeypatch):
    s2 = _small(monkeypatch)
    _wire(monkeypatch, {s: BUSY for s in ("BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT")})
    universe.build(s2, refresh=True)
    for _ in range(8):           # long enough for the median to turn
        _wire(monkeypatch, {"BTCUSDT": BUSY, "ETHUSDT": BUSY, "ADAUSDT": THIN, "SOLUSDT": BUSY})
        u = universe.build(s2, refresh=True)
    assert "ADAUSDT" not in u["symbols"], "a lasting drought should remove it"


def test_leaving_is_harder_than_joining(settings, monkeypatch):
    """An incumbent is judged against 0.7x the floor, a newcomer against the full floor."""
    s2 = _small(monkeypatch)
    mid = s2.min_volume_usdt * 0.8                # below the entry bar, above the exit bar
    _wire(monkeypatch, {s: BUSY for s in ("BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT")})
    universe.build(s2, refresh=True)              # ADA becomes an incumbent
    for _ in range(5):
        _wire(monkeypatch, {"BTCUSDT": BUSY, "ETHUSDT": BUSY, "ADAUSDT": mid, "SOLUSDT": BUSY})
        u = universe.build(s2, refresh=True)
    assert "ADAUSDT" in u["symbols"], "an incumbent holds its place at 0.8x the floor"

    (s2.data_dir / "universe.json").unlink()          # same volume, but now a newcomer
    (s2.data_dir / "universe_volume.json").unlink()
    _wire(monkeypatch, {"BTCUSDT": BUSY, "ETHUSDT": BUSY, "ADAUSDT": mid, "SOLUSDT": BUSY})
    fresh = universe.build(s2, refresh=True)
    assert "ADAUSDT" not in fresh["symbols"], "a newcomer at the same volume is refused"


def test_a_held_coin_is_never_dropped_for_volume(settings, monkeypatch):
    _wire(monkeypatch, {"BTCUSDT": BUSY, "ETHUSDT": BUSY, "ADAUSDT": 1.0, "SOLUSDT": BUSY})
    u = universe.build(settings, refresh=True, held_symbols={"ADAUSDT"})
    assert "ADAUSDT" in u["symbols"], "selling a position because of a thin day is not a strategy"


def test_the_list_does_not_collapse_in_a_drought(settings, monkeypatch):
    """Everything thin: the list is topped up by volume rather than left nearly empty."""
    monkeypatch.setenv("UNIVERSE_MIN_NAMES", "3")
    from app import config as cfg
    s2 = cfg.Settings()
    _wire(monkeypatch, {"BTCUSDT": BUSY, "ETHUSDT": THIN, "ADAUSDT": THIN * 2, "SOLUSDT": THIN * 3})
    u = universe.build(s2, refresh=True)
    assert len(u["symbols"]) >= 3
    assert "SOLUSDT" in u["symbols"] and "ADAUSDT" in u["symbols"], "the most traded come back first"
    recalled = [c for c in u["coins"] if c.get("recalled")]
    assert recalled, "recalls are labelled, not silent"


def test_volume_history_is_bounded_and_persisted(settings, monkeypatch):
    _wire(monkeypatch, {s: BUSY for s in ("BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT")})
    for _ in range(40):
        universe.build(settings, refresh=True)
    hist = json.loads((settings.data_dir / "universe_volume.json").read_text())
    assert hist["BTCUSDT"], "history is kept"
    assert len(hist["BTCUSDT"]) <= max(settings.volume_window_days * 2, 4), "and bounded"


def test_btc_must_survive_the_filters(settings, monkeypatch):
    s2 = _small(monkeypatch)
    _wire(monkeypatch, {"BTCUSDT": 1.0, "ETHUSDT": BUSY, "ADAUSDT": BUSY, "SOLUSDT": BUSY})
    with pytest.raises(RuntimeError):
        universe.build(s2, refresh=True)
