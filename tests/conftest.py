import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402

H4 = 14_400_000
D1 = 86_400_000


def make_bars(n, tf_ms=H4, start_ms=1_500_000_000_000 // H4 * H4, drift=0.0, seed=0, start_price=100.0):
    rng = np.random.default_rng(seed)
    close = start_price * np.exp(np.cumsum(drift + rng.normal(0, 0.01, n)))
    open_ = np.r_[start_price, close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.005, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.005, n))
    ot = start_ms + np.arange(n, dtype="int64") * tf_ms
    vol = rng.uniform(100, 200, n)
    return pd.DataFrame({"open_time": ot, "open": open_, "high": high, "low": low, "close": close,
                         "volume": vol, "quote_volume": vol * close, "taker_buy_volume": vol / 2,
                         "close_time": ot + tf_ms - 1})


def daily_from_4h(b4):
    g = b4.groupby(b4["open_time"] // D1)
    d = pd.DataFrame({"open_time": g["open_time"].first(), "open": g["open"].first(),
                      "high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last(),
                      "volume": g["volume"].sum(), "quote_volume": g["quote_volume"].sum(),
                      "taker_buy_volume": g["taker_buy_volume"].sum()})
    d["close_time"] = d["open_time"] // D1 * D1 + D1 - 1
    return d[g.size() == 6].reset_index(drop=True)


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("MODE", "paper")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    return config.Settings()
