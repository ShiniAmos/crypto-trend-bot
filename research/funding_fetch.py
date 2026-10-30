"""Download perpetual-futures funding rate history - the first data in this project that is not
derived from spot candles.

Funding is what leveraged longs pay shorts (or the reverse) every 8 hours to hold a perpetual
position. It is a direct measure of CROWDING: when everyone is long with leverage, funding goes
positive and longs bleed. Nothing in OHLCV contains it, which is why it is worth a look after
nine families of price-derived features all failed.

History starts September 2019 for BTC, later for coins listed afterwards.

    python research/funding_fetch.py
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FAPI = "https://fapi.binance.com/fapi/v1/fundingRate"
START = 1_500_000_000_000          # well before the first funding payment


def fetch_symbol(symbol, out_dir):
    """All funding payments for one symbol, paged 1000 at a time."""
    rows, start = [], START
    while True:
        try:
            r = requests.get(FAPI, params={"symbol": symbol, "startTime": start, "limit": 1000},
                             timeout=30)
            if r.status_code == 400:
                return None            # no perpetual contract for this coin
            r.raise_for_status()
            page = r.json()
        except requests.RequestException as e:
            print(f"  {symbol}: {e}")
            return None
        if not page:
            break
        rows += page
        last = int(page[-1]["fundingTime"])
        if len(page) < 1000:
            break
        start = last + 1
        time.sleep(0.12)               # stay well inside the rate limit
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["funding_time"] = df["fundingTime"].astype("int64")
    df["funding_rate"] = df["fundingRate"].astype(float)
    df = (df[["funding_time", "funding_rate"]]
          .drop_duplicates("funding_time").sort_values("funding_time").reset_index(drop=True))
    df.to_parquet(out_dir / f"{symbol}.parquet", index=False)
    return df


def main():
    s = config.load()
    u = json.loads((ROOT / "reports" / "_parity_universe.json").read_text())
    out = s.data_dir / "funding"
    out.mkdir(parents=True, exist_ok=True)

    got, missing = [], []
    for sym in u["symbols"]:
        df = fetch_symbol(sym, out)
        if df is None or df.empty:
            missing.append(sym)
            print(f"  {sym:<10} no perpetual funding history")
            continue
        first = pd.to_datetime(df["funding_time"].iloc[0], unit="ms").date()
        got.append(sym)
        print(f"  {sym:<10} {len(df):>6,} payments from {first}  "
              f"mean {df['funding_rate'].mean() * 100:+.4f}% per 8h")
    print(f"\n{len(got)} symbols with funding history, {len(missing)} without: {missing}")


if __name__ == "__main__":
    main()
