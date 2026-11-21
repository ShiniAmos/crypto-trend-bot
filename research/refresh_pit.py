"""Bring the survivorship-free kline cache up to the last closed bar.

    python research/refresh_pit.py

The pit cache under data/pit/klines is what every survivorship-free result in this repository is
built on, and it stops wherever the last download stopped. Training a model that will score live
coins needs it current, and the bars added by a refresh are the only data in the project that no
experiment has looked at yet - so this is also where a fresh out-of-sample check comes from.

Appending is safe for everything already published: backfill_symbol only adds bars after the last
stored one, so no historical row changes, and the cached panels under reports/ are separate files
that this does not touch.

New listings are picked up by re-reading Binance's public archive index, with the same exclusions
the pit universe has always had (stablecoins, fiat, wrapped, leveraged and meme coins).
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config, data, pit, rankuniverse as ru  # noqa: E402


def main(top_n=30):
    s = config.load()
    kdir = s.data_dir / "pit" / "klines"
    before = data.load(kdir, "BTCUSDT", "4h")
    was = pd.to_datetime(before["close_time"].max(), unit="ms", utc=True) if len(before) else None
    print(f"pit cache currently ends {was}")

    have = sorted({p.name.rsplit("_", 1)[0] for p in kdir.glob("*_1d.parquet")})
    try:
        listed = [x for x in pit.all_usdt_symbols() if pit.keep(x, s.exclude_memes)]
    except Exception as e:                                   # noqa: BLE001
        print(f"archive index unavailable ({e}); refreshing the {len(have)} symbols already cached")
        listed = []
    syms = sorted(set(have) | set(listed))
    print(f"{len(syms)} USDT pairs ({len(set(listed) - set(have))} not seen before)")

    pit.fetch(s, syms, "1d", log=print)
    keep = [x for x in syms if pit.keep(x, s.exclude_memes)]
    # 4h bars are needed for every coin either universe rule ever admits. The volume rule alone
    # (app.pit) is what the older research panels use; the coin rule (app.rankuniverse) replaces
    # excluded stocks and pegged tokens with the next coins by volume, and on 21 Nov 2026 three of
    # those - BICO, HEI, LOKA - had never been in the volume top 30 and so had never been
    # downloaded. Training silently lacked them while the live scorer, which fetches its own
    # candles, would have included them.
    member = pit.membership(kdir, keep, top_n)
    vol, close, listed = ru.daily_frames(kdir, keep, data.load)
    coins = ru.membership(vol, close, top_n, listed)
    ever = sorted(set(member.columns[member.any()]) | set(coins.columns[coins.any()]) | {"BTCUSDT"})
    print(f"{len(ever)} coins have been in either top {top_n} at some point; refreshing their 4h bars")
    pit.fetch(s, ever, "4h", log=print)

    after = data.load(kdir, "BTCUSDT", "4h")
    now = pd.to_datetime(after["close_time"].max(), unit="ms", utc=True)
    print(f"pit cache now ends {now}  (+{len(after) - len(before)} BTC bars)")
    # The raw volume top 30 includes tokenized equities and pegged tokens; the model's universe
    # does not (app/rankuniverse.py), so that is the list worth printing.
    vol, close, listed = ru.daily_frames(kdir, keep, data.load)
    coins = ru.membership(vol, close, top_n, listed)
    today = coins.index.max()
    print(f"today's top {top_n} COINS ({today.date()}): "
          f"{', '.join(sorted(coins.columns[coins.loc[today]]))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
