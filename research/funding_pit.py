"""The decisive test: does the funding-crowding filter survive the survivorship-free panel?

Three features in this project passed the standard and held-out panels and then died here, because
the pit panel rebuilds the universe from every USDT pair Binance ever listed - including the ones
that went to zero - instead of from today's survivors.

Funding history is fetched for every coin that was ever in the daily top 30. Coins with no
perpetual contract keep their candidates: a filter cannot act on data that does not exist, and
pretending otherwise would flatter the result.

    python research/funding_pit.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import backtest, config, pit  # noqa: E402
from app.flags import Flags  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SHIPPED = Flags(abs_momentum=True, volume_min=0.8, taker_min=0.5)
FAPI = "https://fapi.binance.com/fapi/v1/fundingRate"
PER_DAY = 3
CUT = 0.8


def fetch(symbol, out_dir):
    path = out_dir / f"{symbol}.parquet"
    if path.exists():
        return True
    rows, start = [], 1_500_000_000_000
    while True:
        try:
            r = requests.get(FAPI, params={"symbol": symbol, "startTime": start, "limit": 1000},
                             timeout=30)
            if r.status_code >= 400:
                return False
            page = r.json()
        except requests.RequestException:
            return False
        if not page:
            break
        rows += page
        if len(page) < 1000:
            break
        start = int(page[-1]["fundingTime"]) + 1
        time.sleep(0.1)
    if not rows:
        return False
    df = pd.DataFrame(rows)
    df["funding_time"] = df["fundingTime"].astype("int64")
    df["funding_rate"] = df["fundingRate"].astype(float)
    (df[["funding_time", "funding_rate"]].drop_duplicates("funding_time")
     .sort_values("funding_time").reset_index(drop=True).to_parquet(path, index=False))
    return True


def attach(panel, fdir):
    out = []
    for sym, g in panel.groupby("symbol", sort=False):
        g = g.sort_values("bar_time")
        path = fdir / f"{sym}.parquet"
        if not path.exists():
            out.append(g.assign(funding_7d=np.nan))
            continue
        f = pd.read_parquet(path)
        f["bar_time"] = pd.to_datetime(f["funding_time"], unit="ms", utc=True)
        f = f.sort_values("bar_time")
        f["funding_7d"] = f["funding_rate"].rolling(7 * PER_DAY).mean()
        out.append(pd.merge_asof(g, f[["bar_time", "funding_7d"]], on="bar_time",
                                 direction="backward", allow_exact_matches=True))
    p = pd.concat(out, ignore_index=True).sort_values(["bar_time", "symbol"]).reset_index(drop=True)
    el = p["eligible"] & p["funding_7d"].notna()
    p["funding_pct"] = np.nan
    p.loc[el, "funding_pct"] = p[el].groupby("bar_time")["funding_7d"].rank(pct=True)
    return p


def main():
    s = config.load()
    print("building the survivorship-free panel (this takes several minutes)...")
    res, panel, _ = pit.run(s, top_n=30, log=print, flags=SHIPPED)
    base = backtest.metrics(res.equity, res.trades, res.exposure)
    rb_b = backtest.random_baselines(panel, res.trades, s) or {}
    print(f"\npit30 baseline: CAGR {base['cagr']:.4f}  Sharpe {base['sharpe']:.4f}  "
          f"DD {base['max_drawdown']:.4f}  trades {base['trades']}  "
          f"p_coin {rb_b.get('p_value_vs_random_coin', float('nan')):.3f}")

    fdir = s.data_dir / "funding_pit"
    fdir.mkdir(parents=True, exist_ok=True)
    syms = sorted(panel["symbol"].unique())
    print(f"\nfetching funding history for {len(syms)} coins...")
    ok = sum(fetch(sym, fdir) for sym in syms)
    print(f"  {ok} of {len(syms)} have a perpetual contract")

    p = attach(panel, fdir)
    cov = p.loc[p["eligible"], "funding_pct"].notna().mean()
    print(f"  funding known for {cov:.1%} of eligible rows")

    v = p.copy()
    v["breakout"] = v["breakout"] & ~(v["funding_pct"] > CUT).fillna(False)
    r2 = backtest.run(v, s, flags=SHIPPED)
    cand = backtest.metrics(r2.equity, r2.trades, r2.exposure)
    rb_c = backtest.random_baselines(v, r2.trades, s) or {}

    print(f"\npit30 + funding filter: CAGR {cand['cagr']:.4f}  Sharpe {cand['sharpe']:.4f}  "
          f"DD {cand['max_drawdown']:.4f}  trades {cand['trades']}  "
          f"p_coin {rb_c.get('p_value_vs_random_coin', float('nan')):.3f}")

    checks = {"CAGR": cand["cagr"] >= base["cagr"],
              "Sharpe": cand["sharpe"] >= base["sharpe"],
              "maxDD": cand["max_drawdown"] >= base["max_drawdown"],
              "p_coin": rb_c.get("p_value_vs_random_coin", 1) <=
                        max(rb_b.get("p_value_vs_random_coin", 0.05), 0.05) + 1e-9}
    print("\n-> " + ("PASS" if all(checks.values()) else "FAIL") + "  " +
          "  ".join(f"{k}:{'ok' if val else 'NO'}" for k, val in checks.items()))

    (ROOT / "reports" / "funding_pit.json").write_text(json.dumps(
        {"baseline": {k: str(v2) for k, v2 in base.items()},
         "candidate": {k: str(v2) for k, v2 in cand.items()},
         "p_coin_baseline": rb_b.get("p_value_vs_random_coin"),
         "p_coin_candidate": rb_c.get("p_value_vs_random_coin"),
         "coverage": cov, "checks": checks}, indent=2, default=str))


if __name__ == "__main__":
    main()
