"""Train the rank model for the shadow scorer, re-validate it, and export it.

    python research/refresh_pit.py          # first: bring the survivorship-free cache up to date
    python research/train_rank_model.py     # then: re-validate, fit, export models/rank_model.json

WHAT THIS DEPLOYS. The model found in RESEARCH_LOG.md P22: gradient boosting on the 57 relationship
features, predicting each coin's 24-hour return against the day's average coin, which in research
ranked coins correctly in 29 of 29 quarters - with all of its skill at the bottom of the ranking.

WHAT IS DIFFERENT FROM P22, and why it is re-validated here before anything is exported:

  1. THE UNIVERSE. P22 used app.pit.membership, the top 30 USDT pairs by volume. Since June 2026
     that has included tokenized equities (7 of the top 30 on 19 Nov), and throughout it included
     stablecoins (PAX for 581 member-days, UST 101, U 97, XUSD 19). app.rankuniverse removes both
     before ranking. A different universe is a different training set, so the P22 numbers do not
     automatically carry over - they are re-measured below, walk-forward, on this universe.
  2. THE DATA. The survivorship-free cache now runs to the last closed bar. Everything after
     2026-10-04 is data no experiment in this repository has seen, and it is scored here once,
     as the fresh period.

THE FEATURE PIPELINE IS app.rankfeatures, the same code the live scorer runs, so the model is
trained on exactly the inputs it will be given.

THE EXPORTED MODEL is fitted the way every walk-forward fold is: a fresh estimator on the most
recent 730 days that have labels, minus the purge gap, labels clipped to that window's 1st and 99th
percentiles. It is written only if app.rankmodel can reproduce its predictions exactly.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config, data, pit, rankfeatures as rf, rankmodel, rankuniverse as ru  # noqa: E402

TOP_N = 30
TRAIN_DAYS = 730
TEST_DAYS = 91
DAY_NS = 86_400 * 10**9
BAR_NS = 4 * 3600 * 10**9
DEV_END = pd.Timestamp("2025-01-01", tz="UTC")
SEEN_END = pd.Timestamp("2026-10-04 08:00", tz="UTC")     # where every earlier experiment stopped
OUT = ROOT / "models" / "rank_model.json"
PREDS = ROOT / "reports" / "_rank_v2_preds.parquet"


def make_model():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.06, max_iter=250,
                                         early_stopping=False, random_state=0)


def build_training_set(s):
    kdir = s.data_dir / "pit" / "klines"
    syms = [x for x in sorted({f.name.rsplit("_", 1)[0] for f in kdir.glob("*_1d.parquet")})
            if pit.keep(x, s.exclude_memes)]
    vol, close, listed = ru.daily_frames(kdir, syms, data.load)
    btc4 = data.load(kdir, "BTCUSDT", "4h")
    last_bar = pd.to_datetime(btc4["close_time"].max() + 1, unit="ms", utc=True)
    member = ru.membership(vol, close, TOP_N, listed, extend_to=last_bar)
    ever = sorted(set(member.columns[member.any()]) | {"BTCUSDT"})
    print(f"  {len(syms)} USDT pairs, {len(ru.tokenized_stocks(syms, listed))} tokenized stocks "
          f"excluded, {len(ever)} coins ever in the top {TOP_N}")
    print("  features for every bar of every coin that was ever a member (several minutes)...")
    full = rf.build(kdir, ever, s.donchian_bars)
    full["fwd"] = rf.forward_returns(full, rf.HORIZON)
    p = rf.add_market_state(rf.attach_members(full, member))
    p = p.sort_values(["bar_time", "symbol"]).reset_index(drop=True)
    p["y"] = p["fwd"] - p.groupby("bar_time")["fwd"].transform("mean")
    return p, member


def walk(X, y, t, t_to):
    """The P22 walk: a fresh model per fold, rolling 730-day window, purge, fold-local clipping."""
    purge = (rf.HORIZON + 1) * BAR_NS
    window = TRAIN_DAYS * DAY_NS
    ok = ~np.isnan(y)
    cut = int(t[ok].min()) + window + purge
    idx_all, pred_all = [], []
    while cut < t_to:
        end = cut + TEST_DAYS * DAY_NS
        tr = ok & (t >= cut - purge - window) & (t < cut - purge)
        te = (t >= cut) & (t < end)
        cut = end
        if tr.sum() < 3000 or te.sum() < 150:
            continue
        lo, hi = np.quantile(y[tr], [0.01, 0.99])
        m = make_model().fit(X[tr], np.clip(y[tr], lo, hi))
        idx_all.append(np.flatnonzero(te))
        pred_all.append(m.predict(X[te]))
        del m
    return np.concatenate(idx_all), np.concatenate(pred_all)


def report(d, periods):
    print(f"\n  {'period':26s} {'quarters':>8s} {'mean IC':>8s} {'positive':>9s} {'z':>6s} | "
          f"{'lowest third':>12s} {'middle':>8s} {'top third':>9s}  (24h return vs average coin)")
    out = {}
    for label, lo, hi in periods:
        w = d[(d["bar_time"] >= lo) & (d["bar_time"] < hi)].dropna(subset=["y"])
        if w.empty:
            continue
        q = w["bar_time"].dt.tz_localize(None).dt.to_period("Q")
        ic = w.groupby(q).apply(lambda g: g["pred"].corr(g["y"], method="spearman"))
        z = ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic))) if len(ic) > 1 else float("nan")
        t3 = pd.cut(w["pct"], [0, 1 / 3, 2 / 3, 1.0001], labels=["low", "mid", "top"])
        by = w.groupby(t3, observed=False)["y"].mean()
        out[label] = {"quarters": int(len(ic)), "ic": float(ic.mean()), "pos": int((ic > 0).sum()),
                      "z": float(z), "low": float(by["low"]), "mid": float(by["mid"]),
                      "top": float(by["top"])}
        print(f"  {label:26s} {len(ic):8d} {ic.mean():+8.4f} {int((ic > 0).sum()):4d}/{len(ic):<4d} "
              f"{z:6.2f} | {by['low']:+12.3%} {by['mid']:+8.3%} {by['top']:+9.3%}")
    return out


def main():
    t0 = time.time()
    s = config.load()
    print("building the training set on the corrected universe...")
    p, member = build_training_set(s)
    names = rf.feature_names(p)
    X = rf.design(p, names)
    y = p["y"].to_numpy(dtype=float)
    t = p["bar_time"].astype("int64").to_numpy()
    print(f"  {len(p):,} member rows, {p.symbol.nunique()} coins, {len(names)} features, "
          f"{p.bar_time.min().date()} to {p.bar_time.max().date()}  ({time.time() - t0:.0f}s)")

    # ---- re-validation: walk-forward on THIS universe --------------------------------------
    print("\nwalk-forward re-validation (fresh model per fold)...")
    idx, pred = walk(X, y, t, int(t.max()) + 1)
    d = p.iloc[idx][["bar_time", "symbol", "y"]].copy()
    d["pred"] = pred
    d["pct"] = d.groupby("bar_time")["pred"].rank(pct=True)
    d.to_parquet(PREDS)
    last = p["bar_time"].max()
    rep = report(d, (("development (to 2024)", d.bar_time.min(), DEV_END),
                     ("holdout 2025 -> 2026-10-04", DEV_END, SEEN_END),
                     ("FRESH 2026-10-04 -> now", SEEN_END, last + pd.Timedelta(hours=4))))
    print("\n  P22 on the old universe, for comparison: development IC +0.076 (22/22),")
    print("  holdout +0.063 (7/7); lowest decile -0.68% / -0.87% a day.")

    # What a working model looks like ONE BAR A DAY since 2025. The forward record holds about one
    # independent observation a day (30 coins move together; 24h windows overlap), so these set
    # how long a shadow evaluation must run before it can say anything. Stored with the model and
    # read by app/rankreport.py.
    from app.rankreport import days_to_verdict
    daily = d[(d["bar_time"] >= DEV_END) & (d["bar_time"].dt.hour == 0)].dropna(subset=["y"])
    ic_d = daily.groupby("bar_time").apply(lambda g: g["pred"].corr(g["y"], method="spearman"),
                                           include_groups=False).dropna()
    sp_d = (daily[daily["pct"] <= 1 / 3].groupby("bar_time")["y"].mean()
            - daily[daily["pct"] > 1 / 3].groupby("bar_time")["y"].mean()).dropna()
    forward = {"ic_mean": float(ic_d.mean()), "ic_sd": float(ic_d.std()),
               "spread_mean": float(sp_d.mean()), "spread_sd": float(sp_d.std())}
    print(f"\n  one bar a day since 2025 ({len(ic_d)} days): rank correlation "
          f"{forward['ic_mean']:+.4f} (sd {forward['ic_sd']:.4f}) -> about "
          f"{days_to_verdict(forward['ic_mean'], forward['ic_sd'])} forward days to z = 2;")
    print(f"  bottom third minus the rest {forward['spread_mean']:+.3%} (sd {forward['spread_sd']:.3%}) "
          f"-> about {days_to_verdict(forward['spread_mean'], forward['spread_sd'])} days")

    ok = all(r["ic"] > 0 for r in rep.values()) and rep.get(
        "holdout 2025 -> 2026-10-04", {}).get("z", 0) > 2
    if not ok:
        print("\nRE-VALIDATION FAILED on the corrected universe: no model is exported.")
        return 1

    # ---- the deployed model: the most recent window, fitted the way every fold is -----------
    labelled = ~np.isnan(y)
    t_last = int(t[labelled].max())
    tr = labelled & (t > t_last - TRAIN_DAYS * DAY_NS)
    lo, hi = np.quantile(y[tr], [0.01, 0.99])
    print(f"\nfitting the deployed model on {int(tr.sum()):,} rows, "
          f"{pd.Timestamp(int(t[tr].min()), tz='UTC').date()} to "
          f"{pd.Timestamp(t_last, tz='UTC').date()}...")
    model = make_model().fit(X[tr], np.clip(y[tr], lo, hi))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    meta = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "trained_through": str(pd.Timestamp(t_last, tz="UTC")),
            "train_rows": int(tr.sum()), "train_days": TRAIN_DAYS, "horizon_bars": rf.HORIZON,
            "label": "24h forward return minus the bar's average member, clipped to [1%, 99%]",
            "label_clip": [float(lo), float(hi)], "universe": f"rankuniverse top {TOP_N} coins",
            "revalidation": rep, "forward_expectation": forward}
    mid, worst = rankmodel.export(model, names, X[tr][-20000:], OUT, meta=meta)
    print(f"  exported {OUT.relative_to(ROOT)}: model {mid}, {len(model._predictors)} trees, "
          f"export check max difference {worst:.1e}")
    print(f"\ndone in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
