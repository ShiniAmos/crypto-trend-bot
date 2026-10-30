"""Judge the strategy in the market it is actually trading in, not in 2020.

Calendar-year attribution (10 Oct 2026) showed the edge is 2019-2021: survivorship-free, those
three years compound at +28.70% a year and 2022-2026 at +2.13%. Every research program in this
repository until now scored candidates on the FULL sample, where those three years dominate the
objective. That has a consequence nobody checked:

    a change that helped 2023-2025 but diluted 2020 was rejected - for the wrong reason.

~655 variants were rejected under that objective. This script re-opens the question with the
objective the user actually trades against.

THE MODERN ERA is 2022-01-01 onward, and it is split so that re-testing old rejections here cannot
quietly become a second fit to a second sample:

    DISCOVER   2022-01-01 .. 2024-12-31   where a flag is allowed to look good
    VALIDATE   2025-01-01 .. now          where it has to still be good

A flag only becomes a candidate if it clears the baseline on BOTH halves. Three years of discovery
is a thin sample, which is exactly why the validate half is not optional and why anything that
passes here is a candidate for more work, never a ship.

EXPOSURE IS REPORTED EVERYWHERE, because "stands in the market and does not stop" is a stated
requirement and the current configuration took ZERO trades in 2022. avg_exposure is the fraction of
the time capital is actually at work. A variant that earns the same return while being invested
more of the time is better for the operator even when the ship rule calls it a tie.

TWO QUESTIONS, in order:

  1. Do the four SHIPPED flags still earn their place? Each is removed in turn (leave-one-out). Any
     whose removal improves the modern era was fitted to a market that has gone.
  2. Do the REJECTED flags look different here? Each is added in turn to the shipped set. The list
     leans on the ones with a reason to be era-dependent: BTC-led exposure (btc_momentum,
     btc_sleeve), faster exits for shorter trends (exit_donchian, exit_rank, time_stop_bars),
     and continuous risk-scaled exposure instead of on/off (vol_target, regime_risk,
     equity_curve_risk, hold_through_regime).

The pit panel is cached to reports/_pit_panel.parquet on first run; delete that file to rebuild.

    python research/modern_era.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from app import backtest, config, features, pit  # noqa: E402
import funding_pit  # noqa: E402  - the pit funding cache and its percentile ranking
from app.flags import Flags  # noqa: E402

SHIPPED = dict(abs_momentum=True, volume_min=0.8, taker_min=0.5, funding_max_pct=0.8)
MODERN = "2022-01-01"
DISCOVER_END = "2024-12-31"
VALIDATE_START = "2025-01-01"
def cache_path(exit_donchian=0):
    tag = "" if not exit_donchian else f"_x{exit_donchian}"
    return ROOT / "reports" / f"_pit_panel{tag}.parquet"


# Added one at a time on top of the shipped set; every one was rejected on the full sample.
# `exit_donchian` needs its own panel - the breakdown_exit column is built at panel-build time from
# that specific window - so those entries carry the window rather than only a flag.
# `vol_target` is a BOOLEAN switch in signals.vol_scale (the magnitude is never read), so it is
# tested once; passing 0.5 and 0.8 produced identical rows on the first run, correctly.
REOPEN = [
    ("btc_momentum", dict(btc_momentum=True), 0),
    ("btc_sleeve", dict(btc_sleeve=True), 0),
    ("breadth", dict(breadth=True), 0),
    ("rs_cap=2.0", dict(rs_cap=2.0), 0),
    ("corr_cap=0.8", dict(corr_cap=0.8), 0),
    ("vol_target", dict(vol_target=1.0), 0),
    ("regime_risk", dict(regime_risk=True), 0),
    ("equity_curve_risk", dict(equity_curve_risk=True), 0),
    ("hold_through_regime", dict(hold_through_regime=True), 0),
    # one value only: each needs its own 590MB panel, and one is enough to learn the sign
    ("exit_donchian=30", dict(exit_donchian=30), 30),
    ("exit_rank=5", dict(exit_rank=5), 0),
    ("exit_rank=15", dict(exit_rank=15), 0),
    ("time_stop_bars=30", dict(time_stop_bars=30), 0),
    ("time_stop_bars=60", dict(time_stop_bars=60), 0),
    ("scale_out_r", dict(scale_out_r=True), 0),
    ("scale_out_pct", dict(scale_out_pct=True), 0),
    ("stop_exec", dict(stop_exec=True), 0),
    ("atr_band", dict(atr_band=True), 0),
]


def build_pit(s, top_n=30, exit_donchian=0):
    """The survivorship-free panel, with funding attached so funding_max_pct is not a no-op.

    Two columns here are built, not inferred, and both were silently missing on the first run:
      funding_pct      signals.extra_filters falls back to "nothing is crowded" when it is absent,
                       so funding_max_pct=0.8 - a SHIPPED flag, in the baseline - did nothing.
      breakdown_exit   built by features.build_panel from a specific exit window, so each
                       exit_donchian value needs its own panel. One panel made 10 and 30 identical.
    """
    cp = cache_path(exit_donchian)
    if cp.exists():
        return pd.read_parquet(cp)
    kdir = s.data_dir / "pit" / "klines"
    syms = sorted({p.name.rsplit("_", 1)[0] for p in kdir.glob("*_1d.parquet")})
    syms = [x for x in syms if pit.keep(x, s.exclude_memes)]
    member = pit.membership(kdir, syms, top_n)
    ever = sorted(set(member.columns[member.any()]) | {"BTCUSDT"})
    print(f"building pit panel (exit_donchian={exit_donchian}): "
          f"{len(ever)} coins ever in the top {top_n}...")
    panel = features.build_panel(kdir, ever, s.donchian_bars, exit_donchian=exit_donchian)
    panel["day"] = (panel["bar_time"] - pd.Timedelta(milliseconds=1)).dt.normalize()
    m = member.rename_axis(index="day", columns="symbol").stack().rename("member").reset_index()
    panel = panel.merge(m, on=["day", "symbol"], how="left")
    panel["member"] = panel["member"].astype("boolean").fillna(False).astype(bool)
    panel["eligible"] = panel["eligible"] & (panel["member"] | (panel["symbol"] == "BTCUSDT"))
    panel = features.score(panel.drop(columns=["score", "rank", "n_eligible"]))
    panel = funding_pit.attach(panel, s.data_dir / "funding_pit")
    cov = panel.loc[panel["eligible"], "funding_pct"].notna().mean()
    print(f"  funding attached: {cov:.1%} of eligible rows have a crowding percentile")
    panel.to_parquet(cp)
    return panel


def run(panel, s, flags, start, end=None):
    r = backtest.run(panel, s, flags=flags, start=pd.Timestamp(start, tz="UTC"),
                     end=pd.Timestamp(end, tz="UTC") if end else None)
    m = backtest.metrics(r.equity, r.trades, r.exposure)
    return {"cagr": m["cagr"], "sharpe": m["sharpe"], "maxdd": m["max_drawdown"],
            "trades": m.get("trades", 0), "win": m.get("win_rate"),
            "exposure": m.get("avg_exposure")}


def both_halves(panel, s, flags):
    out = {}
    for tag, lo, hi in (("disc", MODERN, DISCOVER_END), ("val", VALIDATE_START, None),
                        ("all", MODERN, None)):
        for k, v in run(panel, s, flags, lo, hi).items():
            out[f"{tag}_{k}"] = v
    return out


def verdict(row, base, eps=1e-9):
    """Ship rule on BOTH halves of the modern era. No epsilon generosity on a 3-year sample.

    INERT is reported before anything else. A variant whose three metrics match the baseline to
    1e-12 has not tied - it has not run, and calling that a pass is how a disconnected wire gets
    promoted. This guard exists because the first version of this script did exactly that for
    exit_donchian=10 and =30.
    """
    if all(abs(row[f"all_{k}"] - base[f"all_{k}"]) < 1e-12
           for k in ("cagr", "sharpe", "maxdd")):
        return "INERT - no effect"
    ok = []
    for tag in ("disc", "val"):
        ok.append(row[f"{tag}_cagr"] >= base[f"{tag}_cagr"] - eps
                  and row[f"{tag}_sharpe"] >= base[f"{tag}_sharpe"] - eps
                  and row[f"{tag}_maxdd"] >= base[f"{tag}_maxdd"] - eps)
    return "CANDIDATE" if all(ok) else ("disc only" if ok[0] else ("val only" if ok[1] else "no"))


def show(df, cols, title):
    print(f"\n{title}")
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main():
    s = config.load()
    panel = build_pit(s)
    print(f"panel: {panel['bar_time'].min().date()} .. {panel['bar_time'].max().date()}, "
          f"{panel['symbol'].nunique()} symbols")

    base = both_halves(panel, s, Flags(**SHIPPED))
    print(f"\nMODERN baseline (shipped flags, {MODERN} onward)")
    print(f"  discover 2022-2024: CAGR {base['disc_cagr']:+.2%}  Sharpe {base['disc_sharpe']:.4f}"
          f"  maxDD {base['disc_maxdd']:.2%}  exposure {base['disc_exposure']:.1%}"
          f"  trades {base['disc_trades']}")
    print(f"  validate 2025-26  : CAGR {base['val_cagr']:+.2%}  Sharpe {base['val_sharpe']:.4f}"
          f"  maxDD {base['val_maxdd']:.2%}  exposure {base['val_exposure']:.1%}"
          f"  trades {base['val_trades']}")
    print(f"  whole modern era  : CAGR {base['all_cagr']:+.2%}  Sharpe {base['all_sharpe']:.4f}"
          f"  maxDD {base['all_maxdd']:.2%}  exposure {base['all_exposure']:.1%}"
          f"  trades {base['all_trades']}")

    # ---- question 1: do the shipped flags still earn their place here?
    loo = []
    for name in SHIPPED:
        f = {k: v for k, v in SHIPPED.items() if k != name}
        r = both_halves(panel, s, Flags(**f))
        r["removed"] = name
        r["modern_better_without"] = bool(r["all_sharpe"] > base["all_sharpe"]
                                          and r["all_cagr"] > base["all_cagr"])
        loo.append(r)
    ldf = pd.DataFrame(loo)
    show(ldf, ["removed", "all_cagr", "all_sharpe", "all_maxdd", "all_exposure",
               "modern_better_without"],
         "Q1 - leave one shipped flag out (modern era). True = the flag HURTS today:")

    # ---- question 2: do the full-sample rejections look different here?
    rows = []
    for name, extra, xd in REOPEN:
        try:
            pan = panel if not xd else build_pit(s, exit_donchian=xd)
            r = both_halves(pan, s, Flags(**{**SHIPPED, **extra}))
        except Exception as e:
            print(f"  {name}: skipped ({type(e).__name__}: {e})")
            continue
        r["flag"] = name
        r["verdict"] = verdict(r, base)
        r["d_sharpe"] = r["all_sharpe"] - base["all_sharpe"]
        r["d_cagr"] = r["all_cagr"] - base["all_cagr"]
        r["d_expo"] = r["all_exposure"] - base["all_exposure"]
        rows.append(r)
    rdf = pd.DataFrame(rows).sort_values("d_sharpe", ascending=False)
    show(rdf, ["flag", "all_cagr", "all_sharpe", "all_maxdd", "all_exposure", "all_trades",
               "d_sharpe", "d_cagr", "d_expo", "verdict"],
         "Q2 - full-sample rejections, re-scored on the modern era (sorted by Sharpe gain):")

    cands = rdf[rdf["verdict"] == "CANDIDATE"]
    print(f"\nclear both halves of the modern era: {len(cands)}")
    if len(cands):
        show(cands, ["flag", "disc_cagr", "disc_sharpe", "val_cagr", "val_sharpe", "all_exposure"],
             "  candidates, both halves:")
    print("\nNOTE: a modern-era candidate is NOT a ship. Three years of discovery and under two of "
          "validation is a thin sample; anything here needs cost stress, placebo nulls and a "
          "flag-off parity check before it goes near app/.")

    (ROOT / "reports" / "modern_era.json").write_text(json.dumps(
        {"baseline": base, "leave_one_out": loo, "reopened": rdf.to_dict("records"),
         "candidates": cands["flag"].tolist()}, indent=2, default=str))


if __name__ == "__main__":
    main()
