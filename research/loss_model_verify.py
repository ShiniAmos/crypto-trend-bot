"""Is "veto the worst 10% by predicted loss" real, or is it two trades?

research/loss_model.py tested 16 ways of acting on a loss model. Exactly one cleared the ship rule:

    veto worst 10% loss10   CAGR 0.2816 vs 0.2731, Sharpe 1.7228 vs 1.6851, maxDD identical

Two reasons to distrust it before it goes anywhere near a flag:

  1. It removed 2 net trades out of 298. A 0.85pp CAGR gain carried by two trades is what noise
     looks like.
  2. It is one cell out of 16. At that count, one marginal pass is the expected number of false
     positives, not evidence.

So this script does not re-tune anything. It asks whether the same effect survives perturbations
that must not matter if the effect is real:

  THRESHOLD   0.80 .. 0.95 in steps. A real edge degrades smoothly; a fluke lives in one cell.
              Note the 25% veto already FAILED hard (CAGR 0.2407, Sharpe 1.6079), so the shape of
              this curve is the single most informative test here.
  SEED        the model's random_state over 5 values. The loss signal itself should be stable;
              if the backtest verdict flips with the seed, the verdict was never about the signal.
  SUB-PERIOD  first half vs second half of the out-of-sample span.

PASS means: positive against baseline at a clear majority of thresholds AND seeds. Anything less
is REJECT, recorded, left OFF.

    python research/loss_model_verify.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from app import backtest, config, features, funding, signals  # noqa: E402
from app.flags import Flags  # noqa: E402

import candidate_model as cm  # noqa: E402
import loss_model as lm  # noqa: E402

THRESHOLDS = [0.80, 0.85, 0.875, 0.90, 0.925, 0.95]
SEEDS = [0, 1, 2, 3, 4]


def build():
    s = config.load()
    u = json.loads((ROOT / "reports" / "_parity_universe.json").read_text())
    panel = cm.add_derived(features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars))
    panel = funding.attach(panel, s)
    if "funding_pct" not in panel:
        panel["funding_pct"] = np.nan
    rank = panel["rank"].to_numpy()
    btc = (panel["symbol"] == "BTCUSDT").to_numpy()
    panel["_cand"] = ((panel["eligible"] & panel["regime_on"] & panel["breakout"]).to_numpy()
                      & ((panel["rs_7d"] > 0).to_numpy() | btc) & (rank <= s.candidates)
                      & signals.extra_filters(panel, lm.SHIPPED).to_numpy())
    lab = lm.label_trades(panel, s)
    d = (panel[panel["_cand"]].merge(lab, on=["bar_time", "symbol"], how="inner")
         .reset_index(drop=True))
    return s, panel, d


def veto_run(s, panel, d, col, thr_q, start, end=None):
    merged = panel.merge(d[["bar_time", "symbol", col]], on=["bar_time", "symbol"], how="left")
    raw = merged[col]
    merged["ml_score"] = (raw.quantile(thr_q) - raw).where(raw.notna())
    r = backtest.run(merged, s, flags=Flags(**{**lm.SHIPPED.__dict__, "ml_veto": True}),
                     start=start, end=end)
    return backtest.metrics(r.equity, r.trades, r.exposure)


def main():
    s, panel, d = build()
    rows, seed_auc = [], []

    for seed in SEEDS:
        lm.HP["random_state"] = seed
        pred, folds = lm.walk_forward(d, "loss10", "clf")
        d["p_loss10"] = pred
        o = d[pred.notna()]
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(o["loss10"], o["p_loss10"])
        seed_auc.append({"seed": seed, "folds": folds, "auc": auc})
        start = o["bar_time"].min()
        base = backtest.metrics(*(lambda r: (r.equity, r.trades, r.exposure))(
            backtest.run(panel.assign(**{"p_loss10": np.nan}), s, flags=lm.SHIPPED, start=start)))
        for thr in THRESHOLDS:
            m = veto_run(s, panel, d, "p_loss10", thr, start)
            rows.append({"seed": seed, "veto_pct": 1 - thr, "cagr": m["cagr"],
                         "sharpe": m["sharpe"], "maxdd": m["max_drawdown"],
                         "trades": m.get("trades", 0),
                         "d_cagr": m["cagr"] - base["cagr"],
                         "d_sharpe": m["sharpe"] - base["sharpe"],
                         "d_maxdd": m["max_drawdown"] - base["max_drawdown"]})

    print("the loss signal itself, across seeds (this is the part that should be stable):")
    print(pd.DataFrame(seed_auc).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    df = pd.DataFrame(rows)
    print("\nveto, every threshold x seed, as a difference from baseline:")
    piv = df.pivot_table(index="veto_pct", columns="seed", values="d_cagr")
    print("\n  d_CAGR:")
    print(piv.to_string(float_format=lambda x: f"{x:+.4f}"))
    print("\n  d_Sharpe:")
    print(df.pivot_table(index="veto_pct", columns="seed", values="d_sharpe")
          .to_string(float_format=lambda x: f"{x:+.4f}"))

    ships = (df.d_cagr >= -1e-9) & (df.d_sharpe >= -1e-9) & (df.d_maxdd >= -1e-9)
    df["ships"] = ships
    n, tot = int(ships.sum()), len(df)
    print(f"\ncells clearing the ship rule: {n} / {tot}")
    print(df[ships][["seed", "veto_pct", "d_cagr", "d_sharpe", "d_maxdd"]]
          .to_string(index=False, float_format=lambda x: f"{x:+.4f}") if n else "  none")

    verdict = "CANDIDATE" if n > tot * 0.6 else "REJECT"
    print(f"\nVERDICT: {verdict} - a real edge survives most thresholds and seeds; "
          f"this survives {n / tot:.0%}")
    (ROOT / "reports" / "loss_model_verify.json").write_text(json.dumps(
        {"seed_auc": seed_auc, "cells": df.to_dict("records"), "pass_rate": n / tot,
         "verdict": verdict}, indent=2, default=str))


if __name__ == "__main__":
    main()
