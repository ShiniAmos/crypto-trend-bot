"""The last honest use of a loss model: normalise risk per trade with the MAE forecast.

What research/loss_model.py established, and it is worth stating plainly because it is the first
model in this repo that predicts anything well:

    predicted MAE vs realised MAE   Spearman +0.3976
    P(net < -20%)                   AUC 0.7819
    P(net < -10%)                   AUC 0.6625
    (for contrast, every RETURN model in this repo: AUC 0.4688 - 0.5219, i.e. nothing)

Losses are forecastable. Returns are not. But the quartile tables killed every way of ACTING on it
by selection, because the dangerous trades are the paying trades:

    by predicted loss10:  safest +6.80%  ...  riskiest +9.60% mean net
    by predicted MAE:     shallowest-expected +9.75%  ...  deepest-expected +6.73%

The model is reading volatility, and volatility is where the payoff is. So selection is dead, and
research/loss_model_verify.py already rejected the one veto that looked alive (survived 1 threshold
out of 6; +0.0085 CAGR at exactly 10% and -0.0179, -0.0195, -0.0277 either side of it).

That leaves the one use that does not select: if the forecast says this entry will go 20%
underwater and that one 6%, they should not carry the same number of dollars. The Donchian stop
distance is the bot's current risk proxy; a forecast with rho = 0.40 against realised MAE may be a
better one.

    qty = equity * risk_pct / (entry - stop_level) * scale          (the engine, unchanged)
    scale = target_mae / predicted_mae, renormalised to mean 1.0

EXPOSURE-NEUTRAL BY CONSTRUCTION. The multiplier is divided by its own mean over scored rows, so
average position size is unchanged and this cannot win by quietly levering up - the failure mode
that made explore_ml_right_tail measure leverage instead of skill.

THE CONTROL THAT MATTERS is not the shuffle, it is the arithmetic. In 4 of 5 earlier studies a
plain ratio beat the model. So the same sizing is run three ways:

    forecast    scale from predicted MAE
    realised    scale from vol_30d, the number the bot already has, same normalisation
    shuffled    the forecast's own multipliers, randomly reassigned, same mean

"forecast" has to beat BOTH to mean anything. Beating only the shuffle means the mapping's shape
did the work; losing to "realised" means the model is an expensive way to read volatility.

    python research/loss_risk_sizing.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from app import backtest  # noqa: E402

import loss_model as lm  # noqa: E402
import loss_model_verify as lv  # noqa: E402

TARGETS = [0.08, 0.10, 0.12]       # the MAE each trade is sized to carry
FLOOR = 0.03                       # a forecast shallower than this would ask for silly size


def scaled(series):
    """Renormalise to mean 1.0 FIRST, then clip, so exposure is unchanged on average.

    The order matters and getting it wrong voided this study's most important control on the first
    run. vol_30d is annualised (median 0.83), so target/vol lands around 0.04 and clipping to
    [0.5, 1.5] before normalising pinned 98.8% of rows to the floor; dividing a constant by its own
    mean then gave exactly 1.0 everywhere, and the "arithmetic" arm silently became the baseline.
    Normalising first makes each arm scale-free, which is what a control for SHAPE requires - the
    units of the two signals are not comparable and are not supposed to be.
    """
    m = series[series.notna()].mean()
    v = series / m if (m and np.isfinite(m)) else series
    return v.clip(0.5, 1.5)


def main():
    s, panel, d = lv.build()
    lm.HP["random_state"] = 0
    d["p_mae"], folds = lm.walk_forward(d, "mae", "reg")
    o = d[d["p_mae"].notna()]
    start = o["bar_time"].min()
    print(f"{folds} folds, {len(o)} scored candidates, out of sample from {start.date()}")
    print(f"predicted vs realised MAE: Spearman "
          f"{o['p_mae'].corr(o['mae'], method='spearman'):+.4f}")

    merged = panel.merge(d[["bar_time", "symbol", "p_mae"]], on=["bar_time", "symbol"], how="left")
    expected = (-merged["p_mae"]).clip(lower=FLOOR)          # positive depth
    realised = (merged["vol_30d"] * np.sqrt(6)).clip(lower=FLOOR)   # ~ a 6-bar move, the proxy

    base = backtest.run(merged, s, flags=lm.SHIPPED, start=start)
    mb = backtest.metrics(base.equity, base.trades, base.exposure)
    rows = [{"run": "baseline", "target": "", **{k: mb[k] for k in
             ("cagr", "sharpe", "max_drawdown")}, "trades": mb.get("trades", 0),
             "mean_scale": 1.0, "note": ""}]

    rng = np.random.default_rng(23)
    for tgt in TARGETS:
        variants = {"forecast": scaled(tgt / expected),
                    "realised": scaled(tgt / realised)}
        sh = variants["forecast"].copy()
        have = sh.notna()
        sh.loc[have] = rng.permutation(sh.loc[have].to_numpy())
        variants["shuffled"] = sh
        for tag, sz in variants.items():
            m2 = merged.copy()
            m2["sz"] = sz.fillna(1.0)
            r = backtest.run(m2, s, flags=lm.SHIPPED, start=start, size_scale_col="sz")
            mm = backtest.metrics(r.equity, r.trades, r.exposure)
            rows.append({"run": tag, "target": f"{tgt:.2f}",
                         "cagr": mm["cagr"], "sharpe": mm["sharpe"],
                         "max_drawdown": mm["max_drawdown"], "trades": mm.get("trades", 0),
                         "mean_scale": float(sz[have].mean()),
                         "note": "null" if tag == "shuffled" else ""})

    out = pd.DataFrame(rows)
    print("\n" + out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    eps = 1e-9
    fc = out[out.run == "forecast"].set_index("target")
    rl = out[out.run == "realised"].set_index("target")
    sh = out[out.run == "shuffled"].set_index("target")
    print("\nper target: does the FORECAST beat the baseline, the arithmetic and the shuffle?")
    verdicts = {}
    for tgt in fc.index:
        f, r_, s_ = fc.loc[tgt], rl.loc[tgt], sh.loc[tgt]
        ship = (f.cagr >= mb["cagr"] - eps and f.sharpe >= mb["sharpe"] - eps
                and f.max_drawdown >= mb["max_drawdown"] - eps)
        beats_arith = f.sharpe >= r_.sharpe - eps and f.cagr >= r_.cagr - eps
        beats_shuf = f.sharpe >= s_.sharpe - eps and f.cagr >= s_.cagr - eps
        verdicts[tgt] = {"ship_rule": bool(ship), "beats_realised": bool(beats_arith),
                         "beats_shuffled": bool(beats_shuf)}
        print(f"  target {tgt}: ship_rule={ship}  beats_realised={beats_arith}  "
              f"beats_shuffled={beats_shuf}")

    ok = [t for t, v in verdicts.items() if all(v.values())]
    print(f"\nVERDICT: {'CANDIDATE - run pit30: ' + ', '.join(ok) if ok else 'REJECT'}")
    (ROOT / "reports" / "loss_risk_sizing.json").write_text(json.dumps(
        {"start": str(start), "runs": out.to_dict("records"), "verdicts": verdicts,
         "any_pass": bool(ok)}, indent=2, default=str))


if __name__ == "__main__":
    main()
