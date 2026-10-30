"""Does app/forecast.py, the SHIPPED code, still have the skill the research measured?

research/forecast_volatility.py and research/forecast_regime.py established the skill. This file
re-runs the same walk-forward design through the module that actually ships, so a refactor that
quietly breaks the model fails here instead of in the live bot. It imports app.forecast's own
fit/predict functions - nothing is reimplemented.

Three things are measured, each against its naive baseline:
    volatility    pooled out-of-sample log-R2 of the HAR fit vs "next vol = vol_30d" and "= vol_7d"
    regime        AUC of the 2-coefficient logistic vs persistence ("it stays as it is")
    direction     the honesty control: sign of BTC's next 7 days. Expect ~0.50. If this comes back
                  at 0.56 the harness leaks and nothing above it is believable.

    cd crypto-bot && PYTHONIOENCODING=utf-8 python research/forecast_module_check.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config, features, forecast  # noqa: E402

TRAIN_DAYS, TEST_DAYS, PURGE_DAYS = 730, 91, 10


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def auc(y, p):
    """Rank AUC via the Mann-Whitney identity, with ties at half credit."""
    y = np.asarray(y, int)
    if y.min() == y.max():
        return float("nan")
    r = pd.Series(p).rank().to_numpy()
    n1, n0 = int(y.sum()), int((1 - y).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def folds(first, last):
    purge = pd.Timedelta(days=PURGE_DAYS)
    t0 = first + pd.Timedelta(days=TRAIN_DAYS) + purge
    while t0 < last:
        t1 = t0 + pd.Timedelta(days=TEST_DAYS)
        yield t0 - purge - pd.Timedelta(days=TRAIN_DAYS), t0 - purge, t0, min(t1, last)
        t0 = t1


def check_vol(panel):
    v = forecast.vol_frame(panel)
    d = v[v["y"].notna()].reset_index(drop=True)
    rows, per_fold = [], []
    for a, b, c, e in folds(d["bar_time"].min(), d["bar_time"].max()):
        tr = d[(d["bar_time"] > a) & (d["bar_time"] <= b)]
        te = d[(d["bar_time"] > c) & (d["bar_time"] <= e)]
        if len(tr) < 2000 or len(te) < 200:
            continue
        fit = forecast.fit_vol(tr)
        f = te.assign(pred=forecast.predict_vol(fit, te))
        # The level correction k is a constant shift in log space. It is what makes the LEVEL
        # forecast unbiased - which is what the UI shows - but it costs a little log-R2 because the
        # OLS is already unbiased there. Both are reported so the number is comparable with
        # research/forecast_volatility.py, which scored the uncorrected log prediction.
        f["pred_nolevel"] = f["pred"] / fit["k"]
        rows.append(f)
        per_fold.append({"test_start": str(c),
                         "r2_model": r2(np.log(f["fwd_vol"]), np.log(f["pred"])),
                         "r2_model_nolevel": r2(np.log(f["fwd_vol"]), np.log(f["pred_nolevel"])),
                         "r2_v30": r2(np.log(f["fwd_vol"]), np.log(f["vol_30d"]))})
    oos = pd.concat(rows, ignore_index=True)
    ly = np.log(oos["fwd_vol"])
    out = {"n_oos": len(oos), "n_folds": len(per_fold),
           "span": [str(oos["bar_time"].min()), str(oos["bar_time"].max())],
           "log_r2_model": r2(ly, np.log(oos["pred"])),
           "log_r2_model_nolevel": r2(ly, np.log(oos["pred_nolevel"])),
           "log_r2_naive_vol30": r2(ly, np.log(oos["vol_30d"])),
           "log_r2_naive_vol7": r2(ly, np.log(oos["vol_7d"])),
           "level_r2_model": r2(oos["fwd_vol"], oos["pred"]),
           "level_r2_naive_vol30": r2(oos["fwd_vol"], oos["vol_30d"]),
           "rmse_log_model": float(np.sqrt(np.mean((ly - np.log(oos["pred"])) ** 2))),
           "rmse_log_naive_vol30": float(np.sqrt(np.mean((ly - np.log(oos["vol_30d"])) ** 2))),
           "spearman_pred_realised": float(oos["pred"].corr(oos["fwd_vol"], method="spearman")),
           "folds_model_beats_vol30": int(sum(f["r2_model"] > f["r2_v30"] for f in per_fold)),
           "folds_nolevel_beats_vol30":
               int(sum(f["r2_model_nolevel"] > f["r2_v30"] for f in per_fold))}
    out["margin_vs_vol30"] = out["log_r2_model"] - out["log_r2_naive_vol30"]
    out["margin_nolevel_vs_vol30"] = out["log_r2_model_nolevel"] - out["log_r2_naive_vol30"]
    return out


def check_regime(panel):
    b = forecast.btc_frame(panel)
    d = b[b["y"].notna()]
    rows, won = [], 0
    for a, lo, c, e in folds(d.index.min(), d.index.max()):
        tr = d[(d.index > a) & (d.index <= lo)]
        te = d[(d.index > c) & (d.index <= e)]
        if len(tr) < 1000 or len(te) < 50:
            continue
        fit = forecast.fit_regime(tr)
        p = np.array([forecast.predict_regime(fit, x) for x in te["x"]])
        f = pd.DataFrame({"y": te["y"].astype(int).to_numpy(), "p": p,
                          "pers": te["state"].astype(float).to_numpy()})
        rows.append(f)
        am, ap = auc(f["y"], f["p"]), auc(f["y"], f["pers"])
        won += int(np.isfinite(am) and np.isfinite(ap) and am > ap)
    oos = pd.concat(rows, ignore_index=True)
    am, ap = auc(oos["y"], oos["p"]), auc(oos["y"], oos["pers"])
    return {"n_oos": len(oos), "n_folds": len(rows), "base_rate_on": float(oos["y"].mean()),
            "auc_model": am, "auc_naive_persistence": ap, "margin_auc": am - ap,
            "accuracy_model": float(((oos["p"] > 0.5).astype(int) == oos["y"]).mean()),
            "accuracy_naive_persistence": float((oos["pers"].astype(int) == oos["y"]).mean()),
            "brier_model": float(np.mean((oos["p"] - oos["y"]) ** 2)),
            "brier_naive_persistence": float(np.mean((oos["pers"] - oos["y"]) ** 2)),
            "folds_model_beats_persistence_auc": won}


def check_direction(panel):
    """Same folds, same single feature, target = sign of BTC's next 7 days. The control."""
    b = forecast.btc_frame(panel)
    b = b.assign(up=(b["close"].shift(-forecast.HORIZON) > b["close"]))
    d = b[b["up"].notna() & b["x"].notna()]
    rows = []
    for a, lo, c, e in folds(d.index.min(), d.index.max()):
        tr, te = d[(d.index > a) & (d.index <= lo)], d[(d.index > c) & (d.index <= e)]
        if len(tr) < 1000 or len(te) < 50:
            continue
        fit = forecast.fit_regime(tr.assign(y=tr["up"]))
        rows.append(pd.DataFrame({"y": te["up"].astype(int).to_numpy(),
                                  "p": [forecast.predict_regime(fit, x) for x in te["x"]]}))
    oos = pd.concat(rows, ignore_index=True)
    return {"n_oos": len(oos), "base_rate_up": float(oos["y"].mean()),
            "auc_model": auc(oos["y"], oos["p"]), "n_folds": len(rows)}


def main():
    s = config.load()
    u = json.loads((ROOT / "reports" / "_parity_universe.json").read_text())
    panel = features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars)

    v, g, dr = check_vol(panel), check_regime(panel), check_direction(panel)

    print("== VOLATILITY (7-day forward realised, pooled over coins) ==")
    print(f"  {v['n_oos']:,} out-of-sample rows, {v['n_folds']} folds, {v['span'][0][:10]} to "
          f"{v['span'][1][:10]}")
    print(f"  log-R2  module {v['log_r2_model']:.4f}   naive vol_30d {v['log_r2_naive_vol30']:.4f}"
          f"   naive vol_7d {v['log_r2_naive_vol7']:.4f}")
    print(f"  margin over vol_30d {v['margin_vs_vol30']:+.4f}, won in "
          f"{v['folds_model_beats_vol30']}/{v['n_folds']} folds")
    print(f"  before the level correction (the research's number): {v['log_r2_model_nolevel']:.4f}, "
          f"margin {v['margin_nolevel_vs_vol30']:+.4f}, "
          f"{v['folds_nolevel_beats_vol30']}/{v['n_folds']} folds")
    print(f"  level-space R2 {v['level_r2_model']:.4f} vs {v['level_r2_naive_vol30']:.4f}")
    print(f"  log RMSE {v['rmse_log_model']:.4f} vs {v['rmse_log_naive_vol30']:.4f} "
          f"({1 - v['rmse_log_model'] / v['rmse_log_naive_vol30']:+.1%}), "
          f"spearman {v['spearman_pred_realised']:.4f}")

    print("\n== REGIME PERSISTENCE (is BTC risk-on in 7 days?) ==")
    print(f"  {g['n_oos']:,} rows, {g['n_folds']} folds, base rate on {g['base_rate_on']:.3f}")
    print(f"  AUC   module {g['auc_model']:.4f}   naive persistence {g['auc_naive_persistence']:.4f}"
          f"   margin {g['margin_auc']:+.4f}  ({g['folds_model_beats_persistence_auc']}"
          f"/{g['n_folds']} folds)")
    print(f"  accuracy {g['accuracy_model']:.4f} vs {g['accuracy_naive_persistence']:.4f}   "
          f"Brier {g['brier_model']:.4f} vs {g['brier_naive_persistence']:.4f}")

    print("\n== DIRECTION CONTROL (sign of BTC's next 7 days) ==")
    print(f"  AUC {dr['auc_model']:.4f} on {dr['n_oos']:,} rows (base rate up "
          f"{dr['base_rate_up']:.3f}). ~0.50 is the honest answer and the proof the harness is not "
          f"leaking.")

    ok = (v["margin_vs_vol30"] > 0.05 and g["margin_auc"] > 0.01
          and abs(dr["auc_model"] - 0.5) < 0.06)
    print("\nVERDICT:", "the shipped module reproduces the measured skill."
          if ok else "the shipped module has LOST the skill - investigate before trusting the UI.")
    out = {"volatility": v, "regime": g, "direction_control": dr, "reproduces": bool(ok),
           "note": "skill only. research/forecast_integration.py showed none of it becomes P&L; "
                   "app/forecast.py therefore changes no trading decision."}
    (ROOT / "reports" / "forecast_module_check.json").write_text(json.dumps(out, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
