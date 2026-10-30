"""A model aimed at the LOSSES, used only to make bad trades smaller.

Every earlier model in this repo predicted something good and tried to buy more of it:

    candidate_model.py      win = net > 0            -> inverted (confident quartile +3.65%,
                                                        unconfident +8.79%)
    explore_ml_right_tail   net > +20%, net > +50%    -> REJECT
    ml_veto / app/ml.py     excess return, regression -> REJECT

That inversion is the reason to try the other side. If the model's confidence runs BACKWARDS to
profit, the model is not blind - it is reading trade SIZE, and reading it correctly. A trend
system's winners are a few huge ones, so "likely to win" selects the small, tidy, mean-reverting
trades. The same information pointed at the left tail is the question actually asked here: not
"which coin will go up" but "which of these entries is about to lose, so put less into it".

TARGETS (all four new; none has been tested in this repo)
    loss10     binary: will this trade end below -10% net?
    loss20     binary: below -20%?
    mae        regression on MAXIMUM ADVERSE EXCURSION - the deepest the trade goes underwater
               between entry and exit. The most direct reading of "loses less", and it uses the
               path rather than the endpoint, so a trade that fell 30% and recovered is still
               labelled dangerous.
    downside   regression on min(net, 0). Asymmetric by construction: all upside collapses to one
               label, so the fit spends its whole capacity on the losses.

HOW IT IS ALLOWED TO ACT - size DOWN only, multiplier in [0.5, 1.0].
This is the deliberate difference from every earlier study. Mapping a prediction to a multiplier
that can exceed 1.0 raises return by arithmetic alone, which is how explore_ml_right_tail ended up
measuring leverage instead of skill. Capping at 1.0 means the model can only ever REMOVE risk, so
any improvement has to come from removing it where the losses actually were.

CONTROLS, each one a way for this to be nothing:
    const 0.75 / 0.50   every entry smaller. The "just trade less" null. A size-down cuts return
                        and drawdown together, so beating this line is the whole test.
    shuffled            the model's own multipliers, randomly reassigned among its own rows. Kills
                        the claim that the average multiplier did the work.
    veto worst 25%/10%  the filtering formulation, included because precision in PICKING was asked
                        for, not only in sizing - and because it is the formulation most likely to
                        delete a winner, which is worth measuring rather than assuming.

SHIP RULE, unchanged: CAGR >= baseline AND Sharpe >= baseline AND maxDD not worse. A pure size-down
mechanically lowers CAGR, so passing requires the shrunk trades to be disproportionately losers.
That is a hard bar and it is the right one.

    python research/loss_model.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from app import backtest, config, features, funding, signals  # noqa: E402
from app.flags import Flags  # noqa: E402

import candidate_model as cm  # noqa: E402  - same features, same derived columns, same harness

SHIPPED = Flags(abs_momentum=True, volume_min=0.8, taker_min=0.5, funding_max_pct=0.8)
TRAIN_DAYS, TEST_DAYS, PURGE_DAYS = 730, 91, 30
FEATS = cm.FEATS + cm.DERIVED + ["funding_pct"]
HP = dict(max_iter=250, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=25,
          l2_regularization=1.0, random_state=0)


def label_trades(panel, s):
    """Net return AND maximum adverse excursion for the trade each candidate bar becomes.

    The exit rule mirrors the live one (regime off, breakdown, or rank slipping past 10) and both
    legs cross the spread, so the label is the trade the bot would really have had. MAE walks the
    lows from the entry bar to the exit bar, which is why it cannot be derived from `net`.
    """
    fee, slip = s.fee_rate, s.slippage
    rows = []
    for sym, g in panel.groupby("symbol", sort=False):
        g = g.sort_values("bar_time").reset_index(drop=True)
        o, low = g["open"].to_numpy(), g["low"].to_numpy()
        exit_cond = ((~g["regime_on"].fillna(False).to_numpy())
                     | g["breakdown"].fillna(False).to_numpy()
                     | (g["rank"].fillna(999).to_numpy() > 10))
        exits = np.flatnonzero(exit_cond)
        for i in np.flatnonzero(g["_cand"].to_numpy()):
            if i + 1 >= len(g):
                continue
            entry = o[i + 1] * (1 + slip)
            nxt = exits[np.searchsorted(exits, i + 1, side="left"):]
            j = nxt[0] if len(nxt) else len(g) - 2
            if j + 1 >= len(g):
                continue
            exit_px = o[j + 1] * (1 - slip)
            net = (exit_px * (1 - fee)) / (entry * (1 + fee)) - 1
            worst = low[i + 1:j + 2].min()
            rows.append({"bar_time": g["bar_time"].iloc[i], "symbol": sym, "net": net,
                         "mae": worst / entry - 1.0, "bars_held": j - i})
    d = pd.DataFrame(rows)
    d["loss10"] = (d["net"] < -0.10).astype(int)
    d["loss20"] = (d["net"] < -0.20).astype(int)
    d["downside"] = d["net"].clip(-0.5, 0.0)
    return d


def walk_forward(d, target, kind):
    """Out-of-sample prediction of `target`, 730d train / 91d test, 30d purge gap."""
    X = d[FEATS].replace([np.inf, -np.inf], np.nan)
    y = d["mae"].clip(-0.6, 0.0) if target == "mae" else d[target]
    t = d["bar_time"]
    out = pd.Series(np.nan, index=d.index)
    test_start, folds = t.min() + pd.Timedelta(days=TRAIN_DAYS), 0
    while test_start < t.max():
        test_end = test_start + pd.Timedelta(days=TEST_DAYS)
        tr = (t >= test_start - pd.Timedelta(days=TRAIN_DAYS)) & \
             (t < test_start - pd.Timedelta(days=PURGE_DAYS))
        te = (t >= test_start) & (t < test_end)
        if tr.sum() >= 150 and te.any() and (kind == "reg" or y[tr].nunique() > 1):
            m = (HistGradientBoostingClassifier(**HP) if kind == "clf"
                 else HistGradientBoostingRegressor(**HP))
            m.fit(X[tr], y[tr])
            out.loc[te] = (m.predict_proba(X[te])[:, 1] if kind == "clf" else m.predict(X[te]))
            folds += 1
        test_start = test_end
    return out, folds


def diagnose(o, col, target):
    """Does the prediction order the losses, out of sample? Printed before any backtest."""
    v = o[o[col].notna()]
    line = {"signal": col, "n": len(v)}
    if target in ("loss10", "loss20"):
        if v[target].nunique() > 1:
            line["auc"] = roc_auc_score(v[target], v[col])
            line["brier"] = brier_score_loss(v[target], v[col].clip(0, 1))
        line["base_rate"] = v[target].mean()
    else:
        line["spearman_vs_mae"] = v[col].corr(v["mae"], method="spearman")
    line["spearman_vs_net"] = v[col].corr(v["net"], method="spearman")
    q = pd.qcut(v[col], 4, labels=["safest", "safe", "risky", "riskiest"], duplicates="drop")
    tab = v.groupby(q, observed=True).agg(trades=("net", "size"), mean_net=("net", "mean"),
                                          mean_mae=("mae", "mean"),
                                          loss10_rate=("loss10", "mean"))
    return line, tab


def main():
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
                      & signals.extra_filters(panel, SHIPPED).to_numpy())
    print(f"candidate bars: {int(panel['_cand'].sum())}")

    lab = label_trades(panel, s)
    print(f"labelled trades: {len(lab)}   mean net {lab['net'].mean():+.2%}   "
          f"mean MAE {lab['mae'].mean():+.2%}   below -10%: {lab['loss10'].mean():.1%}   "
          f"below -20%: {lab['loss20'].mean():.1%}")

    d = (panel[panel["_cand"]].merge(lab, on=["bar_time", "symbol"], how="inner")
         .reset_index(drop=True))
    specs = [("loss10", "clf"), ("loss20", "clf"), ("mae", "reg"), ("downside", "reg")]
    for target, kind in specs:
        pred, folds = walk_forward(d, target, kind)
        d[f"p_{target}"] = pred
        print(f"{target}: {folds} folds, {int(pred.notna().sum())} scored")

    o = d[d["p_loss10"].notna()].copy()
    print(f"\nout of sample from {o['bar_time'].min().date()}, {len(o)} candidates\n")

    diags, tables = [], {}
    for target, _ in specs:
        line, tab = diagnose(o, f"p_{target}", target)
        diags.append(line)
        tables[target] = tab
    print(pd.DataFrame(diags).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    for target, tab in tables.items():
        print(f"\nby predicted {target} (riskiest = model says this one loses):")
        print(tab.to_string(float_format=lambda x: f"{x:.4f}"))

    # ---------------------------------------------------------------- backtests
    merged = panel.merge(o[["bar_time", "symbol"] + [f"p_{t}" for t, _ in specs]],
                         on=["bar_time", "symbol"], how="left")
    start = o["bar_time"].min()
    runs = []

    def record(tag, m, note=""):
        runs.append({"run": tag, "cagr": m["cagr"], "sharpe": m["sharpe"],
                     "maxdd": m["max_drawdown"], "trades": m.get("trades", 0),
                     "win": m.get("win_rate"), "note": note})

    base = backtest.run(merged, s, flags=SHIPPED, start=start)
    mb = backtest.metrics(base.equity, base.trades, base.exposure)
    record("baseline", mb)

    for c in (0.75, 0.50):
        m2 = merged.copy()
        m2["sz"] = c
        r = backtest.run(m2, s, flags=SHIPPED, start=start, size_scale_col="sz")
        record(f"const {c:.2f}", backtest.metrics(r.equity, r.trades, r.exposure), "null")

    rng = np.random.default_rng(11)
    for target, _ in specs:
        col = f"p_{target}"
        # riskiest -> 0.5x, safest -> 1.0x. Sign: for loss probabilities HIGH is bad; for the
        # mae/downside regressions LOW (more negative) is bad, so those are inverted first.
        raw = merged[col] if target.startswith("loss") else -merged[col]
        sz = (1.0 - 0.5 * raw.rank(pct=True)).clip(0.5, 1.0)

        m3 = merged.copy()
        m3["sz"] = sz.fillna(1.0)
        r = backtest.run(m3, s, flags=SHIPPED, start=start, size_scale_col="sz")
        record(f"size-down {target}", backtest.metrics(r.equity, r.trades, r.exposure))

        m4 = merged.copy()
        shuf = sz.copy()
        have = shuf.notna()
        shuf.loc[have] = rng.permutation(shuf.loc[have].to_numpy())
        m4["sz"] = shuf.fillna(1.0)
        r = backtest.run(m4, s, flags=SHIPPED, start=start, size_scale_col="sz")
        record(f"  shuffled {target}", backtest.metrics(r.equity, r.trades, r.exposure), "null")

        for keep in (0.75, 0.90):
            m5 = merged.copy()
            thr = raw.quantile(keep)
            m5["ml_score"] = (thr - raw).where(raw.notna())   # positive = keep
            r = backtest.run(m5, s, flags=Flags(**{**SHIPPED.__dict__, "ml_veto": True}),
                             start=start)
            record(f"veto worst {1 - keep:.0%} {target}",
                   backtest.metrics(r.equity, r.trades, r.exposure))

    out = pd.DataFrame(runs)
    print(f"\nbacktests from {start.date()}:")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    eps = 1e-9
    out["passes"] = ((out.cagr >= mb["cagr"] - eps) & (out.sharpe >= mb["sharpe"] - eps)
                     & (out.maxdd >= mb["max_drawdown"] - eps) & (out.run != "baseline")
                     & (out.note != "null"))
    winners = out[out.passes]
    print("\nship rule (CAGR >=, Sharpe >=, maxDD not worse):")
    print("  none pass - REJECT" if winners.empty
          else winners[["run", "cagr", "sharpe", "maxdd"]].to_string(index=False))
    (ROOT / "reports" / "loss_model.json").write_text(json.dumps(
        {"start": str(start), "diagnostics": diags, "runs": out.to_dict("records"),
         "any_pass": bool(not winners.empty)}, indent=2, default=str))


if __name__ == "__main__":
    main()
