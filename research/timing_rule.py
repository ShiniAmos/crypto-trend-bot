"""Can the next 4-hour move of BTC and ETH be timed? The lead the coin-picking search left behind.

    python research/timing_rule.py

WHY THIS IS THE NEXT TRY. predict_search.py scored 102 candidates. The coin-PICKING signal it found
was real on 17 surviving coins and lost money on the survivorship-free universe, because its most
confident calls were crash-buying and the coins that did not come back were not in the sample
(signal_rule_pit.py: the same rule, holdout -68%, with LUNA bought three times on the way to zero).

One row of that search was a different kind of claim and was never priced: "is up over the next 4
hours" scored AUC 0.558 in development for the linear model on the relationship features. That is
not a statement about which coin - every coin moves with the market over four hours - it is a
statement about WHEN. And timing has no survivorship problem if it is applied to BTC and ETH,
because nothing is being picked out of a list of survivors.

It is also a suspiciously large number. A market-timing AUC of 0.558 would be worth a great deal,
and this project's own placebo found the existing rules' entry timing to be a coin flip
(P20: p_time 0.496 and 0.512). So the first job here is to find out whether 0.558 is information
or pooling: scored across 17 coins that rise and fall together, one correct market call counts 17
times. The honest score is per coin, on BTC and on ETH alone.

THE PROTOCOL, fixed before the numbers exist:

  1. INFORMATION. On BTC and on ETH separately, compare the next bar's return when the model's
     score is in its top fifth against its bottom fifth, where "top fifth" is judged against the
     previous 180 days of scores only. Reported for development, holdout and fresh data. If the
     difference is not there on the holdout, nothing below matters.
  2. A RULE, chosen on development only: long when the score is above a trailing quantile q, hold
     K bars, otherwise cash; q in {0.5, 0.7, 0.9}, K in {1, 3, 6}. Nine rules; the one with the
     highest development Sharpe net of 0.27% a round trip is taken, by code.
  3. That one rule, ONCE on the holdout and ONCE on fresh data, against simply holding the coin.
     It works only if it beats buy-and-hold's Sharpe on the holdout AND makes more per unit of
     time invested than holding does on both periods.

The model is the search's, unchanged: regularised logistic regression on the 57 relationship
features, 730-day rolling window, fresh fit per fold, purge gap, pooled across the 17 coins.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from app import config, features, ml  # noqa: E402

import dots_features as dots  # noqa: E402
import predict_search as ps  # noqa: E402
from train_predict_grid import add_targets  # noqa: E402

COINS = ("BTCUSDT", "ETHUSDT")
QS = (0.5, 0.7, 0.9)
KS = (1, 3, 6)
LOOKBACK = 180 * 6
BARS_PER_YEAR = 6 * 365
SIDE = ps.COST / 2


def predictions(kind="linear"):
    s = config.load()
    kdir = Path(s.data_dir) / "klines_plus"
    syms = sorted({f.name.rsplit("_", 1)[0] for f in kdir.glob("*_4h.parquet")})
    p = add_targets(dots.add_dots(features.build_panel(kdir, syms, s.donchian_bars)), 1)
    X = dots.design_dots(p, ml.design).to_numpy(dtype=float)
    y = p["t_abs_sign"].to_numpy(dtype=float)
    t = p["bar_time"].astype("int64").to_numpy()
    r = ps.walk(X, y, t, "auc", kind, 1, 0, int(t.max()) + 1 + 4 * ps.BAR_NS,
                keep_pred=True, threads=12)
    d = p.iloc[r["idx"]][["bar_time", "symbol", "fwd"]].copy()
    d["pred"] = r["pred"]
    d["up"] = (d["fwd"] > 0).astype(float)
    return d.sort_values(["symbol", "bar_time"]).reset_index(drop=True), r


def trailing_rank(x, window=LOOKBACK, min_periods=360):
    """Where each score sits among the scores of the previous `window` bars (0..1), causally."""
    s = pd.Series(x)
    return s.rolling(window + 1, min_periods=min_periods).apply(
        lambda w: (w[:-1] < w[-1]).mean(), raw=True)


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    y, p = np.asarray(y), np.asarray(p)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def info_row(c, lo, hi):
    w = c[(c["bar_time"] >= lo) & (c["bar_time"] < hi)].dropna(subset=["rank", "fwd"])
    if len(w) < 200:
        return None
    top, bot = w[w["rank"] >= 0.8]["fwd"], w[w["rank"] <= 0.2]["fwd"]
    diff = top.mean() - bot.mean()
    se = np.sqrt(top.var(ddof=1) / len(top) + bot.var(ddof=1) / len(bot))
    return {"bars": len(w), "auc": auc(w["up"], w["pred"]), "top": top.mean(), "bot": bot.mean(),
            "diff": diff, "t": diff / se if se > 0 else float("nan"), "all": w["fwd"].mean()}


def simulate(c, q, k, lo, hi):
    """Long when the score's trailing rank is at least q, held k bars; cash otherwise."""
    w = c[(c["bar_time"] >= lo) & (c["bar_time"] < hi)].dropna(subset=["fwd"]).reset_index(drop=True)
    if len(w) < 200:
        return None
    left, pos, rets, trades = 0, [], [], 0
    for rank, fwd in zip(w["rank"].to_numpy(), w["fwd"].to_numpy()):
        cost = 0.0
        fire = rank == rank and rank >= q
        if fire:
            if left <= 0:
                cost += SIDE
                trades += 1
            left = k
        elif left == 0 and pos and pos[-1] == 1:
            cost += SIDE                           # the position closed at this bar's open
        held = 1 if left > 0 else 0
        rets.append(held * fwd - cost)
        pos.append(held)
        left = max(left - 1, 0)
    r = pd.Series(rets)
    eq = (1 + r).cumprod()
    bh = (1 + w["fwd"]).cumprod()
    years = max(len(w) / BARS_PER_YEAR, 1e-9)
    expo = float(np.mean(pos))
    sharpe = float(r.mean() / r.std() * np.sqrt(BARS_PER_YEAR)) if r.std() > 0 else 0.0
    bh_sharpe = float(w["fwd"].mean() / w["fwd"].std() * np.sqrt(BARS_PER_YEAR))
    return {"total": float(eq.iloc[-1] - 1), "cagr": float(eq.iloc[-1] ** (1 / years) - 1),
            "sharpe": sharpe, "maxdd": float((eq / eq.cummax() - 1).min()), "expo": expo,
            "trades": trades, "years": years,
            "bh_total": float(bh.iloc[-1] - 1), "bh_sharpe": bh_sharpe,
            "bh_maxdd": float((bh / bh.cummax() - 1).min()),
            # per unit of time invested: what the rule earned while in, against what holding earns
            "in_market_bar": float(r[np.array(pos) == 1].mean()) if expo > 0 else float("nan"),
            "bh_bar": float(w["fwd"].mean())}


def main():
    print("walk-forward predictions (every one out of sample)...")
    d, r = predictions("linear")
    print(f"  pooled over 17 coins: AUC {np.mean(r['scores']):.4f} across {len(r['scores'])} folds "
          f"- the number that looked like a lead")
    first, last = d["bar_time"].min(), d["bar_time"].max()
    dev_hi = ps.DEV_END - pd.Timedelta(hours=4 * 8)
    hold_hi = ps.HOLD_END - pd.Timedelta(hours=4 * 8)
    periods = (("development", first, dev_hi), ("holdout 2025-26", ps.DEV_END, hold_hi),
               ("fresh data", ps.HOLD_END, last + pd.Timedelta(hours=4)))

    per_coin = {}
    for sym in COINS:
        c = d[d["symbol"] == sym].sort_values("bar_time").reset_index(drop=True)
        c["rank"] = trailing_rank(c["pred"].to_numpy())
        per_coin[sym] = c

    # ---- 1. is there information, coin by coin --------------------------------------------------
    print("\n" + "=" * 100)
    print("1. INFORMATION. Next-bar return when the score is in its top fifth vs its bottom fifth,")
    print("   each coin on its own (the pooled 0.558 counts one market call seventeen times).")
    print()
    print(f"  {'coin':8s} {'period':17s} {'bars':>6s} {'AUC':>7s} {'top fifth':>10s} "
          f"{'bottom fifth':>13s} {'difference':>11s} {'t':>6s} {'any bar':>9s}")
    info = {}
    for sym in COINS:
        for label, lo, hi in periods:
            row = info_row(per_coin[sym], lo, hi)
            info[(sym, label)] = row
            if row is None:
                print(f"  {sym:8s} {label:17s}  too few bars")
                continue
            print(f"  {sym:8s} {label:17s} {row['bars']:6d} {row['auc']:7.4f} {row['top']:+10.3%} "
                  f"{row['bot']:+13.3%} {row['diff']:+11.3%} {row['t']:6.2f} {row['all']:+9.3%}")

    # ---- 2. nine rules on development, per coin ---------------------------------------------------
    print("\n" + "=" * 100)
    print("2. NINE RULES ON DEVELOPMENT, net of 0.27% a round trip. The best Sharpe is taken, by code.")
    chosen = {}
    for sym in COINS:
        print(f"\n  {sym}")
        print(f"  {'enter above':>12s} {'hold':>5s} {'CAGR':>9s} {'Sharpe':>7s} {'maxDD':>8s} "
              f"{'invested':>9s} {'trades':>7s}")
        best = None
        for q in QS:
            for k in KS:
                sres = simulate(per_coin[sym], q, k, first, dev_hi)
                if sres is None:
                    continue
                print(f"  {'rank ' + format(q, '.0%'):>12s} {k * 4:4d}h {sres['cagr']:+9.2%} "
                      f"{sres['sharpe']:7.2f} {sres['maxdd']:+8.2%} {sres['expo']:9.1%} "
                      f"{sres['trades']:7d}")
                if sres["trades"] >= 100 and (best is None or sres["sharpe"] > best[2]["sharpe"]):
                    best = (q, k, sres)
        if best:
            chosen[sym] = best
            b = best[2]
            print(f"  buy and hold: CAGR {(1 + b['bh_total']) ** (1 / b['years']) - 1:+.2%}, "
                  f"Sharpe {b['bh_sharpe']:.2f}, maxDD {b['bh_maxdd']:+.2%}")
            print(f"  CHOSEN: enter above rank {best[0]:.0%}, hold {best[1] * 4}h "
                  f"(development Sharpe {b['sharpe']:.2f} against {b['bh_sharpe']:.2f} for holding)")

    # ---- 3. the chosen rule, once on each period it was not chosen on -----------------------------
    print("\n" + "=" * 100)
    print("3. THE CHOSEN RULE, ONCE, ON DATA IT WAS NOT CHOSEN ON.")
    verdicts = {}
    for sym in COINS:
        if sym not in chosen:
            continue
        q, k, _ = chosen[sym]
        print(f"\n  {sym}: enter above rank {q:.0%}, hold {k * 4}h")
        print(f"  {'period':17s} {'rule total':>11s} {'Sharpe':>7s} {'maxDD':>8s} {'invested':>9s} "
              f"{'trades':>7s} | {'hold total':>11s} {'Sharpe':>7s} {'maxDD':>8s} | "
              f"{'per bar in':>11s} {'per bar hold':>13s}")
        ok = True
        for label, lo, hi in periods:
            sres = simulate(per_coin[sym], q, k, lo, hi)
            if sres is None:
                print(f"  {label:17s}  too few bars")
                ok = False
                continue
            print(f"  {label:17s} {sres['total']:+11.2%} {sres['sharpe']:7.2f} {sres['maxdd']:+8.2%} "
                  f"{sres['expo']:9.1%} {sres['trades']:7d} | {sres['bh_total']:+11.2%} "
                  f"{sres['bh_sharpe']:7.2f} {sres['bh_maxdd']:+8.2%} | "
                  f"{sres['in_market_bar']:+11.4%} {sres['bh_bar']:+13.4%}")
            if label != "development":
                better_per_bar = sres["in_market_bar"] > sres["bh_bar"]
                if label.startswith("holdout"):
                    ok &= sres["sharpe"] > sres["bh_sharpe"] and better_per_bar
                else:
                    ok &= better_per_bar
        verdicts[sym] = ok

    # ---- verdict ------------------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("VERDICT, by the criteria at the top of this file.")
    print()
    for sym in COINS:
        h = info.get((sym, "holdout 2025-26"))
        info_ok = h is not None and h["t"] > 2
        print(f"  {sym}: information on the holdout "
              f"(top fifth minus bottom fifth {('%+.3f%%' % (h['diff'] * 100)) if h else 'n/a'}, "
              f"t {('%.2f' % h['t']) if h else 'n/a'}) -> {'present' if info_ok else 'ABSENT'}; "
              f"rule -> {'pass' if verdicts.get(sym) else 'FAIL'}")
    print()
    if any(verdicts.values()):
        print("  A timing rule passed on data it was not chosen on. Before it is believed it must")
        print("  clear the same gates as everything else: costs at 2x, a placebo of random entry")
        print("  times, and forward data that does not exist yet.")
    else:
        print("  No timing rule passed. If the information row is also absent on the holdout, the")
        print("  pooled 0.558 was one market call counted seventeen times, not seventeen signals.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
