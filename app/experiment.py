"""One record per backtest, machine- and human-readable, so a result can be reproduced later.

Writes reports/experiments/<name>.json and appends a row to reports/experiments/index.csv.
The record carries everything needed to re-run the same experiment: git commit, flags, settings,
dates, panel, costs and seed.
"""

import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest, regime

FIELDS = ["experiment", "panel", "flags", "commit", "start", "end", "cagr", "sharpe",
          "max_drawdown", "total_return", "trades", "win_rate", "profit_factor", "expectancy",
          "avg_exposure", "turnover", "longest_losing_streak", "largest_loss", "largest_win",
          "random_coin_p", "random_time_p", "strategy_avg_trade", "random_coin_avg_trade",
          "random_time_avg_trade"]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or "unversioned"
    except Exception:
        return "unversioned"


def _streak(trades: pd.DataFrame) -> int:
    worst = run = 0
    for pnl in trades["pnl"]:
        run = run + 1 if pnl <= 0 else 0
        worst = max(worst, run)
    return worst


def _turnover(trades: pd.DataFrame, equity: pd.Series) -> float:
    """Round trips per year: how hard the strategy churns."""
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    return len(trades) / years


def summarise(res, panel, settings, flags, name, panel_name, seed=7) -> dict:
    m = backtest.metrics(res.equity, res.trades, res.exposure)
    rb = backtest.random_baselines(panel, res.trades, settings, seed=seed) or {}
    t = res.trades
    rec = {
        "experiment": name,
        "panel": panel_name,
        "flags": flags.enabled or "baseline",
        "commit": git_commit(),
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "start": str(m["start"]), "end": str(m["end"]),
        "cagr": m["cagr"], "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"],
        "total_return": m["total_return"],
        "trades": int(m.get("trades", 0)), "win_rate": m.get("win_rate", np.nan),
        "profit_factor": m.get("profit_factor", np.nan),
        "expectancy": m.get("avg_trade", np.nan),
        "avg_exposure": m.get("avg_exposure", np.nan),
        "turnover": _turnover(t, res.equity) if len(t) else 0.0,
        "longest_losing_streak": _streak(t) if len(t) else 0,
        "largest_loss": float(t["ret"].min()) if len(t) else np.nan,
        "largest_win": float(t["ret"].max()) if len(t) else np.nan,
        "kills": res.kills, "daily_halts": res.halts,
        "strategy_avg_trade": rb.get("strategy_avg_trade", np.nan),
        "random_coin_avg_trade": rb.get("random_coin_avg_trade", np.nan),
        "random_time_avg_trade": rb.get("random_time_avg_trade", np.nan),
        "random_coin_p": rb.get("p_value_vs_random_coin", np.nan),
        "random_time_p": rb.get("p_value_vs_random_time", np.nan),
        "seed": seed,
        "costs": {"slippage_bps": settings.slippage_bps, "fee_rate": settings.fee_rate},
        "settings": {k: getattr(settings, k) for k in
                     ("donchian_bars", "candidates", "exit_rank", "risk_pct", "max_names",
                      "max_weight", "daily_loss_halt", "kill_drawdown", "start_equity")},
        "flags_full": asdict(flags),
        "flags_digest": flags.digest(),
    }
    try:
        rec["regimes"] = regime.per_regime(res.equity, regime.label_bars(panel), t,
                                           res.exposure).to_dict("records")
    except Exception as e:
        rec["regimes"] = f"unavailable: {e}"
    return rec


def save(rec: dict, reports_dir: Path) -> Path:
    d = Path(reports_dir) / "experiments"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{rec['experiment']}.json"
    path.write_text(json.dumps(rec, indent=2, default=str))
    index = d / "index.csv"
    row = pd.DataFrame([{k: rec.get(k) for k in FIELDS}])
    row.to_csv(index, mode="a", header=not index.exists(), index=False)
    return path


def compare(base: dict, cand: dict) -> dict:
    """The ship rule: a candidate must not damage CAGR, Sharpe or max drawdown.

    Drawdown is negative, so 'not worse' means cand >= base. Small tolerances stop noise from
    counting as a difference.
    """
    tol_dd, tol_sharpe, tol_cagr = 0.005, 0.02, 0.002
    dd_ok = cand["max_drawdown"] >= base["max_drawdown"] - tol_dd
    sharpe_ok = cand["sharpe"] >= base["sharpe"] - tol_sharpe
    cagr_ok = cand["cagr"] >= base["cagr"] - tol_cagr
    improves = (cand["cagr"] > base["cagr"] + tol_cagr or cand["sharpe"] > base["sharpe"] + tol_sharpe
                or cand["max_drawdown"] > base["max_drawdown"] + tol_dd)
    return {"cagr": (base["cagr"], cand["cagr"], cagr_ok),
            "sharpe": (base["sharpe"], cand["sharpe"], sharpe_ok),
            "max_drawdown": (base["max_drawdown"], cand["max_drawdown"], dd_ok),
            "passes": bool(dd_ok and sharpe_ok and cagr_ok and improves)}


def decision_table(rows) -> pd.DataFrame:
    """rows: [(feature, panel, base_rec, cand_rec)] -> the table the ship rule is read from."""
    out = []
    for feature, panel_name, base, cand in rows:
        c = compare(base, cand)
        out.append({"feature": feature, "panel": panel_name,
                    "cagr_base": base["cagr"], "cagr_new": cand["cagr"],
                    "sharpe_base": base["sharpe"], "sharpe_new": cand["sharpe"],
                    "maxdd_base": base["max_drawdown"], "maxdd_new": cand["max_drawdown"],
                    "trades_base": base["trades"], "trades_new": cand["trades"],
                    "random_coin_p": cand["random_coin_p"], "random_time_p": cand["random_time_p"],
                    "decision": "CANDIDATE FOR SHIP" if c["passes"] else "KEEP BASELINE"})
    return pd.DataFrame(out)
