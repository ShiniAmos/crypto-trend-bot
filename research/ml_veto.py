"""P3-M: the walk-forward model as a veto, never as a replacement for the ranker.

The model already exists and was rejected as a ranker. The narrower question here: among the
candidates the shipped rules already like, does a positive predicted excess return separate the
good ones from the bad?

Only the out-of-sample span counts — rows before the first trained fold have no ml_score and are
excluded from both sides of the comparison, so the baseline and the candidate see identical bars.

    python research/ml_veto.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import backtest, config, features, ml  # noqa: E402
from app.flags import Flags  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SHIPPED = dict(abs_momentum=True, volume_min=0.8)
HORIZON = 6


def main():
    s = config.load()
    u = json.loads((ROOT / "reports" / "_parity_universe.json").read_text())
    panel = features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars)

    print("training the walk-forward model (730d train / 91d test, purged)...")
    scored = ml.walk_forward(panel, horizon=HORIZON, log=lambda *_: None)
    merged = panel.merge(scored[["bar_time", "symbol", "ml_score"]],
                         on=["bar_time", "symbol"], how="left")
    oos = merged["ml_score"].notna()
    start = merged.loc[oos, "bar_time"].min()
    print(f"out-of-sample from {start.date()}; {int(oos.sum())} scored rows")

    rows = []
    for tag, flags in (("baseline", Flags(**SHIPPED)), ("+ ml veto", Flags(**SHIPPED, ml_veto=True))):
        res = backtest.run(merged, s, flags=flags, start=start)
        m = backtest.metrics(res.equity, res.trades, res.exposure)
        rb = backtest.random_baselines(merged, res.trades, s) or {}
        rows.append({"run": tag, "cagr": m["cagr"], "sharpe": m["sharpe"],
                     "maxdd": m["max_drawdown"], "trades": m.get("trades", 0),
                     "win": m.get("win_rate"), "p_coin": rb.get("p_value_vs_random_coin"),
                     "p_time": rb.get("p_value_vs_random_time")})
    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    b, c = rows[0], rows[1]
    ok = (c["cagr"] >= b["cagr"] and c["sharpe"] >= b["sharpe"] and c["maxdd"] >= b["maxdd"]
          and c["p_coin"] <= max(b["p_coin"], 0.05) + 1e-9)
    print("\nVERDICT:", "CANDIDATE - run pit30" if ok else "REJECT - keep ML disconnected")
    (ROOT / "reports" / "ml_veto.json").write_text(
        json.dumps({"start": str(start), "rows": rows, "passes": bool(ok)}, indent=2, default=str))


if __name__ == "__main__":
    main()
