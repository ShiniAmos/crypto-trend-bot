"""Flag-off parity gate: with every flag off the equity curve must equal the frozen baseline.

    python research/parity_check.py       exit 0 = CLEAN, exit 1 = drift
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import backtest, config, features, universe          # noqa: E402
from app.flags import Flags                                   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "reports" / "_parity_baseline_equity.csv"
# The universe is pinned beside the baseline: data/universe.json is rebuilt daily, and a different
# coin list produces a different curve for reasons that have nothing to do with the code.
PINNED_UNIVERSE = ROOT / "reports" / "_parity_universe.json"


def main():
    s = config.load()
    u = json.loads(PINNED_UNIVERSE.read_text()) if PINNED_UNIVERSE.exists() else universe.load(s)
    panel = features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars)
    res = backtest.run(panel, s, flags=Flags())
    want = pd.read_csv(BASELINE, index_col=0)["equity"]
    got = res.equity
    got.index = got.index.astype(str)
    common = want.index.intersection(got.index)
    diff = (want[common] - got[common]).abs()
    extra = len(want.index.difference(got.index)), len(got.index.difference(want.index))
    worst = float(diff.max()) if len(diff) else float("nan")
    ok = worst < 1e-9 and extra[0] == 0
    print(f"parity: {len(common)} shared bars, max abs diff {worst:.12g}, "
          f"missing {extra[0]}, new {extra[1]}")
    print("CLEAN" if ok else "DRIFT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
