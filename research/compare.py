"""Measure one candidate flag against the shipped baseline, under the ship rule.

    python research/compare.py --label breadth --enable-breadth

The baseline is always the shipped state (abs_momentum on), so every candidate is judged against
what the bot actually runs today, not against the original strategy. The panel is built once and
reused, so the baseline and the candidate see identical data, dates, costs and risk settings.

Standard panel and the held-out period are measured here because they are cheap. The
survivorship-free panel is a separate, slower run (`app.main pit`), and is only worth doing for a
candidate that already passed here.

Ship rule (all must hold, on every panel):
    CAGR >= baseline, Sharpe >= baseline, max drawdown not worse, p_coin not worse than ~0.05
Held-out may have at most one metric flat within: 0 Sharpe, 0.5 CAGR points, 1 DD point.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import backtest, config, experiment, features, universe  # noqa: E402
from app.flags import Flags  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
PINNED_UNIVERSE = REPORTS / "_parity_universe.json"
HELD_OUT = "2022-01-01"
EPS_CAGR, EPS_DD, EPS_SHARPE = 0.005, 0.01, 0.0     # 0.5 CAGR points, 1 DD point, 0 Sharpe


def panel_and_settings():
    s = config.load()
    u = json.loads(PINNED_UNIVERSE.read_text()) if PINNED_UNIVERSE.exists() else universe.load(s)
    return features.build_panel(s.klines_dir, u["symbols"], s.donchian_bars), s, u


def measure(panel, s, flags, start=None):
    res = backtest.run(panel, s, flags=flags, start=start)
    m = backtest.metrics(res.equity, res.trades, res.exposure)
    rb = backtest.random_baselines(panel, res.trades, s) or {}
    return res, {**m, "p_coin": rb.get("p_value_vs_random_coin", float("nan")),
                 "p_time": rb.get("p_value_vs_random_time", float("nan"))}


def verdict(base, cand, held_out=False):
    """Return (passes, checks). Drawdown is negative, so 'not worse' means >= baseline.

    Standard and pit: the three metrics must each hold outright, and p_coin must not be worse than
    the baseline's, floored at 0.05.
    Held-out: the same three, except that ONE of them may be flat inside the pre-declared epsilon
    (0.5 CAGR points, 1 drawdown point, 0 Sharpe). p_coin is not a held-out criterion.
    """
    raw = {"CAGR": (cand["cagr"], base["cagr"], EPS_CAGR),
           "Sharpe": (cand["sharpe"], base["sharpe"], EPS_SHARPE),
           "maxDD": (cand["max_drawdown"], base["max_drawdown"], EPS_DD)}
    checks, used_eps = {}, 0
    for name, (c, b, eps) in raw.items():
        if c >= b:
            checks[name] = True
        elif held_out and c >= b - eps:
            checks[name] = True
            used_eps += 1
        else:
            checks[name] = False
    if held_out:
        if used_eps > 1:
            checks["one-eps-only"] = False
    else:
        checks["p_coin"] = cand["p_coin"] <= max(base["p_coin"], 0.05) + 1e-9
    return all(checks.values()), checks


def line(tag, m):
    return (f"  {tag:<22} CAGR {m['cagr']:+.4f}  Sharpe {m['sharpe']:.4f}  "
            f"maxDD {m['max_drawdown']:.4f}  trades {int(m.get('trades', 0)):>4}  "
            f"win {m.get('win_rate', float('nan')):.3f}  p_coin {m['p_coin']:.3f}  "
            f"p_time {m['p_time']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--baseline-flags", default="abs_momentum,volume_min=0.8,taker_min=0.5",
                    help="shipped flags, comma separated; floats as name=value "
                         "(e.g. abs_momentum,volume_min=0.8)")
    Flags.add_cli(ap)
    args = ap.parse_args()

    shipped = {}
    for item in filter(None, args.baseline_flags.split(",")):
        name, _, value = item.partition("=")
        shipped[name.strip()] = float(value) if value else True
    base_flags = Flags(**shipped)
    cand_flags = Flags(**{**shipped, **{k: v for k, v in vars(args).items()
                                        if k in {f.name for f in Flags.__dataclass_fields__.values()}
                                        and v is not None}})
    if base_flags == cand_flags:
        sys.exit("candidate flags are identical to the baseline; nothing to measure")

    panel, s, u = panel_and_settings()
    print(f"universe: {len(u['symbols'])} coins   baseline: {base_flags.enabled}   "
          f"candidate: {cand_flags.enabled}\n")

    out = {}
    for name, start in (("standard", None), ("held-out", HELD_OUT)):
        res_b, mb = measure(panel, s, base_flags, start)
        res_c, mc = measure(panel, s, cand_flags, start)
        ok, checks = verdict(mb, mc, held_out=(name == "held-out"))
        print(f"{name} panel" + (f" (from {start})" if start else "") + ":")
        print(line("baseline", mb))
        print(line(args.label, mc))
        print(f"  -> {'PASS' if ok else 'FAIL'}  " +
              "  ".join(f"{k}:{'ok' if v else 'NO'}" for k, v in checks.items()) + "\n")
        out[name] = {"base": mb, "cand": mc, "pass": ok, "checks": checks}
        if name == "standard":
            rec = experiment.summarise(res_c, panel, s, cand_flags,
                                       f"standard_{args.label}", "standard")
            experiment.save(rec, REPORTS)

    both = all(v["pass"] for v in out.values())
    print(f"VERDICT: {'run pit30 next' if both else 'REJECT (no pit run needed)'}")
    (REPORTS / "experiments" / f"compare_{args.label}.json").write_text(
        json.dumps(out, indent=2, default=str))
    return 0 if both else 2


if __name__ == "__main__":
    sys.exit(main())
