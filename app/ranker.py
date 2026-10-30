"""Phase 4 output: the ranked table for the latest closed 4h bar, and today's candidates."""

from . import backfill, features, funding, signals


def scan(settings, symbols, refresh=True, log=print, flags=None):
    """Bring 4h/1d history up to the last closed bar, then score everything on it."""
    tf = settings.bar_tf
    if refresh and not settings.shared_data:
        backfill.run(settings, symbols, ("1d", tf), log=lambda *_: None)
    panel = features.build_panel(settings.klines_dir, symbols, settings.donchian_for_tf, tf=tf,
                                 exit_donchian=getattr(flags, "exit_donchian", 0))
    if getattr(flags, "funding_max_pct", 0):
        panel = funding.attach(panel, settings)      # crowding, from outside the price
    bar_time = panel["bar_time"].max()
    rows = panel[panel["bar_time"] == bar_time]
    cands = signals.entry_candidates(rows, settings, flags=flags)
    return panel, bar_time, rows, cands


def explain(rows, top=5):
    """Plain-words reason for each of the top names on a bar (Phase 4 exit check)."""
    out = []
    for r in rows[rows["eligible"]].sort_values("rank").head(top).itertuples():
        bits = [f"7d vs BTC {r.rs_7d * 100:+.1f}%", f"30d vs BTC {r.rs_30d * 100:+.1f}%",
                f"{r.dist_sma50 * 100:+.1f}% from its 50-day average"]
        if r.breakout:
            bits.append("closed above its 20-bar high")
        out.append(f"#{int(r.rank)} {r.symbol}: score {r.score:.0f} ({', '.join(bits)})")
    return out
