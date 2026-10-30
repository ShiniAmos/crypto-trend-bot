#!/usr/bin/env bash
# Bring the server up to date with everything built since 12 October.
#
#   sudo bash /tmp/install_all.sh
#
# WHAT IS CURRENTLY RUNNING THERE IS OLDER THAN THE DEMO BOOK'S OWN POSITIONS. The demo book holds
# two core positions bought on the 200-day rule, under code where:
#
#   * live_tick can close a core position on a DONCHIAN LOW - the reactive exit the core exists to
#     avoid. This is the urgent one.
#   * the core only acts in on_bar_close, so it waits up to an hour past a real crossing
#   * which engine owns a position is inferred from its reason text, which the live path writes
#     differently ("live core: ...") - so a live core buy could never sell itself
#
# Contents:
#   app/main.py        live core (core_tick), engine tags on buys, exits read the tag
#   app/portfolio.py   positions.engine and queued_orders.engine + migration for the live books
#   app/backtest.py    engine tags, capital slices, core risk caps, per-engine P&L curves
#   app/flags.py       capital_slices, slice_max_names, core_target_vol, core_w_max,
#                      core_max_stretch - ALL DEFAULT OFF and all REJECTED by the ship rule
#   app/signals.py     core_weight(), core_entry_allowed(), core symbols reserved from entries
#   app/reconcile.py   dust below 1 USDT is a note, not a safe-mode failure
#   ui/next_actions.py the "Next move" page
#   dashboard.py       adds that page to the navigation
#
# No flag changes. The demo book keeps trend_core=0.25 / taker_min off / 5 names; the paper books
# keep the shipped set. Nothing new is switched on by this script.
set -euo pipefail

FILES_APP="main.py portfolio.py backtest.py flags.py signals.py reconcile.py"
FILES_UI="next_actions.py"
FILES_ROOT="dashboard.py"
STAMP=$(date +%Y%m%d-%H%M%S)

for f in $FILES_APP $FILES_UI $FILES_ROOT; do
  [ -f "/tmp/$f" ] || { echo "missing /tmp/$f - scp it first"; exit 1; }
done

echo "=== backing up to *.bak-$STAMP ==="
for f in $FILES_APP; do cp "/opt/bot/app/$f" "/opt/bot/app/$f.bak-$STAMP" 2>/dev/null || true; done
for f in $FILES_UI;  do cp "/opt/bot/ui/$f"  "/opt/bot/ui/$f.bak-$STAMP"  2>/dev/null || true; done
for f in $FILES_ROOT; do cp "/opt/bot/$f"    "/opt/bot/$f.bak-$STAMP"     2>/dev/null || true; done

echo "=== installing ==="
for f in $FILES_APP;  do install -o bot -g bot -m 644 "/tmp/$f" "/opt/bot/app/$f"; done
for f in $FILES_UI;   do install -o bot -g bot -m 644 "/tmp/$f" "/opt/bot/ui/$f";  done
for f in $FILES_ROOT; do install -o bot -g bot -m 644 "/tmp/$f" "/opt/bot/$f";     done

echo "=== parsing everything on the server before anything restarts ==="
sudo -u bot /opt/bot/venv/bin/python - <<'PY'
import ast, io
paths = ["/opt/bot/app/" + f for f in
         ("main.py", "portfolio.py", "backtest.py", "flags.py", "signals.py", "reconcile.py")]
paths += ["/opt/bot/ui/next_actions.py", "/opt/bot/dashboard.py"]
for p in paths:
    ast.parse(io.open(p, encoding="utf-8").read())
print(f"  {len(paths)} files parse")
PY

echo "=== the schema migration, run once before the bot touches the books ==="
sudo -u bot /opt/bot/venv/bin/python - <<'PY'
import sys
sys.path.insert(0, "/opt/bot")
from app.portfolio import Store
for book in ("paper", "paper100", "demo"):
    p = f"/opt/bot/data/state_{book}.sqlite"
    try:
        s = Store(p)                       # __init__ runs the migration
        pos = s.positions()
        tags = {k: v.get("engine") for k, v in pos.items()}
        print(f"  {book}: {len(pos)} position(s), engine tags {tags or '{}'}")
    except Exception as e:
        print(f"  {book}: {type(e).__name__}: {e}")
PY

echo
echo "NOTE: positions opened before this upgrade are tagged 'breakout' by the migration's default."
echo "The demo book's two CORE positions were opened by the old code, so they carry that tag and"
echo "would be given the breakout engine's Donchian exit. Retagging them below."
sudo -u bot /opt/bot/venv/bin/python - <<'PY'
import sqlite3
p = "/opt/bot/data/state_demo.sqlite"
c = sqlite3.connect(p)
c.row_factory = sqlite3.Row
rows = list(c.execute("SELECT symbol, reason, engine FROM positions"))
fixed = 0
for r in rows:
    if str(r["reason"] or "").startswith(("core", "live core")) and r["engine"] != "core":
        c.execute("UPDATE positions SET engine='core' WHERE symbol=?", (r["symbol"],))
        print(f"  retagged {r['symbol']} -> core ({r['reason']})")
        fixed += 1
c.commit(); c.close()
print(f"  {fixed} position(s) retagged" if fixed else "  nothing needed retagging")
PY

echo
echo "=== restarting ==="
systemctl restart crypto-bot-fleet
systemctl restart crypto-bot-web
sleep 12
systemctl is-active crypto-bot-fleet crypto-bot-web

echo
echo "=== startup, per book ==="
tail -60 /var/log/crypto-bot-fleet.log | grep -E 'fleet started|  (paper|paper100|demo):|Reconcil|note:|SAFE MODE' | tail -10
echo
echo "=== core state ==="
tail -80 /var/log/crypto-bot-fleet.log | grep -iE 'core ' | tail -6 || echo "  no core action this cycle"
echo
echo "Done. The core now acts within LIVE_POLL_SECONDS of a crossing, and a Donchian low can no"
echo "longer close it."
