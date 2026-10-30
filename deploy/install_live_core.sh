#!/usr/bin/env bash
# Make the trend core act live instead of on the hourly bar.
#
#   sudo bash /tmp/install_live_core.sh
#
# Before: live_tick ran every 10 seconds but only for the breakout engine; the core acted only in
# on_bar_close, so it could sit up to an hour past a real crossing of its 200-day average.
# After:  core_tick() checks the live price against the 200-day level every poll - 10 seconds.
#
# Also fixes two bugs that shipped with the core:
#   - live_tick would have sold a core position on a Donchian breakdown (levels["exit"] carries a
#     low for every symbol), which is the reactive exit the core exists to avoid
#   - a core position was identified by reason.startswith("core") while the live path writes
#     "live core: ...", so a live core buy could never sell itself
set -euo pipefail

for f in main.py next_actions.py; do
  [ -f "/tmp/$f" ] || { echo "missing /tmp/$f - scp it first"; exit 1; }
done

cp /opt/bot/app/main.py "/opt/bot/app/main.py.bak-$(date +%Y%m%d-%H%M%S)"
cp /opt/bot/ui/next_actions.py "/opt/bot/ui/next_actions.py.bak-$(date +%Y%m%d-%H%M%S)"
install -o bot -g bot -m 644 /tmp/main.py /opt/bot/app/main.py
install -o bot -g bot -m 644 /tmp/next_actions.py /opt/bot/ui/next_actions.py
rm -f /tmp/main.py /tmp/next_actions.py

sudo -u bot /opt/bot/venv/bin/python - <<'PY'
import ast, io
for f in ("/opt/bot/app/main.py", "/opt/bot/ui/next_actions.py"):
    ast.parse(io.open(f, encoding="utf-8").read())
print("both parse on the server")
PY

systemctl restart crypto-bot-fleet
systemctl restart crypto-bot-web
sleep 12
echo "--- services ---"; systemctl is-active crypto-bot-fleet crypto-bot-web
echo "--- startup, per book ---"
tail -40 /var/log/crypto-bot-fleet.log | grep -E 'fleet started|  (paper|paper100|demo):|Reconcil|note:' | tail -8
echo
echo "--- core activity (may be empty until the price is checked) ---"
tail -60 /var/log/crypto-bot-fleet.log | grep -iE 'core ' | tail -6 || echo "  none yet"
