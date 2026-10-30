#!/usr/bin/env bash
# Install the "Next move" website page and the snapshot it reads.
#
#   sudo bash /opt/bot/install_next_move.sh
#
# Three files:
#   ui/next_actions.py   the new page
#   dashboard.py         adds it to the navigation
#   app/main.py          writes the snapshot on EVERY bar (it used to be live-cycle only, so the
#                        page would have had nothing to show) and puts the core's own trigger,
#                        the core weight and the slot count into it
#
# Restarts the website. The fleet is restarted too because app/main.py changed; if the fleet is
# mid-trade that is safe - orders are queued in sqlite and re-read on start.
set -euo pipefail

for f in next_actions.py dashboard.py main.py; do
  [ -f "/tmp/$f" ] || { echo "missing /tmp/$f - scp it first"; exit 1; }
done

cp /opt/bot/ui/next_actions.py "/opt/bot/ui/next_actions.py.bak-$(date +%Y%m%d-%H%M%S)" 2>/dev/null || true
cp /opt/bot/dashboard.py "/opt/bot/dashboard.py.bak-$(date +%Y%m%d-%H%M%S)"
cp /opt/bot/app/main.py "/opt/bot/app/main.py.bak-$(date +%Y%m%d-%H%M%S)"

install -o bot -g bot -m 644 /tmp/next_actions.py /opt/bot/ui/next_actions.py
install -o bot -g bot -m 644 /tmp/dashboard.py   /opt/bot/dashboard.py
install -o bot -g bot -m 644 /tmp/main.py        /opt/bot/app/main.py
rm -f /tmp/next_actions.py /tmp/dashboard.py /tmp/main.py

sudo -u bot /opt/bot/venv/bin/python - <<'PY'
import ast, io
for f in ("/opt/bot/ui/next_actions.py", "/opt/bot/dashboard.py", "/opt/bot/app/main.py"):
    ast.parse(io.open(f, encoding="utf-8").read())
print("all three parse on the server")
PY

systemctl restart crypto-bot-web
systemctl restart crypto-bot-fleet
sleep 10
echo "--- services ---"
systemctl is-active crypto-bot-web crypto-bot-fleet
echo "--- fleet startup, per book ---"
journalctl -u crypto-bot-fleet -n 30 --no-pager | tail -12
echo
echo "Open the website and pick the 'Next move' page."
