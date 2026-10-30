#!/usr/bin/env bash
# Deploy core_dual_ma and switch it on for the DEMO book only.
#
#   sudo bash /tmp/enable_dual_ma.sh
#
# The rule: hold a core coin only while it is above BOTH its 50-day and its 200-day average.
# It cleared the full ship rule on 18 Oct (pit 16.96 -> 21.55 CAGR, Sharpe 0.9437 -> 1.0971,
# maxDD -25.54 -> -22.02) and was then never switched on. In the twelve days since, ETH has been
# below its 50-day average EVERY day while remaining above its 200-day - exactly the case this
# flag exists for - and that position is the demo book's largest loss.
#
# paper and paper100 are untouched and remain the control.
set -euo pipefail

for f in flags.py signals.py main.py; do
  [ -f "/tmp/$f" ] || { echo "missing /tmp/$f"; exit 1; }
done
S=$(date +%Y%m%d-%H%M%S)
for f in flags.py signals.py main.py; do
  cp "/opt/bot/app/$f" "/opt/bot/app/$f.bak-$S"
  install -o bot -g bot -m 644 "/tmp/$f" "/opt/bot/app/$f"
done
rm -f /tmp/flags.py /tmp/signals.py /tmp/main.py

sudo -u bot /opt/bot/venv/bin/python - <<'PY'
import ast, io
for f in ("flags.py", "signals.py", "main.py"):
    ast.parse(io.open(f"/opt/bot/app/{f}", encoding="utf-8").read())
print("  all three parse")
PY

ENV=/opt/bot/.env
cp "$ENV" "$ENV.bak-$S"
sed -i '/^DEMO_ENABLE_CORE_DUAL_MA=/d' "$ENV"
echo 'DEMO_ENABLE_CORE_DUAL_MA=true' >> "$ENV"
chown bot:bot "$ENV"; chmod 600 "$ENV"
echo "=== demo overrides now ==="; grep -E '^DEMO_' "$ENV"

systemctl restart crypto-bot-fleet
sleep 15
systemctl is-active crypto-bot-fleet
echo "=== per-book flags ==="
tail -40 /var/log/crypto-bot-fleet.log | grep -E '  (paper|paper100|demo):' | tail -3
echo "=== what the core did on the first pass ==="
tail -60 /var/log/crypto-bot-fleet.log | grep -iE 'core |SOLD|BOUGHT' | tail -8 || echo "  nothing yet - it acts on the next price check"
