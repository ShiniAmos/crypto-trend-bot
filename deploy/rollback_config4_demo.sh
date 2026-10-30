#!/usr/bin/env bash
# Undo config 4 on the demo book: remove the DEMO_ overrides and restart.
#   sudo bash /opt/bot/rollback_config4_demo.sh
set -euo pipefail
ENV=/opt/bot/.env
cp "$ENV" "$ENV.bak-rollback-$(date +%Y%m%d-%H%M%S)"
sed -i '/^DEMO_/d;/^# --- config 4, demo book only/d' "$ENV"
chown bot:bot "$ENV"; chmod 600 "$ENV"
systemctl restart crypto-bot-fleet
sleep 8
systemctl is-active crypto-bot-fleet
grep -E '^(MODE|MAX_NAMES|ENABLE_|DEMO_)' "$ENV"
echo "config 4 removed; all books back on config 1"
