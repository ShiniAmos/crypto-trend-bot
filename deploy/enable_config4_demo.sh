#!/usr/bin/env bash
# Config 4 on the DEMO book only (10 Oct 2026).
#
#   trend core 0.25 BTC + 5 slots + taker_min off
#
# The shared ENABLE_* lines in .env are left alone, so paper and paper100 keep running config 1.
# That is deliberate: all three books see the same bars from the same scan, so paper (control)
# against demo (treatment) is a clean live comparison rather than a before-and-after.
#
# Backtested, pit panel, survivorship-free:
#   modern 2022-26   CAGR +10.76%  Sharpe 0.650  maxDD -22.78%  exposure 20.8%
#   holdout 2025-26  CAGR  +4.27%  Sharpe 0.335  maxDD -18.15%
# against config 1's +0.54% / -2.06% at 5.8% exposure.
#
#   sudo bash /opt/bot/enable_config4_demo.sh
set -euo pipefail

ENV=/opt/bot/.env
[ -f "$ENV" ] || { echo "no $ENV"; exit 1; }

cp "$ENV" "$ENV.bak-$(date +%Y%m%d-%H%M%S)"
sed -i '/^DEMO_/d' "$ENV"                 # idempotent: drop any previous demo block first
sed -i '/^# --- config 4, demo book only/d' "$ENV"
cat >> "$ENV" <<'EOF'
# --- config 4, demo book only. paper/paper100 remain the control. ---
DEMO_ENABLE_TREND_CORE=0.25
DEMO_ENABLE_TAKER_MIN=0
DEMO_MAX_NAMES=5
EOF
chown bot:bot "$ENV"
chmod 600 "$ENV"

echo "--- strategy lines in .env ---"
grep -E '^(MODE|MAX_NAMES|ENABLE_|DEMO_)' "$ENV"

systemctl restart crypto-bot-fleet
sleep 8
echo "--- service ---"
systemctl is-active crypto-bot-fleet
echo "--- startup log: each book's own flags ---"
journalctl -u crypto-bot-fleet -n 40 --no-pager | grep -E 'fleet started|paper|paper100|demo|core|flags' | tail -15
