#!/usr/bin/env bash
# Every book runs the SHIPPED configuration (6 Nov 2026).
#
#   abs_momentum + volume_min=0.8 + funding_max_pct=0.8 + taker_min OFF
#   + trend_core=0.25 on BTC,ETH + core_dual_ma, 5 slots
#
# This is the variant that cleared the full ship rule on 18 Oct (P15), the only change to do so
# since the core itself:
#
#   panel             CAGR      Sharpe   maxDD      placebo p
#   standard          +27.03%   1.2480   -26.09%
#   pit               +21.55%   1.0971   -22.02%    0.0000
#   pit held-out      +12.45%   0.7758   -22.02%
#   pit at 2x costs   +18.92%   0.9728
#
# WHY THE PER-BOOK SPLIT GOES AWAY. The DEMO_ block was config 4, opened 10 Oct as a live A/B with
# paper and paper100 held back on config 1 as the control. The 18 Oct ship decision settled that
# question on nine years of survivorship-free data, and P15's own baseline row IS config 4 - so the
# shipped configuration is config 4 plus the dual MA, which is what the demo book has been running
# since 30 Oct. Keeping paper on config 1 means spending forward-test time on an arm this project
# measures at +0.54% a year (held-out -2.06%). The trial is finished; this closes it.
#
# The three books now differ only in what fleet.py says they are for: 10,000 paper, 100 paper where
# minimum orders and rounding bite, and real signed orders on Binance Demo Trading.
set -euo pipefail

ENV=/opt/bot/.env
[ -f "$ENV" ] || { echo "no $ENV"; exit 1; }

cp "$ENV" "$ENV.bak-$(date +%Y%m%d-%H%M%S)"

# drop the per-book trial block and every strategy line this script owns, then re-add them
sed -i '/^DEMO_/d' "$ENV"
sed -i '/^# --- config 4, demo book only/d' "$ENV"
sed -i '/^# --- shipped configuration/d' "$ENV"
sed -i '/^MAX_NAMES=/d; /^ENABLE_ABS_MOMENTUM=/d; /^ENABLE_VOLUME_MIN=/d' "$ENV"
sed -i '/^ENABLE_TAKER_MIN=/d; /^ENABLE_FUNDING_MAX_PCT=/d' "$ENV"
sed -i '/^ENABLE_TREND_CORE=/d; /^ENABLE_TREND_CORE_SYMBOLS=/d; /^ENABLE_CORE_DUAL_MA=/d' "$ENV"

cat >> "$ENV" <<'EOF'
# --- shipped configuration (P15, 18 Oct 2026). Every book, no per-account overrides. ---
MAX_NAMES=5
ENABLE_ABS_MOMENTUM=true
ENABLE_VOLUME_MIN=0.8
ENABLE_FUNDING_MAX_PCT=0.8
ENABLE_TAKER_MIN=0
ENABLE_TREND_CORE=0.25
ENABLE_TREND_CORE_SYMBOLS=BTCUSDT,ETHUSDT
ENABLE_CORE_DUAL_MA=true
EOF

chown bot:bot "$ENV"
chmod 600 "$ENV"
echo "--- strategy lines now ---"
grep -E '^(MODE|MAX_NAMES|ENABLE_|DEMO_|DECISION_TF|LEVEL_TF)' "$ENV"
