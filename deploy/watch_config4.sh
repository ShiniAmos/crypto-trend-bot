#!/usr/bin/env bash
# The kill criteria for the config-4 demo trial, in one command.
#
#   sudo bash /opt/bot/watch_config4.sh
#
# Declared before the trial started, so none of them can be renegotiated later:
#
#   1. demo drawdown worse than -25%          -> roll back
#   2. a core position on demo with no        -> roll back. A divergence here means the
#      equivalent on paper, when paper has       per-account flag wiring is leaking rather than
#      no core flag                              that the strategy is wrong - but either way, stop.
#   3. any order rejected by Binance          -> investigate before continuing
#   4. the fleet restarting on its own        -> investigate (memory, as on 4 Oct)
#
# Trial length: 3-4 weeks from the enable date. Nothing is promoted to the paper books before then.
set -uo pipefail

DB=/opt/bot/data

echo "=== service ==="
systemctl is-active crypto-bot-fleet
systemctl show crypto-bot-fleet -p NRestarts --value | sed 's/^/restarts since boot: /'
echo

echo "=== each book: the flags actually in force ==="
journalctl -u crypto-bot-fleet --no-pager | grep -E '  (paper|paper100|demo): ' | tail -3
echo

echo "=== equity, drawdown and core positions per book ==="
for acct in paper paper100 demo; do
  f="$DB/state_${acct}.sqlite"
  [ -f "$f" ] || continue
  sudo -u bot /opt/bot/venv/bin/python - "$acct" "$f" <<'PY'
import sqlite3, sys
acct, path = sys.argv[1], sys.argv[2]
try:
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        eq = [r[0] for r in c.execute("SELECT equity FROM equity ORDER BY time")]
    except sqlite3.OperationalError:
        eq = []
    if eq:
        peak, dd = eq[0], 0.0
        for v in eq:
            peak = max(peak, v)
            dd = min(dd, v / peak - 1)
        flag = "*** PAST -25%, ROLL BACK ***" if dd < -0.25 else "ok"
        print(f"  {acct:9s} equity {eq[-1]:>12,.2f}   maxDD {dd:>7.2%}   {flag}")
    else:
        print(f"  {acct:9s} no equity history yet")
    pos = list(c.execute("SELECT symbol, qty, entry_price, reason FROM positions"))
    core = [p for p in pos if str(p[3] or "").startswith("core")]
    line = f"  {acct:9s} positions {len(pos)}, of which core {len(core)}"
    if core:
        line += "  -> " + ", ".join(p[0] for p in core)
    print(line)
except Exception as e:
    print(f"  {acct:9s} could not read: {type(e).__name__}: {e}")
PY
done
echo

echo "=== order rejections and errors, last 24h ==="
journalctl -u crypto-bot-fleet --since '24 hours ago' --no-pager \
  | grep -iE 'reject|APIError|code=-[0-9]+|skipped:|safe mode|OFFLINE' | tail -12 \
  || echo "  none"
echo

echo "=== what the core has actually done ==="
journalctl -u crypto-bot-fleet --no-pager | grep -E 'core (BUY|exit|buy)' | tail -10 \
  || echo "  nothing yet - the core only acts when BTC closes above its 200-day average"
