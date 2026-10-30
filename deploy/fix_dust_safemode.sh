#!/usr/bin/env bash
# Unblock the demo book: dust is not a lost trade.
#
#   sudo bash /tmp/fix_dust_safemode.sh
#
# The demo book is in safe mode over 0.0877 ADA worth about two cents, left behind when Binance
# took its fee in the asset bought. That balance is below the 5 USDT minimum order so it can never
# be sold, which means the safe mode can never clear by itself and every entry stays blocked -
# which is why switching config 4 on produced no trades.
#
# Installs the reconcile fix, clears the stale safe-mode flag, and restarts the fleet.
set -euo pipefail

[ -f /tmp/reconcile.py ] || { echo "missing /tmp/reconcile.py - scp it first"; exit 1; }

cp /opt/bot/app/reconcile.py "/opt/bot/app/reconcile.py.bak-$(date +%Y%m%d-%H%M%S)"
install -o bot -g bot -m 644 /tmp/reconcile.py /opt/bot/app/reconcile.py
rm -f /tmp/reconcile.py
sudo -u bot /opt/bot/venv/bin/python -c "import ast,io;ast.parse(io.open('/opt/bot/app/reconcile.py',encoding='utf-8').read());print('reconcile.py parses')"

# Clear the stale flag. The next boot reconciliation re-runs immediately and will re-raise it if
# anything is genuinely wrong, so this cannot hide a real mismatch.
sudo -u bot /opt/bot/venv/bin/python - <<'PY'
import sqlite3
for book in ("demo", "paper", "paper100"):
    p = f"/opt/bot/data/state_{book}.sqlite"
    try:
        c = sqlite3.connect(p)
        n = c.execute("DELETE FROM kv WHERE key='safe_mode'").rowcount
        c.commit(); c.close()
        print(f"  {book}: cleared {n} safe-mode flag(s)")
    except Exception as e:
        print(f"  {book}: {type(e).__name__}: {e}")
PY

systemctl restart crypto-bot-fleet
sleep 12
echo "--- service ---"; systemctl is-active crypto-bot-fleet
echo "--- reconciliation and core, after restart ---"
tail -40 /var/log/crypto-bot-fleet.log | grep -iE 'Reconcil|SAFE MODE|note:|core |flags' | tail -12
