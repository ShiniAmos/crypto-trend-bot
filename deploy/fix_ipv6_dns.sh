#!/usr/bin/env bash
# Stop outbound calls failing with "Network is unreachable" on a host with no IPv6 route.
#
#   sudo bash /tmp/fix_ipv6_dns.sh
#
# THE PROBLEM. api.telegram.org resolves to both an IPv4 and an IPv6 address:
#     IPv4  149.154.166.110        reachable, 302 in 0.45s
#     IPv6  2001:67c:4e8:f004::9   NOT reachable - this VM has no IPv6 default route
# getaddrinfo returns both. When the IPv6 address is tried first the connection fails instantly
# with errno 101, which is the "telegram connection failed" line in the bot log - 11 times between
# 11 and 16 Oct. Trading is unaffected (Binance resolves to IPv4 only) but ALERTING is not, and
# alerting is the safety net while the demo trial runs.
#
# THE FIX. RFC 3484 / getaddrinfo precedence: raising the precedence of IPv4-mapped addresses makes
# glibc sort IPv4 ahead of IPv6 for every lookup on this host. This is the documented remedy for a
# machine with no working IPv6, it is one line, and it is reversible by deleting that line.
#
# It does not disable IPv6, and it does not change anything for Binance, which has no AAAA record.
set -euo pipefail

LINE='precedence ::ffff:0:0/96  100'
STAMP=$(date +%Y%m%d-%H%M%S)

echo "=== before ==="
echo -n "  telegram resolves to: "; getent hosts api.telegram.org | head -1
grep -q "^${LINE}$" /etc/gai.conf 2>/dev/null && { echo "already applied; nothing to do"; exit 0; }

cp /etc/gai.conf "/etc/gai.conf.bak-$STAMP" 2>/dev/null || true
printf '\n# Added %s: this host has no IPv6 route, so prefer IPv4 for every lookup.\n# Remove this line to revert.\n%s\n' "$STAMP" "$LINE" >> /etc/gai.conf

echo "=== after ==="
echo -n "  telegram resolves to: "; getent hosts api.telegram.org | head -1
echo "  python order:"
sudo -u bot /opt/bot/venv/bin/python -c "
import socket
seen=[]
for fam,_,_,_,addr in socket.getaddrinfo('api.telegram.org',443):
    k=('IPv6' if fam==socket.AF_INET6 else 'IPv4', addr[0])
    if k not in seen: seen.append(k)
for k in seen: print('   ', k[0], k[1])"

echo "=== live check, 5 calls ==="
ok=0
for i in 1 2 3 4 5; do
  if curl -s -o /dev/null --max-time 10 https://api.telegram.org/ ; then ok=$((ok+1)); fi
done
echo "  $ok/5 reached api.telegram.org"

echo "=== restarting the fleet so it picks up the new resolution order ==="
systemctl restart crypto-bot-fleet
sleep 12
systemctl is-active crypto-bot-fleet
tail -30 /var/log/crypto-bot-fleet.log | grep -E 'fleet started|  (paper|paper100|demo):|telegram' | tail -6
echo
echo "Done. Watch for 'telegram connection failed' - it should stop appearing."
