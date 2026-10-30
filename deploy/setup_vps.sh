#!/usr/bin/env bash
# Phase 9: install the bot on a fresh Ubuntu server (tested for Oracle Cloud Always Free).
# Run from the unpacked crypto-bot folder:   sudo bash deploy/setup_vps.sh
set -euo pipefail
cd "$(dirname "$0")/.."

apt-get update
apt-get install -y python3-venv python3-pip iptables-persistent

# Small servers: add 2 GB of swap so installs and the history test never run out of memory.
if ! swapon --show | grep -q /swapfile; then
  fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

id bot >/dev/null 2>&1 || useradd -m -s /bin/bash bot
mkdir -p /opt/bot
cp -r app ui dashboard.py requirements.txt .env.example /opt/bot/
[ -d data ] && cp -rn data /opt/bot/          # keeps your paper account and history if you uploaded them
[ -f /opt/bot/.env ] || cp .env.example /opt/bot/.env
chmod 600 /opt/bot/.env
chown -R bot:bot /opt/bot

sudo -u bot python3 -m venv /opt/bot/venv
sudo -u bot /opt/bot/venv/bin/pip install --quiet -r /opt/bot/requirements.txt

touch /var/log/crypto-bot.log /var/log/crypto-bot-watchdog.log
chown bot:bot /var/log/crypto-bot*.log
cp deploy/crypto-bot.service deploy/crypto-bot-web.service /etc/systemd/system/
cp deploy/logrotate /etc/logrotate.d/crypto-bot
sudo -u bot crontab deploy/crontab

# Oracle's Ubuntu images block every port except SSH inside the server too; open the website port.
iptables -C INPUT -p tcp --dport 8601 -j ACCEPT 2>/dev/null || iptables -I INPUT 6 -p tcp --dport 8601 -j ACCEPT
netfilter-persistent save

cd /opt/bot
[ -f data/universe.json ] || sudo -u bot venv/bin/python -m app.main universe
sudo -u bot venv/bin/python -m app.main backfill --tf 1d 4h
systemctl daemon-reload
systemctl enable --now crypto-bot crypto-bot-web

IP=$(curl -s https://api.ipify.org || echo "<server-ip>")
echo
echo "Done. Bot and website are running and restart by themselves after a reboot."
echo "Website: http://$IP:8601"
grep -q '^DASHBOARD_PASSWORD=.\+' /opt/bot/.env || echo "SET A PASSWORD: sudo nano /opt/bot/.env  (DASHBOARD_PASSWORD=...), then: sudo systemctl restart crypto-bot-web"
