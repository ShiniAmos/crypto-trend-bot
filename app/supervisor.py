"""Keeps the bot and the website running: if either one crashes or exits, it is restarted in 3 s.

    python -m app.supervisor          (start.bat runs this in a minimized window)

Closing the supervisor's window stops everything. Output goes to data/bot.log and data/web.log.
"""

import os
import subprocess
import sys
import time
from datetime import datetime

from .config import ROOT

DATA = ROOT / "data"
LOCK = DATA / "supervisor.pid"
# The $100 account trades the same signals with a small wallet, so the effect of Binance's
# minimum order size and rounding shows up in real time. It never sends Telegram messages.
SMALL = {"ACCOUNT": "paper100", "START_EQUITY": "100", "TELEGRAM_BOT_TOKEN": "",
         "SHARED_DATA": "true"}
JOBS = {
    "bot": ([sys.executable, "-u", "-m", "app.main", "run"], DATA / "bot.log"),
    "bot100": ([sys.executable, "-u", "-m", "app.main", "run"], DATA / "bot100.log"),
    "website": ([sys.executable, "-m", "streamlit", "run", "dashboard.py", "--server.port", "8601",
                 "--server.headless", "true", "--browser.gatherUsageStats", "false"], DATA / "web.log"),
}


def say(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(DATA / "supervisor.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def pid_alive(pid):
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def start(name):
    cmd, log = JOBS[name]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    if name == "bot100":
        env.update(SMALL)
    return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=open(log, "a", encoding="utf-8"),
                            stderr=subprocess.STDOUT)


def main():
    DATA.mkdir(exist_ok=True)
    if LOCK.exists() and pid_alive(int(LOCK.read_text() or 0)):
        sys.exit("Already running (another supervisor is active). Close its window first to restart.")
    LOCK.write_text(str(os.getpid()))
    procs = {name: start(name) for name in JOBS}
    say(f"started bot, $100 test account and website (website: http://localhost:8601). Close this window to stop both.")
    restarts = {name: [] for name in JOBS}
    try:
        while True:
            time.sleep(3)
            for name, p in procs.items():
                if p.poll() is None:
                    continue
                recent = [t for t in restarts[name] if time.time() - t < 300]
                restarts[name] = recent + [time.time()]
                wait = 3 if len(recent) < 5 else 60   # crashing over and over: slow down
                say(f"{name} stopped (exit code {p.returncode}); restarting in {wait} s")
                time.sleep(wait - 3)
                procs[name] = start(name)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs.values():
            p.terminate()
        LOCK.unlink(missing_ok=True)
        say("stopped")


if __name__ == "__main__":
    main()
