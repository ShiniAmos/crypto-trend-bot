@echo off
rem Starts the bot and the website under a watchdog that restarts either one if it crashes.
rem Everything runs in one minimized window named "Crypto Bot"; close it to stop both.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
start "Crypto Bot (close this window to stop the bot and website)" /min python -m app.supervisor
timeout /t 8 >nul
start "" http://localhost:8601
