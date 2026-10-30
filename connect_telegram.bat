@echo off
rem Connects Telegram alerts, then restarts the bot so it starts sending them.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
python -m app.main telegram
if errorlevel 1 goto end
echo Restarting the bot so it uses Telegram...
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { $_.CommandLine -like '*app.main run*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
start "Crypto Bot - trading loop (close to stop)" /min cmd /c "python -u -m app.main run >> data\bot.log 2>&1"
echo Done. You will get a Telegram message on every buy and sell, plus a daily summary.
:end
pause
