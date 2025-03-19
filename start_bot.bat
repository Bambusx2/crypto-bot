@echo off
echo Starting Trading Bot (MULTI-COIN MODE)...
echo ========================================
echo MONITORING MULTIPLE COINS: BTC, ETH, XRP, ADA, LINK, HBAR, AVAX, DOT, NEAR, ICP, ONDO, RENDER, S, FET, SEI, STX, IOTA, KDA, FLUX
echo The bot will ONLY trade when it finds a strong signal
echo Do not close this window while the bot is running!
echo ========================================
echo.

:: Create a PID file with a numeric placeholder
echo 0 > bot_process.pid

:: Set debug level
set LOG_LEVEL=INFO

:: Start the bot with the default config (now multi-coin)
python -m cryptobot --config config/default_config.yml

:: Clean up PID file when bot exits
if exist bot_process.pid del /F bot_process.pid

echo.
echo Bot has stopped. Press any key to close this window.
pause > nul 