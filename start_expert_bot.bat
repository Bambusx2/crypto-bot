@echo off
echo ================================================
echo STARTING EXPERT TRADING BOT 
echo ================================================
echo Advanced multi-factor trading strategy with:
echo - Combined SMA, EMA, RSI, MACD and Bollinger Band analysis
echo - Volume confirmation and volatility adaptation
echo - Dynamic position sizing based on signal strength
echo - Optimized risk management with trailing stops
echo.
echo This bot will identify high-quality trading setups
echo across multiple cryptocurrencies using professional
echo analysis techniques.
echo.
echo ========================================
echo Do not close this window while the bot is running!
echo To stop the bot, either press Ctrl+C or use the Dashboard.
echo ========================================
echo.

:: Stop any existing instances
echo Stopping any previous instances...
taskkill /FI "WINDOWTITLE eq Trading Bot*" /T /F > nul 2>&1
taskkill /FI "WINDOWTITLE eq *cryptobot*" /T /F > nul 2>&1

:: Wait for processes to terminate
timeout /t 2 /nobreak > nul

:: Create a PID file with a numeric placeholder
echo 0 > bot_process.pid

:: Set log level
set LOG_LEVEL=INFO

:: Start the bot with the expert strategy configuration
python -m cryptobot --config config/expert_strategy.yml

:: Clean up PID file when bot exits
if exist bot_process.pid del /F bot_process.pid

echo.
echo Bot has stopped. Press any key to close this window.
pause > nul 