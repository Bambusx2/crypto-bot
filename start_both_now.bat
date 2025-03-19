@echo off
echo Starting TRADE NOW Bot System...
echo ========================================
echo This will start both the TRADE NOW Bot and Dashboard
echo This configuration uses SMALLER SMA WINDOWS (3 and 5 periods)
echo The bot should generate trades MUCH FASTER
echo ========================================
echo.

:: Stop any existing instances
echo Stopping any previous instances...
taskkill /FI "WINDOWTITLE eq Trading Bot*" /T /F > nul 2>&1
taskkill /FI "WINDOWTITLE eq *cryptobot*" /T /F > nul 2>&1
taskkill /FI "WINDOWTITLE eq *dashboard*" /T /F > nul 2>&1

:: Wait for processes to terminate
timeout /t 2 /nobreak > nul

:: Start the trading bot in a new command window
start "Trading Bot (TRADE NOW SETTINGS)" cmd /k start_trade_now.bat

:: Wait for the bot to initialize
timeout /t 5 /nobreak > nul

:: Start the dashboard in a new command window
start "Trading Bot Dashboard" cmd /k start_dashboard.bat

echo.
echo Both the TRADE NOW Bot and Dashboard have been started.
echo You can close this window now.
echo. 