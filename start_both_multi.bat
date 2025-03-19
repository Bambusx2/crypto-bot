@echo off
echo Starting Multi-Coin Trading Bot System...
echo ========================================
echo This will start both the Trading Bot and Dashboard
echo The bot will monitor multiple cryptocurrencies and trade on strong signals only
echo ========================================
echo.

:: Start the trading bot in a new command window
start "Trading Bot (MULTI-COIN MODE)" cmd /k start_bot.bat

:: Wait for the bot to initialize
timeout /t 2 /nobreak > nul

:: Start the dashboard in a new command window
start "Trading Bot Dashboard" cmd /k start_dashboard.bat

echo.
echo Both the Trading Bot and Dashboard have been started.
echo You can close this window now.
echo. 