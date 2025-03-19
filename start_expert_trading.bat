@echo off
echo ================================================
echo STARTING EXPERT TRADING SYSTEM
echo ================================================
echo This script will start both the Expert Trading Bot and Dashboard
echo.
echo PROFESSIONAL TRADING FEATURES:
echo - Multi-factor analysis (SMA, EMA, RSI, MACD, Bollinger Bands)
echo - Volume and volatility confirmation
echo - Dynamic position sizing based on signal strength
echo - Intelligent trend detection with multiple timeframes
echo - Optimized risk management with trailing stops
echo.
echo ========================================
echo STARTING SYSTEM...
echo ========================================

:: Stop any existing instances
echo Stopping any previous instances...
taskkill /FI "WINDOWTITLE eq Trading Bot*" /T /F > nul 2>&1
taskkill /FI "WINDOWTITLE eq *cryptobot*" /T /F > nul 2>&1
taskkill /FI "WINDOWTITLE eq *dashboard*" /T /F > nul 2>&1

:: Wait for processes to terminate
timeout /t 2 /nobreak > nul

:: Start the expert trading bot in a new command window
start "Trading Bot (EXPERT MODE)" cmd /k start_expert_bot.bat

:: Wait for the bot to initialize
timeout /t 5 /nobreak > nul

:: Start the dashboard in a new command window
start "Trading Bot Dashboard" cmd /k start_dashboard.bat

echo.
echo Both the EXPERT Trading Bot and Dashboard have been started.
echo You can close this window now.
echo. 