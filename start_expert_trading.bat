@echo off
echo ================================================
echo STARTING PROFESSIONAL CRYPTO TRADING SYSTEM
echo ================================================
echo.
echo ADVANCED TRADING FEATURES:
echo - Multi-strategy approach with trend following and mean reversion
echo - Comprehensive technical analysis tools (SMA, EMA, RSI, MACD, Bollinger Bands)
echo - Advanced risk management with dynamic position sizing
echo - Trailing stops and partial take-profits for maximizing gains
echo - Smart correlation control to minimize portfolio risk
echo - Adaptive stops based on market volatility
echo - Intelligent trade management with performance tracking
echo.

:: Create necessary directories if they don't exist
if not exist "data" mkdir data
if not exist "logs" mkdir logs

:: Stop any existing instances
echo Stopping previous instances...
taskkill /FI "WINDOWTITLE eq Trading Bot*" /T /F > nul 2>&1
taskkill /FI "WINDOWTITLE eq Dashboard*" /T /F > nul 2>&1
timeout /t 2 /nobreak > nul

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Start Trading Bot in a new window
echo Starting Professional Trading Bot...
start "Trading Bot (PROFESSIONAL MODE)" cmd /k "venv\Scripts\activate.bat && python -m cryptobot --config config/expert_strategy.yml"

:: Wait a moment for the bot to initialize
timeout /t 3 /nobreak > nul

:: Start Dashboard in a new window
echo Starting Professional Trading Dashboard...
start "Dashboard" cmd /k "venv\Scripts\activate.bat && python -m streamlit run cryptobot/gui/dashboard.py"

echo.
echo ========================================
echo PROFESSIONAL Trading Bot and Dashboard have been started!
echo Multiple strategies will work together to maximize profits.
echo.
echo IMPORTANT: Keep both the bot and dashboard windows open!
echo Dashboard will be available at: http://localhost:8501
echo ========================================
echo.

:: Deactivate virtual environment
call venv\Scripts\deactivate.bat

pause 