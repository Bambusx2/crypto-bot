@echo off
setlocal EnableDelayedExpansion

:: Default values
set SYMBOL=BTC/USDT
set SIDE=sell
set AMOUNT=11
set LEVERAGE=10

:: Check for command line arguments
if not "%~1"=="" set SYMBOL=%~1
if not "%~2"=="" set SIDE=%~2
if not "%~3"=="" set AMOUNT=%~3
if not "%~4"=="" set LEVERAGE=%~4

:: Show banner
echo ================================================
echo EXECUTING CUSTOM DIRECT TRADE - EMERGENCY SYSTEM
echo ================================================
echo This script will IMMEDIATELY execute a custom trade 
echo regardless of market conditions or trading strategy logic.
echo.
echo TRADE SETTINGS:
echo - Trading pair: %SYMBOL%
echo - Direction: %SIDE% (%SIDE:sell=Short%%SIDE:buy=Long%)
echo - Size: %AMOUNT% contracts
echo - Leverage: %LEVERAGE%x
echo.
echo WARNING: This will create REAL trades if test_mode is disabled!
echo Press Ctrl+C NOW to cancel if you didn't intend to do this.
echo.
timeout /t 5

echo Running custom trade execution...
python execute_direct_trade.py --symbol "%SYMBOL%" --side %SIDE% --amount %AMOUNT% --leverage %LEVERAGE%

echo.
if %ERRORLEVEL% EQU 0 (
    echo ✓ CUSTOM TRADE EXECUTED SUCCESSFULLY
) else (
    echo ✗ CUSTOM TRADE EXECUTION FAILED
    echo Check direct_trade.log for details
)

echo.
echo You can check order status in the exchange dashboard
echo Press any key to close this window...
pause > nul

endlocal 