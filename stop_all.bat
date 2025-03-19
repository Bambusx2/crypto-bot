@echo off
echo Stopping Trading Bot and Dashboard...

:: First try to gracefully terminate by window title
taskkill /FI "WINDOWTITLE eq Trading Bot*" /T /F > nul 2>&1
taskkill /FI "WINDOWTITLE eq Trading Bot Dashboard*" /T /F > nul 2>&1

:: Then also look for python processes
taskkill /FI "IMAGENAME eq python.exe" /FI "WINDOWTITLE eq *cryptobot*" /T /F > nul 2>&1
taskkill /FI "IMAGENAME eq python.exe" /FI "WINDOWTITLE eq *streamlit*" /T /F > nul 2>&1

:: Remove the process ID file if it exists
if exist bot_process.pid del /F bot_process.pid

echo All processes have been stopped.
echo Press any key to close this window.
pause > nul 