@echo off
echo Starting Trading Bot Dashboard...
echo ========================================
echo Do not close this window while the dashboard is running!
echo The dashboard will be available at: http://localhost:8501
echo ========================================
echo.

python -m streamlit run cryptobot/gui/dashboard.py

echo.
echo Dashboard has stopped. Press any key to close this window.
pause > nul 