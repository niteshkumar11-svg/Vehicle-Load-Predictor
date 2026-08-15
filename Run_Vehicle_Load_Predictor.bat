@echo off
title Vehicle Load Predictor - Hajipur MH
echo.
echo  ================================================
echo   Vehicle Load Prediction Dashboard
echo   Flipkart - Hajipur Mother Hub
echo  ================================================
echo.
cd /d "%~dp0"
python -m streamlit run "Vehicle_Load_Predictor.py" --server.port 8504 --browser.gatherUsageStats false
pause
