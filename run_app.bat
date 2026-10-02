@echo off
title EV Charging Station Predictor
echo ==========================================================
echo Starting EV Charging Station Congestion & Grid Load Predictor
echo ==========================================================
cd /d "%~dp0"

if exist "venv\Scripts\python.exe" (
    echo Using virtual environment Python...
    .\venv\Scripts\python.exe -m streamlit run app\app.py
) else (
    echo Virtual environment not found. Using system Python...
    python -m streamlit run app\app.py
)

pause
