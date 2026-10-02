# EV Charging Station Predictor Launcher
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Starting EV Charging Station Congestion & Grid Load Predictor" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan

Set-Location $PSScriptRoot

if (Test-Path ".\venv\Scripts\python.exe") {
    Write-Host "Using virtual environment Python..." -ForegroundColor Yellow
    & ".\venv\Scripts\python.exe" -m streamlit run app\app.py
} else {
    Write-Host "Using system Python..." -ForegroundColor Yellow
    python -m streamlit run app\app.py
}
