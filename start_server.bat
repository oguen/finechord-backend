@echo off
title FineChord Backend
color 0A

echo ====================================
echo    FineChord Backend Server
echo ====================================
echo.

cd /d "%~dp0"

if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    echo [INFO] Virtual environment created.
    echo.
)

echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

echo [INFO] Installing dependencies...
pip install -r requirements.txt --quiet

echo.
echo [INFO] Starting server...
echo    Local:   http://localhost:8000
echo    Docs:    http://localhost:8000/docs
echo    API:     http://localhost:8000/api/health
echo.
echo Press Ctrl+C to stop the server
echo ====================================
echo.

python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000

pause
