@echo off
title Stop FineChord Backend
echo ====================================
echo    Stopping FineChord Backend
echo ====================================
echo.

taskkill /F /IM python.exe /FI "WINDOWTITLE eq FineChord Backend*" 2>nul

if %errorlevel% equ 0 (
    echo [OK] Server stopped.
) else (
    echo [INFO] No running server found.
)

echo.
pause
