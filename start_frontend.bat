# ====================================
# FineChord Frontend - Start Script
# ====================================

# Lance le frontend Next.js en local

cd /d "%~dp0..\frontend"

if not exist "node_modules" (
    echo [INFO] Installing dependencies...
    call npm install
)

echo.
echo [INFO] Starting Next.js dev server...
echo    Local:   http://localhost:3000
echo.
npm run dev

pause
