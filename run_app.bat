@echo off
echo ===================================================
echo PredictWell - One-Click Launcher
echo ===================================================
echo.

REM Check for Node.js
where node >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Node.js is not installed. Please install it first.
    pause
    exit /b 1
)

echo Starting Backend (using existing script)...
start "PredictWell Backend" "backend\start_backend.bat"

echo Starting Frontend...
start "PredictWell Frontend" cmd /k "cd /d %~dp0 && npm install && npm run dev"

echo.
echo ===================================================
echo App is starting!
echo Backend will run on http://localhost:10000 (or 5000)
echo Frontend will run on http://localhost:3000
echo.
echo Please wait for both windows to finish initializing.
echo Then open http://localhost:3000 in your browser.
echo ===================================================
pause
