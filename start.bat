@echo off
title wave.io
cd /d "%~dp0"

echo.
echo  wave.io ^| Physics-Informed Neural Network Wave Simulator
echo  =================================================================
echo.

:: Prefer Python 3.13 (has all deps), fall back to any python on PATH
set PYEXE=
if exist "C:\Users\HP\AppData\Local\Programs\Python\Python313\python.exe" (
    set PYEXE=C:\Users\HP\AppData\Local\Programs\Python\Python313\python.exe
) else (
    set PYEXE=python
)

"%PYEXE%" --version 2>nul || (
    echo  [ERROR] Python not found.
    pause & exit /b 1
)

:: Install deps if missing
"%PYEXE%" -c "import fastapi" 2>nul || (
    echo  Installing dependencies...
    "%PYEXE%" -m pip install fastapi "uvicorn[standard]" websockets numpy -q
)

echo.
echo  Server starting at http://localhost:8000
echo  The browser will open automatically.
echo  Press Ctrl+C to stop.
echo.

:: Open browser after 2-second delay
start "" cmd /c "timeout /t 2 >nul && start http://localhost:8000"

:: Launch server
cd backend
"%PYEXE%" -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

pause
