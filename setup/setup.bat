@echo off
echo ================================
echo   Sinkhole AI - Setup Script
echo ================================

REM 1) Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM 2) Activate virtual environment
echo Activating environment...
call venv\Scripts\activate

REM 3) Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM 4) Install requirements
echo Installing Python dependencies...
pip install -r ./setup/requirements.txt

REM 5) Authenticate Earth Engine (manual step)
echo.
echo ===========================================
echo  IMPORTANT: Authenticate Google Earth Engine
echo  A browser window will open. Log in and copy the auth code.
echo ===========================================
earthengine authenticate

echo.
echo Setup complete. Ready to launch FastAPI server!
echo Run this to start the server:
echo   venv\Scripts\activate
echo   uvicorn main:app --host 0.0.0.0 --port 8000
echo.
pause
