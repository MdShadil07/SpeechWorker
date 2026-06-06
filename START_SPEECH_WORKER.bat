@echo off
REM Speech Worker Startup Script with Memory Optimization
echo.
echo ==========================================
echo Speech Worker - Optimized Startup
echo ==========================================
echo.

REM Set memory optimization environment variables
echo Setting memory optimization environment variables...
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
set NUMEXPR_NUM_THREADS=2
set MKL_THREADING_LAYER=GNU
set NUMEXPR_MAX_THREADS=2

REM Optional: Set smaller model and int8 compute type if you want to override defaults
REM set WHISPER_MODEL=tiny.en
REM set WHISPER_COMPUTE_TYPE=int8

echo.
echo Environment Variables Set:
echo   OMP_NUM_THREADS=2
echo   MKL_NUM_THREADS=2
echo   NUMEXPR_NUM_THREADS=2
echo   MKL_THREADING_LAYER=GNU
echo   NUMEXPR_MAX_THREADS=2
echo.

REM Activate virtual environment
echo Activating virtual environment...
cd /d D:\ENGLISH PRACTICE\Learn English\backend
call ..\..\..\venv\Scripts\activate.bat
if errorlevel 1 (
    echo Error: Failed to activate virtual environment
    pause
    exit /b 1
)

echo.
echo Starting Speech Worker on port 8001...
echo.

REM Start the uvicorn server
cd speech-worker
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

pause
