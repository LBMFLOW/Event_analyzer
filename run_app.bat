@echo off
setlocal

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    where py >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_EXE=py -3"
    ) else (
        set "PYTHON_EXE=python"
    )
)

echo Starting Time Series Threshold App...
%PYTHON_EXE% -m time_series_threshold_app

if errorlevel 1 (
    echo.
    echo The app exited with an error. Make sure dependencies are installed:
    echo     pip install -r requirements.txt
    echo.
    pause
)

endlocal
