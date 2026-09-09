@echo off
rem One-click launcher for the audio data frontend (qt/frontend.py)
rem Launches the GUI and closes this console window immediately.

cd /d "%~dp0"

set "PY=D:\anaconda\envs\vidio_data\pythonw.exe"

if not exist "%PY%" (
    echo [ERROR] Python not found: %PY%
    echo Please reinstall the conda env 'vidio_data' first.
    pause
    exit /b 1
)

start "" "%PY%" "%~dp0qt\frontend.py"

exit
