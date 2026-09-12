@echo off
rem One-click launcher for the audio data frontend (qt/frontend.py)
rem Launches the GUI and closes this console window immediately.

cd /d "%~dp0"

set "PY=D:\model\model\vidio\.venv\pythonw.exe"

if not exist "%PY%" (
    echo [ERROR] Python not found: %PY%
    echo Please check the Python path in the .venv folder.
    pause
    exit /b 1
)

start "" "%PY%" "%~dp0qt\frontend.py"

exit
