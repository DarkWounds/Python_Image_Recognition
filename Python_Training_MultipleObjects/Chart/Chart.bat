@echo off
cd /d "%~dp0"

if not exist "Chart.py" (
    echo Cannot find Chart.py in this folder.
    pause
    exit /b
)

python Chart.py
pause
