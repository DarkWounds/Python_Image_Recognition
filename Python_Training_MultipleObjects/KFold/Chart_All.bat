@echo off
cd /d "%~dp0"

if not exist "Chart_All.py" (
    echo Cannot find Chart_All.py in this folder.
    pause
    exit /b
)

python Chart_All.py
pause
