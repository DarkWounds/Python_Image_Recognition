@echo off
cd /d "%~dp0"

if not exist "Evaluate.py" (
    echo Cannot find Evaluate.py in this folder.
    pause
    exit /b
)

python Evaluate.py
pause
