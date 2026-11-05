@echo off
cd /d "%~dp0"

if not exist "Object_Detection.py" (
    echo Cannot find train12.py in this folder.
    pause
    exit /b
)

python Object_Detection.py
echo.
echo Done. Photos are saved