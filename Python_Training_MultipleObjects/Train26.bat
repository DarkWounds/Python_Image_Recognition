@echo off
rem Put this file in the dataset folder (next to train26.py) and double-click it.
cd /d "%~dp0"

if not exist "train26.py" (
    echo Cannot find train26.py in this folder.
    pause
    exit /b
)

python train26.py
echo.
echo Done. The model is in runs\yolo26\weights\best.pt
pause
