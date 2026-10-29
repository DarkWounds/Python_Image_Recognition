@echo off
rem Put this file in the headphones_dataset folder (next to train.py) and double-click it.
cd /d "%~dp0"

if not exist "train.py" (
    echo Cannot find train.py in this folder.
    pause
    exit /b
)

python train.py
echo.
echo Done. The model is in runs\headphones\weights\best.pt
pause
