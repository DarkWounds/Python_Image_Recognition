@echo off
rem Put this file in the headphones_dataset folder (next to train.py) and double-click it.
rem Stop the detection with Ctrl+C in the black window or by closing it.
cd /d "%~dp0"

set MODEL=runs\headphones\weights\best.pt
set CAMERA=0
set THRESHOLD=0.6

if not exist "%MODEL%" (
    echo Cannot find the model %MODEL%.
    echo Run Train.bat first or check that the file is in the headphones_dataset folder.
    pause
    exit /b
)

echo Starting camera %CAMERA% with threshold %THRESHOLD% ...
yolo predict model="%MODEL%" source=%CAMERA% show=True conf=%THRESHOLD%

if errorlevel 1 (
    echo.
    echo Camera %CAMERA% did not work, trying camera 1 ...
    yolo predict model="%MODEL%" source=1 show=True conf=%THRESHOLD%
)

pause
