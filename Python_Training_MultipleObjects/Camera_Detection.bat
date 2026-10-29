@echo off
rem Put this file in the training folder (next to train.py) or next to your .pt model,
rem then double-click it. Stop the detection by closing the black window or with Ctrl+C.
cd /d "%~dp0"

set CAMERA=0
set THRESHOLD=0.6
set MODEL=

rem Order in which it looks for YOUR model (never yolov8n.pt, which is the base model):
if exist "runs\model\weights\best.pt" set "MODEL=runs\model\weights\best.pt"
if not defined MODEL if exist "runs" for /f "delims=" %%F in ('dir /s /b /o-d "runs\best.pt" 2^>nul') do if not defined MODEL set "MODEL=%%F"
if not defined MODEL if exist "best.pt" set "MODEL=best.pt"
if not defined MODEL if exist "weights.pt" set "MODEL=weights.pt"

if not defined MODEL (
    echo Cannot find your trained model.
    echo Put this .bat next to train.py ^(it looks in runs\model\weights\best.pt^)
    echo or next to the best.pt / weights.pt file.
    echo Note: yolov8n.pt is NOT your model, it is the base model that detects 80 general objects.
    pause
    exit /b
)

echo Model used: %MODEL%
echo Starting camera %CAMERA% with threshold %THRESHOLD% ...
yolo predict model="%MODEL%" source=%CAMERA% show=True conf=%THRESHOLD% device=0

if errorlevel 1 (
    echo.
    echo Camera %CAMERA% did not work, trying camera 1 ...
    yolo predict model="%MODEL%" source=1 show=True conf=%THRESHOLD% device=0
)

pause
