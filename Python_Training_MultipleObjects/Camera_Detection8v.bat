@echo off
cd /d "%~dp0"

set CAMERA=0
set THRESHOLD=0.6
set "MODEL=runs\yolov8n\weights\best.pt"

if not exist "%MODEL%" (
    echo Cannot find %MODEL%
    pause
    exit /b
)

echo Model used: %MODEL%
yolo predict model="%MODEL%" source=%CAMERA% show=True conf=%THRESHOLD% device=0

if errorlevel 1 (
    echo Camera %CAMERA% did not work, trying camera 1 ...
    yolo predict model="%MODEL%" source=1 show=True conf=%THRESHOLD% device=0
)

pause