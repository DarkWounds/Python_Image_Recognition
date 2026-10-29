@echo off
rem Put this file in the same folder as the model (.pt) and double-click it.
rem It automatically uses the first .pt file found in the folder.
rem Stop the detection by closing the black window or with Ctrl+C.
cd /d "%~dp0"

set CAMERA=0
set THRESHOLD=0.6
set MODEL=

for %%F in (*.pt) do if not defined MODEL set "MODEL=%%F"

if not defined MODEL (
    echo Cannot find any .pt file in this folder.
    echo Put the model ^(e.g. weights.pt or best.pt^) next to this .bat.
    echo If you downloaded a .zip from Roboflow, unzip it first.
    pause
    exit /b
)

echo Model: %MODEL%
echo Starting camera %CAMERA% with threshold %THRESHOLD% ...
yolo predict model="%MODEL%" source=%CAMERA% show=True conf=%THRESHOLD%

if errorlevel 1 (
    echo.
    echo Camera %CAMERA% did not work, trying camera 1 ...
    yolo predict model="%MODEL%" source=1 show=True conf=%THRESHOLD%
)

pause
