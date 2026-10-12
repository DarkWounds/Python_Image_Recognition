@echo off
rem Pune acest fisier in acelasi folder cu modelul (.pt) si da dublu-click.
rem Foloseste automat primul fisier .pt gasit in folder.
rem Opresti detectia inchizand fereastra neagra sau cu Ctrl+C.
cd /d "%~dp0"

set CAMERA=0
set PRAG=0.6
set MODEL=

for %%F in (*.pt) do if not defined MODEL set "MODEL=%%F"

if not defined MODEL (
    echo Nu gasesc niciun fisier .pt in acest folder.
    echo Pune modelul ^(de ex. weights.pt sau best.pt^) langa acest .bat.
    echo Daca ai descarcat un .zip de la Roboflow, dezarhiveaza-l intai.
    pause
    exit /b
)

echo Model: %MODEL%
echo Pornesc camera %CAMERA% cu pragul %PRAG% ...
yolo predict model="%MODEL%" source=%CAMERA% show=True conf=%PRAG%

if errorlevel 1 (
    echo.
    echo Camera %CAMERA% nu a mers, incerc camera 1 ...
    yolo predict model="%MODEL%" source=1 show=True conf=%PRAG%
)

pause
