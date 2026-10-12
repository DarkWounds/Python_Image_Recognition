@echo off
rem Pune acest fisier in folderul casti_dataset (langa antrenare.py) si da dublu-click.
rem Opresti detectia cu Ctrl+C in fereastra neagra sau inchizand-o.
cd /d "%~dp0"

set MODEL=runs\casti\weights\best.pt
set CAMERA=0
set PRAG=0.6

if not exist "%MODEL%" (
    echo Nu gasesc modelul %MODEL%.
    echo Ruleaza intai Antrenare.bat sau verifica ca fisierul e in folderul casti_dataset.
    pause
    exit /b
)

echo Pornesc camera %CAMERA% cu pragul %PRAG% ...
yolo predict model="%MODEL%" source=%CAMERA% show=True conf=%PRAG%

if errorlevel 1 (
    echo.
    echo Camera %CAMERA% nu a mers, incerc camera 1 ...
    yolo predict model="%MODEL%" source=1 show=True conf=%PRAG%
)

pause
