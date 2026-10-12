@echo off
rem Pune acest fisier in folderul cu antrenarea (langa antrenare.py) sau langa modelul tau .pt,
rem apoi da dublu-click. Opresti detectia inchizand fereastra neagra sau cu Ctrl+C.
cd /d "%~dp0"

set CAMERA=0
set PRAG=0.6
set MODEL=

rem Ordinea in care cauta modelul TAU (niciodata yolov8n.pt, care e modelul de baza):
if exist "runs\model\weights\best.pt" set "MODEL=runs\model\weights\best.pt"
if not defined MODEL if exist "runs" for /f "delims=" %%F in ('dir /s /b /o-d "runs\best.pt" 2^>nul') do if not defined MODEL set "MODEL=%%F"
if not defined MODEL if exist "best.pt" set "MODEL=best.pt"
if not defined MODEL if exist "weights.pt" set "MODEL=weights.pt"

if not defined MODEL (
    echo Nu gasesc modelul tau antrenat.
    echo Pune acest .bat langa antrenare.py ^(cauta in runs\model\weights\best.pt^)
    echo sau langa fisierul best.pt / weights.pt.
    echo Atentie: yolov8n.pt NU e modelul tau, e modelul de baza care detecteaza 80 de obiecte generale.
    pause
    exit /b
)

echo Model folosit: %MODEL%
echo Pornesc camera %CAMERA% cu pragul %PRAG% ...
yolo predict model="%MODEL%" source=%CAMERA% show=True conf=%PRAG% device=0

if errorlevel 1 (
    echo.
    echo Camera %CAMERA% nu a mers, incerc camera 1 ...
    yolo predict model="%MODEL%" source=1 show=True conf=%PRAG% device=0
)

pause
