@echo off
rem Pune acest fisier in folderul casti_dataset (langa antrenare.py) si da dublu-click.
cd /d "%~dp0"

if not exist "antrenare.py" (
    echo Nu gasesc antrenare.py in acest folder.
    pause
    exit /b
)

python antrenare.py
echo.
echo Gata. Modelul e in runs\casti\weights\best.pt
pause
