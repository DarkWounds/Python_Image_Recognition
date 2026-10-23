@echo off
rem Pune acest fisier in folderul casti_dataset (langa antrenare26.py) si da dublu-click.
cd /d "%~dp0"

if not exist "antrenare26.py" (
    echo Nu gasesc antrenare26.py in acest folder.
    pause
    exit /b
)

python antrenare26.py
echo.
echo Gata. Modelul e in runs\model\weights\best.pt
pause
