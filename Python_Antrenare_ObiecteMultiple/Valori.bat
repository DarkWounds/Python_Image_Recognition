@echo off
cd /d "%~dp0"

if not exist "Verificare.py" (
    echo Nu gasesc Verificare.py in acest folder.
    pause
    exit /b
)

python Verificare.py
pause