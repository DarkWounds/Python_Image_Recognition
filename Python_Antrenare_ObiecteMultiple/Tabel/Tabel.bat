@echo off
cd /d "%~dp0"

if not exist "Tabel.py" (
    echo Nu gasesc Tabel.py in acest folder.
    pause
    exit /b
)

python Tabel.py
pause