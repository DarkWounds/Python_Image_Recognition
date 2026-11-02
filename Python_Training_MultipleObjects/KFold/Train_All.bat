@echo off
rem Antreneaza automat toate modelele din toate fold-urile, unul dupa altul.
rem fold1 YOLOv8 e sarit, pentru ca e deja antrenat.
cd /d "%~dp0"

for %%F in (fold1 fold2 fold3 fold4 fold5) do (
    echo.
    echo ===== %%F =====
    pushd "%%F"
    if /I not "%%F"=="fold1" python train.py
    python train26.py
    python train12.py
    popd
)

echo.
echo Gata, toate antrenarile s-au terminat.
pause
