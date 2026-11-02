@echo off
rem Ruleaza Evaluate.py in toate fold-urile, unul dupa altul,
rem si copiaza rezultatele in KFold\Values_fold1.txt ... Values_fold5.txt
cd /d "%~dp0"

for %%F in (fold1 fold2 fold3 fold4 fold5) do (
    echo.
    echo ===== %%F =====
    pushd "%%F"
    python Evaluate.py
    copy /Y "Chart\Values.txt" "..\Values_%%F.txt" >nul
    popd
)

echo.
echo Gata. Rezultatele sunt in Values_fold1.txt ... Values_fold5.txt
pause
