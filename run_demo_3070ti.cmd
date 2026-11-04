@echo off
setlocal
REM Run from folder where this file is stored (CMD, Windows)
cd /d "%~dp0"
call conda activate neuro_clean
if errorlevel 1 exit /b 1
python -m mtmct.gpu_check || exit /b 1
python -m mtmct || exit /b 1
endlocal
