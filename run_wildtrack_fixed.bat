@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=python"
if exist "%USERPROFILE%\.conda\envs\neuro_clean\python.exe" set "PYTHON=%USERPROFILE%\.conda\envs\neuro_clean\python.exe"
echo Using Python: %PYTHON%
"%PYTHON%" "%~dp0run_wildtrack_fixed.py" %*
if errorlevel 1 (
  echo.
  echo ERROR: WILDTRACK run failed. Check the message above.
  exit /b 1
)
endlocal
