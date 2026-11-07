@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_EXE=%CONDA_PREFIX%\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=%USERPROFILE%\.conda\envs\neuro_clean\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo Python environment neuro_clean or .venv is missing. See docs\WEB_INTERFACE.md.
  pause
  exit /b 1
)
if not exist "frontend\dist\index.html" (
  echo Build frontend first: cd frontend ^&^& pnpm install --frozen-lockfile ^&^& pnpm build
  echo See docs\WEB_INTERFACE.md for setup instructions.
  pause
  exit /b 1
)
set OPENCV_FFMPEG_THREADS=1
set PYTHONIOENCODING=utf-8
echo Open http://127.0.0.1:8000 in your browser. Ctrl+C stops the app.
"%PYTHON_EXE%" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
pause
