@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Python environment missing. See docs\WEB_INTERFACE.md.
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
".venv\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
pause
