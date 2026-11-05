@echo off
setlocal
REM Run from folder where this file is stored (CMD, Windows)
cd /d "%~dp0"
if not defined OPENCV_FFMPEG_THREADS set "OPENCV_FFMPEG_THREADS=1"
if not exist ".venv\Scripts\python.exe" (
    echo Create .venv and install dependencies first. See README.md.
    exit /b 1
)
".venv\Scripts\python.exe" -m mtmct %* || exit /b 1
endlocal
