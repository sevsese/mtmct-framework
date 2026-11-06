#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
if [ ! -f frontend/dist/index.html ]; then
  echo 'Build frontend first: cd frontend && pnpm install --frozen-lockfile && pnpm build'
  exit 1
fi
export OPENCV_FFMPEG_THREADS=1 PYTHONIOENCODING=utf-8
echo 'Open http://127.0.0.1:8000 in your browser. Ctrl+C stops the app.'
exec .venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
