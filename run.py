"""Common local entry point. Run python run.py --help."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='MTMCT: local application and office tracking')
    parser.add_argument('command', choices=['app', 'office'])
    parser.add_argument('--seconds', type=float, default=120)
    parser.add_argument('--run-name', default='office_manual_120s')
    args = parser.parse_args()
    if args.command == 'app':
        if not (ROOT/'frontend/dist/index.html').is_file():
            parser.error('Build frontend first: cd frontend, pnpm install, pnpm build')
        command = [sys.executable, '-m', 'uvicorn', 'backend.main:app', '--host', '127.0.0.1', '--port', '8000']
    else:
        command = [sys.executable, str(ROOT/'run_hackaton_experiment.py'), '--seconds', str(args.seconds), '--run-name', args.run_name]
    raise SystemExit(subprocess.call(command, cwd=ROOT))


if __name__ == '__main__':
    main()
