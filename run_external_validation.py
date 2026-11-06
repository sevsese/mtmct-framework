"""External holdout check after NVIDIA train/val optimization. Run ▶ in VS Code.

Uses frozen best_train_config.json. Does not tune any settings on external GT.
Dataset files must be downloaded manually from official sources first.
"""
from pathlib import Path
import run_dataset as runner

ROOT = Path(__file__).resolve().parent
BEST_CONFIG = ROOT / 'runs' / 'overnight_v1' / 'best_train_config.json'
RUN_NAME = 'external_holdout_v1'

# Which held-out external datasets should be run:
DO_SCOUT = True
DO_CITIC = True
SCOUT_CAMERAS = (0, 1, 2, 3)  # manually verified annotations in sequence_1
CITIC_CAMERAS = None  # available 5 named cameras, when videos are found


def main():
    if not BEST_CONFIG.is_file():
        raise FileNotFoundError(
            f'First finish run_nightly_optimize.py or specify your winner file: {BEST_CONFIG}')
    runner.BEST_CONFIG_FILE = BEST_CONFIG
    runner.RUN_NAME = RUN_NAME
    runner.MAX_FRAMES = 900
    runner.LINK_ONLY = False
    runner.RENDER = True
    if DO_SCOUT:
        runner.DATASET = 'scout'
        runner.SCENE = 'sequence_1'
        runner.CAMERAS = SCOUT_CAMERAS
        runner.EVALUATE = True
        runner.ALLOW_SCOUT_PSEUDO_LABELS = False
        print('=== SCOUT: independent bbox-IDF1 evaluation ===', flush=True)
        runner.main()
    if DO_CITIC:
        runner.DATASET = 'citic'
        runner.SCENE = 'capture1'
        runner.CAMERAS = CITIC_CAMERAS
        runner.EVALUATE = False  # CITIC GT is WORLD POSITION ONLY!
        print('=== CITIC: video inference; world metric requires separate calibration ===', flush=True)
        runner.main()
    print('DONE: see runs/', RUN_NAME)


if __name__ == '__main__':
    main()
