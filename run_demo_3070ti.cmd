@echo off
setlocal
REM Run from folder where this file is stored (CMD, Windows)
cd /d "%~dp0"
call conda activate neuro_clean
if errorlevel 1 exit /b 1
python -m mtmct.gpu_check || exit /b 1
python -m mtmct.download --split train --scene Warehouse_000 --cameras 0 1 || exit /b 1
python -m mtmct.weights --model osnet_x1_0 || exit /b 1
python -m mtmct.tracking --scene-dir data/MTMC_Tracking_2026/train/Warehouse_000 --out runs/3070ti_2cams --model yolo26l.pt --reid osnet --reid-model osnet_x1_0 --device cuda:0 --imgsz 960 --conf 0.15 --max-frames 300 --stride 1 --sample-every 5 || exit /b 1
REM First run: ReID-only matching + time guards. Geometry must be validated before use.
python -m mtmct.link --run-dir runs/3070ti_2cams --scene-id 0 --ignore-geometry --spatial-weight 0 || exit /b 1
python -m mtmct.candidates --run-dir runs/3070ti_2cams || exit /b 1
python -m mtmct.visualize --scene-dir data/MTMC_Tracking_2026/train/Warehouse_000 --pred runs/3070ti_2cams/global_tracks_2d.csv --out runs/3070ti_2cams/mosaic.mp4 --max-frames 0 --stride 1 --scale 0.45 || exit /b 1
echo DONE: runs\3070ti_2cams\mosaic.mp4
endlocal
