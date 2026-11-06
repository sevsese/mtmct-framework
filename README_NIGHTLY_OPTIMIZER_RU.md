# Ночной подбор параметров MTMCT: четыре камеры и проверка на других сценах

## Установка

Распакуй архив в **корень** `MTMCT_GPU_3070Ti` (в ту же папку, где `mtmct/`). В архиве:

- `run_nightly_optimize.py` — ночной оптимизатор, запускаемый кнопкой ▶.
- `run_compare_all_osnet.py` — проверенный ранее код повторного извлечения ReID-признаков с одинаковых bbox.
- `run_best_4cameras.py` — визуализация победителя после завершения поиска, тоже кнопкой ▶.
- `README_NIGHTLY_OPTIMIZER_RU.md` — эта инструкция.

Нужен твой MTMCT с исправлением `occlusion_repair.py` v2 и поддержкой моделей
`osnet_x1_0`, `osnet_ibn_x1_0`, `osnet_ain_x1_0` в `mtmct/weights.py`.
Среда должна содержать `torch`, `ultralytics`, `torchreid`, `opencv-python`,
`numpy`, `scipy`, `ijson`, `huggingface_hub`. В VS Code выбирай Conda `neuro_clean`.

## Просто запустить

1. В редакторе открой `run_nightly_optimize.py`.
2. При необходимости настрой константы в начале файла.
3. Нажми **▶ Run Python File**.
4. Программа сама загрузит **недостающие** RGB MP4, `calibration.json` и
   `ground_truth.json` для выбранных сцен, используя Hugging Face. Видео могут быть большими.
5. Если запуск прервался, запусти файл снова. `RESUME=True` позволяет не повторять
   законченные прогоны. Не редактируй `EXPERIMENT_NAME` для продолжения.

## Как происходит поиск

1. `train/Warehouse_000` и `train/Warehouse_002`: программа выбирает по 4 камеры
   (с достаточной GT-активностью и общими идентичностями). Калибровка и GT проверяются.
2. Тестируются 6 *заранее заданных* конфигураций YOLO+ByteTrack по detection F1.
   **Это приближение:** лучшая детекция не гарантирует лучший полный IDF1.
3. На лучшем детекторе тестируются OSNet, OSNet-IBN, OSNet-AIN, сохраняя
   *одинаковые* локальные треки и кадры для каждой модели.
4. Для каждой модели запускаются reference + 80 случайных + 25 локальных
   испытаний параметров association, memory и occlusion repair.
5. Победитель выбирается **только по среднему IDF1 train-сцен**.
6. **Только победитель** запускается на `val/Warehouse_020` и `val/Warehouse_022`.
   Validation не используется для изменения параметров.

## Какие параметры подбираются

- YOLO: `imgsz`, `conf` (по кандидатам в `DETECTOR_TRIALS`);
- ByteTrack: `track_high_thresh`, `track_low_thresh`, `new_track_thresh`,
  `track_buffer`, `match_thresh`;
- ReID: архитектура OSNet, OSNet-IBN или OSNet-AIN;
- Association: `matcher`, `min_similarity`, `max_cost`, `appearance_weight`,
  `spatial_weight`, `temporal_weight`, `max_gap_seconds`, `max_speed_mps`,
  `spatial_slack_m`, `overlap_distance_m`, geometry, time,
  same-camera fragments и graph identity correction;
- Global Memory: включение и `memory_similarity`;
- Occlusion Repair: включение, IoU и similarity дубликатов, интервал
  восстановления и пороги коррекции с подтверждением соседней камеры.

**Не подбираются автоматически** параметры временной синхронизации (их нельзя
восстанавливать подгонкой к GT), топология помещений, веса самой нейросети,
`sample_every` (в этом поиске фиксирован равным 5), `fuse_score` ByteTrack и
фактический масштаб FPS/калибровки. Полного перебора бесконечного пространства
параметров не существует; это случайный поиск с локальным уточнением.

## Результаты

В `runs/overnight_v1/`:

- `detector_ranking.json` — рейтинг детектора и трекера;
- `search_det_..._osnet_....jsonl` — журнал каждой попытки (дописывается сразу);
- `train_leaderboard.json` — лучшие конфигурации на обучающих сценах;
- `best_train_config.json` — **выбранный до валидации** победитель;
- `best_by_model.json` — лучшие наборы порогов отдельно для каждой OSNet;
- `validation_results.json` — результаты на независимых сценах;
- `selected_cameras.json` — конкретные номера камер для каждой сцены;
- `summary.txt` — главный итог;
- `FAILED_TRACEBACK.txt` — подробности исключения, если что-то не удалось.

После завершения открой `run_best_4cameras.py` и нажми ▶: он построит
`runs/overnight_v1/best_video/train_Warehouse_000/mosaic.mp4`, `metrics.json`.

## Важные ограничения

- Нельзя считать, что подогнанные на синтетических NVIDIA пороги оптимальны
  для настоящих камер. Отдельно проверь CITIC-MTMC и SCOUT.
- Проверка на val-сценах — holdout для этого эксперимента. Не используй её
  для постоянного подбора порогов, иначе она тоже превратится в train.
- Метрика — пользовательская `global-2d-idf1-v1`, не официальный NVIDIA 3D HOTA.
- В память загружаются кэшированные tracklets и записываются временные результаты.
  Если места на диске мало, оставь `KEEP_TRIAL_OUTPUTS=False`.
- Если возникла проблема или кончилась видеопамять, подробности в
  `FAILED_TRACEBACK.txt`. Уменьшай `IMGSZ` в `DETECTOR_TRIALS` только для нового
  `EXPERIMENT_NAME`.
- В тестовом окружении автора проверены синтаксис, генерация конфигураций и
  реальная работа association на сохранённых IBN tracklets. Полного GPU-поиска
  на новых видеозаписях автор не выполнял.

## Внешние датасеты

Подбирать параметры рекомендуем на NVIDIA 2026 (поддерживается из коробки).
Другие наборы потребуют адаптеров к единому формату камер и GT:

- **CITIC-MTMC**: https://github.com/GTEC-UDC/citic-mtmc-dataset — 5 камер в 4 зонах,
  видео на Zenodo, GT положения и маршрутов (не идентичная покадровая bbox GT NVIDIA).
- **SCOUT**: https://scout.epfl.ch/ — реальные люди, multi-view 2D и 3D,
  официальные инструменты: https://github.com/cvlab-epfl/scout_toolkit
- **WILDTRACK**: https://www.epfl.ch/labs/cvlab/data/data-wildtrack/ — 7 камер,
  калибровка, синхронные аннотированные изображения.
- **MultiviewX**: https://github.com/hou-yz/MultiviewX — 6 синтетических камер,
  ground-plane detection.
- **i.c.sens**: https://data.uni-hannover.de/en/dataset/multi-view-pedestrian-tracking

Не смешивай их GT без явного конвертера и проверки определения метрики.
