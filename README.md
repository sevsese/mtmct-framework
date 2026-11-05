# MTMCT Framework

Запускаемый baseline межкамерного трекинга людей: YOLO26l + ByteTrack + OSNet x1.0.

Командная архитектура и распределение задач описаны в `docs/ARCHITECTURE.md` и `docs/ROADMAP.md`. Правила веток и Pull Request находятся в `CONTRIBUTING.md`.

## Быстрый запуск

Из Anaconda Prompt:

```bat
conda activate neuro_clean
cd /d "C:\path\to\mtmct-framework"
python -m mtmct
```

Для разработки:

```bat
python -m pip install -e ".[dev]"
python -m pytest -q
```

Или запусти `run_demo.cmd`. Одна команда скачивает недостающие видео, выполняет локальный трекинг, связывает Local ID в Global ID, создаёт CSV и `mosaic.mp4`.

Стандартный сценарий:

- сцена `Warehouse_000`, камеры `3` и `15`;
- 900 кадров (30 секунд) с каждой камеры;
- ReID + время; геометрия исправлена и доступна опцией, но по умолчанию выключена, так как не улучшила IDF1;
- порог детектора `0.10`;
- жёсткий порог cosine similarity `0.80`;
- максимальная стоимость связи `0.125`;
- результат в `runs/warehouse_000_cam3_15/`.

Камеры `3/15` выбраны по официальному train ground truth: за первые 900 кадров камера 3 видит 36 людей, камера 15 — 33, общих ID — 33. Старые камеры `0/1` в первых 300 кадрах общих людей не имеют. Ground truth используется только для выбора/оценки, но никогда не передаётся алгоритму сопоставления.

## Настройка запуска

```bat
python -m mtmct --cameras 3 15 --max-frames 900 --conf 0.10 --min-similarity 0.80 --max-cost 0.125
```

Полезные параметры:

- `--max-frames 0` — обработать видео целиком;
- `--device cuda:0` — выбрать NVIDIA GPU или `--device cpu` для запуска без CUDA;
- `--imgsz 768` — уменьшить нагрузку на GPU;
- `--min-similarity 0.85` — ещё строже связывать Local ID;
- `--link-only` — пересчитать Global ID без повторного YOLO/OSNet;
- `--no-download` — завершиться ошибкой, если локальных входных файлов нет;
- `--geometry` — включить исправленную проекцию на плоскость пола `(X,Y)`.

При изменении только порогов используй повторно сохранённые признаки:

```bat
python -m mtmct --link-only --no-download --min-similarity 0.85
```

В `--link-only` видео, CUDA и визуализация не нужны. Новые прогоны содержат
`manifest.json`; несовместимая модель, изменённый вход или размер embedding
вызывают понятную ошибку вместо тихого использования устаревшего кэша.

## Результаты

В каталоге запуска создаются:

| Файл | Содержимое |
|---|---|
| `observations.csv` | боксы и Local ID по кадрам |
| `local_tracklets.json` | локальные треки и усреднённые OSNet-признаки |
| `global_ids.json` | `(camera_id, local_id) -> global_id` |
| `global_tracks_2d.csv` | обнаружения с Global ID |
| `performance.json` | FPS и пиковая CUDA-память |
| `manifest.json` | git SHA, модели, конфигурация и хэши входов |
| `association_edges.json` | рассмотренные matcher-рёбра и их стоимость |
| `mosaic.mp4` | синхронная визуализация камер |
| `track1_person_approx.txt` | демонстрационный приближённый 3D-экспорт |

`track1_person_approx.txt` не является полноценным конкурсным 3D-решением. Официальная метрика Track 1 — 3D HOTA.

## Диагностика

```bat
python -m pytest -q
python -m mtmct inspect-scene --scene-dir data/MTMC_Tracking_2026/train/Warehouse_000 --out runs/inspect_warehouse_000
python -m mtmct.audit_gt --gt data/MTMC_Tracking_2026/train/Warehouse_000/ground_truth.json --cameras 3 15 --max-frames 900
python -m mtmct.evaluate --pred runs/warehouse_000_cam3_15/global_tracks_2d.csv --gt data/MTMC_Tracking_2026/train/Warehouse_000/ground_truth.json --cameras 3 15 --max-frames 900 --stride 1 --out runs/warehouse_000_cam3_15/metrics.json
python -m mtmct.validate_geometry --scene-dir data/MTMC_Tracking_2026/train/Warehouse_000 --max-frames 100
```

Чтобы скачать только GT и калибровку без тяжёлых видео:

```bat
python -m mtmct.download --split val --scene Warehouse_022 --ground-truth --no-videos
```

`evaluate` считает версионированный custom Global 2D IDF1 по bbox внутри
`(камера, кадр)`. Это внутренняя метрика разработки, не официальный NVIDIA 3D HOTA.

Существующий прогон на 900 кадрах по новому протоколу
`global-2d-idf1-v1` даёт Global IDF1 `0.397140`, IDP `0.851874`, IDR
`0.258925` и Pairwise F1 `0.885740`. Старое значение diagnostic IDF1
`0.3759` вычислялось другим протоколом и напрямую не сравнивается.

## Структура кода

| Модуль | Назначение |
|---|---|
| `mtmct/__main__.py` | единый запуск всего конвейера |
| `mtmct/tracking.py` | YOLO, ByteTrack, OSNet и локальные треки |
| `mtmct/association.py` | строгая связь Local ID в Global ID |
| `mtmct/export.py` | CSV и приближённый 3D-экспорт |
| `mtmct/visualize.py` | итоговый MP4 mosaic |
| `mtmct/audit_gt.py` | проверка пересечения камер по train GT |
| `mtmct/evaluate.py` | приблизительная 2D-оценка |

Данные NVIDIA: https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces
