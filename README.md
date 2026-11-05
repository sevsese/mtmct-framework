# MTMCT Framework

Запускаемый baseline межкамерного трекинга людей: YOLO26l + ByteTrack + OSNet x1.0.

Командная архитектура и распределение задач описаны в `docs/ARCHITECTURE.md` и `docs/ROADMAP.md`. Правила веток и Pull Request находятся в `CONTRIBUTING.md`.

## Быстрый запуск

Из PowerShell (Python 3.11, NVIDIA GPU):

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python.exe -m pip install -c constraints-windows.txt -e ".[dev]"
```

OSNet требует отдельной установки Torchreid; команды и проверенные результаты
приведены в [отчёте Windows](docs/WINDOWS_VALIDATION.md). Завершите установку
перед запуском `.venv\Scripts\python.exe -m mtmct`.

Для разработки:

```bat
python -m pip install -e ".[dev]"
python -m pytest -q
```

Или запусти `run_demo.cmd`: он использует `.venv` рядом с проектом и передаёт параметры (например, `run_demo.cmd --device cpu`). Одна команда скачивает недостающие видео, выполняет локальный трекинг, связывает Local ID в Global ID, создаёт CSV и `mosaic.mp4`.

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

## Результаты

В каталоге запуска создаются:

| Файл | Содержимое |
|---|---|
| `observations.csv` | боксы и Local ID по кадрам |
| `local_tracklets.json` | локальные треки и усреднённые OSNet-признаки |
| `global_ids.json` | `(camera_id, local_id) -> global_id` |
| `global_tracks_2d.csv` | обнаружения с Global ID |
| `performance.json` | FPS и пиковая CUDA-память |
| `mosaic.mp4` | синхронная визуализация камер |
| `track1_person_approx.txt` | демонстрационный приближённый 3D-экспорт |

`track1_person_approx.txt` не является полноценным конкурсным 3D-решением. Официальная метрика Track 1 — 3D HOTA.

## Диагностика

```bat
python -m pytest -q
python -m mtmct.audit_gt --gt data/MTMC_Tracking_2026/train/Warehouse_000/ground_truth.json --cameras 3 15 --max-frames 900
python -m mtmct.evaluate --pred runs/warehouse_000_cam3_15/global_tracks_2d.csv --gt data/MTMC_Tracking_2026/train/Warehouse_000/ground_truth.json --cameras 3 15 --max-frames 900 --stride 1
python -m mtmct.validate_geometry --scene-dir data/MTMC_Tracking_2026/train/Warehouse_000 --max-frames 100
```

`evaluate` считает только приблизительную 2D IDF1 для разработки, не официальный 3D HOTA.

Проверенный прогон на 900 кадрах улучшил диагностическую IDF1 с `0.3569` до `0.3759`, Pairwise F1 с `0.7692` до `0.8000`, а медианную ошибку геометрии камер 3/15 — примерно с `239/228 м` до `0.19/0.13 м`.

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
