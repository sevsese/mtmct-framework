# Прогресс MTMCT Framework 2.0

## Этап 00 — аудит и фиксация исходного состояния

Статус: **PASS**

Проверяемые результаты:

1. Фактический pipeline, CLI, модели, артефакты и ограничения записаны в `docs/CURRENT_STATE.md`.
2. Старые CPU-тесты проходят без GPU и датасета.
3. Существующий реальный CSV повторно оценён на локальном GT без выдачи результата за официальный benchmark.
4. Следующие этапы и gate отражены в `docs/ROADMAP.md`.

Изменённые файлы: `docs/CURRENT_STATE.md`, `docs/ROADMAP.md`,
`docs/PROGRESS.md`; удалён устаревший аппаратно-зависимый
`run_demo_3070ti.cmd`, универсальный `run_demo.cmd` сохранён.

| Проверка | Результат |
|---|---|
| `python -m pytest -q` | PASS: 12 passed in 9.81 s |
| `python -m mtmct --help` | PASS |
| `python -m mtmct.link --help` | PASS |
| evaluation `runs/improved_cam3_15`, камеры 3/15, 900 кадров | PASS: diagnostic IDF1 0.3759 |
| GT audit, камеры 3/15, 900 кадров | PASS: 36/33 людей, 33 общих ID |
| новый полный GPU inference | NOT RUN: существующий run проверен, но его конфигурация не имеет manifest |
| Warehouse_022 | NOT RUN: данные отсутствуют локально |

Артефакты: использованы существующие файлы из `runs/improved_cam3_15`; новые
объёмные артефакты не создавались и в Git не добавлялись.

Ограничения: текущая метрика приблизительная и неверсионированная; provenance
существующего run неполный; общий `--link-only` ещё зависит от видео.

Следующий этап: **01 — NVIDIA adapter, валидатор GT и `inspect-scene`**.

Коммит: `b8d9e71 docs: audit current pipeline`.

## Этап 01 — NVIDIA adapter и инспектор сцены

Статус: **PASS**

Проверяемые результаты:

1. GT читается потоково, строковые номера кадров преобразуются явно, Person и bbox валидируются.
2. Формула времени поддерживает FPS, frame offset, time offset и точные PTS.
3. `python -m mtmct inspect-scene` создаёт `scene_report.json` и `camera_pair_report.csv` без публикации GT identity.
4. GT+калибровка скачиваются отдельно через `snapshot_download(allow_patterns=...)`, без видео.
5. Инспектор проверен на настоящих `Warehouse_000` и `Warehouse_022`.

Изменённые файлы: `mtmct/nvidia.py`, `mtmct/download.py`, `mtmct/__main__.py`,
`tests/test_scene_adapter.py`, `README.md`, `docs/ROADMAP.md`, `docs/PROGRESS.md`.

Результаты:

| Проверка | Результат |
|---|---|
| targeted scene-adapter tests | PASS: 5 passed |
| `Warehouse_000` | PASS: 9000 кадров, 60 людей, 19 камер; лучшая пара 3/15 |
| `Warehouse_022` | PASS: 9270 кадров, 24 человека, 4 камеры; лучшая пара 0/1 |
| GT validation | 66 961 / 564 вырожденных bbox соответственно; явно отражены и пропущены |
| video validation | Warehouse_000: локально 4/19 MP4; Warehouse_022: видео не скачивались |

Артефакты: `runs/stage01_warehouse000/{scene_report.json,camera_pair_report.csv}`
и аналогичный каталог `stage01_warehouse022`. Они локальные и не добавляются в Git.

Ограничения: точные PTS и camera offsets в NVIDIA GT отсутствуют, поэтому отчёты
используют явно записанное допущение 30 FPS и нулевой offset. GT ID применяются
только внутри analysis adapter для подсчёта агрегатов и не импортируются inference-кодом.

Следующий этап: **02 — версионированный Global 2D IDF1 и oracle-тесты**.

Коммит: `543ca7e feat: inspect NVIDIA scenes`.

## Этап 02 — Global 2D IDF1 и oracle-проверки

Статус: **PASS**

Проверяемые результаты:

1. Bbox сопоставляются только внутри одинаковых `(camera, frame)` при IoU ≥ 0.5.
2. GT ID и predicted Global ID назначаются глобально один-к-одному после bbox matching.
3. Сохраняются IDTP/IDFP/IDFN, IDP/IDR/IDF1, detection, per-camera IDF1 и cross-camera Pairwise P/R/F1.
4. Протокол имеет версию `global-2d-idf1-v1` и явно не называется официальной 3D HOTA.
5. Oracle-тесты подтверждают perfect=1.0 и падение метрик при lost/false detection, ID switch, false merge и false split.

Изменённые файлы: `mtmct/evaluate.py`, `mtmct/nvidia.py`,
`tests/test_evaluation_oracles.py`, `README.md`, `docs/ROADMAP.md`, `docs/PROGRESS.md`.

| Проверка | Результат |
|---|---|
| evaluation oracle tests | PASS: 5 passed |
| полный pytest | PASS: 22 passed in 7.90 s |
| real-data evaluation, Warehouse_000 cameras 3/15, frames 0..899 | PASS |

Реальный результат существующего `runs/improved_cam3_15`:

| Метрика | Значение |
|---|---:|
| Global IDF1 | 0.397140 |
| Global IDP / IDR | 0.851874 / 0.258925 |
| IDTP / IDFP / IDFN | 6999 / 1217 / 20032 |
| Detection P / R / F1 | 0.981743 / 0.298398 / 0.457684 |
| Pairwise association P / R / F1 | 0.999718 / 0.795092 / 0.885740 |

Артефакт: `runs/stage02_improved_cam3_15/metrics.json` (локальный, не в Git).

Ограничения: per-camera HOTA не вычисляется без TrackEval и записывается как
`null`; v1 исключает вырожденные GT bbox, поэтому его IDF1 `0.397140` нельзя
напрямую сравнивать со старым diagnostic IDF1 `0.3759`. По умолчанию evaluator
теперь проверяет каждый кадр (`stride=1`), старый sampling доступен явно.

Следующий этап: **03 — manifest, независимый кэш и честный offline link-only**.

Коммит: `965bcde test: validate global identity metrics`.

## Этап 03 — manifest, кэш и offline link-only

Статус: **PASS**

Проверяемые результаты:

1. Новый GPU tracking-run сохраняет `manifest.json` с git SHA, моделями, конфигурацией, feature dimension и SHA-256 входов.
2. Link-only не проверяет MP4, не импортирует CUDA-модели и не запускает visualizer.
3. Изменённый вход, detector или размер embedding отклоняют устаревший кэш.
4. Tracklet без ReID сохраняется и получает отдельный Global ID вместо удаления.
5. Несколько matcher-запусков с разными порогами используют неизменный feature cache.

Изменённые файлы: `mtmct/cache.py`, `mtmct/link.py`, `mtmct/tracking.py`,
`mtmct/__main__.py`, `tests/test_cache_pipeline.py`, `README.md`,
`docs/ARCHITECTURE.md`, `docs/ROADMAP.md`, `docs/PROGRESS.md`.

| Проверка | Результат |
|---|---|
| cache/link-only tests | PASS: 3 passed |
| полный pytest | PASS: 25 passed in 5.39 s |
| real cached tracklets, threshold 0.80 | PASS: 62 candidate edges, 17 cross-camera Global ID |
| те же tracklets, threshold 0.85 | PASS: 35 candidate edges, 16 cross-camera Global ID |
| SHA-256 `local_tracklets.json` между запусками | неизменён |

Артефакт: `runs/stage03_link_only_real/` содержит повторно построенные
`global_ids.json`, `association_edges.json`, 2D CSV и TXT. MP4 и GPU не использовались.

Ограничения: старые runs без manifest продолжают работать с явным warning, но
их provenance нельзя проверить. Отсутствующий исходный файл допустим для offline
link-only; если файл доступен, его SHA-256 обязательно сверяется.

Следующий этап: **04 — воспроизводимый GPU baseline и Oracle Tracklets**.

Коммит: `feat: decouple association and cache` (этот коммит).

