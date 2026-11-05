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

Коммит: `feat: inspect NVIDIA scenes` (этот коммит).

