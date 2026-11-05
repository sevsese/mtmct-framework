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

Коммит: `66815ae feat: decouple association and cache`.

## Этап 04 — GPU baseline и Oracle Tracklets

Статус: **PASS**

Проверяемые результаты:

1. Полный YOLO + ByteTrack + OSNet pipeline повторно выполнен на GPU с manifest и замороженной конфигурацией.
2. Повторный baseline 3/15 воспроизвёл прежние IDTP/IDFP/IDFN точно.
3. Выполнен расширенный четырёхкамерный GPU smoke на камерах 0/1/3/15.
4. Oracle Tracklets используют GT только как bbox, перенумеровывают Local ID отдельно по камерам и извлекают настоящий OSNet из видео.
5. Для full и oracle сохранены config, manifest, metrics, errors, association edges и mosaic.

Изменённые файлы: `mtmct/oracle.py`, `mtmct/cache.py`, `mtmct/__main__.py`,
`mtmct/evaluate.py`, `tests/test_oracle_tracklets.py`, `README.md`,
`docs/ROADMAP.md`, `docs/PROGRESS.md`.

Проверки:

| Проверка | Результат |
|---|---|
| oracle leakage unit test | PASS: 1 passed |
| полный pytest | PASS: 26 passed in 8.21 s |
| CUDA math | PASS |
| full baseline, cameras 3/15, 900 frames | PASS |
| expanded full baseline, cameras 0/1/3/15, 300 frames | PASS |
| Oracle Tracklets, cameras 3/15, 900 frames | PASS |

Сравнение на одинаковых камерах 3/15 и кадрах 0..899:

| Режим | Global IDF1 | IDP | IDR | Pairwise F1 | Detection F1 |
|---|---:|---:|---:|---:|---:|
| Full pipeline | 0.397140 | 0.851874 | 0.258925 | 0.885740 | 0.457684 |
| Oracle GT bbox + real OSNet | 0.938891 | 0.951042 | 0.927047 | 0.914875 | 0.987224 |

Full pipeline runtime: camera 3 — 24.39 FPS, camera 15 — 26.01 FPS; peak
CUDA tensor memory 0.25 GiB на каждой. Oracle feature extraction: 97.24/106.02 FPS.

Расширенный smoke 0/1/3/15, 300 кадров: Global IDF1 `0.517268`, Pairwise
F1 `0.825221`, Detection F1 `0.594132`; FPS по камерам 12.62/13.31/12.76/18.85.

Артефакты:

- `runs/v2_baseline_cam3_15/`;
- `runs/v2_baseline_cam0_1_3_15_300/`;
- `runs/v2_oracle_tracklets_cam3_15/`.

Вывод: крупнейший подтверждённый разрыв находится в detection/local tracking
(IDR 0.258925 против 0.927047 у Oracle), а не только в global association.

Ограничения: Holdout `Warehouse_022` не запускался через модели, потому что его
видео не скачивались и гиперпараметры на val не подбираются. Oracle — диагностика,
не честный inference benchmark и не официальный NVIDIA 3D HOTA.

Следующий этап: **05 — диагностика ReID и quality-aware aggregation на неизменном кэше**.

Коммит: `ffc842d test: establish MTMCT baselines`.

## Этап 05 — диагностика и агрегация ReID

Статус: **PASS**

Проверяемые результаты:

1. На tracklet сохраняются максимум 20 лучших OSNet samples с quality, поэтому память ограничена.
2. Реализованы отключаемые `first`, `mean`, `topk` и `prototypes` без повторного GPU inference.
3. Analysis-only labels не попадают в matcher artifacts; формируются Rank-1, AP, precision@recall, пары CSV и cosine PNG.
4. Все политики сравнены на одном feature cache реальных кропов и одинаковом matcher.
5. Новая политика не включена по умолчанию без выигрыша.

Изменённые файлы: `mtmct/features.py`, `mtmct/tracking.py`, `mtmct/oracle.py`,
`tests/test_feature_aggregation.py`, `README.md`, `docs/ROADMAP.md`, `docs/PROGRESS.md`.

Один cache: Oracle GT bbox, real OSNet crops, камеры 3/15, 900 кадров,
matcher `min_similarity=0.80`, `max_cost=0.125`.

Проверки: targeted feature/ReID tests — PASS; полный pytest — **29 passed**.

| Policy | Rank-1 | AP | P@R=0.90 | Global IDF1 | Pairwise F1 |
|---|---:|---:|---:|---:|---:|
| first | 0.439394 | 0.391090 | 0.051195 | 0.635594 | 0.280579 |
| mean (baseline) | **0.727273** | **0.732361** | **0.118110** | **0.938891** | **0.914875** |
| top-k quality | 0.621212 | 0.612352 | 0.065502 | 0.868902 | 0.821834 |
| 3 prototypes | 0.560606 | 0.560521 | 0.073529 | 0.785238 | 0.662979 |

Артефакты: `runs/v2_reid_feature_cache`, `runs/v2_reid_{first,mean,topk,prototypes}`;
каждый содержит `reid_report.json`, `reid_pairs.csv`, `cosine_distributions.png` и MTMCT metrics.

Решение: сохранить mean по всем sampled crops как default. Top-k и prototypes
остаются экспериментальными OFF, потому что на замороженном сравнении ухудшили
Global IDF1. Сохранённые top-20 samples предназначены для дешёвых будущих ablation.

Следующий этап: **06 — Tracklet Refinement с dry-run и provenance**.

Коммит: `2bbc3b6 feat: aggregate quality-aware features`.

## Этап 06 — Tracklet Refinement

Статус: **PASS**

Проверяемые результаты:

1. Dry-run диагностирует appearance jump и same-camera fragment merge без мутации входа.
2. Auto split создаёт новые camera-local ID по диапазонам кадров.
3. Auto merge разрешён только при положительном gap, сильном appearance и отсутствии overlap.
4. Observation CSV переписывается по frame ranges, а provenance сохраняет исходные Local ID.
5. Режим OFF возвращает побайтно эквивалентные структуры и остаётся default.

Изменённые файлы: `mtmct/refinement.py`, `mtmct/tracking.py`, `mtmct/oracle.py`,
`tests/test_refinement.py`, `README.md`, `docs/ROADMAP.md`, `docs/PROGRESS.md`.

Проверки: targeted refinement tests — PASS: 3; полный pytest — **32 passed**.

Real dry-run на `v2_baseline_cam3_15` нашёл 0 split и 2 merge-кандидата:
camera 3 gap 58/similarity 0.925388 и camera 15 gap 5/similarity 0.925298.

| Режим | Tracklets | Global IDF1 | Pairwise F1 |
|---|---:|---:|---:|
| baseline | 70 | **0.397140** | **0.885740** |
| auto merge | 68 | 0.394133 | 0.879680 |

Решение: auto refinement остаётся **OFF**, потому что merge ухудшил обе метрики.
Split на реальном старом cache не оценивался: временные feature frames появились
только в новом формате; unit-инварианты и provenance проверены синтетически.

Артефакты: `runs/v2_refinement_dry_cam3_15`, `runs/v2_refinement_merge_cam3_15`.

Следующий этап: **07 — constrained graph association и полный журнал решений**.

Коммит: `d34e945 feat: refine tracklets`.

## Этап 07 — Constrained Graph Association

Статус: **PASS**

Проверяемые результаты:

1. Существующий matcher сохранён как `baseline`; новый выбирается только `--matcher graph`.
2. Graph рассматривает cross-camera candidates, сохраняет singleton без ReID и проверяет все пары компонентов перед merge.
3. Одновременные разные tracklet одной камеры не попадают в один Global ID.
4. Одновременное появление на разных камерах разрешено без geometry/time.
5. Все пары получают decision, score breakdown и reason в `association_edges.json`.

Изменённые файлы: `mtmct/association.py`, `mtmct/link.py`, `mtmct/__main__.py`,
`tests/test_association_constraints.py`, `README.md`, `docs/ROADMAP.md`, `docs/PROGRESS.md`.

Проверки: targeted association/baseline — 14 passed; полный pytest — **34 passed**.

A/B использует один `v2_baseline_cam3_15` feature cache, камеры 3/15, 900 кадров:

| Matcher | Global IDF1 | Pairwise P | Pairwise F1 | Global groups |
|---|---:|---:|---:|---:|
| baseline | **0.397140** | 0.999718 | **0.885740** | 17 cross-camera |
| constrained graph, appearance-only | 0.351803 | **1.000000** | 0.796779 | 51 total |

Graph принял 19 и отклонил 2396 рёбер. Измеренный pairwise false-merge rate
`1 - precision` равен 0, но recall/IDF1 хуже. Поэтому graph остаётся
экспериментальным **OFF**, baseline — default.

Артефакт: `runs/v2_graph_cam3_15/` с полным decision journal и metrics.

Следующий этап: **08 — offline Global Memory для long-gap association**.

Коммит: `529e9d6 feat: constrain global graph association`.

## Этап 08 — Offline Global Memory

Статус: **PASS**

Проверяемые результаты:

1. Memory хранит до трёх нормализованных прототипов, camera/frame history и confirmed status.
2. Long-gap synthetic восстанавливает один Global ID после отказа временного matcher.
3. Одновременные tracklet одной камеры не объединяются даже при одинаковом appearance.
4. Memory OFF не меняет graph mapping и остаётся default.
5. Memory обновляет только группы с доступными ReID-прототипами; TTL/LRU не добавлены для offline режима.

Изменённые файлы: `mtmct/memory.py`, `mtmct/link.py`, `mtmct/__main__.py`,
`tests/test_memory.py`, `README.md`, `docs/ROADMAP.md`, `docs/PROGRESS.md`.

Проверки: memory unit tests — 2 passed; полный pytest — **36 passed**.

A/B на одном graph cache, камеры 3/15, 900 кадров:

| Режим | Global IDF1 | Pairwise F1 | Groups |
|---|---:|---:|---:|
| graph, memory OFF | 0.351803 | 0.796779 | 51 |
| graph, memory ON | **0.377167** | **0.858433** | 48 |
| baseline reference | **0.397140** | **0.885740** | — |

Memory подтвердил 3 merge и отклонил 2 temporal conflict. Он улучшил graph,
но не превзошёл baseline, поэтому остаётся экспериментальным **OFF**.

Артефакт: `runs/v2_graph_memory_cam3_15/global_memory.json` и metrics.

Следующий этап: **09 — camera time offsets и optional topology**.

Коммит: `e2d5e14 feat: add global memory`.

## Этап 09 — время, topology и CITIC protocol

Статус: **PASS**

Проверяемые результаты:

1. Tracklet использует явные `start_s/end_s` либо `frame/fps + camera_offset`.
2. Unknown topology не запрещает match; forbidden и выход за measured route запрещают.
3. Измеренный допустимый маршрут добавляет soft temporal penalty.
4. Time/topology включаются только флагом и не меняют appearance-only default graph.
5. CITIC world-position evaluator имеет отдельную версию протокола и не называется bbox IDF1.

Изменённые файлы: `mtmct/time_sync.py`, `mtmct/citic.py`,
`mtmct/association.py`, `mtmct/link.py`, `mtmct/__main__.py`,
`tests/test_time_sync.py`, `README.md`, `docs/ROADMAP.md`, `docs/PROGRESS.md`.

Проверки: time/topology/CITIC + constraints — 5 passed; полный pytest — **39 passed**.

Real A/B на одном graph cache, камеры 3/15, 900 кадров:

| Режим | Global IDF1 | Pairwise F1 |
|---|---:|---:|
| appearance-only | 0.351803 | 0.796779 |
| appearance + generic time | 0.351803 | 0.796779 |

Результат не изменился, поэтому time остаётся **OFF**. Измеренную topology для
Warehouse_000 не выдумывали; её нужно получать из данных переходов.

CITIC real-data: **NOT RUN — dataset и `timesyncs.csv` отсутствуют локально**.
Синтетический perfect world-position protocol даёт IDF1 1.0.

Артефакт: `runs/v2_graph_time_cam3_15/metrics.json`.

Следующий этап: **10 — reversible decisions и Identity Correction**.

Коммит: `8036ff1 feat: add optional camera transitions`.

## Этап 10 — Reversible Identity Correction

Статус: **PASS**

Проверяемые результаты:

1. Graph journal использует статусы `provisional`, `confirmed`, `rejected` и сохраняет score breakdown.
2. Validator проверяет все пары объединяемых компонентов, включая appearance и hard constraints.
3. False transitive merge отклоняется перестроением из сохранённых рёбер, без попытки разъединить Union-Find.
4. Исходные observations, Local ID и tracklets не мутируются.
5. Assignment после correction детерминирован и не требует повторного YOLO/ReID.

Изменённые файлы: `mtmct/correction.py`, `mtmct/association.py`,
`mtmct/link.py`, `mtmct/__main__.py`, `tests/test_identity_correction.py`,
`README.md`, `docs/ROADMAP.md`, `docs/PROGRESS.md`.

Проверки: correction + graph tests — 4 passed; полный pytest — **41 passed**.

Real graph journal, камеры 3/15, 900 кадров:

| Режим | Global IDF1 | Pairwise F1 | Conflicts |
|---|---:|---:|---:|
| correction OFF | 0.351803 | 0.796779 | — |
| correction ON | 0.351803 | 0.796779 | 0 |

Все 19 provisional merge подтверждены; 2396 ранее отклонённых рёбер остались
rejected. На этом run correction корректно является no-op и остаётся **OFF**,
поскольку улучшения метрики нет.

Артефакт: `runs/v2_graph_correction_cam3_15/identity_correction.json`.

Следующий этап: **11 — WILDTRACK adapter и portability без обязательной geometry**.

Коммит: `feat: validate and correct identities` (этот коммит).

