# MTMCT: единый интерфейс NVIDIA / CITIC / SCOUT

## Что изменено

1. `mtmct/datasets.py` — общий `MTMCTDataset` и реализации `NvidiaDataset`, `CITICDataset`, `SCOUTDataset`.
2. `mtmct/frame_sources.py` — чтение MP4/AVI или кадров `image_N.jpg` без перекодирования.
3. `mtmct/tracking.py` — теперь принимает `camera_sources` и может работать вообще без `calibration.json`. Сохраняет старый API.
4. `mtmct/visualize.py` — отображает произвольное число камер сеткой, в том числе 2×2.
5. `mtmct/dataset_evaluate.py` — общая **пользовательская** метрика Global 2D IDF1 для наборов с настоящими bbox и глобальными ID.
6. `mtmct/cache.py` — кэширует происхождение JPG-папок по имени, размеру и времени изменения файлов; не читает содержимое всех JPEG. Папки не считаются полностью криптографически проверенными.
7. `run_dataset.py` — запуск одного датасета кнопкой ▶ в VS Code.
8. `run_external_validation.py` — проверка готового победителя ночной оптимизации на новых датасетах.
9. `run_nightly_optimize.py` — оставлен прежний подбор на независимых сценах NVIDIA (все исходные файлы оптимизатора включены в комплект).

## Что нужно установить

Используй прежнее окружение `neuro_clean`. Для полного запуска нужны существующие зависимости проекта (PyTorch с CUDA, Ultralytics, Torchreid, OpenCV и т.д.). Для оценки NVIDIA установи `ijson` (`python -m pip install ijson`). `scipy` нужен для метрик. Файлы `run_dataset.py` и `run_external_validation.py` запускаются кнопкой ▶; терминал не обязателен.

## Директории данных

```text
data/
  MTMC_Tracking_2026/
    train/
      Warehouse_000/
        videos/Camera_0000.mp4
        videos/Camera_0001.mp4
        videos/Camera_0003.mp4
        videos/Camera_0015.mp4
        ground_truth.json
        calibration.json          # необязателен при GEOMETRY=False
    val/Warehouse_020/...

  CITIC-MTMC/
    capture1/
      HallWide.mp4                 # имя может отличаться
      HallSeg.mp4
      PF.mp4
      PC.mp4
      Showroom.mp4
    capture1_gt.mot.txt           # файл из официального GitHub
    capture2/...
    capture2_gt.mot.txt
    capture3/...
    capture3_gt.mot.txt
    map_config/Node*.json         # необязательно для режима без геометрии

  SCOUT/
    images/
      sequence_1/
        cam_0/image_0.jpg
        cam_0/image_1.jpg
        cam_1/image_0.jpg
        cam_1/image_1.jpg
        ...
    annotations/
      mot_format/
        sequence_1/
          cam_0.txt
          cam_1.txt
          ...
    timestamps/timestamps_sequence_1.json    # сохранён как метаданные
    calibrations/cam_0.json                  # необязательно
```

**CITIC:** видеозаписи скачиваются отдельно с официального Zenodo: https://zenodo.org/records/17512043; разметка и метаданные — https://github.com/GTEC-UDC/citic-mtmc-dataset . Если имена файлов отличаются от `HallWide`, `HallSeg`, `PF`, `PC`, `Showroom`, укажи в `run_dataset.py` словарь `CITIC_VIDEO_MAP`, например `{0: 'capture1/my_wide_file.mp4', 2: 'capture1/other_video.mp4'}`. Камеры обозначаются кодами `0=HallWide`, `1=HallSeg`, `2=PF`, `3=PC`, `4=Showroom`. Если видео физически в других каталогах — подставь их относительные пути от `data/CITIC-MTMC`.

**SCOUT:** скачай необходимые данные через https://scout.epfl.ch/download ; полный набор занимает много места. Для первых проверок достаточно последовательности 1 и камер из первой восьмёрки с ручными метками, плюс MOT-аннотации. Официальный toolkit: https://github.com/cvlab-epfl/scout_toolkit . Скрипт не загружает видео с сайта автоматически и не пытается скачивать огромный набор без твоего решения.

## Запуск одного набора через ▶

Открой `run_dataset.py`, в начале поменяй:

```python
DATASET = 'scout'    # nvidia | citic | scout
SCENE = 'sequence_1'
CAMERAS = (0, 1, 2, 3)
MAX_FRAMES = 900
REID_MODEL = 'osnet_ibn_x1_0'
RUN_NAME = 'my_scout_test'
EVALUATE = True
```

Нажми **Run Python File**. Результат: `runs/my_scout_test/scout/sequence_1/`.

Для **CITIC** установи `DATASET='citic'`, `SCENE='capture1'`, `CAMERAS=None`, `EVALUATE=False`. Видеозаписи будут обработаны, но **метрика 2D IDF1 не будет сфабрикована**.

Для **NVIDIA**: `DATASET='nvidia'`, `SCENE='Warehouse_000'`, `CAMERAS=(0,1,3,15)`, `EVALUATE=True`, `SPLIT='train'`.

Не включай `GEOMETRY=True` для CITIC/SCOUT: их файлы калибровки имеют другие форматы и не являются матрицами NVIDIA `cameraMatrix`. Для NVIDIA можно включить геометрию только после проверки калибровки.

## Проверка лучшей конфигурации после ночного подбора

Сначала запусти `run_nightly_optimize.py` на NVIDIA: он сохранит

`runs/overnight_v1/best_train_config.json`.

После того как ты **зафиксировал** победителя (не меняя его в зависимости от SCOUT!), открой `run_external_validation.py`, настрой:

```python
BEST_CONFIG = ROOT / 'runs' / 'overnight_v1' / 'best_train_config.json'
DO_SCOUT = True
DO_CITIC = True
SCOUT_CAMERAS = (0, 1, 2, 3)
CITIC_CAMERAS = None
```

Нажми ▶. Он воспроизведёт выбранные на NVIDIA параметры YOLO/ByteTrack, модель OSNet и параметры association, memory и repair. **Геометрия NVIDIA не переносится на чужую систему координат автоматически**; для CITIC без проверенного выравнивания времени временные штрафы также выключены. Это означает, что внешняя проверка не полностью идентична NVIDIA-конфигурации, и результат нельзя трактовать как строгое сравнение неизменного pipeline с той же геометрией.

Папки результатов: `runs/external_holdout_v1/scout/sequence_1/` и `runs/external_holdout_v1/citic/capture1/`.

## Чего программа НЕ делает и почему

- CITIC `capture*_gt.mot.txt` содержит **траектории мира**, а bbox в GT установлены в `-1`. Поэтому пользовательская 2D-метрика на CITIC **не вычисляется**. В проекте уже есть `mtmct/citic.py` для отдельного world-position протокола после согласования координат и timestamp и генерации двух CSV (`timestamp_s, global_id, x, y` и `timestamp_s, person_id, x, y`). Нельзя сравнивать значения Global IDF1 этих разных протоколов напрямую.
- Для SCOUT первая выпущенная версия включает ручную разметку первых восьми камер sequence_1, для остальных есть шумные псевдометки. По умолчанию `ALLOW_SCOUT_PSEUDO_LABELS=False`; используем только соответствующие камеры. Флаг True разрешает экспериментальные оценки на псевдометках, но не делает их надёжной GT.
- При отсутствии калибровки `world_x`/`world_y` пустые; это штатная ситуация. Специальные пространственно-подтверждённые исправления окклюзии будут недоступны.
- В SCOUT лежат синхронные JPEG-кадры, а не MP4; они используются непосредственно. Реальные `timestamps_sequence_*.json` доступны в адаптере, но алгоритм сопоставления пока использует FPS и номера кадров; для произвольных асинхронных камер нужна будущая реализация синхронизации по timestamp.
- Ни NVIDIA, ни SCOUT bbox-IDF1 здесь не являются официальным 3D HOTA.
- Полный GPU-run CITIC/SCOUT **не проверен на реальных файлах**, так как они не были приложены к запросу; проверены синтетические фикстуры и тестовый проход через локальный tracking/link pipeline.

## Проведённые тесты

`python -m pytest -q tests/test_datasets_unified.py` — 6 успешных тестов: изображения, мировая GT CITIC, MOT GT SCOUT, запрет псевдометок, отсутствие калибровки и синтетический полный tracking/link.

Общий набор текущей среды: **54 успешно выполненных теста, 9 падают на `ModuleNotFoundError: ijson`**; это недостающая зависимость, а не ошибка алгоритма. На Windows в окружении с установленным `ijson` перепроверь полный `python -m pytest -q`.

## Официальные источники

- NVIDIA: https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces
- CITIC: https://github.com/GTEC-UDC/citic-mtmc-dataset и https://zenodo.org/records/17512043
- SCOUT: https://scout.epfl.ch/download и https://github.com/cvlab-epfl/scout_toolkit
