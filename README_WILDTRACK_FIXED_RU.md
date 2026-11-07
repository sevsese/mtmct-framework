# MTMCT — исправленный запуск WILDTRACK (v3)

Это **дополнение** к существующему проекту MTMCT. Старые файлы `mtmct/tracking.py`,
`run_wildtrack.py` и `bytetrack.yaml` не изменяются. Новые файлы:

- `run_wildtrack_fixed.py` — полный запуск YOLO → OSNet → ReID-aware local tracker → graph association → оценка → видео.
- `run_wildtrack_fixed.bat` — один клик / запуск в Windows.
- `mtmct/tracking_reid.py` — новый локальный трекер с ReID и проверкой перемещения.
- `baseline_wildtrack_v2_metrics.json` — **реальные** метрики предыдущего архива, используются только для сравнения, не для принятия решений о треках.
- `tests/test_reid_motion.py` и `tests/test_fixed_runner_integration.py` — тесты (без настоящей GPU).

## Установка

1. Распаковать архив **поверх корневой папки существующего проекта** `MTMCT_GPU_3070Ti`, сохраняя вложенный путь `mtmct/tracking_reid.py`. Не создавать лишнюю папку `mtmct` внутри `mtmct`.
2. Убедиться, что имеются `data/WILDTRACK/Image_subsets/C1 ... C4` и `data/WILDTRACK/annotations_positions` (или `annotation_positions`).
3. Запустить в терминале VS Code (`PowerShell` или `cmd`):

```powershell
python run_wildtrack_fixed.py
```

Или в PowerShell:

```powershell
.\run_wildtrack_fixed.bat
```

`.bat` пытается использовать `%USERPROFILE%\.conda\envs\neuro_clean\python.exe`; иначе берёт `python` из PATH.

## Дополнительные команды

Только проверить данные (без загрузки моделей):

```powershell
python run_wildtrack_fixed.py --check-only
```

Повторить запуск в ту же папку, перезаписав известные артефакты:

```powershell
python run_wildtrack_fixed.py --overwrite
```

Изменить количество размеченных моментов, качество / набор камер:

```powershell
python run_wildtrack_fixed.py --max-frames 60 --imgsz 1920 --cameras 0 1 2 3 --out runs/wildtrack_4cam_60f_reid_v3
```

Меньше видеопамяти при извлечении ReID:

```powershell
python run_wildtrack_fixed.py --reid-batch 12
```

Если датасет расположен иначе:

```powershell
python run_wildtrack_fixed.py --root "D:\datasets\WILDTRACK"
```

## Что именно изменено относительно v2

1. Сохраняются **все сырые YOLO bbox**, включая отфильтрованные локальным трекером, в `raw_detections.csv`.
2. Локальное сопоставление учитывает OSNet и перемещение человека, а не один лишь ByteTrack IoU.
3. Трек теряет право на присвоение своего старого ID через `max_missed` пропущенных *обработанных размеченных моментов* (2 FPS, независимо от номера файла `0,5,10,...`).
4. Сильно отличающуюся внешность нельзя молча добавить к старому локальному ID; создаётся новый фрагмент.
5. Для возможной повторной ассоциации сохраняются признаки **каждого выведенного наблюдения** в `observation_embeddings.npz`.
6. Глобальное объединение выполняется через graph matcher со строгим запретом перекрывающихся во времени локальных треков одной камеры, а не простым объединением любых похожих эмбеддингов. Временные ограничения и геометрия на этом запуске выключены.
7. Программа сама считает Global IDF1, Detection Precision/Recall, сырые показатели YOLO и сравнивает с архивным v2 (только при совпадающих кадрах и камерах).
8. `mosaic.mp4` имеет 2 FPS, а не 0.4 FPS: `0,5,10` — имена кадров, **не пропуски**.

## Результаты

В `runs/wildtrack_4cam_40f_reid_v3`:

- `mosaic.mp4` — 4 камеры со стабильными (насколько удалось) глобальными ID;
- `comparison.json` — сравнение IDF1/Precision/Recall с v2;
- `metrics.json` — метрики нового запуска;
- `raw_detection_metrics.json` — YOLO без трекера, чтобы понять, где возникают пропуски;
- `raw_detections.csv` — детекции + причина, по которой bbox сохранён/отброшен;
- `global_tracks_2d.csv` — новые bbox с глобальным ID;
- `observations.csv`, `local_tracklets.json`, `observation_embeddings.npz` — подробные локальные данные;
- `performance.json` — производительность и количество bbox по камерам.

## Границы улучшения

Этот запуск **не гарантирует** рост IDF1: результат пока не измерен на твоём GPU и настоящем WILDTRACK. Это новый алгоритм, а не набор уже подобранных лучших гиперпараметров.
У исходных детекций (YOLO) может быть низкая полнота или ошибки относительно GT из-за зоны интереса WILDTRACK. Новый ReID-трекер **не восстанавливает боксы людей, которых YOLO не обнаружила**.
Пороговые значения нужны для эксперимента, а не подтверждены подбором на независимых данных.
Оценка в `metrics.json` — кастомный bbox IoU 0.5 Global IDF1 на размеченных моментах, **не официальная метрика WILDTRACK**.

Для сравнения обязательно сохраняй **те же 4 камеры и 40 кадров**, иначе результаты несопоставимы; при несовпадении сравнение отключается автоматически.
