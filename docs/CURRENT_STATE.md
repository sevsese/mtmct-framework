# Текущее состояние MTMCT Framework

Аудит выполнен 5 ноября 2026 года в локальной ветке `Sevsese-Branch` на
коммите `6a5707e38403278fea43dc2ba64bbaffb10818da`.

## Рабочий pipeline

Основная команда — `python -m mtmct`. Она последовательно:

1. проверяет наличие калибровки и выбранных MP4, при необходимости скачивает их;
2. запускает Ultralytics YOLO `yolo26l.pt` с `bytetrack.yaml` отдельно для каждой камеры;
3. извлекает OSNet `osnet_x1_0` признаки для кропов людей и усредняет нормализованные признаки по локальному треку;
4. сохраняет наблюдения и локальные tracklet;
5. связывает tracklet жадным графовым объединением с complete-link проверкой физических ограничений;
6. экспортирует Global ID, 2D CSV, приближённый 3D TXT и видео mosaic.

Доступен лёгкий цветовой ReID только для проверки pipeline. Это не модель качества ReID.

## Входы и выходы

Входы:

- `data/MTMC_Tracking_2026/<split>/<scene>/videos/Camera_XXXX.mp4`;
- `calibration.json` для проекции нижней точки bbox на плоскость `(X,Y)`;
- `ground_truth.json` используется только диагностикой и evaluation, не inference.

Артефакты запуска:

| Файл | Фактическое содержимое |
|---|---|
| `observations.csv` | камера, кадр, Local ID, bbox, confidence, опциональные координаты пола |
| `local_tracklets.json` | границы трека, средний embedding, число samples, позиции |
| `performance.json` | время, FPS и CUDA tensor memory по камерам |
| `global_ids.json` | `(camera_id, local_id) -> global_id` |
| `global_tracks_2d.csv` | наблюдения с Global ID |
| `track1_person_approx.txt` | демонстрационный 3D-подобный экспорт, не официальный Track 1 output |
| `mosaic.mp4` | визуализация предсказаний |

Manifest с хэшами входов, весов, конфигурацией и git SHA отсутствует. Поэтому
существующие `runs/` нельзя считать полностью воспроизводимым кэшем.

## CLI и совместимость

- `python -m mtmct` — полный pipeline; прежний интерфейс необходимо сохранить.
- `python -m mtmct.tracking` — tracking/ReID.
- `python -m mtmct.link` — association/export без видео и GPU.
- `python -m mtmct.evaluate` — приблизительная Global 2D IDF1 диагностика.
- `python -m mtmct.audit_gt` — число людей и общих GT ID для выбранных камер.
- `python -m mtmct.synthetic` — маленький синтетический smoke.

Ограничение: `python -m mtmct --link-only` всё ещё проверяет наличие видео и
всегда запускает visualizer. Только отдельная команда `mtmct.link` сейчас
является честным offline link-only.

## Association и метрики

`mtmct.association.associate` перебирает все пары tracklet, применяет порог
cosine similarity, временные и опциональные геометрические ограничения,
сортирует допустимые рёбра по стоимости и объединяет компоненты. Перед каждым
merge проверяется совместимость всех пар компонентов. Одновременные разные
локальные треки одной камеры объединять запрещено.

`mtmct.evaluate.eval_2d` сначала сопоставляет bbox по IoU внутри пары
`(frame, camera)`, затем вычисляет диагностические IDTP/IDFP/IDFN. Протокол не
версионирован и не покрыт oracle-тестами на false merge/split, поэтому результат
нельзя называть официальной NVIDIA 3D HOTA или окончательной Global 2D IDF1.

## Проверенное состояние baseline

Локально доступны GT, калибровка и MP4 камер 0, 1, 3 и 15 сцены
`Warehouse_000`. Существующий run `runs/improved_cam3_15` содержит по 900
обработанных кадров камер 3 и 15, 70 local tracklet и 8216 предсказаний.

Повторная оценка сохранённого CSV на локальном GT, без повторного inference:

```text
diagnostic_IDF1 = 0.3759
IDTP / IDFP / IDFN = 6999 / 1217 / 22020
detection precision / recall / F1 = 0.9817 / 0.2780 / 0.4332
GT / prediction observations = 29019 / 8216
```

Сохранённая производительность: камера 3 — 20.60 FPS и 0.25 GiB CUDA tensor
memory; камера 15 — 26.71 FPS и 0.30 GiB. Эти числа реальны для существующего
артефакта, но его исходная конфигурация и git SHA не сохранены, поэтому
сравнения A/B с ним требуют сначала создать воспроизводимый baseline.

GT подтверждает за первые 900 кадров: камера 3 видит 36 людей, камера 15 — 33,
общих ID — 33.

## Тесты

`python -m pytest -q` на Python 3.13.18: **12 passed**. Тесты покрывают
геометрию, основные ограничения association, NVIDIA-совместимые имена полей,
синтетический экспорт и mocked двухкамерный pipeline. Mocked pipeline не является
проверкой YOLO, OSNet или качества на реальных данных.

## Соответствие плану 2.0

| Возможность | Состояние |
|---|---|
| Рабочий detection/tracking/ReID/export pipeline | существует |
| Повторная association по сохранённым tracklet | частично: отдельная команда работает, общий `--link-only` требует доработки |
| NVIDIA parser и статистика выбранных камер | частично |
| Версионированный Global 2D IDF1 с oracle-тестами | требует доработки |
| Manifest и строгая инвалидация кэша | отсутствует |
| Oracle Tracklets и зафиксированный baseline | отсутствует |
| Quality-aware ReID, refinement, memory, correction | отсутствуют |
| Complete-link hard constraints | частично существуют в baseline matcher |
| Camera offsets/topology/CITIC/WILDTRACK | отсутствуют |
| Опциональная геометрия | существует, выключена по умолчанию в общем CLI |

## Технический долг и непроверенное

- Нет конфигурации запуска рядом с артефактами и журнала экспериментов.
- Tracklet без embedding отбрасывается вместо сохранения singleton.
- FPS association фиксирован как 30 и не связан с FPS каждого видео.
- `evaluate` не сохраняет metrics JSON и не имеет стабильной версии протокола.
- Holdout `Warehouse_022`, CITIC и WILDTRACK локально не проверялись.
- Полный GPU baseline в рамках этого аудита не перезапускался; оценён уже существующий CSV.

