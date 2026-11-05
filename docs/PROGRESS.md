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

Коммит: `docs: audit current pipeline` (этот коммит).

