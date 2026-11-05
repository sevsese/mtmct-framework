# Архитектура MTMCT

Проект разделён по работающим этапам, а не по абстракциям «на будущее».

| Направление | Код | Результат этапа |
|---|---|---|
| Detection & Tracking | `mtmct/tracking.py` | `observations.csv` и локальные ID |
| ReID & Features | `mtmct/reid.py`, `mtmct/weights.py` | embedding каждого tracklet |
| Data & Geometry | `mtmct/download.py`, `mtmct/geometry.py` | общие координаты пола `(X,Y)` |
| Fusion & Evaluation | `mtmct/association.py`, `mtmct/evaluate.py`, `mtmct/export.py` | Global ID, метрики и submission |

`python -m mtmct` связывает этапы. Дорогой результат tracking/ReID сохраняется в `local_tracklets.json`; глобальное сопоставление можно повторять через `--link-only`.
Новый tracking-run также сохраняет `manifest.json`. Link-only проверяет detector,
размер признаков и доступные исходные файлы, но допускает offline-работу, когда
исходные MP4 намеренно отсутствуют.

## Контракты между этапами

### observations.csv

Одна строка на обнаружение: `camera_id`, `frame_id`, `local_id`, 2D-бокс, confidence и опциональные `world_x`, `world_y`.

### local_tracklets.json

Один объект на локальный трек: камера, Local ID, первый/последний кадр, позиции на полу, нормализованный ReID embedding и число использованных кропов.

### global_ids.json

Список соответствий `(camera_id, local_id) -> global_id`.

### manifest.json

Версия схемы, git SHA, сцена, камеры, tracking-конфигурация, модели, размер
embedding, размеры и SHA-256 входов. Параметры association не входят в tracking
cache key, поэтому их можно менять без повторного GPU inference.

Эти три формата — стабильная граница командной работы. Новые интерфейсы добавляются только при появлении второй реальной реализации компонента.

## Пространство и время

Калиброванные сцены используют геометрическую карту пола `(X,Y)`. Для некалиброванных или непересекающихся камер сначала достаточно топологии переходов: пары камер, зоны выхода/входа и распределения времени перехода. Детальная 3D-сцена остаётся отдельным бонусным этапом.
