# MTMCT Local API

Запуск: `python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000`
из корня репозитория с активированным окружением. Интерактивная схема:
http://127.0.0.1:8000/docs, OpenAPI JSON: `/openapi.json`.
Полная установка и использование: [WEB_INTERFACE.md](../docs/WEB_INTERFACE.md).

## Endpoints

| Метод / путь | Назначение |
|---|---|
| GET `/api/health` | Состояние backend и версия адаптера |
| GET `/api/runs` | Доступные каталоги результатов в runs |
| POST `/api/datasets/import` | Импорт `{relative_path, name}` из runs |
| POST `/api/datasets/upload` | Multipart-загрузка файлов |
| GET `/api/datasets` | Список наборов |
| GET `/api/datasets/{id}` | Сводка и задача обработки |
| POST `/api/datasets/{id}/process` | `{max_frames:900, device:"cuda:0"}` для raw-набора |
| GET `/api/jobs/{id}` | Статус/этап/время/ошибка и последние 12 KB журнала |
| GET `/api/datasets/{id}/cameras` | Camera DTO и локальные video_url |
| GET `/api/datasets/{id}/video/{camera_id}` | Поток видео с HTTP Range, ответ 206 |
| GET `/api/datasets/{id}/tracks` | GlobalTrack DTO, фильтры и пагинация |
| GET `/api/datasets/{id}/tracks/{global_id}` | Трек, локальные связи, рёбра ассоциации |
| GET `/api/datasets/{id}/observations` | Страницы Observation DTO |
| GET `/api/datasets/{id}/performance` | Производительность и сохранённый metrics.json |
| GET `/api/datasets/{id}/manifest` | manifest, config, файлы и предупреждения |
| GET `/api/datasets/{id}/analytics` | Агрегаты по камерам/времени/длительности |
| GET / POST / PUT `/api/datasets/{id}/floorplan` | Сведения / загрузка схемы / метки камер |
| GET `/api/datasets/{id}/floorplan/image` | Изображение схемы |

Tracks: `search`, `camera_id`, `start`, `end`, `sort`, `descending`, `offset`,
`limit` (1–200, по умолчанию 25). Sort: global_id/start_time/end_time/duration/observations_count.
Временной фильтр выбирает пересекающиеся с интервалом треки.
Observations: camera_id/global_id/start/end/frame_id/offset/limit (1–10000).
Ответы страниц: `{total, items}`. Время в секундах, frame_id начинается с 0.

Процент прогресса `null`: worker сообщает реальные этапы tracking/association/completed.
Состояния: queued/running/completed/failed. ID задачи совпадает с ID raw-набора.
Повторный запуск выполняющегося или уже завершённого набора возвращает 409.
Неверные данные — 422; слишком большой файл — 413; отсутствующая сущность — 404.

## Multipart-загрузка

Поля: `name` (название), `mode` (`raw` или `results`), `files` (повторяемое),
`cameras` (JSON-массив). Например, **только пример схемы, не результат алгоритма**:

```json
[{"camera_id":3,"filename":"Camera_0003.mp4","name":"Вход","offset":0},
 {"camera_id":15,"filename":"Camera_0015.mp4","name":"Коридор","offset":0.2}]
```

Camera ID и имена файлов уникальны. `fps` можно передать через API как метаданные;
если приложено видео, его измеренный FPS имеет приоритет. Offset означает добавку
к локальному времени камеры. В режиме results видео необязательны, `cameras=[]`
допустим. Режим raw принимает только перечисленные видео.

## Собственный формат результатов

| Файл | Поля / поведение |
|---|---|
| global_tracks_2d.csv | camera_id,frame_id,local_id,x1,y1,x2,y2,global_id; confidence необязателен |
| observations.csv | Те же поля без global_id; mapping берётся из global_ids.json |
| global_ids.json | Массив `{camera_id,local_id,global_id}` |
| local_tracklets.json | Массив треклетов; camera_id/local_id, fps и frame bounds; тяжёлые embeddings не передаются в UI |
| refined_tracklets.json | При наличии определяет порядок треклетов для association edges |
| association_edges.json | Массив связей i/j и сохранённых оценок; решение не домысливается |
| performance.json | Массив camera_id/processed_frames/seconds/fps/peak_cuda_tensor_gib |
| manifest.json | Объект mtmct-cache-v1: cameras, models, inputs, tracking_config, git_sha |
| config.yaml | JSON-объект фактического конфига проекта |
| metrics.json | Сохранённый результат оценщика; protocol_version, global, detection, pairwise_association |

При наличии global_tracks_2d.csv используется именно он: добавлять к нему все строки
observations.csv нельзя, поскольку экспорт мог удалить дубли. Если есть оба CSV,
исходный observations.csv не увеличивает число наблюдений. Mapping сверяется с экспортом.
Новых Global ID адаптер не назначает. При отсутствии глобальной разметки доступны Local ID.

Пример CSV для проверки парсера (**DEMO**, искусственные координаты):

```csv
camera_id,frame_id,local_id,x1,y1,x2,y2,confidence,global_id
3,0,1,10,20,30,80,0.9,7
15,0,2,40,20,60,80,0.8,7
```

Ему соответствует один глобальный трек с двумя локальными. Без видео/FPS timestamp
остаётся null. Observation DTO: camera_id/frame_id/local_id/global_id/timestamp,
bbox `[x1,y1,x2,y2]`, confidence. Координаты исходные, в пикселях.
Camera DTO содержит доступные fps/duration/width/height/offset и video_url либо null.
GlobalTrack: global_id/start_time/end_time/duration/cameras/local_tracklets/
observations_count/status; status не выводится из эвристик и может быть null.

## Слой адаптации

`adapter.py` — единственное место чтения файлов результатов и построения SQLite-индекса.
`main.py` — REST, реестр, фильтры, очередь и выдача production frontend.
`worker.py` — вызовы штатных tracking/link в изолированных процессах.
Для нового формата добавляется преобразование к тем же Camera/Observation/GlobalTrack
DTO; страницы React не должны читать его CSV/JSON напрямую. Неподтверждённые форматы
сейчас отклоняются.
