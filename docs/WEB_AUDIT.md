# Аудит перед созданием интерфейса

Исходная ветка: codex/adaptive-identity-memory, 89d6813. Алгоритм и его файлы
не перемещаются и не меняются для веб-интерфейса.

- Главная команда: `python -m mtmct`. Фактические defaults CLI: YOLO26l,
  imgsz=1920, confidence=0.10, OSNet x1.0, sample_every=5, baseline,
  min_similarity=0.80, max_cost=0.125, geometry/memory/refinement выключены.
  README местами устарел (указывает 1536); интерфейс использует фактический CLI.
- `tracking.run` принимает произвольный словарь camera_sources, видео OpenCV,
  набор camera IDs, путь output и параметры. Новые YOLO/ByteTrack на камеру.
  `link.run` читает кэш, проверяет manifest и экспортирует ID. Эти функции
  вызываются отдельным Python-процессом, чтобы GPU не занимал процесс HTTP.
- `python -m mtmct` сам выбирает NVIDIA layout; для пользовательских видео
  нужен тонкий worker с camera_sources, а не копия алгоритма.
- observations.csv: camera_id, frame_id, local_id, x1,y1,x2,y2, confidence,
  world_x/world_y. global_tracks_2d.csv добавляет global_id. Координаты в пикселях.
- global_ids.json: массив camera_id/local_id/global_id. local_tracklets.json:
  массив треков с диапазоном кадров, embedding, samples и иногда fps/start_s.
  association_edges.json ссылается индексами i/j на треклеты; при наличии
  refined_tracklets.json следует использовать его. Рёбра не всегда приняты:
  старый baseline сохранял кандидатов без decision.
- performance.json: массив camera_id, processed_frames, seconds, fps,
  peak_cuda_tensor_gib. Это скорость обработки, не FPS видеозаписи.
- manifest.json: schema_version=mtmct-cache-v1, git_sha, scene, cameras,
  inputs с SHA256, models, feature_dimension, tracking_config. config.yaml
  в текущем проекте содержит JSON. metrics.json — необязательный результат
  оценщика; global-2d-idf1-v1 не является официальным NVIDIA 3D HOTA.
- mosaic.mp4 и track1_person_approx.txt существуют в обычных прогонах.
  3D TXT — приближённый экспорт, не источник рамок интерфейса.

На диске есть настоящие видео Warehouse_000 камер 3/15 и результаты прогонов.
Они используются для интеграционной проверки; демонстрационные числа не нужны.
Готовые данные открываются без нейросетей. Приоритет у global_tracks_2d.csv:
он уже учитывает исключённые дубли, поэтому нельзя добавлять к нему все строки
observations.csv. Если экспорта нет, mapping можно применить к observations,
не придумывая новых global ID. Неизвестные поля остаются null.

Backend индексирует наблюдения в SQLite и отдаёт отфильтрованные страницы.
Исходные результаты остаются неизменными. Видео обслуживается локально с HTTP
Range. Процент прогресса не выдумывается: показываются состояние, этап и лог.
Без калибровки план помещения только схематический. Поддержка неизвестных
форматов CITIC/SCOUT не обещается: текущий адаптер — native MTMCT artifacts.
