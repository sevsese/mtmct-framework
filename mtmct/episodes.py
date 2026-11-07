"""Split one identity into visits separated by more than the configured gap."""
import csv
import json
from collections import defaultdict
from pathlib import Path


def export_episodes(predictions, camera_fps, offsets, max_gap=60.):
    if max_gap <= 0:
        raise ValueError('Episode gap must be positive')
    source = Path(predictions)
    with source.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        columns, rows = reader.fieldnames, list(reader)
    observations = defaultdict(list)
    for row in rows:
        camera, frame = int(row['camera_id']), int(row['frame_id'])
        timestamp = frame/camera_fps[camera]+offsets.get(camera,0.)
        observations[int(row['global_id'])].append((timestamp,row))
    episodes = []
    for person, samples in sorted(observations.items()):
        previous, episode = None, None
        for timestamp, row in sorted(samples, key=lambda item:item[0]):
            if previous is None or timestamp-previous > max_gap:
                episode = dict(person_id=person, global_track_id=f'{person}:{sum(e["person_id"]==person for e in episodes)+1}',
                               start_time=timestamp,end_time=timestamp,cameras=[])
                episodes.append(episode)
            episode['end_time'] = timestamp
            camera = int(row['camera_id'])
            if camera not in episode['cameras']:
                episode['cameras'].append(camera)
            row.update(person_id=person,global_track_id=episode['global_track_id'],global_timestamp_s=f'{timestamp:.6f}')
            previous = timestamp
    target = source.with_name('global_track_episodes.csv')
    with target.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns+['person_id','global_track_id','global_timestamp_s'])
        writer.writeheader()
        writer.writerows(rows)
    source.with_name('global_track_episodes.json').write_text(json.dumps(episodes,indent=2),encoding='utf-8')
    return episodes
