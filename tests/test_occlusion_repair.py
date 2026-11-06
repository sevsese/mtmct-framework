import csv
import json

from mtmct.association import Config, associate
from mtmct.export import FIELDS, export_predictions
from mtmct.occlusion_repair import (RepairConfig, find_duplicate_tracklets,
                                    read_observations, relink_after_occlusion)


def row(cam, frame, lid, box=(0, 0, 20, 60), world=(0., 0.)):
    return dict(camera_id=cam, frame_id=frame, local_id=lid,
                x1=box[0], y1=box[1], x2=box[2], y2=box[3],
                confidence=0.9, world_x=world[0] if world else "",
                world_y=world[1] if world else "")


def save_obs(path, rows):
    with path.open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)


def track(cam, lid, start, end, embedding, positions=None):
    return {"camera_id": cam, "local_id": lid, "start_frame": start,
            "end_frame": end, "embedding": embedding,
            "positions": positions or []}


def test_short_shadow_duplicate_is_suppressed_but_cache_is_unchanged(tmp_path):
    rows = [row(3, t, 2, (10, 10, 110, 110), (t*.01, 0)) for t in range(15)]
    rows += [row(3, t, 31, (12, 12, 102, 105), (t*.01 + .03, 0)) for t in (9, 10)]
    p = tmp_path / "observations.csv"
    save_obs(p, rows)
    tracks = [track(3, 2, 0, 14, [1, 0]), track(3, 31, 9, 10, [.98, .2])]
    original = p.read_bytes()
    cleaned, suppressed, journal = find_duplicate_tracklets(tracks, read_observations(p))
    assert suppressed == {(3, 31)}
    assert [t['local_id'] for t in cleaned] == [2]
    assert len(journal) == 1
    assert p.read_bytes() == original
    # Export keeps original CSV intact but excludes the suppressed detection.
    mp = tmp_path / 'global_ids.json'
    mp.write_text(json.dumps([dict(camera_id=3, local_id=2, global_id=1)]))
    exported, _ = export_predictions(p, mp, tmp_path, excluded_keys=suppressed)
    assert len(list(csv.DictReader(exported.open()))) == 15


def test_real_overlapping_person_not_suppressed(tmp_path):
    rows = [row(3, t, 2, (10, 10, 110, 110), (0, 0)) for t in range(15)]
    rows += [row(3, t, 31, (12, 12, 102, 105), (0.03, 0)) for t in (9, 10)]
    p = tmp_path / "observations.csv"
    save_obs(p, rows)
    # Looks spatially similar, but strongly different ReID must preserve both.
    tracks = [track(3, 2, 0, 14, [1, 0]), track(3, 31, 9, 10, [0, 1])]
    _, suppressed, _ = find_duplicate_tracklets(tracks, read_observations(p))
    assert not suppressed


def test_duplicate_must_overlap_every_short_frame(tmp_path):
    rows = [row(3, t, 2, (10, 10, 110, 110)) for t in range(15)]
    rows += [row(3, t, 31, (12, 12, 102, 105)) for t in (9, 16)]
    p = tmp_path / "observations.csv"
    save_obs(p, rows)
    tracks = [track(3, 2, 0, 14, [1, 0]), track(3, 31, 9, 16, [1, 0])]
    _, suppressed, _ = find_duplicate_tracklets(tracks, read_observations(p))
    assert not suppressed


def test_occlusion_relink_reuses_existing_global_id_if_other_camera_witnesses(tmp_path):
    # Local track 2 is interrupted by an 8-frame gap. A different camera
    # continues to observe the same floor position (independent witness).
    a = track(15, 2, 0, 81, [1, 0], [[81, 0., 0.]])
    b = track(15, 21, 90, 112, [.76, .65], [[90, 1.1, 0.], [91, 1.1, 0.]])
    witness = track(3, 2, 0, 112, [1, 0], [[90, 1.1, .1], [91, 1.1, .1]])
    tracks = [a, b, witness]
    rows = [row(15, 81, 2, world=(0, 0)), row(15, 90, 21, world=(1.1, 0)),
            row(15, 91, 21, world=(1.1, 0)),
            row(3, 90, 2, world=(1.1, .1)), row(3, 91, 2, world=(1.1, .1))]
    p = tmp_path / "observations.csv"
    save_obs(p, rows)
    # Sim=0.76 is below baseline threshold 0.80: requires the special rescue.
    cfg = Config(use_geometry=False, appearance_weight=.95, spatial_weight=0.,
                 temporal_weight=.05, max_cost=.125, min_similarity=.8)
    mapping, _ = associate(tracks, cfg)
    assert mapping[(15, 2)] == mapping[(3, 2)]
    assert mapping[(15, 21)] != mapping[(15, 2)]
    repaired, journal = relink_after_occlusion(tracks, mapping, cfg, read_observations(p))
    assert repaired[(15, 21)] == repaired[(15, 2)]
    assert journal[0]["decision"] == "accepted"
    assert journal[0]["witness_distance_m"] < .75


def test_no_witness_cannot_join_low_similarity_or_implausible_motion(tmp_path):
    tracks = [track(15, 2, 0, 81, [1, 0]), track(15, 21, 90, 112, [.76, .65])]
    p = tmp_path / "observations.csv"
    save_obs(p, [row(15, 81, 2), row(15, 90, 21)])
    result, journal = relink_after_occlusion(tracks, {(15, 2): 2, (15, 21): 19},
                                             Config(use_geometry=False), read_observations(p))
    assert result[(15, 2)] != result[(15, 21)]
    assert journal[0]["reason"] == "no_independent_witness"
    # Even exact same appearance cannot rescue someone who teleported.
    tracks[1]["embedding"] = [1, 0]
    save_obs(p, [row(15, 81, 2, world=(0, 0)), row(15, 90, 21, world=(100, 100))])
    result, _ = relink_after_occlusion(tracks, {(15, 2): 2, (15, 21): 19},
                                       Config(use_geometry=False), read_observations(p))
    assert result[(15, 2)] != result[(15, 21)]


def test_no_relink_when_groups_contain_overlapping_tracks_same_camera(tmp_path):
    old = track(15, 2, 0, 81, [1, 0])
    later = track(15, 21, 90, 112, [1, 0])
    alive = track(3, 2, 0, 110, [1, 0])
    bad_duplicate = track(3, 31, 99, 100, [1, 0])
    ts = [old, later, alive, bad_duplicate]
    p = tmp_path / "observations.csv"
    save_obs(p, [row(15, 81, 2), row(15, 90, 21)])
    mapping = {(15, 2): 2, (3, 2): 2, (15, 21): 19, (3, 31): 19}
    result, journal = relink_after_occlusion(ts, mapping,
                       Config(use_geometry=False), read_observations(p))
    assert result[(15, 2)] != result[(15, 21)]
    assert journal[0]["reason"] == "component_conflict"


def test_low_quality_two_frame_duplicate_needs_strict_geometry(tmp_path):
    # Reproduce the actual failure mode: two-frame short tracker fragment with
    # unreliable appearance, but matching floor and substantial box overlap.
    long_box = (11, 593, 126, 816)
    rows = [row(3, t, 2, long_box, (-10.45, -61.5)) for t in range(40)
            if t not in (25, 26)]
    rows += [row(3, 25, 2, (13.73, 593.36, 126.83, 816.14), (-10.4512, -61.4962)),
             row(3, 26, 2, (11.21, 595.16, 125.90, 815.95), (-10.4520, -61.5102)),
             row(3, 25, 31, (0., 593.12, 80.82, 809.20), (-10.5206, -61.7839)),
             row(3, 26, 31, (6.97, 624.69, 78.26, 809.06), (-10.5254, -61.7705))]
    p = tmp_path / "observations.csv"
    save_obs(p, rows)
    import math
    long_t = track(3, 2, 0, 39, [1, 0])
    short_t = track(3, 31, 25, 26, [.626, math.sqrt(1-.626**2)])
    short_t["feature_quality"] = [.128, .104]
    cleaned, suppressed, journal = find_duplicate_tracklets(
        [long_t, short_t], read_observations(p))
    assert suppressed == {(3, 31)}
    assert journal[0]["reason"] == "short_low_quality_duplicate"

    # If the quality is good, low cosine similarity signals a real ambiguity:
    # do not suppress an overlapping second person.
    short_t["feature_quality"] = [.65, .7]
    _, suppressed, _ = find_duplicate_tracklets([long_t, short_t], read_observations(p))
    assert suppressed == set()


def test_unique_cross_camera_witness_rescues_069_similarity(tmp_path):
    import math
    old = track(15, 2, 0, 81, [1, 0], [[81, 0, 0]])
    new = track(15, 21, 90, 112, [.699, math.sqrt(1-.699**2)],
                [[f, 1.1, 0] for f in range(90, 95)])
    witness = track(3, 2, 0, 112, [1, 0],
                    [[f, 1.1, .2] for f in range(90, 95)])
    unrelated = track(3, 3, 0, 112, [0, 1],
                      [[f, 6, 0] for f in range(90, 95)])
    p = tmp_path / 'observations.csv'
    save_obs(p, [row(15, 81, 2, world=(0, 0)),
                 row(15, 90, 21, world=(1.1, 0))])
    mapping = {(15, 2): 2, (3, 2): 2, (15, 21): 19, (3, 3): 3}
    config = Config(use_geometry=False)
    repaired, journal = relink_after_occlusion(
        [old, new, witness, unrelated], mapping, config, read_observations(p))
    assert repaired[(15, 21)] == 2
    assert any(e.get('rescue') == 'unique_cross_camera_geometry'
               for e in journal if e['decision'] == 'accepted')


def test_weak_appearance_rescue_rejected_if_second_person_nearby(tmp_path):
    import math
    old = track(15, 2, 0, 81, [1, 0], [[81, 0, 0]])
    new = track(15, 21, 90, 112, [.699, math.sqrt(1-.699**2)],
                [[f, 1.1, 0] for f in range(90, 95)])
    witness = track(3, 2, 0, 112, [1, 0],
                    [[f, 1.1, .2] for f in range(90, 95)])
    competitor = track(3, 3, 0, 112, [0, 1],
                      [[f, 1.2, .1] for f in range(90, 95)])
    p = tmp_path / 'observations.csv'
    save_obs(p, [row(15, 81, 2, world=(0, 0)),
                 row(15, 90, 21, world=(1.1, 0))])
    mapping = {(15, 2): 2, (3, 2): 2, (15, 21): 19, (3, 3): 3}
    repaired, journal = relink_after_occlusion(
        [old, new, witness, competitor], mapping,
        Config(use_geometry=False), read_observations(p))
    assert repaired[(15, 21)] == 19
    assert journal[0]['reason'] == 'insufficient_unique_witness'


def test_low_similarity_without_projected_positions_stays_unmerged(tmp_path):
    import math
    old = track(15, 2, 0, 81, [1, 0])
    new = track(15, 21, 90, 112, [.699, math.sqrt(1-.699**2)])
    witness = track(3, 2, 0, 112, [1, 0])
    p = tmp_path / 'observations.csv'
    save_obs(p, [row(15, 81, 2, world=None),
                 row(15, 90, 21, world=None)])
    mapping = {(15, 2): 2, (3, 2): 2, (15, 21): 19}
    repaired, journal = relink_after_occlusion(
        [old, new, witness], mapping, Config(use_geometry=False),
        read_observations(p))
    assert repaired[(15, 21)] == 19
    assert journal[0]['reason'] == 'insufficient_unique_witness'
