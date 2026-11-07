"""SQLite registry scoped to one recording. Replaying identical input is idempotent."""
from __future__ import annotations

from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import sqlite3


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def prior_revocations(path):
    if not path.is_file():
        return []
    with closing(sqlite3.connect(path)) as db:
        return [(a, b) for a, b, evidence in db.execute(
            "SELECT left_uid,right_uid,evidence FROM edges WHERE revision=(SELECT MAX(id) FROM revisions)"
        ) if json.loads(evidence)["state"] == "revoked"]


def register(path, scope, tracks, groups, edges):
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS revisions(id INTEGER PRIMARY KEY, scope TEXT NOT NULL,
                digest TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS identities(revision INTEGER, global_id INTEGER,
                state TEXT, PRIMARY KEY(revision, global_id));
            CREATE TABLE IF NOT EXISTS tracklets(revision INTEGER, uid TEXT, global_id INTEGER,
                camera_id INTEGER, local_id INTEGER, start_s REAL, end_s REAL, quality REAL,
                provenance TEXT, PRIMARY KEY(revision, uid));
            CREATE TABLE IF NOT EXISTS prototypes(revision INTEGER, global_id INTEGER, sample TEXT);
            CREATE TABLE IF NOT EXISTS edges(revision INTEGER, left_uid TEXT, right_uid TEXT,
                evidence TEXT, PRIMARY KEY(revision, left_uid, right_uid));
            CREATE TABLE IF NOT EXISTS events(revision INTEGER, type TEXT, payload TEXT);
        """)
        db.execute("BEGIN IMMEDIATE")
        previous = db.execute("SELECT id,scope,digest FROM revisions ORDER BY id DESC LIMIT 1").fetchone()
        if previous and previous[1] != scope:
            raise ValueError("Identity registry belongs to a different recording")
        digest = hashlib.sha256()
        digest.update(_json([[tracks[i].uid for i in g] for g in groups]).encode())
        for track in tracks:
            digest.update(_json([track.uid, track.interval, track.quality, track.anchor, track.prototypes,
                                 track.row.get("provenance")]).encode())
        for edge in edges:
            digest.update(_json(edge).encode())
        digest = digest.hexdigest()
        if previous and previous[2] == digest:
            revision = previous[0]
        else:
            revision = _save_revision(db, previous, scope, digest, tracks, groups, edges)
        mapping = {(c, l): g for c, l, g in db.execute(
            "SELECT camera_id,local_id,global_id FROM tracklets WHERE revision=?", (revision,))}
        states = dict(db.execute("SELECT global_id,state FROM identities WHERE revision=?", (revision,)))
        events = [{"type": kind, **json.loads(payload)} for kind, payload in db.execute(
            "SELECT type,payload FROM events WHERE revision=?", (revision,))]
        return mapping, {"revision": revision, "states": states, "events": events}


def _save_revision(db, previous, scope, digest, tracks, groups, edges):
    old_owner = dict(db.execute("SELECT uid,global_id FROM tracklets WHERE revision=?",
                                (previous[0],))) if previous else {}
    last_id = db.execute("SELECT COALESCE(MAX(global_id),0) FROM identities").fetchone()[0]
    candidates = defaultdict(int)
    owners = {}
    for k, group in enumerate(groups):
        for i in group:
            track = tracks[i]
            old = old_owner.get(track.uid)
            if old is None:
                source = next((p["source_uid"] for p in track.row.get("provenance", [])
                               if "source_uid" in p), None)
                old = old_owner.get(source)
            if old is not None:
                owners[i] = old
                candidates[k, old] += 1
    new_ids, claimed = {}, set()
    for (k, gid), _ in sorted(candidates.items(), key=lambda pair: (-pair[1], pair[0][1], pair[0][0])):
        if k not in new_ids and gid not in claimed:
            new_ids[k] = gid
            claimed.add(gid)
    for k in range(len(groups)):
        if k not in new_ids:
            last_id += 1
            new_ids[k] = last_id
    revision = db.execute(
        "INSERT INTO revisions(scope,digest,created_at) VALUES (?,?,?)",
        (scope, digest, datetime.now(timezone.utc).isoformat())).lastrowid
    confirmed = {i for e in edges if e["state"] == "confirmed" for i in (e["i"], e["j"])}
    replacements = defaultdict(set)
    ancestors = defaultdict(set)
    for k, group in enumerate(groups):
        gid = new_ids[k]
        state = "confirmed" if confirmed.intersection(group) else "provisional"
        db.execute("INSERT INTO identities VALUES (?,?,?)", (revision, gid, state))
        gallery = []
        for i in group:
            track = tracks[i]
            old = owners.get(i)
            if old is not None:
                replacements[old].add(gid)
                ancestors[gid].add(old)
            db.execute("INSERT INTO tracklets VALUES (?,?,?,?,?,?,?,?,?)",
                       (revision, track.uid, gid, *track.key, *track.interval, track.quality,
                        _json(track.row.get("provenance", []))))
            if i in confirmed and track.anchor:
                gallery.extend(track.prototypes)
        # Retain multiple views, with a bounded gallery per camera.
        cameras = defaultdict(list)
        for sample in sorted(gallery, key=lambda sample: (-sample["q"], sample["source"])):
            if len(cameras[sample["camera_id"]]) < 4:
                cameras[sample["camera_id"]].append(sample)
        db.executemany("INSERT INTO prototypes VALUES (?,?,?)",
                       ((revision, gid, _json(sample)) for samples in cameras.values() for sample in samples))
    db.executemany("INSERT INTO edges VALUES (?,?,?,?)",
                   ((revision, e["left_uid"], e["right_uid"], _json(e)) for e in edges))
    events = [("split", {"from": old, "to": sorted(new)}) for old, new in replacements.items() if len(new) > 1]
    events += [("merge", {"from": sorted(old), "to": new}) for new, old in ancestors.items() if len(old) > 1]
    prior_ids = {gid for gid, in db.execute("SELECT global_id FROM identities WHERE revision=?",
                                           (previous[0],))} if previous else set()
    for gid in sorted(prior_ids - set(new_ids.values())):
        db.execute("INSERT INTO identities VALUES (?,?,?)", (revision, gid, "archived"))
        if gid in old_owner.values() and gid not in replacements:
            events.append(("archive", {"global_id": gid}))
    db.executemany("INSERT INTO events VALUES (?,?,?)", ((revision, kind, _json(event)) for kind, event in events))
    return revision
