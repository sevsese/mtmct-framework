"""Optional offline identity memory over already-associated tracklet groups."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .association import unit


def apply_memory(tracklets: list[dict], mapping: dict[tuple[int, int], int],
                 similarity_threshold: float = 0.9, max_prototypes: int = 3):
    key_to_index = {(int(row["camera_id"]), int(row["local_id"])): index
                    for index, row in enumerate(tracklets)}
    groups = defaultdict(list)
    for key, global_id in mapping.items():
        groups[global_id].append(key_to_index[key])
    group_ids = sorted(groups)
    parent = list(range(len(group_ids)))
    members = {index: set(groups[global_id]) for index, global_id in enumerate(group_ids)}

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def prototypes(indices):
        vectors = []
        for index in sorted(indices):
            row = tracklets[index]
            candidates = row.get("prototypes") or ([row["embedding"]] if row.get("embedding") else [])
            vectors.extend(unit(vector) for vector in candidates if len(vector))
        return vectors[:max_prototypes]

    def compatible(left_indices, right_indices):
        for left in left_indices:
            for right in right_indices:
                a, b = tracklets[left], tracklets[right]
                if (int(a["camera_id"]) == int(b["camera_id"])
                        and max(int(a["start_frame"]), int(b["start_frame"]))
                        <= min(int(a["end_frame"]), int(b["end_frame"]))):
                    return False
        return True

    candidates = []
    for left in range(len(group_ids)):
        for right in range(left + 1, len(group_ids)):
            left_vectors, right_vectors = prototypes(members[left]), prototypes(members[right])
            if not left_vectors or not right_vectors:
                continue
            similarity = max(float(np.clip(a @ b, -1, 1)) for a in left_vectors for b in right_vectors)
            if similarity >= similarity_threshold:
                candidates.append((similarity, left, right))
    decisions = []
    for similarity, left, right in sorted(candidates, reverse=True):
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            continue
        if not compatible(members[left_root], members[right_root]):
            decisions.append({"decision": "rejected", "reason": "temporal_conflict",
                              "similarity": round(similarity, 6)})
            continue
        parent[right_root] = left_root
        members[left_root].update(members.pop(right_root))
        decisions.append({"decision": "confirmed", "reason": "memory_similarity",
                          "similarity": round(similarity, 6)})

    roots = sorted({find(index) for index in range(len(group_ids))}, key=lambda root: min(
        (int(tracklets[index]["start_frame"]), int(tracklets[index]["camera_id"]),
         int(tracklets[index]["local_id"])) for index in members[root]))
    root_to_global = {root: index + 1 for index, root in enumerate(roots)}
    output = {}
    records = []
    for root in roots:
        indices = sorted(members[root])
        global_id = root_to_global[root]
        for index in indices:
            row = tracklets[index]
            output[(int(row["camera_id"]), int(row["local_id"]))] = global_id
        records.append({
            "global_id": global_id,
            "status": "confirmed",
            "tracklets": [[int(tracklets[index]["camera_id"]), int(tracklets[index]["local_id"])]
                          for index in indices],
            "history": [{"camera_id": int(tracklets[index]["camera_id"]),
                         "start_frame": int(tracklets[index]["start_frame"]),
                         "end_frame": int(tracklets[index]["end_frame"])} for index in indices],
            "prototypes": [vector.tolist() for vector in prototypes(indices)],
        })
    return output, records, decisions
