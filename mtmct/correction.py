"""Rebuild graph components from saved edges to reject inconsistent merges."""
from __future__ import annotations

import copy

from .association import Config, pair_score


def rebuild(tracklets: list[dict], journal: list[dict], cfg: Config):
    keys = [(int(row["camera_id"]), int(row["local_id"])) for row in tracklets]
    parent = list(range(len(tracklets)))
    members = {index: {index} for index in range(len(tracklets))}

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    corrected = copy.deepcopy(journal)
    accepted = sorted(
        (edge for edge in corrected if edge.get("decision") == "accepted"),
        key=lambda edge: (float(edge.get("cost", 1e9)), int(edge["i"]), int(edge["j"])),
    )
    conflicts = []
    for edge in accepted:
        left_root, right_root = find(int(edge["i"])), find(int(edge["j"]))
        if left_root == right_root:
            edge["status"] = "confirmed"
            continue
        incompatible = []
        for left in members[left_root]:
            for right in members[right_root]:
                score = pair_score(tracklets[left], tracklets[right], cfg)
                if score["cost"] is None or score["cost"] > cfg.max_cost:
                    incompatible.append({"i": left, "j": right, "reason": score["reason"],
                                         "similarity": score["similarity"]})
        if incompatible:
            edge.update(decision="rejected", status="rejected",
                        reason="identity_correction_conflict")
            conflicts.append({"edge": [edge["i"], edge["j"]],
                              "incompatible_pairs": incompatible})
            continue
        parent[right_root] = left_root
        members[left_root].update(members.pop(right_root))
        edge["status"] = "confirmed"

    roots = sorted({find(index) for index in range(len(tracklets))}, key=lambda root: min(
        (int(tracklets[index]["start_frame"]), keys[index]) for index in members[root]))
    root_to_global = {root: index + 1 for index, root in enumerate(roots)}
    mapping = {keys[index]: root_to_global[find(index)] for index in range(len(tracklets))}
    return mapping, corrected, conflicts
