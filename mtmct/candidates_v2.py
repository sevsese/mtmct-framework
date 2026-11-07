"""Indexed appearance candidates plus exact same-camera temporal neighbors."""
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree


def candidate_pairs(tracks, top_k):
    pairs = set()
    dimensions = defaultdict(list)
    cameras = defaultdict(list)
    for i, track in enumerate(tracks):
        if track.mean.size:
            dimensions[track.mean.size].append(i)
        cameras[track.key[0]].append(i)
    for indexes in dimensions.values():
        vectors = np.array([tracks[i].mean for i in indexes])
        tree = cKDTree(vectors)
        _, neighbors = tree.query(vectors, k=min(len(indexes), top_k+1))
        neighbors = np.asarray(neighbors).reshape(len(indexes), -1)
        for i, nearest in zip(indexes, neighbors):
            for j in nearest:
                if i != indexes[j]:
                    pairs.add(tuple(sorted((i, indexes[j]))))
        # Query multiple views too: an average alone can hide a valid view.
        owners=[]
        prototypes=[]
        for i in indexes:
            for p in tracks[i].prototypes:
                if len(p['e'])==vectors.shape[1]:
                    owners.append(i)
                    prototypes.append(p['e'])
        if prototypes:
            tree=cKDTree(np.asarray(prototypes))
            _,nearest=tree.query(prototypes,k=min(len(prototypes),4*(top_k+1)))
            for owner,neighbors_ in zip(owners,np.asarray(nearest).reshape(len(owners),-1)):
                selected=set()
                for neighbor in neighbors_:
                    other=owners[neighbor]
                    if other!=owner:
                        selected.add(other)
                        pairs.add(tuple(sorted((owner,other))))
                    if len(selected)>=top_k:
                        break
    # Preserve short handoffs and overlapping duplicate candidates even when
    # their embeddings are poor. Physical vetoes are still checked on merges.
    for indexes in cameras.values():
        active = []
        for i in sorted(indexes, key=lambda i: tracks[i].interval[0]):
            start = tracks[i].interval[0]
            active = [j for j in active if tracks[j].interval[1]+1.8 >= start]
            pairs.update(tuple(sorted((i,j))) for j in active)
            active.append(i)
    return sorted(pairs)
