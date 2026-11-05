from mtmct.refinement import diagnose, refine


def row(local_id, start, end, embedding, frames=None, vectors=None):
    return {"camera_id": 0, "local_id": local_id, "start_frame": start, "end_frame": end,
            "embedding": embedding, "positions": [], "feature_frames": frames or [],
            "feature_vectors": vectors or [], "feature_quality": [1] * len(frames or [])}


def test_dry_run_detects_appearance_jump_without_mutation():
    tracklets = [row(1, 0, 20, [1, 0], [0, 10], [[1, 0], [0, 1]])]
    refined, report, remap = refine(tracklets, split_similarity=0.5)
    assert refined == tracklets and remap == []
    assert report["splits"][0]["split_frame"] == 10


def test_auto_split_creates_provenance_and_new_local_id():
    tracklets = [row(1, 0, 20, [1, 0], [0, 10], [[1, 0], [0, 1]])]
    refined, _report, remap = refine(tracklets, auto_split=True, split_similarity=0.5)
    assert len(refined) == 2 and {item["local_id"] for item in refined} == {1, 2}
    assert len(remap) == 2 and all(item["local_id"] == 1 for item in remap)


def test_merge_fragment_and_reject_overlap():
    fragments = [row(1, 0, 9, [1, 0]), row(2, 20, 30, [0.99, 0.01])]
    refined, report, _remap = refine(fragments, auto_merge=True, merge_similarity=0.9, max_gap_frames=20)
    assert len(report["merges"]) == 1 and len(refined) == 1
    overlapping = [row(1, 0, 20, [1, 0]), row(2, 10, 30, [1, 0])]
    assert diagnose(overlapping)["merges"] == []
