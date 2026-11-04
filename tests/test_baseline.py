import csv
from pathlib import Path
import numpy as np
from mtmct.geometry import pixel_to_floor_matrix, project_pixel, camera_number
from mtmct.association import associate, Config, pair_cost
from mtmct.gt import value
from mtmct.synthetic import make_demo


def track(camera, lid, start, end, embedding, positions):
    return {"camera_id":camera,"local_id":lid,
            "start_frame":start,"end_frame":end,
            "embedding":embedding,"positions":positions}


def test_camera_conversion_and_projection():
    assert camera_number("Camera_0007")==7
    # Synthetic affine floor mapping u=2X+10, v=3Y+20.
    P=np.array([[2,0,0,10],[0,3,0,20],[0,0,0,1]],dtype=float)
    H=pixel_to_floor_matrix({"cameraMatrix":P.tolist()})
    np.testing.assert_allclose(project_pixel(H,18,35),(4,5))


def test_same_person_merges_across_cameras():
    a=track(0,1,0,30,[1,0],[[30,2,2]])
    b=track(1,9,30,60,[1,0],[[30,2,2]])
    result,_=associate([a,b],Config())
    assert result[(0,1)]==result[(1,9)]


def test_overlap_same_camera_forbidden_even_same_appearance():
    a=track(0,1,0,60,[1,0],[[30,0,0]])
    b=track(0,2,30,90,[1,0],[[30,0,0]])
    result,_=associate([a,b],Config())
    assert result[(0,1)]!=result[(0,2)]


def test_cross_camera_overlap_far_apart_forbidden():
    a=track(0,1,0,60,[1,0],[[30,0,0]])
    b=track(1,2,0,60,[1,0],[[30,10,10]])
    assert pair_cost(a,b,Config()) is None


def test_cross_camera_transition_too_fast_forbidden():
    a=track(0,1,0,29,[1,0],[[29,0,0]])
    b=track(1,2,30,60,[1,0],[[30,40,40]])
    assert pair_cost(a,b,Config()) is None


def test_weak_appearance_is_rejected_even_if_time_matches():
    a=track(0,1,0,30,[1,0],[])
    b=track(1,2,0,30,[0.7,0.714],[])
    assert pair_cost(a,b,Config(use_geometry=False)) is None


def test_nvidia_ground_truth_space_separated_fields():
    obj={"object type":"Person","object id":7,
         "2d bounding box visible":{"Camera_0003":[1,2,3,4]}}
    assert value(obj,"object_type")=="Person"
    assert value(obj,"object_id")==7
    assert "Camera_0003" in value(obj,"2d_bounding_box_visible")


def test_transitive_contradictions_rejected():
    # A-B close, B-C close, A-C too far; no all-three merger.
    a=track(0,1,0,60,[1,0],[[30,0,0]])
    b=track(1,2,0,60,[1,0],[[30,1.5,0]])
    c=track(2,3,0,60,[1,0],[[30,3,0]])
    mapping,_=associate([a,b,c],Config())
    assert len(set(mapping.values()))>=2


def test_synthetic_end_to_end(tmp_path):
    make_demo(tmp_path)
    assert (tmp_path/"global_tracks_2d.csv").exists()
    assert (tmp_path/"track1_person_approx.txt").exists()
    lines=(tmp_path/"track1_person_approx.txt").read_text().splitlines()
    assert lines and all(len(row.split())==11 for row in lines)


def test_mocked_two_camera_video_pipeline(tmp_path, monkeypatch):
    """End-to-end video and CSV paths with a fake detector, no model download.

    IMPORTANT: this is not a neural model quality test.
    """
    import json
    import sys
    import types
    import cv2
    import torch
    from mtmct.tracking import run
    from mtmct.association import load_tracklets, associate, save_mapping
    from mtmct.export import export_predictions
    from mtmct.visualize import render

    scene=tmp_path/"Warehouse_000"
    videos=scene/"videos"
    videos.mkdir(parents=True)
    P=[[2,0,0,10],[0,3,0,20],[0,0,0,1]]
    scene.joinpath("calibration.json").write_text(json.dumps({
        "sensors":[{"id":f"Camera_{i:04d}","type":"camera","cameraMatrix":P}
                   for i in range(2)]}),encoding="utf-8")
    for cam in range(2):
        path=videos/f"Camera_{cam:04d}.mp4"
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),
                               10.,(64,64))
        assert writer.isOpened()
        for _ in range(5):
            image=np.zeros((64,64,3),dtype=np.uint8)
            cv2.rectangle(image,(10,10),(30,50),(60,180,230),-1)
            writer.write(image)
        writer.release()

    class FakeYOLO:
        models_created=0
        def __init__(self,_weights):
            self.cam=FakeYOLO.models_created
            FakeYOLO.models_created+=1
        def track(self,frame,**kwargs):
            assert kwargs["persist"] is True
            boxes=types.SimpleNamespace(
                id=torch.tensor([7 if self.cam==0 else 12]),
                xyxy=torch.tensor([[10.,10.,30.,50.]]),
                conf=torch.tensor([.90]))
            return [types.SimpleNamespace(boxes=boxes)]
    monkeypatch.setitem(sys.modules,"ultralytics",types.SimpleNamespace(YOLO=FakeYOLO))
    out=tmp_path/"run"
    run(scene,out,"fake.pt","color",None,"cpu",5,1,1,64,.1)
    assert FakeYOLO.models_created==2  # local ByteTrack state isolated
    ts=load_tracklets(out/"local_tracklets.json")
    assert len(ts)==2
    assert (out/"performance.json").exists()
    ids,_=associate(ts,Config())
    assert ids[(0,7)]==ids[(1,12)]
    save_mapping(ids,out/"global_ids.json")
    csv_path,txt_path=export_predictions(out/"observations.csv",out/"global_ids.json",out)
    with open(csv_path,newline="") as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==10
    assert len({r["global_id"] for r in rows})==1
    assert len(txt_path.read_text().splitlines())==5  # fused per frame
    render(scene,csv_path,out/"mosaic.mp4",max_frames=5,stride=1,scale=1.)
    assert (out/"mosaic.mp4").stat().st_size>0


def test_reid_only_ignores_bad_geometry_if_disabled():
    a = track(0, 1, 0, 30, [1, 0], [[15, 0, 0]])
    b = track(1, 9, 0, 30, [1, 0], [[15, 1000, 1000]])
    cfg = Config(use_geometry=False, spatial_weight=0)
    result, _ = associate([a, b], cfg)
    assert result[(0, 1)] == result[(1, 9)]
    assert pair_cost(a, b, Config()) is None


def test_reid_only_respects_same_camera_overlap():
    a = track(0, 1, 0, 60, [1, 0], [])
    b = track(0, 2, 30, 90, [1, 0], [])
    result, _ = associate([a, b], Config(use_geometry=False, spatial_weight=0))
    assert result[(0, 1)] != result[(0, 2)]
