"""Global association CLI; separate from expensive neural inference."""
from __future__ import annotations
import argparse
from pathlib import Path
from .association import Config, associate, associate_graph, load_tracklets, save_mapping
from .export import export_predictions
from .cache import validate_cache
from .memory import apply_memory
from .time_sync import CameraTransitionModel, parse_offsets
from .correction import rebuild
from .occlusion_repair import (RepairConfig, read_observations,
                               find_duplicate_tracklets, relink_after_occlusion)
import json
from dataclasses import asdict


def run(run_dir: Path, cfg: Config, scene_id: int = 0,
        expected_detector: str | None = None, matcher: str = "baseline",
        allow_same_camera_fragments: bool = False, memory: bool = False,
        memory_similarity: float = 0.9, correct_identities: bool = False,
        repair_occlusions: bool = False, repair_config: RepairConfig | None = None,
        min_observations: int = 1, *, v2_settings=None, v2_features=None, revocations=None):
    manifest = validate_cache(run_dir, expected_detector)
    if manifest.get("legacy"):
        print("WARNING:", manifest["warning"])
    if min_observations < 1:
        raise ValueError("min_observations must be positive")
    all_tracklets=load_tracklets(run_dir/"local_tracklets.json")
    if matcher == "v2":
        from .identity_v2 import Settings, link_v2
        if v2_features:
            enriched = load_tracklets(v2_features)
            signature = lambda rows: sorted((r["camera_id"], r["local_id"], r["start_frame"], r["end_frame"]) for r in rows)
            if signature(enriched) != signature(all_tracklets):
                raise ValueError("Feature cache must describe the same original tracklets")
            all_tracklets = enriched
        return link_v2(run_dir, manifest, all_tracklets, cfg, scene_id=scene_id,
                       same_camera=allow_same_camera_fragments, min_observations=min_observations,
                       settings=v2_settings or Settings(), revocations=revocations,
                       repair_config=(repair_config or RepairConfig()) if repair_occlusions else None)
    def observation_count(row):
        frames = row.get("observation_frames")
        return len(frames) if frames is not None else 1

    suppressed = {
        (int(row["camera_id"]), int(row["local_id"]))
        for row in all_tracklets
        if observation_count(row) < min_observations
    }
    tracklets = [row for row in all_tracklets
                 if (int(row["camera_id"]), int(row["local_id"])) not in suppressed]
    observations = None
    repair_log = None
    if repair_occlusions:
        repair_config = repair_config or RepairConfig()
        observations = read_observations(run_dir / "observations.csv")
        tracklets, duplicate_keys, duplicates = find_duplicate_tracklets(
            tracklets, observations, repair_config)
        suppressed.update(duplicate_keys)
        repair_log = {"experimental": True, "settings": asdict(repair_config),
                      "suppressed_duplicates": duplicates}
    if matcher == "graph":
        mapping,edges=associate_graph(tracklets,cfg,allow_same_camera_fragments)
    elif matcher == "baseline":
        mapping,edges=associate(tracklets,cfg)
    else:
        raise ValueError("matcher must be baseline, graph or v2")
    if correct_identities:
        if matcher != "graph":
            raise ValueError("identity correction requires --matcher graph")
        mapping,edges,conflicts=rebuild(tracklets,edges,cfg)
        (run_dir/"identity_correction.json").write_text(
            __import__("json").dumps({"conflicts":conflicts,"journal":edges},indent=2),
            encoding="utf-8")
    if memory:
        mapping,records,memory_decisions=apply_memory(
            tracklets,mapping,similarity_threshold=memory_similarity,config=cfg)
        (run_dir/"global_memory.json").write_text(
            __import__("json").dumps({"groups":records,"decisions":memory_decisions},indent=2),
            encoding="utf-8")
    if repair_occlusions:
        mapping, recoveries = relink_after_occlusion(
            tracklets, mapping, cfg, observations, repair_config)
        repair_log["occlusion_relinks"] = recoveries
        repair_log["summary"] = {
            "suppressed_tracklets": len(suppressed),
            "accepted_relinks": sum(x["decision"] == "accepted" for x in recoveries),
        }
        (run_dir / "occlusion_repair.json").write_text(
            json.dumps(repair_log, indent=2), encoding="utf-8")
    save_mapping(mapping,run_dir/"global_ids.json")
    (run_dir/"association_edges.json").write_text(
        __import__("json").dumps(edges, indent=2), encoding="utf-8")
    out_csv,out_txt=export_predictions(
        run_dir/"observations.csv",run_dir/"global_ids.json",run_dir,
        scene_id=scene_id, excluded_keys=suppressed)
    return tracklets, mapping, edges, out_csv, out_txt


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--scene-id",type=int,default=0)
    p.add_argument("--max-cost",type=float,default=0.125)
    p.add_argument("--min-similarity",type=float,default=0.80,
                   help="Hard cosine-similarity gate before weighted scoring")
    p.add_argument("--max-gap",type=float,default=12.0)
    p.add_argument("--max-speed",type=float,default=3.5)
    p.add_argument("--max-overlap-distance",type=float,default=2.0)
    p.add_argument("--appearance-weight",type=float,default=0.75)
    p.add_argument("--spatial-weight",type=float,default=0.20)
    p.add_argument("--temporal-weight",type=float,default=0.05)
    p.add_argument("--fps",type=float,default=30.)
    p.add_argument("--require-geometry",action="store_true")
    p.add_argument("--ignore-geometry",action="store_true",
                   help="Ignore projected positions until calibration has been validated")
    p.add_argument("--geometry",action="store_true",help="Explicitly use validated geometry in v2")
    p.add_argument("--model",default=None,help="Reject cache created with another detector")
    p.add_argument("--matcher",choices=["baseline","graph","v2"],default="baseline")
    p.add_argument("--v2-gallery",choices=["mean","max","topk"],default="max")
    p.add_argument("--office-guards",action="store_true",help="Office appearance, continuity and lifetime checks for v2")
    p.add_argument("--v2-features",type=Path,help="Optional feature cache for the same tracklets")
    p.add_argument("--revoke",type=Path,help="JSON list of UID pairs to keep separate on this v2 pass")
    p.add_argument("--camera-start",action="append",default=[],metavar="CAMERA=SECONDS")
    p.add_argument("--allow-same-camera-fragments",action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--time",action=argparse.BooleanOptionalAction,default=True,
                   help="Enable time constraints (use --no-time only for diagnostics)")
    p.add_argument("--gallery-weight",type=float,default=None)
    p.add_argument("--part-weight",type=float,default=0.30)
    p.add_argument("--min-observations",type=int,default=3,
                   help="Suppress very short unconfirmed local tracks")
    p.add_argument("--memory",action="store_true",help="Enable offline prototype memory")
    p.add_argument("--memory-similarity",type=float,default=0.90)
    p.add_argument("--camera-offset",action="append",default=[],metavar="CAMERA=SECONDS")
    p.add_argument("--topology",type=Path,help="JSON measured/forbidden camera routes")
    p.add_argument("--correct-identities",action="store_true")
    p.add_argument("--repair-occlusions",action="store_true",
                   help="Experimental: suppress short duplicate tracklets and relink short occlusions")
    a=p.parse_args()
    cfg=Config(max_cost=a.max_cost,min_similarity=a.min_similarity,max_gap_seconds=a.max_gap,
               max_speed_mps=a.max_speed,overlap_distance_m=a.max_overlap_distance,
               appearance_weight=a.appearance_weight,spatial_weight=a.spatial_weight,
               temporal_weight=a.temporal_weight,fps=a.fps,
               gallery_weight=a.gallery_weight if a.gallery_weight is not None else (.25 if a.matcher == "v2" else .65),
               part_weight=a.part_weight,
               require_geometry=a.require_geometry,
               use_geometry=not a.ignore_geometry and (a.matcher != "v2" or a.geometry or a.require_geometry),
               use_time=a.time,
               camera_offsets_s=parse_offsets(a.camera_offset),
               camera_start_s=parse_offsets(a.camera_start),
               transition_model=CameraTransitionModel.from_json(a.topology) if a.topology else None)
    if not a.time:
        cfg.temporal_weight=0.0
    from dataclasses import replace
    from .identity_v2 import OFFICE_SETTINGS, Settings
    if a.office_guards and a.matcher != "v2":
        p.error("--office-guards requires --matcher v2")
    tracklets,mapping,edges,out_csv,out_txt=run(
        a.run_dir,cfg,a.scene_id,a.model,a.matcher,a.allow_same_camera_fragments,
        a.memory,a.memory_similarity,a.correct_identities,
        repair_occlusions=a.repair_occlusions,min_observations=a.min_observations,
        v2_settings=replace(OFFICE_SETTINGS if a.office_guards else Settings(), gallery=a.v2_gallery),
        v2_features=a.v2_features,
        revocations=json.loads(a.revoke.read_text(encoding="utf-8")) if a.revoke else None)
    accepted=sum(edge.get("decision")=="accepted" for edge in edges) if a.matcher!="baseline" else len(edges)
    print(f"Local tracklets: {len(tracklets)}, accepted edges: {accepted}, global IDs: {len(set(mapping.values()))}")
    print("Results:", out_csv, out_txt)
    print("NOTE: person-only approximate 3D boxes are NOT a competitive full Track-1 solution.")

if __name__=="__main__": main()
