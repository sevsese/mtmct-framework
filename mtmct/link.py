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
from .identity_graph import associate_adaptive, resume_identity_memory
from .refinement import refine, rewrite_observations


def run(run_dir: Path, cfg: Config, scene_id: int = 0,
        expected_detector: str | None = None, matcher: str = "baseline",
        allow_same_camera_fragments: bool = False, memory: bool = False,
        memory_similarity: float = 0.9, correct_identities: bool = False,
        repair_occlusions: bool = False, repair_config: RepairConfig | None = None,
        memory_mode: str = "legacy", split_tracklets: bool = False,
        memory_state: Path | None = None):
    manifest = validate_cache(run_dir, expected_detector)
    if manifest.get("legacy"):
        print("WARNING:", manifest["warning"])
    tracklets=load_tracklets(run_dir/"local_tracklets.json")
    observation_path = run_dir / 'observations.csv'
    if split_tracklets:
        tracklets, report, remap = refine(tracklets, auto_split=True, sustained_samples=3)
        observation_path = run_dir / 'refined_observations.csv'
        rewrite_observations(run_dir / 'observations.csv', observation_path, remap)
        (run_dir / 'refined_tracklets.json').write_text(json.dumps(tracklets), encoding='utf-8')
        (run_dir / 'refinement_report.json').write_text(json.dumps({'diagnosis': report, 'remap': remap}, indent=2), encoding='utf-8')
    suppressed = set()
    observations = None
    repair_log = None
    if repair_occlusions:
        repair_config = repair_config or RepairConfig()
        observations = read_observations(observation_path)
        tracklets, suppressed, duplicates = find_duplicate_tracklets(
            tracklets, observations, repair_config, use_geometry=cfg.use_geometry)
        repair_log = {"experimental": True, "settings": asdict(repair_config),
                      "suppressed_duplicates": duplicates}
    if matcher == "graph":
        mapping,edges=associate_graph(tracklets,cfg,allow_same_camera_fragments)
    elif matcher == "baseline":
        mapping,edges=associate(tracklets,cfg)
    elif matcher == "adaptive":
        mapping,edges=associate_adaptive(tracklets,cfg)
    else:
        raise ValueError("matcher must be baseline, graph or adaptive")
    if correct_identities:
        if matcher != "graph":
            raise ValueError("identity correction requires --matcher graph")
        mapping,edges,conflicts=rebuild(tracklets,edges,cfg)
        (run_dir/"identity_correction.json").write_text(
            __import__("json").dumps({"conflicts":conflicts,"journal":edges},indent=2),
            encoding="utf-8")
    if memory:
        mapping,records,memory_decisions=apply_memory(
            tracklets,mapping,similarity_threshold=memory_similarity,cfg=cfg,mode=memory_mode)
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
    if memory_state:
        from .time_sync import track_times
        if manifest.get('legacy'):
            raise ValueError('Persistent memory requires a manifest for scene/model isolation')
        scope = json.dumps({k: manifest.get(k) for k in ('scene', 'models', 'feature_dimension')}
                           | {'offsets': cfg.camera_offsets_s, 'fps': cfg.fps}, sort_keys=True)
        mapping, _ = resume_identity_memory(memory_state, tracklets, mapping, scope=scope,
                    now_s=max((track_times(t, cfg.fps, cfg.camera_offsets_s)[1] for t in tracklets), default=0),
                    cfg=cfg, threshold=memory_similarity)
    save_mapping(mapping,run_dir/"global_ids.json")
    (run_dir/"association_edges.json").write_text(
        __import__("json").dumps(edges, indent=2), encoding="utf-8")
    out_csv,out_txt=export_predictions(
        observation_path,run_dir/"global_ids.json",run_dir,
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
    p.add_argument("--model",default=None,help="Reject cache created with another detector")
    p.add_argument("--matcher",choices=["baseline","graph","adaptive"],default="baseline")
    p.add_argument("--memory-mode", choices=["legacy","constrained","gallery"], default="legacy")
    p.add_argument("--split-tracklets", action="store_true")
    p.add_argument("--memory-state", type=Path)
    p.add_argument("--allow-same-camera-fragments",action="store_true")
    p.add_argument("--time",action="store_true",help="Enable time constraints for graph matcher")
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
               require_geometry=a.require_geometry,
               use_geometry=not a.ignore_geometry,
               use_time=True if a.matcher in {"baseline", "adaptive"} else a.time,
               camera_offsets_s=parse_offsets(a.camera_offset),
               transition_model=CameraTransitionModel.from_json(a.topology) if a.topology else None)
    if a.matcher=="graph" and not a.time:
        cfg.temporal_weight=0.0
    tracklets,mapping,edges,out_csv,out_txt=run(
        a.run_dir,cfg,a.scene_id,a.model,a.matcher,a.allow_same_camera_fragments,
        a.memory,a.memory_similarity,a.correct_identities,
        repair_occlusions=a.repair_occlusions,memory_mode=a.memory_mode,
        split_tracklets=a.split_tracklets,memory_state=a.memory_state)
    accepted=sum(edge.get("decision")=="accepted" for edge in edges) if a.matcher!="baseline" else len(edges)
    print(f"Local tracklets: {len(tracklets)}, accepted edges: {accepted}, global IDs: {len(set(mapping.values()))}")
    print("Results:", out_csv, out_txt)
    print("NOTE: person-only approximate 3D boxes are NOT a competitive full Track-1 solution.")

if __name__=="__main__": main()
