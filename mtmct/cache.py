"""Small JSON manifest for reusable tracking/ReID artifacts."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

SCHEMA_VERSION = "mtmct-cache-v1"


def write_config(run_dir: str | Path, config: dict) -> Path:
    """JSON is valid YAML 1.2 and keeps this dependency-free."""
    path = Path(run_dir) / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def fingerprint(path: str | Path) -> dict:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "sha256": digest.hexdigest()}


def _git_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def write_manifest(run_dir: str | Path, *, scene: str, cameras: list[int],
                   inputs: list[str | Path], detector: str, reid: str,
                   feature_dimension: int | None, config: dict) -> Path:
    run_dir = Path(run_dir)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "git_sha": _git_sha(),
        "scene": scene,
        "cameras": sorted(cameras),
        "inputs": [fingerprint(path) for path in inputs if Path(path).is_file()],
        "models": {
            "detector": {"name": detector, **(fingerprint(detector) if Path(detector).is_file() else {})},
            "reid": {"name": reid, **(fingerprint(reid) if Path(reid).is_file() else {})},
        },
        "feature_dimension": feature_dimension,
        "tracking_config": config,
    }
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def validate_cache(run_dir: str | Path, expected_detector: str | None = None) -> dict:
    run_dir = Path(run_dir)
    path = run_dir / "manifest.json"
    if not path.exists():
        return {"legacy": True, "warning": "manifest.json missing; cache provenance cannot be verified"}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported cache manifest: {manifest.get('schema_version')!r}")
    detector = manifest.get("models", {}).get("detector", {}).get("name")
    if expected_detector and detector != expected_detector:
        raise ValueError(f"Cache detector mismatch: stored {detector!r}, requested {expected_detector!r}")
    for stored in manifest.get("inputs", []):
        input_path = Path(stored["path"])
        if input_path.exists() and fingerprint(input_path)["sha256"] != stored["sha256"]:
            raise ValueError(f"Stale cache: input changed: {input_path}")

    tracklets = json.loads((run_dir / "local_tracklets.json").read_text(encoding="utf-8"))
    dimensions = {len(row.get("embedding", [])) for row in tracklets if row.get("embedding")}
    if len(dimensions) > 1:
        raise ValueError(f"Inconsistent feature dimensions in cache: {sorted(dimensions)}")
    actual_dimension = next(iter(dimensions), None)
    if actual_dimension != manifest.get("feature_dimension"):
        raise ValueError(
            f"Cache feature dimension mismatch: stored {manifest.get('feature_dimension')}, actual {actual_dimension}"
        )
    return manifest
