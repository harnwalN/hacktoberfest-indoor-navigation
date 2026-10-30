"""File-based persistence: one portable "map bundle" directory per building.

data/buildings/<id>/
    map.json            the navigable graph (nodes, edges, merges)
    gemma_cache.json    Gemma answers keyed by image hash + prompt version
    frames/*.jpg        keyframes shown to users
    videos/<vid>/       normalized videos, track.json, walk.json (pipeline outputs)
"""
import json
import os
import re
import tempfile
import threading
from pathlib import Path

from . import config
from .schemas import MapBundle

_locks: dict[str, threading.RLock] = {}
_locks_guard = threading.Lock()


def lock_for(building_id: str) -> threading.RLock:
    with _locks_guard:
        return _locks.setdefault(building_id, threading.RLock())


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "building"


def building_dir(building_id: str) -> Path:
    if not re.fullmatch(r"[a-z0-9-]+", building_id):
        raise ValueError(f"invalid building id: {building_id!r}")
    return config.BUILDINGS_DIR / building_id


def frames_dir(building_id: str) -> Path:
    return building_dir(building_id) / "frames"


def video_dir(building_id: str, video_id: str) -> Path:
    # Generated ids are "v" + HHMMSS + 4 hex chars; plain alphanumerics keep "..", separators and glob chars out.
    if not re.fullmatch(r"[A-Za-z0-9]{1,64}", video_id or ""):
        raise ValueError(f"invalid video id: {video_id!r}")
    return building_dir(building_id) / "videos" / video_id


def frame_url(building_id: str, frame_id: str | None) -> str | None:
    return f"/bfiles/{building_id}/frames/{frame_id}" if frame_id else None


def write_json_atomic(path: Path, data, mkdir: bool = True) -> None:
    """mkdir=False: fail (FileNotFoundError) if the directory is gone, e.g. a video deleted mid-processing."""
    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def list_buildings() -> list[dict]:
    if not config.BUILDINGS_DIR.exists():
        return []
    out = []
    for d in sorted(config.BUILDINGS_DIR.iterdir()):
        if (d / "map.json").exists():
            m = load_map(d.name)
            out.append({"id": m.building_id, "name": m.name, "nodes": len(m.nodes), "edges": len(m.edges)})
    return out


def create_building(name: str) -> MapBundle:
    base = slugify(name)
    building_id, n = base, 2
    while (config.BUILDINGS_DIR / building_id).exists():
        building_id, n = f"{base}-{n}", n + 1
    bundle = MapBundle(building_id=building_id, name=name)
    frames_dir(building_id).mkdir(parents=True, exist_ok=True)
    save_map(bundle)
    return bundle


def load_map(building_id: str) -> MapBundle:
    data = read_json(building_dir(building_id) / "map.json")
    if data is None:
        raise FileNotFoundError(building_id)
    return MapBundle.model_validate(data)


def save_map(bundle: MapBundle) -> None:
    with lock_for(bundle.building_id):
        write_json_atomic(building_dir(bundle.building_id) / "map.json", bundle.model_dump())
