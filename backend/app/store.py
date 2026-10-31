"""File-based persistence: one portable "map bundle" directory per building.

data/buildings/<id>/
    map.json            the navigable graph (nodes, edges, merges)
    gemma_cache.jsonl   Gemma answers, one JSON line per answer (append-only; a legacy gemma_cache.json is migrated to .bak)
    frames/*.jpg        keyframes shown to users
    videos/<vid>/       normalized videos, track.json, walk.json (pipeline outputs)
    access.json         invite tokens (hashes only, never served; see auth.py)
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


MAX_BUILDING_ID = 64  # longer ids are refused here, before any filesystem call (a 5000-char id gave ENAMETOOLONG)


def building_dir(building_id: str) -> Path:
    if not isinstance(building_id, str) or len(building_id) > MAX_BUILDING_ID \
            or not re.fullmatch(r"[a-z0-9-]+", building_id):
        # The id itself is not echoed: it may be a few thousand characters long.
        raise ValueError(f"invalid building id (1-{MAX_BUILDING_ID} characters of a-z, 0-9, -)")
    return config.BUILDINGS_DIR / building_id


def frames_dir(building_id: str) -> Path:
    return building_dir(building_id) / "frames"


def video_dir(building_id: str, video_id: str) -> Path:
    # Generated ids are "v" + HHMMSS + 4 hex chars; plain alphanumerics keep "..", separators and glob chars out.
    if not re.fullmatch(r"[A-Za-z0-9]{1,64}", video_id or ""):
        raise ValueError(f"invalid video id: {video_id!r}")
    return building_dir(building_id) / "videos" / video_id


def frame_url(building_id: str, frame_id: str | None) -> str | None:
    """Browser URL of a keyframe: signed and expiring with access control on, the plain path with it off.
    Built at response time, never stored (see auth.frame_urls)."""
    from . import auth  # local import: auth imports store

    return auth.frame_urls(building_id, [frame_id])[0]


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


def list_buildings(only: set[str] | None = None) -> list[dict]:
    """Summaries of stored buildings; `only` limits the result to these folder ids."""
    if not config.BUILDINGS_DIR.exists():
        return []
    out = []
    for d in sorted(config.BUILDINGS_DIR.iterdir()):
        if only is not None and d.name not in only:
            continue
        if (d / "map.json").exists():
            m = load_map(d.name)
            out.append({"id": m.building_id, "name": m.name, "nodes": len(m.nodes), "edges": len(m.edges)})
    return out


def create_building(name: str) -> MapBundle:
    # Leave room for the "-2", "-3"... suffix so the id stays within MAX_BUILDING_ID.
    base = slugify(name)[:MAX_BUILDING_ID - 12].strip("-") or "building"
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


# Signed photo URLs exist only in API responses (schemas.MapOut). Never write them to map.json, even if a
# response-model object is handed in by mistake.
_RESPONSE_ONLY_FIELDS = {"nodes": {"__all__": {"frame_url"}},
                         "edges": {"__all__": {"frames_fwd_urls", "frames_rev_urls"}}}


def save_map(bundle: MapBundle) -> None:
    with lock_for(bundle.building_id):
        write_json_atomic(building_dir(bundle.building_id) / "map.json",
                          bundle.model_dump(exclude=_RESPONSE_ONLY_FIELDS))
