import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import networkx as nx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from . import gemma, jobs, mapper, routing, store
from .schemas import (
    CreateBuilding, FrameAnalysis, ManualMerge, MapBundle, MergeDecision, Node, RouteRequest, RouteResponse,
)

api = APIRouter(prefix="/api")


def _load(building_id: str) -> MapBundle:
    try:
        return store.load_map(building_id)
    except (FileNotFoundError, ValueError):
        raise HTTPException(404, f"Unknown building '{building_id}'")


@api.get("/health")
def health():
    ffmpeg = shutil.which("ffmpeg")
    version = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True).stdout.split("\n")[0] if ffmpeg else None
    return {"ffmpeg": version, "gemma": gemma.health()}


# ---------- buildings & map ----------
@api.get("/buildings")
def list_buildings():
    return store.list_buildings()


@api.post("/buildings")
def create_building(body: CreateBuilding):
    return store.create_building(body.name.strip() or "Building")


@api.get("/buildings/{building_id}/map")
def get_map(building_id: str) -> MapBundle:
    return _load(building_id)


@api.put("/buildings/{building_id}/map")
def save_map(building_id: str, bundle: MapBundle):
    _load(building_id)
    bundle.building_id = building_id
    ids = {n.id for n in bundle.nodes}
    bundle.edges = [e for e in bundle.edges if e.from_node in ids and e.to_node in ids and e.from_node != e.to_node]
    mapper.finalize(bundle)
    store.save_map(bundle)
    return {"status": "ok", "map": bundle}


@api.post("/buildings/{building_id}/merges/{merge_id}")
def resolve_merge(building_id: str, merge_id: str, body: MergeDecision):
    with store.lock_for(building_id):
        bundle = _load(building_id)
        s = next((s for s in bundle.suggested_merges if s.id == merge_id), None)
        if s is None:
            raise HTTPException(404, "Unknown merge suggestion")
        if body.accept:
            mapper.merge_nodes(bundle, s.keep, s.remove)
        else:
            bundle.suggested_merges = [x for x in bundle.suggested_merges if x.id != merge_id]
        store.save_map(bundle)
    return bundle


@api.post("/buildings/{building_id}/merge")
def manual_merge(building_id: str, body: ManualMerge):
    with store.lock_for(building_id):
        bundle = _load(building_id)
        ids = {n.id for n in bundle.nodes}
        if body.keep not in ids or body.remove not in ids or body.keep == body.remove:
            raise HTTPException(400, "Pick two different existing nodes")
        mapper.merge_nodes(bundle, body.keep, body.remove)
        store.save_map(bundle)
    return bundle


# ---------- processing ----------
@api.post("/buildings/{building_id}/videos")
async def upload_video(
    building_id: str,
    video: UploadFile = File(...),
    sensors: UploadFile | None = File(None),
    model: str | None = Form(None),
    priority: bool = Form(False),
):
    _load(building_id)
    tmp = Path(tempfile.mkstemp(suffix=Path(video.filename or "v.mp4").suffix)[1])
    with open(tmp, "wb") as f:
        while chunk := await video.read(1 << 20):
            f.write(chunk)
    sensor_data = None
    if sensors is not None:
        try:
            sensor_data = json.loads(await sensors.read())
        except json.JSONDecodeError:
            raise HTTPException(400, "sensors must be JSON")
    job = jobs.submit_video(building_id, tmp, video.filename or "video.mp4", sensor_data, model or None, front=priority)
    if priority:
        jobs.run_next(job.id)
    return {"job_id": job.id, "video_id": job.video_id}


@api.post("/buildings/{building_id}/rebuild")
def rebuild(building_id: str):
    _load(building_id)
    return {"job_id": jobs.submit_rebuild(building_id).id}


@api.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job (server restarted?)")
    return job.public()


@api.get("/buildings/{building_id}/queue")
def job_queue(building_id: str):
    return [j.public() | {"log": j.log[-1:]} for j in jobs.queue(building_id)]


@api.post("/jobs/{job_id}/{action}")
def job_action(job_id: str, action: str):
    """action: pause | resume | run-next (move to front; pauses the running job, which resumes after)."""
    if jobs.get(job_id) is None:
        raise HTTPException(404, "Unknown job")
    fn = {"pause": jobs.pause, "resume": jobs.resume, "run-next": jobs.run_next}.get(action)
    if fn is None:
        raise HTTPException(400, "action must be pause, resume or run-next")
    return fn(job_id).public()


@api.get("/buildings/{building_id}/videos/{video_id}/debug")
def video_debug(building_id: str, video_id: str):
    """Motion track + events + keyframes with Gemma output: the 'what did the AI see' view."""
    _load(building_id)
    vdir = store.video_dir(building_id, video_id)
    track = store.read_json(vdir / "track.json")
    walk = store.read_json(vdir / "walk.json")
    if track is None:
        raise HTTPException(404, "Video not processed yet")
    step = max(1, len(track["t"]) // 600)
    return {
        "source": track.get("source"),
        "t": track["t"][::step], "heading": track["heading"][::step], "flow": track["flow"][::step],
        "events": track.get("events", []),
        "keyframes": [
            {**k, "url": store.frame_url(building_id, k["frame_id"])} for k in (walk or {}).get("keyframes", [])
        ],
    }


@api.post("/buildings/{building_id}/frames/manual")
async def manual_frame(building_id: str, image: UploadFile = File(...), label: str = Form("")):
    """Add a photo by hand (e.g. a sign the video blurred). Gemma labels it; connect it in the editor."""
    bundle = _load(building_id)
    frames = store.frames_dir(building_id)
    frames.mkdir(parents=True, exist_ok=True)
    n = len([p for p in frames.glob("manual_*")])
    frame_id = f"manual_{n:04d}{Path(image.filename or 'x.jpg').suffix.lower() or '.jpg'}"
    (frames / frame_id).write_bytes(await image.read())
    try:
        a = gemma.analyze_frame(frames / frame_id, gemma.cache_for(store.building_dir(building_id)))
    except gemma.GemmaUnavailable:
        a = FrameAnalysis()
    with store.lock_for(building_id):
        bundle = store.load_map(building_id)
        node = Node(
            id=f"n{mapper._next_id('n', [x.id for x in bundle.nodes]):03d}",
            label=label or a.suggested_label or "Manual photo", type=a.place_type, frame_id=frame_id,
            texts=a.texts + a.room_numbers, description=a.description, verified=False,
        )
        bundle.nodes.append(node)
        mapper.finalize(bundle)
        store.save_map(bundle)
    return {"node": node, "analysis": a}


# ---------- navigation ----------
@api.post("/buildings/{building_id}/route")
def route(building_id: str, body: RouteRequest) -> RouteResponse:
    bundle = _load(building_id)
    try:
        return routing.route(bundle, body.start_node_id, body.target_node_id)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown node {exc}")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        raise HTTPException(400, routing.NO_PATH_MESSAGE)
