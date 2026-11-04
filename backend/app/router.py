import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import cv2
import networkx as nx
import numpy as np
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Path as FloorPath, Query, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, ImageOps

from . import auth, config, gemma, jobs, mapper, moderation, ocr, reports, routing, store
from .schemas import (
    CreateBuilding, EdgeOut, FloorMapOut, FrameAnalysis, ManualMerge, MapBundle, MapEdit, MapOut, MergeDecision,
    Account, Node, NodeOut, ReportCreate, ReportStatusUpdate, RouteRequest, RouteResponse, VideoPublic, VideoUpdate,
)

log = logging.getLogger(__name__)
api = APIRouter(prefix="/api")


def _load(building_id: str) -> MapBundle:
    try:
        return store.load_map(building_id)
    except (FileNotFoundError, ValueError):
        raise HTTPException(404, f"Unknown building '{building_id}'")


def _map_out(bundle: MapBundle) -> MapOut:
    """The map as an API response: each photo reference (node frame_id, edge frames_fwd/frames_rev) also comes
    with its URL, built now. The bundle is not changed, so none of this reaches map.json."""
    bid = bundle.building_id
    names = {n.frame_id for n in bundle.nodes} | {f for e in bundle.edges for f in e.frames_fwd + (e.frames_rev or [])}
    names.discard(None)
    ordered = list(names)
    urls = {f: store.frame_url(bid, f) for f in ordered}
    nodes = [NodeOut(**n.model_dump(), frame_url=urls.get(n.frame_id)) for n in bundle.nodes]
    edges = [EdgeOut(**e.model_dump(), frames_fwd_urls=[urls[f] for f in e.frames_fwd],
                     frames_rev_urls=None if e.frames_rev is None else [urls[f] for f in e.frames_rev])
             for e in bundle.edges]
    # Walks without uploader_email or moderated_by: a public response must not carry anyone's email.
    videos = [VideoPublic(**v.model_dump(exclude={"uploader_email", "moderated_by"})) for v in bundle.videos]
    return MapOut(**bundle.model_dump(exclude={"nodes", "edges", "videos"}), nodes=nodes, edges=edges, videos=videos)


def _node_out(building_id: str, node: Node) -> NodeOut:
    return NodeOut(**node.model_dump(), frame_url=store.frame_url(building_id, node.frame_id))


_CHUNK = 1 << 20


async def _copy_capped(upload: UploadFile, f, limit: int, what: str) -> None:
    """Stream an upload into an open file, refusing with 413 once it exceeds `limit` bytes."""
    total = 0
    while chunk := await upload.read(_CHUNK):
        total += len(chunk)
        if total > limit:
            raise HTTPException(413, f"{what} is too large (limit {limit} bytes)")
        f.write(chunk)


async def _read_capped(upload: UploadFile, limit: int, what: str) -> bytes:
    buf = io.BytesIO()
    await _copy_capped(upload, buf, limit, what)
    return buf.getvalue()


# PIL format -> extension of the stored frame (the client's filename is never trusted).
_IMAGE_EXT = {"JPEG": ".jpg", "MPO": ".jpg", "PNG": ".png", "WEBP": ".webp", "GIF": ".gif", "BMP": ".bmp"}
_BAD_IMAGE = "image must be a valid JPEG, PNG, WebP, GIF or BMP file"


def _image_ext(data: bytes) -> str:
    """File extension for an uploaded image; 400 if it is not a decodable image of an allowed format."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            im.load()  # verify() misses truncated pixel data for some formats
    except Exception:  # PIL raises many types: UnidentifiedImageError, OSError, SyntaxError, DecompressionBombError
        raise HTTPException(400, _BAD_IMAGE)
    if fmt not in _IMAGE_EXT:
        raise HTTPException(400, _BAD_IMAGE)
    return _IMAGE_EXT[fmt]


@api.get("/health")
def health():
    ffmpeg = shutil.which("ffmpeg")
    version = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True).stdout.split("\n")[0] if ffmpeg else None
    return {"ffmpeg": version, "gemma": gemma.health(), "hosted": gemma.hosted_health(),
            "vision_provider": gemma.vision_provider(), "ocr": {"engine": ocr.engine(), "available": ocr.available()}}


# ---------- buildings & map ----------
@api.get("/buildings")
def list_buildings(authorization: str | None = Header(None),
                   x_owner_token: str | None = Header(None, alias="X-Owner-Token")):
    """Everyone sees buildings that have a map (nodes > 0). Owners see every building, including empty ones."""
    rows = store.list_buildings()
    if not auth.caller_is_owner(authorization, x_owner_token):
        rows = [r for r in rows if r["nodes"] > 0]
    return rows


@api.post("/buildings")
def create_building(body: CreateBuilding, account: Account | None = Depends(auth.require_owner)):
    """Owner only. The owner who creates a building becomes its moderator (none when the legacy token is used)."""
    return store.create_building(body.name.strip() or "Building", creator_id=account.id if account else None)


@api.delete("/buildings/{building_id}", dependencies=[Depends(auth.require_owner)])
def delete_building(building_id: str):
    """Owner only. Removes the building's map, walks, photos and reports, and stops its queued jobs. 204 on success."""
    _load(building_id)
    jobs.cancel_building(building_id)
    try:
        store.delete_building(building_id)
    except (KeyError, ValueError):
        raise HTTPException(404, f"Unknown building '{building_id}'")
    return Response(status_code=204)


@api.get("/buildings/{building_id}/map")
def get_map(building_id: str) -> MapOut:
    return _map_out(_load(building_id))


@api.get("/buildings/{building_id}/floors/{floor}/map")
def get_floor_map(building_id: str, floor: int = FloorPath(ge=1, le=config.MAX_FLOOR)) -> FloorMapOut:
    """One floor of the map: its places, the connections between them, and the links to other floors.
    A floor with no places is an empty map, not a 404. Floors run 1 to MAX_FLOOR, as in the video upload."""
    bundle = _load(building_id)
    bid = bundle.building_id
    floor_of = {n.id: n.floor for n in bundle.nodes}
    nodes = [_node_out(bid, n) for n in bundle.nodes if n.floor == floor]
    edges, links = [], []
    for e in bundle.edges:
        ends = [floor_of.get(e.from_node), floor_of.get(e.to_node)]
        if floor not in ends:
            continue
        out = EdgeOut(**e.model_dump(), frames_fwd_urls=[store.frame_url(bid, f) for f in e.frames_fwd],
                      frames_rev_urls=None if e.frames_rev is None
                      else [store.frame_url(bid, f) for f in e.frames_rev])
        (edges if ends[0] == ends[1] == floor else links).append(out)
    return FloorMapOut(building_id=bid, floor=floor, nodes=nodes, edges=edges, links=links)


@api.put("/buildings/{building_id}/map", dependencies=[Depends(auth.require_owner)])
def save_map(building_id: str, body: MapEdit):
    """Save the editor's places and connections. Sign-in required.

    Only `nodes` and `edges` are taken from the request, and both are required (a body missing either is a
    422, so a partial body cannot wipe the map; an explicit empty list is allowed). Everything else (videos,
    processed_with, name, suggested merges) is owned by the server and read from disk, so a stale editor copy cannot reset
    walk statuses or persist bad video ids. Load + apply + save happen under the building lock, so the
    save serializes with background jobs instead of overwriting their video/merge bookkeeping.
    A `frame_url` sent back in the nodes is ignored: photo URLs are built by the server for each response.
    A node's `contributor_id` is server-owned too: each place keeps the value already on disk (none for a new place).
    """
    with store.lock_for(building_id):
        bundle = _load(building_id)
        owners = {n.id: n.contributor_id for n in bundle.nodes}
        for n in body.nodes:
            n.contributor_id = owners.get(n.id)
        bundle.nodes = body.nodes
        ids = {n.id for n in bundle.nodes}
        bundle.edges = [e for e in body.edges if e.from_node in ids and e.to_node in ids and e.from_node != e.to_node]
        bundle.suggested_merges = [s for s in bundle.suggested_merges if s.keep in ids and s.remove in ids]
        mapper.finalize(bundle)
        store.save_map(bundle)  # re-takes the same RLock: fine
        out = _map_out(bundle)
    return {"status": "ok", "map": out}


@api.post("/buildings/{building_id}/merges/{merge_id}", dependencies=[Depends(auth.require_owner)])
def resolve_merge(building_id: str, merge_id: str, body: MergeDecision) -> MapOut:
    with store.lock_for(building_id):
        bundle = _load(building_id)
        s = next((s for s in bundle.suggested_merges if s.id == merge_id), None)
        if s is None:
            raise HTTPException(404, "Unknown merge suggestion")
        if body.accept:
            ids = {n.id for n in bundle.nodes}
            if (gone := [i for i in (s.keep, s.remove) if i not in ids]) or s.keep == s.remove:
                # The map changed since this was suggested (a place was deleted in the editor or merged away):
                # drop the stale suggestion so it stops showing, and say why nothing was merged.
                bundle.suggested_merges = [x for x in bundle.suggested_merges if x.id != merge_id]
                store.save_map(bundle)
                what = f"place {', '.join(gone)} no longer exists" if gone else "both sides are the same place"
                raise HTTPException(409, f"Merge suggestion {merge_id} is out of date ({what}); it has been "
                                         "removed. Reload the map.")
            _merge_or_400(bundle, s.keep, s.remove)
        else:
            bundle.suggested_merges = [x for x in bundle.suggested_merges if x.id != merge_id]
        store.save_map(bundle)
        return _map_out(bundle)


@api.post("/buildings/{building_id}/merge", dependencies=[Depends(auth.require_owner)])
def manual_merge(building_id: str, body: ManualMerge) -> MapOut:
    with store.lock_for(building_id):
        bundle = _load(building_id)
        ids = {n.id for n in bundle.nodes}
        if body.keep not in ids or body.remove not in ids or body.keep == body.remove:
            raise HTTPException(400, "Pick two different existing nodes")
        _merge_or_400(bundle, body.keep, body.remove)
        store.save_map(bundle)
        return _map_out(bundle)


def _merge_or_400(bundle: MapBundle, keep: str, remove: str) -> None:
    """merge_nodes refuses (before changing anything) to merge places on two different known floors."""
    try:
        mapper.merge_nodes(bundle, keep, remove)
    except ValueError as exc:
        raise HTTPException(400, f"Cannot merge: {exc}")


# ---------- processing ----------
@api.post("/buildings/{building_id}/videos")
async def upload_video(
    building_id: str,
    video: UploadFile = File(...),
    sensors: UploadFile | None = File(None),
    model: str | None = Form(None),
    priority: bool = Form(False),
    start_floor: int | None = Form(None, ge=1, le=config.MAX_FLOOR),
    account: Account = Depends(auth.require_contributor),
):
    """Signed in and not blocked (403 when blocked). The walk is recorded under the uploader's account (contributor_id,
    uploader_email) and goes through moderation."""
    _load(building_id)
    fd, tmp_name = tempfile.mkstemp(suffix=Path(video.filename or "v.mp4").suffix)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            await _copy_capped(video, f, config.MAX_UPLOAD_MB * 1024 * 1024, "video")
        sensor_data = None
        if sensors is not None:
            raw = await _read_capped(sensors, config.MAX_SENSOR_BYTES, "sensors")
            try:
                sensor_data = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise HTTPException(400, "sensors must be JSON")
        job = jobs.submit_video(building_id, tmp, video.filename or "video.mp4", sensor_data, model or None,
                               front=priority, start_floor=start_floor, contributor_id=account.id,
                               uploader_email=account.email)
    except BaseException:
        tmp.unlink(missing_ok=True)  # on success submit_video has moved it; this only drops partial/rejected uploads
        raise
    if priority:
        jobs.run_next(job.id)
    return {"job_id": job.id, "video_id": job.video_id}


@api.patch("/buildings/{building_id}/videos/{video_id}", dependencies=[Depends(auth.require_owner)])
def update_video(building_id: str, video_id: str, body: VideoUpdate):
    """Set the floor a walk starts on. Places from an already-processed walk move instantly. Sign-in required."""
    with store.lock_for(building_id):
        bundle = _load(building_id)
        v = next((v for v in bundle.videos if v.id == video_id), None)
        if v is None:
            raise HTTPException(404, "Unknown video")
        v.start_floor = body.start_floor
        if v.status == "done":
            mapper.set_walk_floor(bundle, video_id, body.start_floor)  # also updates v.floor_offset
            mapper.finalize(bundle)  # re-lay out the plan around any new stairs/elevator links
        store.save_map(bundle)
    return v


@api.delete("/buildings/{building_id}/videos/{video_id}", dependencies=[Depends(auth.require_owner)])
def delete_video(building_id: str, video_id: str):
    """Remove a video: cancels its processing, drops its places/connections, frames and files.
    Places that other videos also pass through are kept. Sign-in required."""
    _load(building_id)
    try:
        store.video_dir(building_id, video_id)  # reject unsafe ids before anything is modified
    except ValueError:
        raise HTTPException(404, "Unknown video")
    try:
        return jobs.delete_video(building_id, video_id)
    except KeyError:
        raise HTTPException(404, "Unknown video")


@api.post("/buildings/{building_id}/consolidate", dependencies=[Depends(auth.require_owner)])
def consolidate(building_id: str):
    """Link all walks into one map now (runs next in the queue; fast, Gemma answers are cached). Sign-in required."""
    _load(building_id)
    return {"job_id": jobs.submit_consolidate(building_id).id}


@api.post("/buildings/{building_id}/rebuild", dependencies=[Depends(auth.require_owner)])
def rebuild(building_id: str):
    """Re-process every finished walk and rebuild the map from scratch. Sign-in required."""
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


@api.post("/jobs/{job_id}/{action}", dependencies=[Depends(auth.require_owner)])
def job_action(job_id: str, action: str):
    """action: pause | resume | run-next (move to front; pauses the running job, which resumes after). Sign-in required."""
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
    try:
        vdir = store.video_dir(building_id, video_id)
    except ValueError:
        raise HTTPException(404, "Unknown video")
    track = store.read_json(vdir / "track.json")
    walk = store.read_json(vdir / "walk.json")
    if track is None:
        raise HTTPException(404, "Video not processed yet")
    # Only an approved walk's frames are public; for any other walk the url is null (its frames are not served).
    video = store.get_video(building_id, video_id)
    public = video is not None and video.moderation_status == "approved"
    step = max(1, len(track["t"]) // 600)
    return {
        "source": track.get("source"),
        "t": track["t"][::step], "heading": track["heading"][::step], "flow": track["flow"][::step],
        "events": track.get("events", []),
        "keyframes": [
            {**k, "url": store.frame_url(building_id, k["frame_id"]) if public else None}
            for k in (walk or {}).get("keyframes", [])
        ],
    }


_PHOTO_CHECK_FAILED = ("The photo could not be checked (face blur or content check) and was not added. "
                       "Try again later.")


def _moderate_photo(data: bytes) -> bytes:
    """One uploaded photo through the walk's moderation (moderation.moderate_keyframe, as jobs.py does per keyframe):
    upright, faces blurred, then content checked. Returns the blurred photo as JPEG; the original bytes are never kept.
    Fails closed. 400: not a decodable image. 422: the content check rejected it. 503: the check could not run."""
    _image_ext(data)  # 400 unless a decodable image of an allowed format
    with Image.open(io.BytesIO(data)) as im:
        upright = ImageOps.exif_transpose(im).convert("RGB")  # EXIF rotation applied; the stored copy has no EXIF
        img = cv2.cvtColor(np.asarray(upright), cv2.COLOR_RGB2BGR)
    try:
        blurred, verdict = moderation.moderate_keyframe(img)
    except Exception:  # FaceBlurError (no detector / model) or anything else: nothing is stored
        log.exception("manual photo: face blur or content check could not run")
        raise HTTPException(503, _PHOTO_CHECK_FAILED)
    if verdict.error:
        raise HTTPException(503, _PHOTO_CHECK_FAILED)
    if not verdict.approved:
        raise HTTPException(422, f"Photo rejected by moderation: {verdict.reason}. It was not added.")
    ok, encoded = cv2.imencode(".jpg", blurred, [cv2.IMWRITE_JPEG_QUALITY, jobs.JPEG_QUALITY])
    if not ok:
        raise HTTPException(503, _PHOTO_CHECK_FAILED)
    return encoded.tobytes()


@api.post("/buildings/{building_id}/frames/manual")
async def manual_frame(building_id: str, image: UploadFile = File(...), label: str = Form(""),
                       account: Account | None = Depends(auth.require_owner)):
    """Add a photo by hand (e.g. a sign the video blurred). Sign-in required. The photo gets the same moderation as a
    walk's keyframes (faces blurred, then content checked) before anything is stored: a rejected photo is not added
    (422 with the reason, or 503 when the check could not run). The stored copy is the blurred JPEG and the new place
    records the owner's account (contributor_id; none when the legacy token is used). Gemma labels it; connect it in the editor.
    Owner only."""
    _load(building_id)
    frames = store.frames_dir(building_id)
    frames.mkdir(parents=True, exist_ok=True)
    data = await _read_capped(image, config.MAX_IMAGE_BYTES, "image")
    jpeg = await run_in_threadpool(_moderate_photo, data)
    with store.lock_for(building_id):
        n = len([p for p in frames.glob("manual_*")])
        frame_id = f"manual_{n:04d}.jpg"
        (frames / frame_id).write_bytes(jpeg)
    warning = None
    try:
        a = gemma.analyze_frame(frames / frame_id, gemma.cache_for(store.building_dir(building_id)))
    except gemma.GemmaUnavailable as exc:
        a, warning = FrameAnalysis(), f"Gemma unavailable, photo not analysed: {exc}"
    except Exception as exc:  # unreadable image, OCR failure, bug: keep the photo, label it by hand
        log.exception("manual frame %s: analysis failed", frame_id)
        a, warning = FrameAnalysis(), f"Photo could not be analysed ({type(exc).__name__}); label it by hand"
    with store.lock_for(building_id):
        bundle = store.load_map(building_id)
        node = Node(
            id=f"n{mapper._next_id('n', [x.id for x in bundle.nodes]):03d}",
            label=label or a.suggested_label or "Manual photo", type=a.place_type, frame_id=frame_id,
            texts=a.texts + a.room_numbers, description=a.description, verified=False,
            contributor_id=account.id if account else None,
        )
        bundle.nodes.append(node)
        mapper.finalize(bundle)
        store.save_map(bundle)
    return {"node": _node_out(building_id, node), "analysis": a, **({"warning": warning} if warning else {})}


# ---------- navigation ----------
@api.post("/buildings/{building_id}/route")
def route(building_id: str, body: RouteRequest) -> RouteResponse:
    bundle = _load(building_id)
    try:
        return routing.route(bundle, body.start_node_id, body.target_node_id, avoid_stairs=body.avoid_stairs)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown node {exc}")
    except routing.NoStepFreePath as exc:
        raise HTTPException(400, str(exc))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        raise HTTPException(400, routing.NO_PATH_MESSAGE)


# ---------- problem reports (reports.py) ----------
def _refused(exc: reports.Refused) -> HTTPException:
    return HTTPException(exc.status, exc.detail)


def _with_labels(bundle: MapBundle, items: list[dict]) -> list[dict]:
    """Adds node_label: the label of node_id on the current map, or null if that place no longer exists."""
    labels = {n.id: n.label for n in bundle.nodes}
    return [{**item, "node_label": labels.get(item["node_id"]) if item["node_id"] else None} for item in items]


@api.post("/buildings/{building_id}/reports")
def create_report(building_id: str, body: ReportCreate):
    """Flag a wrong step or place. Anyone may send one. node_id, from_node and to_node must exist on the current
    map. 507 once the building holds REPORTS_MAX reports."""
    with store.lock_for(building_id):
        bundle = _load(building_id)
        ids = {n.id for n in bundle.nodes}
        for field in ("node_id", "from_node", "to_node"):
            if getattr(body, field) is not None and getattr(body, field) not in ids:
                raise HTTPException(422, f"{field}: no place with that id on this map")
        fields = {"kind": body.kind, "note": body.note, "node_id": body.node_id, "step_index": body.step_index,
                  "from_node": body.from_node, "to_node": body.to_node}
        try:
            return reports.add(building_id, fields)
        except reports.Refused as exc:
            raise _refused(exc)


@api.get("/buildings/{building_id}/reports")
def list_reports(building_id: str, status: Literal["open", "resolved", "all"] = "open",
                 limit: int = Query(100, ge=1, le=500)):
    """Newest first. status: open (default), resolved or all. Each item has its full history."""
    bundle = _load(building_id)
    return _with_labels(bundle, reports.list_reports(building_id, status, limit))


@api.patch("/buildings/{building_id}/reports/{report_id}", dependencies=[Depends(auth.require_owner)])
def update_report(building_id: str, report_id: str, body: ReportStatusUpdate):
    """Resolve or reopen a report. Owner only. Appends a status event (the history is kept). resolution_note is
    kept only when resolving; reopening clears it."""
    bundle = _load(building_id)
    item = reports.set_status(building_id, report_id, body.status, body.resolution_note)
    if item is None:
        raise HTTPException(404, "Unknown report")
    return _with_labels(bundle, [item])[0]
