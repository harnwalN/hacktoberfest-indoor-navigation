"""Owner moderation and contributor views: the logic behind owner_router.py.

Map membership rule: a walk is on the map when its status is "done" and its moderation_status is "approved".
- approve: the walk is put on the map. A walk already processed is rebuilt in (rebuild job, which re-blurs its frames
  without a new content check). A walk that never reached the map is processed again from its source (reprocess job).
- reject / hide: the walk comes off the map at once. Its frames that no other walk uses move to the walk's own archive
  folder, out of the public frames folder. The processed data is kept, so a later approve can bring it back.
Owner decisions are sticky: the pipeline's automatic moderation no longer changes them (store.set_auto_moderation).
"""
import re
from pathlib import Path

from fastapi import HTTPException

from . import auth, config, jobs, mapper, store
from .schemas import MapBundle, VideoInfo

MAX_PREVIEW_FRAMES = 6
_FRAME_NAME = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]*")  # one path segment, never "." or ".."
UNKNOWN_EMAIL = "unknown"


def _when(value: str | None) -> str:
    """Sort key for newest first: ISO strings compare correctly; a missing time sorts last."""
    return value or ""


def _preview_names(building_id: str, video: VideoInfo) -> list[str]:
    """Up to MAX_PREVIEW_FRAMES keyframe file names of a walk, spread evenly over the walk (from its walk.json)."""
    try:
        walk = store.read_json(store.video_dir(building_id, video.id) / "walk.json") or {}
    except ValueError:
        return []
    names = [k.get("frame_id") for k in walk.get("keyframes", []) if isinstance(k, dict)]
    names = [n for n in names if isinstance(n, str) and _FRAME_NAME.fullmatch(n)]
    if len(names) <= MAX_PREVIEW_FRAMES:
        return names
    step = (len(names) - 1) / (MAX_PREVIEW_FRAMES - 1)
    return [names[round(i * step)] for i in range(MAX_PREVIEW_FRAMES)]


def frame_urls(bundle: MapBundle, video: VideoInfo) -> list[str]:
    """Preview image URLs for one walk. A published walk uses its public /bfiles URL. Any other walk (pending, rejected,
    hidden) gets a signed owner link, because its frames are not public. Files that exist nowhere are left out."""
    bid = bundle.building_id
    urls = []
    for name in _preview_names(bid, video):
        public = store.frames_dir(bid) / name
        archived = store.held_frames_dir(bid, video.id) / name
        if video.moderation_status == "approved" and public.is_file():
            urls.append(store.frame_url(bid, name))
        elif public.is_file() or archived.is_file():
            urls.append(auth.frame_link(bid, video.id, name))
    return urls


def frame_file_path(building_id: str, video_id: str, filename: str) -> Path | None:
    """Where a moderation frame is stored now: the walk's public frames folder, or its held folder. None when absent."""
    if not _FRAME_NAME.fullmatch(filename):
        return None
    try:
        held = store.held_frames_dir(building_id, video_id)
        public = store.frames_dir(building_id) / filename
    except ValueError:
        return None
    for path in (public, held / filename):
        if path.is_file():
            return path
    return None


def moderation_items(status: str) -> list[dict]:
    """Walks across all buildings with this moderation status ("all" for every walk), newest first."""
    out = []
    for bundle in store.all_bundles():
        for v in bundle.videos:
            if status != "all" and v.moderation_status != status:
                continue
            out.append(_moderation_item(bundle, v))
    out.sort(key=lambda item: _when(item["created_at"]), reverse=True)
    return out


def _moderation_item(bundle: MapBundle, v: VideoInfo) -> dict:
    return {
        "building_id": bundle.building_id, "building_name": bundle.name, "video_id": v.id, "filename": v.filename,
        "uploader_email": v.uploader_email or UNKNOWN_EMAIL, "status": v.moderation_status,
        "reason": v.moderation_notes, "frames": frame_urls(bundle, v), "created_at": v.created_at,
    }


def _find(bundle: MapBundle, video_id: str) -> VideoInfo:
    v = next((v for v in bundle.videos if v.id == video_id), None)
    if v is None:
        raise HTTPException(404, "Unknown video")
    return v


def _on_map(v: VideoInfo) -> bool:
    return v.status == "done" and v.moderation_status == "approved"


def decide(building_id: str, video_id: str, action: str, note: str | None, moderator: str) -> tuple[dict, str | None]:
    """Apply an owner's approve, reject or hide. Returns (updated moderation item, job id or None).
    404: unknown building or walk. 409: a walk being processed right now cannot be approved yet, and a walk that has
    no source file cannot be processed again."""
    try:
        with store.lock_for(building_id):
            bundle = store.load_map(building_id)
            v = _find(bundle, video_id)
            was_on_map = _on_map(v)
            if action == "approve":
                if v.status == "processing":
                    raise HTTPException(409, "This walk is being processed. Approve it when processing has finished.")
                job_kind = None
                vdir = store.video_dir(building_id, video_id)
                if not was_on_map and v.status == "done" and (vdir / "analysis.mp4").exists():
                    job_kind = "rebuild"  # processed walk, off the map: rebuild puts it back
                elif not was_on_map and v.status not in ("queued", "paused"):
                    # never processed (failed, or decided before processing): process again from the source file
                    if next(iter(vdir.glob("source.*")), None) is None:
                        raise HTTPException(409, "The walk's source file is gone. It has to be uploaded again.")
                    job_kind = "reprocess"
                # queued or paused: the walk's own job runs later and sees the approval
                v.moderation_status, v.moderation_notes, v.moderated_by = "approved", note, moderator
                store.save_map(bundle)
                jobs._publish_walk_frames(building_id, video_id)  # its blurred frames leave the held folder
            elif action in ("reject", "hide"):
                keep = jobs.frames_kept_without(bundle, video_id)  # frames other walks' places still use stay public
                v.moderation_status = "rejected" if action == "reject" else "hidden"
                v.moderation_notes, v.moderated_by = note, moderator
                job_kind = None
                if was_on_map:
                    mapper.remove_video(bundle, video_id)  # also drops the walk's entry from bundle.videos
                    bundle.videos.append(v)
                store.save_map(bundle)
                jobs._archive_walk_frames(building_id, video_id, keep=keep)  # its frames leave the public folder
            else:  # pragma: no cover - the schema allows only the three actions
                raise HTTPException(422, "action must be approve, reject or hide")
            item = _moderation_item(bundle, v)
    except (FileNotFoundError, ValueError):  # unknown building, or an id that cannot be a building or walk folder
        raise HTTPException(404, "Unknown building or walk")
    job_id = None
    if action == "approve" and job_kind == "rebuild":
        job_id = jobs.submit_rebuild(building_id).id
    elif action == "approve" and job_kind == "reprocess":
        try:
            job_id = jobs.submit_reprocess(building_id, video_id).id
        except FileNotFoundError:
            raise HTTPException(409, "The walk's source file is gone. It has to be uploaded again.")
    return item, job_id


def uploader_status(v: VideoInfo) -> str:
    """What the uploader sees: processing | live | needs_review | rejected."""
    if v.moderation_status in ("rejected", "hidden"):
        return "rejected"
    if v.status == "done" and v.moderation_status == "approved":
        return "live"
    if v.status == "error":
        return "needs_review"
    return "processing"


def my_uploads(account) -> list[dict]:
    """The walks this account uploaded, across all buildings, newest first."""
    out = []
    for bundle in store.all_bundles():
        for v in bundle.videos:
            if v.contributor_id != account.id:
                continue
            status = uploader_status(v)
            out.append({
                "building_id": bundle.building_id, "building_name": bundle.name, "video_id": v.id,
                "filename": v.filename, "status": status,
                "reason": v.moderation_notes if status == "rejected" else None,
                "created_at": v.created_at,
            })
    out.sort(key=lambda item: _when(item["created_at"]), reverse=True)
    return out


def users() -> list[dict]:
    """Every contributor account with counts of their walks. live = on the map; rejected = rejected or hidden."""
    counts: dict[str, dict] = {}
    for bundle in store.all_bundles():
        for v in bundle.videos:
            if not v.contributor_id:
                continue
            c = counts.setdefault(v.contributor_id, {"uploads": 0, "live": 0, "rejected": 0})
            c["uploads"] += 1
            if _on_map(v):
                c["live"] += 1
            if v.moderation_status in ("rejected", "hidden"):
                c["rejected"] += 1
    out = []
    for acct in store.list_accounts():
        c = counts.get(acct.id, {"uploads": 0, "live": 0, "rejected": 0})
        out.append(_user_item(acct, c))
    out.sort(key=lambda item: item["email"])
    return out


def _user_item(acct, c: dict) -> dict:
    return {"email": acct.email, "uploads": c["uploads"], "live": c["live"], "rejected": c["rejected"],
            "blocked": acct.blocked, "last_seen": acct.last_seen, "is_owner": auth.is_owner(acct)}


def set_blocked(email: str, blocked: bool) -> dict:
    """Block or unblock a contributor. 409 for an owner account. 422 for an invalid email."""
    try:
        key = store.normalize_email(email)
    except ValueError:
        raise HTTPException(422, "invalid email address")
    if blocked and key in config.OWNER_EMAILS:
        raise HTTPException(409, "An owner account cannot be blocked.")
    acct = store.set_blocked(key, blocked)
    counts = {"uploads": 0, "live": 0, "rejected": 0}
    for bundle in store.all_bundles():
        for v in bundle.videos:
            if v.contributor_id == acct.id:
                counts["uploads"] += 1
                counts["live"] += int(_on_map(v))
                counts["rejected"] += int(v.moderation_status in ("rejected", "hidden"))
    return _user_item(acct, counts)
