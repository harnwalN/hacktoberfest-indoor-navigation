"""File-based persistence: one portable "map bundle" directory per building.

data/buildings/<id>/
    map.json            the navigable graph (nodes, edges, merges)
    gemma_cache.jsonl   Gemma answers, one JSON line per answer (append-only; a legacy gemma_cache.json is migrated to .bak)
    frames/*.jpg        keyframes of approved walks (served by /bfiles)
    videos/<vid>/       normalized videos, track.json, walk.json (pipeline outputs)
    videos/<vid>/rejected_frames/   blurred keyframes of walks that are not approved (never served)
    videos/<vid>/staging_frames/    raw keyframes while a walk is being moderated (never served)

data/accounts.json      contributor accounts: {"accounts": [Account, ...]} (one file for all buildings)
data/auth.json          sign-in links and sessions, token hashes only (auth.py)

The API never serves data/ directly: map.json is read through the routes (no uploader emails in public responses).
"""
import json
import os
import re
import shutil
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .schemas import DEFAULT_CAMPUS, Account, MapBundle, VideoInfo

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


def held_frames_dir(building_id: str, video_id: str) -> Path:
    """Blurred keyframes of a walk that is not approved (pending, needs review, rejected, hidden). Inside the walk's own
    folder, never under frames/, so /bfiles cannot serve them. The name is kept from the first version of this folder."""
    return video_dir(building_id, video_id) / "rejected_frames"


def staging_frames_dir(building_id: str, video_id: str) -> Path:
    """Keyframes of a walk while it is being extracted and face-blurred. Unblurred files live only here, so they are
    never public and never held. Emptied on every start (an interrupted walk is extracted again)."""
    return video_dir(building_id, video_id) / "staging_frames"


def frame_url(building_id: str, frame_id: str | None) -> str | None:
    """Browser URL of a keyframe, built at response time and never stored."""
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
            out.append({"id": m.building_id, "name": m.name, "campus": m.campus,
                        "nodes": len(m.nodes), "edges": len(m.edges)})
    return out


def create_building(name: str, creator_id: str | None = None) -> MapBundle:
    """creator_id: Account.id of the contributor who created the building; they become its moderator.
    Unknown creator_id raises KeyError before anything is written."""
    if creator_id is not None and get_account(creator_id) is None:
        raise KeyError(creator_id)
    # Leave room for the "-2", "-3"... suffix so the id stays within MAX_BUILDING_ID.
    base = slugify(name)[:MAX_BUILDING_ID - 12].strip("-") or "building"
    building_id, n = base, 2
    while (config.BUILDINGS_DIR / building_id).exists():
        building_id, n = f"{base}-{n}", n + 1
    bundle = MapBundle(building_id=building_id, name=name, campus=DEFAULT_CAMPUS)
    frames_dir(building_id).mkdir(parents=True, exist_ok=True)
    save_map(bundle)
    if creator_id is not None:
        add_moderator(creator_id, building_id)
    return bundle


def delete_building(building_id: str) -> None:
    """Remove a building's whole folder (map, frames, walks, reports) and its moderator grants. Caller checks the
    building exists and cancels its queued jobs. KeyError: no such building. ValueError: invalid id."""
    folder = building_dir(building_id)
    with lock_for(building_id):
        if not (folder / "map.json").exists():
            raise KeyError(building_id)
        shutil.rmtree(folder)
    with _accounts_lock:
        rows = _read_accounts()
        changed = False
        for row in rows:
            if building_id in row["moderator_of"]:
                row["moderator_of"] = [b for b in row["moderator_of"] if b != building_id]
                changed = True
        if changed:
            _write_accounts(rows)


def load_map(building_id: str) -> MapBundle:
    data = read_json(building_dir(building_id) / "map.json")
    if data is None:
        raise FileNotFoundError(building_id)
    return MapBundle.model_validate(data)


# Photo URLs exist only in API responses (schemas.MapOut). Never write them to map.json, even if a
# response-model object is handed in by mistake.
_RESPONSE_ONLY_FIELDS = {"nodes": {"__all__": {"frame_url"}},
                         "edges": {"__all__": {"frames_fwd_urls", "frames_rev_urls"}}}


def save_map(bundle: MapBundle) -> None:
    with lock_for(bundle.building_id):
        write_json_atomic(building_dir(bundle.building_id) / "map.json",
                          bundle.model_dump(exclude=_RESPONSE_ONLY_FIELDS))


# ---------- contributor accounts (data/accounts.json) ----------
# Email-identified, no passwords. Sign-in is not here; it will look accounts up with get_account().
_accounts_lock = threading.RLock()
_EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
MAX_EMAIL = 254


def accounts_path() -> Path:
    # Resolved on each call, so a test that moves config.DATA_DIR moves this file too.
    return config.DATA_DIR / "accounts.json"


def normalize_email(email: str) -> str:
    """Stripped and lowercased. Raises ValueError for anything that is not a plain address.
    The address is not echoed in the error (it is personal data)."""
    value = email.strip().lower() if isinstance(email, str) else ""
    if not value or len(value) > MAX_EMAIL or not _EMAIL_RE.fullmatch(value):
        raise ValueError("invalid email address")
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_accounts() -> list[dict]:
    return read_json(accounts_path(), default={"accounts": []})["accounts"]


def _write_accounts(rows: list[dict]) -> None:
    write_json_atomic(accounts_path(), {"accounts": rows})


def create_account_if_missing(email: str) -> Account:
    """The account for this email, created on first use. Any letter case of the same address gives the same account."""
    key = normalize_email(email)
    with _accounts_lock:
        rows = _read_accounts()
        for row in rows:
            if row["email"] == key:
                return Account(**row)
        acct = Account(id=uuid.uuid4().hex, email=key, created_at=_now())
        rows.append(acct.model_dump())
        _write_accounts(rows)
        return acct


def get_account(email_or_id: str) -> Account | None:
    """Find by email (any letter case) or by id. A value containing "@" is read as an email; ids never contain "@"."""
    if not isinstance(email_or_id, str) or not email_or_id:
        return None
    rows = _read_accounts()
    if "@" in email_or_id:
        try:
            key = normalize_email(email_or_id)
        except ValueError:
            return None
        row = next((r for r in rows if r["email"] == key), None)
    else:
        row = next((r for r in rows if r["id"] == email_or_id), None)
    return Account(**row) if row else None


def get_account_by_id(account_id: str) -> Account | None:
    row = next((r for r in _read_accounts() if r["id"] == account_id), None)
    return Account(**row) if row else None


def list_accounts() -> list[Account]:
    return [Account(**r) for r in _read_accounts()]


def set_blocked(email: str, blocked: bool) -> Account:
    """Block or unblock the contributor with this email (created if it has never signed in, so a spammer can be
    blocked before their first sign-in). Returns the account. ValueError: invalid email."""
    key = normalize_email(email)
    with _accounts_lock:
        rows = _read_accounts()
        row = next((r for r in rows if r["email"] == key), None)
        if row is None:
            row = Account(id=uuid.uuid4().hex, email=key, created_at=_now()).model_dump()
            rows.append(row)
        row["blocked"] = bool(blocked)
        _write_accounts(rows)
        return Account(**row)


_SEEN_EVERY_SEC = 3600


def touch_last_seen(account_id: str) -> None:
    """Record the account's last use. Written at most once an hour per account, so reads stay cheap."""
    with _accounts_lock:
        rows = _read_accounts()
        row = next((r for r in rows if r["id"] == account_id), None)
        if row is None:
            return
        now = datetime.now(timezone.utc)
        last = row.get("last_seen")
        if last and (now - datetime.fromisoformat(last)).total_seconds() < _SEEN_EVERY_SEC:
            return
        row["last_seen"] = now.isoformat(timespec="seconds")
        _write_accounts(rows)


def add_moderator(account_id: str, building_id: str) -> None:
    """Grant the account moderation of the building. Idempotent. KeyError: unknown account.
    ValueError: invalid building id."""
    building_dir(building_id)  # validates the id; no filesystem access
    with _accounts_lock:
        rows = _read_accounts()
        row = next((r for r in rows if r["id"] == account_id), None)
        if row is None:
            raise KeyError(account_id)
        if building_id not in row["moderator_of"]:
            row["moderator_of"].append(building_id)
            _write_accounts(rows)


def is_moderator(account_id: str, building_id: str) -> bool:
    row = next((r for r in _read_accounts() if r["id"] == account_id), None)
    return row is not None and building_id in row["moderator_of"]


def all_bundles() -> list[MapBundle]:
    """Every building's map, in folder order. Buildings whose map cannot be read are skipped (logged by the caller)."""
    if not config.BUILDINGS_DIR.exists():
        return []
    out = []
    for d in sorted(config.BUILDINGS_DIR.iterdir()):
        if (d / "map.json").exists():
            try:
                out.append(load_map(d.name))
            except (ValueError, OSError):
                continue
    return out


def backfill_uploader_emails() -> int:
    """One-off, safe on every start: a walk with a contributor_id but no recorded uploader email gets it from the
    account. Walks with no contributor_id stay "unknown". Returns the number of walks changed."""
    if not config.BUILDINGS_DIR.exists():
        return 0
    changed = 0
    for d in sorted(config.BUILDINGS_DIR.iterdir()):
        if not (d / "map.json").exists():
            continue
        with lock_for(d.name):
            bundle = load_map(d.name)
            dirty = False
            for v in bundle.videos:
                if v.uploader_email != "unknown" or not v.contributor_id:
                    continue
                acct = get_account_by_id(v.contributor_id)
                if acct is not None:
                    v.uploader_email = acct.email
                    dirty = True
                    changed += 1
            if dirty:
                save_map(bundle)
    return changed


# ---------- walk moderation (VideoInfo in the building's map.json) ----------
MODERATION_STATUSES = ("pending", "approved", "rejected", "hidden")
MAX_MODERATION_NOTES = 1000
UNCHANGED = object()  # set_video_moderation(notes=UNCHANGED) keeps the notes already stored


def get_video(building_id: str, video_id: str) -> VideoInfo | None:
    """The walk's record in the building's map, or None if no walk has that id.
    FileNotFoundError: unknown building."""
    bundle = load_map(building_id)
    return next((v for v in bundle.videos if v.id == video_id), None)


def set_video_moderation(building_id: str, video_id: str, status: str, notes: object = UNCHANGED,
                         moderated_by: object = UNCHANGED) -> VideoInfo:
    """Set a walk's moderation_status, and its moderation_notes when notes is given (None or "" clears them).
    moderated_by: the owner's email (or "owner-token") for an owner decision; None clears it. Left alone by default.
    ValueError: status not in MODERATION_STATUSES, or notes not a string / over MAX_MODERATION_NOTES.
    KeyError: no walk with that id. FileNotFoundError: unknown building."""
    if status not in MODERATION_STATUSES:
        raise ValueError(f"moderation_status must be one of: {', '.join(MODERATION_STATUSES)}")
    if notes is not UNCHANGED:
        notes = _clean_notes(notes)
    with lock_for(building_id):
        bundle = load_map(building_id)
        v = next((v for v in bundle.videos if v.id == video_id), None)
        if v is None:
            raise KeyError(video_id)
        v.moderation_status = status
        if notes is not UNCHANGED:
            v.moderation_notes = notes
        if moderated_by is not UNCHANGED:
            v.moderated_by = moderated_by
        save_map(bundle)
        return v


def set_auto_moderation(building_id: str, video_id: str, status: str, notes: object = UNCHANGED) -> VideoInfo:
    """The pipeline's own moderation result (jobs.py). Once an owner has decided the walk (VideoInfo.moderated_by is set),
    the owner's decision stands: nothing is written and the walk as stored is returned."""
    with lock_for(building_id):
        bundle = load_map(building_id)
        v = next((v for v in bundle.videos if v.id == video_id), None)
        if v is None:
            raise KeyError(video_id)
        if v.moderated_by is not None:
            return v
        return set_video_moderation(building_id, video_id, status, notes=notes)


def _clean_notes(notes: object) -> str | None:
    if notes is None:
        return None
    if not isinstance(notes, str):
        raise ValueError("moderation_notes must be a string or None")
    value = notes.strip()
    if len(value) > MAX_MODERATION_NOTES:
        raise ValueError(f"moderation_notes must be at most {MAX_MODERATION_NOTES} characters (after stripping spaces)")
    return value or None


def approve_legacy_walks() -> int:
    """One-off backfill, safe to run on every start. A walk published before walk moderation existed (status "done",
    and its walk.json has no "moderation" record) is approved. Walks processed since then always have that record.
    Walks still queued or failed are left pending, so they are checked when they run. Returns the number of walks changed."""
    if not config.BUILDINGS_DIR.exists():
        return 0
    changed = 0
    for d in sorted(config.BUILDINGS_DIR.iterdir()):
        if not (d / "map.json").exists():
            continue
        with lock_for(d.name):
            bundle = load_map(d.name)
            dirty = False
            for v in bundle.videos:
                if v.status != "done" or v.moderation_status != "pending":
                    continue
                try:
                    walk = read_json(video_dir(d.name, v.id) / "walk.json") or {}
                except ValueError:  # an id that cannot be a video folder: no walk.json to read
                    walk = {}
                if "moderation" not in walk:
                    v.moderation_status = "approved"
                    dirty = True
                    changed += 1
            if dirty:
                save_map(bundle)
    return changed
