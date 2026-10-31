"""'Something's wrong here' reports: a visitor flags a bad step or place, an editor reviews it and resolves it.

Storage: buildings/<id>/reports.jsonl, append-only, one JSON object per line:
    {"type": "report", "id", "kind", "note", "node_id", "step_index", "from_node", "to_node", "created_at", "reporter"}
    {"type": "status", "id", "status", "resolution_note", "at", "by"}
A report's current state is folded from its lines in file order (fold()), so every status change stays in the file
and shows as the report's history. Lines that cannot be read (e.g. a torn last line after a crash) are skipped.

reporter is the invite id of the token that sent the report, "owner" for the owner key, or None when access control
is off. Raw tokens and their hashes are never written here. The file is never served by /bfiles and is never part of
a map response: it is read only by the reports routes in router.py.

Every read and append happens under store.lock_for(building_id), the same lock as map saves.
"""
import json
import logging
import math
import os
import secrets
from datetime import UTC, datetime, timedelta
from pathlib import Path

from . import auth, config, store

log = logging.getLogger(__name__)

FILE = "reports.jsonl"
WINDOW_SEC = 3600  # the per-reporter rate limit is "REPORTS_PER_HOUR in any 3600 seconds"
STATUSES = ("open", "resolved")


class Refused(Exception):
    """A report action the caller may not perform right now. The router turns it into an HTTP response."""

    def __init__(self, status: int, detail: str, retry_after: int | None = None):
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.retry_after = retry_after


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _path(building_id: str) -> Path:
    return store.building_dir(building_id) / FILE


def _records(building_id: str) -> list[dict]:
    """Every readable line, in file order. Unreadable lines are skipped (and counted in the log)."""
    try:
        raw = _path(building_id).read_bytes()
    except FileNotFoundError:
        return []
    out: list[dict] = []
    bad = 0
    for line in raw.split(b"\n"):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)  # bytes: UnicodeDecodeError and JSONDecodeError are both ValueError
        except ValueError:
            bad += 1
            continue
        if not isinstance(rec, dict) or rec.get("type") not in ("report", "status") or not isinstance(rec.get("id"), str):
            bad += 1
            continue
        out.append(rec)
    if bad:
        log.warning("reports: skipped %d unreadable line(s) in %s", bad, _path(building_id))
    return out


def _append(building_id: str, rec: dict) -> None:
    """Append one line and flush it to disk. Caller holds the building lock."""
    path = _path(building_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_newline = False
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            needs_newline = f.read(1) != b"\n"  # a torn last line: start on a fresh line, do not glue onto it
    with open(path, "ab") as f:
        if needs_newline:
            f.write(b"\n")
        f.write((json.dumps(rec) + "\n").encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())


def fold(records: list[dict]) -> dict[str, dict]:
    """Current state of every report, keyed by id, in creation order. Each has its full history."""
    reports: dict[str, dict] = {}
    for rec in records:
        if rec["type"] == "report":
            if rec["id"] in reports:
                continue  # ids are random; a repeated id is ignored rather than overwriting the first
            reporter = rec.get("reporter")
            created = rec.get("created_at")
            reports[rec["id"]] = {
                "id": rec["id"], "kind": rec.get("kind"), "note": rec.get("note"), "node_id": rec.get("node_id"),
                "step_index": rec.get("step_index"), "from_node": rec.get("from_node"), "to_node": rec.get("to_node"),
                "created_at": created, "reporter": reporter, "status": "open", "resolution_note": None,
                "updated_at": created,
                "history": [{"status": "open", "at": created, "by": reporter, "resolution_note": None}],
            }
        else:
            r = reports.get(rec["id"])
            status = rec.get("status")
            if r is None or status not in STATUSES:
                continue
            note = rec.get("resolution_note") if status == "resolved" else None  # reopening clears the old note
            r["status"], r["resolution_note"], r["updated_at"] = status, note, rec.get("at")
            r["history"].append({"status": status, "at": rec.get("at"), "by": rec.get("by"), "resolution_note": note})
    return reports


def reporter_for(building_id: str, authorization: str | None) -> str | None:
    """Who is acting: the invite id of the caller's token, "owner" for the owner key, None with access control off.
    The route's auth dependency has already checked the token, so the lookup should always succeed; if it does
    not, the request is refused rather than stored without a reporter."""
    if not config.AUTH_REQUIRED:
        return None
    token = auth._bearer(authorization)
    if token is None:
        raise Refused(401, "Missing bearer token")
    if auth._owner_match(token):
        return "owner"
    h = auth._sha(token)
    try:
        invites = auth.read_invites(building_id)
    except ValueError:
        invites = []
    for entry in invites:
        if auth._hash_matches(entry, h):
            return entry["id"]
    raise Refused(401, "Invalid or revoked token")


def _retry_after(in_window: list[datetime], limit: int, now: datetime) -> int:
    """Seconds until this reporter may send again: when enough of their oldest reports leave the window."""
    if limit > 0 and in_window:
        free_at = in_window[len(in_window) - limit] + timedelta(seconds=WINDOW_SEC)
        wait = (free_at - now).total_seconds()
    else:
        wait = WINDOW_SEC
    return max(1, math.ceil(wait))


def add(building_id: str, fields: dict, reporter: str | None) -> dict:
    """Store a new report. Raises Refused: 429 past REPORTS_PER_HOUR for this reporter (not checked when reporter
    is None, because then there is no identity to count against), 507 once the building holds REPORTS_MAX reports.
    The caller has already checked that node_id, from_node and to_node exist on the map."""
    with store.lock_for(building_id):
        recs = _records(building_id)
        reports = [r for r in recs if r["type"] == "report"]
        if len(reports) >= config.REPORTS_MAX:
            raise Refused(507, f"This building already holds its maximum of {config.REPORTS_MAX} reports, "
                               "so no new reports can be stored")
        if reporter is not None:
            now = datetime.now(UTC)
            cutoff = now - timedelta(seconds=WINDOW_SEC)
            times = []
            for r in reports:
                if r.get("reporter") != reporter:
                    continue
                try:
                    t = datetime.fromisoformat(r["created_at"])
                except (KeyError, TypeError, ValueError):
                    continue
                if t > cutoff:
                    times.append(t)
            times.sort()
            limit = config.REPORTS_PER_HOUR
            if limit <= 0 or len(times) >= limit:
                raise Refused(429, f"Too many reports from you for this building (limit {limit} per hour); "
                                   "try again later", retry_after=_retry_after(times, limit, now))
        known = {r["id"] for r in recs}
        report_id = secrets.token_hex(6)
        while report_id in known:
            report_id = secrets.token_hex(6)
        rec = {"type": "report", "id": report_id, **fields, "created_at": _now_iso(), "reporter": reporter}
        _append(building_id, rec)
    return {"id": report_id, "status": "open"}


def list_reports(building_id: str, status: str, limit: int) -> list[dict]:
    """Newest first. status: open | resolved | all."""
    folded = list(fold(_records(building_id)).values())
    folded.reverse()  # file order is creation order, so reversed is newest first
    if status != "all":
        folded = [r for r in folded if r["status"] == status]
    return folded[:limit]


def set_status(building_id: str, report_id: str, status: str, resolution_note: str | None,
               by: str | None) -> dict | None:
    """Append a status event and return the folded report. None if there is no such report."""
    with store.lock_for(building_id):
        recs = _records(building_id)
        if report_id not in fold(recs):
            return None
        event = {"type": "status", "id": report_id, "status": status,
                 "resolution_note": resolution_note if status == "resolved" else None, "at": _now_iso(), "by": by}
        _append(building_id, event)
        return fold(recs + [event])[report_id]


def open_count(building_id: str) -> int:
    return sum(1 for r in fold(_records(building_id)).values() if r["status"] == "open")
