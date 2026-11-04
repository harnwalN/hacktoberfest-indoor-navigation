"""Owner and contributor routes. Mounted from main.py under /api.

GET  /api/me/uploads                              signed in: your walks and their status (processing|live|needs_review|rejected)
GET  /api/owner/moderation?status=...             owner: walks by moderation status (pending|rejected|approved|hidden|all)
POST /api/owner/videos/{building_id}/{video_id}/moderation   owner: {action: approve|reject|hide, note?}
GET  /api/owner/videos/{building_id}/{video_id}/frames/{filename}?exp=&sig=   signed link to a preview frame
GET  /api/owner/users                             owner: contributor accounts with their walk counts
POST /api/owner/users/{email}/block               owner: {blocked: bool}

Owner routes take an owner session (Authorization: Bearer) or the legacy X-Owner-Token (auth.require_owner).
"""
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from . import auth, owner
from .schemas import Account, ModerationDecision, UserBlock

owner_api = APIRouter(prefix="/api")


def _moderator(account: Account | None) -> str:
    """Who made a decision, as stored on the walk: the owner's email, or "owner-token" for the legacy token."""
    return account.email if account is not None else "owner-token"


@owner_api.get("/me/uploads")
def my_uploads(account: Account = Depends(auth.require_session)):
    """The signed-in account's walks, newest first. Blocked accounts can still read their own list."""
    return owner.my_uploads(account)


@owner_api.get("/owner/moderation")
def moderation_list(status: Literal["pending", "rejected", "approved", "hidden", "all"] = "pending",
                    _owner: Account | None = Depends(auth.require_owner)):
    return owner.moderation_items(status)


@owner_api.post("/owner/videos/{building_id}/{video_id}/moderation")
def moderate(building_id: str, video_id: str, body: ModerationDecision,
             account: Account | None = Depends(auth.require_owner)):
    """approve puts the walk on the map (a rebuild or a reprocess job is queued when needed; its id is job_id).
    reject and hide take it off the map. The answer is the walk's moderation item plus job_id (null when no job)."""
    item, job_id = owner.decide(building_id, video_id, body.action, body.note, _moderator(account))
    return {**item, "job_id": job_id}


@owner_api.get("/owner/videos/{building_id}/{video_id}/frames/{filename}")
def moderation_frame(building_id: str, video_id: str, filename: str,
                     exp: str | None = None, sig: str | None = None):
    """A preview frame of a walk that is not public. Access is the signed link from the owner list (no session header,
    because an <img> cannot send one). 403 for a missing, forged or expired link; 404 when the file is gone."""
    if not auth.frame_link_valid(building_id, video_id, filename, exp, sig):
        raise HTTPException(403, "This frame link is invalid or has expired. Reload the moderation list.")
    path = owner.frame_file_path(building_id, video_id, filename)
    if path is None:
        raise HTTPException(404, "Not found")
    return FileResponse(path, headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "private, max-age=300"})


@owner_api.get("/owner/users")
def users(_owner: Account | None = Depends(auth.require_owner)):
    return owner.users()


@owner_api.post("/owner/users/{email}/block")
def block_user(email: str, body: UserBlock, _owner: Account | None = Depends(auth.require_owner)):
    """blocked: true refuses the account's uploads (403). Sign-in and reading the own uploads still work."""
    return owner.set_blocked(email, body.blocked)
