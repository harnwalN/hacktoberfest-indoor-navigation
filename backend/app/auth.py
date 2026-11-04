"""Magic-link sign-in and session tokens, kept in data/auth.json (same JSON pattern as accounts.json).

data/auth.json: {"magic_links": [{hash, email, expires_at}], "sessions": [{hash, account_id, expires_at}]}

Only the SHA-256 of each token is stored, never the token itself: a leaked file cannot be used to sign in.
A sign-in link is single use (consumed on verify, whether or not it has expired). Times are Unix seconds.
Tokens survive a restart (the file is read on every call). Like accounts.json, the file is only safe for one
server process: the lock is in-process.
"""
import hashlib
import hmac
import math
import secrets
import threading
import time
from collections import deque

from fastapi import Depends, Header, HTTPException

from . import config, store
from .schemas import Account

_lock = threading.RLock()
MAX_PENDING_LINKS = 1000  # oldest pending links are dropped past this, so the file cannot grow without limit

# In-memory hit times for request-link, keyed by "ip:<addr>" and "email:<address>". Not persisted: a restart resets it.
_link_hits: dict[str, deque] = {}


def auth_path():
    # Resolved on each call, so a test that moves config.DATA_DIR moves this file too.
    return config.DATA_DIR / "auth.json"


def _now() -> float:
    return time.time()


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _read() -> dict:
    data = store.read_json(auth_path(), default={}) or {}
    return {"magic_links": data.get("magic_links", []), "sessions": data.get("sessions", [])}


def _write(data: dict) -> None:
    store.write_json_atomic(auth_path(), data)


def link_request_wait(ip: str, email: str) -> int:
    """Count one request-link call for this IP and this email. Returns 0 when both are under AUTH_LINK_RATE_LIMIT
    hits in the last AUTH_LINK_RATE_WINDOW_SEC seconds. Otherwise returns the seconds until the next slot frees up,
    and the refused call is not counted. Nothing here looks at accounts, so the answer never reveals one."""
    limit, window = config.AUTH_LINK_RATE_LIMIT, config.AUTH_LINK_RATE_WINDOW_SEC
    keys = (f"ip:{ip}", f"email:{email}")
    with _lock:
        now = _now()
        for key in list(_link_hits):  # drop expired hits and empty keys, so the dict cannot grow without limit
            q = _link_hits[key]
            while q and q[0] <= now - window:
                q.popleft()
            if not q:
                del _link_hits[key]
        queues = [_link_hits.setdefault(key, deque()) for key in keys]
        if any(len(q) >= limit for q in queues):
            # The slot frees when the oldest hit in the fullest queue leaves the window.
            wait = max(q[0] + window - now for q in queues if len(q) >= limit)
            return max(1, math.ceil(wait))
        for q in queues:
            q.append(now)
        return 0


def create_magic_link(email: str) -> str:
    """A new single-use sign-in token for this (already normalized) email. Returns the raw token; only its hash is kept."""
    token = secrets.token_urlsafe(32)
    with _lock:
        data, now = _read(), _now()
        links = [r for r in data["magic_links"] if r["expires_at"] > now]
        links.append({"hash": _hash(token), "email": email, "expires_at": now + config.AUTH_LINK_TTL_SEC})
        data["magic_links"] = links[-MAX_PENDING_LINKS:]
        data["sessions"] = [r for r in data["sessions"] if r["expires_at"] > now]
        _write(data)
    return token


def consume_magic_link(token: str) -> str | None:
    """The email the link was issued for, or None when the link is unknown, used or expired. Removed either way."""
    digest = _hash(token)
    with _lock:
        data, now = _read(), _now()
        row = next((r for r in data["magic_links"] if r["hash"] == digest), None)
        if row is None:
            return None
        data["magic_links"] = [r for r in data["magic_links"] if r["hash"] != digest and r["expires_at"] > now]
        _write(data)
    return row["email"] if row["expires_at"] > now else None


def create_session(account_id: str) -> str:
    """A new session token for this account, valid for AUTH_SESSION_TTL_SEC. Returns the raw token."""
    token = secrets.token_urlsafe(32)
    with _lock:
        data, now = _read(), _now()
        data["magic_links"] = [r for r in data["magic_links"] if r["expires_at"] > now]
        data["sessions"] = [r for r in data["sessions"] if r["expires_at"] > now]
        data["sessions"].append({"hash": _hash(token), "account_id": account_id,
                                 "expires_at": now + config.AUTH_SESSION_TTL_SEC})
        _write(data)
    return token


def session_account(token: str | None) -> Account | None:
    """The account a session token belongs to, or None (no token, unknown, or expired)."""
    if not token:
        return None
    digest = _hash(token)
    row = next((r for r in _read()["sessions"] if r["hash"] == digest and r["expires_at"] > _now()), None)
    if row is None:
        return None
    account = store.get_account_by_id(row["account_id"])
    if account is not None:
        store.touch_last_seen(account.id)
    return account


def _bearer(authorization: str | None) -> str | None:
    scheme, _, value = (authorization or "").strip().partition(" ")
    value = value.strip()
    return value if scheme.lower() == "bearer" and value else None


def require_session(authorization: str | None = Header(None)) -> Account:
    """FastAPI dependency: the signed-in account, from `Authorization: Bearer <session_token>`. 401 otherwise."""
    account = session_account(_bearer(authorization))
    if account is None:
        raise HTTPException(401, "Sign-in required", headers={"WWW-Authenticate": "Bearer"})
    return account


def require_contributor(account: Account = Depends(require_session)) -> Account:
    """Signed in and not blocked: may upload walks. 403 for a blocked account."""
    if account.blocked:
        raise HTTPException(403, "This account is blocked from uploading walks.")
    return account


def optional_account(authorization: str | None = Header(None)) -> Account | None:
    """The signed-in account if a valid session is sent, else None. Never refuses (for routes open to everyone)."""
    return session_account(_bearer(authorization))


def is_owner(account: Account | None) -> bool:
    """Owner = a signed-in account whose email is in config.OWNER_EMAILS (read on each call)."""
    return account is not None and account.email in config.OWNER_EMAILS


def token_ok(given: str | None) -> bool:
    """The legacy X-Owner-Token matches OWNER_TOKEN. Always False when OWNER_TOKEN is unset."""
    if not config.OWNER_TOKEN or not given:
        return False
    return hmac.compare_digest(given.encode("utf-8"), config.OWNER_TOKEN.encode("utf-8"))


def caller_is_owner(authorization: str | None, x_owner_token: str | None) -> bool:
    """Owner by token or by owner session, without refusing anyone. For routes open to all that show more to owners."""
    return token_ok(x_owner_token) or is_owner(optional_account(authorization))


def require_owner(authorization: str | None = Header(None),
                  x_owner_token: str | None = Header(None, alias="X-Owner-Token")) -> Account | None:
    """FastAPI dependency for owner-only routes. Passes with a valid X-Owner-Token (scripts; returns None) or with a
    session of an owner account (returns that account). 401 with no usable credentials, 403 for a signed-in non-owner."""
    if token_ok(x_owner_token):
        return None
    if _bearer(authorization) is None:
        raise HTTPException(401, "Owner sign-in required", headers={"WWW-Authenticate": "Bearer"})
    account = session_account(_bearer(authorization))
    if account is None:
        raise HTTPException(401, "Sign-in required", headers={"WWW-Authenticate": "Bearer"})
    if not is_owner(account):
        raise HTTPException(403, "Only the owner can do this.")
    return account


# ---------- signed links to moderation frames ----------
# The owner's moderation list shows keyframes of walks that are not public (pending, rejected, hidden). A browser <img>
# cannot send the session header, so each frame link carries an HMAC over (building, walk, file, expiry). The key is
# random per process: a restart invalidates the links, and the owner list hands out fresh ones.
_FRAME_KEY = secrets.token_bytes(32)
FRAME_LINK_SEC = 3600


def _frame_sig(building_id: str, video_id: str, filename: str, expires: int) -> str:
    msg = f"{building_id}\n{video_id}\n{filename}\n{expires}".encode("utf-8")
    return hmac.new(_FRAME_KEY, msg, hashlib.sha256).hexdigest()


def frame_link(building_id: str, video_id: str, filename: str) -> str:
    """Relative URL of one moderation frame, valid for FRAME_LINK_SEC seconds."""
    expires = int(_now()) + FRAME_LINK_SEC
    sig = _frame_sig(building_id, video_id, filename, expires)
    return f"/api/owner/videos/{building_id}/{video_id}/frames/{filename}?exp={expires}&sig={sig}"


def frame_link_valid(building_id: str, video_id: str, filename: str, exp: str | None, sig: str | None) -> bool:
    if not exp or not sig or not exp.isdigit():
        return False
    if int(exp) < _now():
        return False
    expected = _frame_sig(building_id, video_id, filename, int(exp))
    return hmac.compare_digest(sig.encode("utf-8"), expected.encode("utf-8"))  # bytes: str with non-ASCII raises
