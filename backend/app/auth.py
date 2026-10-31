"""Access control: one owner key per server, plus invite tokens per building.

Tokens are random (secrets.token_urlsafe(32)). Only their SHA-256 hex digests are stored: the owner's in
DATA_DIR/auth.json, each invite's in buildings/<id>/access.json. A raw token is returned once (when the owner key
is generated, or when an invite is created) and is never stored or logged afterwards.

Roles, highest first: owner (every building, building creation, invites) > editor (change a building) > viewer
(read a building, plan routes). Routes state their role as a dependency:

    dependencies=[Depends(auth.require("editor"))]    /buildings/{building_id}/... routes
    dependencies=[Depends(auth.require_job("editor"))]  /jobs/{job_id}/... routes
    dependencies=[Depends(auth.require_owner_key)]    server-wide owner routes

Photo URLs (/bfiles) carry an HMAC signature instead, because <img> tags cannot send headers.
"""
import hashlib
import hmac
import logging
import math
import os
import re
import secrets
import threading
import time
import uuid
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

from fastapi import Header, HTTPException, Request
from fastapi.responses import JSONResponse

from . import config, jobs, store

log = logging.getLogger(__name__)

RANK = {"viewer": 1, "editor": 2, "owner": 3}
MAX_ACCESS_TOKENS = 50  # invite tokens read from one X-Access-Tokens header; the rest are ignored
FAIL_WINDOW_SEC = 60  # sliding window for AUTH_MAX_FAILS
_MAX_TRACKED_IPS = 10_000  # bound on the failure table (spoofed CF-Connecting-IP values cannot grow it forever)
_WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

# --- owner key and frame signing secret (DATA_DIR/auth.json) ---
_state_lock = threading.RLock()
_cache: tuple[tuple, dict] | None = None  # (file stamp, loaded state)


def _sha(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _auth_path() -> Path:
    return config.DATA_DIR / "auth.json"


def _read_file() -> dict:
    return store.read_json(_auth_path(), {})


def _write_file(data: dict) -> None:
    path = _auth_path()
    store.write_json_atomic(path, data)  # mkstemp creates the temp file 0600 and os.replace keeps that
    os.chmod(path, 0o600)


def _stamp() -> tuple:
    """Changes when auth.json or ADMIN_TOKEN changes, so a rotation in another process takes effect at once.
    The file's bytes are compared, not its mtime: some filesystems only keep whole seconds."""
    path = _auth_path()
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        content = None
    return (str(path), content, config.ADMIN_TOKEN)


def _load_state() -> dict:
    data = _read_file()
    changed = False
    if not data.get("signing_secret"):
        data["signing_secret"] = secrets.token_hex(32)
        changed = True
    raw = None  # the owner key in clear: exists only in this variable and the one log record below
    if not config.ADMIN_TOKEN and not data.get("owner_sha256"):
        raw = secrets.token_urlsafe(32)
        data["owner_sha256"] = _sha(raw)
        changed = True
    if changed:
        _write_file(data)
    if raw is not None:
        log.warning(
            "OWNER KEY, shown only once: %s\n"
            "Send it as 'Authorization: Bearer <key>'. Only its SHA-256 is stored, in %s. To use your own key, set "
            "the ADMIN_TOKEN environment variable. To replace this one, run: python -m app.cli owner-key --rotate",
            raw, _auth_path())
    owner = _sha(config.ADMIN_TOKEN) if config.ADMIN_TOKEN else data["owner_sha256"]
    return {"owner_sha256": owner, "signing_secret": data["signing_secret"].encode()}


def _state() -> dict:
    global _cache
    with _state_lock:
        if _cache is None or _cache[0] != _stamp():
            state = _load_state()
            _cache = (_stamp(), state)  # stamp taken after any write above
        return _cache[1]


def _owner_match(token: str) -> bool:
    return hmac.compare_digest(_sha(token).encode(), _state()["owner_sha256"].encode())


def startup() -> None:
    """Run once when the server starts: refuses a short ADMIN_TOKEN, creates auth.json on first run, and warns
    loudly if access control is off. Raising here stops the server from starting."""
    if config.AUTH_REQUIRED and config.ADMIN_TOKEN and len(config.ADMIN_TOKEN) < config.ADMIN_TOKEN_MIN_CHARS:
        # The token itself is never printed: this message only states the length problem.
        raise RuntimeError(
            f"ADMIN_TOKEN is set but only {len(config.ADMIN_TOKEN)} characters long; it must be at least "
            f"{config.ADMIN_TOKEN_MIN_CHARS}. Use a longer random value (for example: python -c "
            "\"import secrets; print(secrets.token_urlsafe(32))\"), or unset ADMIN_TOKEN to get a generated owner key.")
    _state()
    if not config.AUTH_REQUIRED:
        log.warning("=" * 72)
        log.warning("AUTH_REQUIRED=0: ACCESS CONTROL IS OFF. Anyone who can reach this server can read, change, "
                    "delete and rebuild every building. Use this for local development only.")
        log.warning("=" * 72)


def owner_key_status() -> dict:
    """Where the owner key comes from (for `app.cli owner-key`). Never returns the key or its hash."""
    if config.ADMIN_TOKEN:
        return {"set": True, "source": "ADMIN_TOKEN environment variable", "path": str(_auth_path())}
    has = bool(_read_file().get("owner_sha256"))
    return {"set": has, "source": str(_auth_path()) if has else None, "path": str(_auth_path())}


def rotate_owner_key() -> str:
    """Replace the owner key with a new one and return it. The old key stops working at once."""
    global _cache
    if config.ADMIN_TOKEN:
        raise RuntimeError("ADMIN_TOKEN is set in the environment and takes precedence; unset it to use a stored key")
    raw = secrets.token_urlsafe(32)
    with _state_lock:
        data = _read_file()  # not _state(): that would log a generated key that is about to be replaced
        if not data.get("signing_secret"):
            data["signing_secret"] = secrets.token_hex(32)  # keeps existing photo URLs valid
        data["owner_sha256"] = _sha(raw)
        _write_file(data)
        _cache = None
    return raw


# --- signed photo URLs (/bfiles/<building>/frames/<file>?exp=&sig=) ---
_EXP = re.compile(r"[0-9]{1,15}")


def _sign_frame(key: bytes, building_id: str, filename: str, exp: int, generation: int) -> str:
    return hmac.new(key, f"{building_id}/{filename}/{exp}/{generation}".encode(), hashlib.sha256).hexdigest()


def _frame_sig(building_id: str, filename: str, exp: int, generation: int | None = None) -> str:
    if generation is None:
        generation = _generation(building_id) or 0
    return _sign_frame(_state()["signing_secret"], building_id, filename, exp, generation)


def frame_urls(building_id: str, filenames: list[str | None]) -> list[str | None]:
    """Browser URLs for these photos, in the same order (None stays None). Built at response time, never stored.
    Access control on: each URL is signed and expires after FRAME_URL_TTL_SEC. Off: the plain path, as before.
    The signing key and the building's generation are read once for the whole list, so a large map does not
    re-read auth.json for every photo."""
    if not config.AUTH_REQUIRED:
        return [f"/bfiles/{building_id}/frames/{f}" if f else None for f in filenames]
    key = _state()["signing_secret"]
    generation = _generation(building_id) or 0  # unreadable access file: URLs are refused on verify anyway
    exp = int(time.time()) + config.FRAME_URL_TTL_SEC
    return [f"/bfiles/{building_id}/frames/{f}?exp={exp}&sig={_sign_frame(key, building_id, f, exp, generation)}"
            if f else None for f in filenames]


def _generation(building_id: str) -> int | None:
    """The building's current generation, or None if the id is malformed or its access file is unreadable."""
    try:
        path = store.building_dir(building_id) / "access.json"
    except ValueError:
        return None
    try:
        return _load_access(path)[0]
    except (ValueError, OSError):
        log.error("unreadable access file, photo URLs refused: %s", path)
        return None


def frame_signature_ok(building_id: str, filename: str, exp: str | None, sig: str | None) -> bool:
    if not exp or not sig or not _EXP.fullmatch(exp):
        return False
    expires = int(exp)
    if expires < time.time():
        return False
    generation = _generation(building_id)
    if generation is None:
        return False
    return hmac.compare_digest(sig.encode(), _frame_sig(building_id, filename, expires, generation).encode())


# --- invites (see the generation note above) ---
# access.json: {"generation": int, "invites": [{id, role, label, token_sha256, created_at, revoked}, ...]}
# generation starts at 0 and goes up by one on every revoke. It is part of each signed photo URL, so a revoke
# makes that building's existing URLs stop verifying at once. It is read at verify time and never put in the URL.
# A bare list (the layout before generations) reads as generation 0 with those invites.
def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _load_access(path: Path) -> tuple[int, list[dict]]:
    """(generation, invites) from an access file. A missing file is (0, []). Raises ValueError if the file is
    unreadable or malformed (callers treat that as no access)."""
    data = store.read_json(path, None)
    if data is None:
        return 0, []
    if isinstance(data, list):
        return 0, data
    is_dict = isinstance(data, dict)
    generation = data.get("generation", 0) if is_dict else None
    invites = data.get("invites", []) if is_dict else None
    if type(generation) is not int or generation < 0 or not isinstance(invites, list):
        raise ValueError(f"malformed access file: {path}")
    return generation, invites


def _save_access(building_id: str, generation: int, invites: list[dict]) -> None:
    store.write_json_atomic(store.building_dir(building_id) / "access.json",
                            {"generation": generation, "invites": invites})


def read_invites(building_id: str) -> list[dict]:
    """Raises ValueError for a malformed building id."""
    return _load_access(store.building_dir(building_id) / "access.json")[1]


def public_invite(entry: dict) -> dict:
    """The only shape an invite is ever returned in: no token and no hash."""
    return {"id": entry["id"], "role": entry["role"], "label": entry.get("label", ""),
            "created_at": entry["created_at"], "revoked": bool(entry.get("revoked"))}


def list_invites(building_id: str) -> list[dict]:
    return [public_invite(e) for e in read_invites(building_id)]


def create_invite(building_id: str, role: str, label: str) -> tuple[dict, str]:
    """Returns (stored entry, raw token). The raw token is not kept anywhere; the caller hands it out once."""
    token = secrets.token_urlsafe(32)
    entry = {"id": uuid.uuid4().hex[:12], "role": role, "label": label, "token_sha256": _sha(token),
             "created_at": _now_iso(), "revoked": False}
    with store.lock_for(building_id):
        generation, invites = _load_access(store.building_dir(building_id) / "access.json")
        invites.append(entry)
        _save_access(building_id, generation, invites)
    return entry, token


def revoke_invite(building_id: str, invite_id: str) -> dict | None:
    """Marks the invite revoked (the record stays, so it still shows in the list). None if there is no such invite."""
    with store.lock_for(building_id):
        generation, invites = _load_access(store.building_dir(building_id) / "access.json")
        entry = next((e for e in invites if e.get("id") == invite_id), None)
        if entry is None:
            return None
        if not entry.get("revoked"):
            # Bumped once per real revoke (not on a repeat): every photo URL signed for this building stops verifying.
            entry["revoked"] = True
            generation += 1
            _save_access(building_id, generation, invites)
    return public_invite(entry)


def _hash_matches(entry: dict, token_hash: str) -> bool:
    return hmac.compare_digest(str(entry.get("token_sha256", "")).encode(), token_hash.encode())


def _iter_access():
    """(building_id, invites) for every building that has an access file."""
    if not config.BUILDINGS_DIR.exists():
        return
    for d in sorted(config.BUILDINGS_DIR.iterdir()):
        path = d / "access.json"
        if path.is_file():
            try:
                yield d.name, _load_access(path)[1]
            except ValueError:
                log.error("unreadable access file: %s", path)


def role_for_token(token: str, building_id: str) -> str | None:
    """The token's role on this building: 'owner', 'editor', 'viewer', or None if it grants nothing here
    (unknown token, revoked invite, invite for another building, or a malformed building id)."""
    if _owner_match(token):
        return "owner"
    try:
        invites = read_invites(building_id)
    except ValueError:
        return None
    h = _sha(token)
    for entry in invites:
        if _hash_matches(entry, h):
            return None if entry.get("revoked") else entry.get("role")
    return None


def token_known(token: str) -> bool:
    """True if the token is the owner key or a live invite on any building (no role check)."""
    if _owner_match(token):
        return True
    h = _sha(token)
    return any(not e.get("revoked") and _hash_matches(e, h) for _, invites in _iter_access() for e in invites)


def grants_for_tokens(tokens: list[str]) -> dict[str, str]:
    """{building_id: role} for every building a live invite token opens. Invalid tokens are ignored."""
    hashes = [_sha(t) for t in tokens]
    out: dict[str, str] = {}
    for bid, invites in _iter_access():
        best = None
        for entry in invites:
            if entry.get("revoked") or entry.get("role") not in RANK:
                continue
            matched = any(_hash_matches(entry, h) for h in hashes)
            if matched and (best is None or RANK[entry["role"]] > RANK[best]):
                best = entry["role"]
        if best:
            out[bid] = best
    return out


# --- brute-force limit: failed attempts per client IP in a sliding window ---
_fails: dict[str, deque] = {}
_fails_lock = threading.Lock()


def _now() -> float:
    return time.monotonic()


_LOOPBACK_PEERS = frozenset({"127.0.0.1", "::1", "::ffff:127.0.0.1"})  # where cloudflared connects from (IPv4-mapped too)


def client_ip(request: Request) -> str:
    """The address a failed attempt is counted against. Behind the cloudflared tunnel every request arrives from
    loopback, so the real client is in CF-Connecting-IP. That header is believed only when the socket peer is
    loopback: from any other peer anyone can send any value, so the peer address is used instead."""
    peer = request.client.host if request.client else "unknown"
    cf = request.headers.get("cf-connecting-ip", "").strip()
    if cf and peer in _LOOPBACK_PEERS:
        return cf
    return peer


def _recent(ip: str) -> deque:
    """Failures of this IP still inside the window (caller holds _fails_lock)."""
    dq = _fails.get(ip)
    if dq is None:
        return deque()
    cutoff = _now() - FAIL_WINDOW_SEC
    while dq and dq[0] <= cutoff:
        dq.popleft()
    if not dq:
        del _fails[ip]
    return dq


def _record_fail(request: Request) -> None:
    ip = client_ip(request)
    with _fails_lock:
        if len(_fails) > _MAX_TRACKED_IPS:
            for other in list(_fails):
                _recent(other)
        _fails.setdefault(ip, deque()).append(_now())


def _check_lock(request: Request) -> None:
    """429 once AUTH_MAX_FAILS failures fall in the last FAIL_WINDOW_SEC seconds: attempt N+1 is refused."""
    limit = config.AUTH_MAX_FAILS
    with _fails_lock:
        dq = _recent(client_ip(request))
        if dq and len(dq) >= limit:
            # The request is allowed again when enough of the oldest failures have aged out to leave fewer than `limit`.
            wait = dq[len(dq) - limit] + FAIL_WINDOW_SEC - _now()
            raise HTTPException(429, "Too many failed attempts; wait a minute and try again",
                                headers={"Retry-After": str(max(1, math.ceil(wait)))})


def _deny(request: Request, status: int, detail: str, counted: bool = True):
    """Refuse the request. counted=False for a request with no credential at all (not a guessing attempt)."""
    if counted:
        _record_fail(request)
    raise HTTPException(status, detail, headers={"WWW-Authenticate": "Bearer"} if status == 401 else None)


def _bearer(authorization: str | None) -> str | None:
    scheme, _, value = (authorization or "").strip().partition(" ")
    token = value.strip()
    if scheme.lower() != "bearer" or not token:
        return None
    return token


# --- dependencies ---
def _authorize_building(request: Request, authorization: str | None, building_id: str, min_role: str) -> str:
    if not config.AUTH_REQUIRED:
        return "owner"
    _check_lock(request)
    token = _bearer(authorization)
    if token is None:
        _deny(request, 401, "Missing bearer token", counted=False)
    # Unknown building and a building this token does not open get the same answer, so IDs cannot be probed.
    role = role_for_token(token, building_id)
    if role is None:
        _deny(request, 401, "Invalid or revoked token")
    if RANK[role] < RANK[min_role]:
        _deny(request, 403, f"This needs {min_role} access to the building")
    return role


def require(min_role: str):
    """Dependency for building routes: the caller's role on {building_id} must be at least min_role."""
    if min_role not in RANK:
        raise ValueError(f"unknown role {min_role!r}")

    def check(request: Request, building_id: str, authorization: str | None = Header(None)) -> str:
        return _authorize_building(request, authorization, building_id, min_role)

    return check


def require_job(min_role: str):
    """Dependency for /jobs/{job_id} routes: checked against the building the job belongs to."""
    if min_role not in RANK:
        raise ValueError(f"unknown role {min_role!r}")

    def check(request: Request, job_id: str, authorization: str | None = Header(None)) -> str:
        if not config.AUTH_REQUIRED:
            return "owner"
        _check_lock(request)
        token = _bearer(authorization)
        if token is None:
            _deny(request, 401, "Missing bearer token", counted=False)
        if _owner_match(token):
            return "owner"  # an unknown job then gets the handler's 404
        job = jobs.get(job_id)
        role = role_for_token(token, job.building_id) if job else None
        if role is None:
            _deny(request, 401, "Invalid or revoked token")  # same answer for an unknown job
        if RANK[role] < RANK[min_role]:
            _deny(request, 403, f"This needs {min_role} access to the building")
        return role

    return check


def require_owner_key(request: Request, authorization: str | None = Header(None)) -> str:
    """Server-wide owner routes (building creation). Invite tokens get 403, unknown tokens 401."""
    if not config.AUTH_REQUIRED:
        return "owner"
    _check_lock(request)
    token = _bearer(authorization)
    if token is None:
        _deny(request, 401, "Missing bearer token", counted=False)
    if _owner_match(token):
        return "owner"
    if token_known(token):
        _deny(request, 403, "This needs the owner key")
    _deny(request, 401, "Invalid or revoked token")


def caller_grants(request: Request, authorization: str | None, x_access_tokens: str | None) -> dict[str, str] | None:
    """Which buildings GET /api/buildings may list: None = all (owner key), else {building_id: role}.

    Authorization is the owner-key check only: if it is present (any scheme or value) it must be the owner key,
    otherwise 401 (counted as a failed attempt). Invite tokens go in the X-Access-Tokens header (comma-separated,
    first MAX_ACCESS_TOKENS used). No Authorization and no valid invite token: 200 with no buildings. A request
    whose invite tokens are all invalid is counted as a failed attempt, but still answers 200 with no buildings."""
    if not config.AUTH_REQUIRED:
        return None
    _check_lock(request)
    if authorization is not None:
        bearer = _bearer(authorization)
        if bearer is not None and _owner_match(bearer):
            return None
        # One message for every non-owner value (wrong key, invite token, odd scheme): no token oracle.
        _deny(request, 401, "Authorization on this route takes only the owner key; send invite tokens in X-Access-Tokens")
    candidates = [t.strip() for t in (x_access_tokens or "").split(",") if t.strip()][:MAX_ACCESS_TOKENS]
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        return {}
    grants = grants_for_tokens(candidates)
    if not grants:
        _record_fail(request)
    return grants


class WriteGuard:
    """Refuses POST/PUT/PATCH/DELETE that carry no known token, before FastAPI reads the body.

    A multipart upload is spooled to disk before any route dependency runs, so without this an anonymous client
    could fill the disk. Route dependencies still decide the role; this only rejects the obviously unauthenticated."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and config.AUTH_REQUIRED and scope["method"] in _WRITE_METHODS:
            request = Request(scope)
            try:
                _check_lock(request)
                token = _bearer(request.headers.get("authorization"))
                if token is None:
                    _deny(request, 401, "Missing bearer token", counted=False)
                if not token_known(token):
                    _deny(request, 401, "Invalid or revoked token")
            except HTTPException as exc:
                await JSONResponse({"detail": exc.detail}, status_code=exc.status_code,
                                   headers=exc.headers)(scope, receive, send)
                return
        await self.app(scope, receive, send)
