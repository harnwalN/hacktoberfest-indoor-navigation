import hmac
import logging
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.routing import Match, Mount, get_route_path
from starlette.types import Scope

from . import config, jobs, store
from .auth_router import auth_api
from .owner_router import owner_api
from .router import api

log = logging.getLogger(__name__)

class BodySizeLimit:
    """Refuse requests whose declared Content-Length exceeds the largest allowed upload, before
    Starlette spools the multipart body to disk. Per-file limits are enforced again in the routes."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            limit = config.MAX_UPLOAD_MB * 1024 * 1024 + config.MAX_SENSOR_BYTES + (1 << 20)  # + form overhead
            length = dict(scope["headers"]).get(b"content-length", b"")
            if length.isdigit() and int(length) > limit:
                await JSONResponse({"detail": "Request body too large"}, status_code=413)(scope, receive, send)
                return
        await self.app(scope, receive, send)


# public by design; sign-in is open too, so it works before anyone has the owner token
_OPEN_WRITES = re.compile(r"/api/(route|buildings/[^/]+/route|buildings/[^/]+/reports|auth/(request-link|verify))")


class OwnerWriteGate:
    """With OWNER_TOKEN set, every map-changing /api request must send either a valid X-Owner-Token (scripts, acts as
    the owner) or an Authorization header (a sign-in session). The route then decides what that caller may do: owner
    routes need an owner session, upload needs a contributor session that is not blocked. A request with neither is
    401. Reads and the public routing and reports endpoints stay open."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and config.OWNER_TOKEN and scope["method"] not in ("GET", "HEAD", "OPTIONS"):
            path = scope["path"]
            if path.startswith("/api/") and not _OPEN_WRITES.fullmatch(path):
                headers = dict(scope["headers"])
                given = headers.get(b"x-owner-token", b"").decode("latin-1")
                token_ok = hmac.compare_digest(given.encode(), config.OWNER_TOKEN.encode())
                has_session = b"authorization" in headers
                if not (token_ok or has_session):
                    await JSONResponse({"detail": "Owner token or sign-in required"}, status_code=401)(scope, receive, send)
                    return
        await self.app(scope, receive, send)


app = FastAPI(title="Indoor Visual Navigation", version="0.2.0")
app.add_middleware(OwnerWriteGate)
app.add_middleware(BodySizeLimit)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])  # outermost
app.include_router(api)
app.include_router(auth_api)
app.include_router(owner_api)


@app.on_event("startup")
def _owners_configured():
    if not config.OWNER_EMAILS:
        log.warning("OWNER_EMAILS is not set: no one can open the owner page")


@app.on_event("startup")
def _approve_legacy_walks():
    store.approve_legacy_walks()  # walks published before moderation existed: approved, not re-checked


@app.on_event("startup")
def _uploader_emails():
    store.backfill_uploader_emails()  # walks from before uploader emails were recorded get the account's email


@app.on_event("startup")
def _settle_walk_frames():
    jobs.settle_walk_frames()  # frames of walks not approved move out of the public folder (safe to repeat)


@app.on_event("startup")
def _recover_queue():
    jobs.recover()  # re-queue videos interrupted by a restart

config.BUILDINGS_DIR.mkdir(parents=True, exist_ok=True)

_FRAME_NAME = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]*")  # one path segment, never "." or ".."


@app.api_route("/bfiles/{building_id}/frames/{filename}", methods=["GET", "HEAD"])
def frame_file(building_id: str, filename: str):
    """Serve keyframe images only; the rest of a building bundle (map, cache, videos) stays private."""
    try:
        frames = store.frames_dir(building_id)
    except ValueError:
        raise HTTPException(404, "Not found")
    path = frames / filename
    if not _FRAME_NAME.fullmatch(filename) or not path.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(path, headers={"X-Content-Type-Options": "nosniff"})


class _FrontendMount(Mount):
    """The built frontend under /. It never matches an /api path, so an /api path that no route answers gets the
    router's own JSON 404, and a known API path with the wrong method gets its JSON 405 with Allow. Mounted at / it
    would answer every method itself, with StaticFiles' 405 text for DELETE and the like."""

    def matches(self, scope: Scope) -> tuple[Match, Scope]:
        route_path = get_route_path(scope) if scope["type"] == "http" else ""
        if route_path == "/api" or route_path.startswith("/api/"):
            return Match.NONE, {}
        return super().matches(scope)


def serve_frontend(application: FastAPI, static_dir: Path) -> None:
    """Serve the built frontend (or a stub page) on / from the same origin. /api paths are never served as files."""
    if (static_dir / "index.html").exists():
        # Built frontend (npm run build) served from the same origin, so the phone needs one URL only.
        # Appended to the route list directly: FastAPI's mount() would build a plain Mount, which matches /api too.
        application.router.routes.append(
            _FrontendMount("/", app=StaticFiles(directory=static_dir, html=True), name="spa"))
    else:
        @application.get("/", response_class=HTMLResponse)
        def index():
            return ("<h3>API running.</h3><p>Frontend not built: run <code>npm run build</code> in frontend/, "
                    "or use the Vite dev server on :5173.</p>")


serve_frontend(app, config.STATIC_DIR)
