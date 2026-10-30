import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config, jobs, store
from .router import api


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


app = FastAPI(title="Indoor Visual Navigation", version="0.2.0")
app.add_middleware(BodySizeLimit)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])  # outermost
app.include_router(api)


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


if (config.STATIC_DIR / "index.html").exists():
    # Built frontend (npm run build) served from the same origin, so the phone needs one URL only.
    app.mount("/", StaticFiles(directory=config.STATIC_DIR, html=True), name="spa")
else:
    @app.get("/", response_class=HTMLResponse)
    def index():
        return ("<h3>API running.</h3><p>Frontend not built: run <code>npm run build</code> in frontend/, "
                "or use the Vite dev server on :5173.</p>")
