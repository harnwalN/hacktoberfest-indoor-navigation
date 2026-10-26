from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import config, jobs
from .router import api

app = FastAPI(title="Indoor Visual Navigation", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(api)


@app.on_event("startup")
def _recover_queue():
    jobs.recover()  # re-queue videos interrupted by a restart

config.BUILDINGS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/bfiles", StaticFiles(directory=config.BUILDINGS_DIR), name="bfiles")

if (config.STATIC_DIR / "index.html").exists():
    # Built frontend (npm run build) served from the same origin, so the phone needs one URL only.
    app.mount("/", StaticFiles(directory=config.STATIC_DIR, html=True), name="spa")
else:
    @app.get("/", response_class=HTMLResponse)
    def index():
        return ("<h3>API running.</h3><p>Frontend not built: run <code>npm run build</code> in frontend/, "
                "or use the Vite dev server on :5173.</p>")
