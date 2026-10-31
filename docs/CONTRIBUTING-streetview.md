# Contributing: Street View viewer (parallel work guide)
Goal: work in parallel and merge into `Soo` at the end with a working app and no conflicts.
Shared base: `Soo` @ 1812dcb (on GitHub).

## 1. Branches
| Who | Branch | Base | Merges into |
|---|---|---|---|
| Sudarshan | `Soo` directly | Soo @ 1812dcb | Soo |
| Neel | `streetview` | Soo @ 1812dcb | Soo, via a pull request, AFTER the fixes land |

Neel: `git fetch origin && git checkout -b streetview origin/Soo`
Neel should rebase on Soo once the fixes are pushed: `git fetch origin && git rebase origin/Soo`.

## 2. File ownership (the main thing that prevents merge conflicts)
**Sudarshan owns (Neel must NOT edit):**
backend/app/mapper.py, gemma.py, ocr.py, matching.py, floors.py, motion.py, jobs.py, routing.py, store.py,
backend/eval/**, the existing tests in backend/tests/.

**Neel owns: NEW files only:**
- backend/app/panorama.py: stitching and the coverage mask
- backend/app/streetview.py: its own FastAPI `APIRouter` with all Street View endpoints
- backend/tests/test_streetview*.py
- frontend/src/components/StreetView*.jsx (+ any new CSS file it imports)

**Shared files: small, append-only edits by Neel, one block each:**
| File | Allowed change |
|---|---|
| backend/app/main.py | 2 lines: `from .streetview import router as streetview_router` and `app.include_router(streetview_router)`, placed next to the existing `app.include_router(api)` |
| backend/app/config.py | one block at the END of the file, every name prefixed `STREETVIEW_` |
| backend/app/schemas.py | new classes appended at the END only; never change existing classes |
| frontend/src/App.jsx | one route/tab entry for the viewer |
| frontend/src/api.js | new functions appended at the end |
| README.md | one new section "Street View" |
| backend/pyproject.toml / frontend/package.json | new dependencies only, listed in the PR description |

## 3. Data contract (what Street View may read; READ-ONLY)
Street View consumes the map and never changes it. Fields from backend/app/schemas.py (current Soo):
- `Node`: `id, label, type, floor, x, y, frame_id, frame_heading, vertical, video_id, t`
- `Edge`: `from_node, to_node, heading_out, heading_in, vertical, length_m, frames_fwd, frames_rev`
- `Keyframe` (in each walk's walk.json): `frame_id, video_id, t, kind ("turn_start" / "turn_end" / "glance" ...), heading, pitch`
- Photos are served at `/bfiles/<building>/frames/<frame_id>`
- Load data with the existing helpers in store.py (`store.building_dir`, `store.frames_dir`, the map loader). Read-only.

Rules:
1. **Never write map.json or walk.json.** Street View output goes in its own folder:
   `data/buildings/<id>/streetview/` (panoramas + `index.json`). It must be safe to delete at any time.
2. **Rebuild-safe:** key each panorama by the node id + a hash of its source frame_ids. After a rebuild or a merge
   (node ids can change), regenerate lazily on request or via `POST /api/buildings/{id}/streetview/rebuild`.
   Never assume node ids are stable.
3. **Real pixels only in phase 1.** Every panorama ships a coverage mask (which directions are real).
4. **Gemma/OCR isolation (important):** Street View images must NEVER be passed to `gemma.analyze_frame`,
   `gemma.same_place`, OCR, or the mapper. Panoramas and AI-filled imagery must never affect labels, room numbers,
   floors or merges. If AI extension is added later, every image carries a provenance mask (real vs AI-filled),
   and any Gemma call on it must state which regions are AI-generated. Coordinate that with Sudarshan first,
   because it touches gemma.py.
5. API under the prefix `/api/buildings/{building_id}/streetview/...`, e.g.
   `GET .../streetview/{node_id}`: `{pano_url, mask_url, heading0, neighbours:[{node_id, heading, label, vertical}]}`.
   Neighbour headings come from the edges (heading_out from this node; for reversed edges, heading_in + 180).

## 4. Merge order and gates
1. Sudarshan finishes the 3 fixes and pushes them to Soo. This is the bug-fix work on Soo.
   Gates: `cd backend && .venv/bin/python -m pytest -q` green, plus `.venv/bin/python -m eval.run all` hard gate passes.
2. Sudarshan tells Neel the new Soo commit hash.
3. Neel rebases `streetview` on origin/Soo and resolves any conflicts, which should only be in the shared files from section 2.
4. Neel runs, before opening the PR:
   - `cd backend && .venv/bin/python -m pytest -q`: all green (fix tests + streetview tests)
   - `cd backend && .venv/bin/python -m eval.run all --oracle`: same scores as Soo. This proves Street View didn't change the map.
   - `cd frontend && npm run build`: OK
   - manual check: upload `backend/eval/scenarios/two_floor/media/{up,down,f1_revisit}.mp4`, open the viewer,
     step between places across both floors.
5. PR `streetview` -> `Soo`. Sudarshan reviews, then merges.

## 5. Environment both of you need
- macOS (Apple Vision OCR), `brew install ffmpeg ollama`, `ollama pull gemma3:4b`, `brew services start ollama`.
- Backend: python 3.12 venv in backend/.venv (`uv sync` or `python -m venv .venv && .venv/bin/pip install -e .`).
- Run: `cd backend && .venv/bin/uvicorn app.main:app --port 8000`; frontend dev: `cd frontend && npm install && npm run dev`.
- Generated test videos: `cd backend && .venv/bin/python -m eval.generate all` (creates media in eval/scenarios/*/media; gitignored).

## 6. Things to talk about early
- Whether Neel adds a JS panorama library (Pannellum / Marzipano / three.js): licence and bundle size.
- Whether node ids should become stable across rebuilds. That would be Sudarshan's change in mapper.py, if Street View needs it.
- Phase C (AI extension): local vs cloud model, and the provenance-mask format. It needs both people.
