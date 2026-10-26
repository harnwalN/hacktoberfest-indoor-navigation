# Waypoint: indoor navigation from a walkthrough video

Record one walk through a building on a phone and upload it. **Gemma, running locally and offline**, reads the door plaques and signs. Computer vision recovers every turn, glance, and staircase. The result is a navigation graph that can close loops.

Users then pick *Start* and *Destination* on their phone and follow step-by-step photos with arrows. Routing at query time is pure graph math, with **no model calls**.

```
video ─ffmpeg─▶ analysis.mp4 (10 fps) ─optical flow─▶ heading track ─▶ turns / glances / pauses
       └──────▶ hq.mp4 (15 fps) ─adaptive sharpness─▶ keyframes ─Gemma (local)─▶ signs, rooms, stairs
                                                         │
                       walk nodes ─▶ loop closure (room no. + ORB + Gemma "same place?") ─▶ map.json
```

## Run it

```bash
# one-time setup (needs internet once)
brew install ffmpeg uv cloudflared ollama
ollama pull gemma3:4b            # fast default; gemma3:12b for higher-quality pre-processing

# 1. local model server (parallel requests ~25% faster on Apple Silicon)
OLLAMA_NUM_PARALLEL=4 ollama serve

# 2. backend (serves the API and the built frontend on :8000)
cd backend && uv sync && uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

# 3. frontend: build once into backend/static ...
cd frontend && npm install && npm run build
#    ... or develop with hot reload on :5173 (proxies /api to :8000)
npm run dev

# 4. phone access over HTTPS (needed for camera + gyro on iOS); works on any network
cloudflared tunnel --url http://localhost:8000
```

Open the printed `https://….trycloudflare.com` URL on the phone. Bookmark `#/navigate?b=<building-id>`.

### Batch pre-processing (no UI)
```bash
cd backend
uv run python -m app.cli ingest "Sample Building" ../IMG_31*.MOV
uv run python -m app.cli route sample-building n001 n012
uv run python -m app.cli rebuild sample-building   # re-run CV + mapping; Gemma answers come from cache
```

### Tests
`cd backend && uv run pytest`. The tests use synthetic videos with a known camera yaw:
- 90° turns are measured to within 0.1°.
- A glance (looking at something and back) is not counted as a turn.
- A 4-turn loop closes into a cycle.
- On a fixture map, routes mirror in reverse, stairs flip up↔down, and "no path" returns HTTP 400.

## Working as a team

**One machine is the "map server".** It runs Ollama + the backend, processes all videos, and holds
`backend/data/` (git-ignored). Everyone else points at it, so there is one map and one Gemma.

| Role | What to run |
|---|---|
| Map server (Mac, 16 GB+) | full setup above + `cloudflared tunnel --url http://localhost:8000`, share the URL |
| Frontend dev | `cd frontend && npm install && API_TARGET=https://<tunnel>.trycloudflare.com npm run dev` (no Python/Ollama needed) |
| Backend / pipeline dev | full setup on your own laptop; copy a building folder to test on real data (below) |
| Recording / map curation | phone only: tunnel URL → Record, Map, Navigate |

- Get real data locally: zip `backend/data/buildings/<id>/` on the map server (skip `videos/` to keep it
  small) and unzip into the same path; everything runs offline from that folder.
- OCR grounding uses the macOS Vision framework. On Linux/Windows it is skipped automatically and Gemma
  runs ungrounded (expect some invented room numbers), so process final videos on a Mac.
- Map edits save the whole map: only one person should edit the Map tab at a time.
- Git: branch per task (`git switch -c feat/...`), open a PR to `master`, never commit `backend/data/` or
  videos. Run `cd backend && uv run pytest` before pushing backend changes.

## How it works

| Stage | What happens |
|---|---|
| **Normalize** | ffmpeg converts any input (iPhone HEVC `.MOV` with rotation, WebM, MP4) into constant-frame-rate H.264 copies with rotation applied |
| **Motion** | Lucas–Kanade corners plus a RANSAC similarity fit in centred coordinates. Horizontal shift gives **yaw**; expansion gives **forward motion**. The **travel heading** is a running median over frames that move forward and look ahead, so standing and looking around never counts as a turn. If a phone gyro log is attached, it replaces vision yaw, with its sign auto-calibrated against vision |
| **Events** | *Turn*: a sustained change in travel heading. *Glance*: the camera points more than 35° off the travel direction (usually at a plaque). *Pause*: the camera is nearly still |
| **Keyframes** | Event-driven (before and after turns, glances, pauses, scene changes, plus a 3 s heartbeat). In each ±0.5 s window, pick the frame with the highest sharpness relative to the video's own rolling median, divided by motion. Blurry-only windows still keep their best frame, flagged `soft`, so no place is ever lost |
| **Gemma** | Structured JSON per keyframe: sign text, room numbers, label, place type, stairs up or down, floor, and a short landmark phrase. If a sign is visible but unreadable, Gemma is asked again with zoomed left and right crops. All answers are cached on disk |
| **Mapping** | Turns, pauses, signs, and stair boundaries become nodes. Revisits are detected by room-number text and ORB homography, then verified by Gemma ("same physical spot?"). Confident matches merge automatically; others go to a review queue. Each verified match also constrains heading drift (fit `c0 + k·t`). Edge headings are snapped to the building's dominant axis (Manhattan prior) |
| **Routing** | `turn = heading_out(next) − heading_in(arrival)`, quantized to straight, slight, turn, or u-turn. Walking an edge backwards uses headings + 180° and stairs up↔down, so one recording works in both directions. Each step shows the photo taken while approaching that point *in your direction of travel*; if a stretch was only walked the other way, the step carries a note |

## Recording tips
- 1080p with stabilization on. Hold the phone upright and point it where you walk. Turn at a steady pace.
- **Pause about 1 s** at plaques, junctions, and stairs. Glancing at a plaque and back is fine.
- Walk corridors in both directions when you can, and pass known spots again to close loops.
- Best accuracy: record from the **Record** tab on the phone, which logs the gyro alongside the video.

## Persistence
Each building is a portable folder, `backend/data/buildings/<id>/`:
- `map.json`: the graph
- `frames/`: photos users see
- `gemma_cache.json`
- `videos/<vid>/`: normalized video, track, walk

Copy the folder to the demo machine and everything works offline.
