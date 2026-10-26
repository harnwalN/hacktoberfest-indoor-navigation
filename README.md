# Waypoint: indoor navigation from a walkthrough video

Record walks through a building on a phone and upload them. **Gemma, running locally and offline**, reads door plaques and signs; on-device OCR checks every word it reports. Computer vision (plus the phone gyro, when available) recovers every turn, glance, and staircase. All videos are merged into **one multi-floor navigation map** that closes loops and links floors through shared stairwells and elevators.

Users pick *From* and *To* on their phone and follow step-by-step photos with wayfinding arrows. Routing at query time is pure graph math, with **no model calls**.

```
video ─ffmpeg─▶ analysis.mp4 (10 fps) ─optical flow / gyro─▶ travel heading ─▶ turns / glances / pauses
       └──────▶ hq.mp4 (15 fps) ─adaptive sharpness─▶ keyframes ─OCR + Gemma (local)─▶ signs, rooms, stairs
                                                         │
     walk nodes ─▶ loop closure ─▶ floors (start floor, room numbers, stairs) ─▶ link all videos ─▶ map.json
```

## AI models

**Used by the app (all run locally, offline after one download):**

| Model | Role |
|---|---|
| **Gemma 3 4B** (`gemma3:4b`, Google, open weights, via Ollama) | Default vision-language model. Per keyframe it returns structured JSON: sign text, room numbers, place type, stairs, floor hints and a landmark phrase. It also answers "same place?" and "same stairwell?" when linking walks and floors |
| **Gemma 3 12B** (`gemma3:12b`, optional) | Higher-quality, ~2.5× slower alternative for overnight pre-processing on machines with 32 GB+ |
| **Apple Vision text recognizer** (macOS `VNRecognizeTextRequest`, on-device) | Grounds Gemma: only text this OCR actually found may become a sign or room number, which stops small-model hallucinations. Skipped automatically off macOS |

Everything else is classical computer vision with no learned model: Lucas–Kanade optical flow, RANSAC similarity fits, ORB features and homographies (OpenCV), plus graph search (NetworkX).

**Used to design and build it:** the architecture, pipeline, backend, frontend, tests and this README were designed and implemented with **Claude Opus 5.5** (Anthropic, model ID `claude-opus-5-5`) running in **Claude Code**, working with the team below. The team chose the building, recorded and curated the walkthroughs, tested on site and directed the design.

## Contributors

- Evan Schaller
- Naagaa Prachet Kadaba Pradeep
- Sudarshan Kotlanka
- Neelaksh Harnwal

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

Open the printed `https://….trycloudflare.com` URL on the phone. Bookmark `#/navigate?b=<building-id>`;
teammates record new walks at `#/record?b=<building-id>`.

### Batch pre-processing (no UI)
```bash
cd backend
uv run python -m app.cli ingest "Sample Building" ../IMG_31*.MOV
uv run python -m app.cli route sample-building n001 n012
uv run python -m app.cli rebuild sample-building   # re-run CV + mapping; Gemma answers come from cache
```

### Tests
`cd backend && uv run pytest` (19 tests). They use synthetic videos with a known camera yaw and fixture maps:
- 90° turns are measured to within 0.1°; a glance (look at something and back) is not a turn.
- A 4-turn loop closes into a cycle.
- Routes mirror in reverse, stairs flip up↔down, and "no path" returns HTTP 400.
- Floors from room numbers (121 → 1, 208 → 2), elevator jumps without stairs, one misread number ignored.
- Walks merge across videos by room number or named sign but never across floors; guessed floors are
  adopted from linked walks; single staircases on adjacent floors are linked; removing a video keeps
  places other walks still use.

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
| **Gemma + OCR** | On-device OCR lists the text really present; Gemma returns structured JSON (sign text, room numbers, label, place type, stairs, floor, landmark phrase) and may only copy text from that list. Labels built on an unverified number are rebuilt. Without OCR (non-macOS) Gemma re-reads unclear signs with zoomed crops. All answers are cached on disk |
| **Mapping** | Turns, pauses, signs, and stair boundaries become nodes; consecutive frames of one named place collapse. Revisits are detected by room-number text and ORB homography, verified by Gemma ("same physical spot?"); automatic merges need geometric agreement, others go to a review queue. Each verified match constrains heading drift (fit `c0 + k·t`). Edge headings are snapped to the building's dominant axis (Manhattan prior) |
| **Floors** | Floors start at 1. Each walk's floor comes from the uploader's start floor, room numbers (first digit = floor), and detected stairs; a room-number floor change without visible stairs inserts a flagged stairs/elevator link. Places on different floors never merge |
| **One map from n videos** | After every video: same room number or same named sign on the same floor ⇒ same place (even when walked in opposite directions); strong ORB or ORB + Gemma matches across walks; walks with a guessed floor adopt the floor of a known place they share; unanchored walks are rotated into the building frame; the same stairwell or elevator on adjacent floors is linked |
| **Routing** | `turn = heading_out(next) − heading_in(arrival)`, quantized to straight, slight, turn, or u-turn. Walking an edge backwards uses headings + 180° and stairs up↔down, so one recording works in both directions. Floor changes say "take the stairs up to floor 2"; across a link between two separate walks the step says where to head instead of guessing a turn. Each step shows the photo taken while approaching that point *in your direction of travel* |
| **App** | Directions: full-bleed photo with a wayfinding sign (bent-arrow pictograms, stair yellow for floor changes) and a route strip. Map: per-floor schematic with an elevator-style floor selector, pan/zoom, merge review and editing. Build: pausable processing queue (run next, resume, survives restarts), per-video start floor, video removal |

## Recording tips
- 1080p with stabilization on. Hold the phone upright and point it where you walk. Turn at a steady pace.
- **Pause about 1 s** at plaques, junctions, and stairs. Glancing at a plaque and back is fine.
- Walk corridors in both directions when you can, and pass known spots again to close loops.
- Best accuracy: record from the **Record** tab on the phone, which logs the gyro alongside the video, and
  enter the floor the walk starts on.
- Overlap something already mapped (a room number, named sign or staircase) so the walk links into the map;
  film staircases from both floors to connect floors.

## Persistence
Each building is a portable folder, `backend/data/buildings/<id>/`:
- `map.json`: the graph
- `frames/`: photos users see
- `gemma_cache.json`
- `videos/<vid>/`: normalized video, track, walk

Copy the folder to the demo machine and everything works offline.
