# Waypoint: indoor navigation from a walkthrough video

Record a walk through a building on a phone and upload it. The app merges all your walks into one multi-floor navigation map. Users pick *From* and *To* on their phone and follow step-by-step photos with wayfinding arrows.

- **Gemma runs locally and offline** (or optionally a hosted Gemini model) to read door plaques and signs.
- **On-device OCR** checks every word Gemma reports, so the model cannot invent room numbers.
- **Computer vision** (plus the phone gyro, when available) finds every turn, glance and staircase.
- **Routing is pure graph math.** No model is called when a user asks for directions.
- **Sign-in by email link.** Visitors can read and get directions without an account. Uploads need a signed-in account.

> **Screenshots are from an earlier version.** The Build tab is now **Contribute**, and the landing page, the owner page and the new theme are not shown. See [Screenshots](#screenshots).

<p align="center">
  <img src="docs/screenshots/directions-plan.png" alt="Trip planner: choose where you are and where you are going" width="31%">
  <img src="docs/screenshots/directions-turn.jpg" alt="A direction step: photo of the spot with a turn sign" width="31%">
  <img src="docs/screenshots/directions-stairs.jpg" alt="A floor change: yellow stairs sign and route strip marking floors 2 and 3" width="31%">
</p>

<p align="center"><em>Plan a trip, follow photo-and-sign steps, and change floors on the way.</em></p>

<p align="center">
  <img src="docs/screenshots/map.png" alt="Building map with an elevator-style floor selector" width="94%">
</p>

<p align="center"><em>One map per floor, built from every approved walkthrough video. Stairs show the floor they reach.</em></p>

<p align="center">
  <img src="docs/screenshots/build.png" alt="Build tab: upload walks, processing queue and what the pipeline saw" width="94%">
</p>

<p align="center"><em>Upload or record walks; a pausable queue processes them locally.</em></p>

## Contents
- [How it works](#how-it-works)
- [Features](#features)
- [AI models](#ai-models)
- [Run it](#run-it)
- [Share over the internet](#share-over-the-internet)
- [Settings](#settings)
- [Access and roles](#access-and-roles)
- [Configuration](#configuration)
- [Frontend](#frontend)
- [Tests and evaluation](#tests-and-evaluation)
- [Recording tips](#recording-tips)
- [Persistence](#persistence)
- [Screenshots](#screenshots)
- [Changelog](#changelog)
- [Contributing](#contributing)
- [Contributors](#contributors)

## How it works

```
video ─ffmpeg─▶ analysis.mp4 (10 fps) ─optical flow / gyro─▶ travel heading ─▶ turns / glances / pauses
       └──────▶ hq.mp4 (15 fps) ─adaptive sharpness─▶ keyframes ─OCR + Gemma─▶ signs, rooms, stairs
                                                         │
     walk nodes ─▶ loop closure ─▶ floors (start floor, room numbers, stairs) ─▶ link all videos ─▶ map.json
                                                         │
              automatic checks (faces blurred, content checked) ─▶ approved walk joins the map
                                                         (the owner can reject or hide it at any time)
```

| Stage | What happens |
|---|---|
| **Normalize** | ffmpeg converts any input (iPhone HEVC `.MOV` with rotation, WebM, MP4) into constant-frame-rate H.264 copies with rotation applied. |
| **Motion** | Lucas–Kanade corners and a RANSAC similarity fit. Horizontal shift gives **yaw**; expansion gives **forward motion**. The **travel heading** is a running median over frames that move forward, so looking around never counts as a turn. A phone gyro log, when attached, replaces vision yaw. |
| **Events** | *Turn*: a sustained change in travel heading. *Glance*: the camera points more than 35° off the travel direction (usually at a plaque). *Pause*: the camera is nearly still. |
| **Keyframes** | Event-driven (before and after turns, glances, pauses, scene changes, plus a 3 s heartbeat). In each ±0.5 s window the sharpest frame is kept. Blurry windows keep their best frame, flagged `soft`, so no place is lost. |
| **OCR + Gemma** | OCR lists the text really present. Gemma returns structured JSON (sign text, room numbers, label, place type, stairs, floor, landmark phrase) and may only copy text from that list. Answers are cached on disk. See [How Gemma answers are checked](#how-gemma-answers-are-checked). |
| **Mapping** | Turns, pauses, signs and stair boundaries become nodes. Revisits are found by room number and ORB homography, then verified. Places with different plaque numbers never merge. Edge headings snap to the building's dominant axis. |
| **Floors** | Floors start at 1 and come from the start floor, confirmed room numbers, floor signs and detected stairs. See [How floors work](#how-floors-work). |
| **One map from n videos** | Same room number or named sign on the same floor means same place, even when walked in opposite directions. Strong image matches across walks merge too. Unanchored walks are rotated into the building frame. |
| **Moderation** | Every frame is checked (faces blurred, then a vision model checks it is an indoor photo). Any failing frame rejects the walk; otherwise it is approved. The owner can reject, hide or re-approve it. Frames are public only while the walk is approved. |
| **Routing** | `turn = heading_out(next) − heading_in(arrival)`, quantized to straight, slight, turn or u-turn. A walk works in both directions. Floor changes say "take the stairs up to floor 2". Each step shows the photo taken while approaching that point in your direction of travel. |
| **App** | Directions: full-bleed photo with a wayfinding sign and a route strip. Map: per-floor schematic with a floor selector, pan and zoom (read-only for visitors; owners get merge review and editing). Contribute: upload a walk and watch **My uploads**. Owner page: moderation, videos, buildings, reports and users. |

## Features

- **Trip planner and step-by-step directions**, with photo-and-sign steps and floor changes.
- **Step-free routing.** `avoid_stairs` uses only flat walks, ramps and elevators. Every route says whether it is step-free.
- **Shareable route links**, for example `#/navigate?b=<building-id>&from=<place-id>&to=<place-id>`.
- **Measured distances**, shortest-walk routing and a 3D map view.
- **Contribute tab** (signed-in contributors) with upload or in-browser recording and **My uploads**, which shows each walk's status.
- **Owner page** (`#/owner`, owners only) for moderation, videos and map, buildings, reports and users.
- **Problem reports.** Visitors can report a wrong sign, direction or place ("Something's wrong here"). Owners resolve or reopen them on the owner page. Reporting is rate-limited.
- **Entrance posters.** A printable poster with a QR code for a viewer link, opened from the share panel.
- **Email sign-in** with roles for visitors, contributors and owners. See [Access and roles](#access-and-roles).
- **Share script** `scripts/share.sh` for a Cloudflare tunnel, so phones can reach the app over HTTPS.
- **Start script** `scripts/start.sh` runs the backend and the built frontend on one address.
- **Batch CLI** for ingesting, routing and rebuilding without the UI.
- **Evaluation harness** that scores the whole pipeline on synthetic buildings with known ground truth.

## AI models

**Used by the app (run locally, offline after one download):**

| Model | Role |
|---|---|
| **Gemma 3 4B** (`gemma3:4b`, Google, open weights, via Ollama) | Default vision-language model. Per keyframe it returns structured JSON. It also answers "same place?" and "same stairwell?" when linking walks and floors. |
| **Gemma 3 12B** (`gemma3:12b`, optional) | Higher quality, about 2.5× slower. For overnight pre-processing on machines with 32 GB+. |
| **Apple Vision text recognizer** (macOS, on-device) | OCR engine on macOS. Grounds Gemma: only text this OCR found may become a sign or room number. |
| **RapidOCR** (PP-OCRv5 ONNX models on onnxruntime) | OCR engine on Linux and Windows. Install with `uv sync --extra linux-ocr`. |
| **Gemini** (`gemini-2.5-flash-lite` by default, optional, **hosted**) | Can replace local Gemma. Photos leave the machine and each call costs money. See [Hosted vision](#hosted-vision-optional). |

Everything else is classical computer vision with no learned model: Lucas–Kanade optical flow, RANSAC similarity fits, ORB features and homographies (OpenCV), plus graph search (NetworkX). Faces are blurred on the server before a frame is stored or sent to a model (OpenCV YuNet, bundled in `backend/models/`).

**Optional hosted model.** Google Gemini can read walk frames in place of local Gemma. See *Hosted vision* below. It is off unless you set `GEMINI_API_KEY`.

**Speed.** Tested with `gemma3:4b`: about 5 s per keyframe on Apple Silicon at concurrency 3 (`GEMMA_CONCURRENCY`). Each upload can name its own model (the `model` form field, or `--model` in the CLI). That model is used for every Gemma call on the walk.

**Built with:** the architecture, pipeline, backend, frontend, tests and this README were designed and implemented with **Claude Opus 5.5** (Anthropic, model ID `claude-opus-5-5`) in **Claude Code**. The team chose the building, recorded and curated the walkthroughs, tested on site and directed the design.

## Run it

### One-time setup

```bash
# needs internet once
brew install ffmpeg uv cloudflared ollama
ollama pull gemma3:4b            # fast default; gemma3:12b for higher-quality pre-processing
```

Node.js and npm are needed only for the frontend (`npm ci` and `npm run build`).

### Start the app (the simple way)

```bash
# terminal 1: local model server (parallel requests ~25% faster on Apple Silicon)
OLLAMA_NUM_PARALLEL=4 ollama serve

# terminal 2: backend and built frontend on 127.0.0.1:8000
scripts/start.sh            # Ctrl-C stops it
scripts/start.sh --build    # first runs npm ci and npm run build in frontend/, then starts
scripts/start.sh -h         # help
```

`scripts/start.sh` starts the backend only. It serves the API and the built frontend from one address. It reads
`backend/.env` (see [Settings](#settings)). `HOST` and `PORT` in the shell change the listen address. Run Ollama
before you upload a walk.

Open `http://localhost:8000` on this machine. The first page is the landing page. Use `#/navigate` for Directions.

Linux: run `cd backend && uv sync --extra linux-ocr` once for RapidOCR.

### Develop the frontend (hot reload)

```bash
# terminal 1: backend
cd backend && uv sync && uv run uvicorn app.main:app --host 127.0.0.1 --port 8000

# terminal 2: frontend dev server on :5173 (proxies /api and /bfiles to :8000)
cd frontend && npm install && npm run dev
```

Use `API_TARGET=https://<tunnel-host> npm run dev` to point the dev server at another backend. Then you need no local
Gemma or map data.

### Phone access over HTTPS (tunnel)

Phones need HTTPS for the camera and the gyro on iOS:

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

Or use `scripts/share.sh`, which starts Ollama, the backend and the tunnel together. See [Share over the internet](#share-over-the-internet).

> **Warning: a tunnel makes this machine reachable from the internet.** Anyone with the address can read the maps,
> route, file "something's wrong" reports and ask for a sign-in link. They cannot sign in as someone else, because a
> link goes only to that person's inbox (or to the server log in `log` mode). Contributors and owners still need
> their own email link. Do not share the address widely, and do not set `OWNER_TOKEN` on a public tunnel. Anyone with
> that token is the owner. Use a named tunnel with a private address for anything beyond a demo.

Open the printed `https://….trycloudflare.com` URL on your phone. Walks you upload from it still need a signed-in
contributor account.

### Health check

```bash
curl http://localhost:8000/api/health
```

`gemma.ok` is true when Ollama answers, and `gemma.selected` is the model in use. If Ollama is stopped, the reply comes
back within a few seconds with `ok: false`.

Deep links:
- `#/navigate?b=<building-id>` opens Directions for a building.
- `#/record?b=<building-id>` opens the Record tab.
- `#/navigate?b=<building-id>&from=<place-id>&to=<place-id>` shares a route directly.
- `#/owner` opens the owner page (owners only).

### Batch pre-processing (no UI)
```bash
cd backend
uv run python -m app.cli ingest "Sample Building" ../IMG_31*.MOV
uv run python -m app.cli route sample-building n001 n012
uv run python -m app.cli route sample-building n001 n012 --avoid-stairs   # only flat walks, ramps and elevators
uv run python -m app.cli rebuild sample-building   # re-run CV + mapping; Gemma answers come from cache
# add --model gemma3:12b to ingest for a slower, higher-quality run
# add --model gemini-2.5-flash-lite to ingest for the hosted model (see Hosted vision)
```

## Share over the internet

`scripts/share.sh` starts the backend and Ollama if needed, then opens a Cloudflare tunnel. Use it to reach the app on a phone (camera and gyro need HTTPS on iOS).

```bash
scripts/share.sh            # start (quick mode unless a named tunnel is configured)
scripts/share.sh --build    # rebuild the frontend first; needed once, and after frontend changes
```

- **Quick mode (default, no account):** a random `https://<words>.trycloudflare.com` address. It changes on every restart.
- **Named mode:** a stable address on your own domain. Set `TUNNEL_NAME` and `TUNNEL_HOSTNAME` in the environment or in `scripts/share.env` (copy `scripts/share.env.example`; gitignored). Setting only one is an error. Full steps are in `scripts/README.md`.

Logs go to `logs/` (gitignored). Press Ctrl-C to stop everything the script started.

**Sign-in links need `PUBLIC_BASE_URL` set to the tunnel address** (or the address your users open), or the links they receive will point to the wrong place. See [Sign-in mail and owners](#sign-in-mail-and-owners-backendenv).

## Settings

Settings come from three places, in this order: the shell, then `backend/.env`, then the built-in default.

- **`backend/.env`** is read once, when the backend starts. Copy `backend/.env.example` to start.
  Restart the backend after every change.
- **A variable already set in the shell wins** over the file.
- **An empty line in the file** means "use the default".
- **Never commit `backend/.env`.** It is in `.gitignore`, but check before you `git add`. It can hold an SMTP password and the owner token.
- **Keep secrets in the shell, not in the file.** The loader reads every variable in `backend/.env`, so the file is not a safe place for `GEMINI_API_KEY` or `SMTP_PASSWORD`. Export them in the terminal that starts the backend.

The loader is `backend/app/envfile.py`. It logs only the names of the variables it set, never their values.

### Sign-in mail and owners (`backend/.env`)

| Variable | Default | What it does |
|---|---|---|
| `MAIL_BACKEND` | `log` | `log` prints each sign-in link to the server log and sends nothing (local dev only). `smtp` sends the mail through `SMTP_HOST`. Any other value stops the backend |
| `SMTP_HOST` | unset | SMTP server. **Required when `MAIL_BACKEND=smtp`** |
| `SMTP_PORT` | 587 | 587 = submission with STARTTLS |
| `SMTP_USER` | unset | Login name. Empty if the relay needs no login |
| `SMTP_PASSWORD` | unset | SMTP or app password. Never logged, never returned |
| `SMTP_FROM` | unset | Sender address on sign-in mail |
| `SMTP_STARTTLS` | 1 | `0` turns STARTTLS off. Only for a trusted local relay |
| `PUBLIC_BASE_URL` | unset | The origin users open, for example `https://nav.example.org`. Sign-in links are built from it. **Required when `MAIL_BACKEND=smtp`** |
| `AUTH_LINK_TTL_SEC` | 900 | Sign-in link lifetime. A link works once |
| `AUTH_SESSION_TTL_SEC` | 2592000 | How long a signed-in browser stays signed in (30 days) |
| `AUTH_LINK_RATE_LIMIT` | 5 | Sign-in link requests allowed per window, per client IP and per address |
| `AUTH_LINK_RATE_WINDOW_SEC` | 900 | The window for that limit (15 minutes). Past the limit, the answer is HTTP 429 with `Retry-After` |
| `OWNER_EMAILS` | (none) | Comma-separated owner addresses, e.g. `you@example.org`. Blank = nobody can open the owner page (a startup warning is logged) |
| `OWNER_TOKEN` | unset | Legacy shared secret for scripts, sent as `X-Owner-Token`. Unset = not accepted |

**SMTP needs `PUBLIC_BASE_URL`.** With `MAIL_BACKEND=smtp` the backend refuses to start if `SMTP_HOST` or
`PUBLIC_BASE_URL` is missing. `scripts/start.sh` says which one to fix. A production deploy must set `MAIL_BACKEND=smtp`
on purpose. Log mode prints live sign-in links, so never run a public server in it.

For Gmail, see [backend/EMAIL_SETUP.md](backend/EMAIL_SETUP.md).

Photos are served through signed links that last `FRAME_URL_TTL_SEC` (one day by default).

## Access and roles

Sign-in is by emailed link. There are no passwords. Request a link in the app. The link opens
`{PUBLIC_BASE_URL}/#/verify?token=…`, signs you in for `AUTH_SESSION_TTL_SEC`, and works once.

| Who | Sees and can do |
|---|---|
| **Visitor** (no sign-in) | Landing page, **Directions**, a **view-only Map** (floors, 2D/3D, place details), **Record** (film a walk; uploading needs sign-in), and "Something's wrong here" reports on Directions |
| **Contributor** (signed in) | Everything a visitor sees, plus the **Contribute** tab: upload a walk and **My uploads**, which lists your walks and their status: processing, live, needs review, rejected (`GET /api/me/uploads`) |
| **Blocked contributor** | Signed in and can still read **My uploads**. Uploads get HTTP 403 |
| **Owner** (`OWNER_EMAILS`) | Everything above, plus the **Owner** tab (`#/owner`) with five sections, listed below |

Contribute is shown only to signed-in accounts. A visitor who presses Contribute is sent to sign-in first.

**Owner page (`#/owner`).** Visible only to owners. Any other visitor who opens the URL gets a "not allowed" page.
- **Moderation:** the queue of walks by status. **Approve** puts a walk on the map (a rebuild or reprocess job runs when needed).
  **Reject** and **Hide** take it off the map at once. An owner's decision is final: later automatic checks do not change it.
- **Videos & map:** videos per building, the start floor, the map editor, and linking walks.
- **Buildings:** create and delete buildings (delete asks you to confirm), and open the entrance poster.
- **Reports:** problem reports for one building. Resolve or reopen each one, and jump to its place on the map.
- **Users:** every contributor account with its walk counts. **Block** or **Unblock**. A blocked account's uploads are refused (403). It can still sign in.

**What visitors can see.** Owners see every building. Everyone else sees only buildings that have a map.

**Walk checks and privacy.** Every uploaded walk goes through automatic checks: faces are blurred, and a vision model
checks that each frame is a plausible indoor photo. A walk with any frame that fails is rejected. A walk that passes is
approved and joins the map. The owner can then reject, hide or re-approve it. A walk's keyframe photos are public only
while the walk is approved. Rejected, hidden and pending walks' frames are not public, and the public `/bfiles` route does
not serve them. The owner sees up to six preview frames per walk through a signed link that expires. Public map
responses never include an uploader's or a moderator's email.

**Owner routes need a session.** A map-changing request needs either a sign-in session (`Authorization: Bearer`) or the
legacy `X-Owner-Token`. Without one of them the request gets 401. Reads, routing and reports stay open.

`OWNER_TOKEN` (optional, legacy) is a shared secret for scripts: send it as the header `X-Owner-Token` and the request
counts as the owner. It is never accepted in a query string, a body, a cookie or as a sign-in session.

**Problem reports** let a visitor flag a bad step or place. Reports carry no reporter. Each building holds at most
`REPORTS_MAX` reports (5000 by default). Past that limit, new reports get HTTP 507. The count has no reset. Reports are
stored in `backend/data/buildings/<id>/reports.jsonl` and are never served as files.

**Known limits**
- **A sign-in link works once** and expires after `AUTH_LINK_TTL_SEC`. A request for a new link is rate-limited.
- **A session lasts `AUTH_SESSION_TTL_SEC`** (30 days by default) in that browser.
- **The owner token opens every owner route.** Keep it out of public tunnels and shared logs.
- **Log mode shows live sign-in links** in the server log. Keep that log private.

## Configuration
Every setting has a default and can be overridden with an environment variable, for example
`MAX_UPLOAD_MB=4096 uv run uvicorn app.main:app --port 8000`. The full list is in `backend/app/config.py`.
Settings you can also put in `backend/.env` are in [Settings](#settings) above.

### Access and uploads
| Variable | Default | What it does |
|---|---|---|
| `DATA_DIR` | `backend/data` | Where `buildings/` and `hosted_usage.json` are kept |
| `REPORTS_MAX` | 5000 | Problem reports one building may hold. Past this, new reports get HTTP 507 |
| `MAX_UPLOAD_MB` | 2048 | Largest walk video, in MiB. Bigger uploads get HTTP 413 |
| `MAX_SENSOR_BYTES` | 52428800 (50 MiB) | Largest `sensors.json` sent with a video. Bigger gets 413 |
| `MAX_IMAGE_BYTES` | 26214400 (25 MiB) | Largest manually added photo. Bigger gets 413 |

### Local Gemma and OCR
| Variable | Default | What it does |
|---|---|---|
| `GEMMA_MODEL` | gemma3:4b | Model used when an upload names none. `gemma3:12b` is slower and more accurate |
| `GEMMA_KEEP_ALIVE` | 30m | How long Ollama keeps the model loaded after a request. Also `1h`, or seconds (`-1` = forever) |
| `GEMMA_NUM_CTX` | 4096 | Context size sent with every request, so the loaded model is reused |
| `GEMMA_NUM_PREDICT_ANALYZE` | 512 | Answer token cap for frame analysis. A truncated answer fails the call |
| `GEMMA_NUM_PREDICT_COMPARE` | 320 | Answer token cap for "same place?" and "same stairwell?" |
| `GEMMA_LIST_TIMEOUT_SEC` | 3 | Timeout for model list, model check and health calls, so a stopped Ollama fails fast |
| `GEMMA_BREAKER_FAILS` | 3 | Failures in a row before the rest of the walk uses cached answers only |
| `OCR_ENGINE` | auto | `auto` (Apple Vision on macOS, else RapidOCR if installed), `vision`, `rapidocr` or `none` |
| `OCR_ENABLED` | 1 | `0` turns OCR off whatever the engine |
| `VISION_PROVIDER` | auto | Model for new walks: `auto`, `gemma` (local only) or `hosted` (Gemini only). See *Hosted vision* |

### Hosted vision (optional)
| Variable | Default | What it does |
|---|---|---|
| `GEMINI_API_KEY` | unset | Key for the hosted Gemini model. **Set it in the environment only, never in a file.** Unset: no hosted calls |
| `HOSTED_MODEL` | gemini-2.5-flash-lite | Gemini model for hosted walks. Must start with `gemini` |
| `HOSTED_MAX_CALLS_PER_DAY` | 5000 | Cost guard: most Gemini requests per server per UTC day. Past it, cached answers only until the next UTC day |
| `HOSTED_TIMEOUT_SEC` | 60 | Timeout for one Gemini request |
| `HOSTED_CHECK_TIMEOUT_SEC` | 5 | Timeout for the reachability check of the hosted model |
| `HOSTED_CHECK_TTL_SEC` | 300 | How long a reachability result is reused |
| `HOSTED_RETRY_MAX_SEC` | 10 | Longest server `Retry-After` wait before the one retry. Longer waits mean no retry |
| `HOSTED_CONCURRENCY` | 4 | Frames sent at once for a hosted walk |

- **Privacy and cost.** Hosted walk photos leave your machine and go to Google. Use it only if you accept both.
- **`auto` (default).** A new walk uses `HOSTED_MODEL` when `GEMINI_API_KEY` is set, the API answers and today's calls are under `HOSTED_MAX_CALLS_PER_DAY`. Otherwise the walk uses local Gemma. Each walk picks its model once.
- **`gemma`.** Local only. The hosted API is never called.
- **`hosted`.** Always Gemini. Fails if `GEMINI_API_KEY` is missing.
- **An explicit model wins.** An upload's `model` field, or `--model gemini-...` on the CLI, selects that model.
- **Retries.** One retry after a network error, HTTP 429, a 5xx or an unreadable answer. Other 4xx errors do not retry.
- **Call count** is kept in `backend/data/hosted_usage.json`, shared by all walks on the server. The key is never logged or shown by `/api/health`.

### Floors and stairs
| Variable | Default | What it does |
|---|---|---|
| `MAX_FLOOR` | 9 | Highest floor number. `start_floor` must be 1 to this value |
| `FLOOR_SIGNS` | 1 | Set to `0` to ignore floor signs and Gemma's floor hints |
| `FLOOR_SIGN_WEIGHT` | 2.0 | Evidence weight of one OCR-confirmed floor sign. A room number weighs 1, and a floor needs 2 |
| `STAIR_LANDING_GAP_S` | 9.0 | Longest gap (seconds) between two same-direction flights that can still be one climb with a landing between. **Untuned guess: tune on real walks** |
| `STAIR_LANDING_MAX_S` | 15.0 | Longer gap, allowed only when every frame between is a stairwell. **Untuned guess: tune on real walks** |

Set both `STAIR_LANDING_*` values to `0` to count every flight as a floor change. The `CLIMB_*` thresholds in
`config.py` (stair climbs from camera motion) were also set on synthetic renders. Re-check them on real walks.

### Gemma calls and failures
- Every call uses temperature 0 and the `GEMMA_NUM_*` limits above.
- **Model check first.** A job checks its model once. If the model is not pulled, the log shows `Run: ollama pull <model>` and the walk uses cached answers only.
- **Retries.** A dropped connection, timeout, HTTP 5xx or unreadable JSON is retried once. A 404 or other 4xx fails at once. Truncated output is not retried: raise the matching `GEMMA_NUM_PREDICT_*` value.
- **Bad frames are skipped.** The walk goes on.
- **Circuit breaker.** After `GEMMA_BREAKER_FAILS` failures in a row, the rest of that walk uses cached answers only, and the job log says so.
- **Cache first.** Each answer is read from `gemma_cache.jsonl` before Ollama is contacted. A rebuild with Ollama stopped keeps every Gemma result.
- **Model per walk.** The upload's model is used for every Gemma call of that walk. A rebuild reuses each walk's stored model.

### Limits and API behaviour
- Uploads over a limit get HTTP 413, and the partial file is deleted. A request whose declared size is more than
  the video limit + the sensor limit + 1 MiB is refused before the body is read.
- `/bfiles/<building>/frames/<file>` serves keyframe photos of approved walks only. The map, the Gemma cache and the videos are
  not downloadable.
- `start_floor` (on upload, or with `PATCH /api/buildings/<id>/videos/<video>`) must be 1 to `MAX_FLOOR`. Other
  values get HTTP 422.
- `PUT /api/buildings/<id>/map` takes only `nodes` and `edges`, and both are required (a body missing either gets
  422). Videos, processing status and suggested merges come from the server.
- Merging two places on different known floors, by hand or by accepting a suggestion, returns HTTP 400.
- `POST /api/buildings/<id>/route` takes an optional `avoid_stairs` boolean (default `false`). `true` uses only flat
  walks, ramps and elevators. If the only way climbs stairs, the API returns HTTP 400. Every answer has `step_free`,
  which is `true` when the route uses no stairs.
- Moderation endpoints: `GET /api/me/uploads` (signed in), and under `/api/owner/` the moderation list, the
  approve / reject / hide decision, the users list and the block switch. Owner routes need a session or `X-Owner-Token`.

## Frontend

The app is React with Vite. Its source is in `frontend/src/`. The build goes into `backend/static/`.

- **Landing page** (`#/` or `#/landing`). Lists the buildings and links into Directions and each building. Signed-out
  visitors who press Contribute go to sign-in first.
- **Theme.** Display type is **Instrument Serif**. Labels, eyebrows and small text use **JetBrains Mono**. Body text
  is **Atkinson Hyperlegible**. The app follows the OS light or dark setting. On desktop the tab bar is a centred pill.
- **Motion.** Built on `motion` (v14, imported from `motion/react`). Large numbers animate with `@number-flow/react`.
  Shared pieces live in `frontend/src/motion/` (see its README). Tab changes slide, dialogs trap focus, and toasts
  announce messages.
- **Reduced motion.** `main.jsx` wraps the app in `<MotionConfig reducedMotion="user">`. When the OS asks for less
  motion, movement is dropped and content still appears. The 3D map and confetti-style effects check `isLowEnd()` and
  reduced motion.
- **Lazy pages.** Map, Contribute, Owner and Record load on first visit, each in its own chunk. Directions and the
  landing page stay in the main bundle. A placeholder appears only after 180 ms, so a cached chunk never flashes it.
  The 3D map (three.js) sits inside the Map chunk and draws only while something moves.
- **Offline.** A service worker precaches the lazy chunks, so every page works offline after one online visit.

## Tests and evaluation

### Backend
Run from `backend/` after `uv sync` (which also installs pytest):
```bash
cd backend && uv run pytest -q
# or, with the venv active:  python -m pytest -q
```
**ffmpeg must be installed** (`brew install ffmpeg`). The tests that run the real normalize step
(`test_pipeline.py`, `test_jobs_pipeline.py`, `test_motion.py`) need it. Gemma tests replace the Ollama client, so no
model is needed to run them.

**Data-dir guard.** `backend/tests/conftest.py` points every test at a temporary folder, and it ignores `backend/.env`.
Before and after the run it compares `backend/data/` (the live data). If the run created, changed or removed anything
there, the run fails with `ERROR: the test run changed backend/data`. A test that writes to the live data is a bug, not a
flaky test. `tests/test_data_guard.py` checks the guard itself.

Other areas, for example:
- 90° turns are measured to within 0.1°; a glance (look at something and back) is not a turn.
- Routes mirror in reverse, stairs flip up↔down, and "no path" returns HTTP 400.
- Floors from room numbers (121 → 1, 208 → 2), elevator jumps without stairs, one misread number ignored.
- Uploads over the size limits get 413; `/bfiles` serves frames only; `PUT /map` ignores server-owned fields.
- Gemma: a missing model (404) fails at once; transient errors and bad JSON are retried once; the circuit breaker stops calls
  after repeated failures.
- Sign-in: link rate limits, single-use links, owner routes, and the `.env` loader (`test_env_file.py`, `test_config_mail.py`).
- Moderation and owner routes (`test_moderation.py`, `test_api_security*.py`, `test_job_action_auth.py`).

### Frontend (mocked, Playwright)

Install once, from `frontend/`:
```bash
cd frontend && npm install
npx playwright install chromium
```

Then run the suites:
```bash
npm run test:e2e        # vite dev server; the mocked suites in tests/run-all.mjs
npm run test:offline    # production build + vite preview; the offline suites
```

`tests/run-all.mjs` starts its own server on the first free port in 5470–5479 and stops it at the end. Each suite runs in
its own process group, so a crashed browser does not carry on. The exit code is 0 only when every suite passes.

**Manual: owner split against a real backend.** `frontend/tests/e2e-owner-realbackend.mjs` is not in `run-all`. It needs a
real backend and a log file. Start the backend in log mode with a test owner, then run the script:

```bash
# backend: MAIL_BACKEND=log, OWNER_EMAILS=you@example.org (in the shell or backend/.env), port 8030, logs to /path/be.log
# frontend: vite dev server on 5431, proxying /api to the backend
cd frontend && API_TARGET=http://127.0.0.1:8030 npx vite --host 127.0.0.1 --port 5431 --strictPort
# in another terminal:
cd frontend && LOG=/path/be.log BASE=http://127.0.0.1:5431 API=http://127.0.0.1:8030 node tests/e2e-owner-realbackend.mjs
```

It checks for owner-only text leaking to visitors and contributors, nav flicker on reload, and horizontal overflow at 390 px.
It exits 1 on a leak or an overflow.

### Evaluation harness
`backend/eval/` scores the whole pipeline (motion, keyframes, Gemma + OCR, mapping, floors, routing) on synthetic walkthroughs with known ground truth. It never touches `backend/data`.

```bash
cd backend
.venv/bin/python -m eval.run all                       # live, GEMMA_MODEL; about 20 min for all (M3 Pro)
.venv/bin/python -m eval.run two_floor --model gemma3:12b
.venv/bin/python -m eval.run all --oracle              # perfect perception from ground truth, about 1 min, no Ollama
```

Scenarios: `small_loop`, `dead_ends`, `confusers`, `two_floor`. See `backend/eval/README.md`.

## How it works in detail

| Stage | What happens |
|---|---|
| **Motion** | Lucas–Kanade corners plus a RANSAC similarity fit in centred coordinates. The **travel heading** is a running median over frames that move forward and look ahead, so standing and looking around never counts as a turn. If a phone gyro log is attached, its sign is auto-calibrated against vision |
| **Keyframes** | In each ±0.5 s window, pick the frame with the highest sharpness relative to the video's own rolling median, divided by motion. Blurry-only windows still keep their best frame, flagged `soft` |
| **Gemma + OCR** | Labels built on an unverified number are rebuilt. Without OCR (non-macOS) Gemma re-reads unclear signs with zoomed crops; with OCR, OCR reads the same crops |
| **Mapping** | Consecutive frames of one named place collapse. Revisits verified by Gemma ("same physical spot?"); automatic merges need geometric agreement, others go to a review queue. Each verified match constrains heading drift (fit `c0 + k·t`) |
| **Routing** | Walking an edge backwards uses headings + 180° and stairs up↔down, so one recording works in both directions. Across a link between two separate walks the step says where to head instead of guessing a turn |
| **App** | Directions use bent-arrow pictograms, with stair yellow for floor changes. Map: elevator-style floor selector. Owner page: moderation, videos, buildings, reports, users |

## How Gemma answers are checked
- **Grounded in OCR.** A word or number counts only if OCR read it as a whole token. `ROOM 12` does not confirm room `121`.
- **Room numbers.** `RM 118` and `ROOM-118` become `118`. `118A` is kept. Numbers from posters, phone numbers, dates and prices are dropped.
- **Directional signs.** `ROOMS 120-130 →` names rooms down the hall, so it never gives this place a room number. The text stays in the place's texts.
- **Floors.** A floor hint needs a floor word: `LEVEL 2`, `2ND FLOOR` or `FL2`. Negative and basement floors (`LEVEL -1`, `BASEMENT`) are rejected.
- **Photos.** Phone EXIF rotation is applied before analysis. OCR reads the full frame and zoomed left and right crops.
- **Merges.** Places with different room or plaque numbers never merge automatically. A Gemma "different" does not block a strong ORB match when no plaques conflict, because gemma3:4b was unreliable at same-place calls. Manual merges skip this check. The floor rule still applies.
- **Known limits.** gemma3:4b can call a blank corridor or an EXIT sign "same place". That alone never merges two places; a geometric match is still needed.

## How floors work
- **Confirmed evidence only.** A walk gets a floor from two room numbers that agree, or from one floor sign that OCR confirmed (`LEVEL 2`, `2ND FLOOR`, `FL2`). Without OCR, a floor sign needs a second reading.
- **Ignored readings.** Basement rooms (`B112`), hyphenated numbers (`2-105`) and 4-digit numbers (usually phone extensions) are not floor evidence. A sign naming several floors (a directory) is ignored.
- **Stairs.** A staircase with a landing counts one floor per storey, not one per flight. A single staircase on each of two adjacent floors is linked as a guess, flagged "check" in the log, and Gemma can still veto it.
- **Climbs from motion.** Gemma often calls a whole flight "stairs ahead". Camera motion measures the climb, but only when Gemma saw stairs nearby, so a flat walk never changes floor from motion alone. Thresholds are `CLIMB_*` in `config.py`.
- **Unknown start floor.** A walk with no start floor takes its floor from confirmed evidence or from a match to a known place.
- **Merging.** Places on two different known floors cannot merge. The API returns HTTP 400 with the reason.
- **Layout.** Stairs and elevators sit at the same x/y on every floor they connect, so floors stack directly above each other on the map.

## Recording tips
- 1080p with stabilization on. Hold the phone upright and point it where you walk. Turn at a steady pace.
- **Pause about 1 s** at plaques, junctions and stairs. Glancing at a plaque and back is fine.
- Walk corridors in both directions when you can, and pass known spots again to close loops.
- Best accuracy: record from the **Record** tab on the phone, which logs the gyro alongside the video, and enter the floor the walk starts on.
- Overlap something already mapped (a room number, named sign or staircase) so the walk links into the map.
- Film staircases from both floors and **climb them**, so the climb is recorded and the floors link.
- After arriving on a new floor, point the camera at a room number or floor sign within a few seconds. That corrects the stair count.
- Known limits: elevators with no visible sign may not be detected as a floor change; a climb of several storeys with no sign on arrival counts as one floor; basements are not modelled.
- After you upload, the walk is checked automatically. Check **My uploads** on the Contribute tab for its status.

## Persistence
Each building is a portable folder, `backend/data/buildings/<id>/`:
- `map.json`: the graph, plus `processed_with` (model and prompt version of the last Gemma-processed walk)
- `frames/`: keyframe photos of **approved** walks only. This is the only folder served over HTTP
- `reports.jsonl`: problem reports and their status changes, one JSON line each (created only once a report is filed)
- `gemma_cache.jsonl`: Gemma answers, one JSON line per answer. Each new answer is appended. The file is only
  rewritten to remove duplicates when it is loaded. An old `gemma_cache.json` is converted the first time the cache is used, and kept as `gemma_cache.json.bak`
- `videos/<vid>/`: normalized video, track, `walk.json` (keyframes, events, `gemma`, `gemma_frames_ok`, `model`, moderation record)
- `videos/<vid>/rejected_frames/`: blurred keyframes of walks that are not approved. Never served

Only `frames/` is served over HTTP. The rest stays on disk. Copy the folder to the demo machine and everything works offline.

## Screenshots

The images above are from an earlier version of the app. They show the Build tab, which is now **Contribute**. They
do not show the landing page, the owner page or the new theme. New screenshots will replace them.

## Changelog

### Unreleased

- **Animation pass.** Motion layer (`motion`, `@number-flow/react`): tab slide and pill, dialogs with focus trap,
  toasts, skeletons on slow loads, press feedback, and animated numbers. All of it respects reduced motion.
- **Owner and visitor split.** Visitors get Landing, Directions, a read-only Map and Record. Signed-in contributors get
  **Contribute** with **My uploads** and their statuses. Owners get the **Owner** page (`#/owner`): Moderation,
  Videos & map, Buildings, Reports and Users. Owners are set in `OWNER_EMAILS`.
- **Moderation.** Every walk gets automatic checks (faces blurred, content checked). The owner can approve, reject or hide
  any walk. Frames of walks that are not approved are not public. Owners can block a contributor from uploading.
- **Email sign-in.** Magic-link sign-in by email (no passwords), with rate limits per IP and per address. `MAIL_BACKEND=log`
  for local dev, `smtp` for real mail. The owner key and invite links are gone; access is by sign-in.
- **Landing redesign.** New landing page and theme: Instrument Serif for display type, JetBrains Mono for labels.
- **Map3D on demand.** The 3D map draws only while something is moving and stops when idle.
- **Smaller first load.** Heavy pages (Map, Contribute, Owner, Record) load on first visit. The initial JavaScript bundle
  went from 310 kB to 138 kB gzip.
- **Test isolation fix.** The test suite writes only to a temporary data folder. A guard fails the run if anything
  under `backend/data/` changes.
- **`.env` loading.** `backend/.env` is read at startup. Shell variables win. See [Settings](#settings).
- **Start script.** `scripts/start.sh` (with `--build`) starts the backend and checks the mail settings first.

## Contributing
Working on the Street View viewer? Read [docs/CONTRIBUTING-streetview.md](docs/CONTRIBUTING-streetview.md) for the branch and file rules.

## Contributors
- Neel
- Evan
- Prachet
- Sudarshan
