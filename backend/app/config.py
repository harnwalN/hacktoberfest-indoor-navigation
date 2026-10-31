"""Central configuration. Every value can be overridden with an environment variable."""
import os
import re
from pathlib import Path


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BACKEND_DIR / "data"))
BUILDINGS_DIR = DATA_DIR / "buildings"
STATIC_DIR = BACKEND_DIR / "static"

# --- Access control (auth.py) ---
# Every /api route except /api/health needs a token: the owner key, or an invite token for one building.
# AUTH_REQUIRED=0 turns all checks off (local development only). There is no localhost bypass: the tunnel
# delivers public traffic from 127.0.0.1 too.
AUTH_REQUIRED = os.getenv("AUTH_REQUIRED", "1") != "0"
# Owner key from the environment. Unset: a key is generated on first start and only its hash is kept in DATA_DIR/auth.json.
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()
ADMIN_TOKEN_MIN_CHARS = 24  # with AUTH_REQUIRED=1 the server refuses to start when ADMIN_TOKEN is set and shorter
FRAME_URL_TTL_SEC = int(os.getenv("FRAME_URL_TTL_SEC", 86400))  # how long a signed photo URL stays valid (one day)
AUTH_MAX_FAILS = int(os.getenv("AUTH_MAX_FAILS", 20))  # failed auth attempts per client IP per minute before HTTP 429

# --- Upload limits (requests over these get HTTP 413) ---
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", 2048))  # one walk video
MAX_SENSOR_BYTES = int(os.getenv("MAX_SENSOR_BYTES", 50 * 1024 * 1024))  # sensors.json recorded with a video
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", 25 * 1024 * 1024))  # one manually added photo

# --- Gemma (local, via Ollama) ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
# 4b reads plaques well and is ~2.5x faster on a 16 GB M-series Mac; pass model=gemma3:12b per upload
# (or set GEMMA_MODEL) for a slower, higher-quality pre-processing run.
GEMMA_MODEL = os.getenv("GEMMA_MODEL", "gemma3:4b")
GEMMA_FALLBACK_MODEL = os.getenv("GEMMA_FALLBACK_MODEL", "gemma3:12b")
GEMMA_IMAGE_MAX = int(os.getenv("GEMMA_IMAGE_MAX", 896))  # Gemma 3 vision encoder native size
GEMMA_TIMEOUT_SEC = _f("GEMMA_TIMEOUT_SEC", 300)  # one chat request (model load + image + answer)
GEMMA_LIST_TIMEOUT_SEC = _f("GEMMA_LIST_TIMEOUT_SEC", 3)  # list/show/health: fail fast when Ollama is down
GEMMA_CONCURRENCY = int(os.getenv("GEMMA_CONCURRENCY", 3))  # pair with OLLAMA_NUM_PARALLEL>=3
# How long Ollama keeps the model loaded after a request: a duration ("30m", "1h") or seconds ("-1" = forever).
_keep_alive = os.getenv("GEMMA_KEEP_ALIVE", "30m").strip()
GEMMA_KEEP_ALIVE: str | float = float(_keep_alive) if re.fullmatch(r"-?\d+(\.\d+)?", _keep_alive) else _keep_alive
GEMMA_NUM_CTX = int(os.getenv("GEMMA_NUM_CTX", 4096))  # fixed so every request reuses the loaded model
GEMMA_NUM_PREDICT_ANALYZE = int(os.getenv("GEMMA_NUM_PREDICT_ANALYZE", 512))  # answer token cap, frame analysis
GEMMA_NUM_PREDICT_COMPARE = int(os.getenv("GEMMA_NUM_PREDICT_COMPARE", 320))  # same place / same stairwell
GEMMA_BREAKER_FAILS = int(os.getenv("GEMMA_BREAKER_FAILS", 3))  # consecutive failures that stop Gemma for a walk

# --- Vision model choice (gemma.choose_model): hosted Gemini (hosted.py) or local Gemma (Ollama) ---
# auto: HOSTED_MODEL when GEMINI_API_KEY is set, the API answers and the daily call limit has room; else local Gemma.
# gemma: local Gemma only, the hosted API is never called. hosted: HOSTED_MODEL only (needs the key).
# A walk picks its model once; an explicit model (upload form field, CLI --model, eval --model) always wins.
VISION_PROVIDER = os.getenv("VISION_PROVIDER", "auto").strip().lower()
if VISION_PROVIDER not in ("auto", "gemma", "hosted"):
    raise ValueError(f"VISION_PROVIDER must be auto, gemma or hosted (got {VISION_PROVIDER!r})")
# Env only: never logged, never returned by /api/health. Unset: no hosted calls at all.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
HOSTED_MODEL = os.getenv("HOSTED_MODEL", "gemini-2.5-flash-lite").strip()  # cheapest Gemini model with image input
if not HOSTED_MODEL.startswith("gemini"):
    raise ValueError(f"HOSTED_MODEL must be a Gemini model name (got {HOSTED_MODEL!r})")
HOSTED_TIMEOUT_SEC = _f("HOSTED_TIMEOUT_SEC", 60)  # one generateContent request
HOSTED_CHECK_TIMEOUT_SEC = _f("HOSTED_CHECK_TIMEOUT_SEC", 5)  # the reachability GET (model metadata)
HOSTED_CHECK_TTL_SEC = _f("HOSTED_CHECK_TTL_SEC", 300)  # how long a reachability result is reused
HOSTED_RETRY_MAX_SEC = _f("HOSTED_RETRY_MAX_SEC", 10)  # longest Retry-After waited before the one retry
HOSTED_CONCURRENCY = int(os.getenv("HOSTED_CONCURRENCY", 4))  # frames in flight for a hosted walk
HOSTED_MAX_CALLS_PER_DAY = int(os.getenv("HOSTED_MAX_CALLS_PER_DAY", 5000))  # cost guard, per server, UTC day

# On-device OCR grounds Gemma's sign text so it cannot invent room numbers (ocr.py).
# auto: Apple Vision on macOS, else RapidOCR when installed (uv sync --extra linux-ocr), else none (OCR skipped, logged).
# vision / rapidocr: that engine only; none, with a log, when it is not available here. none: OCR off.
OCR_ENGINE = os.getenv("OCR_ENGINE", "auto").strip().lower()
if OCR_ENGINE not in ("auto", "vision", "rapidocr", "none"):
    raise ValueError(f"OCR_ENGINE must be auto, vision, rapidocr or none (got {OCR_ENGINE!r})")
OCR_ENABLED = os.getenv("OCR_ENABLED", "1") != "0"  # master switch: 0 turns OCR off whatever the engine
OCR_MIN_CONFIDENCE = _f("OCR_MIN_CONFIDENCE", 0.5)  # Apple Vision lines
# RapidOCR scores on its own scale (not Vision's): on the 10-frame check real sign text scored 0.90 and up, and the
# noise lines 0.62 and below. Tune on real footage (eval/README "Real or Habitat data") before changing it.
OCR_MIN_CONFIDENCE_RAPIDOCR = _f("OCR_MIN_CONFIDENCE_RAPIDOCR", 0.7)

# --- Video normalization ---
ANALYSIS_FPS = int(os.getenv("ANALYSIS_FPS", 10))
ANALYSIS_LONG_SIDE = int(os.getenv("ANALYSIS_LONG_SIDE", 640))
HQ_FPS = int(os.getenv("HQ_FPS", 15))
HQ_LONG_SIDE = int(os.getenv("HQ_LONG_SIDE", 1280))
# Horizontal FOV across the *long* side of the frame (iPhone main camera, 16:9 video ~ 69 deg).
LONG_SIDE_FOV_DEG = _f("LONG_SIDE_FOV_DEG", 69.0)

# --- Motion / event detection ---
TURN_RATE_DEG_S = _f("TURN_RATE_DEG_S", 12.0)  # smoothed yaw rate that counts as "turning"
TURN_MIN_DEG = _f("TURN_MIN_DEG", 35.0)  # total rotation for a turn event
GLANCE_RETURN_DEG = _f("GLANCE_RETURN_DEG", 25.0)  # turn + opposite turn that cancel = a glance
GLANCE_MAX_GAP_SEC = _f("GLANCE_MAX_GAP_SEC", 4.0)
JOG_MAX_GAP_SEC = _f("JOG_MAX_GAP_SEC", 5.0)  # opposite travel turns this close that cancel = sidestep
FWD_WALKING = _f("FWD_WALKING", 8.0)  # smoothed expansion (x1000/frame) that means "walking forward"
TRAVEL_WINDOW_SEC = _f("TRAVEL_WINDOW_SEC", 4.0)  # running median over forward-looking samples
TRAVEL_TURN_RATE_DEG_S = _f("TRAVEL_TURN_RATE_DEG_S", 8.0)
GLANCE_DEG = _f("GLANCE_DEG", 35.0)  # camera this far off the walking direction = looking at something
DWELL_FLOW_PX = _f("DWELL_FLOW_PX", 1.5)  # median feature motion per analysis frame
DWELL_MIN_SEC = _f("DWELL_MIN_SEC", 0.7)

# --- Keyframes ---
HEARTBEAT_SEC = _f("HEARTBEAT_SEC", 3.0)
SCENE_CHANGE_HAMMING = int(os.getenv("SCENE_CHANGE_HAMMING", 18))
MIN_CANDIDATE_GAP_SEC = _f("MIN_CANDIDATE_GAP_SEC", 1.5)  # for scene/heartbeat candidates
DUPLICATE_HAMMING = int(os.getenv("DUPLICATE_HAMMING", 5))
CANDIDATE_WINDOW_SEC = _f("CANDIDATE_WINDOW_SEC", 0.5)
SOFT_SCORE = _f("SOFT_SCORE", 0.6)

# --- Mapping ---
NODE_MIN_GAP_SEC = _f("NODE_MIN_GAP_SEC", 1.2)
SAME_PLACE_WINDOW_SEC = _f("SAME_PLACE_WINDOW_SEC", 8.0)  # collapse consecutive frames of one place
# A corridor turn this close to the first/last frame of a climb is the turn onto (or off) the flight: the landing
STAIR_APPROACH_SEC = _f("STAIR_APPROACH_SEC", 3.0)
ORB_STRONG_INLIERS = int(os.getenv("ORB_STRONG_INLIERS", 40))
ORB_WEAK_INLIERS = int(os.getenv("ORB_WEAK_INLIERS", 20))
AUTO_MERGE_CONFIDENCE = _f("AUTO_MERGE_CONFIDENCE", 0.8)
MAX_FLOOR = int(os.getenv("MAX_FLOOR", 9))  # room-number floors above this are treated as misreads
DEFAULT_FLOOR = int(os.getenv("DEFAULT_FLOOR", 1))  # ground floor number when nothing says otherwise
WALK_SPEED_MPS = _f("WALK_SPEED_MPS", 1.25)  # typical indoor walking pace while filming
DEFAULT_EDGE_M = _f("DEFAULT_EDGE_M", 6.0)  # length used when a connection has no measurement
STAIRS_COST_M = _f("STAIRS_COST_M", 15.0)  # routing cost of one flight (time/effort, in metres of walking)
TURN_COST_M = _f("TURN_COST_M", 2.0)  # small penalty per place passed, prefers simpler routes
ROOM_NUMBER_FLOORS = os.getenv("ROOM_NUMBER_FLOORS", "1") != "0"  # 121 -> floor 1, 208 -> floor 2
FLOOR_SIGNS = os.getenv("FLOOR_SIGNS", "1") != "0"  # 'LEVEL 3', '2ND FLOOR' and Gemma's floor_hint as evidence
# Evidence weight of one floor sign; one room number weighs 1 and a floor needs 2 to be confirmed, so the
# default lets one clear floor sign stand alone while a single room number needs a second agreeing one.
FLOOR_SIGN_WEIGHT = _f("FLOOR_SIGN_WEIGHT", 2.0)
SNAP_TOLERANCE_DEG = _f("SNAP_TOLERANCE_DEG", 20.0)
# Stair landings: two flights in the same direction count as ONE floor change when only a landing lies
# between them (no room numbers, floor signs or doors). Gap = time from the last on-stairs keyframe of one
# flight to the first of the next, so it includes up to one keyframe interval on each side of the landing.
# Up to STAIR_LANDING_GAP_S any landing-like frames qualify; up to STAIR_LANDING_MAX_S only if every frame
# between is a stairwell (place_type stairs, or the next flight ahead). Set both to 0 to count every flight.
# Both defaults come from synthetic walks only: TUNE THEM against real walk.json files (landing durations of
# switchback stairwells vs short corridors between separate staircases).
STAIR_LANDING_GAP_S = _f("STAIR_LANDING_GAP_S", 9.0)  # about 3 heartbeats (3 s each), so a landing with up to two keyframes joins; tune on real walk.json
STAIR_LANDING_MAX_S = _f("STAIR_LANDING_MAX_S", 15.0)

# --- Stair climbs from camera motion (motion.detect_climbs, floors.mark_climbs) ---
# Gemma often calls every frame of a flight "stairs ahead" and never "on the stairs", so the climb is also
# measured from the vertical image motion. A motion climb only counts when Gemma saw stairs near it (stairs
# ahead, or a stairwell) within CLIMB_HINT_SEC before it or during it, so a flat walk never changes floor from
# motion alone. All defaults were set on synthetic renders (eval/walk.py), not on phone footage: keep them
# conservative and re-check them on real walks (track.json 'pitch' + walk.json Gemma stairs labels).
CLIMB_DETECT = os.getenv("CLIMB_DETECT", "1") != "0"  # 0 = stairs from Gemma's on_stairs labels only
# Flow: the scene keeps sliding vertically (camera rising/falling), relative to level walking, at least this
# fast (deg/s, running median over CLIMB_WINDOW_SEC). Measured on eval renders: climb ~4.3 deg/s; level walks
# stay <= 1.2, except ~1.5 s bursts up to ~4 when looking up at a sign (cut by CLIMB_MIN_SEC + the Gemma gate).
CLIMB_FLOW_DEG_S = _f("CLIMB_FLOW_DEG_S", 2.5)
CLIMB_WINDOW_SEC = _f("CLIMB_WINDOW_SEC", 3.0)
# Tilt: the camera tilts by CLIMB_TILT_DEG within CLIMB_TILT_STEP_SEC, holds it away from the walker's level
# before and after for most of the stretch, and tilts back (looking down a flight while descending). Eval
# renders: descent ~24 deg; level walks <= 3, except short looks at high signs (~21 deg, < CLIMB_MIN_SEC).
CLIMB_TILT_DEG = _f("CLIMB_TILT_DEG", 15.0)
CLIMB_TILT_STEP_SEC = _f("CLIMB_TILT_STEP_SEC", 1.5)
CLIMB_MIN_SEC = _f("CLIMB_MIN_SEC", 4.0)  # shorter = a few steps or a look at the floor, not a storey
CLIMB_MAX_SEC = _f("CLIMB_MAX_SEC", 60.0)  # longer = not one climb (drift, a long look down)
CLIMB_MERGE_GAP_SEC = _f("CLIMB_MERGE_GAP_SEC", 2.0)  # flow dips this short (a landing) do not split a climb
CLIMB_HINT_SEC = _f("CLIMB_HINT_SEC", 5.0)  # Gemma's stairs hint may come this long before the climb starts
# A floor sign read in the last seconds before a climb (from motion or Gemma's on_stairs frames) is usually the
# stairwell sign of the floor the stairs lead to ('LEVEL 2' seen from the bottom of the flight): it is not floor
# evidence for either floor (floors._approach). Room numbers there still count. 0 = off.
CLIMB_SIGN_APPROACH_SEC = _f("CLIMB_SIGN_APPROACH_SEC", 5.0)

# --- "Something's wrong here" reports (reports.py) ---
REPORTS_PER_HOUR = int(os.getenv("REPORTS_PER_HOUR", 20))  # reports one reporter may send for one building per hour
REPORTS_MAX = int(os.getenv("REPORTS_MAX", 5000))  # reports one building may hold; past this new ones get 507
