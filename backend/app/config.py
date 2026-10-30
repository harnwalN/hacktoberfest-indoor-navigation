"""Central configuration. Every value can be overridden with an environment variable."""
import os
from pathlib import Path


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BACKEND_DIR / "data"))
BUILDINGS_DIR = DATA_DIR / "buildings"
STATIC_DIR = BACKEND_DIR / "static"

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
GEMMA_TIMEOUT_SEC = _f("GEMMA_TIMEOUT_SEC", 300)
GEMMA_CONCURRENCY = int(os.getenv("GEMMA_CONCURRENCY", 3))  # pair with OLLAMA_NUM_PARALLEL>=3

# On-device OCR (macOS Vision) grounds Gemma's sign text so it cannot invent room numbers.
OCR_ENABLED = os.getenv("OCR_ENABLED", "1") != "0"
OCR_MIN_CONFIDENCE = _f("OCR_MIN_CONFIDENCE", 0.5)

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
