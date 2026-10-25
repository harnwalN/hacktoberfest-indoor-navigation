"""Central configuration. Every value can be overridden with an environment variable."""
import os
from pathlib import Path


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BACKEND_DIR / "data"))
BUILDINGS_DIR = DATA_DIR / "buildings"
STATIC_DIR = BACKEND_DIR / "static"

# --- Gemma (local, via Ollama) ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
# 4b reads plaques well and is ~2.5x faster on a 16 GB M-series Mac; pass model=gemma3:12b per upload
# (or set GEMMA_MODEL) for a slower, higher-quality pre-processing run.
GEMMA_MODEL = os.getenv("GEMMA_MODEL", "gemma3:4b")
GEMMA_FALLBACK_MODEL = os.getenv("GEMMA_FALLBACK_MODEL", "gemma3:12b")
GEMMA_IMAGE_MAX = int(os.getenv("GEMMA_IMAGE_MAX", 896))  # Gemma 3 vision encoder native size
GEMMA_TIMEOUT_SEC = _f("GEMMA_TIMEOUT_SEC", 300)
GEMMA_CONCURRENCY = int(os.getenv("GEMMA_CONCURRENCY", 3))  # pair with OLLAMA_NUM_PARALLEL>=3

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
SNAP_TOLERANCE_DEG = _f("SNAP_TOLERANCE_DEG", 20.0)
