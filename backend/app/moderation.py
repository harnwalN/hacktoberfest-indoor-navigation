"""Automatic moderation of crowd-uploaded walk keyframes: no human review for the common case.

Two steps, both meant to run on every keyframe of a contributor's walk before it can reach the public map:

1. Face blur, blur_faces(frame_bgr) -> frame_bgr. Faces are pixelated and blurred on this server, before a frame is
   stored, shown, or sent to a vision model. Detector: OpenCV YuNet (cv2.FaceDetectorYN) with
   face_detection_yunet_2023mar.onnx from opencv_zoo (MIT licence), bundled in backend/models/ and checked against a
   pinned SHA-256 when the detector is first created. The model is never downloaded at run time. Fail closed: when the
   model is missing, does not match its pin, or cannot run, FaceBlurError is raised and no unblurred frame comes back.
   Blur is one safeguard, not a guarantee (see .agent-reports/research-crowd-privacy.md).

2. Content check, classify_content(frame_bgr_or_path, model=None) -> ModerationVerdict. Asks the vision model whether
   the image is a plausible photo inside a building (hallway, room, stairwell, lift, entrance, wayfinding sign) and
   whether it is clearly inappropriate. The call goes through gemma._chat, the same path as the walk's analysis, so
   VISION_PROVIDER / HOSTED_MODEL in config.py decide the model: Ollama (local Gemma) or hosted Gemini. A model name
   is never hard-coded here; pass `model` to override it. Fail safe: any error gives approved=False with the error in
   `notes`. Nothing is approved on failure.

moderate_keyframe(frame_bgr, model=None) runs both in the only safe order: blur first, then classify the blurred copy.
Use it from the pipeline so the vision API never receives an unblurred face.

Not wired into jobs.py yet. The caller decides what to do with a rejected verdict.
"""
import hashlib
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from . import config, gemma

log = logging.getLogger(__name__)

# --- face blur ---------------------------------------------------------------------------------------------------
# Pinned model: opencv_zoo at commit f12e12798e8314f7c074a6656816c048dcc95b7a, same bytes as redact.py on soo-crowd.
FACE_MODEL_PATH = config.BACKEND_DIR / "models" / "face_detection_yunet_2023mar.onnx"
FACE_MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
FACE_MODEL_BYTES = 232589
FACE_SCORE_THRESHOLD = 0.5  # low on purpose: favour recall (research: a missed face is worse than a blurred non-face)
NMS_THRESHOLD = 0.3  # YuNet's own overlap threshold, also used to merge one face found at two scales
TOP_K = 5000
DETECT_SCALES = (1.0, 2.0)  # 2x finds the small faces (about 15 to 20 px) that 1x misses
BOX_GROWTH = 1.3  # blurred box = face box, 30% wider and taller, centred on the face
PIXEL_GRID = 6  # the box is pixelated to this many cells across (and down, at most), then smoothed

_detector = None
_detector_lock = threading.Lock()  # a cv2 detector is not thread-safe


class FaceBlurError(RuntimeError):
    """Faces could not be blurred. The frame must not be stored, shown or classified."""


def _model_ok(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size != FACE_MODEL_BYTES:
        return False
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest() == FACE_MODEL_SHA256


def _get_detector():
    """The YuNet detector, created once per process. The model's checksum is verified at that point only."""
    global _detector
    if _detector is None:
        path = Path(FACE_MODEL_PATH)
        if not path.is_file():
            raise FaceBlurError(f"face model missing: {path} (it is bundled in backend/models/; restore it from git)")
        if not _model_ok(path):
            raise FaceBlurError(f"face model {path} does not match its pinned SHA-256 or size; not used")
        try:
            _detector = cv2.FaceDetectorYN.create(str(path), "", (640, 360), FACE_SCORE_THRESHOLD, NMS_THRESHOLD,
                                                  TOP_K)
        except cv2.error as exc:
            raise FaceBlurError(f"OpenCV could not load the face model {path}: {exc}") from exc
    return _detector


def _check_frame(frame: object) -> np.ndarray:
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 \
            or frame.size == 0:
        raise ValueError("expected a non-empty BGR image: numpy uint8 array of shape (height, width, 3)")
    return frame


def detect_faces(frame_bgr: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Face boxes (x, y, w, h) in the frame's own pixels, one face found at several scales merged."""
    h, w = frame_bgr.shape[:2]
    boxes: list[list[float]] = []
    scores: list[float] = []
    with _detector_lock:
        det = _get_detector()
        for s in DETECT_SCALES:
            frame = frame_bgr if s == 1.0 else cv2.resize(
                frame_bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_LINEAR if s > 1 else cv2.INTER_AREA)
            try:
                det.setInputSize((frame.shape[1], frame.shape[0]))
                _, faces = det.detect(frame)
            except cv2.error as exc:
                raise FaceBlurError(f"face detection failed at scale {s}: {exc}") from exc
            if faces is None:
                continue
            for row in faces:  # YuNet row: x, y, w, h, 10 landmark values, score
                boxes.append([float(row[0]) / s, float(row[1]) / s, float(row[2]) / s, float(row[3]) / s])
                scores.append(float(row[14]))
    if not boxes:
        return []
    ints = [[int(round(v)) for v in b] for b in boxes]
    keep = np.array(cv2.dnn.NMSBoxes(ints, scores, FACE_SCORE_THRESHOLD, NMS_THRESHOLD)).reshape(-1)
    return [tuple(boxes[int(i)]) for i in keep]


def _blur_box(x: float, y: float, bw: float, bh: float, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    cx, cy = x + bw / 2, y + bh / 2
    gw, gh = bw * BOX_GROWTH, bh * BOX_GROWTH
    x0 = max(0, int(np.floor(cx - gw / 2)))
    y0 = max(0, int(np.floor(cy - gh / 2)))
    x1 = min(img_w, int(np.ceil(cx + gw / 2)))
    y1 = min(img_h, int(np.ceil(cy + gh / 2)))
    return x0, y0, x1, y1


def _pixelate_and_blur(img: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    """In place on img[y0:y1, x0:x1]: a coarse grid of colour cells, then a Gaussian blur over the cells. A light blur
    alone can be undone, so the detail is removed first."""
    roi = img[y0:y1, x0:x1]
    rh, rw = roi.shape[:2]
    if rh < 1 or rw < 1:
        return
    gx, gy = min(rw, PIXEL_GRID), min(rh, PIXEL_GRID)
    cells = cv2.resize(roi, (gx, gy), interpolation=cv2.INTER_AREA)
    coarse = cv2.resize(cells, (rw, rh), interpolation=cv2.INTER_NEAREST)
    sigma = max(2.0, min(rw, rh) / 6.0)
    img[y0:y1, x0:x1] = cv2.GaussianBlur(coarse, (0, 0), sigma)


def blur_faces(frame_bgr: np.ndarray) -> np.ndarray:
    """A copy of the BGR frame with every detected face pixelated and blurred. The input is not changed.
    Raises ValueError for a bad array and FaceBlurError when the detector cannot run (never a silent pass-through)."""
    _check_frame(frame_bgr)
    out = frame_bgr.copy()
    ih, iw = out.shape[:2]
    for x, y, bw, bh in detect_faces(frame_bgr):
        _pixelate_and_blur(out, *_blur_box(x, y, bw, bh, iw, ih))
    return out


# --- content check -----------------------------------------------------------------------------------------------
MODERATION_NUM_PREDICT = 256  # answer token cap: a short JSON object
NOTES_MAX = 300  # longest error text kept in a verdict

CONTENT_PROMPT = """You are checking one frame from a walkthrough video of a building. A volunteer uploaded it so that
other people can find their way around the building. Fill in the JSON fields.
- description: what the photo shows, a short noun phrase (at most 12 words).
- indoor_building: true if this is a plausible photo taken inside a building: a hallway, room, stairwell, lift or
  elevator, entrance, or a wayfinding sign (door plaque, room number, exit or directional sign). False for outdoor
  scenes, screens, documents, selfies, animals, memes, objects, or anything unrelated to finding one's way around.
- inappropriate: true for nudity, sexual content, violence or gore, hate symbols, or other clearly unsuitable
  material. Otherwise false.
- reason: one short sentence explaining the answer.
Faces may be blurred on purpose. Do not try to identify anyone, and do not reject a photo only because of blurring."""


class ContentCheck(BaseModel):
    """The vision model's answer. Field order is the generation order: the description comes before the verdicts."""

    description: str = Field(description="What the photo shows, a short noun phrase (at most 12 words)")
    indoor_building: bool = Field(description="True for a plausible photo taken inside a building, or a wayfinding "
                                              "sign; false for outdoor, screen, document, selfie or unrelated images")
    inappropriate: bool = Field(description="True for nudity, sexual content, violence, gore, hate symbols or other "
                                            "clearly unsuitable material")
    reason: str = Field(description="One short sentence explaining the answer")


@dataclass(frozen=True)
class ModerationVerdict:
    approved: bool  # True only for a plausible indoor building photo with no inappropriate content
    reason: str  # one line for the moderation log
    notes: str = ""  # the error, when the check could not run (then approved is always False)
    model: str = ""  # the model that answered; "" when no model was reached
    error: bool = False  # True when the check failed (model, API, image read) rather than judging the photo


def _failed(message: str, model: str = "") -> ModerationVerdict:
    log.warning("content check failed, rejected: %s", message)
    return ModerationVerdict(approved=False, reason="content check failed (rejected)", notes=message[:NOTES_MAX],
                             model=model, error=True)


def _model_jpeg(frame_bgr_or_path) -> bytes:
    """The image as the vision model gets it: upright, RGB, thumbnailed, JPEG (the same encoder as the walk's
    analysis, gemma._encode)."""
    if isinstance(frame_bgr_or_path, (str, Path)):
        return gemma._encode(gemma._open(Path(frame_bgr_or_path)))
    frame = _check_frame(frame_bgr_or_path)
    return gemma._encode(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))


def _decide(check: ContentCheck, model: str) -> ModerationVerdict:
    why = check.reason.strip() or check.description.strip() or "no reason given"
    if check.inappropriate:
        return ModerationVerdict(approved=False, reason=f"inappropriate content: {why}", model=model)
    if not check.indoor_building:
        return ModerationVerdict(approved=False, reason=f"not a plausible indoor building photo: {why}", model=model)
    return ModerationVerdict(approved=True, reason=why, model=model)


def classify_content(frame_bgr_or_path, model: str | None = None) -> ModerationVerdict:
    """Ask the vision model whether a frame is a plausible indoor building photo and not inappropriate.

    frame_bgr_or_path: a BGR numpy array (as from OpenCV) or a path to an image file. Pass the blurred frame
    (moderate_keyframe does this for you): the model may be a hosted API.
    model: model name, or None for the model config.py selects (VISION_PROVIDER / HOSTED_MODEL, via gemma.choose_model).
    Never raises for a model, API or image error: returns approved=False with the error in notes."""
    try:
        image = _model_jpeg(frame_bgr_or_path)
    except (OSError, ValueError, TypeError) as exc:
        return _failed(f"could not read the image: {type(exc).__name__}: {exc}")
    use = ""
    try:
        use = model or gemma.choose_model()
        check = gemma._chat(use, CONTENT_PROMPT, [image], ContentCheck, MODERATION_NUM_PREDICT)
    except Exception as exc:  # GemmaUnavailable, HostedError wrapped by gemma, anything else: fail safe
        return _failed(f"vision model call failed: {type(exc).__name__}: {exc}", model=use)
    return _decide(check, use)


def moderate_keyframe(frame_bgr: np.ndarray, model: str | None = None) -> tuple[np.ndarray, ModerationVerdict]:
    """Blur every face, then classify the blurred copy. Returns (blurred frame, verdict). Store the blurred frame,
    never the input. Raises FaceBlurError when blurring fails: then the frame is neither stored nor classified."""
    blurred = blur_faces(frame_bgr)
    return blurred, classify_content(blurred, model=model)
