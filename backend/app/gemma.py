"""Local Gemma (via Ollama) for landmark/OCR analysis and same-place verification.

Runs fully offline once the model is pulled. Every answer is cached on disk keyed by
(image hash, model, prompt version), so reprocessing a building is free and deterministic.
"""
import hashlib
import io
import threading
from pathlib import Path

import ollama
from PIL import Image

from . import config
from .schemas import FrameAnalysis, SamePlace
from .store import read_json, write_json_atomic

PROMPT_VERSION = "v3"

ANALYZE_PROMPT = """You are helping map the inside of a building from a walkthrough video.
The image is the camera view, looking in the walking direction.

Only WAYFINDING signs count: door plaques, room numbers, room/office name signs, restroom, exit,
stairs and elevator signs, directory or floor signs. IGNORE posters, flyers, bulletin boards,
notices, warning stickers, logos and advertising.

Fill in the JSON fields:
- has_sign: true only if a wayfinding sign is clearly legible in THIS image.
- texts: the wayfinding sign texts exactly as written. Empty list if none. Never guess or invent.
- room_numbers: door/room numbers that are actually visible on a plaque. Empty list if none are
  readable. Do not output any number you cannot see.
- suggested_label: a short name a visitor would recognize for this spot: the room name/number from the
  plaque if there is one, otherwise a plain description of the place (type of space + one distinctive
  feature). Never put a room number in the label unless it is visible.
- place_type: room (a door with a plaque is the main subject), intersection (hallways meet or branch),
  hallway (plain corridor), entrance, stairs, elevator, exit, restroom, or landmark (distinctive feature
  such as a front desk, display case, lounge, artwork).
- stairs: ahead_up / ahead_down if a staircase is directly ahead, on_stairs_up / on_stairs_down if the
  camera is on the stairs, otherwise none.
- floor_hint: the floor number only if a sign states it, else "".
- description: a short noun phrase (max 8 words) that completes "Turn left at ___", describing what is
  visible, e.g. the kind of doors, desk or display. No full sentences.
"""

ZOOM_NOTE = """
Image 1 is the full camera view. Images 2 and 3 are zoomed crops of its LEFT and RIGHT parts:
use them to read small door plaques and signs exactly.
"""

SAME_PLACE_PROMPT = """Both images were taken while walking through the same building, at different times.
Were they taken at the SAME physical spot? They may face the same or roughly opposite directions,
and people or lighting may differ. Buildings have many similar-looking corridors, so decide using
unique details: room numbers, sign text, posters, door positions, fixtures, floor patterns.
First list shared_details (specific things visible in BOTH images) and conflicting_details (things
that clearly differ, e.g. different room numbers). Identical sign text in both images is SHARED, not
conflicting. Then give same_place, confidence (0 to 1), and a one-sentence reason."""


class GemmaUnavailable(RuntimeError):
    pass


_model_lock = threading.Lock()
_resolved_model: str | None = None


def _client() -> ollama.Client:
    return ollama.Client(host=config.OLLAMA_HOST, timeout=config.GEMMA_TIMEOUT_SEC)


def available_models() -> list[str]:
    try:
        return [m.model for m in _client().list().models]
    except Exception as exc:  # connection refused, etc.
        raise GemmaUnavailable(f"Ollama not reachable at {config.OLLAMA_HOST}: {exc}") from exc


def resolve_model() -> str:
    global _resolved_model
    with _model_lock:
        if _resolved_model:
            return _resolved_model
        models = available_models()
        for want in (config.GEMMA_MODEL, config.GEMMA_FALLBACK_MODEL):
            if want in models or f"{want}:latest" in models:
                _resolved_model = want
                return want
        gemmas = [m for m in models if m.startswith("gemma")]
        if gemmas:
            _resolved_model = gemmas[0]
            return gemmas[0]
        raise GemmaUnavailable(f"No Gemma model pulled. Run: ollama pull {config.GEMMA_MODEL}")


def health() -> dict:
    try:
        models = available_models()
        return {"ok": True, "models": models, "selected": resolve_model()}
    except GemmaUnavailable as exc:
        return {"ok": False, "error": str(exc), "models": []}


def _encode(img: Image.Image) -> bytes:
    img = img.convert("RGB")
    img.thumbnail((config.GEMMA_IMAGE_MAX, config.GEMMA_IMAGE_MAX))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _views(path: Path) -> list[bytes]:
    img = Image.open(path)
    w, h = img.size
    return [_encode(img), _encode(img.crop((0, 0, int(w * 0.55), h))), _encode(img.crop((int(w * 0.45), 0, w, h)))]


def _sha(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _parse(text: str, model_cls):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    return model_cls.model_validate_json(text)


class GemmaCache:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.data: dict = read_json(path, {}) or {}

    def get(self, key: str):
        return self.data.get(key)

    def put(self, key: str, value: dict) -> None:
        with self.lock:
            self.data[key] = value
            write_json_atomic(self.path, self.data)


def _chat(model: str, prompt: str, images: list[bytes], schema_cls):
    last_exc = None
    for _ in range(2):
        try:
            resp = _client().chat(
                model=model,
                messages=[{"role": "user", "content": prompt, "images": images}],
                format=schema_cls.model_json_schema(),
                options={"temperature": 0},
            )
            return _parse(resp.message.content, schema_cls)
        except (ollama.ResponseError, ValueError) as exc:
            last_exc = exc
        except Exception as exc:
            raise GemmaUnavailable(str(exc)) from exc
    raise GemmaUnavailable(f"Gemma returned unusable output: {last_exc}")


def analyze_frame(path: Path, cache: GemmaCache, model: str | None = None) -> FrameAnalysis:
    model = model or resolve_model()
    key = f"analyze:{PROMPT_VERSION}:{model}:{_sha(path)}"
    if (hit := cache.get(key)) is not None:
        return FrameAnalysis.model_validate(hit)
    views = _views(path)
    result = _chat(model, ANALYZE_PROMPT, views[:1], FrameAnalysis)
    if (result.has_sign and not result.texts) or (result.place_type == "room" and not result.room_numbers):
        # A sign is visible but unreadable at 896px: re-ask with zoomed side crops.
        zoomed = _chat(model, ANALYZE_PROMPT + ZOOM_NOTE, views, FrameAnalysis)
        if zoomed.texts or zoomed.room_numbers:
            result = zoomed
    result.texts = [t.strip() for t in result.texts if t.strip()]
    result.room_numbers = [r.strip() for r in result.room_numbers if r.strip()]
    cache.put(key, result.model_dump())
    return result


def same_place(path_a: Path, path_b: Path, cache: GemmaCache) -> SamePlace:
    model = resolve_model()
    ha, hb = sorted((_sha(path_a), _sha(path_b)))
    key = f"same:{PROMPT_VERSION}:{model}:{ha}:{hb}"
    if (hit := cache.get(key)) is not None:
        return SamePlace.model_validate(hit)
    imgs = [_encode(Image.open(path_a)), _encode(Image.open(path_b))]
    result = _chat(model, SAME_PLACE_PROMPT, imgs, SamePlace)
    result.confidence = max(0.0, min(1.0, float(result.confidence)))
    cache.put(key, result.model_dump())
    return result


def cache_for(building_path: Path) -> GemmaCache:
    return GemmaCache(building_path / "gemma_cache.json")
