"""Local Gemma (via Ollama) for landmark/OCR analysis and same-place verification.

Runs fully offline once the model is pulled. Every answer is cached on disk keyed by
(image hash, model, prompt version), so reprocessing a building is free and deterministic.
"""
import hashlib
import io
import json
import logging
import os
import re
import tempfile
import threading
from pathlib import Path

import httpx
import ollama
from PIL import Image

from . import config, ocr
from .schemas import FrameAnalysis, GemmaFrame, SamePlace

log = logging.getLogger(__name__)

PROMPT_VERSION = "v5"  # v5: exact-token OCR grounding

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

OCR_NOTE = """
An on-device text recognizer found exactly these text strings in the image (one per line):
{lines}
texts and room_numbers MUST be copied from these strings (only the wayfinding ones). If none of them
is a wayfinding sign, return empty lists and has_sign=false. Do not report any other text or number.
"""

SAME_PLACE_PROMPT = """Both images were taken while walking through the same building, at different times.
Were they taken at the SAME physical spot? They may face the same or roughly opposite directions,
and people or lighting may differ. Buildings have many similar-looking corridors, so decide using
unique details: room numbers, sign text, posters, door positions, fixtures, floor patterns.
First list shared_details (specific things visible in BOTH images) and conflicting_details (things
that clearly differ, e.g. different room numbers). Identical sign text in both images is SHARED, not
conflicting. Then give same_place, confidence (0 to 1), and a one-sentence reason."""


SAME_STAIRWELL_PROMPT = """These two photos were taken on DIFFERENT floors of the same building.
Do they show the SAME staircase / stairwell (or the same elevator bank), just seen from different floors
or landings? Compare railing style and color, step material, wall finish, windows, and any stair or
elevator number signs. Different stairwells in one building often look alike, so rely on specific details.
First list shared_details and conflicting_details, then same_place, confidence (0 to 1) and a reason."""


class GemmaUnavailable(RuntimeError):
    pass


_model_lock = threading.Lock()
_resolved_model: str | None = None


def _client(timeout: float | None = None) -> ollama.Client:
    """Module-level factory (tests replace it). Default timeout is the long chat one."""
    return ollama.Client(host=config.OLLAMA_HOST, timeout=timeout or config.GEMMA_TIMEOUT_SEC)


def available_models() -> list[str]:
    try:
        return [m.model for m in _client(timeout=config.GEMMA_LIST_TIMEOUT_SEC).list().models]
    except Exception as exc:  # connection refused, etc.
        raise GemmaUnavailable(f"Ollama not reachable at {config.OLLAMA_HOST}: {exc}") from exc


def ensure_model(model: str) -> None:
    """Fail fast if `model` is not pulled (or Ollama is down), instead of failing on every frame."""
    try:
        _client(timeout=config.GEMMA_LIST_TIMEOUT_SEC).show(model)
    except ollama.ResponseError as exc:
        if exc.status_code == 404:
            raise GemmaUnavailable(f"Model {model!r} is not pulled in Ollama. Run: ollama pull {model}") from exc
        raise GemmaUnavailable(f"Ollama cannot use model {model!r}: {exc}") from exc
    except Exception as exc:
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


CACHE_FILE = "gemma_cache.jsonl"


class GemmaCache:
    """Append-only JSONL cache: one {"k": key, "v": value} object per line.

    Each put appends one line and flushes, so writes are O(1) instead of rewriting the whole
    file. On load, unreadable lines (e.g. a truncated last line after a crash) are skipped and
    the last line wins for duplicate keys. A legacy gemma_cache.json next to the file is
    migrated once, then renamed to gemma_cache.json.bak. Use cache_for() to get the shared
    per-building instance rather than constructing this directly.
    """

    def __init__(self, path: Path):
        path = Path(path)
        if path.suffix == ".json":  # tolerate callers passing the legacy filename
            path = path.with_suffix(".jsonl")
        self.path = path
        self.lock = threading.Lock()
        self.data: dict = {}
        self._needs_newline = False  # file ends mid-line: start the next append on a fresh line
        with self.lock:
            self._migrate_legacy()
            self._load()

    def _migrate_legacy(self) -> None:
        legacy = self.path.with_suffix(".json")
        if self.path.exists() or not legacy.exists():
            return
        try:
            with open(legacy, encoding="utf-8") as f:
                old = json.load(f)
        except (OSError, ValueError) as exc:
            log.warning("gemma cache: cannot read legacy %s (%s); starting empty", legacy, exc)
            return
        if not isinstance(old, dict):
            log.warning("gemma cache: legacy %s is not a JSON object; starting empty", legacy)
            return
        self._write_all(old)
        try:
            os.replace(legacy, legacy.with_name(legacy.name + ".bak"))
        except OSError as exc:  # harmless: the JSONL exists now, so migration won't rerun
            log.warning("gemma cache: could not rename legacy %s (%s)", legacy, exc)

    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, "rb") as f:
            raw = f.read()
        lines = bad = 0
        for line in raw.split(b"\n"):
            if not line.strip():
                continue
            lines += 1
            try:
                rec = json.loads(line)
                self.data[rec["k"]] = rec["v"]
            except (ValueError, KeyError, TypeError):
                bad += 1
        if bad:
            log.warning("gemma cache: skipped %d unreadable line(s) in %s", bad, self.path)
        self._needs_newline = bool(raw) and not raw.endswith(b"\n")
        wasted = lines - len(self.data)  # unreadable lines + superseded duplicates
        if wasted > max(100, len(self.data)):
            self._write_all(self.data)  # compact

    def _write_all(self, data: dict) -> None:
        """Atomically replace the JSONL file with one line per entry."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for k, v in data.items():
                    f.write(json.dumps({"k": k, "v": v}) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        self._needs_newline = False

    def get(self, key: str):
        return self.data.get(key)

    def put(self, key: str, value) -> None:
        line = json.dumps({"k": key, "v": value}) + "\n"  # serialize first: fail before mutating
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                if self._needs_newline:
                    f.write("\n")
                    self._needs_newline = False
                f.write(line)
                f.flush()
            self.data[key] = value


# Worth one retry: dropped/reset connections, timeouts (httpx.TransportError), Ollama not answering.
_TRANSIENT = (ConnectionError, TimeoutError, httpx.TransportError)


def _chat(model: str, prompt: str, images: list[bytes], schema_cls, num_predict: int):
    """One structured answer. At most 2 attempts: a transient error (connection reset, timeout, HTTP 5xx) or
    unparseable JSON is retried once; anything else (404 model not found, other 4xx, truncated output)
    raises GemmaUnavailable immediately."""
    last_exc: Exception | None = None
    for _ in range(2):
        try:
            resp = _client().chat(
                model=model,
                messages=[{"role": "user", "content": prompt, "images": images}],
                format=schema_cls.model_json_schema(),
                options={"temperature": 0, "num_ctx": config.GEMMA_NUM_CTX, "num_predict": num_predict},
                keep_alive=config.GEMMA_KEEP_ALIVE,
            )
        except ollama.ResponseError as exc:
            if exc.status_code == 404:
                raise GemmaUnavailable(f"Model {model!r} not found in Ollama. Run: ollama pull {model}") from exc
            if exc.status_code < 500:  # bad request etc.: retrying gives the same answer
                raise GemmaUnavailable(f"Ollama rejected the request ({exc.status_code}): {exc.error}") from exc
            last_exc = exc
            continue
        except _TRANSIENT as exc:
            last_exc = exc
            continue
        except Exception as exc:
            raise GemmaUnavailable(f"{type(exc).__name__}: {exc}") from exc
        try:
            return _parse(resp.message.content or "", schema_cls)
        except ValueError as exc:  # invalid JSON or schema mismatch (pydantic ValidationError is a ValueError)
            if getattr(resp, "done_reason", None) == "length":
                raise GemmaUnavailable(
                    f"Gemma output truncated at num_predict={num_predict} tokens (raise GEMMA_NUM_PREDICT_*)") from exc
            last_exc = exc
    raise GemmaUnavailable(f"Gemma failed after a retry: {type(last_exc).__name__}: {last_exc}")


def _strip_ungrounded_numbers(text: str, lines: list[str]) -> tuple[str, bool]:
    """Remove number-bearing words that OCR never read as whole tokens. Returns (cleaned, changed)."""
    words, changed = [], False
    for w in text.split():
        core = re.sub(r"[^A-Za-z0-9]", "", w).upper()
        if any(ch.isdigit() for ch in core) and not ocr.grounded(core, lines):
            changed = True
            continue
        words.append(w)
    return " ".join(words).strip(" -,–:"), changed


def _ground(result: FrameAnalysis, lines: list[str]) -> FrameAnalysis:
    result.texts = [t for t in result.texts if ocr.grounded(t, lines)]
    result.room_numbers = [r for r in result.room_numbers if ocr.grounded(r, lines)]
    result.has_sign = bool(result.texts or result.room_numbers)
    if result.floor_hint and not ocr.floor_grounded(result.floor_hint, lines):
        result.floor_hint = ""  # only a line like 'LEVEL 2' / '2ND FLOOR' confirms it, never a room number
    result.description, _ = _strip_ungrounded_numbers(result.description, lines)
    label, changed = _strip_ungrounded_numbers(result.suggested_label, lines)
    if changed:  # the label was built on an invented number: rebuild it from what is real
        if result.room_numbers:
            label = f"Room {result.room_numbers[0]}"
        elif result.description:
            label = result.description[:1].upper() + result.description[1:]
        else:
            label = result.place_type.capitalize()
    result.suggested_label = label
    result.grounded = True
    return result


def analyze_frame(path: Path, cache: GemmaCache, model: str | None = None) -> FrameAnalysis:
    model = model or resolve_model()
    use_ocr = ocr.available()
    key = f"analyze:{PROMPT_VERSION}:{model}:{'ocr' if use_ocr else 'plain'}:{_sha(path)}"
    if (hit := cache.get(key)) is not None:
        return FrameAnalysis.model_validate(hit)
    views = _views(path)
    if use_ocr:
        lines = ocr.read_text(path)
        listing = "\n".join(f"- {l}" for l in lines) if lines else "(none)"
        raw = _chat(model, ANALYZE_PROMPT + OCR_NOTE.format(lines=listing), views[:1], GemmaFrame,
                    config.GEMMA_NUM_PREDICT_ANALYZE)
        result = _ground(FrameAnalysis(**raw.model_dump(), ocr=lines), lines)
    else:
        raw = _chat(model, ANALYZE_PROMPT, views[:1], GemmaFrame, config.GEMMA_NUM_PREDICT_ANALYZE)
        if (raw.has_sign and not raw.texts) or (raw.place_type == "room" and not raw.room_numbers):
            # A sign is visible but unreadable at 896px: re-ask with zoomed side crops.
            zoomed = _chat(model, ANALYZE_PROMPT + ZOOM_NOTE, views, GemmaFrame, config.GEMMA_NUM_PREDICT_ANALYZE)
            if zoomed.texts or zoomed.room_numbers:
                raw = zoomed
        result = FrameAnalysis(**raw.model_dump())
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
    result = _chat(model, SAME_PLACE_PROMPT, imgs, SamePlace, config.GEMMA_NUM_PREDICT_COMPARE)
    result.confidence = max(0.0, min(1.0, float(result.confidence)))
    cache.put(key, result.model_dump())
    return result


def same_stairwell(path_a: Path, path_b: Path, cache: GemmaCache) -> SamePlace:
    model = resolve_model()
    ha, hb = sorted((_sha(path_a), _sha(path_b)))
    key = f"stairs:{PROMPT_VERSION}:{model}:{ha}:{hb}"
    if (hit := cache.get(key)) is not None:
        return SamePlace.model_validate(hit)
    imgs = [_encode(Image.open(path_a)), _encode(Image.open(path_b))]
    result = _chat(model, SAME_STAIRWELL_PROMPT, imgs, SamePlace, config.GEMMA_NUM_PREDICT_COMPARE)
    result.confidence = max(0.0, min(1.0, float(result.confidence)))
    cache.put(key, result.model_dump())
    return result


_caches: dict[Path, GemmaCache] = {}
_caches_lock = threading.Lock()


def cache_for(building_path: Path) -> GemmaCache:
    """Shared GemmaCache per building, so the pipeline and manual requests see the same data."""
    path = (Path(building_path) / CACHE_FILE).resolve()
    with _caches_lock:
        cache = _caches.get(path)
        if cache is None:
            cache = _caches[path] = GemmaCache(path)
        return cache
