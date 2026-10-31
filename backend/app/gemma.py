"""Local Gemma (via Ollama) for landmark/OCR analysis and same-place verification.

Runs fully offline once the model is pulled. Every answer is cached on disk keyed by
(image hash, model, prompt version), so reprocessing a building is free and deterministic.

Cache first: every call looks its answer up before contacting Ollama, so a building that was already
processed can be rebuilt with Ollama stopped and keeps every Gemma answer. The model used for the key is
the explicit `model` argument, else the one of the enclosing `session()` (jobs set it to the model the walk
was processed with), else the last resolved / configured GEMMA_MODEL.
"""
import contextvars
import functools
import hashlib
import io
import json
import logging
import os
import re
import tempfile
import threading
from contextlib import contextmanager
from difflib import SequenceMatcher
from dataclasses import dataclass
from pathlib import Path

import httpx
import ollama
from PIL import Image, ImageOps

from . import config, ocr
from .matching import PLAQUE_PREFIX, complete_room_id, directional, room_id, room_ids
from .schemas import FrameAnalysis, GemmaFrame, SamePlace

log = logging.getLogger(__name__)

PROMPT_VERSION = "v6"  # v6: evidence before verdict, Image 1/2 labels, crop OCR, poster filtering

ANALYZE_PROMPT = """You are helping map the inside of a building from a walkthrough video.
The image is the camera view, looking in the walking direction.

Only WAYFINDING signs count: door plaques, room numbers, room/office name signs, restroom, exit,
stairs and elevator signs, directory or floor signs. IGNORE posters, flyers, bulletin boards,
notices, warning stickers, logos and advertising.

Fill in the JSON fields:
- texts: the wayfinding sign texts exactly as written. Empty list if none. Never guess or invent.
- room_numbers: door/room numbers that are actually visible on a door plaque. Empty list if none are
  readable. Do not output any number you cannot see, and never numbers from posters, phone numbers,
  dates or prices.
- has_sign: true only if a wayfinding sign is clearly legible in THIS image (texts or room_numbers).
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

SAME_PLACE_PROMPT = """Image 1 and Image 2 were taken while walking through the same building, at different times.
Were they taken at the SAME physical spot? They may face the same or roughly opposite directions.

First look at Image 1 ONLY and copy its sign text into image1_sign_text. Then look at Image 2 ONLY and copy
its sign text into image2_sign_text. Each image may show different signs: do not mix them up.

These do NOT mean a different place: a small shift of the viewpoint, a different crop or zoom, a slightly
different angle, lighting, exposure, the wall color or shade looking different, the time the photo was
taken, people present or absent.
The same sign text (same room number or name) in the same layout means the SAME place.
Different room numbers or different sign text mean DIFFERENT places.
Without readable signs, decide using unique details: door positions, fixtures, posters, floor patterns.

Then list shared_details (things visible in BOTH images) and conflicting_details (real differences only,
e.g. different room numbers or sign text; never lighting, color shade, time or viewpoint). Then give
same_place, confidence (0 to 1), and a one-sentence reason."""


SAME_STAIRWELL_PROMPT = """Image 1 and Image 2 were taken on DIFFERENT floors of the same building.
Do they show the SAME staircase / stairwell (or the same elevator bank), just seen from different floors
or landings?

First look at Image 1 ONLY and copy any stair/elevator sign text into image1_sign_text, then the same for
Image 2 into image2_sign_text (floor numbers are expected to differ). Compare railing style, step material,
wall finish, windows, and stair or elevator names/numbers. Lighting, exposure, viewpoint and the wall color
looking a shade different do NOT mean a different stairwell. Different stairwells in one building often look
alike, so rely on specific details.
Then list shared_details (things in BOTH images) and conflicting_details (real differences only), then
same_place, confidence (0 to 1) and a reason."""


class GemmaUnavailable(RuntimeError):
    pass


class NotCached(GemmaUnavailable):
    """Gemma is already known to be offline in this session and the answer is not cached."""


_model_lock = threading.Lock()
_resolved_model: str | None = None


@dataclass
class Session:
    model: str | None = None  # model to key the cache with / ask, when a call passes none
    lookup: tuple[str, ...] = ()  # more models whose cached answers may be reused when Ollama is offline
    offline: bool = False  # Ollama known down (or failed): answer from the cache only, never contact it


_session: contextvars.ContextVar[Session | None] = contextvars.ContextVar("gemma_session", default=None)


@contextmanager
def session(model: str | None = None, lookup=(), offline: bool = False):
    """Scope (a job's mapping step) in which same_place/same_stairwell/analyze_frame default to `model`, and
    in which, once Ollama failed, further calls answer from the cache only (instant misses). Nested sessions
    inherit the outer model, lookup list and offline state unless given."""
    outer = _session.get()
    s = Session(model or (outer.model if outer else None),
                tuple(m for m in lookup if m) or (outer.lookup if outer else ()),
                offline or bool(outer and outer.offline))
    token = _session.set(s)
    try:
        yield s
    finally:
        _session.reset(token)


def session_active() -> bool:
    """True inside session(): calls after a failure are cache-only and cheap, so callers may keep calling."""
    return _session.get() is not None


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


def _has_vision(model: str) -> bool:
    """Ollama reports `capabilities` (e.g. ["completion", "vision"]); older servers omit it: assume yes."""
    try:
        info = _client(timeout=config.GEMMA_LIST_TIMEOUT_SEC).show(model)
    except Exception:
        return False
    caps = info.get("capabilities") if isinstance(info, dict) else getattr(info, "capabilities", None)
    return caps is None or "vision" in caps


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
        gemmas = [m for m in models if m.startswith("gemma") and _has_vision(m)]
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


def _open(path: Path) -> Image.Image:
    """The image upright: phone photos store rotation in EXIF (orientation 6 = portrait shot sideways)."""
    return ImageOps.exif_transpose(Image.open(path))


def _encode(img: Image.Image) -> bytes:
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((config.GEMMA_IMAGE_MAX, config.GEMMA_IMAGE_MAX))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _views(path: Path) -> list[bytes]:
    img = _open(path)
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


def _schema(schema_cls) -> dict:
    """JSON schema sent to Ollama with EVERY field required: optional fields may otherwise be skipped by the
    model, and the evidence fields (texts, shared_details...) only help when they are generated first."""
    schema = schema_cls.model_json_schema()
    schema["required"] = list(schema.get("properties", {}))
    return schema


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
                format=_schema(schema_cls),
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


# A room number seen at one of these places is the place's own; anywhere else (a hallway, a landmark) it only
# counts when it is on a door-plaque-like OCR line, so poster numbers ('Table 77 Hall 9') never become rooms.
_DOOR_PLACES = {"room", "entrance", "restroom", "stairs", "elevator", "exit"}


_NUM34 = re.compile(r"(?<!\d)\d{3,4}(?!\d)")
# 'Room 2', 'RM. 118', 'Office #214A', 'Lab No. 3': the number a plaque word in a label names.
_LABEL_PLAQUE_NUMBER = re.compile(rf"\b{PLAQUE_PREFIX}\b\.?\s*(?:NO\b\.?\s*)?#?\s*([A-Z]?\d+[A-Z]?)\b")
_PLAQUE_WORDS = {"ROOM", "RM", "OFFICE", "OFC", "LAB", "SUITE", "STE"}
_SAYS_POSTER = {"POSTER", "POSTERS", "NOTICE", "NOTICES", "FLYER", "FLYERS", "FLIER", "FLIERS", "BULLETIN"}


def _notice_numbers(lines: list[str]) -> set[str]:
    """3-4 digit numbers printed on notice text of this frame ('4PM ROOM 305', 'CALL 555-0147'): another line of
    the same frame repeating one ('API ROOM 305', a misread of the same poster line) is no room number either."""
    return {n for line in lines if not ocr.sign_line(line) for n in _NUM34.findall(line)}


def _on_notice(text: str, notice_nums: set[str]) -> bool:
    return bool(notice_nums & set(_NUM34.findall(text)))


def _core(word: str) -> str:
    return re.sub(r"^[^A-Z0-9]+|[^A-Z0-9]+$", "", word.upper())


def _alpha_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[A-Z]+", text.upper()) if len(w) >= 4]


def _word_in(w: str, toks: set[str]) -> bool:
    """w (4+ letters) is one of the tokens, part of one ('BOOK' of 'BOOKFAIR': OCR dropped the space) or close
    to one ('GUITAR' ~ 'UITAR', 'SEMINAR' ~ 'EMINAR': an OCR cut)."""
    return any(w in t or SequenceMatcher(None, w, t).ratio() >= 0.8 for t in toks)


def _numbers_ok(label: str, rooms: set[str]) -> bool:
    """Every 3-4 digit number in the label, and every number right after Room/Office/Lab, is a kept room number."""
    up = label.upper()
    if any(room_id(m.group(1)) not in rooms for m in _LABEL_PLAQUE_NUMBER.finditer(up)):
        return False
    return all(room_id(c) in rooms for w in up.split() if _NUM34.search(c := _core(w)))


def _poster_built(label: str, poster_toks: set[str], own_toks: set[str]) -> bool:
    """The label names the poster ('Math Tutoring', 'Guitar Lessons hallway'): one of its words is read only on
    the frame's poster lines. A label that says it is a poster ('Bake sale poster') is fine."""
    words = _alpha_words(label)
    if not poster_toks or _SAYS_POSTER & set(words):
        return False
    return any(_word_in(w, poster_toks) and not _word_in(w, own_toks) for w in words if w not in _PLAQUE_WORDS)


def _drop_plaque_words(label: str, rooms: set[str], own_toks: set[str]) -> str:
    """Drop a Room/Office/Lab/Suite word the label invents: not followed by a kept room number and not read on
    a non-poster OCR line ('Math Tutoring Office' -> 'Math Tutoring', 'ROCI Room' -> 'ROCI')."""
    words = label.split()
    out = []
    for i, w in enumerate(words):
        core = _core(w)
        if core in _PLAQUE_WORDS and core not in own_toks:
            nxt = _core(words[i + 1]) if i + 1 < len(words) else ""
            if not (nxt and room_id(nxt) in rooms):
                continue
        out.append(w)
    return " ".join(out).strip(" -,–:")


def _clean_label(text: str, number_lines: list[str], rooms: set[str], poster_toks: set[str], own_toks: set[str],
                 directional_ok: bool = False) -> str:
    """The label without invented Room/Office words, or '' if it is built on an ungrounded, cut or poster number
    or on poster words. `directional_ok`: a description may repeat a directional sign ('Rooms 220 to 230
    hallway'); its numbers were checked against that sign already."""
    text = text.strip()
    if not (directional_ok and directional(text)):
        text, changed = _strip_ungrounded_numbers(text, number_lines)
        if changed or not _numbers_ok(text, rooms):
            return ""
    text = _drop_plaque_words(text, rooms, own_toks)
    if not re.search(r"[A-Za-z0-9]", text) or _poster_built(text, poster_toks, own_toks):
        return ""
    return text


def _check_ocr(result: FrameAnalysis, lines: list[str]) -> FrameAnalysis:
    """Texts, room numbers, description and label checked against the frame's OCR lines. Idempotent, so it is
    re-applied to cached answers (see _tidy_rooms).

    Notice text ('4PM ROOM 305', phone numbers, dates) and partial plaques ('ROOM 21' of ROOM 216, 'ROOM 10?')
    ground nothing; a room number must be complete (3-4 digits) and not printed on a notice line of the same
    frame. The label may only name kept room numbers, may not invent a Room/Office word, and may not be built
    from a poster's words; a rejected label becomes 'Room N', else the description, else the place type."""
    clean = [line for line in lines if ocr.sign_line(line) and not ocr.partial_plaque(line)]
    notice_nums = _notice_numbers(lines)
    # A directional sign ('ROOMS 120-130 ->', 'ROOM 121 →') names rooms elsewhere: never this place's number.
    own_lines = [line for line in clean if not directional(line)]
    if result.place_type in _DOOR_PLACES:
        number_lines = own_lines
    else:
        number_lines = [line for line in own_lines if ocr.plaque_like(line) or ocr.floor_from_lines([line])]
    result.texts = [t for t in result.texts
                    if ocr.grounded(t, clean) and ocr.sign_line(t) and not ocr.partial_plaque(t)
                    and not _on_notice(t, notice_nums)]
    result.room_numbers = list(dict.fromkeys(
        rn for r in result.room_numbers
        if not directional(r) and complete_room_id(rn := room_id(r))  # 'RM 118' -> '118'
        and ocr.grounded(rn, number_lines) and not _on_notice(rn, notice_nums)))
    result.has_sign = bool(result.texts or result.room_numbers)
    # numbers OCR read on a directional sign are real: fine in a description ('sign to rooms 120-130'), not a label
    result.description, _ = _strip_ungrounded_numbers(
        result.description, number_lines + [line for line in clean if directional(line)])

    rooms = {r.upper() for r in result.room_numbers}
    posters = ocr.poster_lines(lines)
    poster_toks = {w for line in posters for w in _alpha_words(line)}
    own_toks = {w for line in lines if line not in posters for w in re.findall(r"[A-Z]+", line.upper())}
    label = _clean_label(result.suggested_label, number_lines, rooms, poster_toks, own_toks)
    if not label:  # built on an invented, cut or poster number or on poster words: rebuild it from what is real
        desc = _clean_label(result.description, number_lines, rooms, poster_toks, own_toks,
                            directional_ok=True)
        if result.room_numbers:
            rn = result.room_numbers[0]
            label = f"Room {rn}" if rn[:1].isdigit() else rn  # '132' (read as 'RM 132') -> 'Room 132'; 'B204' as read
        elif desc:
            label = desc[:1].upper() + desc[1:]
        else:
            label = result.place_type.capitalize()
    result.suggested_label = label
    return result


def _ground(result: FrameAnalysis, lines: list[str]) -> FrameAnalysis:
    _check_ocr(result, lines)
    sign_lines = [line for line in lines if ocr.sign_line(line)]  # not phone numbers, notices or running text
    if result.floor_hint and not ocr.floor_grounded(result.floor_hint, lines):
        result.floor_hint = ""  # only a line like 'LEVEL 2' / '2ND FLOOR' confirms it, never a room number
    if not result.floor_hint:
        result.floor_hint = ocr.floor_from_lines(sign_lines)  # Gemma missed a clear 'LEVEL 2' sign
    result.grounded = True
    return result


def _tidy_rooms(result: FrameAnalysis) -> FrameAnalysis:
    """Room numbers in one canonical form ('RM 118' / 'ROOM-118' -> '118', '118A' kept) and never from a
    directional sign: 'ROOMS 120-130 ->' names rooms down the hall, so it gives no room number and does not
    make the spot a "room" (the text stays in `texts`). Idempotent; also applied to cached answers, so maps
    built from an older cache get the same room numbers and label checks without asking Gemma again."""
    nums = [room_id(r) for r in result.room_numbers if r.strip() and not directional(r)]
    # Re-check texts, numbers and label against the frame's OCR (_check_ocr). Lines recorded at a known
    # revision count even when empty: OCR read nothing there (e.g. only a plaque cut at the image edge), so
    # nothing Gemma reported is grounded.
    if result.grounded and (result.ocr or result.ocr_rev):
        result.room_numbers = nums
        _check_ocr(result, list(result.ocr))
        nums = result.room_numbers
    elif not result.grounded:  # no OCR: Gemma's own texts tell which numbers are only on a directional sign
        elsewhere = {n for t in result.texts if directional(t) for n in re.findall(r"\d+[A-Z]?", t.upper())}
        own = room_ids(result.texts)
        nums = [r for r in nums if r not in elsewhere or r in own]
    result.room_numbers = list(dict.fromkeys(nums))
    if (result.place_type == "room" and not result.room_numbers and result.texts
            and all(directional(t) for t in result.texts)):
        result.place_type = "hallway"  # only a directional sign: a corridor, not a door
    if result.grounded:
        result.has_sign = bool(result.texts or result.room_numbers)
    return result


def _cached(cache: GemmaCache, key_for, models, model_cls):
    for m in dict.fromkeys(x for x in models if x):
        if (hit := cache.get(key_for(m))) is not None:
            result = model_cls.model_validate(hit)
            if isinstance(result, FrameAnalysis):
                result._model_used = m
            return result
    return None


def _offline_models(explicit: str | None) -> list[str]:
    """Models whose cached answers are acceptable when Ollama cannot be asked."""
    s = _session.get()
    out = list(s.lookup) if s else []
    if not explicit:  # nobody chose a model: any of the usual ones
        out += [_resolved_model, config.GEMMA_MODEL, config.GEMMA_FALLBACK_MODEL]
    return out


def _cache_first(cache: GemmaCache, key_for, model_cls, model: str | None, compute, mark_offline: bool):
    """Cached answer for the model in charge, else ask Ollama (and cache). Raises GemmaUnavailable only when
    the answer is not cached and Ollama cannot give it."""
    s = _session.get()
    explicit = model or (s.model if s else None)
    if (hit := _cached(cache, key_for, [explicit or _resolved_model or config.GEMMA_MODEL], model_cls)) is not None:
        return hit
    if s is not None and s.offline:
        if (hit := _cached(cache, key_for, _offline_models(explicit), model_cls)) is not None:
            return hit
        raise NotCached("Gemma offline in this job and the answer is not cached")
    try:
        use = explicit or resolve_model()
    except GemmaUnavailable:
        if s is not None:
            s.offline = True
        if (hit := _cached(cache, key_for, _offline_models(explicit), model_cls)) is not None:
            return hit
        raise
    if (hit := _cached(cache, key_for, [use], model_cls)) is not None:
        return hit
    try:
        result = compute(use)
    except GemmaUnavailable:
        if s is not None and mark_offline:
            s.offline = True
        raise
    cache.put(key_for(use), result.model_dump())
    if isinstance(result, FrameAnalysis):
        result._model_used = use
    return result


def _frame_key(path: Path):
    mode = "ocr" if ocr.available() else "plain"
    sha = _sha(path)
    return lambda m: f"analyze:{PROMPT_VERSION}:{m}:{mode}:{sha}"


def _compute_frame(path: Path, model: str) -> FrameAnalysis:
    views = _views(path)
    if ocr.available():
        lines = ocr.read_text_with_crops(path)
        listing = "\n".join(f"- {l}" for l in lines) if lines else "(none)"
        raw = _chat(model, ANALYZE_PROMPT + OCR_NOTE.format(lines=listing), views[:1], GemmaFrame,
                    config.GEMMA_NUM_PREDICT_ANALYZE)
        result = _ground(FrameAnalysis(**raw.model_dump(), ocr=lines, ocr_rev=ocr.REVISION), lines)
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
    _tidy_rooms(result)
    if not result.suggested_label.strip():  # never an empty node name
        d = result.description.strip()
        result.suggested_label = d[:1].upper() + d[1:] if d else result.place_type.capitalize()
    return result


def _refresh_ocr(path: Path, result: FrameAnalysis, cache: GemmaCache, key_for) -> FrameAnalysis:
    """A cached answer grounded on OCR lines from an older ocr.REVISION: re-read the lines from the photo (OCR
    only, never Ollama) and store them, so _tidy_rooms re-checks Gemma's answer against what OCR keeps now.
    Without this, a line the current OCR drops (a number cut at the image edge: '110' of plaque 110A) would
    still ground a room number from the cache. Unchanged when OCR is off or the photo cannot be read."""
    if not result.grounded or result.ocr_rev >= ocr.REVISION or not ocr.available():
        return result
    try:
        st = Path(path).stat()
        lines = list(_ocr_lines(str(path), st.st_mtime_ns, st.st_size))
    except (OSError, ValueError) as exc:
        log.debug("OCR refresh skipped for %s: %s", path, exc)
        return result
    if lines != result.ocr:
        log.info("OCR lines of cached %s re-read: %s -> %s", Path(path).name, result.ocr, lines)
    result.ocr, result.ocr_rev = lines, ocr.REVISION
    if result._model_used:
        cache.put(key_for(result._model_used), result.model_dump())  # Gemma's raw answer, newer OCR lines
    return result


def analyze_frame(path: Path, cache: GemmaCache, model: str | None = None) -> FrameAnalysis:
    """Cache first; `result._model_used` tells which model the answer is from (not serialized)."""
    key_for = _frame_key(path)
    hit = _cache_first(cache, key_for, FrameAnalysis, model, lambda m: _compute_frame(path, m), mark_offline=False)
    return _tidy_rooms(_refresh_ocr(path, hit, cache, key_for))


def cached_analysis(path: Path, cache: GemmaCache, model: str | None = None) -> FrameAnalysis | None:
    """The cached answer for a frame without ever contacting Ollama (None if not cached or unreadable)."""
    try:
        key_for = _frame_key(path)
    except OSError:
        return None
    s = _session.get()
    explicit = model or (s.model if s else None)
    hit = _cached(cache, key_for, [explicit] + _offline_models(explicit), FrameAnalysis)
    return _tidy_rooms(_refresh_ocr(path, hit, cache, key_for)) if hit is not None else None


OCR_CONFLICT = "OCR reads different room numbers"


def ocr_conflict(verdict: SamePlace | None) -> bool:
    """True for a verdict decided by OCR (different plaque numbers), not by the model: always binding."""
    return bool(verdict is not None and not verdict.same_place and verdict.reason.startswith(OCR_CONFLICT))


def _room_conflict(path_a: Path, path_b: Path) -> SamePlace | None:
    """Different door-plaque numbers read by OCR in the two images: different places, no need to ask Gemma
    (small models call 'ROOM 121' and 'ROOM 305' plaques on identical walls the same place)."""
    if not ocr.available():
        return None
    return _plaque_verdict(ocr.read_text_with_crops(path_a), ocr.read_text_with_crops(path_b))


@functools.lru_cache(maxsize=1024)
def _ocr_lines(path: str, mtime_ns: int, size: int) -> tuple[str, ...]:
    return tuple(ocr.read_text_with_crops(Path(path)))


def plaque_conflict(path_a: Path, path_b: Path) -> SamePlace | None:
    """OCR-only 'different room numbers' check (never contacts Ollama), for pairs decided without asking Gemma.
    None when OCR is off, an image cannot be read, or the plaques do not conflict."""
    if not ocr.available():
        return None
    try:
        lines = []
        for p in (Path(path_a), Path(path_b)):
            st = p.stat()
            lines.append(list(_ocr_lines(str(p), st.st_mtime_ns, st.st_size)))
    except (OSError, ValueError) as exc:  # missing / unreadable image: no evidence either way
        log.debug("plaque check skipped: %s", exc)
        return None
    return _plaque_verdict(*lines)


def _plaque_verdict(la: list[str], lb: list[str]) -> SamePlace | None:
    ra, rb = ocr.plaque_numbers(la), ocr.plaque_numbers(lb)
    if not ra or not rb or ra & rb:
        return None
    return SamePlace(image1_sign_text=la, image2_sign_text=lb, conflicting_details=[
        f"different room numbers: {'/'.join(sorted(ra))} vs {'/'.join(sorted(rb))}"], same_place=False,
        confidence=0.95, reason=f"{OCR_CONFLICT}: {'/'.join(sorted(ra))} in Image 1, {'/'.join(sorted(rb))} in Image 2")


def _compare(kind: str, prompt: str, path_a: Path, path_b: Path, cache: GemmaCache, model: str | None) -> SamePlace:
    sa, sb = _sha(path_a), _sha(path_b)
    (ha, first), (hb, second) = sorted(((sa, path_a), (sb, path_b)))  # Image 1 = smaller hash: either order

    def compute(m: str) -> SamePlace:  # gives the same (cached) question
        if kind == "same" and (conflict := _room_conflict(first, second)) is not None:
            return conflict
        imgs = [_encode(_open(first)), _encode(_open(second))]
        result = _chat(m, prompt, imgs, SamePlace, config.GEMMA_NUM_PREDICT_COMPARE)
        result.confidence = max(0.0, min(1.0, float(result.confidence)))
        return result

    return _cache_first(cache, lambda m: f"{kind}:{PROMPT_VERSION}:{m}:{ha}:{hb}", SamePlace, model, compute,
                        mark_offline=True)


def same_place(path_a: Path, path_b: Path, cache: GemmaCache, model: str | None = None) -> SamePlace:
    return _compare("same", SAME_PLACE_PROMPT, path_a, path_b, cache, model)


def same_stairwell(path_a: Path, path_b: Path, cache: GemmaCache, model: str | None = None) -> SamePlace:
    return _compare("stairs", SAME_STAIRWELL_PROMPT, path_a, path_b, cache, model)


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
