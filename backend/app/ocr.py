"""On-device OCR (Apple Vision framework) used to ground Gemma's sign reading.

Small vision-language models confidently invent room numbers. Vision's text recognizer is offline,
~50 ms per frame, and only reports strings that are really there, so Gemma is told which strings
exist and anything it reports outside that list is dropped. Gemma still does all the understanding:
labels, place types, stairs, landmark phrases and same-place checks.

Not on macOS (or pyobjc missing)? `available()` is False and Gemma runs ungrounded.
"""
import io
import re
from pathlib import Path

from PIL import Image, ImageOps

from . import config
from .matching import ROOM_ID, complete_room_id, cut_number, directional, notice, phone_like, room_id

try:
    import objc
    import Vision
    from Foundation import NSURL, NSData

    _AVAILABLE = True
except ImportError:  # Linux/Windows, or pyobjc not installed
    _AVAILABLE = False


# Version of what read_text_with_crops keeps. Bump it whenever that changes (e.g. the image-edge filter, rev 2):
# a cached frame analysis stores the OCR lines it was grounded on, and one from an older revision has its lines
# re-read from the photo (no Gemma call) so stale lines ('110' of a 110A plaque cut at the edge) cannot ground.
REVISION = 2


def available() -> bool:
    return _AVAILABLE and config.OCR_ENABLED


def _handler(source):
    """VNImageRequestHandler for a file path or an in-memory PIL image."""
    if isinstance(source, (str, Path)):
        return Vision.VNImageRequestHandler.alloc().initWithURL_options_(NSURL.fileURLWithPath_(str(source)), None)
    buf = io.BytesIO()
    source.convert("RGB").save(buf, format="PNG", compress_level=1)
    data = buf.getvalue()
    return Vision.VNImageRequestHandler.alloc().initWithData_options_(NSData.dataWithBytes_length_(data, len(data)), None)


def _observations(source) -> list[tuple[str, float, float]]:
    """(text, x_min, x_max) per recognized line with confidence >= OCR_MIN_CONFIDENCE; x normalized to 0..1."""
    with objc.autorelease_pool():
        handler = _handler(source)
        req = Vision.VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        req.setUsesLanguageCorrection_(False)  # room numbers are not dictionary words
        ok, _ = handler.performRequests_error_([req], None)
        if not ok:
            return []
        out = []
        for obs in req.results() or []:
            cand = obs.topCandidates_(1)
            if cand and cand[0].confidence() >= config.OCR_MIN_CONFIDENCE:
                text = str(cand[0].string()).strip()
                if text:
                    box = obs.boundingBox()
                    out.append((text, float(box.origin.x), float(box.origin.x + box.size.width)))
        return out


def read_text(source: Path | Image.Image) -> list[str]:
    """Return recognized text lines (confidence >= OCR_MIN_CONFIDENCE) of an image file or a PIL image."""
    if not available():
        return []
    return [t for t, _, _ in _observations(source)]


CROP_EDGE = 0.02  # a line this close to a crop's cut edge may be cut in half: the full view or other crop has it


def _edge_cut(text: str, x0: float, x1: float, left_edge: bool, right_edge: bool) -> bool:
    """A line touching the image's left/right edge with a number at that end: the plaque runs off the frame,
    so the number may be missing digits ('ROOM 21' of ROOM 216, '14A' of 214A). Words cut there are harmless
    ('OM 220' still says 220)."""
    toks = text.split()
    if not toks:
        return False
    return ((right_edge and x1 >= 1 - CROP_EDGE and any(ch.isdigit() for ch in toks[-1]))
            or (left_edge and x0 <= CROP_EDGE and any(ch.isdigit() for ch in toks[0])))


def read_text_with_crops(path: Path) -> list[str]:
    """OCR of the full (EXIF-upright) image plus zoomed LEFT/RIGHT crops (the same 55% halves Gemma sees), so
    small corner plaques that the full view misses are read too. Lines are deduplicated; a crop line touching
    the cut edge (possibly half a word) or contained in an already-read line is dropped, and so is any line
    whose number touches the image edge (possibly cut, see _edge_cut)."""
    if not available():
        return []
    img = ImageOps.exif_transpose(Image.open(path))
    img.load()
    w, h = img.size
    out = [t for t, x0, x1 in _observations(img) if not _edge_cut(t, x0, x1, True, True)]
    seen = [_norm(t) for t in out]
    for box, cut_left, cut_right in (((0, 0, int(w * 0.55), h), False, True),
                                     ((int(w * 0.45), 0, w, h), True, False)):
        crop = img.crop(box)
        if crop.width < 1200:  # small text: upscale so the recognizer sees enough pixels per glyph
            crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
        for text, x0, x1 in _observations(crop):
            if (cut_left and x0 <= CROP_EDGE) or (cut_right and x1 >= 1 - CROP_EDGE):
                continue
            if _edge_cut(text, x0, x1, not cut_left, not cut_right):  # the crop's outer side is the image edge
                continue
            n = _norm(text)
            if not n or any(n in s for s in seen):
                continue
            seen = [s for s in seen if s not in n]  # a longer reading of an earlier fragment replaces it
            out = [t for t in out if _norm(t) not in n]
            out.append(text)
            seen.append(n)
    return out


def _norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def _spans(s: str) -> list[tuple[str, bool]]:
    """Upper-case alphanumeric tokens, also split where letters meet digits: 'Rm.121A' -> RM, 121, A.
    Each token comes with `glued`: True if it touches the previous token with no separator ('1ST' -> ST)."""
    out, end = [], -1
    for m in re.finditer(r"[A-Z]+|[0-9]+", s.upper()):
        out.append((m.group(), m.start() == end))
        end = m.end()
    return out


def _tokens(s: str) -> list[str]:
    """Digit runs are never split, so a number only matches a whole number ('12' never matches '121')."""
    return [t for t, _ in _spans(s)]


def _starts_word(spans: list[tuple[str, bool]], i: int) -> bool:
    """A run may not start on letters glued to a number: the 'ST' of '1ST' or the 'A' of '121A' is no word."""
    return not (spans[i][1] and spans[i][0][0].isalpha() and spans[i - 1][0][-1].isdigit())


def _concat_run(c: str, spans: list[tuple[str, bool]]) -> bool:
    """True if c equals the concatenation of a contiguous run of tokens ('ROOM121' ~ ROOM, 121). Two digit
    tokens are never glued: '1 21' / '1-21' are two numbers, not 121."""
    for i in range(len(spans)):
        if not _starts_word(spans, i):
            continue
        acc = ""
        for j in range(i, len(spans)):
            t = spans[j][0]
            if j > i and t[0].isdigit() and acc[-1].isdigit():
                break
            acc += t
            if acc == c:
                return True
            if len(acc) >= len(c) or not c.startswith(acc):
                break
    return False


def _has_run(cand: list[str], spans: list[tuple[str, bool]]) -> bool:
    n = len(cand)
    return any([t for t, _ in spans[i:i + n]] == cand and _starts_word(spans, i)
               for i in range(len(spans) - n + 1))


def grounded(candidate: str, ocr_lines: list[str]) -> bool:
    """True if the candidate is read by OCR as whole tokens (ignoring case, spacing and punctuation).

    It must equal a contiguous token run of one OCR line ('Room-121' grounds 'ROOM 121'; 'ROOM 12' does
    not ground '121', 'ROOM 121' does not ground '1', '1 21' does not ground '121', '1ST' does not ground
    'ST'), or, for a sign split over several OCR lines, its tokens must appear in order across consecutive
    lines ('ROOM 121 LAB' on lines 'ROOM 121' + 'LAB')."""
    c, cand = _norm(candidate), _tokens(candidate)
    if not c:
        return False
    if any(_concat_run(c, _spans(line)) for line in ocr_lines):
        return True
    flat = []
    for line in ocr_lines:
        flat += [(t, glued and k > 0) for k, (t, glued) in enumerate(_spans(line))]
    return len(cand) > 1 and _has_run(cand, flat)


_FLOOR_WORDS = {"FLOOR", "FLR", "FL", "LEVEL", "LVL", "STOREY", "STORY"}
_ORDINAL_SUFFIXES = {"ST", "ND", "RD", "TH"}
_ORDINAL_WORDS = {"FIRST": 1, "SECOND": 2, "THIRD": 3, "FOURTH": 4, "FIFTH": 5, "SIXTH": 6, "SEVENTH": 7,
                  "EIGHTH": 8, "NINTH": 9}
# Floors are 1-based and basements are not modelled (config.MIN floor is 1, VideoUpdate.start_floor >= 1):
# a level below or beside the numbered floors ('LOWER LEVEL 1', 'B1', 'LEVEL -1') is never floor N.
# Same list as floors._NOT_A_FLOOR.
_NOT_A_FLOOR = {"LOWER", "SUB", "BASEMENT", "UNDERGROUND", "PARKING", "GARAGE", "MEZZANINE"}
_NEGATIVE = re.compile(r"[-\u2212\u2013]\s*$")  # '-', minus sign or en dash right before a number


def _line_floors(line: str) -> set[int]:
    """Floor numbers an OCR line states: 'LEVEL 2', 'FL2', '2ND FLOOR', 'SECOND FLOOR', or a lone '2ND' / 'L2'.
    Negative levels ('LEVEL -1') and basement/parking levels state no floor."""
    up = line.upper()
    toks = _tokens(up)
    if _NOT_A_FLOOR & set(toks):
        return set()
    starts = [m.start() for m in re.finditer(r"[A-Z]+|[0-9]+", up)]
    out = set()
    for i, t in enumerate(toks):
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        after = toks[i + 2] if i + 2 < len(toks) else ""
        if t in _FLOOR_WORDS and nxt.isdigit():  # LEVEL 2, FLOOR 02, FL2
            if not _NEGATIVE.search(up[:starts[i + 1]]):  # not LEVEL -1
                out.add(int(nxt))
        elif t.isdigit() and nxt in _ORDINAL_SUFFIXES and after in _FLOOR_WORDS:  # 2ND FLOOR
            if not _NEGATIVE.search(up[:starts[i]]):
                out.add(int(t))
        elif t in _ORDINAL_WORDS and nxt in _FLOOR_WORDS:  # SECOND FLOOR
            out.add(_ORDINAL_WORDS[t])
    if len(toks) == 2 and toks[0].isdigit() and toks[1] in _ORDINAL_SUFFIXES and not _NEGATIVE.search(up[:starts[0]]):
        out.add(int(toks[0]))  # a lone '2ND' sign
    if len(toks) == 2 and toks[0] == "L" and toks[1].isdigit():  # a lone 'L2' sign
        out.add(int(toks[1]))
    return out


def _below_ground(hint: str) -> bool:
    """'-1', 'B1', 'B 2', 'basement', 'lower level 1': not one of the 1-based floors this app models."""
    s = hint.strip().upper()
    toks = _tokens(s)
    return bool(re.match(r"[-\u2212\u2013]\s*\d", s) or (_NOT_A_FLOOR & set(toks))
                or any(t == "B" and k + 1 < len(toks) and toks[k + 1].isdigit() for k, t in enumerate(toks)))


def floor_grounded(hint: str, ocr_lines: list[str]) -> bool:
    """True if some OCR line states the floor in Gemma's floor_hint ('2', '2nd', 'Level 2', 'second floor')
    next to a floor word. A bare number does not count: 'ROOM 121' never confirms floor 1, and a basement
    hint ('-1', 'B1') is never confirmed (floors are 1-based here)."""
    if _below_ground(hint):
        return False
    toks = _tokens(hint)
    num = next((int(t) for t in toks if t.isdigit()), None)
    if num is None:
        num = next((_ORDINAL_WORDS[t] for t in toks if t in _ORDINAL_WORDS), None)
    return num is not None and any(num in _line_floors(line) for line in ocr_lines)


def floor_from_lines(ocr_lines: list[str]) -> str:
    """The floor a short floor sign in the OCR states ('LEVEL 2' -> '2'), or '' if none or several do.
    Only short lines count (at most 5 tokens), so a poster sentence mentioning a floor is ignored."""
    found = set()
    for line in ocr_lines:
        if len(_tokens(line)) <= 5 and not phone_like(line):
            found |= {f for f in _line_floors(line) if 1 <= f <= config.MAX_FLOOR}
    return str(found.pop()) if len(found) == 1 else ""


# ---------------------------------------------------------------- which text is a wayfinding sign
_PLAQUE_PREFIX = r"(?:ROOM|RM|OFFICE|OFC|LAB|STE|SUITE)"
_ROOM_NO = r"[A-Z]?\d{1,4}[A-Z]?"
_PLAQUE = [
    re.compile(rf"(?:{_PLAQUE_PREFIX}\s*)?{_ROOM_NO}"),  # 121, 121A, ROOM 121, RM118, LAB 214
    re.compile(rf"{_ROOM_NO}(?:\s+[A-Z&']+){{1,3}}"),  # 214 CHEMISTRY LAB
]


def long_sentence(text: str) -> bool:
    """Running text (flyer, notice), not a sign: many words, or a sentence of several words."""
    words = text.split()
    return len(words) > 7 or (len(words) >= 5 and bool(re.search(r"[.!?](\s|$)", text)))


def sign_line(text: str) -> bool:
    """Could be a sign: not a phone number, running text or notice text ('4PM ROOM 305', 'FRIDAY 10AM')."""
    return not phone_like(text) and not long_sentence(text) and not notice(text)


def _plaque_shape(line: str) -> bool:
    s = re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9&' ]", " ", line.upper())).strip()
    return any(p.fullmatch(s) for p in _PLAQUE)


def plaque_like(line: str) -> bool:
    """A door plaque line: a number alone, ROOM/RM/OFFICE/LAB/STE + number, or number + short name.
    A directional sign ('ROOMS 120-130 ->', 'ROOM 121 →'), notice text and a partial read ('ROOM 21',
    'ROOM 10?') are not plaques."""
    if directional(line):
        return False
    return sign_line(line) and _plaque_shape(line) and not partial_plaque(line)


def partial_plaque(line: str) -> bool:
    """A plaque read only in part: a number followed by '?' / an apostrophe ('ROOM 10?'), or a plaque-shaped
    line whose number is shorter than a room number ('ROOM 21' of ROOM 216, a lone '2' of 214A). Its numbers
    are no room numbers and must not end up in a label."""
    if cut_number(line):
        return True
    if directional(line) or not _plaque_shape(line):
        return False
    nums = re.findall(r"[A-Z]?\d+[A-Z]?", re.sub(r"[^A-Z0-9 ]", " ", line.upper()))
    return bool(nums) and not any(re.fullmatch(ROOM_ID, n) or complete_room_id(n) for n in nums)


# Words that make an OCR line a wayfinding sign even next to a poster.
_WAYFINDING_WORDS = {"EXIT", "STAIR", "STAIRS", "STAIRWAY", "STAIRWELL", "ELEVATOR", "ELEVATORS", "LIFT", "RESTROOM",
                     "RESTROOMS", "TOILET", "TOILETS", "WC", "MEN", "WOMEN", "LEVEL", "FLOOR", "ENTRANCE", "LOBBY"}


def poster_lines(ocr_lines: list[str]) -> list[str]:
    """OCR lines that belong to a poster or notice: notice text itself ('4PM ROOM 305', 'CALL 555-0199'), and,
    when the frame shows notice text, its other lines that are not wayfinding (the headline 'SEMINAR',
    'MATH TUTORING' printed above the time or phone number). Plaques, floor, directional and exit/stairs/
    restroom signs in the same frame are kept apart."""
    notes = [line for line in ocr_lines if not sign_line(line)]
    if not notes:
        return []
    out = list(notes)
    for line in ocr_lines:
        if line in out or directional(line) or _plaque_shape(line) or _line_floors(line):
            continue
        if _WAYFINDING_WORDS & set(_tokens(line)):
            continue
        out.append(line)
    return out


def plaque_numbers(ocr_lines: list[str]) -> set[str]:
    """Room numbers (3-4 digits, optional letter suffix: 121, RM118 -> 118, 121A) on door-plaque-like lines."""
    out = set()
    for line in ocr_lines:
        if plaque_like(line):
            out |= set(re.findall(r"(?<![0-9])(\d{3,4}[A-Z]?)(?![0-9A-Z])", line.upper()))
    return out
