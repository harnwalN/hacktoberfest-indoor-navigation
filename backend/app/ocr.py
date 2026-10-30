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
from .matching import directional, phone_like

try:
    import objc
    import Vision
    from Foundation import NSURL, NSData

    _AVAILABLE = True
except ImportError:  # Linux/Windows, or pyobjc not installed
    _AVAILABLE = False


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


def read_text_with_crops(path: Path) -> list[str]:
    """OCR of the full (EXIF-upright) image plus zoomed LEFT/RIGHT crops (the same 55% halves Gemma sees), so
    small corner plaques that the full view misses are read too. Lines are deduplicated; a crop line touching
    the cut edge (possibly half a word) or contained in an already-read line is dropped."""
    if not available():
        return []
    img = ImageOps.exif_transpose(Image.open(path))
    img.load()
    w, h = img.size
    out = [t for t, _, _ in _observations(img)]
    seen = [_norm(t) for t in out]
    for box, cut_left, cut_right in (((0, 0, int(w * 0.55), h), False, True),
                                     ((int(w * 0.45), 0, w, h), True, False)):
        crop = img.crop(box)
        if crop.width < 1200:  # small text: upscale so the recognizer sees enough pixels per glyph
            crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
        for text, x0, x1 in _observations(crop):
            if (cut_left and x0 <= CROP_EDGE) or (cut_right and x1 >= 1 - CROP_EDGE):
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
    return not phone_like(text) and not long_sentence(text)


def plaque_like(line: str) -> bool:
    """A door plaque line: a number alone, ROOM/RM/OFFICE/LAB/STE + number, or number + short name.
    A directional sign ('ROOMS 120-130 ->', 'ROOM 121 →') is not a plaque."""
    if directional(line):
        return False
    s = re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9&' ]", " ", line.upper())).strip()
    return sign_line(line) and any(p.fullmatch(s) for p in _PLAQUE)


def plaque_numbers(ocr_lines: list[str]) -> set[str]:
    """Room numbers (3-4 digits, optional letter suffix: 121, RM118 -> 118, 121A) on door-plaque-like lines."""
    out = set()
    for line in ocr_lines:
        if plaque_like(line):
            out |= set(re.findall(r"(?<![0-9])(\d{3,4}[A-Z]?)(?![0-9A-Z])", line.upper()))
    return out
