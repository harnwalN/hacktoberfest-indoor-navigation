"""On-device OCR (Apple Vision framework) used to ground Gemma's sign reading.

Small vision-language models confidently invent room numbers. Vision's text recognizer is offline,
~50 ms per frame, and only reports strings that are really there, so Gemma is told which strings
exist and anything it reports outside that list is dropped. Gemma still does all the understanding:
labels, place types, stairs, landmark phrases and same-place checks.

Not on macOS (or pyobjc missing)? `available()` is False and Gemma runs ungrounded.
"""
import re
from pathlib import Path

from . import config

try:
    import objc
    import Vision
    from Foundation import NSURL

    _AVAILABLE = True
except ImportError:  # Linux/Windows, or pyobjc not installed
    _AVAILABLE = False


def available() -> bool:
    return _AVAILABLE and config.OCR_ENABLED


def read_text(path: Path) -> list[str]:
    """Return recognized text lines with confidence >= OCR_MIN_CONFIDENCE."""
    if not available():
        return []
    with objc.autorelease_pool():
        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(NSURL.fileURLWithPath_(str(path)), None)
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
                    out.append(text)
        return out


def _norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def _tokens(s: str) -> list[str]:
    """Upper-case alphanumeric tokens, also split where letters meet digits: 'Rm.121A' -> RM, 121, A.

    Digit runs are never split, so a number only matches a whole number ('12' never matches '121')."""
    return re.findall(r"[A-Z]+|[0-9]+", s.upper())


def _concat_run(c: str, toks: list[str]) -> bool:
    """True if c equals the concatenation of a contiguous run of toks ('ROOM121' ~ ROOM, 121)."""
    for i in range(len(toks)):
        s = ""
        for t in toks[i:]:
            s += t
            if s == c:
                return True
            if len(s) >= len(c) or not c.startswith(s):
                break
    return False


def _has_run(cand: list[str], toks: list[str]) -> bool:
    n = len(cand)
    return any(toks[i:i + n] == cand for i in range(len(toks) - n + 1))


def grounded(candidate: str, ocr_lines: list[str]) -> bool:
    """True if the candidate is read by OCR as whole tokens (ignoring case, spacing and punctuation).

    It must equal a contiguous token run of one OCR line ('Room-121' grounds 'ROOM 121'; 'ROOM 12' does
    not ground '121', 'ROOM 121' does not ground '1'), or, for a sign split over several OCR lines, its
    tokens must appear in order across consecutive lines ('ROOM 121 LAB' on lines 'ROOM 121' + 'LAB')."""
    c, cand = _norm(candidate), _tokens(candidate)
    if not c:
        return False
    if any(_concat_run(c, _tokens(line)) for line in ocr_lines):
        return True
    return len(cand) > 1 and _has_run(cand, [t for line in ocr_lines for t in _tokens(line)])


_FLOOR_WORDS = {"FLOOR", "FLR", "FL", "LEVEL", "LVL", "STOREY", "STORY"}
_ORDINAL_SUFFIXES = {"ST", "ND", "RD", "TH"}
_ORDINAL_WORDS = {"FIRST": 1, "SECOND": 2, "THIRD": 3, "FOURTH": 4, "FIFTH": 5, "SIXTH": 6, "SEVENTH": 7,
                  "EIGHTH": 8, "NINTH": 9}


def _line_floors(line: str) -> set[int]:
    """Floor numbers an OCR line states: 'LEVEL 2', 'FL2', '2ND FLOOR', 'SECOND FLOOR', or a lone '2ND' / 'L2'."""
    toks = _tokens(line)
    out = set()
    for i, t in enumerate(toks):
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        after = toks[i + 2] if i + 2 < len(toks) else ""
        if t in _FLOOR_WORDS and nxt.isdigit():  # LEVEL 2, FLOOR 02, FL2
            out.add(int(nxt))
        elif t.isdigit() and nxt in _ORDINAL_SUFFIXES and after in _FLOOR_WORDS:  # 2ND FLOOR
            out.add(int(t))
        elif t in _ORDINAL_WORDS and nxt in _FLOOR_WORDS:  # SECOND FLOOR
            out.add(_ORDINAL_WORDS[t])
    if len(toks) == 2 and toks[0].isdigit() and toks[1] in _ORDINAL_SUFFIXES:  # a lone '2ND' sign
        out.add(int(toks[0]))
    if len(toks) == 2 and toks[0] == "L" and toks[1].isdigit():  # a lone 'L2' sign
        out.add(int(toks[1]))
    return out


def floor_grounded(hint: str, ocr_lines: list[str]) -> bool:
    """True if some OCR line states the floor in Gemma's floor_hint ('2', '2nd', 'Level 2', 'second floor')
    next to a floor word. A bare number does not count: 'ROOM 121' never confirms floor 1."""
    toks = _tokens(hint)
    num = next((int(t) for t in toks if t.isdigit()), None)
    if num is None:
        num = next((_ORDINAL_WORDS[t] for t in toks if t in _ORDINAL_WORDS), None)
    return num is not None and any(num in _line_floors(line) for line in ocr_lines)

