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


def grounded(candidate: str, ocr_lines: list[str]) -> bool:
    """True if the candidate string is supported by OCR (ignoring case, spacing and punctuation)."""
    c = _norm(candidate)
    if not c:
        return False
    for line in ocr_lines:
        n = _norm(line)
        if c in n or (len(n) >= 3 and n in c):
            return True
    return False


def ocr_tokens(ocr_lines: list[str]) -> set[str]:
    return {_norm(t) for line in ocr_lines for t in re.split(r"[\s,;:/()]+", line) if _norm(t)}
