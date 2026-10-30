"""Small helpers shared by mapper.py and floors.py (kept here so neither imports the other)."""
import re

PLAQUE_PREFIX = r"(?:ROOM|RM|OFFICE|OFC|LAB|STE|SUITE)"
ROOM_ID = r"[A-Z]?\d{3,4}[A-Z]?"
_GLUED_PREFIX = re.compile(rf"{PLAQUE_PREFIX}({ROOM_ID})")  # ROOM118 / RM118A (after punctuation is dropped)


def text_tokens(texts: list[str]) -> set[str]:
    """Distinctive tokens only: things containing a digit (room numbers). 'EXIT' repeats on every floor.
    A plaque word glued to a room number counts as the number: 'ROOM-118' / 'RM118' -> 118 (as 'Room 118')."""
    out = set()
    for t in texts:
        for tok in re.split(r"[\s,;:/()]+", t.upper()):
            tok = re.sub(r"[^A-Z0-9]", "", tok)
            if m := _GLUED_PREFIX.fullmatch(tok):
                tok = m.group(1)
            if len(tok) >= 2 and any(ch.isdigit() for ch in tok):
                out.add(tok)
    return out


# ---------------------------------------------------------------- room numbers vs directional signs
ARROWS = "←→↑↓↖↗↘↙⇐⇒⇑⇓⇦⇨⇧⇩➔➙➛➜➝➞➟➠➡⬅⬆⬇"
_ASCII_ARROW = re.compile(r"->|<-|=>|<=|>>|<<")
_CHAIN = re.compile(r"(?<![A-Z0-9])\d+[A-Z]?(?:\s*(?:[-‐-―−~]|\bTO\b|\bTHRU\b|\bTHROUGH\b)\s*\d+[A-Z]?)+"
                    r"(?![A-Z0-9])")
_PART = re.compile(r"(\d+)[A-Z]?")


def _range_spans(up: str) -> list[tuple[int, int]]:
    """Spans of number ranges ('120-130', '120 – 130', '120 TO 130') in an upper-case text. Exactly two
    numbers of 2-4 digits; with a dash both must have as many digits, so '555-0147' (a phone number) and
    '2-105' (floor-room) are no ranges."""
    out = []
    for m in _CHAIN.finditer(up):
        parts = _PART.findall(m.group())
        if len(parts) != 2 or not all(2 <= len(p) <= 4 for p in parts):
            continue
        if len(parts[0]) == len(parts[1]) or re.search(r"\b(?:TO|THRU|THROUGH)\b", m.group()):
            out.append(m.span())
    return out


def is_range(text: str) -> bool:
    """True if the text contains a number range such as 'ROOMS 120-130'."""
    return bool(_range_spans(text.upper()))


def directional(text: str) -> bool:
    """A directional / hallway sign, not a door plaque: an arrow, a number range or a plural 'ROOMS ...'
    ('ROOMS 120-130 ->', '120-130', 'ROOM 121 →'). Its numbers are rooms somewhere else, never this place's."""
    up = text.strip().upper()
    return bool(any(ch in ARROWS for ch in up) or _ASCII_ARROW.search(up) or re.match(r"ROOMS\b", up)
                or _range_spans(up))


def room_id(text: str) -> str:
    """Canonical room number: 'RM 118' / 'Room 118' / 'ROOM-118' / 'rm.118a' -> '118' / '118A'.
    Anything else (a name, '2-105') is returned stripped and unchanged."""
    s = text.strip()
    m = re.fullmatch(rf"(?:{PLAQUE_PREFIX}\b\.?)?[\s.:#-]*(?:NO\b\.?[\s.:#-]*)?([A-Z]?\d{{1,4}})(?:[\s-]?([A-Z]))?",
                     s.upper())
    if not m:
        m = re.fullmatch(rf"{PLAQUE_PREFIX}([A-Z]?\d{{1,4}})([A-Z])?", s.upper())  # glued: RM118, ROOM118A
    return (m.group(1) + (m.group(2) or "")) if m else s


def room_ids(texts: list[str]) -> set[str]:
    """Room numbers (3-4 digits, optional letter prefix/suffix) a place's own sign texts state. Directional
    signs ('ROOMS 120-130 ->') say nothing about which room this is."""
    return {t for t in text_tokens([x for x in texts if not directional(x)]) if re.fullmatch(ROOM_ID, t)}


def _next_id(prefix: str, existing: list[str]) -> int:
    nums = [int(m.group(1)) for x in existing if (m := re.fullmatch(rf"{prefix}(\d+)", x))]
    return max(nums, default=0) + 1
