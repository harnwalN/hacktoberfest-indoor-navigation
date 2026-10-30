"""Small helpers shared by mapper.py and floors.py (kept here so neither imports the other)."""
import re

PLAQUE_PREFIX = r"(?:ROOM|RM|OFFICE|OFC|LAB|STE|SUITE)"
ROOM_ID = r"[A-Z]?\d{3,4}[A-Z]?"
_GLUED_PREFIX = re.compile(rf"{PLAQUE_PREFIX}({ROOM_ID})")  # ROOM118 / RM118A (after punctuation is dropped)
PHONE = re.compile(r"(?:\+?\d[\s().\-]*){7,}")  # 7+ digits with only separators between: 555 0147


def phone_like(text: str) -> bool:
    """'Call 555 0147', '(858) 534-1234': seven or more digits in one number."""
    return bool(PHONE.search(text))


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
    signs ('ROOMS 120-130 ->') say nothing about which room this is, notice text ('SEMINAR 4PM ROOM 305')
    names a room somewhere else, and a cut read ('ROOM 21?') may be part of another number."""
    own = [x for x in texts if not directional(x) and not notice(x) and not cut_number(x)]
    return {t for t in text_tokens(own) if re.fullmatch(ROOM_ID, t)}


def place_tokens(texts: list[str]) -> set[str]:
    """Tokens that identify this place (text-hit evidence for 'same place'): text_tokens of its own signs only.
    A directional sign ('ROOMS 120-130 ->' -> '120130') is shown at many hallway spots, and a phone number
    ('555-0147' -> '5550147', '555 0147' -> '555') or any other notice text ('FRIDAY 10AM', '4PM ROOM 305')
    on every copy of a flyer: neither says where you are."""
    return text_tokens([PHONE.sub(" ", t) for t in texts if not directional(t) and not notice(t)])


# ---------------------------------------------------------------- notices and incomplete room numbers
# Poster / flyer / notice text: times, dates, weekdays, prices. Such a line names an event, not this place.
_MONTH = (r"(?:JAN(?:UARY)?|FEB(?:RUARY)?|MAR(?:CH)?|APR(?:IL)?|MAY|JUNE?|JULY?|AUG(?:UST)?|SEPT?(?:EMBER)?"
          r"|OCT(?:OBER)?|NOV(?:EMBER)?|DEC(?:EMBER)?)")
_ORD = r"(?:ST|ND|RD|TH)?"
_NOTICE = re.compile("|".join([
    r"(?<![A-Z0-9])\d{1,2}(?::\d{2})?\s*[AP]\.?M\b",  # 4PM, 10 AM, 10:30PM
    r"(?<![\d.:])\d{1,2}:\d{2}(?![\d:])",  # 10:30
    rf"\b{_MONTH}\b\.?\s*\d{{1,2}}{_ORD}\b",  # OCT 31, MARCH 3RD
    rf"(?<![A-Z0-9])\d{{1,2}}{_ORD}\s+{_MONTH}\b",  # 31 OCT
    r"(?<![\d/])\d{1,2}/\d{1,2}(?:/\d{2,4})?(?![\d/])",  # 10/31, 10/31/2026
    r"\b(?:MON|TUES?|WEDNES|THURS?|FRI|SATUR|SUN)DAYS?\b",  # FRIDAY, MONDAYS
    r"\b(?:MON|TUES?|WED|THU|THURS?|FRI|SAT|SUN)\b\.?,?\s*(?=\d)",  # FRI 10, TUE 5
    r"\b(?:TODAY|TONIGHT|TOMORROW)\b",
    r"[$€£]\s*\d|\d\s*[$€£]",  # $12, 12$
]))
# 'ROOM 305' inside running text: '4PM ROOM 305', 'MEET IN ROOM 305 AT NOON'. A plaque name before it
# ('SEMINAR ROOM 305', 'CONFERENCE ROOM 305') is not enough; a digit, time or sentence word is.
_EMBEDDED_ROOM = re.compile(rf"\b(?:ROOM|RM)\b\.?\s*#?\s*{ROOM_ID}\b")
_SENTENCE_WORDS = {"AM", "PM", "AT", "IN", "ON", "MEET", "MEETING", "JOIN", "SEE", "VISIT", "COME", "HELD",
                   "NOON", "UNTIL", "FROM", "EVERY", "CALL", "CONTACT", "EMAIL", "FREE"}
_FLOOR_PREFIX = re.compile(r"^(?:(?:LEVEL|FLOOR|FLR|FL|LVL)\s*\d+|\d+(?:ST|ND|RD|TH)\s+FLOOR|L\d)\s*$")


def _embedded_room(up: str) -> bool:
    m = _EMBEDDED_ROOM.search(up)
    if not m:
        return False
    before = re.sub(r"[^A-Z0-9:]+", " ", up[:m.start()]).strip()
    after = re.sub(r"[^A-Z0-9]+", " ", up[m.end():]).split()
    if before and not _FLOOR_PREFIX.match(before):  # 'LEVEL 2 ROOM 214' is a plaque
        words = before.split()
        if (any(ch.isdigit() for ch in before) or _SENTENCE_WORDS & set(words)
                or any(len(w) <= 3 and w.endswith(("AM", "PM")) for w in words)):  # 'APM ROOM 305' (4PM misread)
            return True
    return len(before.split()) + len(after) >= 4 or bool(_SENTENCE_WORDS & set(after))


def notice(text: str) -> bool:
    """Poster / flyer / notice text, never a wayfinding sign: a time ('4PM', '10:30'), date ('OCT 31',
    '10/31'), weekday, price ('$12'), phone number, or a 'ROOM 305' inside a longer text ('4PM ROOM 305').
    It never gives a room number, a label or same-place evidence."""
    up = text.upper()
    return bool(_NOTICE.search(up) or phone_like(text) or _embedded_room(up))


_CUT = re.compile(r"\d\s*[?¿…’'_]|\d\.\.\.")  # '10?' '2'' '21…': the recognizer was unsure what follows


def cut_number(text: str) -> bool:
    """A number that may continue past what was read: followed by '?', an apostrophe or an ellipsis
    ('ROOM 10?' for ROOM 101, 'ROOM 2'' for a plaque cut at the image edge)."""
    return bool(_CUT.search(text))


def complete_room_id(r: str) -> bool:
    """A plausible whole room number: 3-4 digits with an optional letter prefix/suffix ('121', '214A', 'B204'),
    or a hyphenated floor-room form ('2-105'). A shorter number ('21', '2', 'ROOM 10') is a partial read of a
    plaque (cut at the edge, too small or oblique) in buildings numbered like every scenario here.
    Text without digits (a name) is not a number and is left alone."""
    rid = room_id(r).upper()
    if not any(ch.isdigit() for ch in rid):
        return True
    return bool(re.fullmatch(ROOM_ID, rid) or re.fullmatch(r"[A-Z]?\d{1,2}-\d{3,4}[A-Z]?", rid))


def _next_id(prefix: str, existing: list[str]) -> int:
    nums = [int(m.group(1)) for x in existing if (m := re.fullmatch(rf"{prefix}(\d+)", x))]
    return max(nums, default=0) + 1
