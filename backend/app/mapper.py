"""Turn one processed walk (keyframes + Gemma analyses + motion track) into graph nodes/edges and
merge it into the building map, closing loops when the walker revisits a known place.

Pipeline per walk:
  1. walk nodes:    keyframes that matter (turns, glances, dwells, signs, stairs boundaries, ends)
  2. loop closure:  match each walk node against known nodes by room-number text and ORB geometry,
                    verified by Gemma ("same physical spot?"); auto-merge or suggest for review. Gemma's
                    "same" alone never merges: it needs strong ORB or a shared plaque number (_gemma_backed)
  3. drift fix:     every ORB-verified match yields a heading constraint; fit c(t) = c0 + k*t so the
                    walk's integrated heading lines up with the building frame (and itself, on loops)
  4. edges:         consecutive walk nodes -> edges with heading_out / heading_in and vertical
  5. finalize:      Manhattan snap of all edge headings to the building's dominant axis
"""
import math
import re
from difflib import SequenceMatcher
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from . import config, gemma
from .floors import (  # noqa: F401  (re-exported: mapper.room_floor etc. keep working)
    _RANK, _add_vertical_edge, _adopt_floor, _connected, _floor_rank, _floors_compatible, _is_vertical,
    _link_floors, _merge_floor_conflict, _shift_guessed_walk, assign_floors, mark_climbs, room_floor,
    set_walk_floor, stair_rel_floors, stair_states,
)
from .matching import PLAQUE_PREFIX, ROOM_ID, _next_id, directional, place_tokens, room_ids
from .motion import detect_climbs, focal_px, median_heading
from .ocr import sign_line
from .schemas import (
    FLIP_VERTICAL, Edge, FrameAnalysis, Keyframe, MapBundle, Node, SuggestedMerge,
)

# Structural moments that always become nodes. Glances only do if Gemma finds a sign/landmark there.
NODE_KINDS = {"start", "end", "turn_start", "dwell"}
IMPORTANT_TYPES = {"intersection", "entrance", "stairs", "elevator", "exit", "restroom", "room", "landmark"}
KEY_TYPES = {"entrance", "stairs", "elevator", "exit", "restroom"}  # always worth a node + auto-verify
# Key places a visitor routes to, by family: an exit door is also a way in, so exit and entrance are one family
# (one place seen twice may be typed either; eval/score.py counts false key places by the same table).
KEY_FAMILY = {"exit": "exit", "entrance": "exit", "restroom": "restroom", "elevator": "elevator"}
KIND_PRIORITY = {"turn_start": 0, "glance": 1, "start": 2, "end": 2, "dwell": 3}


def wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def wrap360(a: float) -> float:
    return a % 360.0


# ---------------------------------------------------------------- text + geometry matching
GENERIC_WORDS = {
    "ENTRANCE", "EXIT", "ROOM", "ROOMS", "HALL", "HALLWAY", "STAIR", "STAIRS", "STAIRWELL", "ELEVATOR",
    "ELEVATORS", "RESTROOM", "RESTROOMS", "MENS", "WOMEN", "WOMENS", "FLOOR", "LEVEL", "OFFICE", "OFFICES",
    "LOBBY", "DOOR", "DOORS", "CLOSED", "OPEN", "CAUTION", "NOTICE", "WARNING", "PLEASE", "THIS", "THAT",
    "WITH", "FROM", "ONLY", "AREA", "MAIN", "BUILDING", "SUITE", "LABORATORY", "CENTER", "CENTRE",
}


def sign_words(texts: list[str]) -> set[str]:
    """Distinctive words from sign text (named places like MAKER SPACE, QUALCOMM INSTITUTE)."""
    out = set()
    for t in texts:
        for w in re.split(r"[^A-Za-z]+", t.upper()):
            if len(w) > 4 and w.endswith("S"):
                w = w[:-1]  # MAKERS == MAKER
            if len(w) >= 4 and w not in GENERIC_WORDS:
                out.add(w)
    return out


def shared_sign_words(a: list[str], b: list[str]) -> set[str]:
    """Shared distinctive words, tolerant of OCR merges/fragments (MAKERSPACE vs MAKER SPACE)."""
    wa, wb = sign_words(a), sign_words(b)
    shared = wa & wb
    ja, jb = "".join(sorted(wa)), "".join(sorted(wb))
    shared |= {w for w in wa if len(w) >= 5 and w in jb.replace(" ", "")}
    shared |= {w for w in wb if len(w) >= 5 and w in ja.replace(" ", "")}
    return shared


def strong_name_match(a: list[str], b: list[str]) -> bool:
    """Same named place: shared distinctive words that make up most of the (short) sign.
    Busy text (posters, wall graphics) is never used: it is visible from many places."""
    wa, wb = sign_words(a), sign_words(b)
    if not wa or not wb or len(wa) > 6 or len(wb) > 6:
        return False
    s = shared_sign_words(a, b)
    return (len(s) >= 2 or any(len(w) >= 7 for w in s)) and len(s) >= 0.5 * min(len(wa), len(wb))


def _letters(texts: list[str]) -> str:
    return re.sub(r"[^A-Z]", "", " ".join(texts).upper())


def fuzzy_name_match(a: list[str], b: list[str], threshold: float = 0.65) -> bool:
    """OCR reads one sign in pieces from different angles ('MAK SPAC', 'KERSPAGE', 'MAKER SPACE').
    Compare the letters of the shorter text with the best-matching stretch of the longer one."""
    la, lb = _letters(a), _letters(b)
    if min(len(la), len(lb)) < 7 or max(len(la), len(lb)) > 40:
        return False  # too little to go on, or busy text (posters, wall graphics)
    short, long_ = sorted((la, lb), key=len)
    width = len(short) + 3
    best = max(SequenceMatcher(None, short, long_[i:i + width]).ratio()
               for i in range(0, max(1, len(long_) - len(short) + 1)))
    return best >= threshold


@lru_cache(maxsize=2048)
def _orb_features(path: str, zoom: float = 1.0):
    """ORB keypoints of a photo; zoom < 1 keeps only the centre (that fraction of width and height), enlarged: the
    view ahead as it looks a few metres closer."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None, (0, 0)
    h, w = img.shape
    if zoom < 1.0:
        ch, cw = int(h * zoom), int(w * zoom)
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        img = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_CUBIC)
    s = 640 / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    kp, des = cv2.ORB_create(nfeatures=1500).detectAndCompute(img, None)
    return [k.pt for k in kp], des, img.shape


def _homography(fa, fb):
    """(inliers, homography or None) between two _orb_features results."""
    (pa, da, _), (pb, db, _) = fa, fb
    if da is None or db is None or len(da) < 10 or len(db) < 10:
        return 0, None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m for m, *rest in pairs if rest and m.distance < 0.75 * rest[0].distance]
    if len(good) < 10:
        return len(good) // 2, None
    src = np.float32([pa[m.queryIdx] for m in good])
    dst = np.float32([pb[m.trainIdx] for m in good])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return 0, None
    return int(mask.sum()), H


def zoomed_inliers(path_a: Path, path_b: Path, zoom: float = 0.67) -> int:
    """Feature matches between two photos of one sign taken from different distances: the farther view's centre,
    enlarged, against the nearer view (either photo may be the farther one). Plain ORB loses a sign that is a
    third of the size in one photo."""
    a, b = _orb_features(str(path_a)), _orb_features(str(path_b))
    za, zb = _orb_features(str(path_a), zoom), _orb_features(str(path_b), zoom)
    return max(_homography(za, b)[0], _homography(zb, a)[0])


def orb_match(path_a: Path, path_b: Path) -> tuple[int, float | None]:
    """Return (homography inliers, yaw of B relative to A in degrees, if well-determined)."""
    fa = _orb_features(str(path_a))
    inliers, H = _homography(fa, _orb_features(str(path_b)))
    if H is None:
        return inliers, None
    h, w = fa[2]
    centre = cv2.perspectiveTransform(np.float32([[[w / 2, h / 2]]]), H)[0, 0]
    dx = float(centre[0] - w / 2)
    rel = -math.degrees(math.atan2(dx, focal_px(w, h)))
    return inliers, (rel if inliers >= config.ORB_WEAK_INLIERS and abs(rel) < 45 else None)


# ---------------------------------------------------------------- walk nodes
@dataclass
class WalkNode:
    kf: Keyframe
    t_arrive: float
    t_depart: float
    analysis: FrameAnalysis
    depart_frame: str | None = None
    rel_floor: int = 0
    node_id: str | None = None  # map node this walk node became / merged into
    vertical: str | None = None  # "stairs" / "elevator" access point (see _walk_vertical), whatever Gemma called it
    extra_texts: list[str] = field(default_factory=list)
    glances: int = 0  # glance events (looks at the side) this node is made of

    @property
    def texts(self) -> list[str]:
        return list(dict.fromkeys(wayfinding_texts(self.analysis) + self.extra_texts))


def _stairs_state(a: FrameAnalysis | None) -> str:
    if a is None:
        return "none"
    return {"on_stairs_up": "up", "on_stairs_down": "down"}.get(a.stairs, "none")


# Safety/notice words OCR picks up that are never wayfinding (Gemma sometimes keeps them anyway).
NOT_WAYFINDING = {"CAUTION", "NOTICE", "WARNING", "SECURITY", "CAMERAS", "CAMERA", "PREMISES", "AUTOMATIC",
                  "DOOR", "PUSH", "PULL", "ON", "THE", "AND", "FOR", "OPERATE", "ACTIVATE", "SWITCH", "TO",
                  "WET", "FLOOR", "DANGER", "EMERGENCY", "EVACUATION", "PLAN", "DO", "NOT", "ENTER"}


def wayfinding_texts(a: FrameAnalysis | None) -> list[str]:
    if a is None:
        return []
    keep = []
    for t in a.texts + a.room_numbers:
        words = [re.sub(r"[^A-Z0-9]", "", w) for w in t.upper().split()]
        words = [w for w in words if w]
        if words and not all(w in NOT_WAYFINDING for w in words):
            keep.append(t)
    return list(dict.fromkeys(keep))


def _has_sign(a: FrameAnalysis | None) -> bool:
    return bool(wayfinding_texts(a))


# ---------------------------------------------------------------- key places need their own sign
# A place is an exit / entrance / restroom / elevator only if a sign on ITS frame says so. Gemma's place_type
# alone is a guess: small models call the end wall of a corridor, a door at a corridor end, a 'LEVEL 1' sign or
# a green 'NO EXIT' plate an exit or entrance. Staircases are recognizable from the picture (no sign needed).
SIGNED_TYPES = ("exit", "entrance", "restroom", "elevator")
KEY_SIGN_WORDS = {
    "exit": {"EXIT", "EXITS"},
    "entrance": {"ENTRANCE", "ENTRANCES", "ENTRY", "ENTER"},
    "restroom": {"RESTROOM", "RESTROOMS", "TOILET", "TOILETS", "WC", "LAVATORY", "BATHROOM", "BATHROOMS", "MEN",
                 "MENS", "WOMEN", "WOMENS", "LADIES", "GENTLEMEN", "GENTS"},
    "elevator": {"ELEVATOR", "ELEVATORS", "LIFT", "LIFTS"},
}
_NEGATION = {"NO", "NOT", "NON"}  # 'NO EXIT', 'NOT AN EXIT', 'NON-EXIT', 'DO NOT ENTER', 'NO ENTRY'
_NEG_SCOPE = 2  # a negation reaches the next word or the one after ('NOT AN EXIT', 'NO THROUGH EXIT')
_ARTICLES = {"A", "AN", "THE"}  # 'NOT AN' + 'EXIT' on two lines: the negation still reaches the next line
_KEY_WORDS = set().union(*KEY_SIGN_WORDS.values())
_OCR_LOOKALIKE = str.maketrans("01", "OI")


def _ocr_letters(token: str) -> str:
    """A word OCR spelled with a 0 for an O or a 1 for an I ('N0', 'EX1T', 'N0EXIT') back in letters. Only a token
    that is mostly letters and has no other digit: plaques ('107', '1O7', '214A', '101') keep their digits."""
    digits = sum(ch.isdigit() for ch in token)
    if not digits or digits > len(token) - digits or any(ch.isdigit() and ch not in "01" for ch in token):
        return token
    return token.translate(_OCR_LOOKALIKE)


def _sign_words(text: str) -> list[str]:
    """Upper-case words, with a negation OCR glued on split off again ('NOEXIT' -> 'NO', 'EXIT') and OCR's 0-for-O /
    1-for-I read back as letters ('N0 EXIT' -> 'NO', 'EXIT')."""
    out = []
    text = re.sub(r"[A-Z0-9]+", lambda m: _ocr_letters(m.group()), text.upper())
    for w in re.findall(r"[A-Z]+", text):
        neg = next((n for n in ("NOT", "NON", "NO") if w.startswith(n) and w[len(n):] in _KEY_WORDS), None)
        out += [neg, w[len(neg):]] if neg else [w]
    return out


def _negation_of(words: list[str], i: int) -> int | None:
    """Index of the negation word that negates key word `words[i]`, else None. Its scope is the next one or two
    words, and it negates only the first key word in it: 'NO ENTRY EXIT ONLY', 'DO NOT ENTER EXIT ONLY' negate
    the entry, not the exit. 'DO NOT <verb>' negates the verb: 'DO NOT BLOCK EXIT' is an exit."""
    for j in range(i - 1, max(-1, i - 1 - _NEG_SCOPE), -1):
        if words[j] not in _NEGATION:
            continue
        between = words[j + 1:i]
        if any(w in _KEY_WORDS for w in between) or (between and j > 0 and words[j - 1] == "DO"):
            return None
        return j
    return None


def _key_hits(text: str) -> tuple[list[str], list[tuple[int, str, int | None]]]:
    """The text's words, and (index, kind, index of its negation or None) for each key word in it."""
    words = _sign_words(text)
    hits = [(i, k, _negation_of(words, i)) for i, w in enumerate(words)
            for k, kw in KEY_SIGN_WORDS.items() if w in kw]
    return words, hits


def names_key(text: str, kind: str) -> bool:
    """The sign text says this is a `kind` place: one of its words, not negated by the one or two words before it
    ('EMERGENCY EXIT', 'EXIT ONLY', 'NO ENTRY EXIT ONLY', 'DO NOT BLOCK EXIT' yes; 'NO EXIT', 'NOEXIT', 'NON-EXIT',
    'NOT AN EXIT', 'DO NOT ENTER' no). A text that negates the kind anywhere never names it ('NOT AN EXIT. EXIT
    AHEAD'): which word is right is unknown, and a false exit is worse than a missed one."""
    negs = [neg is not None for _, k, neg in _key_hits(text)[1] if k == kind]
    return bool(negs) and not any(negs)


def negated_key(text: str) -> bool:
    """A sign that says a place is NOT a key place ('NO EXIT', 'DO NOT ENTER')."""
    return any(neg is not None for _, _, neg in _key_hits(text)[1])


def _negated_spans(text: str, kind: str) -> list[list[str]]:
    """The words of each negated `kind` sign in the text: from the negation to the key word, plus the word after
    ('NO EXIT DOOR')."""
    words, hits = _key_hits(text)
    return [words[neg:i + 2] for i, k, neg in hits if k == kind and neg is not None]


def _split_signs(lines: list[str]) -> list[str]:
    """Neighbouring lines read as one sign where a line ends in a negation, or a negation and an article: OCR splits
    a 'NO EXIT' plate into 'NO' + 'EXIT', 'NOT AN EXIT' into 'NOT AN' + 'EXIT' or 'NOT' + 'AN' + 'EXIT' (an
    article alone on the next line: the chain goes on to the line after, three lines at most; a line that is only
    the negation may come after the line it belongs to). 'NO SMOKING' + 'EXIT' stays two."""
    out = []
    for i, line in enumerate(lines):
        words = _sign_words(line)
        while words and words[-1] in _ARTICLES and len(words) > 1:
            words = words[:-1]  # 'NOT AN' / 'NOT A' / 'NOT THE': the negation reaches past the article
        if not words or words[-1] not in _NEGATION:
            continue
        if i + 1 < len(lines):
            out.append(f"{line} {lines[i + 1]}")
            nxt = _sign_words(lines[i + 1])
            if i + 2 < len(lines) and nxt and all(w in _ARTICLES for w in nxt):
                out.append(f"{line} {lines[i + 1]} {lines[i + 2]}")  # 'NOT' + 'AN' + 'EXIT'
        if i > 0 and len(words) == 1:
            out.append(f"{line} {lines[i - 1]}")
    return out


def _within(words: list[str], span: list[str]) -> bool:
    return any(span[i:i + len(words)] == words for i in range(len(span) - len(words) + 1))


def key_sign(a: FrameAnalysis | None, kind: str) -> bool:
    """A sign naming `kind` on this frame: in Gemma's texts (OCR-checked when grounded), or an OCR line of the frame
    that Gemma left out. Without OCR, Gemma's texts are the only reading, but its place_type is still not enough.

    Negation is read over the whole frame, not one line at a time: OCR splits a 'NO EXIT' plate into the lines
    'NO', 'EXIT', or Gemma reads 'NO EXIT' where OCR only caught 'EXIT'. A frame carrying a negated `kind` sign is
    that kind only through a separate positive sign, one that is not a piece of the negated one ('NO EXIT' +
    'FIRE EXIT'); a bare 'EXIT' next to 'NO EXIT' is taken for a piece of it (a false exit is worse than a missed
    one)."""
    if a is None:
        return False
    lines = list(a.texts)
    if a.grounded:
        lines += [line for line in a.ocr if sign_line(line)]
    positive = [t for t in lines if names_key(t, kind)]
    if not positive:
        return False
    sources = [*a.texts, *a.ocr, *_split_signs(list(a.texts)), *_split_signs(list(a.ocr))]
    spans = [s for src in sources for s in _negated_spans(src, kind)]
    return any(not any(_within(_sign_words(t), s) for s in spans) for t in positive)


def place_type(a: FrameAnalysis | None) -> str:
    """Gemma's place_type, grounded: an exit/entrance/restroom/elevator without its sign on the frame is the room
    whose plaque the frame shows, else a hallway spot."""
    if a is None:
        return "hallway"
    t = a.place_type
    if t in SIGNED_TYPES and not key_sign(a, t):
        return "room" if room_ids(a.room_numbers) else "hallway"
    return t


def _unsigned_key(a: FrameAnalysis | None) -> str | None:
    """The key type Gemma claimed but no sign on the frame confirms (see place_type), else None."""
    return a.place_type if a is not None and a.place_type in SIGNED_TYPES and place_type(a) != a.place_type else None


def _key_place(a: FrameAnalysis | None) -> bool:
    """Entrances/restrooms/exits/elevators need their own sign on the frame (place_type);
    a staircase is recognizable from the picture alone."""
    if a is None:
        return False
    if a.place_type == "stairs" or a.stairs in ("ahead_up", "ahead_down"):
        return True
    return place_type(a) in KEY_TYPES


def _worthy(kf: Keyframe) -> bool:
    a = kf.analysis
    if kf.kind in NODE_KINDS:
        return True
    return bool(a and (a.room_numbers or _key_place(a) or (place_type(a) == "room" and _has_sign(a))
                       or _no_entry_sign(a)))


def _no_entry_sign(a: FrameAnalysis | None) -> str | None:
    """A 'NO EXIT' / 'DO NOT ENTER' sign that Gemma took for an exit/entrance: the door is worth a place (a
    landmark named by its sign), but it is not a key place."""
    if _unsigned_key(a) is None:
        return None
    return next((t for t in a.texts if negated_key(t)), None)


ROOM_LIKE = {"room", "restroom", "exit"}  # places with their own identity, never a stair landing


def _walk_vertical(a: FrameAnalysis, boundary: bool) -> str | None:
    """Stairs/elevator access point from what the walk shows, not from the place label: Gemma often calls the
    top or bottom of a flight a "hallway". `boundary` = last frame before a climb or first frame after it.
    Stairs merely seen ahead (ahead_up/ahead_down) do not count: the walker may never go near them, and a
    flagged node on each of two floors would be enough for _link_floors to guess a floor link."""
    if boundary or a.place_type == "stairs":
        return "stairs"
    if place_type(a) == "elevator":  # an elevator needs its sign, like any key place (place_type)
        return "elevator"
    return None


def walk_nodes(keyframes: list[Keyframe]) -> list[WalkNode]:
    kfs = sorted(keyframes, key=lambda k: k.t)
    # Landings between two same-direction flights count as mid-flight (floors.stair_states, B6): the landing
    # is not a node or a floor boundary; its photos still appear along the stairs edge.
    states = stair_states(kfs)
    worthy = [_worthy(k) for k in kfs]
    boundary = [False] * len(kfs)
    # Stair boundaries: last frame before climbing and first frame after.
    for i in range(1, len(kfs)):
        if states[i] != states[i - 1]:
            if states[i - 1] == "none":
                worthy[i - 1] = boundary[i - 1] = True
            if states[i] == "none":
                worthy[i] = boundary[i] = True

    nodes: list[WalkNode] = []
    for i, k in enumerate(kfs):
        if k.kind == "turn_end" and nodes and nodes[-1].kf.kind == "turn_start":
            nodes[-1].depart_frame = k.frame_id
        if not worthy[i] or states[i] != "none":
            continue  # never put a node in the middle of a staircase
        a = k.analysis or FrameAnalysis()
        wn = WalkNode(kf=k, t_arrive=k.t_arrive if k.t_arrive is not None else k.t,
                      t_depart=k.t_depart if k.t_depart is not None else k.t, analysis=a,
                      vertical=_walk_vertical(a, boundary[i]), glances=int(k.kind == "glance"))
        prev = nodes[-1] if nodes else None
        if prev and _absorbs(kfs, states, prev, wn):
            _absorb(prev, wn)
            continue
        nodes.append(wn)

    # Relative floor per walk node: count completed climbs (a flight, or flights joined by landings).
    j = 0
    for k, floor in zip(kfs, stair_rel_floors(states)):
        while j < len(nodes) and nodes[j].kf.t <= k.t:
            nodes[j].rel_floor = floor
            j += 1
    return nodes


def _absorbs(kfs, states, prev: WalkNode, wn: WalkNode) -> bool:
    """Whether walk node `wn` is just more of `prev` (one place) rather than a place of its own.

    Close in time is not enough on its own: walking past doors, the next plaque is in view a second after
    the last glance. Different room numbers are always two places, and two separate glances (look left at
    one door, then right at another) are two places unless they show the same place (_same_spot)."""
    if _crosses_stairs(kfs, states, prev, wn) or _keep_apart(prev, wn):
        return False
    if _rooms_conflict(_walk_rooms(prev), _walk_rooms(wn)):
        return False
    gap = wn.t_arrive - prev.t_depart
    same_spot = gap < config.SAME_PLACE_WINDOW_SEC and _same_spot(prev, wn)
    if prev.glances and wn.glances:
        return same_spot
    return gap < config.NODE_MIN_GAP_SEC or same_spot or (gap < config.STAIR_APPROACH_SEC and _stair_turn(prev, wn))


def _stair_turn(a: WalkNode, b: WalkNode) -> bool:
    """A plain corridor turn right before / after a stair boundary: the walker turned onto (or off) the flight there,
    so it is the landing, not a junction of its own (two nodes would split the stair place between walks)."""
    stairs, other = (a, b) if a.vertical == "stairs" else (b, a)
    return (stairs.vertical == "stairs" and not other.vertical and other.kf.kind == "turn_start"
            and place_type(other.analysis) in ("hallway", "intersection") and not _walk_rooms(other))


def _walk_rooms(wn: WalkNode) -> set[str]:
    """Room numbers a walk node states: its own frame's plaques plus those of the frames it absorbed (which
    _absorb keeps consistent, so absorbed text never links two rooms)."""
    return room_ids(wn.texts)


def _norm_label(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _keep_apart(a: WalkNode, b: WalkNode) -> bool:
    """Keep the walk's start/end separate from an adjacent turn, so that turn stays routable."""
    kinds = {a.kf.kind, b.kf.kind}
    if kinds & {"start", "end"} and "turn_start" in kinds:
        return True
    # A stair landing and the door beside it are two places: one node would either be a "room" that is
    # stairs (and then never merges with the same room seen on another walk) or lose the stairs.
    return bool(a.vertical) != bool(b.vertical) and bool({place_type(a.analysis), place_type(b.analysis)} & ROOM_LIKE)


def _same_spot(a: WalkNode, b: WalkNode) -> bool:
    """Consecutive frames describing the same place (e.g. three frames of the same entrance)."""
    if b.kf.kind == "turn_start" and a.kf.kind == "turn_start":
        return False  # two real junctions
    ra, rb = place_tokens(a.texts), place_tokens(b.texts)
    if ra and rb:
        return bool(ra & rb)  # room numbers decide
    if strong_name_match(wayfinding_texts(a.analysis), wayfinding_texts(b.analysis)):
        return True  # several frames of the same named place (e.g. MAKER SPACE)
    la, lb = _norm_label(a.analysis.suggested_label), _norm_label(b.analysis.suggested_label)
    if la and la == lb:
        return True
    if _stairish(a) and _stairish(b):
        return True  # the landing and the stairs seen from it ('Level 1' sign, then the 'STAIRS' sign)
    ta = place_type(a.analysis)
    return ta == place_type(b.analysis) and ta in {"entrance", "stairs", "elevator", "exit", "restroom"}


def _stairish(w: WalkNode) -> bool:
    return w.vertical == "stairs" or place_type(w.analysis) == "stairs"


def _crosses_stairs(kfs, states, a: WalkNode, b: WalkNode) -> bool:
    return any(s != "none" for k, s in zip(kfs, states) if a.kf.t < k.t < b.kf.t)


def _absorb(keep: WalkNode, other: WalkNode) -> None:
    """Merge two walk nodes that are too close in time into one (callers check _absorbs first).

    A place that already states room numbers takes no other room's plaque from the frames it absorbs (a frame
    showing this door and the next one): otherwise that number would make the next door "the same spot" too
    and chain two rooms into one node."""
    def rank(w: WalkNode):
        # a frame of the place itself (the glance at the door) describes it better than a corridor view of
        # its plaque: a turn that also glanced at Room 105 is Room 105, a corner merely in view of it is not
        corridor = place_type(w.analysis) in ("hallway", "intersection")
        return (0 if _has_sign(w.analysis) else 1, int(corridor), KIND_PRIORITY.get(w.kf.kind, 9))
    mine = _walk_rooms(keep)
    rooms_k, rooms_o = room_ids(wayfinding_texts(keep.analysis)), room_ids(wayfinding_texts(other.analysis))
    # rank before the frame swap below: the glance's "Room 101" must win over the start's "Hallway with ROOM 101"
    takes_analysis = rank(other) < rank(keep) and (not mine or rooms_o <= mine)
    # A numbered door and an exit (entrance...) sign seen from it are one stop: the door, where the walker stood.
    # The exit is down the hall; as one node it would be an 'Exit' claiming the plaques ('EXIT, ROOM 114, RM 112').
    if rooms_k and place_type(other.analysis) in SIGNED_TYPES and not rooms_o:
        takes_analysis = False
    elif rooms_o and place_type(keep.analysis) in SIGNED_TYPES and not rooms_k:
        takes_analysis = not mine or rooms_o <= mine
    final = other.analysis if takes_analysis else keep.analysis
    key = place_type(final) in SIGNED_TYPES and not room_ids(wayfinding_texts(final))

    def no_key(texts: list[str]) -> list[str]:  # a room keeps no exit sign seen from its door
        return texts if key else [t for t in texts if not any(names_key(t, k) for k in SIGNED_TYPES)]

    def own(texts: list[str]) -> list[str]:  # a plaqueless exit takes no plaque from the frames it absorbs
        return no_key([t for t in texts if (room_ids([t]) <= mine if mine else not (key and room_ids([t])))])

    if KIND_PRIORITY.get(other.kf.kind, 9) < KIND_PRIORITY.get(keep.kf.kind, 9):
        keep.kf, keep.depart_frame = other.kf, other.depart_frame or keep.depart_frame
    if takes_analysis:
        keep.extra_texts += no_key(wayfinding_texts(keep.analysis)) + own(other.extra_texts)
        keep.analysis = other.analysis
    else:
        keep.extra_texts += own(other.texts)
    keep.extra_texts = list(dict.fromkeys(t for t in no_key(keep.extra_texts) if not (key and room_ids([t]))))
    keep.t_arrive, keep.t_depart = min(keep.t_arrive, other.t_arrive), max(keep.t_depart, other.t_depart)
    keep.vertical = keep.vertical or other.vertical  # stair-ness survives even when the other frame's analysis wins
    keep.glances += other.glances


# ---------------------------------------------------------------- labels
_VERTICAL_WORDS = {"stairs": ("STAIR",), "elevator": ("ELEVATOR", "LIFT")}
_FLOOR_WORDS = re.compile(r"\b(?:LEVEL|FLOOR|FLR|LVL)\b|\b\d+(?:ST|ND|RD|TH)\b|\bL\d+\b")
_BARE_LABELS = {"room", "hallway", "intersection", "landmark", "corridor", "door"}  # a type, not a name


def _label(wn: WalkNode) -> tuple[str, str]:
    label, ptype = _raw_label(wn)
    return tidy_label(label, wn.texts, ptype), ptype


def _raw_label(wn: WalkNode) -> tuple[str, str]:
    a = wn.analysis
    ptype = place_type(a)
    if wn.vertical and ptype in ("hallway", "intersection"):
        ptype = wn.vertical  # the walk climbed here: it is the stairwell, not a corridor
    if ptype in _VERTICAL_WORDS:
        # named for what it is: a floor sign in view ('Level 1 Corridor', 'Level 2 Entrance') does not name stairs,
        # and a floor number in the name ('Level 2 Stairwell') is often the sign at the other end of the flight
        up = a.suggested_label.upper()
        if any(w in up for w in _VERTICAL_WORDS[ptype]) and not _FLOOR_WORDS.search(up):
            return a.suggested_label.strip(), ptype
        return ("Stairs" if ptype == "stairs" else "Elevator"), ptype
    label = a.suggested_label.strip()
    if (ptype in ("room", "hallway", "intersection", "landmark") and a.texts and all(map(negated_key, a.texts))
            and not _walk_rooms(wn)):
        return _sign_name(a.texts[0]), "landmark"  # a door whose only sign is 'NO EXIT', whatever Gemma typed it
    if (claimed := _unsigned_key(a)) and ptype in ("hallway", "intersection"):
        # Gemma called it an exit/entrance/... but no sign on the frame says so
        if (no := _no_entry_sign(a) or next((t for t in wn.texts if negated_key(t)), None)):
            return _sign_name(no), "landmark"  # 'NO EXIT': a sign worth showing, not an exit
        if claimed in ("exit", "entrance"):
            # the wall or door at the end of a corridor (a side glance: a door along it)
            return ("Doorway" if wn.kf.kind == "glance" else "Hallway end"), "hallway"
        if any(names_key(label, k) for k in SIGNED_TYPES):
            label = ""  # 'Restroom' / 'Elevator lobby' without the sign
    if wn.kf.kind == "turn_start" and ptype == "hallway":
        ptype = "intersection"
    if _norm_label(label) in _BARE_LABELS:
        # 'Room' is a type, not a name: named by its own plaque or sign ('SEMINAR' -> 'Seminar'); with no sign
        # read at all, no name (a floor sign alone leaves the type)
        own = [t for t in wayfinding_texts(a)
               if not directional(t) and not negated_key(t) and not _FLOOR_WORDS.search(t.upper())]
        rooms = sorted(room_ids(own))
        if ptype == "room" and rooms:
            label = f"Room {rooms[0]}"
        elif (named := next((t for t in own if not room_ids([t])), None)):
            label = _sign_name(named)
        elif not _has_sign(a):
            label = ""
    if label and (_has_sign(a) or ptype != "hallway"):
        return label, ptype
    defaults = {"start": "Walk start", "end": "Walk end", "turn_start": "Hallway junction",
                "glance": "Doorway", "dwell": "Hallway point"}
    return defaults.get(wn.kf.kind, label or "Hallway point"), ptype


def _sign_name(text: str) -> str:
    """A sign's text as a place name: title case, a negation OCR glued on split off ('NOEXIT' -> 'No Exit')."""
    return re.sub(r"[A-Za-z]+", lambda m: " ".join(_sign_words(m.group())), text.strip()).title()


_LABEL_PLAQUE = re.compile(rf"(?:ROOM|RM)\b\.?\s*(?:NO\b\.?\s*)?#?\s*({ROOM_ID})\b", re.I)
_PLAQUE_WORDS = {"ROOM", "ROOMS", "RM", "OFFICE", "LAB", "SUITE", "STE"}


def _complete_words(label: str, texts: list[str]) -> str:
    """A label word cut where OCR or the crop cut the sign ('Ker Space 107' of 'MAKER SPACE', 'Maker Sp'), completed
    from a longer reading of the same sign among the place's texts."""
    for m in re.finditer(r"[A-Za-z]+(?:\s+[A-Za-z]+)*", label):
        run = m.group()
        words = run.upper().split()
        if len(run.replace(" ", "")) < 4 or set(words) & _PLAQUE_WORDS:
            continue
        for t in texts:
            if directional(t) or not sign_line(t):
                continue
            up = re.sub(r"\s+", " ", t.upper())
            if (i := up.find(" ".join(words))) < 0:
                continue
            j = i + len(" ".join(words))
            a0, b0 = i, j
            while a0 > 0 and up[a0 - 1].isalpha():
                a0 -= 1
            while b0 < len(up) and up[b0].isalpha():
                b0 += 1
            if (i - a0) + (b0 - j) < 2:
                continue  # whole words already (or one stray letter)
            full = up[a0:b0]
            return label[:m.start()] + (full if run.isupper() else full.title()) + label[m.end():]
    return label


def tidy_label(label: str, texts: list[str], ptype: str) -> str:
    """One spelling per thing: 'RM 112' / 'ROOM 112' -> 'Room 112', cut words completed from the place's own sign
    texts, and no stray OCR token in a key place's name ('Exit - ROCW1' -> 'Exit')."""
    label = _LABEL_PLAQUE.sub(lambda m: f"Room {m.group(1).upper()}", label.strip())
    label = _complete_words(label, texts)
    if ptype in SIGNED_TYPES:
        # 'Exit Door' / 'Exit sign' / 'Exit' are one thing: spelled 'Exit' ('Emergency Exit Door' -> 'Emergency Exit')
        label = re.sub(r"(?i)\s+(?:door|doors|doorway|sign|signs)$", "", label) or label
        rooms = room_ids(texts)
        kept = [w for w in label.split()
                if not any(ch.isdigit() for ch in w) or re.sub(r"[^A-Z0-9]", "", w.upper()) in rooms]
        label = " ".join(kept).strip(" -,:") or label
    return label


# ---------------------------------------------------------------- loop closure
# Two places of one walk recorded less than this apart are never one place: the walker was moving between them
# (a loop back to a spot takes longer). Look-alike stretches of hallway (the same poster every few metres)
# match well on ORB and Gemma, so neither loop closure nor consolidation may join them.
SAME_WALK_GAP_SEC = 15.0


def _moments_apart(a: Node, b: Node) -> bool:
    """Both places come from the same walk, recorded less than SAME_WALK_GAP_SEC apart."""
    return bool(a.video_id and a.video_id == b.video_id and a.t is not None and b.t is not None
                and abs(a.t - b.t) < SAME_WALK_GAP_SEC)


@dataclass
class Match:
    node: Node
    auto: bool
    confidence: float
    reason: str
    signals: dict
    rel_yaw: float | None


def _gemma_backed(inliers: int, shared_plaque: bool) -> bool:
    """Whether a Gemma 'same place' may merge two spots automatically: only with real geometric support, i.e.
    strong ORB, or the same door-plaque room number on both plus weak ORB. Gemma alone is never enough: small
    models call look-alike spots 'identical' (the same poster on two walls, a 'LEVEL 2' sign along a whole
    corridor), and such a merge joins places metres apart. Without support the pair is left for review."""
    return inliers >= config.ORB_STRONG_INLIERS or (shared_plaque and inliers >= config.ORB_WEAK_INLIERS)


def _find_match(wn: WalkNode, bundle: MapBundle, frames: Path, cache, use_gemma: bool,
                exclude: set[str], walk_times: dict[str, float], log, w_floor: int | None = None,
                walk_levels: dict[str, int] | None = None, w_level: int | None = None) -> Match | None:
    """Best known place this walk node may be. `walk_levels` (node id -> floor within this walk) covers places
    this walk created itself: their Node.floor is not set until the walk is done, so the floor filter
    above cannot see them, and a walk that climbed stairs must not loop-close onto its own earlier floor."""
    w_tokens = place_tokens(wn.texts)
    w_plaques = room_ids(wn.texts)  # plaque numbers anywhere in this walk node (frame + absorbed frames)
    w_rooms = room_ids(wayfinding_texts(wn.analysis))  # this frame's own plaque numbers, as Node.sign_texts
    w_path = frames / wn.kf.frame_id
    cands = []
    for node in bundle.nodes:
        if node.id in exclude or not node.frame_id:
            continue
        if _rooms_conflict(w_rooms, _room_ids(node)):
            continue  # different room numbers: different places, however alike the pictures are
        if w_floor is not None and node.floor_source != "default" and node.floor != w_floor:
            continue  # similar corridors on different floors are never the same place
        if walk_levels is not None and node.id in walk_levels and walk_levels[node.id] != w_level:
            continue  # this walk's own place, on another floor of the same walk
        if node.id in walk_times and abs(wn.kf.t - walk_times[node.id]) < SAME_WALK_GAP_SEC:
            continue  # same walk, a few seconds ago: that's just the previous stretch of hallway
        text_hit = bool(w_tokens & place_tokens(node.texts))
        inliers, rel_yaw = orb_match(frames / node.frame_id, w_path)
        if text_hit or inliers >= config.ORB_WEAK_INLIERS:
            cands.append((text_hit, inliers, rel_yaw, node))
    if not cands:
        return None
    cands.sort(key=lambda c: (c[0], c[1]), reverse=True)

    best: Match | None = None
    for text_hit, inliers, rel_yaw, node in cands[:3]:
        signals = {"text": text_hit, "orb_inliers": inliers}
        shared_plaque = bool(w_plaques & room_ids(node.texts))
        verdict = None
        if use_gemma:
            try:
                verdict = gemma.same_place(frames / node.frame_id, w_path, cache)
                signals["gemma"] = verdict.model_dump()
            except gemma.GemmaUnavailable as exc:
                if not isinstance(exc, gemma.NotCached):
                    log(f"Gemma same-place check unavailable: {exc}")
                use_gemma = gemma.session_active()  # in a job session further calls only read the cache
        # Different door-plaque numbers read by OCR are never the same place, however strong the geometry
        # (identical walls and doors match hundreds of features). Without a verdict, ask OCR directly.
        if gemma.ocr_conflict(verdict if verdict is not None
                              else gemma.plaque_conflict(frames / node.frame_id, w_path)):
            continue
        overwhelming = inliers >= 3 * config.ORB_STRONG_INLIERS  # geometry alone is conclusive
        # A Gemma "different place" never vetoes a strong geometric match: small models (gemma3:4b) call the
        # same plaque seen 25 px apart "different" and flip with prompt wording, so geometry decides there.
        gemma_binding = verdict is not None and (verdict.same_place or inliers < config.ORB_STRONG_INLIERS)
        if gemma_binding and not overwhelming:
            if not verdict.same_place and verdict.confidence >= 0.7:
                continue
            conf = verdict.confidence if verdict.same_place else 0.3
            # Gemma alone is not proof (similar corridors look alike): require geometric agreement.
            auto = (verdict.same_place and verdict.confidence >= config.AUTO_MERGE_CONFIDENCE
                    and _gemma_backed(inliers, shared_plaque))
            if verdict.same_place and not auto:
                signals["note"] = "Gemma alone: no strong geometry or shared room number"
            reason = f"Gemma: {verdict.reason}"
        else:
            conf = min(1.0, 0.5 * text_hit + inliers / (2 * config.ORB_STRONG_INLIERS))
            if verdict is not None and not verdict.same_place:
                signals["note"] = "geometry overrode Gemma"
            auto = (text_hit and inliers >= config.ORB_WEAK_INLIERS) or inliers >= 1.5 * config.ORB_STRONG_INLIERS
            reason = ("same room number; " if text_hit else "") + f"{inliers} matching image features"
        m = Match(node, auto, round(conf, 2), reason, signals, rel_yaw)
        if best is None or (m.auto, m.confidence) > (best.auto, best.confidence):
            best = m
    return best


# ---------------------------------------------------------------- heading drift model
def _fit_correction(abs_c: list[tuple[float, float]], rel_c: list[tuple[float, float, float]]) -> tuple[float, float]:
    """Least squares for corrected = raw - (c0 + k t).

    abs_c: (t, value)  ->  c0 + k t = value
    rel_c: (t_new, t_old, d) -> k (t_new - t_old) = d
    Values are angle differences, already wrapped to [-180, 180).
    """
    rows, rhs = [], []
    for t, v in abs_c:
        rows.append([1.0, t]); rhs.append(v)
    for tn, to, d in rel_c:
        rows.append([0.0, tn - to]); rhs.append(d)
    rows.append([0.0, 60.0]); rhs.append(0.0)  # weak prior: no drift
    if not abs_c:
        rows.append([1.0, 0.0]); rhs.append(0.0)  # unknown absolute offset: keep raw frame
    sol, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    return float(sol[0]), float(sol[1])


# ---------------------------------------------------------------- integrate one walk
def integrate_walk(bundle: MapBundle, video_id: str, keyframes: list[Keyframe], track: dict,
                   frames: Path, cache, use_gemma: bool, log=print, start_floor: int | None = None) -> dict:
    # Stair climbs measured from camera motion, where Gemma's stairs hints back them up (floors.mark_climbs).
    keyframes = mark_climbs(keyframes, detect_climbs(track), log=log)
    wnodes = walk_nodes(keyframes)
    if not wnodes:
        return {"nodes": 0, "merged": 0}
    first_walk = not bundle.nodes
    kf_by_t = sorted(keyframes, key=lambda k: k.t)
    states = {k.frame_id: _stairs_state(k.analysis) for k in kf_by_t}

    # --- pass 1: identity decisions (merge or new) ---
    node_counter = _next_id("n", [n.id for n in bundle.nodes])
    walk_times: dict[str, float] = {}
    abs_c: list[tuple[float, float]] = [(0.0, 0.0)] if first_walk else []
    rel_c: list[tuple[float, float, float]] = []
    pending_suggestions: list[tuple[Match, WalkNode]] = []
    # Floors: start floor and room numbers first; still-unknown floors inherit from the first known place
    # this walk passes (or default to 0 for the very first walk).
    floors = assign_floors(wnodes, start_floor, keyframes=keyframes, log=log)
    has_rooms = any(room_floor(wn.texts) is not None for wn in wnodes)
    known_source = "user" if start_floor is not None else ("rooms" if has_rooms else None)
    inherit: int | None = None
    # While this walk's floor is unknown, a look-alike place on a known floor (identical elevator lobbies on
    # every floor) is not trusted alone: it votes for a floor offset and waits. Two agreeing votes (different
    # places on the same floor) set the walk's floor; a room-number/sign match sets it at once.
    votes: dict[int, set[str]] = {}  # floor offset -> known places that voted for it
    tentative: list[tuple[Match, WalkNode, int]] = []  # held matches (match, walk node, offset)
    created: list[tuple[Node, WalkNode]] = []
    merged = 0
    recent: list[str] = []
    # Floor of each place this walk created, within the walk: the assigned floor when known (assign_floors
    # gives a floor to every node or to none), else the stair count. Evidence may overrule the stair count
    # (Gemma saw a flight, rooms say same floor), so the assigned floor wins when there is one.
    walk_levels: dict[str, int] = {}

    for i, wn in enumerate(wnodes):
        w_floor = floors[i] if floors[i] is not None else (wn.rel_floor + inherit if inherit is not None else None)
        w_level = floors[i] if floors[i] is not None else wn.rel_floor
        match = _find_match(wn, bundle, frames, cache, use_gemma, set(recent[-2:]), walk_times, log, w_floor,
                            walk_levels, w_level)
        if (match and match.auto and w_floor is None and match.node.floor_source != "default"
                and not match.signals.get("text")):
            off = match.node.floor - wn.rel_floor
            votes.setdefault(off, set()).add(match.node.id)
            if len(votes[off]) >= 2 and all(len(votes[off]) > len(v) for o, v in votes.items() if o != off):
                inherit = off  # second agreeing place: now the floors are known and equal, merge below
                log(f"Walk floor from {len(votes[off])} matching places on floor {match.node.floor}")
            else:
                match.auto = False  # held: a suggestion unless a second place confirms the floor
                match.signals["note"] = "walk floor unknown: one look-alike place does not decide it"
                tentative.append((match, wn, off))
        if match and match.auto:
            node = match.node
            wn.node_id = node.id
            merged += 1
            node.texts = list(dict.fromkeys(node.texts + wn.texts))
            if wn.vertical and not node.vertical and node.type not in ROOM_LIKE:
                node.vertical = wn.vertical  # this walk climbed here: the known place is a stair landing
            if node.soft and not wn.kf.soft:
                node.frame_id, node.soft = wn.kf.frame_id, False
                node.frame_heading = None  # re-set below once headings are corrected
            log(f"Loop closure: walk t={wn.kf.t:.1f}s is '{node.label}' ({match.reason})")
            if floors[i] is None and inherit is None and node.floor_source != "default":
                inherit = node.floor - wn.rel_floor
            # Same view seen twice => headings must agree (up to the measured relative yaw).
            if match.rel_yaw is not None:
                observed = wn.kf.heading
                if node.id in walk_times:  # loop inside this walk: constrains drift
                    expected = _raw_heading_of(node, created) + match.rel_yaw
                    rel_c.append((wn.kf.t, walk_times[node.id], wrap180(observed - expected)))
                elif node.frame_heading is not None:  # known building node: constrains offset
                    abs_c.append((wn.kf.t, wrap180(observed - (node.frame_heading + match.rel_yaw))))
        else:
            label, ptype = _label(wn)
            node = Node(
                id=f"n{node_counter:03d}", label=label, type=ptype, frame_id=wn.kf.frame_id,
                texts=wn.texts, sign_texts=wayfinding_texts(wn.analysis), description=wn.analysis.description,
                soft=wn.kf.soft, vertical=wn.vertical,
                video_id=video_id, t=wn.kf.t,
                verified=bool(wn.analysis.room_numbers) or _key_place(wn.analysis)
                or (ptype == "room" and _has_sign(wn.analysis)),
            )
            node_counter += 1
            bundle.nodes.append(node)
            created.append((node, wn))
            wn.node_id = node.id
            walk_times[node.id] = wn.kf.t
            walk_levels[node.id] = w_level
            if match and not any(m is match for m, *_ in tentative):
                pending_suggestions.append((match, wn))
        recent.append(wn.node_id)

    # Held matches: merged after all if a later match settled the walk on their floor, else left for review.
    promote: list[tuple[Match, WalkNode]] = []
    for match, wn, off in tentative:
        if inherit is None:
            pending_suggestions.append((match, wn))
        elif off == inherit:
            promote.append((match, wn))
            node = match.node
            if match.rel_yaw is not None and node.frame_heading is not None:
                abs_c.append((wn.kf.t, wrap180(wn.kf.heading - (node.frame_heading + match.rel_yaw))))
        else:
            log(f"Not merging walk t={wn.kf.t:.1f}s with '{match.node.label}': that place is on floor "
                f"{match.node.floor}, this walk is on floor {wn.rel_floor + inherit}")

    # --- drift / alignment ---
    c0, k = _fit_correction(abs_c, rel_c)
    def corr(t: float) -> float:
        return c0 + k * t
    if not first_walk and not abs_c:
        log("No geometric anchor to the existing map for this walk; its headings use their own frame")
    if abs_c or rel_c:
        log(f"Heading correction: offset {c0:+.1f} deg, drift {k * 60:+.1f} deg/min")
    sources = []
    for i, wn in enumerate(wnodes):
        if floors[i] is not None:
            sources.append(known_source or "rooms")
        elif inherit is not None:
            floors[i], _ = wn.rel_floor + inherit, sources.append("linked")
        else:  # nothing known yet: a guess that later linking may correct
            floors[i], _ = wn.rel_floor + config.DEFAULT_FLOOR, sources.append("default")
    floor_of = {id(wn): f for wn, f in zip(wnodes, floors)}
    source_of = {id(wn): src for wn, src in zip(wnodes, sources)}
    for node, wn in created:
        node.frame_heading = round(wrap360(wn.kf.heading - corr(wn.kf.t)), 1)
        node.floor = floor_of[id(wn)]
        node.floor_source = source_of[id(wn)]
    for node in bundle.nodes:
        if node.frame_heading is None and node.frame_id:
            wn = next((w for w in wnodes if w.kf.frame_id == node.frame_id), None)
            if wn:
                node.frame_heading = round(wrap360(wn.kf.heading - corr(wn.kf.t)), 1)

    # --- suggestions for review ---
    for match, wn in pending_suggestions:
        bundle.suggested_merges.append(SuggestedMerge(
            id=f"m{_next_id('m', [s.id for s in bundle.suggested_merges]):03d}",
            keep=match.node.id, remove=wn.node_id, confidence=match.confidence,
            reason=match.reason, signals=match.signals,
        ))

    # --- pass 2: edges ---
    edge_counter = _next_id("e", [e.id for e in bundle.edges])
    new_edges = 0
    for a, b in zip(wnodes, wnodes[1:]):
        if a.node_id == b.node_id:
            continue
        seg = [kf for kf in kf_by_t if a.kf.t < kf.t < b.kf.t]
        lo, hi = a.t_depart, b.t_arrive
        if hi - lo < 0.4:
            lo = hi = (a.kf.t + b.kf.t) / 2
        h_out = wrap360(median_heading(track, lo, min(lo + 1.0, hi)) - corr(lo))
        h_in = wrap360(median_heading(track, max(hi - 1.0, lo), hi) - corr(hi))
        up = sum(states[k.frame_id] == "up" for k in seg)
        down = sum(states[k.frame_id] == "down" for k in seg)
        vertical = "stairs_up" if up > down else "stairs_down" if down > up else "none"
        fa, fb = floor_of[id(a)], floor_of[id(b)]
        inferred = False
        if vertical == "none" and fb != fa:
            vertical = "stairs_up" if fb > fa else "stairs_down"
            inferred = b.rel_floor == a.rel_floor  # room numbers say new floor, but no stairs were seen
            if inferred:
                log(f"Floor {fa} -> {fb} inferred from room numbers between t={a.kf.t:.0f}s and t={b.kf.t:.0f}s "
                    "(no stairs seen: check stairs vs elevator in Map)")
        elif vertical != "none" and fb != fa and (vertical == "stairs_up") != (fb > fa):
            # assign_floors already let room numbers/floor signs overrule the stair count: follow the floors,
            # or the route would say "take the stairs up to floor 1" from floor 2.
            log(f"Stairs between t={a.kf.t:.0f}s and t={b.kf.t:.0f}s looked {vertical[7:]}, but floors go "
                f"{fa} -> {fb}: using the floor direction")
            vertical = "stairs_up" if fb > fa else "stairs_down"
        elif vertical != "none" and fb == fa:
            # Steps that stay on one floor (split level, a few steps, or Gemma misread a flight). Kept as stairs:
            # the walker does climb, and routing only says "to floor N" when the floor changes.
            log(f"Stairs between t={a.kf.t:.0f}s and t={b.kf.t:.0f}s stay on floor {fa} (room numbers/floor "
                "signs): kept as steps on one floor, no floor change")
        frames_seq = [a.depart_frame or a.kf.frame_id] + [k.frame_id for k in seg
                                                          if k.kind not in ("glance",) and k.frame_id != a.depart_frame]
        frames_seq.append(b.kf.frame_id)
        frames_seq = list(dict.fromkeys(frames_seq))

        length = _walk_length(track, a.kf.t, b.kf.t)
        existing = next((e for e in bundle.edges if {e.from_node, e.to_node} == {a.node_id, b.node_id}), None)
        if existing:
            if existing.length_m is None or existing.source != "walk":
                existing.length_m = length
            else:  # walked twice: average the two measurements
                existing.length_m = round((existing.length_m + length) / 2, 1)
            if existing.from_node == b.node_id and existing.frames_rev is None:
                existing.frames_rev = frames_seq  # walked the other way: now we have imagery both ways
            continue
        bundle.edges.append(Edge(
            id=f"e{edge_counter:03d}", from_node=a.node_id, to_node=b.node_id,
            heading_out=h_out, heading_in=h_in, heading_out_raw=round(h_out, 1), heading_in_raw=round(h_in, 1),
            vertical=vertical, frames_fwd=frames_seq, source="walk", video_id=video_id,
            inferred_floor_change=inferred, length_m=length,
        ))
        edge_counter += 1
        new_edges += 1

    promoted = 0
    for match, wn in promote:  # held matches the walk's floor now confirms
        mine = next((n for n in bundle.nodes if n.id == wn.node_id), None)
        if mine is None or not any(n.id == match.node.id for n in bundle.nodes) or mine.id == match.node.id:
            continue
        if match.node.type in ROOM_LIKE:
            mine.vertical = None  # a room is never a stair landing (same rule as a direct match)
        log(f"Loop closure: walk t={wn.kf.t:.1f}s is '{match.node.label}' ({match.reason}; floor confirmed)")
        merge_nodes(bundle, match.node.id, mine.id, log=log)
        wn.node_id = match.node.id
        merged += 1
        promoted += 1

    info = next((v for v in bundle.videos if v.id == video_id), None)
    if info is not None:
        info.anchored = first_walk or bool(abs_c)
    finalize(bundle)
    return {"nodes": len(created) - promoted, "merged": merged, "edges": new_edges,
            "floor_offset": floors[0] - wnodes[0].rel_floor,
            "floors": sorted(set(floors)),
            "suggested": len(pending_suggestions), "drift_deg_per_min": round(k * 60, 1)}


def _walk_length(track: dict, t0: float, t1: float) -> float:
    """Metres walked between two moments: time spent moving forward x walking speed. Standing still,
    looking around or opening doors adds nothing."""
    t = np.array(track["t"])
    fwd = np.array(track.get("fwd") or [config.FWD_WALKING + 1] * len(t))
    m = (t >= min(t0, t1)) & (t <= max(t0, t1))
    if not m.any():
        return round(abs(t1 - t0) * config.WALK_SPEED_MPS, 1)
    moving = float((fwd[m] > config.FWD_WALKING * 0.6).sum()) / track["fps"]
    return round(max(1.0, moving * config.WALK_SPEED_MPS), 1)


def _raw_heading_of(node: Node, created: list[tuple[Node, WalkNode]]) -> float:
    for n, wn in created:
        if n.id == node.id:
            return wn.kf.heading
    return node.frame_heading or 0.0


# ---------------------------------------------------------------- whole-map operations
def dominant_axis(edges: list[Edge]) -> float:
    angles = [math.radians(h) for e in edges for h in (e.heading_out_raw, e.heading_in_raw) if h is not None]
    if not angles:
        return 0.0
    s = sum(math.sin(4 * a) for a in angles); c = sum(math.cos(4 * a) for a in angles)
    return round((math.degrees(math.atan2(s, c)) / 4) % 90, 3) % 90  # -0.0001 must not become 89.999


def snap(h: float, axis: float) -> float:
    rel = wrap180(h - axis)
    nearest90 = round(rel / 90) * 90
    target = nearest90 if abs(rel - nearest90) <= config.SNAP_TOLERANCE_DEG else round(rel / 45) * 45
    return round(wrap360(axis + target), 1)


def finalize(bundle: MapBundle) -> None:
    axis = dominant_axis(bundle.edges)
    bundle.dominant_axis_deg = round(axis, 1)
    for e in bundle.edges:
        if e.source == "manual":
            continue
        if e.heading_out_raw is not None:
            e.heading_out = snap(e.heading_out_raw, axis)
        if e.heading_in_raw is not None:
            e.heading_in = snap(e.heading_in_raw, axis)
    _dedupe_labels(bundle)
    solve_positions(bundle)


def solve_positions(bundle: MapBundle) -> None:
    """Metric x/y for every place, from all connections at once (least squares).

    Each connection says "b is length_m away from a along its heading"; stairs/elevators say "same x/y,
    different floor". Solving everything together averages out errors around loops instead of
    accumulating them. Separate pieces are placed side by side. Axes follow the main hallway axis.
    """
    idx = {n.id: i for i, n in enumerate(bundle.nodes)}
    if not idx:
        return
    axis = bundle.dominant_axis_deg
    rows: list[tuple[int, int, float, float, float]] = []  # (a, b, dx, dy, weight)
    for e in bundle.edges:
        if e.from_node not in idx or e.to_node not in idx:
            continue
        if e.vertical in ("stairs_up", "stairs_down", "elevator"):
            # Walked or linked, a staircase/elevator is one spot on the plan: the floor above sits on top of
            # the floor below. Its length_m is climbing distance, not a horizontal walk, so weight it high.
            rows.append((idx[e.from_node], idx[e.to_node], 0.0, 0.0, 5.0))
            continue
        length = e.length_m or config.DEFAULT_EDGE_M
        # average leaving/arriving directions (vector mean handles 350/10 deg)
        ha, hb = math.radians(e.heading_out - axis), math.radians(e.heading_in - axis)
        vx, vy = math.sin(ha) + math.sin(hb), math.cos(ha) + math.cos(hb)
        norm = math.hypot(vx, vy) or 1.0
        rows.append((idx[e.from_node], idx[e.to_node], length * vx / norm, length * vy / norm,
                     1.0 if e.length_m else 0.4))
    # connected pieces, each anchored at its first place
    parent = list(range(len(idx)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b, *_ in rows:
        parent[find(a)] = find(b)
    pieces: dict[int, list[int]] = {}
    for i in range(len(idx)):
        pieces.setdefault(find(i), []).append(i)
    n = len(idx)
    A = np.zeros((len(rows) + len(pieces), n))
    bx, by = np.zeros(A.shape[0]), np.zeros(A.shape[0])
    for r, (a, b, dx, dy, w) in enumerate(rows):
        A[r, a], A[r, b] = -w, w
        bx[r], by[r] = dx * w, dy * w
    for k, members in enumerate(pieces.values()):
        A[len(rows) + k, members[0]] = 10.0
    xs, *_ = np.linalg.lstsq(A, bx, rcond=None)
    ys, *_ = np.linalg.lstsq(A, by, rcond=None)
    offset = 0.0
    for members in sorted(pieces.values(), key=len, reverse=True):
        mx = min(xs[i] for i in members)
        for i in members:
            xs[i] += offset - mx
        offset = max(xs[i] for i in members) + 15.0
    for node in bundle.nodes:
        i = idx[node.id]
        node.x, node.y = round(float(xs[i]), 1), round(float(ys[i]), 1)


def _dedupe_labels(bundle: MapBundle) -> None:
    seen: dict[str, int] = {}
    for n in bundle.nodes:
        base = re.sub(r" \(\d+\)$", "", n.label)
        seen[base] = seen.get(base, 0) + 1
        n.label = base if seen[base] == 1 else f"{base} ({seen[base]})"


def merge_nodes(bundle: MapBundle, keep_id: str, remove_id: str, log=print) -> None:
    """Make two places one. Raises ValueError (before changing anything) when both floors are known and
    differ: that would join two floors' corridors on the flat. A guessed ('default') floor takes the known
    one, together with the rest of its walk's guessed places. Raises KeyError (also before changing anything)
    for an id that is not a place on the map."""
    by_id = {n.id: n for n in bundle.nodes}
    if (missing := [i for i in (keep_id, remove_id) if i not in by_id]):
        raise KeyError(f"no place {', '.join(repr(i) for i in missing)} on the map")
    keep, remove = by_id[keep_id], by_id[remove_id]
    if (why := _merge_floor_conflict(keep, remove)):
        raise ValueError(why)
    if (keep.floor_source == "default") != (remove.floor_source == "default"):
        guessed, sure = (keep, remove) if keep.floor_source == "default" else (remove, keep)
        if guessed.video_id:
            _adopt_floor(bundle, guessed, sure, log)
        if guessed.floor_source == "default":  # a place no walk owns (added by hand)
            guessed.floor, guessed.floor_source = sure.floor, "linked"
    keep.texts = list(dict.fromkeys(keep.texts + remove.texts))
    keep.vertical = keep.vertical or remove.vertical
    if keep.soft and not remove.soft and remove.frame_id:
        keep.frame_id, keep.frame_heading, keep.soft = remove.frame_id, remove.frame_heading, False
    for e in bundle.edges:
        if e.from_node == remove_id:
            e.from_node = keep_id
        if e.to_node == remove_id:
            e.to_node = keep_id
    kept: list[Edge] = []
    for e in bundle.edges:
        if e.from_node == e.to_node:
            continue
        dup = next((k for k in kept if {k.from_node, k.to_node} == {e.from_node, e.to_node}), None)
        if dup is None:
            kept.append(e)
        elif dup.from_node == e.to_node and dup.frames_rev is None:
            dup.frames_rev = e.frames_fwd
        elif dup.from_node == e.from_node and e.frames_rev and dup.frames_rev is None:
            dup.frames_rev = e.frames_rev
    bundle.edges = kept
    bundle.nodes = [n for n in bundle.nodes if n.id != remove_id]
    bundle.suggested_merges = [
        s for s in bundle.suggested_merges
        if not ({s.keep, s.remove} == {keep_id, remove_id} or remove_id in (s.keep, s.remove) and keep_id in (s.keep, s.remove))
    ]
    for s in bundle.suggested_merges:
        s.keep = keep_id if s.keep == remove_id else s.keep
        s.remove = keep_id if s.remove == remove_id else s.remove
    bundle.suggested_merges = [s for s in bundle.suggested_merges if s.keep != s.remove]
    finalize(bundle)


# ---------------------------------------------------------------- place descriptors
STOP_WORDS = {"WITH", "THE", "AND", "FROM", "NEAR", "NEXT", "BESIDE", "INTO", "THAT", "THIS", "WALL", "AREA",
              "LARGE", "SMALL", "WHITE", "GRAY", "GREY", "SIGN", "VIEW", "SPACE"} | GENERIC_WORDS


@dataclass
class PlaceDescriptor:
    node: Node
    words: set[str]  # distinctive sign words from its own photo
    letters: str  # all sign letters, for OCR fragments
    desc_words: set[str]  # Gemma's landmark phrase
    context: set[str]  # words from neighbouring places (what surrounds it)
    hist: object = None  # colour/texture fingerprint of its photo


def _content_words(text: str) -> set[str]:
    return {w for w in re.split(r"[^A-Z]+", text.upper()) if len(w) >= 4 and w not in STOP_WORDS}


@lru_cache(maxsize=2048)
def _photo_hist(path: str):
    img = cv2.imread(path)
    if img is None:
        return None
    small = cv2.resize(img, (160, int(160 * img.shape[0] / img.shape[1])))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def build_descriptors(bundle: MapBundle, frames: Path) -> dict[str, PlaceDescriptor]:
    nbrs: dict[str, set[str]] = {}
    for e in bundle.edges:
        nbrs.setdefault(e.from_node, set()).add(e.to_node)
        nbrs.setdefault(e.to_node, set()).add(e.from_node)
    by_id = {n.id: n for n in bundle.nodes}
    out = {}
    for n in bundle.nodes:
        ctx: set[str] = set()
        for m in nbrs.get(n.id, ()):
            if m in by_id:
                ctx |= sign_words(by_id[m].sign_texts) | _content_words(by_id[m].label)
        out[n.id] = PlaceDescriptor(
            node=n, words=sign_words(n.sign_texts), letters=_letters(n.sign_texts),
            desc_words=_content_words(n.description), context=ctx,
            hist=_photo_hist(str(frames / n.frame_id)) if n.frame_id else None,
        )
    return out


def word_weights(descs: dict[str, PlaceDescriptor]) -> dict[str, float]:
    """Rarity weights: a word seen at many places (posters, wayfinding arrows, 'THE BASEMENT')
    identifies none of them."""
    n = max(len(descs), 1)
    df: dict[str, int] = {}
    for d in descs.values():
        for w in d.words:
            df[w] = df.get(w, 0) + 1
    return {w: (0.0 if c >= max(3, 0.15 * n) else math.log(n / c) / math.log(n + 1)) for w, c in df.items()}


def _word_match(w: str, others: set[str], other_letters: str) -> bool:
    if w in others or (len(w) >= 4 and w in other_letters):
        return True
    return any(SequenceMatcher(None, w, v).ratio() >= 0.8 for v in others)


def _weighted_overlap(wa: set[str], wb: set[str], la: str, lb: str, weights: dict[str, float]) -> float:
    small, big, big_letters = (wa, wb, lb) if len(wa) <= len(wb) else (wb, wa, la)
    total = sum(weights.get(w, 1.0) for w in small)
    if total <= 1e-6:
        return 0.0
    hit = sum(weights.get(w, 1.0) for w in small if _word_match(w, big, big_letters))
    return hit / total


def place_similarity(a: PlaceDescriptor, b: PlaceDescriptor, frames: Path, weights: dict[str, float]) -> dict:
    """Combine independent descriptors; each one 0..1. Returns score, per-descriptor values and how
    many descriptors independently agree."""
    name = _weighted_overlap(a.words, b.words, a.letters, b.letters, weights) if a.words and b.words else 0.0
    inliers = 0
    if a.node.frame_id and b.node.frame_id:
        inliers, _ = orb_match(frames / a.node.frame_id, frames / b.node.frame_id)
    hist = float(cv2.compareHist(a.hist, b.hist, cv2.HISTCMP_CORREL)) if a.hist is not None and b.hist is not None else 0.0
    visual = max(min(inliers / (1.5 * config.ORB_STRONG_INLIERS), 1.0), max(0.0, (hist - 0.5) * 2) * 0.6)
    desc = (len(a.desc_words & b.desc_words) / max(1, min(len(a.desc_words), len(b.desc_words)))
            if a.desc_words and b.desc_words else 0.0)
    context = _weighted_overlap(a.context, b.context, "", "", weights) if a.context and b.context else 0.0
    agree = {"name": name >= 0.6, "visual": inliers >= config.ORB_WEAK_INLIERS or hist >= 0.85,
             "description": desc >= 0.5, "surroundings": context >= 0.4}
    score = 0.4 * name + 0.3 * visual + 0.1 * desc + 0.2 * context
    return {"score": round(score, 2), "name": round(name, 2), "visual": round(visual, 2), "orb": inliers,
            "description": round(desc, 2), "surroundings": round(context, 2),
            "agree": [k for k, v in agree.items() if v]}


# ---------------------------------------------------------------- map-wide consolidation
def _room_ids(node: Node) -> set[str]:
    """Room numbers on this place's own photo (merged places accumulate other photos' numbers); directional
    signs ('ROOMS 120-130 ->') do not count."""
    return room_ids(node.sign_texts or node.texts)


def _corridor_view(node: Node) -> bool:
    """A hallway spot or junction (not a door, stairwell or exit): any room number on its photo is a plaque in
    view, not the place where the walker stood. A place whose own label names one of its room numbers
    ('Room 208', 'RM 118') is that room, whatever its type says: hand-edited maps leave room nodes at the
    default type."""
    if node.type not in ("hallway", "intersection") or node.vertical:
        return False
    return not (_label_room(node.label) & _room_ids(node))


_LABEL_ROOM = re.compile(rf"\s*(?:{PLAQUE_PREFIX}\b\.?)?[\s.:#-]*(?:NO\b\.?[\s.:#-]*)?({ROOM_ID})\b"
                         rf"|\s*{PLAQUE_PREFIX}({ROOM_ID})\b")


def _label_room(label: str) -> set[str]:
    """The room a label is named after: 'Room 208', 'RM-118 door', '208 Lab'. A label that merely mentions a
    plaque further in ('Hallway with ROOM 101') names a corridor, not that room."""
    m = _LABEL_ROOM.match((label or "").upper())
    return {m.group(1) or m.group(2)} if m else set()


def _rooms_conflict(ra: set[str], rb: set[str]) -> bool:
    """Both places state room numbers and none is shared: different rooms, whatever the pictures say."""
    return bool(ra and rb and not ra & rb)


def _directions_at(bundle: MapBundle, node_id: str, video_id: str | None, include: bool) -> list[float]:
    """Travel directions leaving `node_id` along edges recorded by `video_id` (include) or by others."""
    out = []
    for e in bundle.edges:
        if (e.video_id == video_id) != include or e.heading_out_raw is None:
            continue
        if e.from_node == node_id:
            out.append(e.heading_out_raw)
        elif e.to_node == node_id:
            out.append((e.heading_in_raw + 180) % 360)
    return out


def _align_video(bundle: MapBundle, video_id: str, log) -> bool:
    """Rotate an unanchored walk by the multiple of 90 deg that best lines its corridors up with the
    anchored map at the places they share (buildings are mostly orthogonal)."""
    anchored = {v.id for v in bundle.videos if v.anchored}
    shared = {n for e in bundle.edges if e.video_id == video_id for n in (e.from_node, e.to_node)} & \
             {n for e in bundle.edges if e.video_id in anchored for n in (e.from_node, e.to_node)}
    if not shared:
        return False
    best, best_score = 0, float("inf")
    for rot in (0, 90, 180, 270):
        score = 0.0
        for n in shared:
            mine = _directions_at(bundle, n, video_id, True)
            theirs = _directions_at(bundle, n, video_id, False)
            for d in mine:
                score += min((abs(wrap180(d + rot - o)) for o in theirs), default=45)
        if score < best_score:
            best, best_score = rot, score
    for e in bundle.edges:
        if e.video_id == video_id:
            e.heading_out_raw = round(wrap360((e.heading_out_raw or 0) + best), 1)
            e.heading_in_raw = round(wrap360((e.heading_in_raw or 0) + best), 1)
    for n in bundle.nodes:
        if n.video_id == video_id and n.frame_heading is not None:
            n.frame_heading = round(wrap360(n.frame_heading + best), 1)
    next(v for v in bundle.videos if v.id == video_id).anchored = True
    log(f"Aligned walk {video_id} to the building frame (rotated {best} deg at {len(shared)} shared place(s))")
    return True


def consolidate(bundle: MapBundle, frames: Path, cache, use_gemma: bool, log=print) -> dict:
    """One map from n videos: merge places seen in different walks, independent of upload order.

    1. identity: the same OCR-verified room number on the same floor is the same place (even when the
       two walks faced opposite directions and the photos look nothing alike);
    2. appearance: strong ORB geometry, or Gemma 'same place' backed by geometry (_gemma_backed), between
       walks on the same floor;
    3. heading alignment for walks that joined the map without a geometric anchor.
    Floors are never mixed: similar corridors on different floors stay separate.

    A walk whose floor is only a guess ('default') is placed by vote, after each pass has seen every pair
    (so the result does not depend on node order): a room number or named sign it shares with a known
    place counts 2, a look-alike (appearance only) counts 1. The floor with weight >= 2 and more than any
    other wins; the walk moves there and only the matches on that floor merge. A tie or a lone look-alike
    moves nothing: those matches become suggestions for review.
    """
    merged, suggested = 0, 0
    existing_pairs = {frozenset((s.keep, s.remove)) for s in bundle.suggested_merges}
    deferred: list[tuple] = []  # merges that would place a guessed walk: (a, b, why, conf, signals, video, delta, w)
    merged_into: dict[str, str] = {}

    def alive(nid):
        return any(n.id == nid for n in bundle.nodes)

    def suggest(keep_id: str, remove_id: str, confidence: float, reason: str, signals: dict) -> None:
        nonlocal suggested
        if frozenset((keep_id, remove_id)) in existing_pairs:
            return
        bundle.suggested_merges.append(SuggestedMerge(
            id=f"m{_next_id('m', [x.id for x in bundle.suggested_merges]):03d}", keep=keep_id, remove=remove_id,
            confidence=confidence, reason=reason, signals=signals,
        ))
        existing_pairs.add(frozenset((keep_id, remove_id)))
        suggested += 1

    def do_merge(a: Node, b: Node, why: str) -> None:
        nonlocal merged
        keep, drop = (a, b) if _floor_rank(a) <= _floor_rank(b) else (b, a)
        log(f"{why}: merging '{drop.label}' into '{keep.label}' (floor {keep.floor})")
        merge_nodes(bundle, keep.id, drop.id, log=log)
        merged_into[drop.id] = keep.id
        merged += 1

    def merge(a: Node, b: Node, why: str, confidence: float, signals: dict, strong: bool) -> None:
        # Last gate for every automatic merge: different room numbers (on the places' own signs, or read by
        # OCR on their photos) are different places, whatever matched.
        conflict = _rooms_conflict(_room_ids(a), _room_ids(b)) or (
            bool(a.frame_id and b.frame_id)
            and gemma.ocr_conflict(gemma.plaque_conflict(frames / a.frame_id, frames / b.frame_id)))
        if conflict:
            log(f"Not merging '{b.label}' into '{a.label}' ({why}): different room numbers")
            return
        guessed = [n for n in (a, b) if n.floor_source == "default"]
        if len(guessed) == 1 and guessed[0].video_id:  # would place a guessed walk: vote, decide later
            g = guessed[0]
            sure = b if g is a else a
            deferred.append((sure.id, g.id, why, confidence, signals, g.video_id, sure.floor - g.floor,
                             2.0 if strong else 1.0))
            return
        do_merge(a, b, why)

    def current(nid: str) -> Node | None:
        while nid in merged_into:
            nid = merged_into[nid]
        return next((n for n in bundle.nodes if n.id == nid), None)

    def settle() -> None:
        """Place guessed walks by vote, then apply (or suggest) the merges that were waiting on it."""
        totals: dict[str, dict[int, float]] = {}
        for *_, vid, delta, w in deferred:
            totals.setdefault(vid, {}).setdefault(delta, 0.0)
            totals[vid][delta] += w
        for vid in sorted(totals):
            best = max(totals[vid].values())
            winners = [d for d, w in totals[vid].items() if w == best]
            if best >= 2.0 and len(winners) == 1:
                _shift_guessed_walk(bundle, vid, winners[0], f"{best:g} vote(s) from matching places", log)
            else:
                log(f"Walk {vid}: floor evidence from matching places is "
                    f"{'too weak' if best < 2.0 else 'split'} ({totals[vid]}); left for review")
        for sure_id, g_id, why, confidence, signals, *_ in deferred:
            sure, g = current(sure_id), current(g_id)
            if sure is None or g is None or sure.id == g.id:
                continue
            if _merge_floor_conflict(sure, g):
                log(f"Not merging '{g.label}' into '{sure.label}' ({why}): floor {g.floor} won the vote, "
                    f"not floor {sure.floor}")
            elif (g.floor_source == "default") != (sure.floor_source == "default"):
                suggest(sure.id, g.id, confidence, f"{why} (floor not settled)", signals)
            else:
                do_merge(sure, g, why)
        deferred.clear()

    # 1. identity from several independent descriptors (sign words weighted by rarity, appearance,
    #    Gemma's description, surroundings). A word alone never decides: posters and directional signs
    #    repeat names like THE BASEMENT all over a building.
    descs = build_descriptors(bundle, frames)
    weights = word_weights(descs)
    nodes = list(bundle.nodes)
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            if not (alive(a.id) and alive(b.id)) or not _floors_compatible(a, b):
                continue
            if (_is_vertical(a) is None) != (_is_vertical(b) is None):
                continue  # a staircase is never the same place as the room next to it
            ra, rb = _room_ids(a), _room_ids(b)
            if ra and rb:
                # a corridor spot that merely has the plaque in view (down the hall, from the corner) is not
                # that room: the same plaque seen from both ends of a hallway would make one place of both
                if ra & rb and len(ra) <= 2 and len(rb) <= 2 and not (_corridor_view(a) or _corridor_view(b)):
                    merge(a, b, f"Same room {'/'.join(sorted(ra & rb))}", 1.0,
                          {"rooms": sorted(ra & rb)}, strong=True)
                continue  # different room numbers: different places
            if _moments_apart(a, b):
                continue  # no shared plaque and the walk moved between them (as in loop closure): look-alikes
            sim = place_similarity(descs[a.id], descs[b.id], frames, weights)
            if sim["score"] < 0.35 or not sim["agree"]:
                continue
            why = f"{', '.join(sim['agree'])} agree (score {sim['score']})"
            if sim["score"] >= 0.5 and len(sim["agree"]) >= 2:
                merge(a, b, f"Same place: {why}", sim["score"], sim, strong="name" in sim["agree"])
                continue
            verdict = None
            if use_gemma and a.frame_id and b.frame_id:
                try:
                    verdict = gemma.same_place(frames / a.frame_id, frames / b.frame_id, cache)
                except gemma.GemmaUnavailable:
                    use_gemma = gemma.session_active()  # in a job session further calls only read the cache
            # Gemma's visual judgement is an independent descriptor of its own, but never enough without
            # geometry: a colour histogram, shared poster words or neighbours do not show it is one spot
            if (verdict and verdict.same_place and verdict.confidence >= 0.8 and len(sim["agree"]) >= 1
                    and _gemma_backed(sim["orb"], bool(_room_ids(a) & _room_ids(b)))):
                merge(a, b, f"Same place: {why}; Gemma: {verdict.reason[:120]}", sim["score"],
                      sim | {"gemma": verdict.model_dump()}, strong="name" in sim["agree"])
            elif frozenset((a.id, b.id)) not in existing_pairs and not (verdict and not verdict.same_place):
                bundle.suggested_merges.append(SuggestedMerge(
                    id=f"m{_next_id('m', [x.id for x in bundle.suggested_merges]):03d}", keep=a.id, remove=b.id,
                    confidence=sim["score"], reason=why + (f"; Gemma: {verdict.reason[:120]}" if verdict else ""),
                    signals=sim | ({"gemma": verdict.model_dump()} if verdict else {}),
                ))
                existing_pairs.add(frozenset((a.id, b.id)))
                suggested += 1
    settle()

    # 2. appearance across walks
    nodes = list(bundle.nodes)
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            if (not (alive(a.id) and alive(b.id)) or not _floors_compatible(a, b) or a.video_id == b.video_id
                    or not a.frame_id or not b.frame_id):
                continue
            if _rooms_conflict(_room_ids(a), _room_ids(b)):
                continue  # different room numbers: different places (as in step 1), whatever the geometry
            inliers, _ = orb_match(frames / a.frame_id, frames / b.frame_id)
            if inliers < config.ORB_WEAK_INLIERS:
                continue
            verdict = None
            if use_gemma and inliers < 1.5 * config.ORB_STRONG_INLIERS:
                try:
                    verdict = gemma.same_place(frames / a.frame_id, frames / b.frame_id, cache)
                except gemma.GemmaUnavailable:
                    use_gemma = gemma.session_active()  # in a job session further calls only read the cache
            if gemma.ocr_conflict(verdict):
                continue  # different plaque numbers read by OCR: never merged, never suggested
            strong = inliers >= 1.5 * config.ORB_STRONG_INLIERS
            if strong or (verdict and verdict.same_place and verdict.confidence >= config.AUTO_MERGE_CONFIDENCE
                          and _gemma_backed(inliers, bool(_room_ids(a) & _room_ids(b)))):
                why = f"{inliers} matching features" + (f"; Gemma: {verdict.reason}" if verdict else "")
                merge(a, b, f"Same place across walks ({why})",
                      round(verdict.confidence if verdict else min(1.0, inliers / (2 * config.ORB_STRONG_INLIERS)), 2),
                      {"orb_inliers": inliers, **({"gemma": verdict.model_dump()} if verdict else {})}, strong=False)
            elif frozenset((a.id, b.id)) not in existing_pairs and not (
                    verdict and not verdict.same_place and verdict.confidence >= 0.7
                    and inliers < config.ORB_STRONG_INLIERS):
                # the model never vetoes a strong geometric match (different OCR plaque numbers do, above)
                agree = verdict is not None and verdict.same_place
                bundle.suggested_merges.append(SuggestedMerge(
                    id=f"m{_next_id('m', [s.id for s in bundle.suggested_merges]):03d}", keep=a.id, remove=b.id,
                    confidence=round(verdict.confidence if agree else inliers / (2 * config.ORB_STRONG_INLIERS), 2),
                    reason=(f"Gemma: {verdict.reason}" if agree else f"{inliers} matching image features"
                            + (f"; Gemma disagreed: {verdict.reason}" if verdict else "")),
                    signals={"orb_inliers": inliers, **({"gemma": verdict.model_dump()} if verdict else {})},
                ))
                existing_pairs.add(frozenset((a.id, b.id)))
                suggested += 1
    settle()

    # 3. align walks that joined without a geometric anchor
    aligned = 0
    for _ in range(len(bundle.videos)):
        progress = False
        for v in bundle.videos:
            if not v.anchored and v.status == "done" and _align_video(bundle, v.id, log):
                aligned += 1
                progress = True
        if not progress:
            break

    # 3b. one place seen twice where the photos did not match: an exit faced from two corridors, or read from down the
    #     corridor and then reached; the bottom (or top) of a staircase walked up and walked down. Kind and plan
    #     position are never enough alone: view directions, the walk, sign text or the photos back each pair.
    #     A sign only read from afar (Twin.far) is not merged: the spot it was read from is a real spot of the
    #     corridor (other places may lie between it and the exit), so it stays, as a corridor spot.
    sign_views = 0
    for tw in _twin_candidates(bundle, frames):
        a, b = current(tw.a.id), current(tw.b.id)
        if a is None or b is None or a.id == b.id:
            continue
        if tw.far is not None:
            far, near = (a, b) if tw.far.id == tw.a.id else (b, a)
            if far.id == tw.far.id and far.type in KEY_FAMILY and not far.vertical and not _room_ids(far):
                log(f"'{far.label}' only read the sign of '{near.label}' from {tw.d:.1f} m ({tw.why}): "
                    "a hallway spot, not a second exit")
                far.type, far.label = "hallway", "Hallway point"
                sign_views += 1
            continue
        merge(a, b, f"Same {tw.kind}: {tw.why} ({tw.d:.1f} m apart on the plan)", 0.8,
              {"distance_m": round(tw.d, 1), "kind": tw.kind, "evidence": tw.why}, strong=False)
    settle()

    # 4. connect floors: the same stairwell / elevator bank seen on adjacent floors
    linked = _link_floors(bundle, frames, cache, use_gemma, log)

    finalize(bundle)
    g_components = _components(bundle)
    floors = sorted({n.floor for n in bundle.nodes})
    log(f"Consolidated: {merged} merged, {suggested} suggested, {aligned} walks aligned, {linked} floor links"
        + (f", {sign_views} sign(s) read from afar made hallway spots" if sign_views else "") + "; "
        f"{len(bundle.nodes)} places on floors {floors} in {g_components} connected piece(s)")
    return {"merged": merged, "suggested": suggested, "aligned": aligned, "floor_links": linked,
            "components": g_components}


TWIN_VIEW_M = 4.0  # an exit / stair landing seen twice this close on the plan may be one place (if nothing closer)
TWIN_VIEW_DEG = 45.0  # ... seen from directions at least this far apart (else the photos would have matched)
TWIN_TIGHT_M = 2.5  # stair landings with only their plan positions to go on (no photo, no sign): this close, both ends
APPROACH_M = 5.0  # a sign read from this far, then reached, is one place (eval/score.py's far-merge limit)
APPROACH_DEG = 30.0  # ... both photos (and the walk between them) facing within this of one direction
APPROACH_MERGE_M = 1.5  # ... closer than this the two photos were taken at one spot: one place. Farther, the far
#                         photo is where the sign was read from: a corridor spot of its own (see consolidate 3b)
VIEW_CONE_DEG = 40.0  # two photos can show one sign when a spot lies within this of both views (heading drift)
VIEW_RANGE_M = 6.0  # ... and this close to both walkers
VIEW_CROSS_DEG = 30.0  # views closer than this in direction (or than this short of opposite) are parallel: they
#                        never meet on their own (two doors side by side on one wall, faced from either end)
VIEW_BEHIND_M = 0.5  # where two views cross may lie this far behind a walker (one standing at the sign; drift)
ONE_SPOT_M = 2.5  # two photos facing one side (within 90 deg) from this close: taken at one spot (plan drift puts
#                   one door 2.1-2.3 m apart). Farther, parallel views need the photos or the sign to back them
TWIN_PHOTO_INLIERS = config.ORB_WEAK_INLIERS  # photo match (plain or zoomed) that backs a twin: support, never proof


def _travel_dirs(bundle: MapBundle, node_id: str) -> list[float]:
    """Directions the walks were going at a place (arriving and leaving along its corridor connections), in the
    frame solve_positions uses."""
    out = []
    for e in bundle.edges:
        if e.vertical != "none":
            continue
        if e.from_node == node_id and e.heading_out is not None:
            out.append(e.heading_out)
        if e.to_node == node_id and e.heading_in is not None:
            out.append(e.heading_in)
    return out


def _two_directions(bundle: MapBundle, a: Node, b: Node) -> bool:
    """The walks saw `a` and `b` from different directions: every way a walk went at one is at least TWIN_VIEW_DEG
    from every way a walk went at the other. Two walks going the same way saw the same view: if the photos did
    not match (earlier steps), only _approach evidence makes them one place."""
    da, db = _travel_dirs(bundle, a.id), _travel_dirs(bundle, b.id)
    return bool(da and db) and all(abs(wrap180(x - y)) >= TWIN_VIEW_DEG for x in da for y in db)


def _plan_dir(heading: float, axis: float) -> tuple[float, float]:
    """Unit vector on the plan (solve_positions' axes) of a building-frame heading."""
    r = math.radians(heading - axis)
    return math.sin(r), math.cos(r)


def _off_view(n: Node, x: float, y: float, axis: float) -> float:
    """Degrees between the way photo `n` looks and the plan point (x, y); 0 where the walker stood."""
    dx, dy = x - (n.x or 0.0), y - (n.y or 0.0)
    if math.hypot(dx, dy) < 1e-6:
        return 0.0
    return abs(wrap180(math.degrees(math.atan2(dx, dy)) - (n.frame_heading - axis)))


def _views_meet(a: Node, b: Node, axis: float) -> bool | None:
    """Whether the photos of `a` and `b` can show one sign: a spot within VIEW_RANGE_M of both walkers, within
    VIEW_CONE_DEG of both views (or where the other walker stood). None when a view direction is unknown.
    Exits at the two ends of a short corridor, each faced by its walk, look away from each other: False."""
    if a.frame_heading is None or b.frame_heading is None:
        return None
    ax, ay, bx, by = a.x or 0.0, a.y or 0.0, b.x or 0.0, b.y or 0.0
    step = 0.25
    for off in range(-int(VIEW_CONE_DEG), int(VIEW_CONE_DEG) + 1, 5):
        dx, dy = _plan_dir(a.frame_heading + off, axis)
        for k in range(int(VIEW_RANGE_M / step) + 1):
            x, y = ax + dx * k * step, ay + dy * k * step
            to_b = math.hypot(x - bx, y - by)
            if to_b <= 0.75 or (to_b <= VIEW_RANGE_M and _off_view(b, x, y, axis) <= VIEW_CONE_DEG):
                return True
    return False


def _views_cross(a: Node, b: Node, axis: float) -> bool | None:
    """Whether the two view rays genuinely converge on one spot: at least VIEW_CROSS_DEG apart in direction (and
    not facing each other head on), crossing in front of both walkers (at most VIEW_BEHIND_M behind one) and within
    VIEW_RANGE_M of each. None when a view direction is unknown. Parallel views never cross: two doors on one wall
    each faced by a walk from either end of the corridor look the same way, and _views_meet's cone lets them meet."""
    if a.frame_heading is None or b.frame_heading is None:
        return None
    if not VIEW_CROSS_DEG <= abs(wrap180(a.frame_heading - b.frame_heading)) <= 180.0 - VIEW_CROSS_DEG:
        return False
    (ux, uy), (vx, vy) = _plan_dir(a.frame_heading, axis), _plan_dir(b.frame_heading, axis)
    wx, wy = (b.x or 0.0) - (a.x or 0.0), (b.y or 0.0) - (a.y or 0.0)
    den = ux * vy - uy * vx  # a + t u = b + s v
    t, s = (wx * vy - wy * vx) / den, (wx * uy - wy * ux) / den
    return -VIEW_BEHIND_M <= t <= VIEW_RANGE_M and -VIEW_BEHIND_M <= s <= VIEW_RANGE_M


def _one_spot(a: Node, b: Node, d: float) -> bool:
    """Two photos facing one side (within 90 deg of each other) taken within ONE_SPOT_M: one spot."""
    return d <= ONE_SPOT_M and a.frame_heading is not None and b.frame_heading is not None and \
        abs(wrap180(a.frame_heading - b.frame_heading)) <= 90.0


def _approach(a: Node, b: Node, axis: float) -> tuple[Node, Node] | None:
    """(far, near) when both photos face one way (within APPROACH_DEG) and the far one looks at the spot where the
    near one was taken: a sign read from down the corridor, then the same sign up close. None without view
    directions. Two doors glanced at side by side along a corridor look the same way, but not at each other."""
    if a.frame_heading is None or b.frame_heading is None:
        return None
    if abs(wrap180(a.frame_heading - b.frame_heading)) > APPROACH_DEG:
        return None
    for far, near in ((a, b), (b, a)):
        if _off_view(far, near.x or 0.0, near.y or 0.0, axis) <= APPROACH_DEG:
            return far, near
    return None


def _walked_toward(bundle: MapBundle, far: Node, near: Node) -> str | None:
    """The walk that went from `far` on to `near` the way far's photo looks (every step within APPROACH_DEG of it,
    at most three corridor connections and APPROACH_M + 1 m in all): it read the sign, then walked up to it.
    Its video id, else None."""
    if far.frame_heading is None:
        return None
    def along(e: Edge) -> bool:
        hs = [h for h in (e.heading_out_raw, e.heading_in_raw) if h is not None] or [e.heading_out, e.heading_in]
        return all(h is not None and abs(wrap180(h - far.frame_heading)) <= APPROACH_DEG for h in hs)
    walks: dict[str, list[Edge]] = {}
    for e in bundle.edges:
        if e.vertical == "none" and e.video_id and along(e):
            walks.setdefault(e.video_id, []).append(e)
    for vid in sorted(walks):
        frontier = [(far.id, 0.0)]
        for _ in range(3):
            nxt = []
            for nid, walked in frontier:
                for e in walks[vid]:
                    total = walked + (e.length_m or config.DEFAULT_EDGE_M)
                    if e.from_node != nid or total > APPROACH_M + 1.0:
                        continue
                    if e.to_node == near.id:
                        return vid
                    nxt.append((e.to_node, total))
            frontier = nxt
    return None


def _photo_support(frames: Path | None, a: Node, b: Node) -> int:
    """Best feature match between the two places' photos, plain either way or zoomed (zoomed_inliers)."""
    if frames is None or not a.frame_id or not b.frame_id:
        return 0
    pa, pb = frames / a.frame_id, frames / b.frame_id
    return max(orb_match(pa, pb)[0], orb_match(pb, pa)[0], zoomed_inliers(pa, pb))


# ---------------------------------------------------------------- what tells two signs of one kind apart
_IDENTITY_SKIP = GENERIC_WORDS | NOT_WAYFINDING | _KEY_WORDS | _NEGATION | {
    "AN", "THE", "SIGN", "ONLY", "WAY", "HERE", "THIS", "OF", "IN", "OUT", "AT", "STAIRWAY", "STAIRCASE"}
_GENDER = {"MEN": "M", "MENS": "M", "GENTLEMEN": "M", "GENTS": "M", "WOMEN": "W", "WOMENS": "W", "LADIES": "W"}
_FLOOR_SIGN = re.compile(r"\b(?:LEVEL|FLOOR|FLR|LVL)\b\.?\s*[A-Z0-9]*|\b\d+(?:ST|ND|RD|TH)\b")


def _sign_of_kind(text: str, kind: str) -> bool:
    """`text` is a sign of twin kind `kind`: an exit-family (restroom, elevator) sign, a 'NO <kind>' sign, stairs."""
    if kind == "stairs":
        return "STAIR" in text.upper()
    if kind.startswith("no "):
        return any(k == kind[3:] and neg is not None for _, k, neg in _key_hits(text)[1])
    return any(names_key(text, k) for k, fam in KEY_FAMILY.items() if fam == kind)


def _sign_identity(n: Node, kind: str) -> set[str]:
    """What tells this place's sign from another of its kind: the other words on it ('EXIT 2', 'NORTH EXIT',
    'STAIR B', men / women), without floor signs or OCR junk ('ROCW1'). Empty for a plain 'EXIT' or 'STAIRS'."""
    out: set[str] = set()
    for t in n.sign_texts or n.texts:
        if not _sign_of_kind(t, kind):
            continue
        for tok in re.findall(r"[A-Z0-9]+", _FLOOR_SIGN.sub(" ", t.upper())):
            tok = _ocr_letters(tok)
            for w in (_sign_words(tok) if tok.isalpha() else [tok]):
                if kind == "restroom" and w in _GENDER:
                    out.add(_GENDER[w])
                elif w not in _IDENTITY_SKIP and (w.isalpha() or w.isdigit() or len(w) <= 3):
                    out.add(w)
    return out


def _identity_relation(a: set[str], b: set[str]) -> str | None:
    """'same' / 'differ' when both signs say more than their kind ('EXIT 2' / 'EXIT 3'), else None."""
    if not a or not b:
        return None
    same = a & b or any(len(x) >= 4 and len(y) >= 4 and SequenceMatcher(None, x, y).ratio() >= 0.8
                        for x in a for y in b)
    return "same" if same else "differ"


def _no_sign_kind(n: Node) -> str | None:
    """'no exit' for a landmark named by a 'NO EXIT' plate ('no entrance' for 'DO NOT ENTER'), else None."""
    if n.type != "landmark" or _room_ids(n):
        return None
    kinds = {k for t in [*(n.sign_texts or n.texts), n.label] for _, k, neg in _key_hits(t)[1] if neg is not None}
    return f"no {kinds.pop()}" if len(kinds) == 1 else None


# ---------------------------------------------------------------- one place seen twice: the evidence
def _same_sign_place(bundle: MapBundle, a: Node, b: Node, d: float, sign: str | None, photo,
                     axis: float) -> tuple[str | None, Node | None]:
    """(why, far): why two exits (restrooms, elevators, NO EXIT plates) of one kind are one place, else None; `far`
    the photo that only read the sign from afar (Twin.far), if any. Kind and plan position never decide alone:
    - seen from two directions (_two_directions) within TWIN_VIEW_M, and (when both view directions are known):
      the view rays converge on one spot (_views_cross), or the photos were taken at one spot (_one_spot), or the
      same distinctive sign text, or matching photos (TWIN_PHOTO_INLIERS). Parallel views farther apart (two doors
      on one wall, each faced by a walk from either end of the corridor) are two places without the sign or the
      photos;
    - seen facing one way, one photo looking at where the other was taken (_approach), within APPROACH_M, and
      backed by the walk that read the sign and walked up to it (_walked_toward), the same distinctive sign text,
      or matching photos (TWIN_PHOTO_INLIERS)."""
    meet = _views_meet(a, b, axis)
    if meet is False:
        return None, None  # the photos look away from each other: two signs
    same = "; same sign" if sign == "same" else ""
    if d <= TWIN_VIEW_M and _two_directions(bundle, a, b):
        if (cross := _views_cross(a, b, axis)) is None:
            return "seen from two directions" + same, None  # no view direction to check against
        if cross:
            return "seen from two directions; the views meet" + same, None
        if _one_spot(a, b, d):
            return f"seen from two directions; the views meet at one spot ({d:.1f} m)" + same, None
        if sign == "same":
            return "seen from two directions; same sign", None
        if (n := photo(a, b)) >= TWIN_PHOTO_INLIERS:
            return f"seen from two directions; photos match ({n} features)", None
    pair = _approach(a, b, axis)
    if pair is None:
        return None, None
    reader, near = pair
    far = reader if d > APPROACH_MERGE_M else None  # read from afar: a corridor spot; from one spot: one place
    if (vid := _walked_toward(bundle, reader, near)):
        return f"read from afar, then walked up to (walk {vid})" + same, far
    if sign == "same":
        return "read from afar, then up close; same sign", far
    if (n := photo(a, b)) >= TWIN_PHOTO_INLIERS:
        return f"read from afar, then up close; photos match ({n} features)", far
    return None, None


def _same_flight(flights: dict, a: Node, b: Node, d: float, sign: str | None, photo) -> str | None:
    """Why two stair landings, each on a walk's flight between the same two floors, are one landing, else None.
    Both ends of the flights must line up (here and on the other floor):
    - walked up and walked down (opposite views: the photos never match): within TWIN_TIGHT_M at both ends, or
      TWIN_VIEW_M with the same distinctive stairs sign. Two stairwells 3 m apart stay two;
    - walked the same way twice: within TWIN_TIGHT_M at both ends, the flights heading the same way, and the
      photos at either end match (or the same stairs sign). Position alone is two stairwells side by side."""
    for way_a, floor_a, end_a, e_a in flights.get(a.id, ()):
        for way_b, floor_b, end_b, e_b in flights.get(b.id, ()):
            if floor_a != floor_b:
                continue
            d2 = math.hypot((end_a.x or 0.0) - (end_b.x or 0.0), (end_a.y or 0.0) - (end_b.y or 0.0))
            ends = f"both ends line up ({d:.1f} / {d2:.1f} m)"
            if way_a != way_b:
                limit = TWIN_VIEW_M if sign == "same" else TWIN_TIGHT_M
                if d <= limit and d2 <= limit:
                    return f"walked up and down; {ends}" + ("; same sign" if sign == "same" else "")
                continue
            if d > TWIN_TIGHT_M or d2 > TWIN_TIGHT_M or not all(
                    x is not None and y is not None and abs(wrap180(x - y)) < TWIN_VIEW_DEG
                    for x, y in ((e_a.heading_out, e_b.heading_out), (e_a.heading_in, e_b.heading_in))):
                continue
            if sign == "same":
                return f"climbed the same way twice; {ends}; same sign"
            if (n := max(photo(a, b), photo(end_a, end_b))) >= TWIN_PHOTO_INLIERS:
                return f"climbed the same way twice; {ends}; photos match ({n} features)"
    return None


@dataclass
class Twin:
    """Two map places that show one place (see _twin_candidates). `far`: the one of them where the sign was only read
    from afar (_approach, more than APPROACH_MERGE_M short of it): that spot is not the place, so consolidate makes
    it a corridor spot instead of merging it into the place, which would fold the corridor between them."""
    a: Node
    b: Node
    d: float
    kind: str
    why: str
    far: Node | None = None


def _twin_views(bundle: MapBundle, frames: Path | None = None) -> list[tuple[Node, Node, float, str]]:
    """(a, b, plan distance, kind) for places that are one place seen twice; see _twin_candidates."""
    return [(t.a, t.b, t.d, t.kind) for t in _twin_candidates(bundle, frames)]


def _twin_candidates(bundle: MapBundle, frames: Path | None = None) -> list[Twin]:
    """Pairs of places that are one place seen twice, whose photos did not match well enough for the earlier steps,
    with the evidence (Twin). Kinds: an exit family by its own sign (place_type: exit and entrance are one), a
    'NO EXIT' plate, a stair landing (where a walk went up or down stairs).

    Both in one connected piece of the map (positions elsewhere mean nothing), on one floor, not seconds apart in
    one walk, each the other's nearest place of that kind (two exits side by side stay two), and their sign texts
    not telling them apart ('EXIT 2' / 'EXIT 3', men / women). Then kind and plan position never decide alone:
    _same_sign_place and _same_flight name the evidence needed. `frames` (the photo directory) lets matching
    photos count; without it they never do. Positions come from solve_positions: call this after walks are
    aligned."""
    solve_positions(bundle)
    by_id = {n.id: n for n in bundle.nodes}
    parent = {n.id: n.id for n in bundle.nodes}
    axis = bundle.dominant_axis_deg

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    # landing -> (leave / arrive, the other floor, the landing at the other end, the stairs connection)
    flights: dict[str, list[tuple[str, int, Node, Edge]]] = {}
    for e in bundle.edges:
        if e.from_node in parent and e.to_node in parent:
            parent[find(e.from_node)] = find(e.to_node)
            if e.vertical in ("stairs_up", "stairs_down"):
                lo, hi = by_id[e.from_node], by_id[e.to_node]
                flights.setdefault(lo.id, []).append(("leave", hi.floor, hi, e))
                flights.setdefault(hi.id, []).append(("arrive", lo.floor, lo, e))

    def kind(n: Node) -> str | None:
        if n.type in KEY_FAMILY:
            return KEY_FAMILY[n.type]
        if _is_vertical(n) == "stairs" and n.id in flights:
            return "stairs"
        return _no_sign_kind(n)
    groups: dict[tuple, list[Node]] = {}
    for n in bundle.nodes:
        if (k := kind(n)) is not None:
            groups.setdefault((k, n.floor, find(n.id)), []).append(n)
    photos: dict[frozenset, int] = {}

    def photo(x: Node, y: Node) -> int:
        key = frozenset((x.id, y.id))
        if key not in photos:
            photos[key] = _photo_support(frames, x, y)
        return photos[key]

    def dist(a: Node, b: Node) -> float:
        return math.hypot((a.x or 0.0) - (b.x or 0.0), (a.y or 0.0) - (b.y or 0.0))
    out = []
    for (k, _, _), members in groups.items():
        def nearest(a: Node) -> Node | None:
            return min((b for b in members if b is not a), key=lambda b: dist(a, b), default=None)
        reach_m = TWIN_VIEW_M if k == "stairs" else APPROACH_M
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                d = dist(a, b)
                if _moments_apart(a, b) or d > reach_m or nearest(a) is not b or nearest(b) is not a:
                    continue
                sign = _identity_relation(_sign_identity(a, k), _sign_identity(b, k))
                if sign == "differ":
                    continue  # 'EXIT 2' / 'EXIT 3': two signs, however close
                if k == "stairs":
                    why, far = _same_flight(flights, a, b, d, sign, photo), None
                else:
                    why, far = _same_sign_place(bundle, a, b, d, sign, photo, axis)
                if why:
                    out.append(Twin(a, b, d, k, why, far))
    out += [Twin(landing, view, d, k, "stairs seen ahead, climbed there moments later")
            for landing, view, d, k in _stair_views(bundle, groups, find, flights)]
    return sorted(out, key=lambda t: t.d)


def _stair_views(bundle: MapBundle, groups: dict[tuple, list[Node]], find, reach: dict) -> list:
    """(landing, view) pairs: a place typed stairs where no walk climbed (the 'STAIRS' sign seen ahead down the
    corridor, a walk passing the stairwell) is the landing of the flight next to it - the nearest stairs-edge
    place on its floor, in the same map piece, within TWIN_VIEW_M, with no second landing that close (two
    stairwells side by side stay apart). The same walk may see the stairs and then climb them a few seconds on."""
    out = []
    for n in bundle.nodes:
        if n.id in reach or n.type != "stairs" or _is_vertical(n) != "stairs" or _room_ids(n):
            continue
        landings = sorted(((math.hypot((n.x or 0.0) - (m.x or 0.0), (n.y or 0.0) - (m.y or 0.0)), m)
                           for m in groups.get(("stairs", n.floor, find(n.id)), [])), key=lambda p: p[0])
        close = [p for p in landings if p[0] <= TWIN_VIEW_M]
        if len(close) == 1:
            out.append((close[0][1], n, close[0][0], "stairs"))
    return out


def _components(bundle: MapBundle) -> int:
    parent = {n.id: n.id for n in bundle.nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for e in bundle.edges:
        if e.from_node in parent and e.to_node in parent:
            parent[find(e.from_node)] = find(e.to_node)
    return len({find(x) for x in parent})


def remove_video(bundle: MapBundle, video_id: str) -> dict:
    """Drop one walk from the map. Places another walk also uses are kept (re-owned by that walk);
    returns the frame ids nothing references any more, so the caller can delete the files."""
    prefix = f"{video_id}_"
    bundle.edges = [e for e in bundle.edges if e.video_id != video_id]
    for e in bundle.edges:  # imagery this walk contributed to other walks' edges
        e.frames_fwd = [f for f in e.frames_fwd if not f.startswith(prefix)]
        if e.frames_rev is not None:
            e.frames_rev = [f for f in e.frames_rev if not f.startswith(prefix)] or None
    used_by_others: dict[str, str] = {}
    for e in bundle.edges:
        for nid in (e.from_node, e.to_node):
            used_by_others.setdefault(nid, e.video_id or "")
    kept, removed = [], 0
    for n in bundle.nodes:
        if n.video_id != video_id:
            kept.append(n)
        elif n.id in used_by_others:
            n.video_id = used_by_others[n.id] or None  # still part of another walk
            kept.append(n)
        else:
            removed += 1
    bundle.nodes = kept
    ids = {n.id for n in kept}
    bundle.suggested_merges = [s for s in bundle.suggested_merges if s.keep in ids and s.remove in ids]
    bundle.videos = [v for v in bundle.videos if v.id != video_id]
    referenced = {n.frame_id for n in kept} | {f for e in bundle.edges for f in e.frames_fwd + (e.frames_rev or [])}
    finalize(bundle)
    return {"removed_places": removed, "kept_shared": sum(1 for n in kept if n.id in used_by_others
                                                          and n.frame_id and n.frame_id.startswith(prefix)),
            "referenced_frames": referenced}


def flip_vertical(v: str) -> str:
    return FLIP_VERTICAL.get(v, "none")
