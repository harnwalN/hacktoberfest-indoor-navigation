"""Turn one processed walk (keyframes + Gemma analyses + motion track) into graph nodes/edges and
merge it into the building map, closing loops when the walker revisits a known place.

Pipeline per walk:
  1. walk nodes:    keyframes that matter (turns, glances, dwells, signs, stairs boundaries, ends)
  2. loop closure:  match each walk node against known nodes by room-number text and ORB geometry,
                    verified by Gemma ("same physical spot?"); auto-merge or suggest for review
  3. drift fix:     every ORB-verified match yields a heading constraint; fit c(t) = c0 + k*t so the
                    walk's integrated heading lines up with the building frame (and itself, on loops)
  4. edges:         consecutive walk nodes -> edges with heading_out / heading_in and vertical
  5. finalize:      Manhattan snap of all edge headings to the building's dominant axis
"""
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from . import config, gemma
from .motion import focal_px, median_heading
from .schemas import (
    FLIP_VERTICAL, Edge, FrameAnalysis, Keyframe, MapBundle, Node, SuggestedMerge,
)

# Structural moments that always become nodes. Glances only do if Gemma finds a sign/landmark there.
NODE_KINDS = {"start", "end", "turn_start", "dwell"}
IMPORTANT_TYPES = {"intersection", "entrance", "stairs", "elevator", "exit", "restroom", "room", "landmark"}
KEY_TYPES = {"entrance", "stairs", "elevator", "exit", "restroom"}  # always worth a node + auto-verify
KIND_PRIORITY = {"turn_start": 0, "glance": 1, "start": 2, "end": 2, "dwell": 3}


def wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def wrap360(a: float) -> float:
    return a % 360.0


# ---------------------------------------------------------------- text + geometry matching
def text_tokens(texts: list[str]) -> set[str]:
    """Distinctive tokens only: things containing a digit (room numbers). 'EXIT' repeats on every floor."""
    out = set()
    for t in texts:
        for tok in re.split(r"[\s,;:/()]+", t.upper()):
            tok = re.sub(r"[^A-Z0-9]", "", tok)
            if len(tok) >= 2 and any(ch.isdigit() for ch in tok):
                out.add(tok)
    return out


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


@lru_cache(maxsize=2048)
def _orb_features(path: str):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None, (0, 0)
    h, w = img.shape
    s = 640 / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    kp, des = cv2.ORB_create(nfeatures=1500).detectAndCompute(img, None)
    return [k.pt for k in kp], des, img.shape


def orb_match(path_a: Path, path_b: Path) -> tuple[int, float | None]:
    """Return (homography inliers, yaw of B relative to A in degrees, if well-determined)."""
    pa, da, sa = _orb_features(str(path_a))
    pb, db, _ = _orb_features(str(path_b))
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
    inliers = int(mask.sum())
    h, w = sa
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
    extra_texts: list[str] = field(default_factory=list)

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


def _key_place(a: FrameAnalysis | None) -> bool:
    """Entrances/restrooms/exits/elevators need sign evidence when OCR grounding is on;
    a staircase is recognizable from the picture alone."""
    if a is None:
        return False
    if a.place_type == "stairs" or a.stairs in ("ahead_up", "ahead_down"):
        return True
    return a.place_type in KEY_TYPES and (_has_sign(a) or not a.grounded)


def _worthy(kf: Keyframe) -> bool:
    a = kf.analysis
    if kf.kind in NODE_KINDS:
        return True
    return bool(a and (a.room_numbers or _key_place(a) or (a.place_type == "room" and _has_sign(a))))


def walk_nodes(keyframes: list[Keyframe]) -> list[WalkNode]:
    kfs = sorted(keyframes, key=lambda k: k.t)
    states = [_stairs_state(k.analysis) for k in kfs]
    worthy = [_worthy(k) for k in kfs]
    # Stair boundaries: last frame before climbing and first frame after.
    for i in range(1, len(kfs)):
        if states[i] != states[i - 1]:
            if states[i - 1] == "none":
                worthy[i - 1] = True
            if states[i] == "none":
                worthy[i] = True

    nodes: list[WalkNode] = []
    for i, k in enumerate(kfs):
        if k.kind == "turn_end" and nodes and nodes[-1].kf.kind == "turn_start":
            nodes[-1].depart_frame = k.frame_id
        if not worthy[i] or states[i] != "none":
            continue  # never put a node in the middle of a staircase
        a = k.analysis or FrameAnalysis()
        wn = WalkNode(kf=k, t_arrive=k.t_arrive if k.t_arrive is not None else k.t,
                      t_depart=k.t_depart if k.t_depart is not None else k.t, analysis=a)
        prev = nodes[-1] if nodes else None
        if prev and not _crosses_stairs(kfs, states, prev, wn) and not _keep_apart(prev, wn) and (
                wn.t_arrive - prev.t_depart < config.NODE_MIN_GAP_SEC
                or (wn.t_arrive - prev.t_depart < config.SAME_PLACE_WINDOW_SEC and _same_spot(prev, wn))):
            _absorb(prev, wn)
            continue
        nodes.append(wn)

    # Relative floor per walk node: count completed flights.
    floor, j = 0, 0
    for i, k in enumerate(kfs):
        if i > 0 and states[i - 1] != "none" and states[i] == "none":
            floor += 1 if states[i - 1] == "up" else -1
        while j < len(nodes) and nodes[j].kf.t <= k.t:
            nodes[j].rel_floor = floor
            j += 1
    return nodes


def _norm_label(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _keep_apart(a: WalkNode, b: WalkNode) -> bool:
    """Keep the walk's start/end separate from an adjacent turn, so that turn stays routable."""
    kinds = {a.kf.kind, b.kf.kind}
    return bool(kinds & {"start", "end"}) and "turn_start" in kinds


def _same_spot(a: WalkNode, b: WalkNode) -> bool:
    """Consecutive frames describing the same place (e.g. three frames of the same entrance)."""
    if b.kf.kind == "turn_start" and a.kf.kind == "turn_start":
        return False  # two real junctions
    ra, rb = text_tokens(a.texts), text_tokens(b.texts)
    if ra and rb:
        return bool(ra & rb)  # room numbers decide
    if strong_name_match(wayfinding_texts(a.analysis), wayfinding_texts(b.analysis)):
        return True  # several frames of the same named place (e.g. MAKER SPACE)
    la, lb = _norm_label(a.analysis.suggested_label), _norm_label(b.analysis.suggested_label)
    if la and la == lb:
        return True
    return a.analysis.place_type == b.analysis.place_type and a.analysis.place_type in {"entrance", "stairs", "elevator", "exit", "restroom"}


def _crosses_stairs(kfs, states, a: WalkNode, b: WalkNode) -> bool:
    return any(s != "none" for k, s in zip(kfs, states) if a.kf.t < k.t < b.kf.t)


def _absorb(keep: WalkNode, other: WalkNode) -> None:
    """Merge two walk nodes that are too close in time into one."""
    def rank(w: WalkNode):
        return (0 if _has_sign(w.analysis) else 1, KIND_PRIORITY.get(w.kf.kind, 9))
    if KIND_PRIORITY.get(other.kf.kind, 9) < KIND_PRIORITY.get(keep.kf.kind, 9):
        keep.kf, keep.depart_frame = other.kf, other.depart_frame or keep.depart_frame
    if rank(other) < rank(keep):
        keep.extra_texts += wayfinding_texts(keep.analysis)
        keep.analysis = other.analysis
    else:
        keep.extra_texts += other.texts
    keep.t_arrive, keep.t_depart = min(keep.t_arrive, other.t_arrive), max(keep.t_depart, other.t_depart)


# ---------------------------------------------------------------- labels
def _label(wn: WalkNode) -> tuple[str, str]:
    a = wn.analysis
    ptype = a.place_type
    if wn.kf.kind == "turn_start" and ptype == "hallway":
        ptype = "intersection"
    if a.suggested_label and (_has_sign(a) or ptype != "hallway"):
        return a.suggested_label.strip(), ptype
    defaults = {"start": "Walk start", "end": "Walk end", "turn_start": "Hallway junction",
                "glance": "Doorway", "dwell": "Hallway point"}
    return defaults.get(wn.kf.kind, a.suggested_label or "Hallway point"), ptype


def _next_id(prefix: str, existing: list[str]) -> int:
    nums = [int(m.group(1)) for x in existing if (m := re.fullmatch(rf"{prefix}(\d+)", x))]
    return max(nums, default=0) + 1


# ---------------------------------------------------------------- floors
def room_floor(texts: list[str]) -> int | None:
    """Floor implied by room numbers, using the 'first digit(s) = floor' convention (121 -> 1, 1210 -> 12)."""
    if not config.ROOM_NUMBER_FLOORS:
        return None
    floors = []
    for tok in text_tokens(texts):
        m = re.fullmatch(r"[A-Z]?(\d{3,4})[A-Z]?", tok)
        if m:
            floors.append(int(m.group(1)[:-2]))
    return max(set(floors), key=floors.count) if floors else None


def assign_floors(wnodes: list[WalkNode], start_floor: int | None) -> list[int | None]:  # noqa: C901
    """Absolute floor per walk node, or None when unknown.

    Relative floor changes come from detected stairs (rel_floor). The absolute offset comes from the
    uploader's start floor and/or room-number evidence; a persistent change in room-number floors that
    stairs do not explain (e.g. an elevator ride, missed stairs) starts a new floor segment.
    """
    ev = [(i, f - wn.rel_floor) for i, wn in enumerate(wnodes) if (f := room_floor(wn.texts)) is not None]
    offsets: list[int | None] = []
    cur = start_floor if start_floor is not None else (ev[0][1] if ev else None)
    j = 0
    for i in range(len(wnodes)):
        while j < len(ev) and ev[j][0] == i:
            off = ev[j][1]
            confirmed = j + 1 >= len(ev) or ev[j + 1][1] == off  # one misread number is not a floor change
            if off != cur and confirmed and not (cur == start_floor and j == 0 and len(ev) == 1):
                cur = off
            j += 1
        offsets.append(cur)
    return [wn.rel_floor + o if o is not None else None for wn, o in zip(wnodes, offsets)]


# ---------------------------------------------------------------- loop closure
@dataclass
class Match:
    node: Node
    auto: bool
    confidence: float
    reason: str
    signals: dict
    rel_yaw: float | None


def _find_match(wn: WalkNode, bundle: MapBundle, frames: Path, cache, use_gemma: bool,
                exclude: set[str], walk_times: dict[str, float], log, w_floor: int | None = None) -> Match | None:
    w_tokens = text_tokens(wn.texts)
    w_path = frames / wn.kf.frame_id
    cands = []
    for node in bundle.nodes:
        if node.id in exclude or not node.frame_id:
            continue
        if w_floor is not None and node.floor_source != "default" and node.floor != w_floor:
            continue  # similar corridors on different floors are never the same place
        if node.id in walk_times and abs(wn.kf.t - walk_times[node.id]) < 15:
            continue  # same walk, a few seconds ago: that's just the previous stretch of hallway
        text_hit = bool(w_tokens & text_tokens(node.texts))
        inliers, rel_yaw = orb_match(frames / node.frame_id, w_path)
        if text_hit or inliers >= config.ORB_WEAK_INLIERS:
            cands.append((text_hit, inliers, rel_yaw, node))
    if not cands:
        return None
    cands.sort(key=lambda c: (c[0], c[1]), reverse=True)

    best: Match | None = None
    for text_hit, inliers, rel_yaw, node in cands[:3]:
        signals = {"text": text_hit, "orb_inliers": inliers}
        verdict = None
        if use_gemma:
            try:
                verdict = gemma.same_place(frames / node.frame_id, w_path, cache)
                signals["gemma"] = verdict.model_dump()
            except gemma.GemmaUnavailable as exc:
                log(f"Gemma same-place check unavailable: {exc}")
                use_gemma = False
        overwhelming = inliers >= 3 * config.ORB_STRONG_INLIERS  # geometry alone is conclusive
        if verdict is not None and not overwhelming:
            if not verdict.same_place and verdict.confidence >= 0.7:
                continue
            conf = verdict.confidence if verdict.same_place else 0.3
            # Gemma alone is not proof (similar corridors look alike): require geometric agreement.
            auto = (verdict.same_place and verdict.confidence >= config.AUTO_MERGE_CONFIDENCE
                    and inliers >= config.ORB_WEAK_INLIERS)
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
    floors = assign_floors(wnodes, start_floor)
    has_rooms = any(room_floor(wn.texts) is not None for wn in wnodes)
    known_source = "user" if start_floor is not None else ("rooms" if has_rooms else None)
    inherit: int | None = None
    created: list[tuple[Node, WalkNode]] = []
    merged = 0
    recent: list[str] = []

    for i, wn in enumerate(wnodes):
        w_floor = floors[i] if floors[i] is not None else (wn.rel_floor + inherit if inherit is not None else None)
        match = _find_match(wn, bundle, frames, cache, use_gemma, set(recent[-2:]), walk_times, log, w_floor)
        if match and match.auto:
            node = match.node
            wn.node_id = node.id
            merged += 1
            node.texts = list(dict.fromkeys(node.texts + wn.texts))
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
                soft=wn.kf.soft,
                video_id=video_id, t=wn.kf.t,
                verified=bool(wn.analysis.room_numbers) or _key_place(wn.analysis)
                or (ptype == "room" and _has_sign(wn.analysis)),
            )
            node_counter += 1
            bundle.nodes.append(node)
            created.append((node, wn))
            wn.node_id = node.id
            walk_times[node.id] = wn.kf.t
            if match:
                pending_suggestions.append((match, wn))
        recent.append(wn.node_id)

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
        frames_seq = [a.depart_frame or a.kf.frame_id] + [k.frame_id for k in seg
                                                          if k.kind not in ("glance",) and k.frame_id != a.depart_frame]
        frames_seq.append(b.kf.frame_id)
        frames_seq = list(dict.fromkeys(frames_seq))

        existing = next((e for e in bundle.edges if {e.from_node, e.to_node} == {a.node_id, b.node_id}), None)
        if existing:
            if existing.from_node == b.node_id and existing.frames_rev is None:
                existing.frames_rev = frames_seq  # walked the other way: now we have imagery both ways
            continue
        bundle.edges.append(Edge(
            id=f"e{edge_counter:03d}", from_node=a.node_id, to_node=b.node_id,
            heading_out=h_out, heading_in=h_in, heading_out_raw=round(h_out, 1), heading_in_raw=round(h_in, 1),
            vertical=vertical, frames_fwd=frames_seq, source="walk", video_id=video_id,
            inferred_floor_change=inferred,
        ))
        edge_counter += 1
        new_edges += 1

    info = next((v for v in bundle.videos if v.id == video_id), None)
    if info is not None:
        info.anchored = first_walk or bool(abs_c)
    finalize(bundle)
    return {"nodes": len(created), "merged": merged, "edges": new_edges, "floor_offset": floors[0] - wnodes[0].rel_floor,
            "floors": sorted(set(floors)),
            "suggested": len(pending_suggestions), "drift_deg_per_min": round(k * 60, 1)}


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
    return (math.degrees(math.atan2(s, c)) / 4) % 90


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


def _dedupe_labels(bundle: MapBundle) -> None:
    seen: dict[str, int] = {}
    for n in bundle.nodes:
        base = re.sub(r" \(\d+\)$", "", n.label)
        seen[base] = seen.get(base, 0) + 1
        n.label = base if seen[base] == 1 else f"{base} ({seen[base]})"


def merge_nodes(bundle: MapBundle, keep_id: str, remove_id: str) -> None:
    keep = next(n for n in bundle.nodes if n.id == keep_id)
    remove = next(n for n in bundle.nodes if n.id == remove_id)
    keep.texts = list(dict.fromkeys(keep.texts + remove.texts))
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


# ---------------------------------------------------------------- map-wide consolidation
def _room_ids(node: Node) -> set[str]:
    return {t for t in text_tokens(node.texts) if re.fullmatch(r"[A-Z]?\d{3,4}[A-Z]?", t)}


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
    2. appearance: strong ORB geometry, or ORB + Gemma 'same place', between walks on the same floor;
    3. heading alignment for walks that joined the map without a geometric anchor.
    Floors are never mixed: similar corridors on different floors stay separate.
    """
    merged, suggested = 0, 0
    existing_pairs = {frozenset((s.keep, s.remove)) for s in bundle.suggested_merges}

    def alive(nid):
        return any(n.id == nid for n in bundle.nodes)

    def merge(a: Node, b: Node, why: str) -> None:
        nonlocal merged
        keep, drop = (a, b) if _floor_rank(a) <= _floor_rank(b) else (b, a)
        _adopt_floor(bundle, drop, keep, log)
        log(f"{why}: merging '{drop.label}' into '{keep.label}' (floor {keep.floor})")
        merge_nodes(bundle, keep.id, drop.id)
        merged += 1

    # 1. identity: same room number, or the same named sign (MAKER SPACE / QUALCOMM INSTITUTE)
    nodes = list(bundle.nodes)
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            if not (alive(a.id) and alive(b.id)) or not _floors_compatible(a, b):
                continue
            ra, rb = _room_ids(a), _room_ids(b)
            if ra and rb:
                if ra & rb and len(ra) <= 2 and len(rb) <= 2:
                    merge(a, b, f"Same room {'/'.join(sorted(ra & rb))}")
                continue  # different room numbers: different places
            if strong_name_match(a.sign_texts, b.sign_texts) and a.video_id != b.video_id:
                merge(a, b, f"Same sign {'/'.join(sorted(shared_sign_words(a.sign_texts, b.sign_texts)))}")

    # 2. appearance across walks
    nodes = list(bundle.nodes)
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            if (not (alive(a.id) and alive(b.id)) or not _floors_compatible(a, b) or a.video_id == b.video_id
                    or not a.frame_id or not b.frame_id):
                continue
            inliers, _ = orb_match(frames / a.frame_id, frames / b.frame_id)
            if inliers < config.ORB_WEAK_INLIERS:
                continue
            verdict = None
            if use_gemma and inliers < 1.5 * config.ORB_STRONG_INLIERS:
                try:
                    verdict = gemma.same_place(frames / a.frame_id, frames / b.frame_id, cache)
                except gemma.GemmaUnavailable:
                    use_gemma = False
            strong = inliers >= 1.5 * config.ORB_STRONG_INLIERS
            if strong or (verdict and verdict.same_place and verdict.confidence >= config.AUTO_MERGE_CONFIDENCE):
                why = f"{inliers} matching features" + (f"; Gemma: {verdict.reason}" if verdict else "")
                merge(a, b, f"Same place across walks ({why})")
            elif frozenset((a.id, b.id)) not in existing_pairs and not (verdict and not verdict.same_place
                                                                        and verdict.confidence >= 0.7):
                bundle.suggested_merges.append(SuggestedMerge(
                    id=f"m{_next_id('m', [s.id for s in bundle.suggested_merges]):03d}", keep=a.id, remove=b.id,
                    confidence=round(verdict.confidence if verdict else inliers / (2 * config.ORB_STRONG_INLIERS), 2),
                    reason=(f"Gemma: {verdict.reason}" if verdict else f"{inliers} matching image features"),
                    signals={"orb_inliers": inliers, **({"gemma": verdict.model_dump()} if verdict else {})},
                ))
                existing_pairs.add(frozenset((a.id, b.id)))
                suggested += 1

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

    # 4. connect floors: the same stairwell / elevator bank seen on adjacent floors
    linked = _link_floors(bundle, frames, cache, use_gemma, log)

    finalize(bundle)
    g_components = _components(bundle)
    floors = sorted({n.floor for n in bundle.nodes})
    log(f"Consolidated: {merged} merged, {suggested} suggested, {aligned} walks aligned, {linked} floor links; "
        f"{len(bundle.nodes)} places on floors {floors} in {g_components} connected piece(s)")
    return {"merged": merged, "suggested": suggested, "aligned": aligned, "floor_links": linked,
            "components": g_components}


_RANK = {"user": 0, "rooms": 1, "linked": 2, "default": 3}


def _floor_rank(n: Node) -> int:
    return _RANK.get(n.floor_source, 3)


def _floors_compatible(a: Node, b: Node) -> bool:
    return a.floor == b.floor or "default" in (a.floor_source, b.floor_source)


def _adopt_floor(bundle: MapBundle, guessed: Node, sure: Node, log) -> None:
    """A walk whose floor was only a guess learns its floor from a place it shares with a known walk."""
    if guessed.floor_source != "default" or sure.floor_source == "default" or not guessed.video_id:
        return
    delta = sure.floor - guessed.floor
    moved = 0
    for n in bundle.nodes:
        if n.video_id == guessed.video_id and n.floor_source == "default":
            n.floor += delta
            n.floor_source = "linked"
            moved += 1
    if delta:
        log(f"Walk {guessed.video_id} placed on floor {sure.floor} via '{sure.label}' ({moved} places moved)")


def _is_vertical(n: Node) -> str | None:
    words = " ".join(n.texts).upper() + " " + n.label.upper()
    if n.type == "elevator" or "ELEVATOR" in words:
        return "elevator"
    if n.type == "stairs" or "STAIR" in words:
        return "stairs"
    return None


def _connected(bundle: MapBundle, a: str, b: str) -> bool:
    adj: dict[str, set[str]] = {}
    for e in bundle.edges:
        adj.setdefault(e.from_node, set()).add(e.to_node)
        adj.setdefault(e.to_node, set()).add(e.from_node)
    seen, stack = {a}, [a]
    while stack:
        u = stack.pop()
        if u == b:
            return True
        for v in adj.get(u, ()):
            if v not in seen:
                seen.add(v); stack.append(v)
    return False


def _link_floors(bundle: MapBundle, frames: Path, cache, use_gemma: bool, log) -> int:
    """Add stairs/elevator connections between adjacent floors that no single walk climbed."""
    linked = 0
    by_kind: dict[tuple[str, int], list[Node]] = {}
    for n in bundle.nodes:
        if (kind := _is_vertical(n)):
            by_kind.setdefault((kind, n.floor), []).append(n)
    floors = sorted({f for _, f in by_kind})
    for kind in ("stairs", "elevator"):
        for f in floors:
            lower, upper = by_kind.get((kind, f), []), by_kind.get((kind, f + 1), [])
            if not lower or not upper:
                continue
            pairs = []
            for a in lower:
                for b in upper:
                    if _connected(bundle, a.id, b.id) and any(
                            e for e in bundle.edges if e.vertical != "none" and {e.from_node, e.to_node} & {a.id, b.id}):
                        continue  # a walk already climbed between these
                    score, why = 0.0, ""
                    ids_a, ids_b = text_tokens(a.texts), text_tokens(b.texts)
                    if ids_a & ids_b:
                        score, why = 1.0, f"same {kind} sign {'/'.join(sorted(ids_a & ids_b))}"
                    elif use_gemma and a.frame_id and b.frame_id:
                        try:
                            v = gemma.same_stairwell(frames / a.frame_id, frames / b.frame_id, cache)
                            if v.same_place:
                                score, why = v.confidence, f"Gemma: {v.reason}"
                        except gemma.GemmaUnavailable:
                            use_gemma = False
                    pairs.append((score, why, a, b))
            pairs.sort(key=lambda p: -p[0])
            used: set[str] = set()
            for score, why, a, b in pairs:
                if a.id in used or b.id in used:
                    continue
                inferred = score < 0.7
                if inferred and not (len(lower) == 1 and len(upper) == 1):
                    continue  # ambiguous: several staircases and no evidence which one
                _add_vertical_edge(bundle, a, b, kind, inferred)
                used |= {a.id, b.id}
                linked += 1
                log(f"Floor link: '{a.label}' (floor {f}) <-> '{b.label}' (floor {f + 1}) by {kind} "
                    f"({why or 'only ' + kind + ' on both floors; check'})")
    return linked


def _add_vertical_edge(bundle: MapBundle, lower: Node, upper: Node, kind: str, inferred: bool) -> None:
    h = lower.frame_heading or 0.0
    bundle.edges.append(Edge(
        id=f"e{_next_id('e', [e.id for e in bundle.edges]):03d}", from_node=lower.id, to_node=upper.id,
        heading_out=h, heading_in=upper.frame_heading if upper.frame_heading is not None else h,
        heading_out_raw=h, heading_in_raw=upper.frame_heading if upper.frame_heading is not None else h,
        vertical="stairs_up" if kind == "stairs" else "elevator",
        frames_fwd=[f for f in (lower.frame_id, upper.frame_id) if f],
        frames_rev=[f for f in (upper.frame_id, lower.frame_id) if f],
        source="merge", verified=not inferred, inferred_floor_change=inferred,
    ))


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
