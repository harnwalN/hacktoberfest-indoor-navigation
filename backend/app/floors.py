"""Floors: absolute floor per walk node (stairs + room numbers), floor bookkeeping when places merge,
and stairs/elevator links between adjacent floors that no single walk climbed."""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from . import config, gemma
from .matching import _next_id, text_tokens
from .schemas import Edge, MapBundle, Node

if TYPE_CHECKING:  # annotation only; a runtime import would be circular
    from .mapper import WalkNode
    from .schemas import FrameAnalysis, Keyframe

_HYPHENATED = re.compile(r"[A-Z0-9]*\d\s*[-–]\s*\d[A-Z0-9]*")  # 2-105, 858-534-1234: not a plain room number


def room_floor(texts: list[str]) -> int | None:
    """Floor implied by room numbers, using the 'first digit(s) = floor' convention (121 -> 1, 1210 -> 12).

    Only safe readings count: basement rooms (B112) and hyphenated numbers (2-105) give no evidence, and a
    4-digit number only counts when its first two digits are a plausible floor (10..MAX_FLOOR), since
    4-digit numbers are usually phone extensions or codes.
    """
    if not config.ROOM_NUMBER_FLOORS:
        return None
    floors = []
    for tok in text_tokens([_HYPHENATED.sub(" ", t.upper()) for t in texts]):
        if (m := re.fullmatch(r"([A-Z]?)(\d{3})[A-Z]?", tok)):
            f = int(m.group(2)[0])
        elif (m := re.fullmatch(r"([A-Z]?)(\d{4})[A-Z]?", tok)) and m.group(2)[0] != "0":
            f = int(m.group(2)[:2])
        else:
            continue
        if m.group(1) == "B":
            continue  # basement: not the ground floor, and floors below 1 are not modelled
        if 1 <= f <= config.MAX_FLOOR:
            floors.append(f)
    return max(set(floors), key=floors.count) if floors else None


# ---------------------------------------------------------------- floor signs
_ORDINALS = {"FIRST": 1, "SECOND": 2, "THIRD": 3, "FOURTH": 4, "FIFTH": 5, "SIXTH": 6, "SEVENTH": 7,
             "EIGHTH": 8, "NINTH": 9}
_FLOOR_WORD = r"(?:FLOOR|FLR|FL|LEVEL|LVL)"
_SIGN_PATTERNS = [
    re.compile(rf"\b(\d{{1,2}})\s*(?:ST|ND|RD|TH)\.?[\s-]+{_FLOOR_WORD}\b"),  # 3RD FLOOR
    re.compile(rf"\b{_FLOOR_WORD}\.?\s*(\d{{1,2}})\b(?!\s*[-–]\s*\d)"),  # FLOOR 2, LEVEL 2 (not 'FLOOR 1-3')
    re.compile(rf"\b({'|'.join(_ORDINALS)})[\s-]+{_FLOOR_WORD}\b"),  # SECOND FLOOR
]
# Levels below or beside the numbered floors ('LOWER LEVEL 1', 'PARKING LEVEL 2') are not floor N.
_NOT_A_FLOOR = re.compile(r"\b(?:LOWER|SUB|BASEMENT|UNDERGROUND|PARKING|GARAGE|MEZZANINE)\b")
_CONFIRM = 2.0  # evidence weight that settles a floor: two room numbers, or one floor sign
_ROOM_WEIGHT = 1.0


def _floors_in(text: str) -> set[int]:
    s = text.upper()
    if _NOT_A_FLOOR.search(s):
        return set()
    out = {_ORDINALS[m.group(1)] if m.group(1) in _ORDINALS else int(m.group(1))
           for p in _SIGN_PATTERNS for m in p.finditer(s)}
    if (m := re.fullmatch(r"\s*L\s?(\d{1,2})\s*", s)):  # a lone 'L3' sign; inside other text it is a code
        out.add(int(m.group(1)))
    return out


def sign_floor(texts: list[str]) -> int | None:
    """Floor stated by sign text ('3RD FLOOR', 'FLOOR 2', 'LEVEL 2', a lone 'L3'). None if no text states
    a floor, or if texts name several floors (a directory or elevator panel)."""
    found: set[int] = set()
    for t in texts:
        found |= _floors_in(t)
    if len(found) != 1:
        return None
    f = found.pop()
    return f if 1 <= f <= config.MAX_FLOOR else None


def _hint_floor(hint: str) -> int | None:
    """Parse Gemma's free-text floor_hint ('2', '2nd', 'Level 2', 'second floor'); basements give None."""
    s = (hint or "").upper().strip()
    if (m := re.fullmatch(r"(\d{1,2})(?:ST|ND|RD|TH)?", s)):
        f = int(m.group(1))
    elif s in _ORDINALS:
        f = _ORDINALS[s]
    else:
        return sign_floor([s]) if s else None
    return f if 1 <= f <= config.MAX_FLOOR else None


def _frame_readings(a: FrameAnalysis | None) -> list[tuple[int, float]]:
    """(floor, weight) readings of one frame's floor signs.

    Strong (FLOOR_SIGN_WEIGHT): a floor in the OCR-grounded sign texts, or a floor_hint that an OCR line
    actually reads as that floor. Weak (like one room number): floor words anywhere in the OCR (may be a
    poster), or a floor_hint/sign that OCR could not check (no OCR on this machine). A grounded floor_hint
    with no OCR line stating it is dropped: gemma's grounding is a substring test ('2' passes on '125').
    """
    if a is None or not config.FLOOR_SIGNS:
        return []
    strong = config.FLOOR_SIGN_WEIGHT if a.grounded else _ROOM_WEIGHT
    out = []
    if (f := sign_floor(a.texts)) is not None:
        out.append((f, strong))
    elif (f := sign_floor(a.ocr)) is not None:
        out.append((f, _ROOM_WEIGHT))
    if (h := _hint_floor(a.floor_hint)) is not None:
        if not a.grounded:
            out.append((h, _ROOM_WEIGHT))
        elif any(_hint_floor(line) == h for line in a.ocr):
            out.append((h, strong))
    return out


def _combine(readings: list[tuple[int, float]]) -> tuple[int, float] | None:
    """One (offset, weight) per walk node: the strongest readings, if they agree."""
    if not readings:
        return None
    top = max(w for _, w in readings)
    offs = {o for o, w in readings if w == top}
    return (offs.pop(), top) if len(offs) == 1 else None


def _node_sign(wn: WalkNode) -> tuple[int, float] | None:
    readings = [(f - wn.rel_floor, w) for f, w in _frame_readings(wn.analysis)]
    if config.FLOOR_SIGNS and (f := sign_floor(wn.extra_texts)) is not None:  # absorbed neighbour frames
        readings.append((f - wn.rel_floor, config.FLOOR_SIGN_WEIGHT if wn.analysis.grounded else _ROOM_WEIGHT))
    return _combine(readings)


_STAIRS = {"on_stairs_up": "up", "on_stairs_down": "down"}


def _signs_from_keyframes(wnodes: list[WalkNode], keyframes: list[Keyframe]) -> list[tuple[int, float] | None]:
    """Floor signs from every keyframe, not only the ones that became walk nodes (a 'LEVEL 2' plaque in a
    plain corridor frame is not a node of its own). Each reading is converted to a floor offset with the
    frame's own stair count (same counting as walk_nodes) and attached to the nearest walk node."""
    per_node: list[list[tuple[int, float]]] = [[] for _ in wnodes]
    rel, prev = 0, "none"

    def gap(wn: WalkNode, t: float) -> float:
        return 0.0 if wn.t_arrive <= t <= wn.t_depart else min(abs(t - wn.t_arrive), abs(t - wn.t_depart))

    for k in sorted(keyframes, key=lambda k: k.t):
        state = _STAIRS.get(k.analysis.stairs if k.analysis else "none", "none")
        if prev != "none" and state == "none":
            rel += 1 if prev == "up" else -1
        prev = state
        if state != "none" or not wnodes:
            continue  # mid-flight it is unclear which floor a landing sign belongs to
        if not (readings := _frame_readings(k.analysis)):
            continue
        i = min(range(len(wnodes)), key=lambda i: (wnodes[i].rel_floor != rel, gap(wnodes[i], k.t)))
        per_node[i] += [(f - rel, w) for f, w in readings]
    return [_combine(r) for r in per_node]


def _confirmed(ev: list[tuple[int, int, float]], k: int) -> bool:
    """Evidence point k is believed if, among the previous point, itself and the next two, its offset has
    weight >= _CONFIRM and outweighs every other offset. So one misread room number (leading, trailing or
    in the middle) never sets a floor, while two agreeing numbers or one clear floor sign do."""
    support: dict[int, float] = {}
    for _, o, w in ev[max(k - 1, 0):k + 3]:
        support[o] = support.get(o, 0.0) + w
    off = ev[k][1]
    return support[off] >= _CONFIRM and all(support[off] > s for o, s in support.items() if o != off)


def assign_floors(wnodes: list[WalkNode], start_floor: int | None,
                  keyframes: list[Keyframe] | None = None) -> list[int | None]:
    """Absolute floor per walk node, or None when unknown.

    Relative floor changes come from detected stairs (rel_floor). The absolute offset comes from the
    uploader's start floor, room numbers and floor signs (Gemma's floor_hint, 'LEVEL 3' / '2ND FLOOR' text;
    read from all `keyframes` when given, else from the walk nodes). Evidence only counts once confirmed
    (see _confirmed). Without a start floor, the first confirmed floor also covers the nodes before it; a
    later confirmed change that stairs do not explain (an elevator ride, missed stairs) starts a new floor
    segment.
    """
    signs = _signs_from_keyframes(wnodes, keyframes) if keyframes is not None else [_node_sign(w) for w in wnodes]
    ev: list[tuple[int, int, float]] = []  # (walk node index, floor offset, weight)
    for i, (wn, sign) in enumerate(zip(wnodes, signs)):
        if sign is not None:
            ev.append((i, *sign))
        if (f := room_floor(wn.texts)) is not None:
            ev.append((i, f - wn.rel_floor, _ROOM_WEIGHT))
    changes: list[tuple[int, int]] = []
    cur = start_floor
    for k, (i, off, _) in enumerate(ev):
        if off != cur and _confirmed(ev, k):
            changes.append((i, off))
            cur = off
    o = start_floor if start_floor is not None else (changes[0][1] if changes else None)
    offsets: list[int | None] = []
    c = 0
    for i in range(len(wnodes)):
        while c < len(changes) and changes[c][0] <= i:
            o = changes[c][1]
            c += 1
        offsets.append(o)
    return [wn.rel_floor + o if o is not None else None for wn, o in zip(wnodes, offsets)]


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
    if delta and moved:  # keep the walk's recorded offset in step with its places (PATCH start floor uses it)
        for v in bundle.videos:
            if v.id == guessed.video_id:
                v.floor_offset += delta
    if delta:
        log(f"Walk {guessed.video_id} placed on floor {sure.floor} via '{sure.label}' ({moved} places moved)")


_STAIR_WORD = re.compile(r"\bSTAIR(S|WAYS?|WELLS?|CASES?)?\b")  # not UPSTAIRS / DOWNSTAIRS (a lounge, a cafe)
_ABOUT_A_SIGN = re.compile(r"\b(POSTERS?|SIGNS?|SIGNAGE|NOTICES?|BANNERS?|FLYERS?|ARROWS?|DIRECTORY|MAP)\b")


def _is_vertical(n: Node) -> str | None:
    """Stairs/elevator from what the place IS, never from text read nearby: posters like 'take the stairs,
    not the elevator' hang in ordinary hallways.

    In order: what the walk did here (`vertical`: climbed a flight / faced the stairs, even if Gemma called
    it a hallway), the place type, then the label, unless the label is about a sign or poster
    ('Take the stairs poster', 'Elevator sign')."""
    if n.vertical:
        return n.vertical
    if n.type in ("elevator", "stairs"):
        return n.type
    if n.type in ("room", "restroom", "exit"):
        return None
    label = n.label.upper()
    if _ABOUT_A_SIGN.search(label):
        return None
    if "ELEVATOR" in label:
        return "elevator"
    if _STAIR_WORD.search(label):
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


SURE_LINK = 0.7  # evidence (same sign, Gemma "same stairwell") this strong makes a verified link
VETO_LINK = 0.7  # Gemma "different stairwell" this sure blocks a link, including the only-staircase guess


def _is_guess(e: Edge) -> bool:
    """A link _link_floors added with no evidence (the only staircase on both floors)."""
    return e.source == "merge" and e.inferred_floor_change and e.vertical in ("stairs_up", "elevator")


def _reaches_floor(bundle: MapBundle, node_id: str, floor: int, floor_of: dict[str, int], ignore: set[str]) -> bool:
    """Does this node already have a stairs/elevator edge to `floor` (a walk climbed it, or an earlier link)?"""
    for e in bundle.edges:
        if e.vertical == "none" or e.id in ignore or node_id not in (e.from_node, e.to_node):
            continue
        if floor_of.get(e.to_node if e.from_node == node_id else e.from_node) == floor:
            return True
    return False


def _link_floors(bundle: MapBundle, frames: Path, cache, use_gemma: bool, log) -> int:  # noqa: C901
    """Add stairs/elevator connections between adjacent floors that no single walk climbed.

    A stair node that already has a vertical edge to the other floor is left alone. Links this function
    guessed earlier (no evidence, `inferred_floor_change`) are revisited each run, conservatively: kept as
    they are by default, confirmed when the same pair gains evidence, dropped when Gemma is now sure they
    are different stairwells, and replaced when a pair with real evidence claims one of their places.
    Walk, manual and evidence-backed links are never changed. Returns the number of links added."""
    linked = 0
    floor_of = {n.id: n.floor for n in bundle.nodes}
    by_kind: dict[tuple[str, int], list[Node]] = {}
    for n in bundle.nodes:
        if (kind := _is_vertical(n)):
            by_kind.setdefault((kind, n.floor), []).append(n)
    floors = sorted({f for _, f in by_kind})

    def drop(e: Edge, why: str) -> None:
        bundle.edges = [x for x in bundle.edges if x is not e]
        log(f"Floor link removed: {e.from_node} <-> {e.to_node} ({why})")

    for kind in ("stairs", "elevator"):
        for f in floors:
            lower, upper = by_kind.get((kind, f), []), by_kind.get((kind, f + 1), [])
            if not lower or not upper:
                continue
            lo_ids, up_ids = {a.id for a in lower}, {b.id for b in upper}
            guesses = {(e.from_node, e.to_node): e for e in bundle.edges
                       if _is_guess(e) and e.from_node in lo_ids and e.to_node in up_ids}
            revisit = {e.id for e in guesses.values()}
            done_lo = {a.id for a in lower if _reaches_floor(bundle, a.id, f + 1, floor_of, revisit)}
            done_up = {b.id for b in upper if _reaches_floor(bundle, b.id, f, floor_of, revisit)}
            pairs = []
            for a in lower:
                for b in upper:
                    if a.id in done_lo or b.id in done_up:
                        continue  # already reaches that floor: a walk climbed here, or a sure link exists
                    score, why = 0.0, ""
                    ids_a, ids_b = text_tokens(a.texts), text_tokens(b.texts)
                    if ids_a & ids_b:
                        score, why = 1.0, f"same {kind} sign {'/'.join(sorted(ids_a & ids_b))}"
                    elif use_gemma and a.frame_id and b.frame_id:
                        try:
                            v = gemma.same_stairwell(frames / a.frame_id, frames / b.frame_id, cache)
                        except gemma.GemmaUnavailable:
                            use_gemma, v = False, None
                        if v is not None and v.same_place:
                            score, why = v.confidence, f"Gemma: {v.reason}"
                        elif v is not None and v.confidence >= VETO_LINK:
                            if (old := guesses.pop((a.id, b.id), None)) is not None:
                                drop(old, f"Gemma: different {kind}: {v.reason}")
                            else:
                                log(f"No floor link '{a.label}' (floor {f}) <-> '{b.label}' (floor {f + 1}): "
                                    f"Gemma says different {kind} ({v.reason})")
                            continue
                    known = (a.floor_source != "default") + (b.floor_source != "default")
                    pairs.append((score, known, why, a, b))
            # Strongest evidence first; among equals, places whose floor is known rather than guessed.
            pairs.sort(key=lambda p: (-p[0], -p[1]))
            single = len(lower) == 1 and len(upper) == 1
            used: set[str] = set()
            for score, _, why, a, b in pairs:
                if a.id in used or b.id in used:
                    continue
                sure = score >= SURE_LINK
                if (old := guesses.pop((a.id, b.id), None)) is not None:
                    used |= {a.id, b.id}  # guessed before: keep it, and confirm it if there is evidence now
                    if sure:
                        old.verified, old.inferred_floor_change = True, False
                        log(f"Floor link confirmed: '{a.label}' (floor {f}) <-> '{b.label}' (floor {f + 1}) ({why})")
                    continue
                if not sure and not single:
                    continue  # ambiguous: several staircases and no evidence which one
                if sure:  # real evidence beats an earlier guess that used either of these places
                    for key in [k for k in guesses if a.id in k or b.id in k]:
                        drop(guesses.pop(key), f"replaced by '{a.label}' <-> '{b.label}': {why}")
                _add_vertical_edge(bundle, a, b, kind, not sure)
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
        vertical="stairs_up" if kind == "stairs" else "elevator", length_m=6.0,
        frames_fwd=[f for f in (lower.frame_id, upper.frame_id) if f],
        frames_rev=[f for f in (upper.frame_id, lower.frame_id) if f],
        source="merge", verified=not inferred, inferred_floor_change=inferred,
    ))


def set_walk_floor(bundle: MapBundle, video_id: str, floor: int, log=print) -> dict:
    """The uploader says which floor a processed walk starts on: move that walk's own places there.

    The shift is measured from the floor of the walk's earliest own place as it is now, not from the
    stored floor_offset, so it is right even after consolidation adopted the walk onto another floor.
    Places the walk shares with other walks belong to those walks and stay put. Floor links that no
    longer join a lower floor to the one above are dropped, then floors are linked again.
    """
    own = [n for n in bundle.nodes if n.video_id == video_id]
    first = min(own, key=lambda n: n.t if n.t is not None else float("inf"), default=None)  # ties: walk order
    delta = floor - first.floor if first is not None else 0
    for n in own:
        n.floor += delta
        n.floor_source = "user"
    for v in bundle.videos:
        if v.id == video_id:
            v.floor_offset = floor
    dropped = 0
    if delta:
        moved = {n.id for n in own}
        floor_of = {n.id: n.floor for n in bundle.nodes}
        kept = [e for e in bundle.edges
                if not (e.source == "merge" and e.vertical != "none" and {e.from_node, e.to_node} & moved
                        and floor_of.get(e.to_node) != floor_of.get(e.from_node, 0) + 1)]
        dropped = len(bundle.edges) - len(kept)
        bundle.edges = kept
        log(f"Walk {video_id} set to start on floor {floor} ({len(own)} places moved by {delta:+d}, "
            f"{dropped} stale floor links dropped)")
    linked = _link_floors(bundle, Path("."), None, False, log)  # no Gemma: this runs inside an API request
    return {"delta": delta, "moved": len(own) if delta else 0, "dropped_links": dropped, "floor_links": linked}
