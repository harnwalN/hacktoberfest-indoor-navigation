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


def room_floor(texts: list[str]) -> int | None:
    """Floor implied by room numbers, using the 'first digit(s) = floor' convention (121 -> 1, 1210 -> 12)."""
    if not config.ROOM_NUMBER_FLOORS:
        return None
    floors = []
    for tok in text_tokens(texts):
        m = re.fullmatch(r"[A-Z]?(\d{3})[A-Z]?", tok)  # 4+ digits are usually phone/codes, not rooms
        if m and 1 <= int(m.group(1)[0]) <= config.MAX_FLOOR:
            floors.append(int(m.group(1)[0]))
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


def _is_vertical(n: Node) -> str | None:
    """Stairs/elevator from what the place IS (type, label), never from text read nearby: posters
    like 'take the stairs, not the elevator' hang in ordinary hallways."""
    label = n.label.upper()
    if n.type == "elevator" or (n.type not in ("room", "restroom", "exit") and "ELEVATOR" in label):
        return "elevator"
    if n.type == "stairs" or (n.type not in ("room", "restroom", "exit") and "STAIR" in label):
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
