"""A perfect-perception stand-in for Gemma, built from the generator's per-frame sidecars.

`--oracle` answers every Gemma question (frame analysis, same place, same stairwell) from ground truth, so a
run scores the mapping / floors / routing logic alone. Comparing a live run with an oracle run splits the
errors into "Gemma/OCR saw it wrong" and "the map logic got it wrong". It needs `<video>.frames.json`
sidecars, which only the synthetic generator writes; it also keeps the main test suite model-free.
"""
from __future__ import annotations

import json
import re
from contextlib import contextmanager
from pathlib import Path

from app import gemma
from app.schemas import FrameAnalysis, SamePlace

from .trace import Tracer
from .truth import Truth

LEGIBLE_PX = {"plaque": 22.0, "sign": 16.0, "poster": 30.0}  # projected decal height at 720p to read it
MIN_COS = 0.35  # flatter than ~70 degrees off-axis is unreadable
PROMINENT_PX = 55.0  # a plaque this big is the subject of the photo (place_type room)
_STAIRS = {"up": "on_stairs_up", "down": "on_stairs_down", "ahead_up": "ahead_up", "ahead_down": "ahead_down"}


class Oracle:
    def __init__(self, truth: Truth, scenario_dir: Path, tracer: Tracer):
        self.truth = truth
        self.tracer = tracer
        self.sidecars: dict[str, dict] = {}
        for v in truth.videos:
            side = scenario_dir / Path(v.file).with_suffix(".frames.json")
            if not side.exists():
                raise FileNotFoundError(f"--oracle needs the generator sidecar {side}")
            self.sidecars[v.id] = json.loads(side.read_text())
        self.video_names: dict[str, str] = {}  # app video id -> truth video id (filled by the runner)

    # ------------------------------------------------------------ lookups
    def _frame(self, path: Path) -> tuple[str, float] | None:
        hit = self.tracer.frames.get(Path(path).name)
        if hit is None:
            return None
        vid, t = hit
        name = self.video_names.get(vid)
        return (name, t) if name else None

    def _state(self, name: str, t: float) -> dict:
        side = self.sidecars[name]
        frames = side["frames"]
        return frames[max(0, min(len(frames) - 1, int(round(t * side["fps"]))))]

    def place_at(self, name: str, t: float, tol: float = 2.0) -> str | None:
        video = next(v for v in self.truth.videos if v.id == name)
        best = min(video.visits, key=lambda v: abs(v.t - t), default=None)
        return best.place if best is not None and abs(best.t - t) <= tol else None

    # ------------------------------------------------------------ Gemma replacements
    def analyze(self, path: Path) -> FrameAnalysis:
        where = self._frame(path)
        if where is None:
            return FrameAnalysis(suggested_label="Hallway", description="corridor")
        name, t = where
        side = self.sidecars[name]
        st = self._state(name, t)
        scale = 720.0 / side["size"][1]
        texts, rooms, ocr, floor_hint = [], [], [], ""
        prominent_room = prominent_exit = None
        for did, h, _w, cx, cos in st["see"]:
            d = side["decals"][did]
            h *= scale
            if h < LEGIBLE_PX[d["kind"]] or cos < MIN_COS:
                continue
            ocr.append(d["text"])
            if not d["wayfinding"]:
                continue
            texts.append(d["text"])
            if d["room"]:
                rooms.append(d["room"])
                if h >= PROMINENT_PX and abs(cx - 0.5) < 0.35:
                    prominent_room = d["room"]
            if d["style"] == "exit" and h >= 20:
                prominent_exit = d["text"]
            if d["style"] == "floor" and (m := re.search(r"\d+", d["text"])):
                floor_hint = m.group()
        stairs = _STAIRS.get(st["stairs"], "none")
        if stairs != "none":
            ptype, label = "stairs", "Stairs"
        elif prominent_room:
            ptype, label = "room", f"Room {prominent_room}"
        elif prominent_exit:
            # place_type follows the plate's look, like a vision model: a green 'NO EXIT' plate is typed an exit
            # too, and the mapper must check what the sign says (its label/texts are read correctly)
            ptype, label = "exit", prominent_exit.title()
        else:
            ptype, label = "hallway", ("Hallway with " + texts[0]) if texts else "Hallway"
        return FrameAnalysis(texts=texts, room_numbers=rooms, has_sign=bool(texts), suggested_label=label,
                             place_type=ptype, stairs=stairs, floor_hint=floor_hint,
                             description=label.lower(), ocr=ocr, grounded=True)

    def _compare(self, path_a: Path, path_b: Path, stairwell: bool) -> SamePlace:
        a, b = self._frame(path_a), self._frame(path_b)
        pa = self.place_at(*a) if a else None
        pb = self.place_at(*b) if b else None
        if stairwell and pa and pb:
            ga, gb = self.truth.place(pa).group, self.truth.place(pb).group
            same = ga is not None and ga == gb
            return SamePlace(same_place=same, confidence=0.95, reason=f"oracle: {pa} / {pb}")
        if pa and pb:
            return SamePlace(same_place=pa == pb, confidence=0.95, reason=f"oracle: {pa} / {pb}")
        return SamePlace(same_place=False, confidence=0.5, reason="oracle: not at a known place")

    @contextmanager
    def installed(self):
        names = ("resolve_model", "ensure_model", "analyze_frame", "cached_analysis", "same_place", "same_stairwell")
        saved = {n: getattr(gemma, n) for n in names}
        gemma.resolve_model = lambda: "oracle"
        gemma.ensure_model = lambda model: None
        gemma.analyze_frame = lambda path, cache, model=None: self.analyze(path)
        gemma.cached_analysis = lambda path, cache, model=None: self.analyze(path)
        gemma.same_place = lambda a, b, cache, model=None: self._compare(a, b, False)
        gemma.same_stairwell = lambda a, b, cache, model=None: self._compare(a, b, True)
        try:
            yield self
        finally:
            for n, f in saved.items():
                setattr(gemma, n, f)
