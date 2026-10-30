"""Procedural indoor buildings for the evaluation harness: floor plans, textures, doors, plaques, signs,
posters and straight staircases, turned into `render.Quad`s.

A building spec (see eval/scenarios/*/spec.py) gives, per level, an ASCII floor plan on a 1 m grid (north up):
    '#' solid   '.' corridor   'o' corridor cell that opens into a stair shaft   'S' stair shaft (all levels)
plus decals placed on walls (`wall: [x, y, facing]`: a point on a wall line and the compass direction the wall
faces, i.e. into the corridor), stair flights inside the shaft, and extra places (junctions, stairs).
Walls are one storey (`storey` m) high; ceilings are at `ceiling` m above each level's floor.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .render import Quad

PX_M = 128  # texture density of walls, floors and ceilings (pixels per metre)
FACING = {"N": (0.0, 1.0), "S": (0.0, -1.0), "E": (1.0, 0.0), "W": (-1.0, 0.0)}
UP = np.array([0.0, 0.0, 1.0])


def _h(*parts) -> int:
    """Stable hash (Python's hash() is salted per process)."""
    return int.from_bytes(hashlib.blake2b(repr(parts).encode(), digest_size=8).digest(), "little")


def _rng(*parts) -> np.random.Generator:
    return np.random.default_rng(_h(*parts))


# ---------------------------------------------------------------- text panels
def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


def _fit(draw: ImageDraw.ImageDraw, text: str, max_w: int, max_h: int) -> ImageFont.FreeTypeFont:
    size = max_h
    while size > 8:
        f = _font(size)
        x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=f)
        if x1 - x0 <= max_w and y1 - y0 <= max_h:
            return f
        size = int(size * 0.92)
    return _font(8)


def _arrow(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], direction: str, fill) -> None:
    x0, y0, x1, y1 = box
    cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    s = min(w, h) / 2
    shaft, head = s * 0.28, s * 0.95
    pts = [(-s, -shaft), (s - head * 0.8, -shaft), (s - head * 0.8, -head * 0.8), (s, 0),
           (s - head * 0.8, head * 0.8), (s - head * 0.8, shaft), (-s, shaft)]
    rot = {"right": 0, "down": 90, "left": 180, "up": 270}[direction]
    c, sn = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    draw.polygon([(cx + x * c - y * sn, cy + x * sn + y * c) for x, y in pts], fill=fill)


def text_panel(lines: list[str], size_m: tuple[float, float], bg, fg, px_m: int = 1600, border=None,
               arrow: str | None = None, picture: bool = False, seed: int = 0) -> np.ndarray:
    """RGB->BGR texture of a sign: centred lines of text (auto-fitted), optional border, arrow and picture."""
    w, h = int(size_m[0] * px_m), int(size_m[1] * px_m)
    img = Image.new("RGB", (w, h), tuple(bg))
    d = ImageDraw.Draw(img)
    pad = int(min(w, h) * 0.08)
    if border is not None:
        d.rectangle([pad // 3, pad // 3, w - pad // 3, h - pad // 3], outline=tuple(border), width=max(2, pad // 4))
    x0, x1, y0, y1 = pad, w - pad, pad, h - pad
    if picture:  # a poster: abstract picture in the top half
        rng = _rng("poster", seed)
        ph = int((y1 - y0) * 0.45)
        for _ in range(6):
            r = int(rng.integers(ph // 6, ph // 2))
            cx, cy = int(rng.integers(x0 + r, max(x0 + r + 1, x1 - r))), int(rng.integers(y0 + r // 2, y0 + ph))
            col = tuple(int(v) for v in rng.integers(40, 230, 3))
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
        y0 += ph + pad // 2
    if arrow:
        aw = int((y1 - y0) * 1.1)
        _arrow(d, (x1 - aw, y0, x1, y1), arrow, tuple(fg))
        x1 -= aw + pad
    n = len(lines)
    lh = (y1 - y0) / n
    for i, line in enumerate(lines):
        f = _fit(d, line, x1 - x0, int(lh * 0.82))
        bx0, by0, bx1, by1 = d.textbbox((0, 0), line, font=f)
        tx = x0 + (x1 - x0 - (bx1 - bx0)) / 2 - bx0
        ty = y0 + i * lh + (lh - (by1 - by0)) / 2 - by0
        d.text((tx, ty), line, font=f, fill=tuple(fg))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


SIGN_STYLES = {  # bg, fg, border (RGB)
    "plaque": ((245, 245, 240), (20, 20, 25), (60, 60, 70)),
    "plaque_dark": ((25, 45, 90), (250, 250, 250), (200, 200, 210)),
    "floor": ((20, 70, 150), (255, 255, 255), (230, 230, 240)),
    "exit": ((10, 140, 60), (255, 255, 255), None),
    "stairs": ((40, 40, 45), (255, 255, 255), (220, 220, 220)),
    "directional": ((35, 35, 40), (255, 255, 255), None),
    "name": ((235, 235, 230), (30, 30, 30), (90, 90, 90)),
}


# ---------------------------------------------------------------- surface textures
def wall_texture(level: int, axis_key: tuple, a0: float, a1: float, flip: bool, height: float, ceiling: float,
                 stripe_bgr, period: float | None) -> np.ndarray:
    """Painted wall from a0 to a1 (metres along its axis, world order), floor to `height` m. `flip`: the
    texture's x runs in decreasing world coordinate. Details repeat every `period` m when given (twin spots)."""
    w, h = max(2, int(round((a1 - a0) * PX_M))), int(round(height * PX_M))
    world = a0 + (np.arange(w) + 0.5) / PX_M  # world coordinate of each column
    key = world % period if period else world
    rng = _rng("wallnoise", level, axis_key)
    base = np.array([214, 222, 226], np.float32) + rng.normal(0, 3, 3).astype(np.float32)
    img = np.empty((h, w, 3), np.float32)
    img[:] = base
    # low-frequency mottling that is a pure function of the (periodic) world coordinate
    zz = (h - np.arange(h)[:, None] - 0.5) / PX_M
    ph = rng.uniform(0, 6.28, 4)
    mott = (np.sin(key[None, :] * 3.1 + ph[0]) * np.sin(zz * 2.3 + ph[1]) * 4
            + np.sin(key[None, :] * 7.7 + ph[2] + zz * 1.3) * 3)
    img += mott[..., None]
    z_of_row = zz[:, 0]
    img[z_of_row < 0.12] = (70, 78, 84)  # baseboard
    band = (z_of_row > 0.95) & (z_of_row < 1.03)
    img[band] = stripe_bgr
    # panel seams every 1.2 m
    seam = (np.abs((key + 0.6) % 1.2 - 0.6) < 0.008)
    img[:, seam & True] *= 0.8
    # fixtures: outlets, switches, fire alarm pulls, small notices; position from the (periodic) metre index
    for m in np.unique(np.floor(key)).astype(int):
        r = _rng("fixture", level, axis_key if not period else ("p", axis_key[0]), int(m))
        kind = r.integers(0, 6)
        cols = np.where(np.floor(key).astype(int) == m)[0]
        if not len(cols) or kind >= 4:
            continue
        cx = cols[0] + int(r.uniform(0.2, 0.8) * PX_M)
        if kind == 0:  # outlet
            zc, sw, sh, col = 0.3, 0.08, 0.12, (235, 235, 235)
        elif kind == 1:  # light switch
            zc, sw, sh, col = 1.2, 0.08, 0.12, (240, 240, 240)
        elif kind == 2:  # fire alarm pull
            zc, sw, sh, col = 1.25, 0.12, 0.16, (40, 40, 200)
        else:  # thermostat
            zc, sw, sh, col = 1.5, 0.12, 0.09, (200, 205, 205)
        x0, x1 = cx - int(sw * PX_M / 2), cx + int(sw * PX_M / 2)
        y0, y1 = int(h - (zc + sh / 2) * PX_M), int(h - (zc - sh / 2) * PX_M)
        if 0 <= x0 and x1 < w:
            img[y0:y1, x0:x1] = col
            img[y0:y1, x0] *= 0.6
            img[y1 - 1, x0:x1] *= 0.6
    img = cv2.GaussianBlur(img, (0, 0), 0.6)
    if flip:
        img = img[:, ::-1]
    return np.clip(img, 0, 255).astype(np.uint8)


def floor_texture(level: int, x0: float, y0: float, x1: float, y1: float, period: float | None) -> np.ndarray:
    """Tiles (0.6 m, grout lines) in world alignment. Row 0 of the texture is the NORTH edge (y1)."""
    w, h = int(round((x1 - x0) * PX_M)), int(round((y1 - y0) * PX_M))
    xs = x0 + (np.arange(w) + 0.5) / PX_M
    ys = y1 - (np.arange(h) + 0.5) / PX_M
    tile = 0.6
    ix, iy = np.floor(xs / tile).astype(int), np.floor(ys / tile).astype(int)
    pidx = (ix % int(round(period / tile))) if period else ix
    shades = np.array([[(_h("tile", level, int(a), int(b)) % 1000) / 1000.0 for a in np.unique(pidx)]
                       for b in np.unique(iy)], np.float32)
    ua, ub = {v: i for i, v in enumerate(np.unique(pidx))}, {v: i for i, v in enumerate(np.unique(iy))}
    sh = shades[np.vectorize(ub.get)(iy)[:, None], np.vectorize(ua.get)(pidx)[None, :]]
    base = np.array([150, 160, 165], np.float32) if level % 2 else np.array([125, 140, 150], np.float32)
    img = base[None, None, :] * (0.88 + 0.24 * sh[..., None])
    gx = np.abs(((xs + tile / 2) % tile) - tile / 2) < 0.012
    gy = np.abs(((ys + tile / 2) % tile) - tile / 2) < 0.012
    img[:, gx] *= 0.55
    img[gy, :] *= 0.55
    # sparse speckles (terrazzo-like) for texture; with a period, the same pattern repeats along x
    rng = _rng("speck", level, round(y0, 2)) if period else _rng("speck", level, round(x0, 2), round(y0, 2))
    span = int(round(period * PX_M)) if period else w
    n = int(span * h / 400)
    sx, py = rng.integers(0, span, n), rng.integers(0, h, n)
    fac = rng.uniform(0.6, 1.25, n).astype(np.float32)
    col_key = (np.floor(xs * PX_M).astype(int) % span) if period else np.arange(w)
    spk = np.ones((h, span), np.float32)
    spk[py, sx] = fac
    img *= spk[:, col_key][..., None]
    return np.clip(cv2.GaussianBlur(img, (0, 0), 0.5), 0, 255).astype(np.uint8)


def ceiling_texture(level: int, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    w, h = int(round((x1 - x0) * PX_M)), int(round((y1 - y0) * PX_M))
    xs = x0 + (np.arange(w) + 0.5) / PX_M
    ys = y1 - (np.arange(h) + 0.5) / PX_M
    img = np.full((h, w, 3), 205, np.float32)
    tile = 0.6
    ix, iy = np.floor(xs / tile).astype(int), np.floor(ys / tile).astype(int)
    light = ((ix % 4) == 1)[None, :] & ((iy % 3) == 1)[:, None]
    img[light] = (250, 250, 248)
    gx = np.abs(((xs + tile / 2) % tile) - tile / 2) < 0.01
    gy = np.abs(((ys + tile / 2) % tile) - tile / 2) < 0.01
    img[:, gx] = (150, 150, 150)
    img[gy, :] = (150, 150, 150)
    rng = _rng("ceil", level, round(x0, 2), round(y0, 2))
    img += rng.normal(0, 4, (h, w, 1)).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def door_texture(color_bgr, latch_right: bool, px_m: int = 200) -> np.ndarray:
    w, h = int(0.9 * px_m), int(2.1 * px_m)
    img = np.empty((h, w, 3), np.float32)
    img[:] = color_bgr
    rng = _rng("door", tuple(color_bgr))
    img += np.repeat(rng.normal(0, 5, (1, w, 1)), h, axis=0).astype(np.float32)  # grain
    for (a, b) in ((0.12, 0.45), (0.55, 0.92)):  # inset panels
        y0, y1, x0, x1 = int(h * a), int(h * b), int(w * 0.15), int(w * 0.85)
        img[y0:y1, x0:x1] *= 0.9
        img[y0:y0 + 3, x0:x1] *= 0.7
        img[y0:y1, x0:x0 + 3] *= 0.7
    img[int(h * 0.88):, :] = (150, 155, 160)  # kick plate
    hx = int(w * 0.85) if latch_right else int(w * 0.08)
    img[int(h * 0.48):int(h * 0.52), hx:hx + int(w * 0.07)] = (60, 60, 60)
    return np.clip(img, 0, 255).astype(np.uint8)


def stair_textures() -> dict[str, np.ndarray]:
    rng = _rng("stairs")
    tread = np.full((40, 256, 3), (120, 125, 128), np.float32) + rng.normal(0, 6, (40, 256, 1)).astype(np.float32)
    tread[:7] = (40, 160, 200)  # yellow-ish anti-slip nosing (BGR)
    riser = np.full((24, 256, 3), (185, 190, 192), np.float32) + rng.normal(0, 4, (24, 256, 1)).astype(np.float32)
    rail = np.full((8, 64, 3), (40, 40, 40), np.float32)
    return {k: np.clip(v, 0, 255).astype(np.uint8) for k, v in
            (("tread", tread), ("riser", riser), ("rail", rail))}


# ---------------------------------------------------------------- building
@dataclass
class Decal:
    id: str
    kind: str  # plaque | sign | poster
    text: str  # the text as written (lines joined by a space)
    level: int
    pos: np.ndarray  # world centre
    wayfinding: bool  # a sign a perfect reader would report (not posters)
    room: str | None = None  # room number for door plaques
    style: str = ""


@dataclass
class Place:
    id: str
    label: str
    type: str
    floor: int
    stand: tuple[float, float]  # where a walker stands to be "at" the place
    look: tuple[float, float, float] | None = None  # what the walker looks at (plaque centre)
    room: str | None = None
    group: str | None = None  # stairwell / elevator bank id (vertical places)
    aliases: list[str] = field(default_factory=list)


@dataclass
class Stairway:
    """A straight stair run inside a shaft, walked as a polyline of (x, y, floor height) points."""
    id: str
    path: list[tuple[float, float, float]]  # bottom (lower level) ... top (upper level)
    lower: int
    upper: int


class Building:
    def __init__(self, spec: dict):
        self.spec = spec
        self.storey = spec.get("storey", 3.4)
        self.ceiling = spec.get("ceiling", 2.7)
        self.levels: dict[int, dict] = {int(k): v for k, v in spec["levels"].items()}
        self.quads: list[Quad] = []
        self.textures: dict[str, np.ndarray] = stair_textures()
        self.decals: dict[str, Decal] = {}
        self.places: dict[str, Place] = {}
        self.stairways: dict[str, Stairway] = {}
        self.shaft_box: tuple[float, float, float, float] | None = None
        rows = {len(v["grid"]) for v in self.levels.values()}
        assert len(rows) == 1, "all level grids must have the same size"
        self.nrows = rows.pop()
        self.ncols = max(len(r) for v in self.levels.values() for r in v["grid"])
        for lvl in sorted(self.levels):
            self._level_geometry(lvl)
        for d in spec.get("decals", []):
            self._decal(d)
        for s in spec.get("stairs", []):
            self._stairs(s)
        for p in spec.get("places", []):
            self.places[p["id"]] = Place(id=p["id"], label=p["label"], type=p["type"], floor=p["floor"],
                                         stand=tuple(p["stand"]), room=p.get("room"), group=p.get("group"),
                                         aliases=p.get("aliases", []),
                                         look=tuple(p["look"]) if p.get("look") else None)

    # ------------------------------------------------------------ grid helpers
    def z_of(self, level: int) -> float:
        return float(self.levels[level].get("z", (level - min(self.levels)) * self.storey))

    def cell(self, level: int, col: int, row: int) -> str:
        if not (0 <= row < self.nrows and 0 <= col < self.ncols):
            return "#"
        line = self.levels[level]["grid"][row]
        return line[col] if col < len(line) else "#"

    def cell_at(self, level: int, x: float, y: float) -> str:
        return self.cell(level, int(math.floor(x)), self.nrows - 1 - int(math.floor(y)))

    def in_shaft(self, x: float, y: float) -> bool:
        return any(self.cell_at(lvl, x, y) == "S" for lvl in self.levels)

    def _levels_with_shaft(self) -> list[int]:
        return sorted(l for l in self.levels if any("S" in r for r in self.levels[l]["grid"]))

    # ------------------------------------------------------------ floors, ceilings, walls
    def _level_geometry(self, lvl: int) -> None:  # noqa: C901
        z = self.z_of(lvl)
        shaft_levels = self._levels_with_shaft()
        period = self.levels[lvl].get("period")
        stripe = tuple(self.levels[lvl].get("stripe", (150, 90, 40)))
        free = {(c, r): self.cell(lvl, c, r) for r in range(self.nrows) for c in range(self.ncols)
                if self.cell(lvl, c, r) != "#"}
        # floor / ceiling rectangles: merge horizontal runs of cells, row by row
        for kind in ("floor", "ceiling"):
            cells = set()
            for (c, r), ch in free.items():
                if ch != "S":
                    cells.add((c, r))
                elif kind == "floor" and lvl == shaft_levels[0]:
                    cells.add((c, r))
                elif kind == "ceiling" and lvl == shaft_levels[-1]:
                    cells.add((c, r))
            for r in range(self.nrows):
                c = 0
                while c < self.ncols:
                    if (c, r) not in cells:
                        c += 1
                        continue
                    c0 = c
                    while (c, r) in cells:
                        c += 1
                    x0, x1 = float(c0), float(c)
                    y0, y1 = float(self.nrows - 1 - r), float(self.nrows - r)
                    if kind == "floor":
                        name = f"floor_{lvl}_{r}_{c0}"
                        self.textures[name] = floor_texture(lvl, x0, y0, x1, y1, period)
                        self.quads.append(Quad([x0, y1, z], [x1 - x0, 0, 0], [0, -(y1 - y0), 0], name, 0.95, lvl))
                    else:
                        name = f"ceil_{lvl}_{r}_{c0}"
                        zc = z + self.ceiling
                        self.textures[name] = ceiling_texture(lvl, x0, y0, x1, y1)
                        self.quads.append(Quad([x0, y1, zc], [x1 - x0, 0, 0], [0, -(y1 - y0), 0], name, 1.05, lvl))
        # walls: faces between a free cell and a solid cell (or '.'-'S'), merged into runs along each grid line
        faces: dict[tuple, list[tuple[int, str]]] = {}  # (facing, line) -> [(cell index along line, kind)]
        for (c, r), ch in free.items():
            for facing, (dc, dr) in (("N", (0, 1)), ("S", (0, -1)), ("E", (-1, 0)), ("W", (1, 0))):
                # the wall faces `facing` and lies on the side of (c, r) opposite to that direction
                nb = self.cell(lvl, c + dc, r + dr)
                if nb == "#":
                    kind = "full"
                elif ch == "S" and nb == "." or ch == "." and nb == "S":
                    kind = "full"
                elif ch == "S" and nb == "o" or ch == "o" and nb == "S":
                    kind = "lintel"
                else:
                    continue
                if facing in ("N", "S"):
                    # facing N: the wall is the cell's south edge; facing S: its north edge
                    line = float(self.nrows - 1 - r) if facing == "N" else float(self.nrows - r)
                    faces.setdefault((facing, line), []).append((c, kind))
                else:
                    line = float(c) if facing == "E" else float(c + 1)
                    faces.setdefault((facing, line), []).append((self.nrows - 1 - r, kind))
        for (facing, line), items in faces.items():
            for kind in ("full", "lintel"):
                idx = sorted(i for i, k in items if k == kind)
                runs, start = [], None
                for j, i in enumerate(idx):
                    if start is None:
                        start = i
                    if j + 1 == len(idx) or idx[j + 1] != i + 1:
                        runs.append((start, i + 1))
                        start = None
                for a0, a1 in runs:
                    self._wall(lvl, facing, line, float(a0), float(a1), kind, period, stripe)

    def _wall(self, lvl, facing, line, a0, a1, kind, period, stripe) -> None:
        z = self.z_of(lvl)
        if kind == "full":
            zb, zt = z, z + self.storey
        else:  # lintel over an opening into the shaft
            zb, zt = z + self.ceiling, z + self.storey
        nx, ny = FACING[facing]
        right = np.cross(-np.array([nx, ny, 0.0]), UP)  # viewer's right when facing the wall
        if facing in ("N", "S"):
            p_lo, p_hi = np.array([a0, line, zt]), np.array([a1, line, zt])
            flip = right[0] < 0
        else:
            p_lo, p_hi = np.array([line, a0, zt]), np.array([line, a1, zt])
            flip = right[1] < 0
        # a hair into the free side, so the two faces of a corridor/shaft wall never z-fight
        off = np.array([nx, ny, 0.0]) * 0.004
        p0 = (p_hi if flip else p_lo) + off
        u = (p_lo - p_hi) if flip else (p_hi - p_lo)
        name = f"wall_{lvl}_{facing}_{line}_{a0}_{kind}"
        height = zt - zb
        tex = wall_texture(lvl, (facing, line), a0, a1, flip, self.storey, self.ceiling, stripe, period)
        tex = tex[: int(round(height * PX_M))]  # rows run top-down: a lintel keeps the top band
        self.textures[name] = tex
        self.quads.append(Quad(p0, u, [0, 0, -height], name, 0.92 if facing in ("E", "W") else 1.0, lvl))

    # ------------------------------------------------------------ decals
    def _wall_frame(self, d: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        x, y, facing = d["wall"]
        lvl = d["level"]
        nx, ny = FACING[facing]
        n = np.array([nx, ny, 0.0])
        right = np.cross(-n, UP)
        return np.array([x, y, self.z_of(lvl)]), n, right, lvl

    def _panel(self, did, base, n, right, lvl, centre_off, z_centre, size, tex, eps=0.012) -> Quad:
        w, h = size
        c = base + right * centre_off + n * eps + UP * z_centre
        p0 = c - right * (w / 2) + UP * (h / 2)
        return Quad(p0, right * w, -UP * h, tex, 1.0, lvl, decal=did)

    def _decal(self, d: dict) -> None:
        base, n, right, lvl = self._wall_frame(d)
        kind = d["kind"]
        did = d["id"]
        if kind == "door":
            latch_right = d.get("plaque_side", "right") == "right"
            dtex = f"door_{did}"
            self.textures[dtex] = door_texture(tuple(d.get("color", (60, 95, 140))), latch_right)
            self.textures.setdefault("frame", np.full((8, 8, 3), (70, 70, 75), np.uint8))
            self.quads.append(self._panel(None, base, n, right, lvl, 0, 1.075, (1.02, 2.15), "frame", 0.004))
            self.quads.append(self._panel(None, base, n, right, lvl, 0, 1.05, (0.9, 2.1), dtex, 0.008))
            if d.get("plaque"):
                lines = d["plaque"] if isinstance(d["plaque"], list) else [d["plaque"]]
                style = d.get("plaque_style", "plaque")
                bg, fg, border = SIGN_STYLES[style]
                size = tuple(d.get("plaque_size", (0.28, 0.14)))
                ptex = f"plaque_{did}"
                self.textures[ptex] = text_panel(lines, size, bg, fg, border=border)
                off = (0.75 if latch_right else -0.75)
                pq = self._panel(f"{did}:plaque", base, n, right, lvl, off, 1.5, size, ptex)
                self.quads.append(pq)
                centre = pq.p0 + pq.u / 2 + pq.v / 2
                self.decals[f"{did}:plaque"] = Decal(f"{did}:plaque", "plaque", " ".join(lines), lvl, centre, True,
                                                     d.get("room"), style)
            if d.get("sign"):  # e.g. EXIT above the door
                self._sign(f"{did}:sign", d["sign"], base, n, right, lvl, 0.0, 2.35)
            if d.get("place"):
                p = d["place"]
                look = self.decals.get(f"{did}:plaque") or self.decals.get(f"{did}:sign")
                side = (0.75 if latch_right else -0.75) if d.get("plaque") else 0.0  # facing the plaque
                stand = base + n * d.get("stand", 1.0) + right * side
                self.places[p["id"]] = Place(
                    id=p["id"], label=p["label"], type=p.get("type", "room"), floor=lvl,
                    stand=(round(float(stand[0]), 2), round(float(stand[1]), 2)), room=d.get("room"),
                    look=tuple(float(v) for v in (look.pos if look else base + UP * 1.5)),
                    aliases=p.get("aliases", []))
        elif kind == "sign":
            self._sign(did, d, base, n, right, lvl, 0.0, d.get("height", 1.7))
            if d.get("place"):
                p = d["place"]
                stand = base + n * d.get("stand", 1.0)
                self.places[p["id"]] = Place(id=p["id"], label=p["label"], type=p.get("type", "landmark"), floor=lvl,
                                             stand=(round(float(stand[0]), 2), round(float(stand[1]), 2)),
                                             look=tuple(float(v) for v in self.decals[did].pos), group=p.get("group"))
        elif kind == "poster":
            size = tuple(d.get("size", (0.6, 0.85)))
            ptex = f"poster_{d.get('look_key', did)}"
            if ptex not in self.textures:
                bg = tuple(d.get("bg", (250, 220, 90)))
                self.textures[ptex] = text_panel(d["lines"], size, bg, (20, 20, 20), px_m=900, picture=True,
                                                 seed=_h(d.get("look_key", did)) % 10_000)
            q = self._panel(did, base, n, right, lvl, 0.0, d.get("height", 1.45), size, ptex)
            self.quads.append(q)
            self.decals[did] = Decal(did, "poster", " ".join(d["lines"]), lvl, q.p0 + q.u / 2 + q.v / 2, False)
        else:
            raise ValueError(f"unknown decal kind {kind!r}")

    def _sign(self, did, d, base, n, right, lvl, off, zc) -> None:
        style = d.get("style", "name")
        bg, fg, border = SIGN_STYLES[style]
        lines = d["text"] if isinstance(d["text"], list) else [d["text"]]
        size = tuple(d.get("size", {"exit": (0.42, 0.18), "floor": (0.6, 0.3), "stairs": (0.6, 0.25),
                                     "directional": (1.1, 0.26)}.get(style, (0.7, 0.25))))
        tex = f"sign_{did}"
        self.textures[tex] = text_panel(lines, size, bg, fg, border=border, arrow=d.get("arrow"))
        q = self._panel(did, base, n, right, lvl, off + d.get("offset", 0.0), zc, size, tex, 0.015)
        self.quads.append(q)
        self.decals[did] = Decal(did, "sign", " ".join(lines), lvl, q.p0 + q.u / 2 + q.v / 2, True, style=style)

    # ------------------------------------------------------------ stairs
    def _stairs(self, s: dict) -> None:
        """Straight flights going `dir` (N/S/E/W) between x0..x1 (or y0..y1 across), with optional landings.
        s = {id, lower, upper, dir, across: [a0, a1], start: along-coordinate of the first riser,
             flights: [steps, ...], rise, run, landing: length between flights,
             bottom: [x, y], top: [x, y]  (walk endpoints on the two levels)}"""
        d = s["dir"]
        sgn = 1.0 if d in ("N", "E") else -1.0
        along_y = d in ("N", "S")
        a0, a1 = s["across"]
        rise, run = s.get("rise", 0.17), s.get("run", 0.28)
        z = self.z_of(s["lower"])
        pos = s["start"]
        path = [(s["bottom"][0], s["bottom"][1], z)]

        def pt(along, across):
            return (across, along) if along_y else (along, across)

        mid = (a0 + a1) / 2
        for fi, steps in enumerate(s["flights"]):
            path.append((*pt(pos, mid), z))
            z_start, pos_start = z, pos
            for i in range(steps):
                y_lo = pos
                p_a, p_b = np.array([*pt(y_lo, a0), z]), np.array([*pt(y_lo, a1), z])
                # riser (vertical face at the front of the step), then tread on top of it
                self.quads.append(Quad(p_a + UP * rise, p_b - p_a, -UP * rise, "riser", 0.95, "shaft"))
                z += rise
                pos += sgn * run
                q_a = np.array([*pt(y_lo, a0), z])
                fwd = np.array([*pt(pos, a0), z]) - q_a
                self.quads.append(Quad(q_a, p_b - p_a, fwd, "tread", 1.0, "shaft"))
            path.append((*pt(pos, mid), z))
            # handrails on both side walls, 0.9 m above the pitch line
            for side, eps in ((a0, 0.03), (a1, -0.03)):
                start = np.array([*pt(pos_start, side + eps), z_start + 0.9])
                end = np.array([*pt(pos, side + eps), z + 0.9])
                self.quads.append(Quad(start, end - start, -UP * 0.06, "rail", 1.0, "shaft"))
            if fi + 1 < len(s["flights"]):  # landing between flights
                L = s.get("landing", 1.2)
                p_a = np.array([*pt(pos, a0), z])
                p_end = np.array([*pt(pos + sgn * L, a0), z])
                name = f"landing_{s['id']}_{fi}"
                self.textures[name] = floor_texture(s["lower"], 0, 0, abs(a1 - a0), L, None)
                self.quads.append(Quad(p_a, np.array([*pt(pos, a1), z]) - p_a, p_end - p_a, name, 0.95, "shaft"))
                pos += sgn * L
        # top landing up to the shaft end
        top_end = s["top_landing_end"]
        p_a = np.array([*pt(pos, a0), z])
        p_end = np.array([*pt(top_end, a0), z])
        name = f"landing_{s['id']}_top"
        self.textures[name] = floor_texture(s["upper"], 0, 0, abs(a1 - a0), abs(top_end - pos), None)
        self.quads.append(Quad(p_a, np.array([*pt(pos, a1), z]) - p_a, p_end - p_a, name, 0.95, "shaft"))
        path.append((s["top"][0], s["top"][1], z))
        assert abs(z - self.z_of(s["upper"])) < 1e-6, f"stairs {s['id']} end at z={z}, level is {self.z_of(s['upper'])}"
        self.stairways[s["id"]] = Stairway(s["id"], path, s["lower"], s["upper"])
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        self.shaft_box = (min(xs), min(ys), max(xs), max(ys))

    # ------------------------------------------------------------ per-frame culling
    def active_quads(self, x: float, y: float, z_eye: float) -> list[Quad]:
        """Quads worth rendering from here: the camera's own level plus the stair shaft; everything when the
        camera is in or near the shaft (it can see up and down the stairs)."""
        near_shaft = False
        if self.shaft_box is not None:
            bx0, by0, bx1, by1 = self.shaft_box
            near_shaft = bx0 - 3 <= x <= bx1 + 3 and by0 - 3 <= y <= by1 + 3
        if near_shaft:
            return self.quads
        lvl = min(self.levels, key=lambda l: abs(z_eye - 1.6 - self.z_of(l)))
        return [q for q in self.quads if q.zone in (lvl, "shaft")]
