"""A tiny textured-quad renderer for synthetic walkthroughs (numpy + OpenCV only).

World frame: x east, y north, z up, metres. Heading (yaw) is degrees clockwise from north, the app's
convention (a right turn increases it); pitch is degrees up. The camera is a pinhole with the same
horizontal field of view the pipeline assumes (config.LONG_SIDE_FOV_DEG), so measured yaw is in true degrees.

Every surface is a textured parallelogram (`Quad`). Per frame and per quad we intersect the pixel rays of
the quad's screen bounding box with its plane, keep pixels inside it and nearer than the z-buffer, and
sample the texture with cv2.remap from a mip level chosen per pixel (so far tiles do not shimmer).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

NEAR = 0.05
FAR = 40.0


@dataclass
class Quad:
    p0: np.ndarray  # texture origin corner (texture pixel 0,0)
    u: np.ndarray  # edge along texture x
    v: np.ndarray  # edge along texture y (down the texture)
    tex: str  # texture name
    shade: float = 1.0
    zone: str | int = 0  # level number, or "shaft" (see Scene.active)
    decal: str | None = None  # id of a text decal, for visibility reports
    # derived
    n: np.ndarray = field(init=False)
    u_dual: np.ndarray = field(init=False)
    v_dual: np.ndarray = field(init=False)
    corners: np.ndarray = field(init=False)

    def __post_init__(self):
        self.p0, self.u, self.v = (np.asarray(a, dtype=np.float64) for a in (self.p0, self.u, self.v))
        self.n = np.cross(self.u, self.v)
        g = np.array([[self.u @ self.u, self.u @ self.v], [self.u @ self.v, self.v @ self.v]])
        gi = np.linalg.inv(g)
        self.u_dual = gi[0, 0] * self.u + gi[0, 1] * self.v  # s = u_dual . (Q - p0)
        self.v_dual = gi[1, 0] * self.u + gi[1, 1] * self.v
        self.corners = np.array([self.p0, self.p0 + self.u, self.p0 + self.u + self.v, self.p0 + self.v])


@dataclass
class Pose:
    x: float
    y: float
    z: float
    yaw: float  # deg clockwise from north
    pitch: float = 0.0  # deg, up positive

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(centre, forward, right, up) in world coordinates."""
        yw, pt = math.radians(self.yaw), math.radians(self.pitch)
        f = np.array([math.sin(yw) * math.cos(pt), math.cos(yw) * math.cos(pt), math.sin(pt)])
        r = np.array([math.cos(yw), -math.sin(yw), 0.0])
        up = np.cross(r, f)
        return np.array([self.x, self.y, self.z]), f, r, up


def mip_chain(img: np.ndarray) -> list[np.ndarray]:
    levels = [img]
    while min(levels[-1].shape[:2]) > 8 and len(levels) < 7:
        levels.append(cv2.pyrDown(levels[-1]))
    return levels


class Camera:
    def __init__(self, width: int, height: int, hfov_deg: float):
        self.w, self.h = width, height
        self.f = (width / 2) / math.tan(math.radians(hfov_deg / 2))
        xs = (np.arange(width, dtype=np.float32) + 0.5 - width / 2) / self.f
        ys = -(np.arange(height, dtype=np.float32) + 0.5 - height / 2) / self.f
        self.a, self.b = np.meshgrid(xs, ys)  # ray = f + a*r + b*up

    def project(self, pts: np.ndarray, c, f, r, up) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d = pts - c
        z = d @ f
        return self.w / 2 + self.f * (d @ r) / z, self.h / 2 - self.f * (d @ up) / z, z


def _clip_near(poly: np.ndarray, c: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clip of a convex polygon against the plane depth >= NEAR."""
    out = []
    n = len(poly)
    depth = (poly - c) @ f
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        dp, dq = depth[i], depth[(i + 1) % n]
        if dp >= NEAR:
            out.append(p)
        if (dp >= NEAR) != (dq >= NEAR):
            out.append(p + (q - p) * ((NEAR - dp) / (dq - dp)))
    return np.array(out)


class Renderer:
    def __init__(self, camera: Camera, textures: dict[str, np.ndarray], background=(40, 40, 44)):
        self.cam = camera
        self.mips = {k: [m.astype(np.float32) for m in mip_chain(v)] for k, v in textures.items()}
        self.background = np.array(background, dtype=np.float32)

    def render(self, quads: list[Quad], pose: Pose) -> tuple[np.ndarray, np.ndarray]:
        cam = self.cam
        img = np.empty((cam.h, cam.w, 3), np.float32)
        img[:] = self.background
        zbuf = np.full((cam.h, cam.w), np.inf, np.float32)
        c, f, r, up = pose.basis()
        for q in quads:
            self._draw(q, img, zbuf, c, f, r, up)
        return np.clip(img, 0, 255).astype(np.uint8), zbuf

    def _bbox(self, q: Quad, c, f, r, up):
        depth = (q.corners - c) @ f
        if depth.max() < NEAR or depth.min() > FAR:
            return None
        poly = q.corners if depth.min() >= NEAR else _clip_near(q.corners, c, f)
        if len(poly) < 3:
            return None
        px, py, _ = self.cam.project(poly, c, f, r, up)
        x0, x1 = max(0, int(math.floor(px.min()))), min(self.cam.w, int(math.ceil(px.max())) + 1)
        y0, y1 = max(0, int(math.floor(py.min()))), min(self.cam.h, int(math.ceil(py.max())) + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, x1, y0, y1

    def _draw(self, q: Quad, img, zbuf, c, f, r, up) -> None:
        box = self._bbox(q, c, f, r, up)
        if box is None:
            return
        x0, x1, y0, y1 = box
        A, B = self.cam.a[y0:y1, x0:x1], self.cam.b[y0:y1, x0:x1]
        n = q.n
        denom = (n @ f) + A * (n @ r) + B * (n @ up)
        num = n @ (q.p0 - c)
        with np.errstate(divide="ignore", invalid="ignore"):
            tz = (num / denom).astype(np.float32)
        cp = c - q.p0
        s = (q.u_dual @ cp) + tz * ((q.u_dual @ f) + A * (q.u_dual @ r) + B * (q.u_dual @ up))
        t = (q.v_dual @ cp) + tz * ((q.v_dual @ f) + A * (q.v_dual @ r) + B * (q.v_dual @ up))
        zb = zbuf[y0:y1, x0:x1]
        mask = (tz > NEAR) & (tz < zb) & (s >= 0) & (s < 1) & (t >= 0) & (t < 1)
        if not mask.any():
            return
        levels = self.mips[q.tex]
        th, tw = levels[0].shape[:2]
        # texels per screen pixel, from neighbouring-pixel differences of (s, t)
        ds = np.abs(np.diff(s, axis=1, append=s[:, -1:])) * tw + np.abs(np.diff(s, axis=0, append=s[-1:, :])) * tw
        dt = np.abs(np.diff(t, axis=1, append=t[:, -1:])) * th + np.abs(np.diff(t, axis=0, append=t[-1:, :])) * th
        foot = np.maximum(ds, dt)
        lod = np.clip(np.floor(np.log2(np.maximum(foot, 1e-6)) + 0.3), 0, len(levels) - 1).astype(np.int8)
        out = np.zeros(s.shape + (3,), np.float32)
        s32, t32 = s.astype(np.float32), t.astype(np.float32)
        for k in np.unique(lod[mask]):
            tex = levels[int(k)]
            hk, wk = tex.shape[:2]
            sel = lod == k
            sampled = cv2.remap(tex, s32 * wk - 0.5, t32 * hk - 0.5, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            out[sel] = sampled[sel]
        with np.errstate(invalid="ignore", over="ignore"):
            light = q.shade * (0.5 + 0.5 * np.exp(-np.nan_to_num(tz, nan=FAR) / 14.0))
        out *= light[..., None]
        sub = img[y0:y1, x0:x1]
        sub[mask] = out[mask]
        zb[mask] = tz[mask]

    def decal_view(self, q: Quad, pose: Pose, zbuf: np.ndarray) -> dict | None:
        """How the decal quad appears in the frame just rendered (None if not visible): projected height and
        width in pixels, horizontal centre (0..1) and the cosine between its normal and the view ray."""
        c, f, r, up = pose.basis()
        centre = q.p0 + q.u / 2 + q.v / 2
        d = centre - c
        z = d @ f
        if z <= NEAR:
            return None
        px, py, _ = self.cam.project(centre[None, :], c, f, r, up)
        ix, iy = int(px[0]), int(py[0])
        if not (0 <= ix < self.cam.w and 0 <= iy < self.cam.h):
            return None
        if zbuf[iy, ix] < z - 0.03:  # something in front of it
            return None
        ends = np.array([q.p0, q.p0 + q.v, q.p0 + q.u, q.p0 + q.u + q.v])
        if ((ends - c) @ f).min() <= NEAR:
            return None
        ex, ey, _ = self.cam.project(ends, c, f, r, up)
        h_px = float(min(math.hypot(ex[1] - ex[0], ey[1] - ey[0]), math.hypot(ex[3] - ex[2], ey[3] - ey[2])))
        w_px = float(min(math.hypot(ex[2] - ex[0], ey[2] - ey[0]), math.hypot(ex[3] - ex[1], ey[3] - ey[1])))
        nrm = q.n / np.linalg.norm(q.n)
        cos = abs(float(nrm @ (d / np.linalg.norm(d))))
        return {"h": round(h_px, 1), "w": round(w_px, 1), "cx": round(float(px[0]) / self.cam.w, 3), "cos": round(cos, 2)}
