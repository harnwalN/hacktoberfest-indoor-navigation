"""Camera trajectories for synthetic walks, and the ground-truth visit times they imply.

A walk script is a list of steps (see eval/README.md):
    {"go": "r110"} / {"go": [x, y]}   walk there in a straight line (turning in place first if needed)
    {"look": "r110"}                   turn the camera to the place's plaque, hold, turn back (a glance)
    {"visit": "r110"}                  go + look
    {"pause": 1.0}                     stand still
    {"turn": 180} / {"face": 90}       turn in place by / to a heading
    {"stairs": "A", "dir": "up"}       walk the stairway's path (bottom -> top, or top -> bottom)
Headings are degrees clockwise from north, like the app.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .building import Building, _h
from .render import Pose

FPS = 10
EYE = 1.6
WALK_SPEED = 1.25  # m/s, the pipeline's WALK_SPEED_MPS
STAIR_SPEED = 0.5  # m/s horizontal on a flight
TURN_RATE = 75.0  # deg/s average for turns in place
LOOK_HOLD = 1.4  # s holding the camera on a plaque
LOOK_RATE = 100.0  # deg/s average when glancing at a plaque and back


def wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def bearing(x0, y0, x1, y1) -> float:
    return math.degrees(math.atan2(x1 - x0, y1 - y0)) % 360.0


@dataclass
class FrameState:
    pose: Pose
    level: int | None  # None while between floors
    stairs: str  # none | ahead_up | ahead_down | up | down
    moving: bool


class Walker:
    def __init__(self, building: Building, level: int, start: tuple[float, float], heading: float, seed: int = 0):
        self.b = building
        self.level = level
        self.x, self.y = start
        self.floor_z = building.z_of(level)
        self.yaw = heading % 360.0
        self.pitch = 0.0
        self.frames: list[FrameState] = []
        self.looks: list[tuple[float, str]] = []
        self.turns: list[dict] = []
        self.dist = 0.0
        r = np.random.default_rng(_h("jitter", seed))
        self._ph = r.uniform(0, 2 * math.pi, 6)

    @property
    def t(self) -> float:
        return len(self.frames) / FPS

    def _emit(self, stairs: str = "none", moving: bool = False, level: int | None = -1) -> None:
        t = self.t
        ph = self._ph
        # gentle hand-held sway (fractions of a degree) and a walking bob
        jy = 0.25 * math.sin(0.7 * t + ph[0]) + 0.12 * math.sin(2.3 * t + ph[1])
        jp = 0.20 * math.sin(0.9 * t + ph[2]) + 0.10 * math.sin(3.1 * t + ph[3])
        bob = 0.012 * math.sin(self.dist / 0.7 * math.pi) if moving else 0.0
        pose = Pose(self.x, self.y, self.floor_z + EYE + bob, self.yaw + jy, self.pitch + jp)
        self.frames.append(FrameState(pose, self.level if level == -1 else level, stairs, moving))

    # ------------------------------------------------------------ primitives
    def rotate(self, delta: float, dur: float | None = None, stairs: str = "none", dpitch: float = 0.0,
               turn: bool = True) -> None:
        """Turn in place. `turn`: a change of walking direction (recorded as a truth turn when >= 30 deg);
        False for glances, which the camera undoes."""
        if abs(delta) < 0.5 and abs(dpitch) < 0.5:
            return
        dur = dur or max(0.7, abs(delta) / TURN_RATE)
        n = max(2, int(round(dur * FPS)))
        if turn and abs(delta) >= 30:
            self.turns.append({"t": round(self.t + dur / 2, 2), "start": round(self.t, 2),
                               "end": round(self.t + n / FPS, 2), "delta": round(delta, 1)})
        y0, p0 = self.yaw, self.pitch
        for i in range(1, n + 1):
            ease = (1 - math.cos(math.pi * i / n)) / 2
            self.yaw = (y0 + delta * ease) % 360.0
            self.pitch = p0 + dpitch * ease
            self._emit(stairs)

    def face(self, heading: float) -> None:
        self.rotate(wrap180(heading - self.yaw))

    def pause(self, sec: float) -> None:
        for _ in range(int(round(sec * FPS))):
            self._emit()

    def walk_to(self, x: float, y: float, speed: float = WALK_SPEED) -> None:
        d = math.hypot(x - self.x, y - self.y)
        if d < 0.05:
            return
        self.face(bearing(self.x, self.y, x, y))
        n = max(1, int(math.ceil(d / (speed / FPS))))
        x0, y0 = self.x, self.y
        for i in range(1, n + 1):
            self.x, self.y = x0 + (x - x0) * i / n, y0 + (y - y0) * i / n
            self.dist += d / n
            self._emit(moving=True)

    def look(self, place_id: str) -> None:
        p = self.b.places[place_id]
        tx, ty = (p.look[0], p.look[1]) if p.look else p.stand
        delta = wrap180(bearing(self.x, self.y, tx, ty) - self.yaw)
        dpitch = 0.0
        if p.look:  # tilt toward signs mounted high (EXIT above a door)
            dist = max(0.3, math.hypot(tx - self.x, ty - self.y))
            dpitch = max(-25.0, min(25.0, math.degrees(math.atan2(p.look[2] - (self.floor_z + EYE), dist))))
            dpitch = dpitch if abs(dpitch) > 6 else 0.0
        dur = max(0.8, abs(delta) / LOOK_RATE)
        self.rotate(delta, dur, dpitch=dpitch, turn=False)
        n = int(round(LOOK_HOLD * FPS))
        self.looks.append((self.t + LOOK_HOLD / 2, place_id))
        for _ in range(n):
            self._emit()
        self.rotate(-delta, dur, dpitch=-dpitch, turn=False)

    def stairs(self, stairway_id: str, direction: str) -> None:
        sw = self.b.stairways[stairway_id]
        path = sw.path if direction == "up" else list(reversed(sw.path))
        self.walk_to(path[0][0], path[0][1])
        if self.b.z_of(sw.lower if direction == "up" else sw.upper) != self.floor_z:
            raise ValueError(f"stairs {stairway_id} {direction}: walker is not on the right level")
        target_pitch = 4.0 if direction == "up" else -22.0
        last = len(path) - 2
        for k, (a, b) in enumerate(zip(path, path[1:])):
            climbing = abs(b[2] - a[2]) > 1e-6
            on_flight = 0 < k < last  # first and last segments are the flat approach / exit
            state = ("up" if direction == "up" else "down") if on_flight else (
                ("ahead_up" if direction == "up" else "ahead_down") if k == 0 else "none")
            d = math.hypot(b[0] - a[0], b[1] - a[1])
            if d > 0.05:
                self.face(bearing(a[0], a[1], b[0], b[1]))
            speed = STAIR_SPEED if climbing else (WALK_SPEED * 0.8)
            n = max(1, int(math.ceil(d / (speed / FPS))))
            p0 = self.pitch
            p1 = target_pitch if on_flight and k < last else 0.0
            if k == last:
                p1 = 0.0
            for i in range(1, n + 1):
                f = i / n
                self.x, self.y = a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
                self.floor_z = a[2] + (b[2] - a[2]) * f
                self.pitch = p0 + (p1 - p0) * min(1.0, i / max(1, int(0.8 * FPS)))
                self.dist += d / n
                lvl = self._level_at(self.floor_z)
                self._emit(state, moving=True, level=lvl if state in ("none", "ahead_up", "ahead_down") else None)
        self.level = sw.upper if direction == "up" else sw.lower
        self.floor_z = self.b.z_of(self.level)
        self.pitch = 0.0

    def _level_at(self, z: float) -> int | None:
        for lvl in self.b.levels:
            if abs(self.b.z_of(lvl) - z) < 0.05:
                return lvl
        return None

    # ------------------------------------------------------------ scripts
    def run(self, script: list[dict]) -> None:
        for step in script:
            if "go" in step or "visit" in step:
                target = step.get("go", step.get("visit"))
                x, y = self.b.places[target].stand if isinstance(target, str) else target
                self.walk_to(x, y, step.get("speed", WALK_SPEED))
                if "visit" in step:
                    self.look(step["visit"])
            elif "look" in step:
                self.look(step["look"])
            elif "pause" in step:
                self.pause(step["pause"])
            elif "turn" in step:
                self.rotate(step["turn"])
            elif "face" in step:
                self.face(step["face"])
            elif "stairs" in step:
                self.stairs(step["stairs"], step.get("dir", "up"))
            else:
                raise ValueError(f"unknown walk step {step}")
        self.pause(0.6)  # settle before the video ends
