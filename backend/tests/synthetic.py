"""Synthetic walkthrough videos with known camera yaw, for testing the motion pipeline.

The "building" is a cylindrical 360-degree panorama of random texture: heading h shows the crop at
x = h_rad * F (mod the panorama period), so turning a full circle returns to an identical view.
"""
import math
from pathlib import Path

import cv2
import numpy as np

W, H, SRC_FPS = 640, 360, 30
F = (W / 2) / math.tan(math.radians(69.0 / 2))
PERIOD = int(round(2 * math.pi * F))


def panorama(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = (rng.random((H, PERIOD)) * 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 3)
    img = cv2.cvtColor(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX), cv2.COLOR_GRAY2BGR)
    for _ in range(PERIOD // 30):
        x, y = int(rng.integers(0, PERIOD - 60)), int(rng.integers(0, H - 60))
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.rectangle(img, (x, y), (x + int(rng.integers(10, 60)), y + int(rng.integers(10, 60))), color, -1)
    return np.concatenate([img, img], axis=1)  # tiled so any crop wraps around


def write_video(path: Path, segments: list[tuple[float, float]], seed: int = 0) -> list[float]:
    """segments: (duration_sec, yaw_rate_deg_per_sec). Returns the true heading per source frame."""
    pano = panorama(seed)
    heading, headings = 0.0, []
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), SRC_FPS, (W, H))
    for dur, rate in segments:
        for _ in range(int(dur * SRC_FPS)):
            heading += rate / SRC_FPS
            x = int(round(math.radians(heading) * F)) % PERIOD  # turning right slides the view right
            out.write(np.ascontiguousarray(pano[:, x:x + W]))
            headings.append(heading)
    out.release()
    return headings
