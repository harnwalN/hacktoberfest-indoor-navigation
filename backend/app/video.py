"""ffmpeg normalization + frame iteration.

Any input (iPhone HEVC .MOV, Android .mp4, WebM from the browser recorder...) is converted into two
constant-frame-rate H.264 files with rotation metadata already applied:
  analysis.mp4  small, 10 fps   -> optical flow / motion track
  hq.mp4        <=1280px, 15 fps -> keyframe selection and export
"""
import json
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from . import config


class VideoError(RuntimeError):
    pass


def _scale_filter(long_side: int) -> str:
    # Fit the long side, keep aspect, even dimensions. Works for portrait and landscape.
    return (
        f"scale='if(gt(iw,ih),min({long_side},iw),-2)':'if(gt(iw,ih),-2,min({long_side},ih))'"
    )


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise VideoError(f"{cmd[0]} failed: {proc.stderr[-800:]}")


def probe_duration(path: Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(json.loads(proc.stdout)["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return 0.0


def normalize(src: Path, out_dir: Path) -> tuple[Path, Path, float]:
    if not shutil.which("ffmpeg"):
        raise VideoError("ffmpeg not found on PATH (brew install ffmpeg)")
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis, hq = out_dir / "analysis.mp4", out_dir / "hq.mp4"
    common = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p"]
    _run(common + ["-vf", f"{_scale_filter(config.ANALYSIS_LONG_SIDE)},fps={config.ANALYSIS_FPS}",
                   "-preset", "veryfast", "-crf", "23", str(analysis)])
    _run(common + ["-vf", f"{_scale_filter(config.HQ_LONG_SIDE)},fps={config.HQ_FPS}",
                   "-preset", "veryfast", "-crf", "18", str(hq)])
    duration = probe_duration(hq) or probe_duration(src)
    if duration <= 0:
        raise VideoError("could not read any video frames")
    return analysis, hq, duration


def iter_frames(path: Path, fps: float) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (index, time_sec, BGR frame). Files are CFR, so t = index / fps."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError(f"OpenCV cannot open {path}")
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield i, i / fps, frame
            i += 1
    finally:
        cap.release()


def grab_frames(path: Path, indices: set[int]) -> dict[int, np.ndarray]:
    """Sequentially decode once and keep only the requested frame indices."""
    wanted, out = set(indices), {}
    if not wanted:
        return out
    last = max(wanted)
    for i, _, frame in iter_frames(path, 1.0):
        if i in wanted:
            out[i] = frame
        if i >= last:
            break
    return out
