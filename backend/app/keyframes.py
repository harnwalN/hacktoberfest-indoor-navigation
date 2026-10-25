"""Event-driven, adaptive keyframe selection.

Instead of sampling every N seconds and throwing away blurry frames (which can silently lose the one
frame that shows a door plaque), we:
  1. decide *when* we need a photo: walk start/end, before and after each turn, glances, dwells
     (pauses), scene changes, plus a heartbeat on long straights;
  2. in a +-0.5 s window around each moment, pick the frame with the best
     score = (Laplacian sharpness / rolling median sharpness of this video) / (1 + motion);
     normalizing by the video's own median makes the threshold resolution- and scene-independent;
  3. always keep the best frame (flagged `soft` if still blurry) so no node is ever lost;
  4. drop near-duplicates by perceptual hash (except structural moments like turns).
"""
from pathlib import Path

import cv2
import numpy as np

from . import config
from .motion import dhash, hamming, heading_at, pitch_at
from .schemas import Keyframe
from .video import grab_frames, iter_frames

PRIORITY = {"start": 0, "end": 0, "turn_start": 1, "glance": 2, "dwell": 3, "turn_end": 4, "scene": 5, "heartbeat": 6}
STRUCTURAL = {"start", "end", "turn_start", "turn_end", "glance"}


def _sharpness(frame: np.ndarray) -> tuple[float, int]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = 640 / max(h, w)
    if scale < 1:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()), dhash(gray)


def _rolling_median(x: np.ndarray, half: int) -> np.ndarray:
    return np.array([np.median(x[max(0, i - half): i + half + 1]) for i in range(len(x))])


def _candidates(track: dict, events: list[dict], duration: float) -> list[dict]:
    c = [
        {"kind": "start", "t": 0.3, "lo": 0.0, "hi": min(1.5, duration)},
        {"kind": "end", "t": duration - 0.3, "lo": max(0.0, duration - 1.5), "hi": duration},
    ]
    for e in events:
        if e["type"] == "turn":
            # approach view (what you see as you reach the junction) and departure view
            c.append({"kind": "turn_start", "t": e["start"], "lo": e["start"] - 0.8, "hi": e["start"] + 0.2, "event": e})
            c.append({"kind": "turn_end", "t": e["end"], "lo": e["end"], "hi": e["end"] + 1.0, "event": e})
        elif e["type"] == "glance" and "peak" in e:
            # the moment the camera points furthest toward the plaque, +-0.6 s for sharpness
            lo, hi = max(e["start"], e["peak"] - 0.6), min(e["end"], e["peak"] + 0.6)
            c.append({"kind": "glance", "t": e["peak"], "lo": lo, "hi": hi, "event": e})
        elif e["type"] in ("glance", "dwell"):
            c.append({"kind": e["type"], "t": (e["start"] + e["end"]) / 2, "lo": e["start"], "hi": e["end"], "event": e})

    # Sweep the dense track for scene changes and heartbeat gaps.
    times, hashes = track["t"], [int(h, 16) for h in track["dhash"]]
    fixed = sorted(x["t"] for x in c)
    last_t, last_h, fi = 0.0, hashes[0], 0
    w = config.CANDIDATE_WINDOW_SEC
    for t, h in zip(times, hashes):
        while fi < len(fixed) and fixed[fi] <= t:
            last_t, last_h, fi = fixed[fi], h, fi + 1
        if t - last_t < config.MIN_CANDIDATE_GAP_SEC:
            continue
        if hamming(h, last_h) > config.SCENE_CHANGE_HAMMING:
            c.append({"kind": "scene", "t": t, "lo": t - w, "hi": t + w}); last_t, last_h = t, h
        elif t - last_t >= config.HEARTBEAT_SEC:
            c.append({"kind": "heartbeat", "t": t, "lo": t - w, "hi": t + w}); last_t, last_h = t, h

    # Collapse candidates that are too close together, keeping the higher-priority one.
    c.sort(key=lambda x: x["t"])
    out: list[dict] = []
    for x in c:
        x["lo"], x["hi"] = max(0.0, x["lo"]), min(duration, x["hi"])
        if out and x["t"] - out[-1]["t"] < 0.6:
            if PRIORITY[x["kind"]] < PRIORITY[out[-1]["kind"]]:
                out[-1] = x
            continue
        out.append(x)
    return out


def select_keyframes(
    hq_path: Path, track: dict, events: list[dict], duration: float, video_id: str, frames_out: Path, log=print
) -> list[Keyframe]:
    fps = config.HQ_FPS
    sharp, hashes = [], []
    for _, _, frame in iter_frames(hq_path, fps):
        s, h = _sharpness(frame)
        sharp.append(s); hashes.append(h)
    n = len(sharp)
    if n == 0:
        return []
    sharp_arr = np.array(sharp)
    norm = sharp_arr / np.maximum(_rolling_median(sharp_arr, int(5 * fps)), 1e-6)
    t_hq = np.arange(n) / fps
    flow = np.interp(t_hq, track["t"], track["flow"])
    score = norm / (1 + flow / 5.0)

    chosen: list[tuple[int, dict]] = []
    used: set[int] = set()
    prev_hash: int | None = None
    for cand in _candidates(track, events, duration):
        lo, hi = int(cand["lo"] * fps), min(n - 1, int(np.ceil(cand["hi"] * fps)))
        if hi < lo:
            continue
        idx = lo + int(np.argmax(score[lo: hi + 1]))
        if idx in used:
            continue
        if (prev_hash is not None and cand["kind"] not in STRUCTURAL
                and hamming(hashes[idx], prev_hash) <= config.DUPLICATE_HAMMING):
            continue
        used.add(idx); prev_hash = hashes[idx]
        chosen.append((idx, cand))

    images = grab_frames(hq_path, {i for i, _ in chosen})
    frames_out.mkdir(parents=True, exist_ok=True)
    keyframes = []
    for k, (idx, cand) in enumerate(chosen):
        frame_id = f"{video_id}_{k:04d}.jpg"
        cv2.imwrite(str(frames_out / frame_id), images[idx], [cv2.IMWRITE_JPEG_QUALITY, 88])
        t = idx / fps
        ev = cand.get("event") or {}
        t_arrive, t_depart = t, t
        if cand["kind"] == "turn_start":
            t_arrive, t_depart = ev["start"], ev["end"]
        elif cand["kind"] == "glance":
            t_arrive, t_depart = ev["outer_start"], ev["outer_end"]
        keyframes.append(Keyframe(
            frame_id=frame_id, video_id=video_id, t=round(t, 2), kind=cand["kind"],
            t_arrive=round(t_arrive, 2), t_depart=round(t_depart, 2),
            heading=round(heading_at(track, t), 1), pitch=round(pitch_at(track, t), 1),
            score=round(float(norm[idx]), 2), soft=bool(norm[idx] < config.SOFT_SCORE),
            dhash=format(hashes[idx], "016x"),
        ))
    soft = sum(k.soft for k in keyframes)
    log(f"Selected {len(keyframes)} keyframes from {n} frames ({soft} soft)")
    return keyframes
