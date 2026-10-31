"""Spatial awareness from the video itself (plus optional phone IMU).

Per pair of consecutive analysis frames we track ~400 corners with Lucas-Kanade optical flow and fit
a RANSAC similarity transform in image-centred coordinates:
    p1 = s * R(roll) * p0 + t
Walking forward shows up as scale (radial expansion about the centre), while turning the camera shows
up as a near-uniform horizontal shift t_x. So yaw = atan(-t_x / f). This also works for pure rotation
(turning in place at a junction), where an essential-matrix fit is degenerate. When there is too
little texture (blank walls) we fall back to FFT phase correlation.

Heading convention everywhere: degrees, clockwise positive (a right turn increases heading).
"""
import math
from pathlib import Path

import cv2
import numpy as np

from . import config
from .video import iter_frames


def focal_px(width: int, height: int) -> float:
    return (max(width, height) / 2) / math.tan(math.radians(config.LONG_SIDE_FOV_DEG / 2))


def dhash(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _pair_motion(prev: np.ndarray, cur: np.ndarray, f: float) -> tuple[float, float, float, int, float]:
    """Return (yaw_deg, pitch_deg, median_flow_px, inliers, log_scale) between two grayscale frames.

    log_scale > 0 means the scene expanded about the centre: the camera moved forward.
    """
    h, w = prev.shape
    pts = cv2.goodFeaturesToTrack(prev, maxCorners=400, qualityLevel=0.01, minDistance=8, blockSize=7)
    if pts is not None and len(pts) >= 20:
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, winSize=(21, 21), maxLevel=3)
        if nxt is not None:
            ok = st.ravel() == 1
            p0, p1 = pts[ok].reshape(-1, 2), nxt[ok].reshape(-1, 2)
            if len(p0) >= 15:
                c = np.array([w / 2, h / 2], dtype=np.float32)
                M, inl = cv2.estimateAffinePartial2D(
                    p0 - c, p1 - c, method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=500, confidence=0.99
                )
                if M is not None and inl is not None and int(inl.sum()) >= 12:
                    mask = inl.ravel() == 1
                    flow = float(np.median(np.linalg.norm(p1[mask] - p0[mask], axis=1)))
                    tx, ty = float(M[0, 2]), float(M[1, 2])
                    scale = math.log(max(1e-6, math.hypot(M[0, 0], M[1, 0])))
                    roll = math.degrees(math.atan2(M[1, 0], M[0, 0]))
                    if abs(roll) > 8:  # phone being rotated (portrait<->landscape): not a heading change
                        return 0.0, 0.0, flow, int(mask.sum()), 0.0
                    return (math.degrees(math.atan2(-tx, f)), math.degrees(math.atan2(ty, f)), flow,
                            int(mask.sum()), scale)
    (dx, dy), _ = cv2.phaseCorrelate(prev.astype(np.float32), cur.astype(np.float32))
    return math.degrees(math.atan2(-dx, f)), math.degrees(math.atan2(dy, f)), math.hypot(dx, dy), 0, 0.0


def _moving_avg(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or len(x) == 0:
        return x.copy()
    k = np.ones(win) / win
    return np.convolve(np.pad(x, (win // 2, win - 1 - win // 2), mode="edge"), k, mode="valid")


def _imu_yaw_per_frame(sensors: dict, times: np.ndarray) -> np.ndarray | None:
    """Integrate gyro rotation about the gravity axis, resampled to the analysis frame times.

    sensors = {"samples": [{"t": sec, "rr": [alpha, beta, gamma] deg/s, "g": [x, y, z] m/s^2}, ...]}
    (DeviceMotionEvent: alpha=z, beta=x, gamma=y). The sign is resolved later against CV yaw.
    """
    samples = sensors.get("samples") or []
    if len(samples) < 10:
        return None
    t = np.array([s["t"] for s in samples], dtype=float) + float(sensors.get("offset_sec", 0.0))
    rr = np.array([[s["rr"][1], s["rr"][2], s["rr"][0]] for s in samples], dtype=float)  # x, y, z
    g = np.array([s["g"] for s in samples], dtype=float)
    g_unit = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-6)
    yaw_rate = np.sum(rr * g_unit, axis=1)  # deg/s about vertical
    order = np.argsort(t)
    t, yaw_rate = t[order], yaw_rate[order]
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (yaw_rate[1:] + yaw_rate[:-1]) * np.diff(t))])
    at = np.interp(times, t, cum)
    return np.diff(at, prepend=at[0])


def compute_track(analysis_path: Path, sensors: dict | None = None, log=print) -> dict:
    fps = config.ANALYSIS_FPS
    times, yaw, pitch, flow, inliers, hashes, scale = [], [], [], [], [], [], []
    prev, f = None, None
    for _, t, frame in iter_frames(analysis_path, fps):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if f is None:
            f = focal_px(gray.shape[1], gray.shape[0])
        if prev is None:
            dyaw = dpitch = fl = sc = 0.0
            n = 0
        else:
            dyaw, dpitch, fl, n, sc = _pair_motion(prev, gray, f)
        times.append(t); yaw.append(dyaw); pitch.append(dpitch); flow.append(fl); inliers.append(n); scale.append(sc)
        hashes.append(dhash(gray))
        prev = gray
    if len(times) < 2:
        raise ValueError("video too short for motion analysis")

    t_arr, yaw_arr = np.array(times), np.array(yaw)
    # Clamp physically impossible per-frame jumps (bad fits) to keep integration sane.
    yaw_arr = np.clip(yaw_arr, -25, 25)
    source = "cv"
    if sensors:
        imu = _imu_yaw_per_frame(sensors, t_arr)
        if imu is not None:
            cv_rate, imu_rate = _moving_avg(yaw_arr, 5), _moving_avg(imu, 5)
            corr = float(np.corrcoef(cv_rate, imu_rate)[0, 1]) if np.std(imu_rate) > 1e-6 else 0.0
            if abs(corr) >= 0.2 or np.std(cv_rate) < 0.05:
                yaw_arr = imu * (1 if corr >= 0 else -1)
                source = "imu"
                log(f"Using phone gyro for heading (corr with vision {corr:+.2f})")
            else:
                log(f"Gyro disagrees with vision (corr {corr:+.2f}); falling back to vision yaw")

    heading = np.cumsum(yaw_arr)
    rate = _moving_avg(yaw_arr, fps) * fps
    fwd = _moving_avg(np.array(scale), fps) * 1000
    travel = travel_heading(t_arr, heading, rate, fwd)
    if travel is None:
        log("Little forward motion detected: using camera heading as travel direction")
    return {
        "fps": fps,
        "source": source,
        "t": t_arr.round(3).tolist(),
        "yaw": yaw_arr.round(3).tolist(),
        "heading": heading.round(2).tolist(),
        "pitch": np.cumsum(pitch).round(2).tolist(),
        "rate": rate.round(2).tolist(),
        "flow": np.round(flow, 2).tolist(),
        "inliers": inliers,
        "fwd": fwd.round(2).tolist(),  # smoothed expansion about the centre, x1000 (>0 = moving forward)
        "travel": travel.round(2).tolist() if travel is not None else None,
        "dhash": [format(h, "016x") for h in hashes],
    }


def travel_heading(t: np.ndarray, heading: np.ndarray, rate: np.ndarray, fwd: np.ndarray) -> np.ndarray | None:
    """Direction of walking, which is not the same as where the camera points.

    People swing the camera toward plaques while walking. Frames where the scene expands about the
    image centre (moving forward) and the camera is not rotating are frames where the camera looks
    where the walker goes; a running median over those samples gives the travel heading. Standing
    still and looking around contributes nothing, so it can never be mistaken for a turn.
    """
    valid = (fwd > config.FWD_WALKING) & (np.abs(rate) < config.TURN_RATE_DEG_S)
    if valid.sum() < max(10, 0.15 * len(t)):
        return None
    tv, hv = t[valid], heading[valid]
    half = config.TRAVEL_WINDOW_SEC / 2
    lo = np.searchsorted(tv, tv - half)
    hi = np.searchsorted(tv, tv + half, side="right")
    smooth = np.array([np.median(hv[a:b]) for a, b in zip(lo, hi)])
    return np.interp(t, tv, smooth)


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            segs.append((start, i - 1)); start = None
    if start is not None:
        segs.append((start, len(mask) - 1))
    return segs


def detect_events(track: dict) -> list[dict]:
    """Turns, glances (look at a plaque and back) and dwells (pauses) from the motion track."""
    events = _travel_events(track) if track.get("travel") else _camera_events(track)
    events += _dwells(track)
    events.sort(key=lambda e: e["start"])
    return events


def _turn_segments(t: np.ndarray, heading: np.ndarray, rate: np.ndarray, rate_thr: float, fps: int) -> list[dict]:
    raw = []
    for sign in (1, -1):
        for a, b in _segments(sign * rate > rate_thr):
            raw.append([a, b, sign])
    raw.sort()
    merged: list[list[int]] = []
    for a, b, s in raw:
        if merged and merged[-1][2] == s and (a - merged[-1][1]) <= int(0.5 * fps):
            merged[-1][1] = b
        else:
            merged.append([a, b, s])
    turns = []
    for a, b, _ in merged:
        delta = float(heading[b] - heading[max(a - 1, 0)])
        if abs(delta) >= config.TURN_MIN_DEG:
            turns.append({"type": "turn", "start": float(t[a]), "end": float(t[b]), "delta": round(delta, 1)})
    return turns


def _cancel_jogs(turns: list[dict]) -> list[dict]:
    """Drop opposite turn pairs that cancel out quickly: sidesteps, door swings, weaving round people."""
    out, i = [], 0
    while i < len(turns):
        cur, nxt = turns[i], turns[i + 1] if i + 1 < len(turns) else None
        if (nxt and cur["delta"] * nxt["delta"] < 0 and nxt["start"] - cur["end"] <= config.JOG_MAX_GAP_SEC
                and abs(cur["delta"] + nxt["delta"]) < config.GLANCE_RETURN_DEG):
            i += 2
            continue
        out.append(cur)
        i += 1
    return out


def _travel_events(track: dict) -> list[dict]:
    fps = track["fps"]
    t = np.array(track["t"]); heading = np.array(track["heading"]); travel = np.array(track["travel"])
    travel_rate = np.gradient(travel, t)
    events = _cancel_jogs(_turn_segments(t, travel, travel_rate, config.TRAVEL_TURN_RATE_DEG_S, fps))
    turning = np.zeros(len(t), bool)
    for e in events:
        turning |= (t >= e["start"] - 0.5) & (t <= e["end"] + 0.5)
    # Glances: camera pointing well away from the walking direction (usually at a door plaque).
    dev = heading - travel
    for sign, side in ((1, "right"), (-1, "left")):
        for a, b in _segments((sign * dev > config.GLANCE_DEG) & ~turning):
            if (b - a + 1) / fps >= 0.4:
                peak = a + int(np.argmax(sign * dev[a:b + 1]))
                events.append({"type": "glance", "start": float(t[a]), "end": float(t[b]),
                               "outer_start": float(t[a]), "outer_end": float(t[b]),
                               "peak": float(t[peak]), "side": side, "delta": round(float(dev[peak]), 1)})
    return events


def _camera_events(track: dict) -> list[dict]:
    """Fallback when there is no forward motion (e.g. panning in place): camera heading = travel."""
    fps = track["fps"]
    t = np.array(track["t"]); heading = np.array(track["heading"]); rate = np.array(track["rate"])
    turns = _turn_segments(t, heading, rate, config.TURN_RATE_DEG_S, fps)

    # --- glances: a turn quickly undone by the opposite turn -> looking at something, not a turn ---
    events, i = [], 0
    while i < len(turns):
        cur = turns[i]
        nxt = turns[i + 1] if i + 1 < len(turns) else None
        if (nxt and cur["delta"] * nxt["delta"] < 0
                and nxt["start"] - cur["end"] <= config.GLANCE_MAX_GAP_SEC
                and abs(cur["delta"] + nxt["delta"]) <= config.GLANCE_RETURN_DEG):
            events.append({"type": "glance", "start": cur["end"], "end": nxt["start"],
                           "outer_start": cur["start"], "outer_end": nxt["end"],
                           "side": "right" if cur["delta"] > 0 else "left", "delta": cur["delta"]})
            i += 2
        else:
            events.append(cur)
            i += 1
    return events


def _dwells(track: dict) -> list[dict]:
    """Pauses: camera nearly still and not rotating (the walker stopped at something)."""
    fps = track["fps"]
    t = np.array(track["t"]); rate = np.array(track["rate"]); flow = np.array(track["flow"])
    still = (flow < config.DWELL_FLOW_PX) & (np.abs(rate) < config.TURN_RATE_DEG_S / 2)
    return [{"type": "dwell", "start": float(t[a]), "end": float(t[b])}
            for a, b in _segments(still) if (b - a + 1) / fps >= config.DWELL_MIN_SEC and t[a] > 0.5]


# ---------------------------------------------------------------- stair climbs
def _vertical_motion(track: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """(t, per-frame vertical image motion in deg relative to level walking, valid mask, forward motion).

    The 'pitch' track is the cumulative vertical image shift, which mixes camera tilt with the camera rising or
    falling. Frames whose fit is unreliable (phase-correlation fallback, turning) are marked invalid: a turn's
    fit often adds a spurious vertical jump. Level walking itself shows a small steady vertical drift (uneven
    depth above and below the horizon), so the median over walking frames is subtracted as the baseline.
    """
    if not track.get("pitch") or len(track["pitch"]) < 3:
        return None
    t = np.asarray(track["t"], dtype=float)
    pitch = np.asarray(track["pitch"], dtype=float)
    n = len(t)
    dp = np.diff(pitch, prepend=pitch[0])
    valid = np.ones(n, bool)
    if track.get("inliers") is not None:
        valid &= np.asarray(track["inliers"]) > 0
    if track.get("rate") is not None:
        valid &= np.abs(np.asarray(track["rate"], dtype=float)) < config.TURN_RATE_DEG_S
    fwd = np.asarray(track["fwd"], dtype=float) if track.get("fwd") is not None else np.zeros(n)
    walking = valid & (fwd > config.FWD_WALKING)
    bias = 0.0
    if walking.sum() >= 10:
        # The median over all walking frames is pulled toward a long climb; refine it on the frames whose
        # running median is near it (the level-walking cluster), so a climb does not eat its own signal.
        fps = float(track.get("fps") or config.ANALYSIS_FPS)
        local = _running_median(dp, walking, max(1, int(config.CLIMB_WINDOW_SEC * fps / 2)))
        bias = float(np.median(dp[walking]))
        for _ in range(2):
            near = walking & (np.abs(local - bias) * fps < config.CLIMB_FLOW_DEG_S / 2)
            if near.sum() < 10:
                break
            bias = float(np.median(local[near]))
    return t, np.where(valid, dp - bias, 0.0), valid, fwd


def _running_median(x: np.ndarray, valid: np.ndarray, half: int) -> np.ndarray:
    out = np.zeros(len(x))
    for i in range(len(x)):
        w = x[max(0, i - half):i + half + 1][valid[max(0, i - half):i + half + 1]]
        out[i] = float(np.median(w)) if len(w) > half else 0.0  # mostly invalid (a turn): no evidence
    return out


def _flow_climbs(t, rel, valid, fps) -> list[dict]:
    """Sustained vertical sliding of the scene: the camera rising (scene slides down, pitch track grows) or
    falling. Only a slope seen from the side or from below shows it; descending while looking down the flight
    moves almost along the view axis and is caught by _tilt_climbs instead."""
    med = _running_median(rel, valid, max(1, int(config.CLIMB_WINDOW_SEC * fps / 2))) * fps  # deg/s
    out = []
    for sign, d in ((1, "up"), (-1, "down")):
        segs = []
        for a, b in _segments(sign * med >= config.CLIMB_FLOW_DEG_S):
            if segs and t[a] - t[segs[-1][1]] <= config.CLIMB_MERGE_GAP_SEC:
                segs[-1] = (segs[-1][0], b)  # a short dip (a landing step, a stumble) does not split the climb
            else:
                segs.append((a, b))
        for a, b in segs:
            out.append({"start": float(t[a]), "end": float(t[b]), "dir": d, "by": "flow",
                        "strength": round(float(np.mean(med[a:b + 1])), 2)})
    return out


def _tilt_climbs(t, rel, fps) -> list[dict]:
    """The camera tilts down (or up) by CLIMB_TILT_DEG within CLIMB_TILT_STEP_SEC, holds that tilt, then tilts
    back: looking down the steps while descending (or up a flight). Short looks (a sign overhead, the floor) are
    dropped later by CLIMB_MIN_SEC; the direction is only a hint (people also look down while climbing)."""
    level = np.cumsum(rel)
    k = max(1, int(config.CLIMB_TILT_STEP_SEC * fps))
    if len(level) <= k:
        return []
    diff = level[k:] - level[:-k]
    steps = []  # (sign, first index, last index, size): the frames that carry the tilt
    for sign in (1, -1):
        for a, b in _segments(sign * diff >= config.CLIMB_TILT_DEG):
            i = a + int(np.argmax(sign * diff[a:b + 1]))
            size = float(diff[i])
            moved = sign * (level[i:i + k + 1] - level[i]) / abs(size)
            s0 = i + max(int(np.argmax(moved >= 0.1)) - 1, 0)
            s1 = i + int(np.argmax(moved >= 0.9))
            steps.append((sign, s0, s1, size))
    steps.sort(key=lambda s: s[1])
    w = max(1, int(config.CLIMB_MIN_SEC * fps))  # the walker's own level: this long before / after the stretch
    out = []
    for j, (sign, a0, a1, size) in enumerate(steps):
        for sign2, b0, b1, size2 in steps[j + 1:]:
            if sign2 == sign:
                break  # tilted further the same way first: a ramp of vertical flow, not a held tilt
            if t[b0] - t[a1] < config.CLIMB_MIN_SEC:
                continue
            if abs(size + size2) > 0.5 * max(abs(size), abs(size2)):
                break  # did not come back to about the same level
            # Held away from the level before AND after the stretch. Measured from the walker's usual level, not
            # from the step: coming back from a glance up at a sign, then glancing up again later, is two
            # opposite steps with nothing held in between.
            before = float(np.median(level[max(0, a0 - w):a0])) if a0 > 0 else float(level[a0])
            after = float(np.median(level[b1 + 1:b1 + 1 + w])) if b1 + 1 < len(level) else float(level[b1])
            stretch = level[a1:b0 + 1]
            held = np.minimum(sign * (stretch - before), sign * (stretch - after)) >= 0.5 * config.CLIMB_TILT_DEG
            if np.median(held) >= 0.5:  # the tilt was held for most of the stretch
                out.append({"start": float(t[a0]), "end": float(t[b1]), "dir": "up" if sign > 0 else "down",
                            "by": "tilt", "strength": round(size, 1)})
            break
    return out


def detect_climbs(track: dict) -> list[dict]:
    """Stretches whose camera motion looks like a stair climb, as candidates: floors.mark_climbs only accepts a
    candidate that Gemma's stairs hints back up (stairs ahead / a stairwell near it).

    [{"start", "end", "dir": "up"|"down", "by": "flow"|"tilt"|"flow+tilt", "strength"}]. Overlapping flow and
    tilt stretches are one climb; its direction comes from the flow (the camera rising or falling is physical,
    a tilt is only where the walker looked). Every candidate lasts >= CLIMB_MIN_SEC with the walker moving
    forward (median forward expansion > FWD_WALKING) and at most CLIMB_MAX_SEC. Thresholds are in config.py
    (CLIMB_*); they were set on synthetic walks and need checking against real phone footage."""
    if not config.CLIMB_DETECT or (vm := _vertical_motion(track)) is None:
        return []
    t, rel, valid, fwd = vm
    fps = float(track.get("fps") or config.ANALYSIS_FPS)
    cands = sorted(_flow_climbs(t, rel, valid, fps) + _tilt_climbs(t, rel, fps), key=lambda c: c["start"])
    merged: list[dict] = []
    for c in cands:
        if merged and c["start"] <= merged[-1]["end"]:
            m = merged[-1]
            m["end"] = max(m["end"], c["end"])
            if c["by"] == "flow" and "flow" not in m["by"]:
                m["dir"], m["strength"] = c["dir"], c["strength"]
            if c["by"] not in m["by"]:
                m["by"] = "flow+tilt"
        else:
            merged.append(dict(c))
    out = []
    for c in merged:
        m = (t >= c["start"]) & (t <= c["end"])
        if not (config.CLIMB_MIN_SEC <= c["end"] - c["start"] <= config.CLIMB_MAX_SEC):
            continue
        if not m.any() or float(np.median(fwd[m])) <= config.FWD_WALKING:
            continue  # standing still (looking around, an elevator, an escalator ride): not a climb on foot
        out.append({**c, "start": round(c["start"], 2), "end": round(c["end"], 2)})
    return out


def heading_at(track: dict, ts: float) -> float:
    return float(np.interp(ts, track["t"], track["heading"]))


def pitch_at(track: dict, ts: float) -> float:
    return float(np.interp(ts, track["t"], track["pitch"]))


def median_heading(track: dict, t0: float, t1: float) -> float:
    """Median *travel* heading over [t0, t1] (falls back to camera heading)."""
    t = np.array(track["t"]); h = np.array(track.get("travel") or track["heading"])
    m = (t >= min(t0, t1)) & (t <= max(t0, t1))
    return float(np.median(h[m])) if m.any() else heading_at(track, (t0 + t1) / 2)
