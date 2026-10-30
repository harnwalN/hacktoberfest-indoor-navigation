"""Stair climbs from camera motion (motion.detect_climbs) gated by Gemma's stairs hints (floors.mark_climbs),
and floor signs read just before a climb (the next floor's stairwell sign seen from the bottom of the flight).

Tracks are built by hand in the shape motion.compute_track returns (10 fps, cumulative 'pitch' in degrees,
'rate' deg/s, 'fwd' expansion x1000, 'inliers'), with noise, so the tests do not depend on rendered video."""
import numpy as np

from app import config, floors, mapper, motion
from app.schemas import FrameAnalysis, Keyframe, MapBundle

FPS = 10


def _track(dur, pitch_rate=None, tilt=None, rate=None, fwd=None, seed=0):
    """pitch_rate(t) deg/s of vertical image motion; tilt(t) deg of held camera tilt (added to the pitch);
    rate(t) yaw rate deg/s; fwd(t) forward expansion (walking > config.FWD_WALKING)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * FPS)) / FPS
    dp = np.array([pitch_rate(x) if pitch_rate else 0.0 for x in t]) / FPS
    dp += 0.02 + rng.normal(0, 0.15, len(t))  # level-walk drift + per-frame fit noise
    pitch = np.cumsum(dp) + np.array([tilt(x) if tilt else 0.0 for x in t])
    r = np.array([rate(x) if rate else 0.0 for x in t]) + rng.normal(0, 1.0, len(t))
    f = np.array([fwd(x) if fwd else 20.0 for x in t])
    inl = np.full(len(t), 120)
    return {"t": t.round(3).tolist(), "fps": FPS, "pitch": pitch.round(2).tolist(), "rate": r.round(2).tolist(),
            "fwd": f.round(2).tolist(), "inliers": inl.tolist(), "yaw": (r / FPS).tolist(),
            "heading": np.cumsum(r / FPS).tolist(), "flow": [3.0] * len(t)}


def _ramp(t0, t1, v):
    return lambda x: v if t0 <= x <= t1 else 0.0


def _held(t0, t1, deg, ease=0.8):
    def f(x):
        if x < t0 or x > t1 + ease:
            return 0.0
        if x < t0 + ease:
            return deg * (x - t0) / ease
        if x > t1:
            return deg * (1 - (x - t1) / ease)
        return deg
    return f


# ---------------------------------------------------------------- motion.detect_climbs
def test_climb_up_from_vertical_flow():
    climbs = motion.detect_climbs(_track(40, pitch_rate=_ramp(12, 26, 4.0)))
    assert len(climbs) == 1, climbs
    c = climbs[0]
    assert c["dir"] == "up" and "flow" in c["by"]
    assert 10.5 <= c["start"] <= 14 and 24 <= c["end"] <= 28


def test_climb_down_from_held_tilt():
    climbs = motion.detect_climbs(_track(40, tilt=_held(12, 25, -22.0)))
    assert len(climbs) == 1, climbs
    assert climbs[0]["dir"] == "down" and climbs[0]["by"] == "tilt"
    assert 11 <= climbs[0]["start"] <= 13 and 24.5 <= climbs[0]["end"] <= 27


def test_flat_walk_with_turns_and_glances_has_no_climb():
    # Two 90-degree turns, glances up at high signs (20 deg, 1.5 s), a glance down at a plaque, steady drift.
    tilt = (lambda x: _held(6, 7.5, 20.0)(x) + _held(18, 19.5, 18.0)(x) + _held(30, 32, -16.0)(x))
    rate = (lambda x: 45.0 if 10 <= x <= 12 or 24 <= x <= 26 else 0.0)
    assert motion.detect_climbs(_track(45, tilt=tilt, rate=rate, pitch_rate=lambda x: 0.6)) == []


def test_turn_fits_with_vertical_jumps_are_ignored():
    # A turn's fit often adds a vertical jump: the frames are invalid (rate above TURN_RATE_DEG_S).
    tr = _track(30, rate=lambda x: 60.0 if 10 <= x <= 16 else 0.0)
    pitch = np.array(tr["pitch"])
    t = np.array(tr["t"])
    pitch[(t >= 10) & (t <= 16)] += np.linspace(0, 30, int(((t >= 10) & (t <= 16)).sum()))
    pitch[t > 16] += 30
    tr["pitch"] = pitch.tolist()
    assert motion.detect_climbs(tr) == []


def test_standing_and_looking_down_is_not_a_climb():
    # Looking at the floor / a phone for 8 s while standing (elevator, escalator, reading a map).
    tr = _track(30, tilt=_held(10, 18, -25.0), fwd=lambda x: 2.0 if 9 <= x <= 20 else 20.0)
    assert motion.detect_climbs(tr) == []


def test_look_up_at_a_high_sign_is_not_a_climb():
    # Stopping to read an EXIT sign above a door: 25 deg up for 1.5 s, while the sign view slides vertically.
    assert motion.detect_climbs(_track(30, tilt=_held(10, 11.5, 25.0), pitch_rate=_ramp(9.5, 12, 4.0))) == []


def test_short_look_down_while_walking_is_not_a_climb():
    assert motion.detect_climbs(_track(30, tilt=_held(10, 12, -25.0))) == []


def test_no_pitch_track_or_disabled(monkeypatch):
    tr = _track(40, pitch_rate=_ramp(12, 26, 4.0))
    assert motion.detect_climbs({k: v for k, v in tr.items() if k != "pitch"}) == []
    monkeypatch.setattr(config, "CLIMB_DETECT", False)
    assert motion.detect_climbs(tr) == []


# ---------------------------------------------------------------- floors.mark_climbs
def _kf(t, kind="heartbeat", rooms=(), texts=(), stairs="none", place="hallway"):
    tx = list(rooms) + list(texts)
    a = FrameAnalysis(has_sign=bool(tx), texts=tx, room_numbers=list(rooms), ocr=list(tx), grounded=True,
                      stairs=stairs, place_type=place)
    return Keyframe(frame_id=f"f{t}.jpg", video_id="v", t=float(t), kind=kind, heading=0.0, analysis=a)


CLIMB = {"start": 23.8, "end": 36.5, "dir": "up", "by": "flow", "strength": 3.5}


def _up_walk():
    """The live two_floor failure: Gemma says 'stairs ahead' on every frame of the flight, never 'on the
    stairs', and reads 'LEVEL 2' from the bottom of the flight on floor 1."""
    return [
        _kf(0, "start"), _kf(2.8, "dwell", rooms=["110"], place="room"), _kf(11.7, rooms=["114"], place="room"),
        _kf(14.6, "glance", rooms=["114"], place="room"),
        _kf(22.3, "turn_start", texts=["LEVEL 2"], stairs="ahead_up", place="stairs"),
        _kf(23.2, texts=["LEVEL 2"], stairs="ahead_up", place="stairs"),
        _kf(25.8, texts=["LEVEL 2"], stairs="ahead_up", place="stairs"),
        _kf(29.5, texts=["LEVEL 2"], stairs="ahead_up", place="stairs"),
        _kf(31.8, texts=["LEVEL 2"], place="entrance"), _kf(34.5, texts=["LEVEL 2"]),
        _kf(37.0, texts=["LEVEL 2"]), _kf(38.9, "turn_start", texts=["LEVEL 2"], place="entrance"),
        _kf(45.1, "glance", rooms=["214A"], place="room"), _kf(51.0, "dwell", rooms=["212"], place="room"),
        _kf(57.9, "end", rooms=["210"], place="room"),
    ]


def _quiet(*_):
    pass


def test_motion_climb_with_gemma_stairs_ahead_counts_one_floor():
    kfs = floors.mark_climbs(_up_walk(), [CLIMB], log=_quiet)
    assert [k.analysis.stairs for k in kfs if 23.8 <= k.t <= 36.5] == ["on_stairs_up"] * 4
    assert _up_walk()[6].analysis.stairs == "ahead_up"  # the caller's keyframes are not modified
    wn = mapper.walk_nodes(kfs)
    assert all(not (23.8 <= w.kf.t <= 36.5) for w in wn)  # no place in the middle of the flight
    bottom = max((w for w in wn if w.kf.t < 23.8), key=lambda w: w.kf.t)
    top = min((w for w in wn if w.kf.t > 36.5), key=lambda w: w.kf.t)
    assert (bottom.vertical, top.vertical) == ("stairs", "stairs")
    assert (bottom.rel_floor, top.rel_floor) == (0, 1)
    fl = mapper.assign_floors(wn, 1, keyframes=kfs, log=_quiet)
    assert [f for w, f in zip(wn, fl) if w.kf.t < 23.8] == [1] * sum(w.kf.t < 23.8 for w in wn)
    assert [f for w, f in zip(wn, fl) if w.kf.t > 36.5] == [2] * sum(w.kf.t > 36.5 for w in wn)


def test_level_sign_before_climb_does_not_move_the_bottom_without_start_floor():
    kfs = floors.mark_climbs(_up_walk(), [CLIMB], log=_quiet)
    wn = mapper.walk_nodes(kfs)
    fl = mapper.assign_floors(wn, None, keyframes=kfs, log=_quiet)
    assert {f for w, f in zip(wn, fl) if w.kf.t < 23.8} == {1}  # rooms 110/114, not the LEVEL 2 sign
    assert {f for w, f in zip(wn, fl) if w.kf.t > 36.5} == {2}


def test_motion_climb_without_gemma_stairs_is_not_counted():
    flat = [_kf(t, rooms=["11" + str(i)], place="room") if i % 2 else _kf(t) for i, t in enumerate(range(0, 50, 3))]
    logs = []
    out = floors.mark_climbs(flat, [CLIMB], log=logs.append)
    assert out is flat and "Gemma saw no stairs" in logs[0]
    assert {w.rel_floor for w in mapper.walk_nodes(out)} == {0}


def test_door_inside_a_motion_climb_drops_it_and_doors_at_its_ends_are_trimmed():
    kfs = [_kf(20, stairs="ahead_up", place="stairs"), _kf(24, rooms=["114"], place="room"),
           _kf(27, stairs="ahead_up"), _kf(30), _kf(36, place="exit")]
    out = floors.mark_climbs(kfs, [CLIMB], log=_quiet)
    assert [k.analysis.stairs for k in out] == ["ahead_up", "none", "on_stairs_up", "on_stairs_up", "none"]
    mid = kfs[:2] + [_kf(27, stairs="ahead_up"), _kf(30, rooms=["120"], place="room"), _kf(33), _kf(36)]
    assert floors.mark_climbs(mid, [CLIMB], log=_quiet) is mid


def test_climb_direction_rules():
    tilt_down = {**CLIMB, "dir": "down", "by": "tilt"}
    hint_up = [_kf(22, stairs="ahead_up"), _kf(26), _kf(30)]
    assert floors.mark_climbs(hint_up, [tilt_down], log=_quiet)[1].analysis.stairs == "on_stairs_up"  # Gemma > tilt
    tie = [_kf(22, stairs="ahead_up"), _kf(26, stairs="ahead_down"), _kf(30)]
    assert floors.mark_climbs(tie, [tilt_down], log=_quiet)[2].analysis.stairs == "on_stairs_down"  # then the tilt
    flow_up = [_kf(22, stairs="ahead_down"), _kf(26), _kf(30)]
    assert floors.mark_climbs(flow_up, [CLIMB], log=_quiet)[1].analysis.stairs == "on_stairs_up"  # flow > ahead
    gemma_on = [_kf(22, stairs="ahead_up"), _kf(26, stairs="on_stairs_down"), _kf(30)]
    assert floors.mark_climbs(gemma_on, [CLIMB], log=_quiet)[2].analysis.stairs == "on_stairs_down"  # on_stairs > flow


def test_sign_before_a_gemma_flight_is_ignored_too():
    kfs = [_kf(0, "start", rooms=["110"], place="room"), _kf(3, rooms=["112"], place="room"),
           _kf(8, "dwell", texts=["FLOOR 2"]), _kf(10, stairs="on_stairs_up"), _kf(12, stairs="on_stairs_up"),
           _kf(15, "dwell", rooms=["210"], place="room"), _kf(18, "end", rooms=["212"], place="room")]
    wn = mapper.walk_nodes(kfs)
    assert mapper.assign_floors(wn, None, keyframes=kfs, log=_quiet) == [1, 1, 1, 2, 2]


def test_approach_window_can_be_disabled(monkeypatch):
    kfs = floors.mark_climbs(_up_walk(), [CLIMB], log=_quiet)
    wn = mapper.walk_nodes(kfs)
    i = max(range(len(wn)), key=lambda i: wn[i].kf.t if wn[i].kf.t < 23.8 else -1)  # bottom of the stairs
    assert floors._signs_from_keyframes(wn, kfs)[i] is None  # 'LEVEL 2' read from the bottom: no evidence
    monkeypatch.setattr(config, "CLIMB_SIGN_APPROACH_SEC", 0.0)
    assert floors._signs_from_keyframes(wn, kfs)[i] == (2, config.FLOOR_SIGN_WEIGHT)  # the old reading


# ---------------------------------------------------------------- integrate_walk end to end
def test_integrate_walk_links_stairs_bottom_and_top(tmp_path):
    tr = _track(60, pitch_rate=_ramp(24, 36.5, 4.0))
    b = MapBundle(building_id="t")
    mapper.integrate_walk(b, "v", _up_walk(), tr, tmp_path, None, False, log=_quiet, start_floor=1)
    climbs = [e for e in b.edges if e.vertical != "none"]
    assert len(climbs) == 1 and climbs[0].vertical == "stairs_up" and not climbs[0].inferred_floor_change
    lo, hi = (next(n for n in b.nodes if n.id == x) for x in (climbs[0].from_node, climbs[0].to_node))
    assert (lo.floor, hi.floor) == (1, 2)
    assert lo.t is not None and 22 <= lo.t < 24 and 36.5 < hi.t <= 39  # at the bottom and top, not Room 114
    assert all(n.floor == 1 for n in b.nodes if any("11" in x for x in n.texts if x[:1].isdigit()))
