"""Floor evidence through the full integrate_walk path, plus realistic walk sequences."""
from pathlib import Path

import numpy as np

from app import mapper
from app.schemas import FrameAnalysis, Keyframe, MapBundle


def _kf(t, kind="scene", rooms=(), texts=(), hint="", stairs="none", place_type="hallway"):
    tx = list(rooms) + list(texts)
    a = FrameAnalysis(has_sign=bool(tx), texts=tx, room_numbers=list(rooms), floor_hint=hint, ocr=list(tx),
                      grounded=True, stairs=stairs, place_type=place_type)
    return Keyframe(frame_id=f"f{int(t)}.jpg", video_id="v", t=float(t), kind=kind, heading=0.0, analysis=a)


def _track(n=400):
    t = np.arange(n) / 10.0
    return {"t": t.tolist(), "fps": 10.0, "yaw": [0.0] * n, "fwd": [1.0] * n, "heading": [0.0] * n}


def _run(kfs, start_floor=None):
    bundle = MapBundle(building_id="t")
    try:
        mapper.integrate_walk(bundle, "v", kfs, _track(), Path("."), None, False, log=lambda *_: None,
                              start_floor=start_floor)
    except Exception as e:  # surface harness problems clearly
        raise AssertionError(f"integrate_walk failed: {e!r}")
    return bundle


def test_b5a_full_path():
    kfs = [_kf(0, "start"), _kf(10, "dwell"), _kf(20, rooms=["321"]), _kf(30, rooms=["121"]),
           _kf(40, rooms=["123"]), _kf(50, "end")]
    b = _run(kfs)
    assert {n.floor for n in b.nodes} == {1}


def test_b5b_full_path_no_fake_floor_no_stairs_edge():
    kfs = [_kf(0, "start", rooms=["121"]), _kf(10, rooms=["123"]), _kf(20, rooms=["321"])]
    b = _run(kfs)
    assert sorted(n.floor for n in b.nodes if n.texts) == [1, 1, 1]
    assert {n.floor for n in b.nodes} == {1}
    assert not [e for e in b.edges if e.inferred_floor_change or e.vertical != "none"]


def test_stairs_up_then_rooms_full_path():
    kfs = [_kf(0, "start", rooms=["121"]), _kf(10, rooms=["123"]), _kf(20, stairs="on_stairs_up"),
           _kf(25, stairs="on_stairs_up"), _kf(30, rooms=["221"]), _kf(40, rooms=["223"])]
    b = _run(kfs)
    assert sorted({n.floor for n in b.nodes}) == [1, 2]


# ---- realistic probes on assign_floors
def _wns(kfs):
    return mapper.walk_nodes(kfs)


def test_probe_stairs_then_single_misread_then_rooms():
    kfs = [_kf(0, "start", rooms=["101"]), _kf(10, rooms=["103"]), _kf(20, stairs="on_stairs_up"),
           _kf(25, stairs="on_stairs_up"), _kf(30, rooms=["777"]), _kf(40, rooms=["205"]), _kf(50, rooms=["207"])]
    w = _wns(kfs)
    f = mapper.assign_floors(w, None, keyframes=kfs)
    assert f[0] == 1 and f[-1] == 2 and f[-2] == 2
    assert all(x is not None and x <= 2 for x in f)


def test_probe_signless_walk_with_one_floor_sign():
    kfs = [_kf(0, "start"), _kf(10), _kf(20, texts=["FLOOR 2"]), _kf(30), _kf(40, "end")]
    w = _wns(kfs)
    assert set(mapper.assign_floors(w, None, keyframes=kfs)) == {2}


def test_probe_down_stairs_from_floor_one():
    kfs = [_kf(0, "start", rooms=["121"]), _kf(10, rooms=["123"]), _kf(20, stairs="on_stairs_down"),
           _kf(25, stairs="on_stairs_down"), _kf(30, rooms=["B12"]), _kf(40)]
    w = _wns(kfs)
    f = mapper.assign_floors(w, None, keyframes=kfs)
    print("DOWN", f)
    assert f[0] == 1  # later nodes: floor 0 expected (basements are not modelled), documented not asserted
