"""Stair landings (B6): a staircase with landings is one floor change per storey, not one per flight.

Synthetic keyframe sequences only; the landing thresholds (config.STAIR_LANDING_GAP_S / _MAX_S) still need
tuning against real walk.json files."""
import logging
from pathlib import Path

import numpy as np

from app import config, floors, mapper
from app.schemas import FrameAnalysis, Keyframe, MapBundle

UP, DOWN = "on_stairs_up", "on_stairs_down"


def _kf(t, kind="heartbeat", rooms=(), texts=(), stairs="none", place="hallway"):
    tx = list(rooms) + list(texts)
    a = FrameAnalysis(has_sign=bool(tx), texts=tx, room_numbers=list(rooms), ocr=list(tx), grounded=True,
                      stairs=stairs, place_type=place)
    return Keyframe(frame_id=f"f{t}.jpg", video_id="v", t=float(t), kind=kind, heading=0.0, analysis=a)


def _track(n=800):
    t = np.arange(n) / 10.0
    return {"t": t.tolist(), "fps": 10.0, "yaw": [0.0] * n, "fwd": [1.0] * n, "heading": [0.0] * n}


def _quiet(*_):
    pass


def _rel_at(wnodes, t):
    return next(w.rel_floor for w in wnodes if w.kf.t == t)


# Switchback staircase: two up-flights with a ~4 s landing (and its 180-degree turn) in between.
SWITCHBACK = [
    _kf(0, "start"), _kf(5, "dwell"),
    _kf(10, stairs=UP), _kf(12, stairs=UP),
    _kf(14, "turn_start"),  # the landing: state none, no room numbers, no doors
    _kf(16, stairs=UP), _kf(18, stairs=UP),
    _kf(20, "dwell"),  # top of the stairs
    _kf(26, "end"),
]


# ---------------------------------------------------------------- (1) one storey, two flights
def test_switchback_with_landing_is_one_floor():
    wn = mapper.walk_nodes(SWITCHBACK)
    assert _rel_at(wn, 20) == 1  # was 2 before the fix: one floor counted per flight
    assert all(w.kf.t != 14 for w in wn)  # the landing is mid-flight, not a floor boundary node
    assert [w.vertical for w in wn if w.kf.t in (5, 20)] == ["stairs", "stairs"]
    assert mapper.assign_floors(wn, 1, keyframes=SWITCHBACK) == [1, 1, 2, 2]


def test_switchback_full_path_one_stairs_edge(tmp_path):
    b = MapBundle(building_id="t")
    mapper.integrate_walk(b, "v", SWITCHBACK, _track(), tmp_path, None, False, log=_quiet, start_floor=1)
    assert sorted(n.floor for n in b.nodes) == [1, 1, 2, 2]
    climbs = [e for e in b.edges if e.vertical != "none"]
    assert len(climbs) == 1 and climbs[0].vertical == "stairs_up" and not climbs[0].inferred_floor_change
    # the landing photo still shows up along the stairs edge
    assert "f14.jpg" in climbs[0].frames_fwd


def test_landing_typed_as_stairwell_allows_a_longer_pause():
    kfs = [_kf(0, "start"), _kf(5, "dwell"), _kf(10, stairs=UP),
           _kf(13, "dwell", place="stairs"), _kf(17, "turn_start", place="stairs"),
           _kf(20, stairs=UP), _kf(22, "dwell"), _kf(28, "end")]
    assert config.STAIR_LANDING_GAP_S < 10 <= config.STAIR_LANDING_MAX_S
    assert _rel_at(mapper.walk_nodes(kfs), 22) == 1


def test_gap_setting_zero_disables_merging(monkeypatch):
    monkeypatch.setattr(config, "STAIR_LANDING_GAP_S", 0.0)
    monkeypatch.setattr(config, "STAIR_LANDING_MAX_S", 0.0)
    assert _rel_at(mapper.walk_nodes(SWITCHBACK), 20) == 2


# ---------------------------------------------------------------- (2) two separate climbs
def test_two_climbs_with_corridor_between_are_two_floors():
    kfs = [_kf(0, "start", rooms=["121"]), _kf(5, rooms=["123"]),
           _kf(10, stairs=UP), _kf(12, stairs=UP), _kf(14, "dwell"),
           _kf(20, rooms=["221"]), _kf(26, rooms=["223"]),  # floor-2 corridor, > STAIR_LANDING_GAP_S
           _kf(32, stairs=UP), _kf(34, stairs=UP), _kf(36, "dwell"),
           _kf(40, rooms=["321"]), _kf(45, "end", rooms=["323"])]
    wn = mapper.walk_nodes(kfs)
    assert _rel_at(wn, 26) == 1 and _rel_at(wn, 45) == 2
    f = mapper.assign_floors(wn, None, keyframes=kfs)
    assert f[0] == 1 and f[-1] == 3 and 2 in f


def test_short_gap_with_room_number_is_a_real_floor():
    # Only 4 s between flights, but the frame between shows a room plaque: a floor landing, not a mid-landing.
    kfs = [_kf(0, "start"), _kf(10, stairs=UP), _kf(12, stairs=UP),
           _kf(14, "glance", rooms=["210"], place="room"),
           _kf(16, stairs=UP), _kf(18, stairs=UP), _kf(20, "dwell"), _kf(26, "end")]
    assert _rel_at(mapper.walk_nodes(kfs), 20) == 2


def test_short_gap_with_floor_sign_is_a_real_floor():
    kfs = [_kf(0, "start"), _kf(10, stairs=UP), _kf(12, stairs=UP), _kf(14, texts=["LEVEL 2"]),
           _kf(16, stairs=UP), _kf(18, stairs=UP), _kf(20, "dwell"), _kf(26, "end")]
    assert _rel_at(mapper.walk_nodes(kfs), 20) == 2


def test_long_plain_hallway_between_climbs_is_not_a_landing():
    # 12 s of plain hallway (no signs, not a stairwell): too long for a landing without stairwell evidence
    kfs = [_kf(0, "start"), _kf(10, stairs=UP), _kf(12, stairs=UP), _kf(17), _kf(21),
           _kf(24, stairs=UP), _kf(26, "dwell"), _kf(30, "end")]
    assert _rel_at(mapper.walk_nodes(kfs), 26) == 2


# ---------------------------------------------------------------- (3) up then down
def test_up_then_down_quickly_nets_zero():
    kfs = [_kf(0, "start"), _kf(10, stairs=UP), _kf(12, stairs=UP), _kf(14, "turn_start"),
           _kf(16, stairs=DOWN), _kf(18, stairs=DOWN), _kf(20, "dwell"), _kf(26, "end")]
    wn = mapper.walk_nodes(kfs)
    assert _rel_at(wn, 14) == 1  # the turnaround point is a real (upper) floor stop, not merged away
    assert _rel_at(wn, 20) == 0 and _rel_at(wn, 26) == 0


def test_up_then_down_without_a_frame_between_nets_zero():
    # Direction flips with no 'none' frame in between: both flights still count.
    kfs = [_kf(0, "start"), _kf(10, stairs=UP), _kf(12, stairs=DOWN), _kf(14, "dwell"), _kf(20, "end")]
    assert _rel_at(mapper.walk_nodes(kfs), 14) == 0


def test_down_switchback_is_minus_one():
    kfs = [_kf(0, "start"), _kf(10, stairs=DOWN), _kf(12, stairs=DOWN), _kf(14, "turn_start"),
           _kf(16, stairs=DOWN), _kf(18, "dwell"), _kf(24, "end")]
    assert _rel_at(mapper.walk_nodes(kfs), 18) == -1


# ---------------------------------------------------------------- evidence vs counted stairs (B19)
def test_evidence_beats_counted_stairs_and_is_logged(caplog):
    # A long plain pause splits one storey into two counted flights (+2); rooms before/after say +1.
    kfs = [_kf(0, "start", rooms=["121"]), _kf(5, rooms=["123"]),
           _kf(10, stairs=UP), _kf(12, stairs=UP), _kf(17), _kf(21),
           _kf(24, stairs=UP), _kf(26, "dwell"), _kf(30, rooms=["221"]), _kf(35, "end", rooms=["223"])]
    wn = mapper.walk_nodes(kfs)
    assert _rel_at(wn, 35) == 2
    with caplog.at_level(logging.WARNING, logger=floors.__name__):
        f = mapper.assign_floors(wn, None, keyframes=kfs)
    assert f[0] == 1 and set(f) == {1, 2}  # no fake floor 3, not even for the top-of-stairs nodes
    assert any("counted +2" in r.message and "say +1" in r.message for r in caplog.records)


def test_evidence_disagreement_goes_to_given_log():
    kfs = [_kf(0, "start", rooms=["121"]), _kf(5, rooms=["123"]), _kf(10, stairs=UP), _kf(12, "dwell"),
           _kf(20, rooms=["121"]), _kf(25, "end", rooms=["125"])]  # Gemma saw stairs; rooms say same floor
    msgs = []
    f = mapper.assign_floors(mapper.walk_nodes(kfs), None, keyframes=kfs, log=msgs.append)
    assert f == [1] * len(f)  # the top of the 'stairs' too: evidence re-anchors from the crossing on
    assert len(msgs) == 1 and "counted +1" in msgs[0] and "say +0" in msgs[0]


def test_agreeing_evidence_logs_nothing(caplog):
    kfs = [_kf(0, "start", rooms=["121"]), _kf(5, rooms=["123"])] + SWITCHBACK[2:7] + [
        _kf(20, "dwell", rooms=["221"]), _kf(26, "end", rooms=["223"])]
    with caplog.at_level(logging.WARNING, logger=floors.__name__):
        f = mapper.assign_floors(mapper.walk_nodes(kfs), None, keyframes=kfs)
    assert f[0] == 1 and f[-1] == 2
    assert not caplog.records
