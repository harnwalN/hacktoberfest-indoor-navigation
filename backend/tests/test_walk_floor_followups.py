"""Follow-ups to the floor work: a walk never loop-closes onto its own earlier floor, a walked stairs edge
agrees with the floors at its ends (B19), and assign_floors warnings reach the job log."""
from pathlib import Path

import numpy as np

from app import mapper, routing
from app.schemas import FrameAnalysis, Keyframe, MapBundle

UP = "on_stairs_up"


def _kf(t, kind="heartbeat", rooms=(), stairs="none"):
    a = FrameAnalysis(has_sign=bool(rooms), texts=list(rooms), room_numbers=list(rooms), ocr=list(rooms),
                      grounded=True, stairs=stairs)
    return Keyframe(frame_id=f"f{t}.jpg", video_id="v", t=float(t), kind=kind, heading=0.0, analysis=a)


def _track(n=600):
    t = np.arange(n) / 10.0
    return {"t": t.tolist(), "fps": 10.0, "yaw": [0.0] * n, "fwd": [1.0] * n, "heading": [0.0] * n}


def _fake_orb(monkeypatch, a, b, inliers=70):
    """Frames a and b ORB-match with `inliers` (70 >= 1.5 * ORB_STRONG: auto-merge without Gemma)."""
    def fake(p, q):
        return (inliers, 0.0) if {Path(p).name, Path(q).name} == {a, b} else (0, None)
    monkeypatch.setattr(mapper, "orb_match", fake)


def _integrate(kfs, start_floor=None):
    logs: list[str] = []
    b = MapBundle(building_id="t")
    res = mapper.integrate_walk(b, "v", kfs, _track(), Path("."), None, False, log=logs.append,
                                start_floor=start_floor)
    return b, res, logs


def _node_at(b, t):
    return next(n for n in b.nodes if n.frame_id == f"f{t}.jpg")


# ---------------------------------------------------------------- 1. same-walk loop closure across floors
def _climb(rooms_before=(), rooms_after=()):
    """Floor-1 start, a flight up, then a look-alike (f45) of the start (f0), well over 15 s later."""
    return [_kf(0, "start", rooms=rooms_before[:1]), _kf(8, "dwell", rooms=rooms_before[1:]),
            _kf(15, stairs=UP), _kf(17, stairs=UP),
            _kf(25, "dwell", rooms=rooms_after[:1]), _kf(32, "dwell", rooms=rooms_after[1:]), _kf(45, "end")]


def test_walk_does_not_loop_close_onto_its_own_earlier_floor(monkeypatch):
    _fake_orb(monkeypatch, "f0.jpg", "f45.jpg")
    b, res, _ = _integrate(_climb())
    assert res["merged"] == 0
    start, end = _node_at(b, 0), _node_at(b, 45)
    assert start.id != end.id and end.floor == start.floor + 1


def test_walk_does_not_loop_close_across_floors_with_known_start_floor(monkeypatch):
    _fake_orb(monkeypatch, "f0.jpg", "f45.jpg")
    b, res, _ = _integrate(_climb(), start_floor=1)
    assert res["merged"] == 0
    assert (_node_at(b, 0).floor, _node_at(b, 45).floor) == (1, 2)


def test_same_floor_loop_closure_within_a_walk_still_merges(monkeypatch):
    _fake_orb(monkeypatch, "f0.jpg", "f45.jpg")
    kfs = [k for k in _climb() if k.analysis.stairs == "none"]  # same walk, no stairs
    b, res, _ = _integrate(kfs)
    assert res["merged"] == 1
    assert not any(n.frame_id == "f45.jpg" for n in b.nodes)  # merged into the start


def test_stairs_overruled_by_rooms_still_loop_closes(monkeypatch):
    # Gemma saw a flight, but four room numbers say the walk never left floor 1: the floor, not the stair
    # count, decides whether two places of one walk can be the same.
    _fake_orb(monkeypatch, "f0.jpg", "f45.jpg")
    b, res, _ = _integrate(_climb(("121", "123"), ("125", "127")))
    assert {n.floor for n in b.nodes} == {1}
    assert res["merged"] == 1


# ---------------------------------------------------------------- 2. stairs edge vs floor delta (B19)
def _route_text(b, a, z):
    return " ".join(s.instruction_text for s in routing.route(b, a, z).steps)


def test_same_floor_stairs_edge_keeps_stairs_but_claims_no_floor_change():
    b, _, logs = _integrate(_climb(("121", "123"), ("125", "127")))
    lo, hi = _node_at(b, 8), _node_at(b, 25)
    edge = next(e for e in b.edges if {e.from_node, e.to_node} == {lo.id, hi.id})
    assert lo.floor == hi.floor == 1
    assert edge.vertical == "stairs_up" and not edge.inferred_floor_change  # the walker did climb steps
    assert any("stay on floor 1" in m for m in logs)
    text = _route_text(b, _node_at(b, 0).id, _node_at(b, 45).id)
    assert "take the stairs up" in text and "to floor" not in text
    back = _route_text(b, _node_at(b, 45).id, _node_at(b, 0).id)
    assert "take the stairs down" in back and "to floor" not in back


def test_stairs_direction_follows_the_floors_when_they_disagree():
    # Gemma says "up", but floor-2 rooms before and floor-1 rooms after: the walker went down.
    b, _, logs = _integrate(_climb(("221", "223"), ("121", "123")))
    lo, hi = _node_at(b, 8), _node_at(b, 25)
    assert (lo.floor, hi.floor) == (2, 1)
    edge = next(e for e in b.edges if {e.from_node, e.to_node} == {lo.id, hi.id})
    assert (edge.from_node, edge.vertical) == (lo.id, "stairs_down")
    assert any("using the floor direction" in m for m in logs)
    assert "take the stairs down to floor 1" in _route_text(b, _node_at(b, 0).id, _node_at(b, 45).id)
    assert "take the stairs up to floor 2" in _route_text(b, _node_at(b, 45).id, _node_at(b, 0).id)


def test_stairs_matching_the_floors_unchanged():
    b, _, logs = _integrate(_climb(("121", "123"), ("221", "223")))
    lo, hi = _node_at(b, 8), _node_at(b, 25)
    edge = next(e for e in b.edges if {e.from_node, e.to_node} == {lo.id, hi.id})
    assert (lo.floor, hi.floor, edge.vertical) == (1, 2, "stairs_up")
    assert not any("floor direction" in m or "stay on floor" in m for m in logs)
    assert "take the stairs up to floor 2" in _route_text(b, _node_at(b, 0).id, _node_at(b, 45).id)


# ---------------------------------------------------------------- 3. assign_floors warnings reach the job log
def test_stair_count_vs_evidence_warning_reaches_the_job_log(caplog):
    with caplog.at_level("WARNING", logger="app.floors"):
        _, _, logs = _integrate(_climb(("121", "123"), ("125", "127")))
    assert any("counted +1" in m and "+0" in m for m in logs), logs
    assert not any("counted" in r.message for r in caplog.records)  # not (only) the module logger


# ---------------------------------------------------------------- probe: up and back down, revisit the start
def _round_trip(start_floor=None):
    return [_kf(0, "start", rooms=("121",)), _kf(8, "dwell", rooms=("123",)),
            _kf(15, stairs=UP), _kf(17, stairs=UP),
            _kf(25, "dwell", rooms=("221",)), _kf(32, "dwell", rooms=("223",)),
            _kf(36, stairs="on_stairs_down"), _kf(38, stairs="on_stairs_down"),
            _kf(45, "dwell", rooms=("125",)), _kf(52, "end")]


def test_up_then_down_walk_still_loop_closes_onto_the_start(monkeypatch):
    _fake_orb(monkeypatch, "f0.jpg", "f52.jpg")
    for sf in (None, 1):
        b, res, _ = _integrate(_round_trip(), start_floor=sf)
        assert _node_at(b, 0).floor == 1
        assert res["merged"] == 1, (sf, res)
        assert not any(n.frame_id == "f52.jpg" for n in b.nodes)
