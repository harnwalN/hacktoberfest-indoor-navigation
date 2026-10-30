"""Probes for B6 stair landings: multi-storey stairwell, straight flight, realistic 3 s heartbeat timing.

Tests marked xfail document known limits (strict, so they flip loudly when the limit is fixed)."""
import pytest

from app import mapper
from app.schemas import MapBundle
from tests.test_stair_landings import UP, _kf, _quiet, _track


def _two_storey(rooms):
    k = [_kf(0, "start"), _kf(5, "dwell"), _kf(10, stairs=UP), _kf(12, stairs=UP),
         _kf(14, "turn_start", place="stairs"), _kf(16, stairs=UP), _kf(18, stairs=UP),
         _kf(20, "dwell", place="stairs"),  # floor-2 landing: no sign, no door
         _kf(22, stairs=UP), _kf(24, stairs=UP), _kf(26, "turn_start", place="stairs"),
         _kf(28, stairs=UP), _kf(30, stairs=UP), _kf(32, "dwell")]
    return k + ([_kf(34, "dwell", rooms=["301"]), _kf(36, rooms=["303"]), _kf(40, "end")] if rooms
                else [_kf(40, "end")])


def test_two_storeys_in_one_stairwell_with_rooms_after_arrival_lands_on_floor_3(caplog):
    kfs = _two_storey(True)
    wn = mapper.walk_nodes(kfs)
    with caplog.at_level("WARNING", logger="app.floors"):
        floors = mapper.assign_floors(wn, 1, keyframes=kfs)
    assert floors == [1, 1, 3, 3, 3, 3]  # old code: [1,1,2,3,4,3,3,3,3] with a phantom floor 4
    assert any("counted +1" in r.message and "+2" in r.message for r in caplog.records)


def test_two_storeys_rooms_after_arrival_one_stairs_edge_floor_1_to_3(tmp_path):
    b = MapBundle(building_id="t")
    mapper.integrate_walk(b, "v", _two_storey(True), _track(), tmp_path, None, False, log=_quiet, start_floor=1)
    assert max(n.floor for n in b.nodes) == 3 and min(n.floor for n in b.nodes) == 1
    fl = {n.id: n.floor for n in b.nodes}
    climbs = [e for e in b.edges if e.vertical != "none"]
    assert len(climbs) == 1
    # Documents B19: the single edge spans two floors (1 -> 3), no floor-2 node exists.
    assert fl[climbs[0].to_node] - fl[climbs[0].from_node] == 2


@pytest.mark.xfail(strict=True, reason="known limit: floor landings with no sign inside one stairwell are "
                                       "joined into one climb; with no later evidence the top is floor 2, not 3")
def test_two_storeys_no_rooms_after_arrival_reaches_floor_3():
    kfs = _two_storey(False)
    assert mapper.assign_floors(mapper.walk_nodes(kfs), 1, keyframes=kfs)[-1] == 3


def test_two_storeys_no_rooms_is_silently_floor_2_but_not_worse_than_old():
    kfs = _two_storey(False)
    new = mapper.assign_floors(mapper.walk_nodes(kfs), 1, keyframes=kfs)
    assert new == [1, 1, 2, 2]  # old code gave [1,1,2,3,4,5,5]: top at floor 5 (3 storeys too many)


def test_single_straight_flight_unchanged():
    kfs = [_kf(0, "start"), _kf(5, "dwell"), _kf(10, stairs=UP), _kf(12, stairs=UP), _kf(14, "dwell"),
           _kf(20, "end")]
    wn = mapper.walk_nodes(kfs)
    assert [(w.kf.t, w.rel_floor) for w in wn] == [(0.0, 0), (5.0, 0), (14.0, 1), (20.0, 1)]


def _gapped(gap, place):
    """Heartbeat 3 s: flights sampled every 3 s, with the landing taking `gap` s between on-stairs frames."""
    ks = [_kf(0, "start"), _kf(6, "dwell"), _kf(9, stairs=UP), _kf(12, stairs=UP)]
    n = int(gap // 3) - 1  # heartbeat frames that land inside the gap
    ks += [_kf(12 + 3 * (i + 1), "heartbeat", place=place) for i in range(n)]
    ks += [_kf(12 + gap, stairs=UP), _kf(15 + gap, stairs=UP), _kf(18 + gap, "dwell"), _kf(30 + gap, "end")]
    return ks


@pytest.mark.parametrize("gap,place,joined", [
    (6, "hallway", True),    # one heartbeat frame on the landing
    (9, "hallway", True),    # two landing frames, typed hallway: joins at the 9 s default (gap <= STAIR_LANDING_GAP_S)
    (12, "hallway", False),  # three landing frames, typed hallway: 12 s is past the 9 s default (documented limit)
    (9, "stairs", True),
    (12, "stairs", True),
    (18, "stairs", False),
])
def test_heartbeat_3s_landing_timing(gap, place, joined):
    wn = mapper.walk_nodes(_gapped(gap, place))
    assert max(w.rel_floor for w in wn) == (1 if joined else 2)
