"""Adversarial probes of the twin-evidence merge (mapper._twin_candidates), written by sentinel-e2e.

Passing tests pin behaviour that holds under noise. Tests marked xfail(strict=True) document confirmed
over-merges / misses: when forge-backend fixes one the strict xfail fails and the marker should be removed.
Plan: corridor along +x (heading 90), exit doors on the north (fh 0) / south (fh 180) walls."""
import random
from pathlib import Path

import pytest

from app import mapper, routing
from tests.test_key_places import _a, _bundle, _edge, _node
from tests.test_twin_evidence import NO_PHOTOS, _pairs, _seen, quiet


def _two_spots(sep, fh_a, fh_b, come_a="W", come_b="E", nz=0.0, rng=None):
    """Walk A reaches a spot at x=6 (from the west), walk B one `sep` m further east (from `come_b`); each photographs
    an exit door. Anchor rooms at x=0 and x=14 are joined by walk A."""
    rng = rng or random.Random(0)
    g = lambda: rng.gauss(0, nz) if nz else 0.0  # noqa: E731
    nodes = [_node("r", "Room 120", "vA", "room"), _node("rE", "Room 130", "vA", "room", t=5)]
    edges = [_edge("e0", "r", "rE", 90, 14.0 + g(), "vA")]
    for nid, vid, x, fh, come in (("a", "vA", 6.0, fh_a, come_a), ("b", "vB", 6.0 + sep, fh_b, come_b)):
        nodes.append(_seen(nid, "Exit", vid, "exit", fh, t=20))
        edges.append(_edge("e" + nid, "r", nid, 90, x + g(), vid) if come == "W"
                     else _edge("e" + nid, "rE", nid, 270, 14.0 - x + g(), vid))
    return _bundle(nodes, edges)


# ---------------------------------------------------------------- confirmed over-merges (different exits)
@pytest.mark.parametrize("sep", [
    pytest.param(2.0, marks=pytest.mark.xfail(strict=True, reason=(
        "limit: within mapper.ONE_SPOT_M (2.5 m) parallel views count as one spot, which "
        "test_one_door_seen_from_both_corridor_ends_merges_under_noise needs (one door, 1.5 m, 0.2 m noise: "
        "plan distance up to 2.3 m, headings as parallel as 16/12 deg); position and heading cannot tell the two "
        "apart"))),
    3.0, 4.0])
def test_two_exits_on_one_wall_reached_from_opposite_ends_stay_two(sep):
    b = _two_spots(sep, 0.0, 0.0)
    assert _pairs(b) == []
    mapper.consolidate(b, NO_PHOTOS, None, use_gemma=False, log=quiet)
    assert len([n for n in b.nodes if n.type == "exit"]) == 2


@pytest.mark.parametrize("fh_a,fh_b", [(40.0, 140.0), (45.0, 135.0)])
def test_exits_on_opposite_walls_with_heading_noise_stay_two(fh_a, fh_b):
    assert _pairs(_two_spots(3.0, fh_a, fh_b)) == []


# ---------------------------------------------------------------- holds: same direction, opposite walls, no noise
@pytest.mark.parametrize("fh_a,fh_b,come_a,come_b", [(0.0, 0.0, "W", "W"), (0.0, 180.0, "W", "E"),
                                                      (0.0, 180.0, "W", "W"), (270.0, 90.0, "W", "E")])
@pytest.mark.parametrize("sep", [2.0, 3.0, 4.0])
def test_distinct_exits_stay_two_in_these_layouts(sep, fh_a, fh_b, come_a, come_b):
    assert _pairs(_two_spots(sep, fh_a, fh_b, come_a, come_b)) == []


@pytest.mark.parametrize("sep", [0.0, 0.8, 1.5])
def test_one_door_seen_from_both_corridor_ends_merges_under_noise(sep):
    rng = random.Random(5)
    for _ in range(40):
        b = _two_spots(sep, rng.uniform(-30, 30) % 360, rng.uniform(-30, 30) % 360, nz=0.2, rng=rng)
        assert len(_pairs(b)) == 1


# ---------------------------------------------------------------- stairs: 2.5 m limit under noise
def _climb_pair(sep, nz, seed):
    from tests.test_twin_evidence import _climbs
    rng = random.Random(seed)
    b = _climbs(sep)
    for e in b.edges:
        e.length_m += rng.gauss(0, nz)
    return b


@pytest.mark.parametrize("sep", [4.0, 5.0])
def test_stairwells_4m_apart_never_merge_under_position_noise(sep):
    assert all(_pairs(_climb_pair(sep, 0.3, s)) == [] for s in range(60))


@pytest.mark.parametrize("sep", [0.0, 1.0])
def test_one_stairwell_walked_up_and_down_merges_under_position_noise(sep):
    assert all(len(_pairs(_climb_pair(sep, 0.3, s))) == 2 for s in range(60))


# ---------------------------------------------------------------- hallway point: route still goes the right way
def test_hallway_point_keeps_route_order_and_the_exit():
    nodes = [_node("r", "Room 110", "vA", "room"), _node("m", "Room 112", "vA", "room", t=5),
             _seen("p", "Exit", "vA", "exit", 90.0, t=10), _node("q", "Room 114", "vB", "room", t=15),
             _seen("x", "Exit", "vB", "exit", 90.0, t=20)]
    edges = [_edge("e1", "r", "m", 90, 3, "vA"), _edge("e2", "m", "p", 90, 2, "vA"),
             _edge("e3", "p", "q", 90, 2, "vB"), _edge("e4", "q", "x", 90, 2.5, "vB")]
    b = _bundle(nodes, edges)
    mapper.consolidate(b, Path("/nonexistent"), None, use_gemma=False, log=quiet)
    assert [n.id for n in b.nodes if n.type == "exit"] == ["x"]
    for start, want in (("r", ["r", "m", "p", "q", "x"]), ("m", ["m", "p", "q", "x"]), ("q", ["q", "x"])):
        assert [s.node_id for s in routing.route(b, start, "x").steps] == want


# ---------------------------------------------------------------- OCR
@pytest.mark.parametrize("plaque", ["ROOM 1O1", "RM 107", "214A", "101", "110A", "B101", "L101", "1000"])
def test_room_plaques_are_not_rewritten(plaque):
    assert mapper._ocr_letters(plaque.split()[-1]) == plaque.split()[-1]


def test_negation_split_over_three_lines():
    lines = ["NOT", "AN", "EXIT"]
    assert mapper.place_type(_a("exit", lines, ocr=lines)) != "exit"
