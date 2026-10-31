"""One exit / NO EXIT plate / stair landing seen twice is merged on combined evidence, never on kind and plan position
alone (mapper._twin_candidates): two different exits or stairwells about 3 m apart stay two, while the same one seen
twice - from opposite directions, or the same way (read from afar, then reached) - becomes one place.

Hand-made maps on a plan with the main axis north (heading 90 = +x). Photos, where a test needs them, are written
to tmp_path; elsewhere there are none (ORB finds nothing), so photos never back a pair."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from app import mapper
from app.schemas import MapBundle, Node
from eval.score import score
from eval.truth import Truth, load
from tests.test_key_places import _a, _bundle, _edge, _node

NO_PHOTOS = Path("/nonexistent")
SCENARIOS = Path(__file__).resolve().parents[1] / "eval" / "scenarios"


def quiet(*_):
    pass


def _seen(nid, label, video, typ="exit", fh=None, t=0.0, signs=(), **kw):
    n = _node(nid, label, video, typ, t=t, **kw)
    n.frame_heading = fh
    n.sign_texts = n.texts = list(signs)
    return n


def _pairs(b, frames=None):
    return sorted((a.id, c.id, k) for a, c, _, k in mapper._twin_views(b, frames))


def _same_photo(frames: Path, *names: str) -> Path:
    """Write one textured photo under every name: the photos of these places match."""
    rng = np.random.default_rng(7)
    img = cv2.resize(rng.integers(0, 255, (48, 64), dtype=np.uint8), (640, 480), interpolation=cv2.INTER_NEAREST)
    for _ in range(40):
        cv2.circle(img, (int(rng.integers(0, 640)), int(rng.integers(0, 480))), int(rng.integers(5, 30)),
                   int(rng.integers(0, 255)), -1)
    for name in names:
        cv2.imwrite(str(frames / name), img)
    return frames


# ---------------------------------------------------------------- exits: the same one seen twice merges
def _exit_two_directions(fh1=180.0, fh2=180.0, signs1=(), signs2=(), typ="exit", label="Exit"):
    # walk A goes east along the corridor and glances south at the exit from x1; walk B comes south down a branch,
    # facing the exit, then turns west to room 120 (dead_ends)
    nodes = [_node("r", "Room 120", "vA", "room"), _seen("x1", label, "vA", typ, fh1, t=10, signs=signs1),
             _node("y", "Room 124", "vA", "room", t=20), _node("s", "Room 130", "vB", "room", t=2),
             _seen("x2", label, "vB", typ, fh2, t=12, signs=signs2)]
    edges = [_edge("e1", "r", "x1", 90, 2.0, "vA"), _edge("e2", "x1", "y", 90, 3.0, "vA"),
             _edge("e3", "s", "x2", 180, 3.6, "vB"), _edge("e4", "x2", "r", 270, 2.5, "vB")]
    return _bundle(nodes, edges)


def test_exit_seen_from_opposite_directions_merges_when_the_views_meet():
    b = _exit_two_directions()
    (tw,) = mapper._twin_candidates(b)
    assert {tw.a.id, tw.b.id} == {"x1", "x2"} and tw.kind == "exit" and "the views meet" in tw.why and tw.far is None
    mapper.consolidate(b, NO_PHOTOS, None, use_gemma=False, log=quiet)
    assert len([n for n in b.nodes if n.type == "exit"]) == 1


def _exit_read_then_reached(gap=4.5, fh_far=90.0, fh_near=90.0, walked=True):
    # walk A stops at p and reads the EXIT sign ahead; walk B passes p (loop closure) and walks on to the exit at x.
    # Without `walked`, walk B reaches x by another way (from room r), never passing p.
    nodes = [_node("r", "Room 110", "vA", "room"), _seen("p", "Exit", "vA", "exit", fh_far, t=10),
             _seen("x", "Exit (2)", "vB", "exit", fh_near, t=20)]
    edges = [_edge("e1", "r", "p", 90, 1.0, "vA")]
    edges.append(_edge("e2", "p", "x", 90, gap, "vB") if walked else _edge("e2", "r", "x", 90, 1.0 + gap, "vB"))
    return _bundle(nodes, edges)


def test_exit_read_from_afar_then_walked_up_to_is_one_exit():
    # two_floor: the down walk stops 3 m short of the exit facing it; the revisit walks on to it (4.6 m on the plan,
    # beyond TWIN_VIEW_M: the same way, so only the walk and the views make it one place)
    b = _exit_read_then_reached()
    (tw,) = mapper._twin_candidates(b)
    assert {tw.a.id, tw.b.id} == {"p", "x"} and "walked up to (walk vB)" in tw.why
    assert tw.d == pytest.approx(4.5, abs=0.1) and tw.far.id == "p"
    edges = {(e.from_node, e.to_node) for e in b.edges}
    mapper.consolidate(b, NO_PHOTOS, None, use_gemma=False, log=quiet)
    assert [(n.id, n.label) for n in b.nodes if n.type == "exit"] == [("x", "Exit")]
    # the spot the sign was read from stays a spot of the corridor (not folded into the exit): a place between it and
    # the exit (two_floor: room 110) keeps its order on the way, so routes do not turn back
    p = next(n for n in b.nodes if n.id == "p")
    assert (p.type, p.label) == ("hallway", "Hallway point")
    assert {(e.from_node, e.to_node) for e in b.edges} == edges


def test_exit_read_twice_from_one_spot_merges():
    b = _exit_read_then_reached(gap=1.0)
    (tw,) = mapper._twin_candidates(b)
    assert tw.far is None  # one spot, two photos: one place
    mapper.consolidate(b, NO_PHOTOS, None, use_gemma=False, log=quiet)
    assert len([n for n in b.nodes if n.type == "exit"]) == 1 and {n.id for n in b.nodes} < {"r", "p", "x"}


def test_exit_read_from_afar_by_another_walk_needs_matching_photos(tmp_path):
    b = _exit_read_then_reached(walked=False)
    assert _pairs(b) == []  # same view direction, no walk between them, no photos: the design limit holds
    assert _pairs(b, _same_photo(tmp_path, "p.jpg", "x.jpg")) == [("p", "x", "exit")]


def test_exit_read_from_afar_with_the_same_distinctive_sign_merges():
    b = _exit_read_then_reached(walked=False)
    for n in b.nodes:
        if n.type == "exit":
            n.sign_texts = n.texts = ["EXIT 4"]
    assert _pairs(b) == [("p", "x", "exit")]


# ---------------------------------------------------------------- exits: two different ones about 3 m apart stay two
def test_exits_at_two_ends_of_a_short_corridor_stay_two():
    # walk A walks east 2 m and faces the east exit; walk B walks west 1 m and faces the west exit: 3 m apart,
    # opposite directions, but the photos look away from each other
    nodes = [_node("r", "Room 120", "vA", "room"), _seen("x1", "Exit", "vA", "exit", 90.0, t=10),
             _seen("x2", "Exit", "vB", "exit", 270.0, t=5)]
    b = _bundle(nodes, [_edge("e1", "r", "x1", 90, 2.0, "vA"), _edge("e2", "r", "x2", 270, 1.0, "vB")])
    assert _pairs(b) == []
    mapper.consolidate(b, NO_PHOTOS, None, use_gemma=False, log=quiet)
    assert {n.id for n in b.nodes if n.type == "exit"} == {"x1", "x2"}


def test_exits_side_by_side_glanced_at_from_one_corridor_stay_two(tmp_path):
    # two exit doors 3 m apart on the north wall; walk B passes both (one the walk A glanced at): the photos look the
    # same way but not at each other's spot. Even identical photos (same door, same sign) do not make them one.
    nodes = [_node("r", "Room 120", "vA", "room"), _seen("x1", "Exit", "vA", "exit", 0.0, t=10),
             _seen("x2", "Exit", "vB", "exit", 0.0, t=20)]
    b = _bundle(nodes, [_edge("e1", "r", "x1", 90, 1.0, "vA"), _edge("e2", "x1", "x2", 90, 3.0, "vB")])
    assert _pairs(b) == []
    assert _pairs(b, _same_photo(tmp_path, "x1.jpg", "x2.jpg")) == []


@pytest.mark.parametrize("s1,s2", [(["EXIT 2"], ["EXIT 3"]), (["NORTH EXIT"], ["SOUTH EXIT"]),
                                   (["EXIT A"], ["EXIT B"])])
def test_exits_with_different_signs_stay_two(s1, s2):
    assert _pairs(_exit_two_directions(signs1=s1, signs2=s2)) == []


@pytest.mark.parametrize("s1,s2", [(["EXIT"], ["EXIT 2"]), (["EXIT", "ROCN"], ["EXIT"]), (["LEVEL 1 EXIT"], ["EXIT"]),
                                   (["EXIT 2"], ["EXIT 2"])])
def test_plain_or_equal_exit_signs_do_not_tell_exits_apart(s1, s2):
    assert _pairs(_exit_two_directions(signs1=s1, signs2=s2)) == [("x1", "x2", "exit")]


def test_mens_and_womens_restrooms_side_by_side_stay_two():
    b = _exit_two_directions(signs1=["MEN"], signs2=["WOMEN"], typ="restroom", label="Restroom")
    assert _pairs(b) == []
    b = _exit_two_directions(signs1=["WOMEN"], signs2=["WOMENS RESTROOM"], typ="restroom", label="Restroom")
    assert _pairs(b) == [("x1", "x2", "restroom")]


# ---------------------------------------------------------------- NO EXIT plates
def test_no_exit_plate_seen_by_two_walks_is_one_landmark():
    # dead_ends: walk A faces the NO EXIT door (looking north), walk B reads it ahead on its right walking west
    b = _exit_two_directions(fh1=0.0, fh2=315.0, signs1=["NO EXIT"], signs2=["NOEXIT"], typ="landmark",
                             label="No Exit")
    assert _pairs(b) == [("x1", "x2", "no exit")]
    mapper.consolidate(b, NO_PHOTOS, None, use_gemma=False, log=quiet)
    assert len([n for n in b.nodes if n.type == "landmark"]) == 1


def test_no_exit_and_do_not_enter_plates_are_different_kinds():
    b = _exit_two_directions(fh1=0.0, fh2=315.0, signs1=["NO EXIT"], signs2=["DO NOT ENTER"], typ="landmark",
                             label="No Exit")
    next(n for n in b.nodes if n.id == "x2").label = "Do Not Enter"
    assert _pairs(b) == []


def test_landmark_without_a_no_sign_is_not_a_twin_kind():
    b = _exit_two_directions(fh1=0.0, fh2=315.0, signs1=["BOOK FAIR"], signs2=["BOOK FAIR"], typ="landmark",
                             label="Poster")
    assert _pairs(b) == []


# ---------------------------------------------------------------- stairwells
def _climbs(sep, down=True, h_b=0.0, signs=((), ())):
    # walk A climbs a1 -> a2 heading north; walk B climbs (or comes down) the flight `sep` m east of it
    nodes = [_node("r", "Room 114", "vA", "room"),
             _seen("a1", "Stairs", "vA", "stairs", 0.0, t=10, signs=signs[0], vertical="stairs"),
             _seen("a2", "Stairs", "vA", "stairs", 0.0, t=30, signs=signs[0], floor=2, vertical="stairs"),
             _seen("b1", "Stairs", "vB", "stairs", 0.0, t=10, signs=signs[1], vertical="stairs"),
             _seen("b2", "Stairs", "vB", "stairs", 0.0, t=30, signs=signs[1], floor=2, vertical="stairs")]
    climb_b = (_edge("e4", "b2", "b1", (h_b + 180) % 360, 6.0, "vB", "stairs_down") if down
               else _edge("e4", "b1", "b2", h_b, 6.0, "vB", "stairs_up"))
    edges = [_edge("e1", "r", "a1", 90, 3.0, "vA"), _edge("e2", "a1", "a2", 0, 6.0, "vA", "stairs_up"),
             _edge("e3", "r", "b1", 90, 3.0 + sep, "vB"), climb_b]
    return _bundle(nodes, edges)


@pytest.mark.parametrize("sep", [0.5, 2.0])
def test_stairwell_walked_up_and_down_merges_on_both_floors(sep):
    assert _pairs(_climbs(sep)) == [("a1", "b1", "stairs"), ("a2", "b2", "stairs")]


@pytest.mark.parametrize("sep", [3.0, 3.5])
def test_two_stairwells_about_3m_apart_walked_up_and_down_stay_two(sep):
    b = _climbs(sep)
    assert _pairs(b) == []
    mapper.consolidate(b, NO_PHOTOS, None, use_gemma=False, log=quiet)
    assert len([n for n in b.nodes if mapper._is_vertical(n)]) == 4


def test_the_same_stairs_sign_lets_landings_merge_a_little_further_apart():
    assert _pairs(_climbs(3.0, signs=(["STAIR B"], ["STAIR B"]))) == [("a1", "b1", "stairs"), ("a2", "b2", "stairs")]
    assert _pairs(_climbs(1.0, signs=(["STAIR A"], ["STAIR B"]))) == []  # different stairwells, however close
    assert _pairs(_climbs(1.0, signs=(["STAIRS", "LEVEL 1"], ["STAIRS", "LEVEL 2"]))) != []  # floor signs: no identity


def test_stairwell_climbed_the_same_way_twice_merges_when_the_photos_match(tmp_path):
    b = _climbs(0.5, down=False)
    assert _pairs(b) == []  # position alone: two stairwells side by side
    frames = _same_photo(tmp_path, "a1.jpg", "b1.jpg")
    assert _pairs(b, frames) == [("a1", "b1", "stairs"), ("a2", "b2", "stairs")]  # either end's photos back both


def test_two_stairwells_climbed_the_same_way_stay_two_even_when_they_look_alike(tmp_path):
    frames = _same_photo(tmp_path, "a1.jpg", "b1.jpg", "a2.jpg", "b2.jpg")
    assert _pairs(_climbs(3.0, down=False), frames) == []  # identical stairwells 3 m apart
    assert _pairs(_climbs(0.5, down=False, h_b=180.0), frames) == []  # scissor stairs: flights run opposite ways


# ---------------------------------------------------------------- sign reading (bugs fixed in this round)
@pytest.mark.parametrize("token,want", [("N0", "NO"), ("EX1T", "EXIT"), ("N0EXIT", "NOEXIT"), ("1O7", "1O7"),
                                        ("107", "107"), ("214A", "214A"), ("101", "101"), ("1O1", "1O1"),
                                        ("110A", "110A"), ("10", "10")])
def test_ocr_zero_and_one_read_as_letters_only_in_words(token, want):
    assert mapper._ocr_letters(token) == want


@pytest.mark.parametrize("text,exit_", [("N0 EXIT", False), ("N0EXIT", False), ("NO EX1T", False), ("EX1T", True),
                                        ("EXIT 10", True), ("EXIT 1O7", True)])
def test_ocr_lookalikes_in_exit_signs(text, exit_):
    assert mapper.names_key(text, "exit") is exit_


@pytest.mark.parametrize("plaque", ["ROOM 1O7", "RM 107", "214A", "ROOM 101"])
def test_plaques_keep_their_numbers(plaque):
    a = _a("room", [plaque], rooms=[plaque.split()[-1]])
    assert mapper.place_type(a) == "room" and mapper.room_ids(a.texts) == mapper.room_ids([plaque])


@pytest.mark.parametrize("lines,exit_", [(["NOT AN", "EXIT"], False), (["EXIT", "NOT AN"], False),
                                         (["NOT A", "EXIT"], False), (["NO", "EXIT"], False),
                                         (["NO SMOKING", "EXIT"], True), (["AN", "EXIT"], True)])
def test_negation_with_an_article_reaches_the_next_line(lines, exit_):
    assert (mapper.place_type(_a("exit", lines, ocr=lines)) == "exit") is exit_


# ---------------------------------------------------------------- eval truth scores these duplicates
def _sight_truth():
    return Truth.model_validate({
        "schema": "waypoint-eval/1", "scenario": "unit",
        "building": {"places": [{"id": "x", "label": "Exit", "type": "exit", "floor": 1, "x": 2.0, "y": 1.0},
                                {"id": "r", "label": "Room 101", "room": "101", "floor": 1, "x": 3.0, "y": 1.0}]},
        "videos": [{"id": "v1", "file": "v1.mp4", "start_floor": 1,
                    "visits": [{"place": "x", "t": 5.0, "kind": "see"}, {"place": "r", "t": 9.0, "kind": "see"}]},
                   {"id": "v2", "file": "v2.mp4", "start_floor": 1,
                    "visits": [{"place": "x", "t": 3.0, "kind": "look"}, {"place": "r", "t": 1.0, "kind": "look"}]}],
        "expect_merged": ["x", "r"],
    })


@pytest.mark.parametrize("far_type,far_texts,merged", [
    ("exit", [], 1),  # a second exit 3 m short of the exit: a missed merge
    ("hallway", [], 2),  # the hallway spot the sign was read from: not a copy of the exit
    ("hallway", ["ROOM 101"], 1),  # a corridor spot claiming the room's plaque is a copy of the room
])
def test_a_sign_read_from_afar_counts_only_a_map_place_claiming_it(far_type, far_texts, merged):
    nodes = [Node(id="far", label="Spot", type=far_type, floor=1, texts=far_texts, sign_texts=far_texts),
             Node(id="x", label="Exit", type="exit", floor=1),
             Node(id="r", label="Room 101", type="room", floor=1, texts=["ROOM 101"], sign_texts=["ROOM 101"])]
    obs = {"v1": [{"t": t, "t_arrive": t - 0.5, "t_depart": t + 0.5, "node": "far", "frame_id": "", "kind": "end"}
                  for t in (5.0, 9.0)],
           "v2": [{"t": t, "t_arrive": t - 0.5, "t_depart": t + 0.5, "node": n, "frame_id": "", "kind": "glance"}
                  for t, n in ((3.0, "x"), (1.0, "r"))]}
    res = score(_sight_truth(), MapBundle(building_id="u", nodes=nodes), obs)
    assert res["metrics"]["merge_recall"] == merged / 2



def test_scenario_truth_scores_the_exit_read_from_afar_and_the_no_exit_door():
    two = load(SCENARIOS / "two_floor" / "truth.json")
    assert "exit_w" in two.expect_merged
    assert any(v.place == "exit_w" and v.kind == "see" for v in next(x for x in two.videos if x.id == "down").visits)
    dead = load(SCENARIOS / "dead_ends" / "truth.json")
    assert "noexit" in dead.expect_merged
    assert any(v.place == "noexit" and v.kind == "see" for v in next(x for x in dead.videos if x.id == "west").visits)
