"""Key places (exit / entrance / restroom / elevator) need their own sign; dead ends and NO EXIT are not exits;
one exit (or stair landing) seen from two directions is one place; labels are spelled one way.

Synthetic keyframes and hand-made maps (no video, no Gemma). The end-to-end check is eval/scenarios/dead_ends."""
import json
from pathlib import Path

import pytest

from app import config, mapper
from app.schemas import Edge, FrameAnalysis, Keyframe, MapBundle, Node, VideoInfo
from eval.score import score
from eval.truth import Truth, load


def _a(place="hallway", texts=(), rooms=(), label="", ocr=None, grounded=True, stairs="none"):
    tx = list(texts)
    return FrameAnalysis(texts=tx, room_numbers=list(rooms), has_sign=bool(tx or rooms), place_type=place,
                         suggested_label=label, ocr=tx if ocr is None else list(ocr), grounded=grounded,
                         stairs=stairs)


def _kf(t, kind="scene", **kw):
    return Keyframe(frame_id=f"f{t}.jpg", video_id="v", t=float(t), kind=kind, heading=0.0, analysis=_a(**kw))


def _wn(t, kind="scene", vertical=None, arrive=None, depart=None, **kw):
    k = _kf(t, kind, **kw)
    return mapper.WalkNode(kf=k, t_arrive=t if arrive is None else arrive, t_depart=t if depart is None else depart,
                           analysis=k.analysis, vertical=vertical, glances=int(kind == "glance"))


# ---------------------------------------------------------------- what a sign says
@pytest.mark.parametrize("text,kind,yes", [
    ("EXIT", "exit", True), ("EMERGENCY EXIT", "exit", True), ("EXIT ONLY", "exit", True),
    ("NO EXIT", "exit", False), ("NOEXIT", "exit", False), ("THIS IS NOT AN EXIT", "exit", False), ("EXITING TRAFFIC", "exit", False),
    ("MAIN ENTRANCE", "entrance", True), ("DO NOT ENTER", "entrance", False), ("NO ENTRY", "entrance", False),
    ("WOMEN", "restroom", True), ("ALL GENDER RESTROOM", "restroom", True), ("ELEVATOR", "elevator", True),
    ("LEVEL 1", "entrance", False),
])
def test_names_key(text, kind, yes):
    assert mapper.names_key(text, kind) is yes


def test_negated_key_signs():
    assert mapper.negated_key("NO EXIT") and mapper.negated_key("Do not enter") and mapper.negated_key("NOEXIT")
    assert not mapper.negated_key("EXIT") and not mapper.negated_key("ROOM 101")


@pytest.mark.parametrize("texts,ocr,yes", [
    (["NO EXIT"], ["NO", "EXIT"], False),  # OCR split the plate
    (["NO EXIT", "EXIT"], None, False),  # a bare EXIT beside NO EXIT: a piece of it
    (["EXIT"], ["NO", "EXIT"], False),  # Gemma missed the NO, OCR has it on the line before
    (["NO EXIT", "FIRE EXIT"], None, True),  # a separate, positive sign
    (["FIRE EXIT"], ["NO", "FIRE EXIT"], False),  # 'NO FIRE EXIT' read in two lines
    (["NO SMOKING", "EXIT"], None, True),  # a negated non-key sign vetoes nothing
    (["DO NOT ENTER", "EXIT ONLY"], None, True),
])
def test_negation_is_read_over_the_whole_frame(texts, ocr, yes):
    assert (mapper.place_type(_a("exit", texts, ocr=ocr)) == "exit") is yes


# ---------------------------------------------------------------- grounded place type
def test_key_type_needs_its_own_sign_on_the_frame():
    pt = mapper.place_type
    assert pt(_a("exit")) == "hallway"  # the end wall of a corridor that Gemma called an exit
    assert pt(_a("exit", ["EXIT"])) == "exit"
    assert pt(_a("exit", [], ocr=["EXIT"])) == "exit"  # OCR read it, Gemma left it out of texts
    assert pt(_a("exit", ["NO EXIT"])) == "hallway"
    assert pt(_a("entrance", ["LEVEL 1"])) == "hallway"  # a floor sign is no entrance sign
    assert pt(_a("restroom", ["ROOM 115"], ["115"])) == "room"  # a plaque, but no restroom sign
    assert pt(_a("elevator")) == "hallway"
    assert pt(_a("exit", grounded=False)) == "hallway"  # without OCR Gemma must still have read the sign
    assert pt(_a("exit", ["EXIT"], grounded=False)) == "exit"
    assert pt(_a("stairs")) == "stairs"  # a staircase is recognizable from the picture
    assert not mapper._key_place(_a("exit")) and mapper._key_place(_a("exit", ["EXIT"]))


def test_dead_end_without_a_sign_is_a_hallway_end():
    nodes = mapper.walk_nodes([_kf(0, "start"), _kf(10, "end", place="exit", label="Exit")])
    end = nodes[-1]
    assert mapper._label(end) == ("Hallway end", "hallway")
    # the U-turn at the dead end is not a junction either
    turn = _wn(5, "turn_start", place="entrance", label="Entrance to a hallway")
    assert mapper._label(turn) == ("Hallway end", "hallway")


def test_no_exit_sign_is_a_landmark_not_an_exit():
    seq = [_kf(0, "start"), _kf(8, "glance", place="exit", texts=["NO EXIT"], label="No Exit"), _kf(20, "end")]
    (door,) = [w for w in mapper.walk_nodes(seq) if w.kf.kind == "glance"]  # still worth a place
    assert mapper._label(door) == ("No Exit", "landmark")
    assert not mapper._key_place(door.analysis)


def test_floor_sign_at_the_stairs_names_stairs_not_an_entrance_or_corridor():
    entrance = _wn(5, vertical="stairs", place="entrance", texts=["LEVEL 1"], label="Level 1 Entrance")
    corridor = _wn(5, vertical="stairs", place="hallway", texts=["LEVEL 2"], label="Level 2 Corridor")
    typed = _wn(5, place="stairs", texts=["LEVEL 2"], label="Level 2 Corridor")
    named = _wn(5, place="stairs", texts=["STAIR B"], label="Stair B")
    other_end = _wn(5, vertical="stairs", place="stairs", texts=["LEVEL 2"], label="Level 2 Stairwell")
    assert mapper._label(entrance) == mapper._label(corridor) == mapper._label(typed) == ("Stairs", "stairs")
    assert mapper._label(other_end) == ("Stairs", "stairs")  # the floor sign may be at the other end
    assert mapper._label(named) == ("Stair B", "stairs")


# ---------------------------------------------------------------- exits do not absorb plaques
@pytest.mark.parametrize("exit_first", [True, False])
def test_room_glance_wins_over_an_exit_sign_seen_from_its_door(exit_first):
    exit_frame = _kf(10 if exit_first else 10.8, place="exit", texts=["EXIT"], label="Exit")
    glance = _kf(10.8 if exit_first else 10, "glance", place="room", texts=["ROOM 114"], rooms=["114"],
                 label="Room 114")
    nodes = mapper.walk_nodes([_kf(0, "start"), exit_frame, glance, _kf(30, "end")])
    (door,) = [w for w in nodes if mapper.room_ids(w.texts)]
    assert mapper._label(door) == ("Room 114", "room")
    assert "EXIT" not in door.texts  # the exit is down the hall, not at this door
    assert not any(mapper.place_type(w.analysis) == "exit" for w in nodes)


def test_exit_takes_no_plaque_from_a_frame_it_absorbs():
    keep = _wn(10, place="exit", texts=["EXIT"], label="Exit")
    other = _wn(10.5, place="hallway", label="Hallway")
    other.extra_texts = ["RM 112"]  # a plaque the corridor frame had absorbed earlier
    mapper._absorb(keep, other)
    assert keep.texts == ["EXIT"] and mapper._label(keep) == ("Exit", "exit")


# ---------------------------------------------------------------- labels
@pytest.mark.parametrize("label,texts,ptype,want", [
    ("RM 112", ["RM 112"], "room", "Room 112"),
    ("ROOM 114", ["ROOM 114"], "room", "Room 114"),
    ("Rm. 118a lab", [], "room", "Room 118A lab"),
    ("Ker Space 107", ["107", "KER SPACE", "MAKER SPACE"], "room", "Maker Space 107"),
    ("Maker Sp", ["MAKER SPACE"], "room", "Maker Space"),
    ("Exit - ROCW1", ["EXIT", "ROCW1"], "exit", "Exit"),
    ("Room 112", ["ROOMS 110-120"], "room", "Room 112"),  # never 'Rooms 112' from a directional sign
    ("Seminar Room 305", ["SEMINAR ROOM 305"], "room", "Seminar Room 305"),
])
def test_tidy_label(label, texts, ptype, want):
    assert mapper.tidy_label(label, texts, ptype) == want


def test_bare_type_label_without_a_sign_is_no_name():
    start = _wn(0, "start", place="room", label="Room")
    assert mapper._label(start) == ("Walk start", "room")


def test_bare_type_label_is_named_by_its_own_sign():
    assert mapper._label(_wn(5, place="room", texts=["LEVEL 2", "SEMINAR"], label="Room")) == ("Seminar", "room")
    assert mapper._label(_wn(5, place="room", texts=["RM 116"], rooms=["116"], label="Room")) == ("Room 116", "room")
    assert mapper._label(_wn(5, place="room", texts=["LEVEL 2"], label="Room")) == ("Room", "room")  # floor sign only


@pytest.mark.parametrize("place,texts,label", [("room", ["NO EXIT"], "No Exit"), ("exit", ["NOEXIT"], "No Exit Door"),
                                               ("hallway", ["NO EXIT"], "Hallway")])
def test_no_exit_door_is_one_landmark_whatever_gemma_typed(place, texts, label):
    assert mapper._label(_wn(5, "glance", place=place, texts=texts, label=label)) == ("No Exit", "landmark")


@pytest.mark.parametrize("label,want", [("Exit Door", "Exit"), ("Emergency Exit Door", "Emergency Exit"),
                                        ("Exit sign", "Exit"), ("Door", "Door")])
def test_key_place_label_drops_door_and_sign(label, want):
    assert mapper.tidy_label(label, ["EXIT"], "exit") == want


# ---------------------------------------------------------------- a turn onto the stairs is the landing
def test_turn_right_before_the_stairs_is_the_landing():
    turn = _wn(18.2, "turn_start", place="hallway", arrive=18.9, depart=20.7)
    near = _wn(23.0, vertical="stairs", place="hallway")
    far = _wn(23.0 + config.STAIR_APPROACH_SEC, vertical="stairs", place="hallway")
    assert mapper._absorbs([], [], turn, near)
    assert not mapper._absorbs([], [], turn, far)
    door = _wn(18.2, "turn_start", place="room", texts=["ROOM 210"], rooms=["210"], arrive=18.9, depart=20.7)
    assert not mapper._absorbs([], [], door, near)  # a door beside the stairs stays a room


# ---------------------------------------------------------------- one place seen from two directions
def _node(nid, label, video, typ="hallway", floor=1, vertical=None, t=0.0):
    return Node(id=nid, label=label, type=typ, floor=floor, floor_source="user", video_id=video, t=t,
                vertical=vertical, frame_id=f"{nid}.jpg")


def _edge(eid, a, b, heading, length, video, vertical="none"):
    return Edge(id=eid, from_node=a, to_node=b, heading_out=heading, heading_in=heading, heading_out_raw=heading,
                heading_in_raw=heading, length_m=length, vertical=vertical, source="walk", video_id=video)


def _bundle(nodes, edges):
    return MapBundle(building_id="t", nodes=nodes, edges=edges,
                     videos=[VideoInfo(id=v, filename=v, status="done", anchored=True) for v in ("vA", "vB")])


def test_exit_seen_from_two_directions_is_one_place():
    # walk A sees the exit 2 m east of room 120, walk B 1.5 m south of it; a far exit 12 m away stays apart
    nodes = [_node("r", "Room 120", "vA", "room"), _node("x1", "Exit", "vA", "exit", t=10),
             _node("x2", "Exit", "vB", "exit", t=5), _node("x3", "Exit", "vB", "exit", t=40)]
    edges = [_edge("e1", "r", "x1", 90, 2.0, "vA"), _edge("e2", "r", "x2", 180, 1.5, "vB"),
             _edge("e3", "r", "x3", 270, 12.0, "vB")]
    b = _bundle(nodes, edges)
    pairs = [(a.id, c.id, kind) for a, c, _, kind in mapper._twin_views(b)]
    assert pairs == [("x1", "x2", "exit")]
    mapper.consolidate(b, Path("/nonexistent"), None, use_gemma=False, log=lambda *_: None)
    assert sorted(n.id for n in b.nodes if n.type == "exit") in (["x1", "x3"], ["x2", "x3"])


def test_exits_seen_walking_the_same_way_stay_two():
    # two walks pass two exits 1.5 m apart going the same way: same view, the photos did not match -> two exits
    nodes = [_node("r", "Room 120", "vA", "room"), _node("x1", "Exit", "vA", "exit", t=10),
             _node("x2", "Exit", "vB", "exit", t=5)]
    edges = [_edge("e1", "r", "x1", 90, 2.0, "vA"), _edge("e2", "r", "x2", 90, 3.5, "vB")]
    assert mapper._twin_views(_bundle(nodes, edges)) == []


def test_exits_in_separate_map_pieces_never_merge_by_position():
    nodes = [_node("x1", "Exit", "vA", "exit"), _node("x2", "Exit", "vB", "exit")]
    assert mapper._twin_views(_bundle(nodes, [])) == []  # positions of unconnected pieces mean nothing


def test_stair_landings_of_two_walks_merge_when_they_lead_to_the_same_floor():
    # walk A climbs 1 -> 2, walk B comes down 2 -> 1 (opposite views of both landings); both pass room 110
    nodes = [_node("r", "Room 110", "vA", "room"),
             _node("a1", "Stairs", "vA", "stairs", vertical="stairs", t=10),
             _node("a2", "Stairs", "vA", "stairs", floor=2, vertical="stairs", t=30),
             _node("b2", "Hallway junction", "vB", "intersection", floor=2, vertical="stairs", t=5),
             _node("b1", "Stairs", "vB", "stairs", vertical="stairs", t=25)]
    edges = [_edge("e1", "r", "a1", 90, 4.0, "vA"), _edge("e2", "a1", "a2", 0, 6.0, "vA", "stairs_up"),
             _edge("e3", "b2", "b1", 180, 6.0, "vB", "stairs_down"), _edge("e4", "b1", "r", 270, 5.0, "vB")]
    b = _bundle(nodes, edges)
    pairs = {frozenset((a.id, c.id)) for a, c, _, kind in mapper._twin_views(b) if kind == "stairs"}
    assert frozenset(("a1", "b1")) in pairs
    mapper.consolidate(b, Path("/nonexistent"), None, use_gemma=False, log=lambda *_: None)
    assert len([n for n in b.nodes if n.floor == 1 and mapper._is_vertical(n)]) == 1


def _view_bundle(view_x, second_landing_x=None):
    # one walk: room 114, the STAIRS sign seen ahead from the corridor (typed stairs, no climb), the landing, up
    nodes = [_node("r", "Room 114", "vA", "room", t=11), _node("v", "Stairs", "vA", "stairs", vertical="stairs", t=13),
             _node("l", "Stairs (2)", "vA", "stairs", vertical="stairs", t=19),
             _node("l2", "Stairs (3)", "vA", "stairs", floor=2, vertical="stairs", t=38)]
    edges = [_edge("e1", "r", "v", 90, view_x, "vA"), _edge("e2", "r", "l", 90, 2.8, "vA"),
             _edge("e3", "l", "l2", 0, 20.0, "vA", "stairs_up")]
    if second_landing_x is not None:  # another stairwell, climbed by walk B
        nodes += [_node("m", "Stairs", "vB", "stairs", vertical="stairs", t=5),
                  _node("m2", "Stairs", "vB", "stairs", floor=2, vertical="stairs", t=25)]
        edges += [_edge("e4", "r", "m", 90, second_landing_x, "vB"), _edge("e5", "m", "m2", 0, 20.0, "vB", "stairs_up")]
    return _bundle(nodes, edges)


def test_stairs_seen_ahead_is_the_landing_climbed_moments_later():
    b = _view_bundle(1.9)
    assert [(a.id, c.id, k) for a, c, _, k in mapper._twin_views(b)] == [("l", "v", "stairs")]
    mapper.consolidate(b, Path("/nonexistent"), None, use_gemma=False, log=lambda *_: None)
    assert sorted(n.id for n in b.nodes if n.floor == 1 and mapper._is_vertical(n)) == ["l"]
    assert any({e.from_node, e.to_node} == {"r", "l"} for e in b.edges)


def test_stairs_view_far_away_or_between_two_stairwells_stays():
    assert not mapper._twin_views(_view_bundle(9.0))  # 6 m short of the landing: not that landing
    # two climbed stairwells within reach of the view: which one it showed is unknown
    assert not [p for p in mapper._twin_views(_view_bundle(1.9, second_landing_x=0.5)) if "v" in (p[0].id, p[1].id)]


def test_stair_landings_to_different_floors_stay_apart():
    nodes = [_node("r", "Room 210", "vA", "room", floor=2),
             _node("up", "Stairs", "vA", "stairs", floor=2, vertical="stairs", t=10),
             _node("up3", "Stairs", "vA", "stairs", floor=3, vertical="stairs", t=30),
             _node("dn", "Stairs", "vB", "stairs", floor=2, vertical="stairs", t=10),
             _node("dn1", "Stairs", "vB", "stairs", floor=1, vertical="stairs", t=30)]
    edges = [_edge("e1", "r", "up", 90, 3.0, "vA"), _edge("e2", "up", "up3", 0, 6.0, "vA", "stairs_up"),
             _edge("e3", "r", "dn", 90, 3.5, "vB"), _edge("e4", "dn", "dn1", 0, 6.0, "vB", "stairs_down")]
    assert not [p for p in mapper._twin_views(_bundle(nodes, edges)) if p[3] == "stairs"]


# ---------------------------------------------------------------- scorer: false key places
def _truth():
    return Truth.model_validate({
        "schema": "waypoint-eval/1", "scenario": "unit",
        "building": {"places": [
            {"id": "r", "label": "Room 101", "room": "101", "floor": 1, "x": 2.0, "y": 1.0},
            {"id": "x", "label": "Exit", "type": "exit", "floor": 1, "x": 20.0, "y": 1.0},
            {"id": "end", "label": "Dead end", "type": "hallway", "floor": 1, "x": 0.0, "y": 1.0},
        ]},
        "videos": [{"id": "v1", "file": "v1.mp4", "start_floor": 1,
                    "visits": [{"place": "end", "t": 1.0}, {"place": "r", "t": 3.0, "kind": "look"},
                               {"place": "x", "t": 20.0, "kind": "look"}],
                    "path": [[1.0, 0.0, 1.0, 1], [3.0, 2.0, 1.0, 1], [17.0, 16.0, 1.0, 1], [20.0, 19.0, 1.0, 1]]}],
    })


def _obs(*rows):
    return {"v1": [{"t": t, "t_arrive": t - 0.5, "t_depart": t + 0.5, "node": n, "frame_id": "", "kind": "glance"}
                   for t, n in rows]}


def test_false_key_places_is_a_hard_gate():
    nodes = [Node(id="n1", label="Exit", type="exit", floor=1), Node(id="n2", label="Room 101", type="room",
             floor=1, texts=["ROOM 101"], sign_texts=["ROOM 101"]), Node(id="n3", label="Exit", type="exit", floor=1),
             Node(id="n4", label="Exit (2)", type="exit", floor=1)]
    # n1 is the dead end typed exit; n3 is the real exit; n4 saw the exit door from 4 m away (t=17, x=16)
    res = score(_truth(), MapBundle(building_id="u", nodes=nodes), _obs((1.0, "n1"), (3.0, "n2"), (20.0, "n3"),
                                                                       (17.0, "n4")))
    assert res["metrics"]["false_key_places"] == 1
    assert [k["node"] for k in res["details"]["false_key_places"]] == ["n1"]
    assert "false_key_places" in res["hard_fail"]


def test_dead_ends_scenario_truth():
    path = Path(__file__).resolve().parents[1] / "eval" / "scenarios" / "dead_ends" / "truth.json"
    truth = load(path)
    places = {p.id: p for p in truth.building.places}
    assert [p for p in places.values() if p.type == "exit"] == [places["exit_s"]]
    assert {"exit_s", "r120", "r122"} <= set(truth.expect_merged)
    assert ["exit_s", "noexit", "a NO EXIT sign is not the exit"] in truth.must_not_merge
    looks = {(v.id, x.place) for v in truth.videos for x in v.visits if x.kind == "look"}
    assert {("east", "exit_s"), ("west", "exit_s"), ("east", "noexit")} <= looks  # the exit from both walks
    assert json.loads(path.read_text())["scenario"] == "dead_ends"
