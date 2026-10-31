"""Adversarial checks on the A2 key-place work: exit wording, NO EXIT, dead ends, stair/exit over-merging.
Defects found by sentinel-e2e (OCR-split NO EXIT, negation scope, NON-EXIT, close stairwells) are fixed; their tests now pass."""
from pathlib import Path

import pytest

from app import mapper
from tests.test_key_places import _a, _bundle, _edge, _kf, _node, _view_bundle, _wn

quiet = lambda *_: None  # noqa: E731


@pytest.mark.parametrize("text", ["NO EXITS", "No-Exit", "no exit", "NO  EXIT", "NOT EXIT", "NOTEXIT", "NO THROUGH EXIT",
                                  "NOT AN EXIT. EXIT AHEAD", "NOEXIT DOOR"])
def test_negated_exit_wording_is_never_an_exit(text):
    assert not mapper.names_key(text, "exit")
    assert mapper.place_type(_a("exit", [text])) != "exit"


@pytest.mark.parametrize("text", ["EXITS", "EXIT 2", "FIRE EXIT", "Emergency Exit", "EXIT ONLY", "exit"])
def test_exit_wording_is_an_exit(text):
    assert mapper.names_key(text, "exit")


@pytest.mark.parametrize("texts", [["DEAD END"], ["EXITING TRAFFIC"], ["LEVEL 1"], ["ROOM 101"], []])
def test_unrelated_text_never_makes_an_exit(texts):
    assert mapper.place_type(_a("exit", texts)) != "exit"
    assert mapper._label(_wn(5, "end", place="exit", texts=texts, label="Exit"))[1] != "exit"


@pytest.mark.parametrize("kind", ["start", "end", "turn_start", "glance", "scene"])
def test_unsigned_exit_typed_node_is_not_exit_for_any_keyframe_kind(kind):
    assert mapper._label(_wn(5, kind, place="exit", label="Exit"))[1] != "exit"


def test_ocr_split_no_exit_sign_is_not_an_exit():
    assert mapper.place_type(_a("exit", ["NO EXIT"], ocr=["NO", "EXIT"])) != "exit"


def test_gemma_no_exit_with_ocr_exit_line_is_not_an_exit():
    assert mapper.place_type(_a("exit", ["NO EXIT"], ocr=["EXIT"])) != "exit"


def test_non_exit_is_not_an_exit():
    assert not mapper.names_key("NON-EXIT", "exit")


@pytest.mark.parametrize("text", ["NO ENTRY EXIT ONLY", "DO NOT ENTER EXIT ONLY", "DO NOT BLOCK EXIT"])
def test_negation_scope_does_not_swallow_a_later_exit(text):
    assert mapper.names_key(text, "exit")


# ---------------------------------------------------------------- over-merging
def test_exit_and_no_exit_landmark_never_merge():
    nodes = [_node("r", "Room 120", "vA", "room"), _node("x", "Exit", "vA", "exit", t=10),
             _node("n", "No Exit", "vB", "landmark", t=5)]
    edges = [_edge("e1", "r", "x", 90, 2.0, "vA"), _edge("e2", "r", "n", 90, 2.2, "vB")]
    b = _bundle(nodes, edges)
    assert mapper._twin_views(b) == []
    mapper.consolidate(b, Path("/nonexistent"), None, use_gemma=False, log=quiet)
    assert {n.id for n in b.nodes} >= {"x", "n"}


def test_exits_on_different_floors_never_merge():
    nodes = [_node("r", "Room 120", "vA", "room"), _node("x1", "Exit", "vA", "exit", t=10),
             _node("x2", "Exit", "vA", "exit", floor=2, t=20)]
    b = _bundle(nodes, [_edge("e1", "r", "x1", 90, 2.0, "vA"), _edge("e2", "r", "x2", 90, 2.0, "vA", "stairs_up")])
    assert mapper._twin_views(b) == []


def test_exits_seconds_apart_in_one_walk_stay_two():
    nodes = [_node("r", "Room 120", "vA", "room"), _node("x1", "Exit", "vA", "exit", t=10),
             _node("x2", "Exit (2)", "vA", "exit", t=12)]
    b = _bundle(nodes, [_edge("e1", "r", "x1", 90, 2.0, "vA"), _edge("e2", "r", "x2", 180, 1.5, "vA")])
    assert mapper._twin_views(b) == []


def _two_stairwells(sep, floor_b=2):
    # two stairwells `sep` m apart, each climbed by its own walk
    nodes = [_node("r", "Room 114", "vA", "room"),
             _node("a1", "Stairs", "vA", "stairs", vertical="stairs", t=10),
             _node("a2", "Stairs", "vA", "stairs", floor=2, vertical="stairs", t=30),
             _node("b1", "Stairs", "vB", "stairs", vertical="stairs", t=10),
             _node("b2", "Stairs", "vB", "stairs", floor=floor_b, vertical="stairs", t=30)]
    edges = [_edge("e1", "r", "a1", 90, 3.0, "vA"), _edge("e2", "a1", "a2", 0, 6.0, "vA", "stairs_up"),
             _edge("e3", "r", "b1", 90, 3.0 + sep, "vB"), _edge("e4", "b1", "b2", 0, 6.0, "vB", "stairs_up")]
    return _bundle(nodes, edges)


def test_stairwells_to_different_floors_do_not_merge():
    b = _two_stairwells(1.0, floor_b=3)
    assert not [p for p in mapper._twin_views(b) if p[3] == "stairs"]


def test_far_apart_stairwells_do_not_merge():
    assert not [p for p in mapper._twin_views(_two_stairwells(15.0)) if p[3] == "stairs"]


def test_two_close_stairwells_to_the_same_floor_do_not_merge():
    assert not [p for p in mapper._twin_views(_two_stairwells(1.5)) if p[3] == "stairs"]


def test_stairs_in_separate_map_pieces_never_merge_by_position():
    nodes = [_node("a1", "Stairs", "vA", "stairs", vertical="stairs"),
             _node("a2", "Stairs", "vA", "stairs", floor=2, vertical="stairs"),
             _node("b1", "Stairs", "vB", "stairs", vertical="stairs"),
             _node("b2", "Stairs", "vB", "stairs", floor=2, vertical="stairs")]
    b = _bundle(nodes, [_edge("e1", "a1", "a2", 0, 6.0, "vA", "stairs_up"), _edge("e2", "b1", "b2", 0, 6.0, "vB", "stairs_up")])
    assert mapper._twin_views(b) == []


def test_stairs_view_exactly_at_limit_and_beyond():
    assert not mapper._twin_views(_view_bundle(2.8 + 6.0))
    assert not mapper._twin_views(_view_bundle(20.0))


def test_room_named_stairs_view_is_not_swallowed_as_a_landing():
    b = _view_bundle(1.9)
    v = next(n for n in b.nodes if n.id == "v")
    v.texts, v.sign_texts, v.label = ["ROOM 115"], ["ROOM 115"], "Room 115"
    assert not [p for p in mapper._twin_views(b) if "v" in (p[0].id, p[1].id)]
