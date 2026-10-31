"""Round-2 adversarial checks (sentinel-e2e) on the negation scope, whole-frame negation and the direction / opposite-climb
merge rules. Known open limits are marked xfail(strict) so a fix shows up as XPASS."""
import pytest

from app import mapper
from tests.test_key_places import _a, _bundle, _edge, _node


def _pairs(b):
    return [(a.id, c.id, k) for a, c, _, k in mapper._twin_views(b)]


@pytest.mark.parametrize("text", ["NO EXIT", "NOT AN EXIT", "NON-EXIT", "NOEXIT", "NO-EXIT", "No Exit", "ANNOUNCE NOT EXIT",
                                  "NO. EXIT", "NO, EXIT", "NO ONE EXIT"])
def test_negated_wordings(text):
    assert not mapper.names_key(text, "exit")


@pytest.mark.parametrize("text", ["KNOW EXIT", "NOTE: EXIT", "NORTH EXIT", "NOW EXIT", "EXIT NO ENTRY", "EXIT - NO RE-ENTRY",
                                  "EXIT NOT HERE", "NO ENTRY EXIT ONLY", "DO NOT BLOCK EXIT"])
def test_words_that_only_start_with_no_are_exits(text):
    assert mapper.names_key(text, "exit")


@pytest.mark.parametrize("texts,ocr", [(["NO EXIT"], ["NO", "EXIT"]), (["EXIT"], ["NO", "EXIT"]), (["NO", "EXIT"], ["NO", "EXIT"]),
                                       (["NO EXIT", "EXIT"], ["NO EXIT", "EXIT"]), (["EXIT", "NO EXIT"], ["EXIT", "NO EXIT"]),
                                       (["NOT AN EXIT", "EXIT"], ["NOT AN EXIT", "EXIT"])])
def test_frame_with_negated_exit_is_not_exit(texts, ocr):
    assert mapper.place_type(_a("exit", texts, ocr=ocr)) != "exit"


def test_separate_positive_exit_beside_no_exit_is_an_exit():
    assert mapper.place_type(_a("exit", ["NO EXIT", "FIRE EXIT"], ocr=["NO EXIT", "FIRE EXIT"])) == "exit"
    assert mapper.place_type(_a("exit", ["EXIT", "NO ENTRY"], ocr=["EXIT", "NO ENTRY"])) == "exit"


@pytest.mark.xfail(strict=True, reason="known open: 'NOT AN' + 'EXIT' on two lines is not joined (builder-disclosed)")
def test_not_an_exit_split_over_two_lines():
    assert mapper.place_type(_a("exit", ["NOT AN", "EXIT"], ocr=["NOT AN", "EXIT"])) != "exit"


@pytest.mark.xfail(strict=True, reason="OCR reads the O of NO as a zero: 'N0 EXIT' is taken for an exit (minor)")
def test_ocr_zero_for_o_in_no_exit():
    assert not mapper.names_key("N0 EXIT", "exit")


def _exit_pair(h2, l2=1.5):
    nodes = [_node("r", "Room 120", "vA", "room"), _node("x1", "Exit", "vA", "exit", t=10), _node("x2", "Exit", "vB", "exit", t=5)]
    return _bundle(nodes, [_edge("e1", "r", "x1", 90, 2.0, "vA"), _edge("e2", "r", "x2", h2, l2, "vB")])


@pytest.mark.parametrize("h2", [135, 180, 270, 225])
def test_exit_seen_from_directions_45_or_more_apart_merges(h2):
    assert _pairs(_exit_pair(h2)) == [("x1", "x2", "exit")]


@pytest.mark.parametrize("h2", [90, 100, 134, 60])
def test_exit_seen_from_nearly_same_direction_stays_two(h2):
    assert _pairs(_exit_pair(h2)) == []


def test_exit_walked_toward_from_both_ends_of_a_corridor_merges():
    nodes = [_node("r", "Room 120", "vA", "room"), _node("x1", "Exit", "vA", "exit", t=10),
             _node("s", "Room 130", "vB", "room"), _node("x2", "Exit", "vB", "exit", t=5)]
    edges = [_edge("e1", "r", "x1", 90, 2.0, "vA"), _edge("e2", "s", "x2", 270, 3.0, "vB"), _edge("e3", "r", "s", 90, 5.0, "vA")]
    assert _pairs(_bundle(nodes, edges)) == [("x1", "x2", "exit")]


def _stairs(down: bool, sep_b=3.5):
    nodes = [_node("r", "Room 114", "vA", "room"), _node("a1", "Stairs", "vA", "stairs", vertical="stairs", t=10),
             _node("a2", "Stairs", "vA", "stairs", floor=2, vertical="stairs", t=30),
             _node("b1", "Stairs", "vB", "stairs", vertical="stairs", t=10),
             _node("b2", "Stairs", "vB", "stairs", floor=2, vertical="stairs", t=30)]
    edges = [_edge("e1", "r", "a1", 90, 3.0, "vA"), _edge("e2", "a1", "a2", 0, 6.0, "vA", "stairs_up"),
             _edge("e3", "r", "b1", 90, sep_b, "vB"),
             _edge("e4", "b2", "b1", 0, 6.0, "vB", "stairs_down") if down else _edge("e4", "b1", "b2", 0, 6.0, "vB", "stairs_up")]
    return _bundle(nodes, edges)


def test_stairwell_walked_up_then_down_merges_on_both_floors():
    assert sorted(_pairs(_stairs(True))) == [("a1", "b1", "stairs"), ("a2", "b2", "stairs")]


def test_stairwell_walked_up_twice_is_not_merged_by_position():
    assert _pairs(_stairs(False)) == []


@pytest.mark.xfail(strict=True, reason="known open: two different stairwells 3 m apart, one climbed up and one down, merge")
def test_two_stairwells_3m_apart_up_and_down_stay_two():
    assert _pairs(_stairs(True, sep_b=6.0)) == []
