"""Routing on a hand-written fixture: a square loop with a staircase off one corner.

    A --e1--> B          headings: A->B east (90), B->C south (180), C->D west (270), D->A north (0)
    ^         |          B->S north (0) then stairs up S->T (0)
    |         v
    D <------ C
"""
import networkx as nx
import pytest

from app import routing
from app.schemas import Edge, MapBundle, Node


def _edge(i, a, b, h_out, h_in=None, vertical="none", rev=True):
    h_in = h_out if h_in is None else h_in
    return Edge(id=f"e{i}", from_node=a, to_node=b, heading_out=h_out, heading_in=h_in, vertical=vertical,
                frames_fwd=[f"{a}_dep.jpg", f"{b}_arr.jpg"], frames_rev=[f"{b}_dep.jpg", f"{a}_arr.jpg"] if rev else None)


@pytest.fixture
def bundle():
    nodes = [Node(id=x, label=f"Room {x}", frame_id=f"{x}.jpg", texts=[x]) for x in "ABCDST"]
    nodes.append(Node(id="Z", label="Island"))
    edges = [
        _edge(1, "A", "B", 90), _edge(2, "B", "C", 180), _edge(3, "C", "D", 270), _edge(4, "D", "A", 0),
        _edge(5, "B", "S", 0, rev=False), _edge(6, "S", "T", 0, vertical="stairs_up"),
    ]
    return MapBundle(building_id="test", nodes=nodes, edges=edges)


def actions(resp):
    return [s.action for s in resp.steps]


def test_forward_and_reverse_mirror(bundle):
    fwd = routing.route(bundle, "A", "C")
    assert fwd.node_path == ["A", "B", "C"]
    assert actions(fwd) == ["start", "turn_right", "destination_reached"]
    back = routing.route(bundle, "C", "A")
    assert actions(back) == ["start", "turn_left", "destination_reached"]


def test_loop_uses_both_ways(bundle):
    assert actions(routing.route(bundle, "D", "B")) == ["start", "turn_right", "destination_reached"]
    assert actions(routing.route(bundle, "B", "D"))[1] in ("turn_right", "turn_left")


def test_stairs_flip(bundle):
    up = routing.route(bundle, "A", "T")
    assert up.node_path == ["A", "B", "S", "T"]
    assert actions(up) == ["start", "turn_left", "stairs_up", "destination_reached"]
    down = routing.route(bundle, "T", "A")
    assert down.steps[0].action == "stairs_down"
    assert actions(down)[1:] == ["go_straight", "turn_right", "destination_reached"]


def test_direction_correct_photos(bundle):
    up = routing.route(bundle, "A", "T")
    assert up.steps[0].frame_url.endswith("A_dep.jpg")
    assert up.steps[1].frame_url.endswith("B_arr.jpg")
    down = routing.route(bundle, "S", "A")  # S->B was never walked: node photo + a note
    assert down.steps[0].frame_url.endswith("/S.jpg") and down.steps[0].view_note
    assert down.steps[1].frame_url.endswith("/B.jpg") and down.steps[1].view_note
    assert down.steps[2].frame_url.endswith("A_arr.jpg") and down.steps[2].view_note is None


def test_no_path(bundle):
    with pytest.raises(nx.NetworkXNoPath):
        routing.route(bundle, "A", "Z")


def test_quantize():
    assert routing.quantize_turn(10) == "go_straight"
    assert routing.quantize_turn(-40) == "slight_left"
    assert routing.quantize_turn(95) == "turn_right"
    assert routing.quantize_turn(-170) == "u_turn"
    assert routing.quantize_turn(350) == "go_straight"


def test_shortest_walk_beats_fewest_places():
    nodes = [Node(id=x, label=x) for x in "ABCDE"]
    edges = [
        Edge(id="long", from_node="A", to_node="E", heading_out=0, heading_in=0, length_m=120),
        Edge(id="e1", from_node="A", to_node="B", heading_out=0, heading_in=0, length_m=10),
        Edge(id="e2", from_node="B", to_node="C", heading_out=0, heading_in=0, length_m=10),
        Edge(id="e3", from_node="C", to_node="D", heading_out=0, heading_in=0, length_m=10),
        Edge(id="e4", from_node="D", to_node="E", heading_out=0, heading_in=0, length_m=10),
    ]
    r = routing.route(MapBundle(building_id="t", nodes=nodes, edges=edges), "A", "E")
    assert r.node_path == ["A", "B", "C", "D", "E"]  # 40 m through 3 places, not 120 m direct


def test_positions_are_metric():
    from app import mapper
    nodes = [Node(id=x, label=x) for x in "ABCD"]
    sq = [("A", "B", 90, 10), ("B", "C", 0, 20), ("C", "D", 270, 10), ("D", "A", 180, 20)]
    edges = [Edge(id=f"e{i}", from_node=a, to_node=b, heading_out=h, heading_in=h, heading_out_raw=h,
                  heading_in_raw=h, length_m=L) for i, (a, b, h, L) in enumerate(sq)]
    m = MapBundle(building_id="t", nodes=nodes, edges=edges)
    mapper.finalize(m)
    p = {n.id: (n.x, n.y) for n in m.nodes}
    dist = lambda a, b: ((p[a][0] - p[b][0]) ** 2 + (p[a][1] - p[b][1]) ** 2) ** 0.5
    assert abs(dist("A", "B") - 10) < 0.5 and abs(dist("B", "C") - 20) < 0.5 and abs(dist("A", "C") - 22.4) < 0.5
