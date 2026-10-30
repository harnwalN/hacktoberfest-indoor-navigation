"""Map layout across floors: a staircase or elevator sits at the same x/y on every floor it connects."""
from app import mapper, routing
from app.schemas import Edge, MapBundle, Node


def _edge(eid, a, b, heading, length, **kw):
    return Edge(id=eid, from_node=a, to_node=b, heading_out=heading, heading_in=heading,
                heading_out_raw=heading, heading_in_raw=heading, length_m=length, **kw)


def _stairs_bundle(vertical="stairs_up", source="walk"):
    nodes = [Node(id=x, label=x, floor=f) for x, f in [("A", 1), ("S1", 1), ("S2", 2), ("B", 2)]]
    edges = [
        _edge("e1", "A", "S1", 0, 10),
        _edge("e2", "S1", "S2", 0, 8, vertical=vertical, source=source),
        _edge("e3", "S2", "B", 90, 10),
    ]
    return MapBundle(building_id="t", nodes=nodes, edges=edges)


def _xy(m):
    return {n.id: (n.x, n.y) for n in m.nodes}


def test_stairs_share_xy_across_floors():
    m = _stairs_bundle()
    mapper.finalize(m)
    p = _xy(m)
    assert abs(p["S1"][0] - p["S2"][0]) < 0.5 and abs(p["S1"][1] - p["S2"][1]) < 0.5
    # the floors above and below still keep their own horizontal layout
    assert abs((p["S1"][1] - p["A"][1]) - 10) < 0.5
    assert abs((p["B"][0] - p["S2"][0]) - 10) < 0.5


def test_walked_elevator_and_stairs_down_share_xy():
    for vertical in ("elevator", "stairs_down"):
        m = _stairs_bundle(vertical=vertical)
        mapper.finalize(m)
        p = _xy(m)
        assert abs(p["S1"][0] - p["S2"][0]) < 0.5 and abs(p["S1"][1] - p["S2"][1]) < 0.5, vertical


def test_walked_stairs_win_over_conflicting_horizontal_loop():
    # A second path between the floors says S2 is 4 m east of S1; the climbed stairs still say "same spot".
    m = _stairs_bundle()
    m.nodes.append(Node(id="C", label="C", floor=2))
    m.edges += [_edge("e4", "S1", "C", 90, 4), _edge("e5", "C", "S2", 270, 0.1)]
    mapper.finalize(m)
    p = _xy(m)
    assert abs(p["S1"][0] - p["S2"][0]) < 1.0 and abs(p["S1"][1] - p["S2"][1]) < 1.0


def test_route_distance_does_not_depend_on_stair_layout():
    m = _stairs_bundle()
    mapper.finalize(m)
    res = routing.route(m, "A", "B")
    assert res.node_path == ["A", "S1", "S2", "B"]
    assert any(s.action == "stairs_up" for s in res.steps)
