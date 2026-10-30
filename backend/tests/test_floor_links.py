"""Stairs/elevator links between floors: created when the evidence says so, and only then."""
import json

from app import floors, gemma, mapper, routing, store
from app.schemas import Edge, FrameAnalysis, Keyframe, MapBundle, Node, SamePlace, VideoInfo

QUIET = dict(log=lambda *_: None)


def _kf(vid, t, kind="scene", place="hallway", stairs="none", rooms=(), label=""):
    a = FrameAnalysis(has_sign=bool(rooms), texts=list(rooms), room_numbers=list(rooms), place_type=place,
                      stairs=stairs, suggested_label=label, grounded=True)
    return Keyframe(frame_id=f"{vid}_{t}.jpg", video_id=vid, t=t, kind=kind, heading=0, analysis=a)


def _track(t_end):
    ts = [i / 10 for i in range(int(t_end * 10) + 1)]
    return {"t": ts, "heading": [0.0] * len(ts), "fps": 10}


# Walk A: floor 1, Room 121, then starts up the stairs (video ends mid-flight). Gemma calls the last
# frame before the climb a plain "Hallway".
WALK_A = [
    _kf("vA", 0, "start"),
    _kf("vA", 4, "glance", "room", rooms=["121"], label="Room 121"),
    _kf("vA", 10, "scene", "hallway", label="Hallway"),
    _kf("vA", 11, "heartbeat", "hallway", "on_stairs_up"),
    _kf("vA", 12, "end", "hallway", "on_stairs_up"),
]
# Walk B: starts on the same flight, arrives on floor 2 (again "Hallway" to Gemma), walks to Room 208.
WALK_B = [
    _kf("vB", 0, "start", "hallway", "on_stairs_up"),
    _kf("vB", 1, "heartbeat", "hallway", "on_stairs_up"),
    _kf("vB", 3, "scene", "hallway", label="Hallway"),
    _kf("vB", 8, "glance", "room", rooms=["208"], label="Room 208"),
    _kf("vB", 14, "end"),
]


def _two_walks(tmp_path):
    b = MapBundle(building_id="t", videos=[VideoInfo(id="vA", filename="a", status="done"),
                                           VideoInfo(id="vB", filename="b", status="done")])
    for vid, kfs in (("vA", WALK_A), ("vB", WALK_B)):
        mapper.integrate_walk(b, vid, kfs, _track(15), tmp_path, None, use_gemma=False, **QUIET)
    mapper.consolidate(b, tmp_path, None, use_gemma=False, **QUIET)
    return b


def _by_label(b, label):
    return next(n for n in b.nodes if n.label == label)


def test_walk_nodes_flag_stair_boundaries():
    wa = mapper.walk_nodes(WALK_A)
    assert [w.vertical for w in wa] == [None, None, "stairs"]  # last frame before the climb
    wb = mapper.walk_nodes(WALK_B)
    assert [w.vertical for w in wb] == ["stairs", None, None]  # first frame after it
    assert mapper._label(wb[0]) == ("Stairs", "stairs")


def test_absorb_keeps_stair_flag():
    # Top-of-stairs frame absorbed into a named place 0.5 s later: the place is still the stair landing.
    kfs = [_kf("v", 0, "heartbeat", "hallway", "on_stairs_up"), _kf("v", 2, "scene", "hallway"),
           _kf("v", 2.5, "dwell", "landmark", label="Trophy case"), _kf("v", 9, "end")]
    kfs[2].analysis.texts, kfs[2].analysis.has_sign = ["TROPHIES"], True
    wn = mapper.walk_nodes(kfs)
    assert len(wn) == 2 and wn[0].analysis.suggested_label == "Trophy case" and wn[0].vertical == "stairs"


def test_stair_landing_and_adjacent_room_stay_two_places():
    kfs = [_kf("v", 0, "heartbeat", "hallway", "on_stairs_up"), _kf("v", 2, "scene", "hallway"),
           _kf("v", 2.5, "glance", "room", rooms=["210"], label="Room 210"), _kf("v", 9, "end")]
    wn = mapper.walk_nodes(kfs)
    assert [(w.analysis.place_type, w.vertical) for w in wn] == [("hallway", "stairs"), ("room", None), ("hallway", None)]


def test_hallway_typed_stair_boundary_links_to_floor_above(tmp_path):  # (a)
    b = _two_walks(tmp_path)
    r121, r208 = _by_label(b, "Room 121"), _by_label(b, "Room 208")
    assert (r121.floor, r208.floor) == (1, 2)
    links = [e for e in b.edges if e.source == "merge" and e.vertical == "stairs_up"]
    assert len(links) == 1
    lo, hi = (next(n for n in b.nodes if n.id == x) for x in (links[0].from_node, links[0].to_node))
    assert (lo.floor, lo.video_id, lo.vertical, hi.floor, hi.video_id, hi.vertical) == (1, "vA", "stairs", 2, "vB", "stairs")
    assert links[0].inferred_floor_change  # one staircase per floor and no other evidence: flagged as a guess


def test_route_climbs_the_linked_stairs(tmp_path):  # (d)
    b = _two_walks(tmp_path)
    r121, r208 = _by_label(b, "Room 121"), _by_label(b, "Room 208")
    up = routing.route(b, r121.id, r208.id)
    assert [s.floor for s in up.steps][0] == 1 and up.steps[-1].floor == 2
    assert any("take the stairs up to floor 2" in s.instruction_text for s in up.steps), [s.instruction_text for s in up.steps]
    assert any(s.instruction_text.startswith("At the top of the stairs") for s in up.steps)
    down = routing.route(b, r208.id, r121.id)
    assert any("take the stairs down to floor 1" in s.instruction_text for s in down.steps)


def _stairs_pair(**extra):
    return [Node(id="n001", label="Stairs", type="stairs", floor=1, floor_source="rooms", frame_id="a.jpg", **extra),
            Node(id="n002", label="Stairwell", type="stairs", floor=2, floor_source="rooms", frame_id="b.jpg", **extra)]


def _gemma_says(monkeypatch, same, conf):
    calls = []

    def fake(pa, pb, cache):
        calls.append((pa.name, pb.name))
        return SamePlace(same_place=same, confidence=conf, reason="different railings and wall colour")
    monkeypatch.setattr(gemma, "same_stairwell", fake)
    return calls


def test_gemma_veto_blocks_single_to_single_guess(tmp_path, monkeypatch):  # (b)
    calls = _gemma_says(monkeypatch, False, 0.9)
    b = MapBundle(building_id="t", nodes=_stairs_pair())
    assert floors._link_floors(b, tmp_path, None, True, **QUIET) == 0
    assert calls and b.edges == []


def test_weak_gemma_no_still_allows_flagged_guess(tmp_path, monkeypatch):
    _gemma_says(monkeypatch, False, 0.4)
    b = MapBundle(building_id="t", nodes=_stairs_pair())
    assert floors._link_floors(b, tmp_path, None, True, **QUIET) == 1
    assert b.edges[0].inferred_floor_change and not b.edges[0].verified


def test_gemma_veto_removes_earlier_guess(tmp_path, monkeypatch):
    b = MapBundle(building_id="t", nodes=_stairs_pair())
    floors._link_floors(b, tmp_path, None, False, **QUIET)  # Gemma offline: guessed link
    assert len(b.edges) == 1 and b.edges[0].inferred_floor_change
    _gemma_says(monkeypatch, False, 0.9)
    floors._link_floors(b, tmp_path, None, True, **QUIET)
    assert b.edges == []


def test_relinking_is_idempotent_and_upgrades_guess(tmp_path, monkeypatch):
    b = MapBundle(building_id="t", nodes=_stairs_pair())
    floors._link_floors(b, tmp_path, None, False, **QUIET)
    floors._link_floors(b, tmp_path, None, False, **QUIET)
    assert len(b.edges) == 1
    _gemma_says(monkeypatch, True, 0.9)
    assert floors._link_floors(b, tmp_path, None, True, **QUIET) == 0
    assert len(b.edges) == 1 and b.edges[0].verified and not b.edges[0].inferred_floor_change


def test_better_evidence_replaces_a_guessed_link(tmp_path):
    b = MapBundle(building_id="t", nodes=[
        Node(id="n001", label="Stairs", type="stairs", texts=["STAIR S2"], floor=1, floor_source="rooms"),
        Node(id="n002", label="Stairs", type="stairs", floor=2, floor_source="rooms")])
    floors._link_floors(b, tmp_path, None, False, **QUIET)
    assert {(e.from_node, e.to_node) for e in b.edges} == {("n001", "n002")}
    # A later walk finds the same stairwell sign on floor 2: that is the real match.
    b.nodes.append(Node(id="n003", label="Stairs", type="stairs", texts=["STAIR S2"], floor=2, floor_source="rooms"))
    floors._link_floors(b, tmp_path, None, False, **QUIET)
    assert [(e.from_node, e.to_node, e.verified) for e in b.edges] == [("n001", "n003", True)]


def test_climbed_stairs_are_not_linked_again_but_other_floors_are(tmp_path):
    # n001 (floor 1) was walked down to n000 (floor 0); that says nothing about floor 1 -> 2, even though
    # an elevator ride elsewhere already connects floors 1 and 2 in the graph.
    def e(i, a, b, v="none"):
        return Edge(id=f"e00{i}", from_node=a, to_node=b, heading_out=0, heading_in=0, vertical=v)
    b = MapBundle(building_id="t", nodes=[
        Node(id="n000", label="Stairs", type="stairs", floor=0, floor_source="rooms"),
        Node(id="n001", label="Stairs", type="stairs", floor=1, floor_source="rooms"),
        Node(id="n002", label="Stairs", type="stairs", floor=2, floor_source="rooms"),
        Node(id="h1", label="Hall", floor=1, floor_source="rooms"),
        Node(id="h2", label="Hall", floor=2, floor_source="rooms")],
        edges=[e(1, "n001", "n000", "stairs_down"), e(2, "n001", "h1"), e(3, "h1", "h2", "elevator"), e(4, "h2", "n002")])
    assert floors._link_floors(b, tmp_path, None, False, **QUIET) == 1
    assert ("n001", "n002", "stairs_up") in {(e.from_node, e.to_node, e.vertical) for e in b.edges}
    assert floors._link_floors(b, tmp_path, None, False, **QUIET) == 0  # pair now has its vertical edge


def test_known_floors_preferred_over_guessed(tmp_path):
    shared = ["STAIR S2"]
    b = MapBundle(building_id="t", nodes=[
        Node(id="n001", label="Stairs", type="stairs", texts=shared, floor=1, floor_source="rooms"),
        Node(id="n002", label="Stairs", type="stairs", texts=shared, floor=2, floor_source="default"),
        Node(id="n003", label="Stairs", type="stairs", texts=shared, floor=2, floor_source="rooms")])
    floors._link_floors(b, tmp_path, None, False, **QUIET)
    assert [(e.from_node, e.to_node) for e in b.edges] == [("n001", "n003")]


def test_poster_label_is_not_a_staircase():
    assert floors._is_vertical(Node(id="x", label="Take the stairs poster", type="landmark")) is None
    assert floors._is_vertical(Node(id="x", label="Stairwell B", type="hallway")) == "stairs"
    assert floors._is_vertical(Node(id="x", label="Hallway", vertical="stairs")) == "stairs"
    assert floors._is_vertical(Node(id="x", label="Hallway", vertical="elevator")) == "elevator"


def test_old_map_without_vertical_field_loads(tmp_path, monkeypatch):  # (c)
    from app import config
    monkeypatch.setattr(config, "BUILDINGS_DIR", tmp_path)
    old = {"building_id": "old", "nodes": [{"id": "n001", "label": "Stairs", "type": "stairs", "floor": 1}],
           "edges": []}
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "map.json").write_text(json.dumps(old))
    m = store.load_map("old")
    assert m.nodes[0].vertical is None and floors._is_vertical(m.nodes[0]) == "stairs"
