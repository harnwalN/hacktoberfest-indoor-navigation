"""Probes for the vertical flag: ahead_* sightings that never climb, and PUT /map round trip."""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import config, floors, mapper, router, store
from app.schemas import MapBundle, VideoInfo
from tests.test_floor_links import QUIET, _kf, _track


def _ahead_walk(vid, room, ahead="ahead_up"):
    # Two agreeing room numbers: one alone is not confirmed floor evidence (floors._confirmed).
    nxt = str(int(room) + 2)
    return [_kf(vid, 0, "start"), _kf(vid, 4, "glance", "room", rooms=[room], label=f"Room {room}"),
            _kf(vid, 8, "scene", "hallway", ahead, label="Hallway"),
            _kf(vid, 10, "glance", "room", rooms=[nxt], label=f"Room {nxt}"), _kf(vid, 14, "end")]


def _bundle(tmp_path, a, b):
    bun = MapBundle(building_id="t", videos=[VideoInfo(id="vA", filename="a", status="done"),
                                             VideoInfo(id="vB", filename="b", status="done")])
    for vid, kfs in (("vA", a), ("vB", b)):
        mapper.integrate_walk(bun, vid, kfs, _track(15), tmp_path, None, use_gemma=False, **QUIET)
    mapper.consolidate(bun, tmp_path, None, use_gemma=False, **QUIET)
    return bun


def test_ahead_up_without_climb_does_not_flag_node():
    # Stairs seen ahead but never climbed: no stair node, and the hallway keeps its own label/type.
    wn = mapper.walk_nodes(_ahead_walk("vA", "121"))
    assert all(w.vertical is None for w in wn)
    assert all(mapper._label(w)[1] != "stairs" for w in wn)


def test_ahead_then_climb_still_flags_boundary():
    # README "film staircases from both floors": stand at the bottom, see the stairs ahead, then climb.
    kfs = [_kf("v", 0, "start"), _kf("v", 4, "glance", "room", rooms=["121"], label="Room 121"),
           _kf("v", 10, "scene", "hallway", "ahead_up", label="Hallway"),
           _kf("v", 11, "heartbeat", "hallway", "on_stairs_up"), _kf("v", 12, "end", "hallway", "on_stairs_up")]
    wn = mapper.walk_nodes(kfs)
    assert wn[-1].vertical == "stairs" and mapper._label(wn[-1]) == ("Stairs", "stairs")
    # Same at the top of a flight: the first frame after the climb looks back down / at the next flight.
    kfs = [_kf("v", 0, "start", "hallway", "on_stairs_up"), _kf("v", 2, "scene", "hallway", "ahead_down"),
           _kf("v", 8, "glance", "room", rooms=["208"], label="Room 208"), _kf("v", 14, "end")]
    assert mapper.walk_nodes(kfs)[0].vertical == "stairs"


def test_ahead_sightings_on_two_floors_do_not_link(tmp_path):
    """PROBE: two walks that only SEE stairs ahead (never climb), on floors 1 and 2."""
    bun = _bundle(tmp_path, _ahead_walk("vA", "121"), _ahead_walk("vB", "208", "ahead_down"))
    assert sorted({n.floor for n in bun.nodes}) == [1, 2]
    links = [e for e in bun.edges if e.vertical != "none" and e.source == "merge"]
    assert links == []


def test_vertical_round_trips_through_put_map(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BUILDINGS_DIR", tmp_path / "buildings")
    app = FastAPI(); app.include_router(router.api)
    c = TestClient(app)
    bun = store.create_building("T")
    bid = bun.building_id
    body = {"nodes": [{"id": "n001", "label": "Hallway", "vertical": "stairs"},
                      {"id": "n002", "label": "Hall"}], "edges": []}
    assert c.put(f"/api/buildings/{bid}/map", json=body).status_code in (200, 204)
    m = store.load_map(bid)
    assert {n.id: n.vertical for n in m.nodes} == {"n001": "stairs", "n002": None}
    bad = {"nodes": [{"id": "n001", "label": "x", "vertical": "ladder"}], "edges": []}
    assert c.put(f"/api/buildings/{bid}/map", json=bad).status_code == 422


def test_consolidate_twice_no_duplicate_links(tmp_path):
    from tests.test_floor_links import _two_walks
    b = _two_walks(tmp_path)
    n = len(b.edges)
    mapper.consolidate(b, tmp_path, None, use_gemma=False, **QUIET)
    assert len(b.edges) == n


def test_walk_and_manual_links_never_removed(tmp_path, monkeypatch):
    from app import gemma
    from app.schemas import Edge, Node, SamePlace
    monkeypatch.setattr(gemma, "same_stairwell", lambda *a: SamePlace(same_place=False, confidence=0.95, reason="x"))
    for src in ("walk", "manual"):
        nodes = [Node(id="n001", label="Stairs", type="stairs", floor=1, floor_source="rooms", frame_id="a.jpg"),
                 Node(id="n002", label="Stairs", type="stairs", floor=2, floor_source="rooms", frame_id="b.jpg")]
        e = Edge(id="e001", from_node="n001", to_node="n002", heading_out=0, heading_in=0,
                 vertical="stairs_up", source=src, inferred_floor_change=True)
        b = MapBundle(building_id="t", nodes=nodes, edges=[e])
        floors._link_floors(b, tmp_path, None, True, **QUIET)
        assert len(b.edges) == 1
