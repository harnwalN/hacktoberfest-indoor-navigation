"""A walk's floor stays consistent: consolidation adopting it onto another floor, the uploader then setting
its start floor (mapper.set_walk_floor and PATCH /videos/{id}), and floor links after the move."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import config, gemma, mapper, store
from app.main import app
from app.schemas import Edge, MapBundle, Node, VideoInfo


def _quiet(*_):
    pass


def _bundle(upper_stairs_floor=4):
    """Walk vA is known (rooms) on floor 2 and has the only other staircase. Walk vB was guessed onto
    floor 1 and starts at vA's Maker Space, so consolidation adopts it onto floor 2."""
    nodes = [
        Node(id="n001", label="Maker Space", texts=["MAKER SPACE", "QUALCOMM INSTITUTE"],
             sign_texts=["MAKER SPACE", "QUALCOMM INSTITUTE"], description="glass doors with logo", floor=2,
             floor_source="rooms", video_id="vA", t=0),
        Node(id="n005", label="Stairs", type="stairs", floor=upper_stairs_floor, floor_source="rooms",
             video_id="vA", t=30),
        Node(id="n002", label="Maker Space Entrance", texts=["MAKERSPACE", "QUALCOMM"],
             sign_texts=["MAKER SPACE", "QUALCOMM"], description="glass doors and logo", floor=1,
             floor_source="default", video_id="vB", t=0),
        Node(id="n003", label="Hallway", floor=1, floor_source="default", video_id="vB", t=5),
        Node(id="n004", label="Stairwell", type="stairs", floor=1, floor_source="default", video_id="vB", t=10),
    ]
    edges = [
        Edge(id="e001", from_node="n002", to_node="n003", heading_out=0, heading_in=0,
             heading_out_raw=0, heading_in_raw=0, video_id="vB", length_m=5),
        Edge(id="e002", from_node="n003", to_node="n004", heading_out=90, heading_in=90,
             heading_out_raw=90, heading_in_raw=90, video_id="vB", length_m=5),
    ]
    videos = [VideoInfo(id="vA", filename="a", status="done", anchored=True, floor_offset=2),
              VideoInfo(id="vB", filename="b", status="done", anchored=True, floor_offset=1)]
    return MapBundle(building_id="t", nodes=nodes, edges=edges, videos=videos)


def _adopted(**kw):
    b = _bundle(**kw)
    mapper.consolidate(b, Path("."), None, use_gemma=False, log=_quiet)
    return b


def _floors(b):
    return {n.id: n.floor for n in b.nodes}


def _video(b, vid):
    return next(v for v in b.videos if v.id == vid)


def _stair_links(b):
    return [(e.from_node, e.to_node) for e in b.edges if e.vertical != "none"]


def test_adoption_updates_floor_offset():
    b = _adopted()
    assert "n002" not in _floors(b)  # merged into vA's Maker Space
    assert _floors(b)["n003"] == 2 and _floors(b)["n004"] == 2
    assert _video(b, "vB").floor_offset == 2
    assert _video(b, "vA").floor_offset == 2  # the walk it was adopted from is untouched


def test_set_walk_floor_to_adopted_floor_is_a_no_op():
    b = _adopted()
    before = _floors(b)
    mapper.set_walk_floor(b, "vB", 2, log=_quiet)
    assert _floors(b) == before
    assert _video(b, "vB").floor_offset == 2


def test_set_walk_floor_moves_only_own_places_and_relinks():
    b = _adopted()
    assert _stair_links(b) == []  # stairs on floors 2 and 4: nothing adjacent to link
    mapper.set_walk_floor(b, "vB", 3, log=_quiet)
    f = _floors(b)
    assert f["n003"] == 3 and f["n004"] == 3  # vB's own places moved up one floor
    assert f["n001"] == 2 and f["n005"] == 4  # vA's places (incl. the shared Maker Space) untouched
    assert _video(b, "vB").floor_offset == 3
    assert {n.floor_source for n in b.nodes if n.video_id == "vB"} == {"user"}
    assert _stair_links(b) == [("n004", "n005")]  # floors 3 and 4 now share a staircase


def test_set_walk_floor_drops_stale_floor_link():
    b = _adopted(upper_stairs_floor=3)
    assert _stair_links(b) == [("n004", "n005")]  # floor 2 <-> floor 3
    mapper.set_walk_floor(b, "vB", 3, log=_quiet)  # both staircases now on floor 3
    assert _stair_links(b) == []


def test_set_walk_floor_without_adoption_uses_current_floor_not_stale_offset():
    b = _bundle()
    _video(b, "vB").floor_offset = 7  # stale offset must not drive the shift
    mapper.set_walk_floor(b, "vB", 1, log=_quiet)
    assert _floors(b)["n002"] == 1 and _floors(b)["n003"] == 1


# ---------- API: PATCH /videos/{id} ----------
@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BUILDINGS_DIR", tmp_path / "buildings")

    def no_gemma(*_a, **_k):
        raise gemma.GemmaUnavailable("disabled in tests")

    monkeypatch.setattr(gemma, "resolve_model", no_gemma)
    bid = store.create_building("Floor Hall").building_id
    b = _adopted()
    b.building_id = bid
    store.save_map(b)
    return TestClient(app), bid


def _patch(c, bid, vid, floor):
    return c.patch(f"/api/buildings/{bid}/videos/{vid}", json={"start_floor": floor})


def test_patch_start_floor_after_adoption(client):
    c, bid = client
    before = _floors(store.load_map(bid))
    r = _patch(c, bid, "vB", 2)
    assert r.status_code == 200, r.text
    assert r.json()["floor_offset"] == 2 and r.json()["start_floor"] == 2
    assert _floors(store.load_map(bid)) == before  # already on floor 2: nothing moves

    r = _patch(c, bid, "vB", 3)
    assert r.status_code == 200, r.text
    b = store.load_map(bid)
    f = _floors(b)
    assert f["n003"] == 3 and f["n004"] == 3 and f["n001"] == 2 and f["n005"] == 4
    assert _video(b, "vB").floor_offset == 3 and _video(b, "vB").start_floor == 3
    assert _stair_links(b) == [("n004", "n005")]


def test_patch_unknown_video_404(client):
    c, bid = client
    assert _patch(c, bid, "vZ", 2).status_code == 404


# ---------- probes: walk-source stair edges after a move, odd start_floor values ----------
def _climb_bundle():
    """vB climbs stairs itself (walk-source edge n004->n006); n006 is on floor 2 in vB's walk, shared with nothing."""
    b = _bundle()
    b.nodes.append(Node(id="n006", label="Upstairs Hall", floor=2, floor_source="default", video_id="vB", t=15))
    b.edges.append(Edge(id="e003", from_node="n004", to_node="n006", heading_out=90, heading_in=90,
                        heading_out_raw=90, heading_in_raw=90, video_id="vB", length_m=6,
                        vertical="stairs_up", source="walk"))
    return b


def test_move_whole_walk_keeps_walk_stairs_and_routes():
    from app import routing
    b = _climb_bundle()
    mapper.set_walk_floor(b, "vB", 3, log=_quiet)  # whole walk (all own nodes) moves up together
    f = _floors(b)
    assert f["n006"] - f["n004"] == 1
    mapper.finalize(b)
    xy = {n.id: (n.x, n.y) for n in b.nodes}
    assert abs(xy["n004"][0] - xy["n006"][0]) < 0.5 and abs(xy["n004"][1] - xy["n006"][1]) < 0.5
    res = routing.route(b, "n003", "n006")
    assert res.node_path[-1] == "n006"


def test_partial_move_leaves_walk_stair_edge_spanning_wrong_floors_but_does_not_crash():
    """Walk vB's stair edge touches a node shared with vA (not moved): label stays, floors no longer adjacent."""
    from app import routing
    b = _climb_bundle()
    b.edges[-1].to_node = "n005"  # stairs edge now ends at vA's node (shared/not moved)
    b.nodes = [n for n in b.nodes if n.id != "n006"]
    mapper.set_walk_floor(b, "vB", 4, log=_quiet)  # vB nodes -> floor 4 (+3 from 1); n005 stays on 4
    f = _floors(b)
    assert f["n004"] == f["n005"]  # edge now spans 0 floors
    mapper.finalize(b)  # must not raise
    xy = {n.id: (n.x, n.y) for n in b.nodes}
    assert routing.route(b, "n003", "n005").node_path[-1] == "n005"
    # report-only: the 0-floor stair edge pins both floors' stairs together in the layout
    assert abs(xy["n004"][0] - xy["n005"][0]) < 0.5


@pytest.mark.parametrize("floor", [0, -1, 10, 99])
def test_patch_out_of_range_start_floor(client, floor):
    c, bid = client
    before = store.load_map(bid).model_dump()
    r = _patch(c, bid, "vB", floor)
    assert r.status_code == 422, r.text
    assert store.load_map(bid).model_dump() == before  # rejected before anything is touched


@pytest.mark.parametrize("floor", [1, config.MAX_FLOOR])
def test_patch_boundary_start_floor(client, floor):
    c, bid = client
    r = _patch(c, bid, "vB", floor)
    assert r.status_code == 200, r.text
    assert r.json()["start_floor"] == floor
    assert _video(store.load_map(bid), "vB").start_floor == floor


@pytest.mark.parametrize("floor", ["0", "-1", "10", "99"])
def test_upload_out_of_range_start_floor(client, floor, monkeypatch):
    from app import jobs
    c, bid = client
    called = []
    monkeypatch.setattr(jobs, "submit_video", lambda *a, **k: called.append(k))
    r = c.post(f"/api/buildings/{bid}/videos", files={"video": ("w.mp4", b"x", "video/mp4")},
               data={"start_floor": floor})
    assert r.status_code == 422, r.text
    assert called == []
