"""GET /buildings/{id}/floors/{floor}/map: one floor's places, its connections and its links to other floors."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import config, router, store
from app.schemas import Edge, Node


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BUILDINGS_DIR", tmp_path / "buildings")
    app = FastAPI()
    app.include_router(router.api)  # bare router: no startup recovery, no static mounts
    bundle = store.create_building("Two Floors")
    bundle.nodes = [
        Node(id="n001", label="Lobby", floor=1, frame_id="a.jpg"),
        Node(id="n002", label="Room 101", floor=1),
        Node(id="n003", label="Stairs 1", floor=1, vertical="stairs"),
        Node(id="n004", label="Stairs 2", floor=2, vertical="stairs"),
        Node(id="n005", label="Room 201", floor=2, frame_id="b.jpg"),
    ]
    bundle.edges = [
        Edge(id="e001", from_node="n001", to_node="n002", heading_out=0, heading_in=0,
             frames_fwd=["a.jpg", "c.jpg"], frames_rev=["d.jpg"]),
        Edge(id="e002", from_node="n002", to_node="n003", heading_out=90, heading_in=90),
        Edge(id="e003", from_node="n003", to_node="n004", heading_out=0, heading_in=0, vertical="stairs_up"),
        Edge(id="e004", from_node="n004", to_node="n005", heading_out=90, heading_in=90),
    ]
    store.save_map(bundle)
    return {"client": TestClient(app), "bid": bundle.building_id}


def _get(env, floor, bid=None):
    return env["client"].get(f"/api/buildings/{bid or env['bid']}/floors/{floor}/map")


def test_floor_one_has_its_places_and_connections_only(env):
    r = _get(env, 1)
    assert r.status_code == 200
    body = r.json()
    assert body["building_id"] == env["bid"] and body["floor"] == 1
    assert [n["id"] for n in body["nodes"]] == ["n001", "n002", "n003"]
    assert [e["id"] for e in body["edges"]] == ["e001", "e002"]
    # e003 leaves floor 1 for floor 2: it is a link, not a floor-1 connection
    assert [e["id"] for e in body["links"]] == ["e003"]


def test_floor_two_mirrors_floor_one(env):
    body = _get(env, 2).json()
    assert [n["id"] for n in body["nodes"]] == ["n004", "n005"]
    assert [e["id"] for e in body["edges"]] == ["e004"]
    assert [e["id"] for e in body["links"]] == ["e003"]


def test_photo_urls_are_built_per_response(env):
    body = _get(env, 1).json()
    lobby = next(n for n in body["nodes"] if n["id"] == "n001")
    assert lobby["frame_url"] == f"/bfiles/{env['bid']}/frames/a.jpg"
    assert next(n for n in body["nodes"] if n["id"] == "n002")["frame_url"] is None
    e001 = next(e for e in body["edges"] if e["id"] == "e001")
    assert e001["frames_fwd_urls"] == [f"/bfiles/{env['bid']}/frames/a.jpg", f"/bfiles/{env['bid']}/frames/c.jpg"]
    assert e001["frames_rev_urls"] == [f"/bfiles/{env['bid']}/frames/d.jpg"]
    assert next(e for e in body["edges"] if e["id"] == "e002")["frames_rev_urls"] is None


def test_floor_response_does_not_change_the_map_on_disk(env):
    before = store.load_map(env["bid"])
    _get(env, 1)
    after = store.load_map(env["bid"])
    assert after.model_dump() == before.model_dump()


def test_floor_with_no_places_is_an_empty_map(env):
    r = _get(env, 5)
    assert r.status_code == 200
    assert r.json() == {"building_id": env["bid"], "floor": 5, "nodes": [], "edges": [], "links": []}


@pytest.mark.parametrize("floor", [0, config.MAX_FLOOR + 1])
def test_floor_outside_range_is_422(env, floor):
    assert _get(env, floor).status_code == 422


def test_non_numeric_floor_is_422(env):
    assert _get(env, "ground").status_code == 422


def test_unknown_building_is_404(env):
    r = _get(env, 1, bid="no-such-building")
    assert r.status_code == 404
    assert r.json()["detail"] == "Unknown building 'no-such-building'"
