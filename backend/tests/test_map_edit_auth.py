"""Map-structure edits are owner only: PUT map, merges, PATCH/DELETE video, consolidate and rebuild.
Reads stay open. A contributor session gets 403, no credentials gets 401, and the legacy X-Owner-Token counts as the
owner. Each refused request is checked for no side effects: the map on disk is unchanged and no job was queued.
"""
import pytest
from fastapi.testclient import TestClient

from app import auth, config, jobs, store
from app.main import app
from app.schemas import Edge, Node, SuggestedMerge, VideoInfo
from tests.auth_helpers import owner_headers, signed_in_headers

client = TestClient(app)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BUILDINGS_DIR", tmp_path / "buildings")
    bundle = store.create_building("Edit Hall")
    bundle.nodes = [Node(id="n001", label="Lobby", video_id="v1"), Node(id="n002", label="Room 101", video_id="v1"),
                    Node(id="n003", label="Stairs", video_id="v1")]
    bundle.edges = [Edge(id="e001", from_node="n001", to_node="n002", heading_out=0, heading_in=0, video_id="v1")]
    bundle.videos = [VideoInfo(id="v1", filename="walk.mp4", status="done", start_floor=1, floor_offset=0,
                               moderation_status="approved")]
    bundle.suggested_merges = [SuggestedMerge(id="m1", keep="n001", remove="n003", confidence=0.6, reason="same sign")]
    store.save_map(bundle)
    jobs_seen = []
    monkeypatch.setattr(jobs, "submit_consolidate", lambda bid: jobs_seen.append(("consolidate", bid)) or _job())
    monkeypatch.setattr(jobs, "submit_rebuild", lambda bid: jobs_seen.append(("rebuild", bid)) or _job())
    return {"bid": bundle.building_id, "jobs": jobs_seen, "headers": owner_headers()}


def _job():
    return jobs.Job(id="j1", building_id="b", video_id=None, kind="rebuild")


def _put_body(bid):
    m = store.load_map(bid)
    return {"nodes": [n.model_dump() for n in m.nodes], "edges": [e.model_dump() for e in m.edges]}


# (method, path template, json body): one representative per owner-only edit route
GATED = [
    ("PUT", "/api/buildings/{bid}/map", "put"),
    ("POST", "/api/buildings/{bid}/merges/m1", {"accept": True}),
    ("POST", "/api/buildings/{bid}/merge", {"keep": "n001", "remove": "n002"}),
    ("PATCH", "/api/buildings/{bid}/videos/v1", {"start_floor": 2}),
    ("DELETE", "/api/buildings/{bid}/videos/v1", None),
    ("POST", "/api/buildings/{bid}/consolidate", None),
    ("POST", "/api/buildings/{bid}/rebuild", None),
]


def _call(env, method, path, body, headers):
    if body == "put":
        body = _put_body(env["bid"])
    return client.request(method, path.format(bid=env["bid"]), json=body, headers=headers)


@pytest.mark.parametrize("method,path,body", GATED, ids=[f"{m} {p}" for m, p, _ in GATED])
def test_edit_route_without_credentials_is_401_and_changes_nothing(env, method, path, body):
    before = store.load_map(env["bid"]).model_dump()
    r = _call(env, method, path, body, headers={})
    assert r.status_code == 401, r.text
    assert r.json()["detail"] == "Owner sign-in required"
    assert store.load_map(env["bid"]).model_dump() == before
    assert env["jobs"] == []


@pytest.mark.parametrize("method,path,body", GATED, ids=[f"{m} {p}" for m, p, _ in GATED])
def test_edit_route_with_bad_session_is_401(env, method, path, body):
    r = _call(env, method, path, body, headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401
    assert env["jobs"] == []


@pytest.mark.parametrize("method,path,body", GATED, ids=[f"{m} {p}" for m, p, _ in GATED])
def test_edit_route_by_contributor_is_403_and_changes_nothing(env, method, path, body):
    before = store.load_map(env["bid"]).model_dump()
    r = _call(env, method, path, body, headers=signed_in_headers())
    assert r.status_code == 403, r.text
    assert store.load_map(env["bid"]).model_dump() == before
    assert env["jobs"] == []


def test_expired_owner_session_cannot_edit(env, monkeypatch):
    token = auth.create_session(store.create_account_if_missing("owner@example.com").id)
    monkeypatch.setattr(auth, "_now", lambda: 10**12)
    r = client.patch(f"/api/buildings/{env['bid']}/videos/v1", json={"start_floor": 2},
                     headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 401


def test_owner_session_passes_for_put_map(env):
    r = client.put(f"/api/buildings/{env['bid']}/map", json=_put_body(env["bid"]), headers=env["headers"])
    assert r.status_code == 200, r.text


def test_owner_session_passes_for_video_patch(env):
    r = client.patch(f"/api/buildings/{env['bid']}/videos/v1", json={"start_floor": 2}, headers=env["headers"])
    assert r.status_code == 200, r.text
    assert store.load_map(env["bid"]).videos[0].start_floor == 2


def test_owner_session_passes_for_consolidate_and_rebuild(env):
    assert client.post(f"/api/buildings/{env['bid']}/consolidate", headers=env["headers"]).status_code == 200
    assert client.post(f"/api/buildings/{env['bid']}/rebuild", headers=env["headers"]).status_code == 200
    assert [k for k, _ in env["jobs"]] == ["consolidate", "rebuild"]


def test_legacy_token_alone_passes_owner_edits(env, monkeypatch):
    monkeypatch.setattr(config, "OWNER_TOKEN", "s3cret")
    r = client.patch(f"/api/buildings/{env['bid']}/videos/v1", json={"start_floor": 2},
                     headers={"X-Owner-Token": "s3cret"})
    assert r.status_code == 200, r.text


def test_wrong_token_without_session_is_refused_by_the_gate(env, monkeypatch):
    monkeypatch.setattr(config, "OWNER_TOKEN", "s3cret")
    r = client.patch(f"/api/buildings/{env['bid']}/videos/v1", json={"start_floor": 2},
                     headers={"X-Owner-Token": "nope"})
    assert r.status_code == 401 and r.json() == {"detail": "Owner token or sign-in required"}


def test_contributor_session_with_token_set_gets_403_from_the_route(env, monkeypatch):
    monkeypatch.setattr(config, "OWNER_TOKEN", "s3cret")
    r = client.patch(f"/api/buildings/{env['bid']}/videos/v1", json={"start_floor": 2}, headers=signed_in_headers())
    assert r.status_code == 403


@pytest.mark.parametrize("path", ["/api/buildings/{bid}/map", "/api/buildings/{bid}/videos/v1/debug"])
def test_reads_stay_open_without_session(env, path):
    r = client.get(path.format(bid=env["bid"]))
    assert r.status_code != 401  # the debug view is 404 here only because this walk has no track.json


def test_put_map_keeps_server_owned_contributor_id(env):
    """A place's contributor_id comes from disk: the editor cannot change it, and a new place cannot claim one."""
    bundle = store.load_map(env["bid"])
    bundle.nodes[0].contributor_id = "real-owner"
    store.save_map(bundle)
    body = _put_body(env["bid"])
    body["nodes"][0]["contributor_id"] = "forged"  # existing place
    body["nodes"].append({"id": "n009", "label": "New place", "contributor_id": "forged"})  # new place
    r = client.put(f"/api/buildings/{env['bid']}/map", json=body, headers=env["headers"])
    assert r.status_code == 200, r.text
    saved = {n.id: n.contributor_id for n in store.load_map(env["bid"]).nodes}
    assert saved["n001"] == "real-owner"
    assert saved["n009"] is None
