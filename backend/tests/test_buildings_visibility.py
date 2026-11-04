"""Building list visibility (GET /api/buildings: a building without a map is hidden from everyone but owners) and
building deletion (DELETE /api/buildings/{id}, owner only, 204)."""
import pytest
from fastapi.testclient import TestClient

from app import config, jobs, store
from app.main import app
from app.schemas import Node
from tests.auth_helpers import owner_headers, signed_in_headers

client = TestClient(app)


def _names(rows):
    return sorted(r["name"] for r in rows)


@pytest.fixture
def buildings():
    """One building with a map (two places) and one empty building."""
    mapped = store.create_building("Mapped Hall").building_id
    b = store.load_map(mapped)
    b.nodes = [Node(id="n1", label="Lobby"), Node(id="n2", label="Room 2")]
    store.save_map(b)
    empty = store.create_building("Empty Hall").building_id
    return mapped, empty


# ---------- GET /api/buildings ----------
def test_public_list_hides_buildings_without_a_map(buildings):
    assert _names(client.get("/api/buildings").json()) == ["Mapped Hall"]


def test_contributor_list_hides_buildings_without_a_map(buildings):
    assert _names(client.get("/api/buildings", headers=signed_in_headers()).json()) == ["Mapped Hall"]


def test_owner_list_shows_every_building_with_its_node_count(buildings):
    rows = {r["name"]: r for r in client.get("/api/buildings", headers=owner_headers()).json()}
    assert set(rows) == {"Mapped Hall", "Empty Hall"}
    assert rows["Mapped Hall"]["nodes"] == 2 and rows["Empty Hall"]["nodes"] == 0


def test_legacy_token_sees_every_building(buildings, monkeypatch):
    monkeypatch.setattr(config, "OWNER_TOKEN", "s3cret")
    assert _names(client.get("/api/buildings", headers={"X-Owner-Token": "s3cret"}).json()) == ["Empty Hall", "Mapped Hall"]


def test_bad_session_on_the_list_is_treated_as_public_not_refused(buildings):
    r = client.get("/api/buildings", headers={"Authorization": "Bearer nope"})
    assert r.status_code == 200 and _names(r.json()) == ["Mapped Hall"]


def test_building_list_detail_for_a_direct_empty_map_is_still_readable(buildings):
    """Hiding is for the list only: a link to an empty building's map still answers."""
    _, empty = buildings
    assert client.get(f"/api/buildings/{empty}/map").status_code == 200


# ---------- DELETE /api/buildings/{id} ----------
def test_owner_deletes_a_building_with_204_and_its_folder(buildings):
    mapped, _ = buildings
    client.post("/api/buildings/{}/reports".format(mapped), json={"kind": "other", "note": "x"})
    assert store.building_dir(mapped).exists()
    r = client.delete(f"/api/buildings/{mapped}", headers=owner_headers())
    assert r.status_code == 204 and r.content == b""
    assert not store.building_dir(mapped).exists()
    assert client.get(f"/api/buildings/{mapped}/map").status_code == 404
    assert client.get(f"/api/buildings/{mapped}/reports").status_code == 404


def test_delete_removes_the_moderator_grant(buildings):
    mapped, _ = buildings
    owner = store.create_account_if_missing("owner@example.com")
    store.add_moderator(owner.id, mapped)
    client.delete(f"/api/buildings/{mapped}", headers=owner_headers())
    assert store.get_account("owner@example.com").moderator_of == []


def test_delete_is_refused_for_anonymous_and_contributors(buildings):
    mapped, _ = buildings
    assert client.delete(f"/api/buildings/{mapped}").status_code == 401
    assert client.delete(f"/api/buildings/{mapped}", headers=signed_in_headers()).status_code == 403
    assert store.building_dir(mapped).exists()


def test_delete_unknown_or_invalid_building_is_404(buildings):
    assert client.delete("/api/buildings/no-such-hall", headers=owner_headers()).status_code == 404
    assert client.delete("/api/buildings/..%2Fetc", headers=owner_headers()).status_code == 404


def test_delete_stops_the_building_queued_jobs(buildings, monkeypatch):
    monkeypatch.setattr(jobs, "JOBS", {})
    monkeypatch.setattr(jobs, "_ensure_worker", lambda: None)
    mapped, empty = buildings
    queued = jobs.Job(id="jdel1", building_id=mapped, video_id=None, kind="rebuild", status="queued")
    paused = jobs.Job(id="jdel2", building_id=mapped, video_id=None, kind="rebuild", status="paused")
    other = jobs.Job(id="jdel3", building_id=empty, video_id=None, kind="rebuild", status="queued")
    jobs.JOBS.update({j.id: j for j in (queued, paused, other)})
    assert client.delete(f"/api/buildings/{mapped}", headers=owner_headers()).status_code == 204
    assert queued.status == "cancelled" and paused.status == "cancelled"
    assert other.status == "queued"  # another building's job is untouched


def test_a_deleted_building_can_be_created_again_with_the_same_name(buildings):
    mapped, _ = buildings
    client.delete(f"/api/buildings/{mapped}", headers=owner_headers())
    r = client.post("/api/buildings", json={"name": "Mapped Hall"}, headers=owner_headers())
    assert r.status_code == 200 and r.json()["building_id"] == mapped
