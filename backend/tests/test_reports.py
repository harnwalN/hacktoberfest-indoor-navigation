"""Problem reports ('something's wrong here'): anyone can file and resolve, validation, the storage cap, status
history, storage and privacy.

Each building has nodes n1 "Lobby" and n2 "Office" and one walk v1200001234.
"""
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import config, jobs, store
from app.main import app
from app.schemas import Node, VideoInfo
from tests.auth_helpers import owner_headers

RECORD_KEYS = {"type", "id", "kind", "note", "node_id", "step_index", "from_node", "to_node", "created_at"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BUILDINGS_DIR", tmp_path / "buildings")
    monkeypatch.setattr(jobs, "JOBS", {})
    a = store.create_building("Hall A").building_id
    b = store.create_building("Hall B").building_id
    for bid in (a, b):
        bundle = store.load_map(bid)
        bundle.nodes = [Node(id="n1", label="Lobby"), Node(id="n2", label="Office")]
        bundle.videos = [VideoInfo(id="v1200001234", filename="walk.mp4")]
        store.save_map(bundle)
    return SimpleNamespace(client=TestClient(app), a=a, b=b)


def _post(env, bid, **body):
    body.setdefault("kind", "wrong_direction")
    return env.client.post(f"/api/buildings/{bid}/reports", json=body)


def _file(bid):
    return store.building_dir(bid) / "reports.jsonl"


def _lines(bid) -> list[dict]:
    return [json.loads(line) for line in _file(bid).read_text().splitlines() if line.strip()]


# ---------- anyone can file, list and resolve ----------
def test_anyone_can_post_and_list_the_owner_resolves_and_reopens(env):
    c = env.client
    url = f"/api/buildings/{env.a}/reports"
    r = _post(env, env.a, kind="blocked", note="Door 4 is locked")
    assert r.status_code == 200, r.text
    assert set(r.json()) == {"id", "status"} and r.json()["status"] == "open"
    rid = r.json()["id"]
    assert [x["id"] for x in c.get(url).json()] == [rid]
    r = c.patch(f"{url}/{rid}", headers=owner_headers(), json={"status": "resolved"})
    assert r.status_code == 200 and r.json()["status"] == "resolved"
    assert c.get(url).json() == []
    assert c.patch(f"{url}/{rid}", headers=owner_headers(), json={"status": "open"}).json()["status"] == "open"


def test_reports_are_per_building(env):
    rid = _post(env, env.a, kind="other").json()["id"]
    assert env.client.get(f"/api/buildings/{env.b}/reports").json() == []
    assert env.client.patch(f"/api/buildings/{env.b}/reports/{rid}", headers=owner_headers(), json={"status": "resolved"}).status_code == 404


def test_unknown_building_gets_404(env):
    assert env.client.post("/api/buildings/nope/reports", json={"kind": "other"}).status_code == 404
    assert env.client.get("/api/buildings/nope/reports").status_code == 404


# ---------- validation ----------
def test_bad_input_is_refused_with_422_and_a_clear_detail(env):
    cases = {
        "bad kind": {"kind": "teleport"},
        "kind missing": {"note": "hi"},
        "note too long": {"kind": "other", "note": "x" * 501},
        "unknown node_id": {"kind": "other", "node_id": "n999"},
        "unknown from_node": {"kind": "other", "from_node": "nope"},
        "unknown to_node": {"kind": "other", "to_node": "nope"},
        "step_index too big": {"kind": "other", "step_index": 501},
        "step_index negative": {"kind": "other", "step_index": -1},
        "step_index a string": {"kind": "other", "step_index": "3"},
        "note not a string": {"kind": "other", "note": 5},
    }
    for name, body in cases.items():
        r = env.client.post(f"/api/buildings/{env.a}/reports", json=body)  # body exactly as given
        assert r.status_code == 422, (name, r.status_code, r.text)
        assert r.json()["detail"], name
    assert "node_id" in _post(env, env.a, kind="other", node_id="n999").text
    assert "500 characters" in _post(env, env.a, kind="other", note="x" * 501).text
    assert "step_index" in _post(env, env.a, kind="other", step_index=501).text
    assert _file(env.a).exists() is False  # nothing was written by any refused request


def test_note_is_stripped_before_the_length_check(env):
    assert _post(env, env.a, kind="other", note="   " + "x" * 500 + "   ").status_code == 200
    assert _post(env, env.a, kind="other", note="   ").status_code == 200
    recs = [r for r in _lines(env.a) if r["type"] == "report"]
    assert recs[0]["note"] == "x" * 500
    assert recs[1]["note"] is None


def test_valid_optional_fields_are_stored_and_labelled(env):
    body = {"kind": "sign_changed", "note": "  Now says 210 ", "node_id": "n2", "step_index": 0,
            "from_node": "n1", "to_node": "n2"}
    rid = _post(env, env.a, **body).json()["id"]
    item = env.client.get(f"/api/buildings/{env.a}/reports").json()[0]
    assert item["id"] == rid and item["note"] == "Now says 210" and item["node_label"] == "Office"
    assert (item["node_id"], item["step_index"], item["from_node"], item["to_node"]) == ("n2", 0, "n1", "n2")
    assert item["kind"] == "sign_changed" and item["status"] == "open" and item["resolution_note"] is None
    assert item["history"][0]["status"] == "open"
    assert "reporter" not in item and "by" not in item["history"][0]


# ---------- limits ----------
def test_building_cap_gives_507_and_resolved_reports_still_count(env, monkeypatch):
    monkeypatch.setattr(config, "REPORTS_MAX", 2)
    rid = _post(env, env.a, kind="other").json()["id"]
    assert _post(env, env.a, kind="other").status_code == 200
    r = _post(env, env.a, kind="other")
    assert r.status_code == 507 and "2" in r.json()["detail"]
    # resolving does not free a slot: the cap counts open and resolved reports together
    assert env.client.patch(f"/api/buildings/{env.a}/reports/{rid}", headers=owner_headers(), json={"status": "resolved"}).status_code == 200
    assert _post(env, env.a, kind="other").status_code == 507
    # the other building has its own cap
    assert _post(env, env.b, kind="other").status_code == 200


# ---------- status folding and listing ----------
def test_resolve_then_reopen_keeps_history_and_the_right_status(env):
    rid = _post(env, env.a, kind="sign_changed", note="Sign says 210").json()["id"]
    url = f"/api/buildings/{env.a}/reports/{rid}"
    c = env.client

    r = c.patch(url, headers=owner_headers(), json={"status": "resolved", "resolution_note": "  Re-hung the sign  "})
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "resolved" and r.json()["resolution_note"] == "Re-hung the sign"
    assert [x["id"] for x in c.get(f"/api/buildings/{env.a}/reports").json()] == []
    assert [x["id"] for x in c.get(f"/api/buildings/{env.a}/reports", params={"status": "resolved"}).json()] == [rid]

    r = c.patch(url, headers=owner_headers(), json={"status": "open", "resolution_note": "ignored when reopening"})
    assert r.status_code == 200
    item = r.json()
    assert item["status"] == "open" and item["resolution_note"] is None
    assert [h["status"] for h in item["history"]] == ["open", "resolved", "open"]
    assert item["history"][1]["resolution_note"] == "Re-hung the sign"
    assert item["history"][2]["resolution_note"] is None
    assert all("by" not in h for h in item["history"])
    assert item["created_at"] == item["history"][0]["at"]
    # the file has the same story: one report line and two status lines, nothing rewritten
    assert [r["type"] for r in _lines(env.a)] == ["report", "status", "status"]


def test_patch_unknown_report_and_bad_bodies(env):
    c = env.client
    url = f"/api/buildings/{env.a}/reports"
    assert c.patch(f"{url}/nope", headers=owner_headers(), json={"status": "resolved"}).status_code == 404
    rid = _post(env, env.a, kind="other").json()["id"]
    assert c.patch(f"{url}/{rid}", headers=owner_headers(), json={"status": "closed"}).status_code == 422
    assert c.patch(f"{url}/{rid}", headers=owner_headers(), json={"status": "resolved", "resolution_note": "x" * 301}).status_code == 422
    assert c.patch(f"{url}/{rid}", headers=owner_headers(), json={"status": "resolved", "resolution_note": "x" * 300}).status_code == 200


def test_list_filters_order_and_limit(env):
    ids = [_post(env, env.a, kind="other", note=f"r{i}").json()["id"] for i in range(5)]
    c = env.client
    url = f"/api/buildings/{env.a}/reports"

    rows = c.get(url).json()
    assert [x["note"] for x in rows] == ["r4", "r3", "r2", "r1", "r0"]  # newest first
    assert [x["note"] for x in c.get(url, params={"limit": 2}).json()] == ["r4", "r3"]
    assert c.get(url, params={"limit": 500}).status_code == 200

    c.patch(f"{url}/{ids[2]}", headers=owner_headers(), json={"status": "resolved"})
    assert [x["note"] for x in c.get(url, params={"status": "resolved"}).json()] == ["r2"]
    assert len(c.get(url, params={"status": "open"}).json()) == 4
    assert len(c.get(url, params={"status": "all"}).json()) == 5
    assert [x["note"] for x in c.get(url, params={"status": "all", "limit": 1}).json()] == ["r4"]

    assert c.get(url, params={"status": "closed"}).status_code == 422
    assert c.get(url, params={"limit": 0}).status_code == 422
    assert c.get(url, params={"limit": 501}).status_code == 422


def test_node_label_is_null_once_the_place_is_gone(env):
    c = env.client
    rid = _post(env, env.a, kind="place_missing", node_id="n2").json()["id"]
    bundle = store.load_map(env.a)
    bundle.nodes = [n for n in bundle.nodes if n.id != "n2"]
    store.save_map(bundle)
    item = c.get(f"/api/buildings/{env.a}/reports").json()[0]
    assert item["id"] == rid and item["node_id"] == "n2" and item["node_label"] is None


def test_reports_survive_a_video_delete(env):
    rid = _post(env, env.a, kind="photo_wrong", note="Photo is from the wrong floor").json()["id"]
    r = env.client.delete(f"/api/buildings/{env.a}/videos/v1200001234", headers=owner_headers())
    assert r.status_code == 200, r.text
    rows = env.client.get(f"/api/buildings/{env.a}/reports").json()
    assert [x["id"] for x in rows] == [rid]


# ---------- storage and privacy ----------
def test_file_holds_only_the_report_fields(env):
    _post(env, env.a, kind="other", note="from the desk")
    rid = _post(env, env.a, kind="other", note="second").json()["id"]
    env.client.patch(f"/api/buildings/{env.a}/reports/{rid}", headers=owner_headers(), json={"status": "resolved"})
    recs = _lines(env.a)
    reports = [r for r in recs if r["type"] == "report"]
    assert len(reports) == 2 and all(set(r) == RECORD_KEYS for r in reports)
    statuses = [r for r in recs if r["type"] == "status"]
    assert len(statuses) == 1 and set(statuses[0]) == {"type", "id", "status", "resolution_note", "at"}


def test_reports_file_is_not_served_and_not_in_map_responses(env):
    _post(env, env.a, kind="other", note="secret-note-xyz")
    c = env.client
    names = ["reports.jsonl", "..%2Freports.jsonl", "reports.jsonl%00"]
    for name in names:
        assert c.get(f"/bfiles/{env.a}/{name}").status_code == 404
        assert c.get(f"/bfiles/{env.a}/frames/{name}").status_code == 404
    assert c.get(f"/bfiles/{env.a}/frames/reports.jsonl?exp=4102444800&sig=abc").status_code == 404
    text = c.get(f"/api/buildings/{env.a}/map").text
    assert "secret-note-xyz" not in text and "reports" not in text


def test_torn_and_garbage_lines_are_skipped_and_appends_start_fresh(env):
    rid = _post(env, env.a, kind="other", note="before the crash").json()["id"]
    with open(_file(env.a), "ab") as f:
        f.write(b'not json at all\n[1, 2]\n{"type": "mystery", "id": "x"}\n{"type": "report", "id": "zz')  # torn
    c = env.client
    url = f"/api/buildings/{env.a}/reports"
    rows = c.get(url, params={"status": "all"})
    assert rows.status_code == 200 and [x["id"] for x in rows.json()] == [rid]

    # a status change after the torn line is written on a fresh line, not glued onto the fragment
    assert c.patch(f"{url}/{rid}", headers=owner_headers(), json={"status": "resolved"}).status_code == 200
    new_id = _post(env, env.a, kind="other", note="after the crash").json()["id"]
    rows = c.get(url, params={"status": "all"}).json()
    assert {x["id"] for x in rows} == {rid, new_id}
    by_id = {x["id"]: x for x in rows}
    assert by_id[rid]["status"] == "resolved" and by_id[new_id]["status"] == "open"
