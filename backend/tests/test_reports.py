"""Problem reports ('something's wrong here'): roles, validation, limits, status history, storage and privacy.

Reuses the env fixture from test_auth.py (AUTH_REQUIRED=1, two buildings, owner key OWNER). Each building has
nodes n1 "Lobby" and n2 "Office".
"""
import hashlib
import json
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from app import auth, config, store
from app.main import app
from app.schemas import Node
from tests.test_auth import OWNER, H, _invite, _token, env  # noqa: F401  (env is a fixture)


def _post(env, bid, token, **body):
    body.setdefault("kind", "wrong_direction")
    return env.client.post(f"/api/buildings/{bid}/reports", json=body, headers=H(token))


def _file(bid):
    return store.building_dir(bid) / "reports.jsonl"


def _lines(bid) -> list[dict]:
    return [json.loads(line) for line in _file(bid).read_text().splitlines() if line.strip()]


def _age_reports(bid, hours: float) -> None:
    """Move every report's created_at back, to test the one-hour window without waiting."""
    old = (datetime.now(UTC) - timedelta(hours=hours)).isoformat(timespec="seconds")
    out = []
    for rec in _lines(bid):
        if rec["type"] == "report":
            rec["created_at"] = old
        out.append(json.dumps(rec))
    _file(bid).write_text("\n".join(out) + "\n")


# ---------- roles and identity ----------
def test_viewer_can_post_but_not_list_or_patch(env):
    tok = _token(env, env.a, "viewer")
    r = _post(env, env.a, tok, kind="blocked", note="Door 4 is locked")
    assert r.status_code == 200, r.text
    assert set(r.json()) == {"id", "status"} and r.json()["status"] == "open"
    rid = r.json()["id"]
    assert env.client.get(f"/api/buildings/{env.a}/reports", headers=H(tok)).status_code == 403
    r = env.client.patch(f"/api/buildings/{env.a}/reports/{rid}", json={"status": "resolved"}, headers=H(tok))
    assert r.status_code == 403


def test_editor_can_post_list_and_patch(env):
    tok = _token(env, env.a, "editor")
    rid = _post(env, env.a, tok, kind="other").json()["id"]
    r = env.client.get(f"/api/buildings/{env.a}/reports", headers=H(tok))
    assert r.status_code == 200 and [x["id"] for x in r.json()] == [rid]
    r = env.client.patch(f"/api/buildings/{env.a}/reports/{rid}", json={"status": "resolved"}, headers=H(tok))
    assert r.status_code == 200 and r.json()["status"] == "resolved"


def test_owner_key_can_post_list_and_patch(env):
    rid = _post(env, env.a, OWNER, kind="other").json()["id"]
    assert [x["id"] for x in env.client.get(f"/api/buildings/{env.a}/reports", headers=H(OWNER)).json()] == [rid]
    r = env.client.patch(f"/api/buildings/{env.a}/reports/{rid}", json={"status": "resolved"}, headers=H(OWNER))
    assert r.status_code == 200


def test_other_buildings_token_and_no_token_get_401(env):
    editor_a = _token(env, env.a, "editor")
    editor_b = _token(env, env.b, "editor")
    rid = _post(env, env.a, editor_a, kind="other").json()["id"]
    url = f"/api/buildings/{env.a}/reports"
    assert _post(env, env.a, editor_b, kind="other").status_code == 401
    assert env.client.get(url, headers=H(editor_b)).status_code == 401
    assert env.client.patch(f"{url}/{rid}", json={"status": "resolved"}, headers=H(editor_b)).status_code == 401
    assert env.client.post(url, json={"kind": "other"}).status_code == 401
    assert env.client.get(url).status_code == 401
    assert env.client.patch(f"{url}/{rid}", json={"status": "resolved"}).status_code == 401
    # the refused attempts stored nothing, and nothing was resolved
    rows = env.client.get(url, headers=H(OWNER), params={"status": "all"}).json()
    assert [(x["id"], x["status"]) for x in rows] == [(rid, "open")]


# ---------- validation ----------
def test_bad_input_is_refused_with_422_and_a_clear_detail(env):
    tok = _token(env, env.a, "viewer")
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
        r = env.client.post(f"/api/buildings/{env.a}/reports", json=body, headers=H(tok))  # body exactly as given
        assert r.status_code == 422, (name, r.status_code, r.text)
        assert r.json()["detail"], name
    assert "node_id" in _post(env, env.a, tok, kind="other", node_id="n999").text
    assert "500 characters" in _post(env, env.a, tok, kind="other", note="x" * 501).text
    assert "step_index" in _post(env, env.a, tok, kind="other", step_index=501).text
    assert _file(env.a).exists() is False  # nothing was written by any refused request


def test_note_is_stripped_before_the_length_check(env):
    tok = _token(env, env.a, "viewer")
    assert _post(env, env.a, tok, kind="other", note="   " + "x" * 500 + "   ").status_code == 200
    assert _post(env, env.a, tok, kind="other", note="   ").status_code == 200
    recs = [r for r in _lines(env.a) if r["type"] == "report"]
    assert recs[0]["note"] == "x" * 500
    assert recs[1]["note"] is None


def test_valid_optional_fields_are_stored_and_labelled(env):
    inv = _invite(env, env.a, "viewer")
    body = {"kind": "sign_changed", "note": "  Now says 210 ", "node_id": "n2", "step_index": 0,
            "from_node": "n1", "to_node": "n2"}
    rid = _post(env, env.a, inv["token"], **body).json()["id"]
    item = env.client.get(f"/api/buildings/{env.a}/reports", headers=H(OWNER)).json()[0]
    assert item["id"] == rid and item["note"] == "Now says 210" and item["node_label"] == "Office"
    assert (item["node_id"], item["step_index"], item["from_node"], item["to_node"]) == ("n2", 0, "n1", "n2")
    assert item["kind"] == "sign_changed" and item["status"] == "open" and item["resolution_note"] is None
    assert item["history"][0]["status"] == "open" and item["reporter"] == inv["id"]


# ---------- limits ----------
def test_rate_limit_is_per_reporter_and_gives_429_with_retry_after(env, monkeypatch):
    monkeypatch.setattr(config, "REPORTS_PER_HOUR", 3)
    tok1 = _token(env, env.a, "viewer")
    tok2 = _token(env, env.a, "viewer")
    for _ in range(3):
        assert _post(env, env.a, tok1).status_code == 200
    r = _post(env, env.a, tok1)
    assert r.status_code == 429, r.text
    assert 3500 <= int(r.headers["Retry-After"]) <= 3600
    assert "3 per hour" in r.json()["detail"]
    # another invite holder is not affected by the first one's count
    assert _post(env, env.a, tok2).status_code == 200
    # the owner key counts separately, and so does the same reporter on another building
    assert _post(env, env.a, OWNER).status_code == 200
    tok_b = _token(env, env.b, "viewer")
    assert all(_post(env, env.b, tok_b).status_code == 200 for _ in range(3))
    # the refused request was not stored: 3 + 1 + 1 for building a
    assert len([r for r in _lines(env.a) if r["type"] == "report"]) == 5


def test_rate_limit_window_is_one_hour(env, monkeypatch):
    monkeypatch.setattr(config, "REPORTS_PER_HOUR", 2)
    tok = _token(env, env.a, "viewer")
    assert _post(env, env.a, tok).status_code == 200
    assert _post(env, env.a, tok).status_code == 200
    assert _post(env, env.a, tok).status_code == 429
    _age_reports(env.a, hours=2)  # the earlier reports are now outside the hour
    assert _post(env, env.a, tok).status_code == 200


def test_building_cap_gives_507_and_resolved_reports_still_count(env, monkeypatch):
    monkeypatch.setattr(config, "REPORTS_MAX", 2)
    rid = _post(env, env.a, OWNER, kind="other").json()["id"]
    assert _post(env, env.a, OWNER, kind="other").status_code == 200
    r = _post(env, env.a, OWNER, kind="other")
    assert r.status_code == 507 and "2" in r.json()["detail"]
    # resolving does not free a slot: the cap counts open and resolved reports together
    assert env.client.patch(f"/api/buildings/{env.a}/reports/{rid}", json={"status": "resolved"},
                            headers=H(OWNER)).status_code == 200
    assert _post(env, env.a, OWNER, kind="other").status_code == 507
    # the other building has its own cap
    assert _post(env, env.b, OWNER, kind="other").status_code == 200


# ---------- status folding and listing ----------
def test_resolve_then_reopen_keeps_history_and_the_right_status(env):
    viewer = _invite(env, env.a, "viewer", label="Front desk")
    editor = _invite(env, env.a, "editor", label="Ops")
    rid = _post(env, env.a, viewer["token"], kind="sign_changed", note="Sign says 210").json()["id"]
    url = f"/api/buildings/{env.a}/reports/{rid}"
    c = env.client

    r = c.patch(url, json={"status": "resolved", "resolution_note": "  Re-hung the sign  "}, headers=H(editor["token"]))
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "resolved" and r.json()["resolution_note"] == "Re-hung the sign"
    assert [x["id"] for x in c.get(f"/api/buildings/{env.a}/reports", headers=H(OWNER)).json()] == []
    assert [x["id"] for x in c.get(f"/api/buildings/{env.a}/reports", headers=H(OWNER),
                                    params={"status": "resolved"}).json()] == [rid]

    r = c.patch(url, json={"status": "open", "resolution_note": "ignored when reopening"}, headers=H(editor["token"]))
    assert r.status_code == 200
    item = r.json()
    assert item["status"] == "open" and item["resolution_note"] is None
    assert [h["status"] for h in item["history"]] == ["open", "resolved", "open"]
    assert item["history"][1]["resolution_note"] == "Re-hung the sign"
    assert item["history"][2]["resolution_note"] is None
    assert [h["by"] for h in item["history"]] == [viewer["id"], editor["id"], editor["id"]]
    assert item["created_at"] == item["history"][0]["at"]
    # the file has the same story: one report line and two status lines, nothing rewritten
    assert [r["type"] for r in _lines(env.a)] == ["report", "status", "status"]


def test_patch_unknown_report_and_bad_bodies(env):
    c = env.client
    url = f"/api/buildings/{env.a}/reports"
    assert c.patch(f"{url}/nope", json={"status": "resolved"}, headers=H(OWNER)).status_code == 404
    rid = _post(env, env.a, OWNER, kind="other").json()["id"]
    assert c.patch(f"{url}/{rid}", json={"status": "closed"}, headers=H(OWNER)).status_code == 422
    assert c.patch(f"{url}/{rid}", json={"status": "resolved", "resolution_note": "x" * 301},
                   headers=H(OWNER)).status_code == 422
    assert c.patch(f"{url}/{rid}", json={"status": "resolved", "resolution_note": "x" * 300},
                   headers=H(OWNER)).status_code == 200


def test_list_filters_order_and_limit(env):
    tok = _token(env, env.a, "viewer")
    ids = [_post(env, env.a, tok, kind="other", note=f"r{i}").json()["id"] for i in range(5)]
    c = env.client
    url = f"/api/buildings/{env.a}/reports"

    rows = c.get(url, headers=H(OWNER)).json()
    assert [x["note"] for x in rows] == ["r4", "r3", "r2", "r1", "r0"]  # newest first
    assert [x["note"] for x in c.get(url, headers=H(OWNER), params={"limit": 2}).json()] == ["r4", "r3"]
    assert c.get(url, headers=H(OWNER), params={"limit": 500}).status_code == 200

    c.patch(f"{url}/{ids[2]}", json={"status": "resolved"}, headers=H(OWNER))
    assert [x["note"] for x in c.get(url, headers=H(OWNER), params={"status": "resolved"}).json()] == ["r2"]
    assert len(c.get(url, headers=H(OWNER), params={"status": "open"}).json()) == 4
    assert len(c.get(url, headers=H(OWNER), params={"status": "all"}).json()) == 5
    assert [x["note"] for x in c.get(url, headers=H(OWNER), params={"status": "all", "limit": 1}).json()] == ["r4"]

    assert c.get(url, headers=H(OWNER), params={"status": "closed"}).status_code == 422
    assert c.get(url, headers=H(OWNER), params={"limit": 0}).status_code == 422
    assert c.get(url, headers=H(OWNER), params={"limit": 501}).status_code == 422


def test_node_label_is_null_once_the_place_is_gone(env):
    c = env.client
    tok = _token(env, env.a, "viewer")
    rid = _post(env, env.a, tok, kind="place_missing", node_id="n2").json()["id"]
    bundle = store.load_map(env.a)
    bundle.nodes = [n for n in bundle.nodes if n.id != "n2"]
    store.save_map(bundle)
    item = c.get(f"/api/buildings/{env.a}/reports", headers=H(OWNER)).json()[0]
    assert item["id"] == rid and item["node_id"] == "n2" and item["node_label"] is None


def test_reports_survive_a_video_delete(env):
    tok = _token(env, env.a, "viewer")
    rid = _post(env, env.a, tok, kind="photo_wrong", note="Photo is from the wrong floor").json()["id"]
    r = env.client.delete(f"/api/buildings/{env.a}/videos/v1200001234", headers=H(OWNER))
    assert r.status_code == 200, r.text
    rows = env.client.get(f"/api/buildings/{env.a}/reports", headers=H(OWNER)).json()
    assert [x["id"] for x in rows] == [rid]


# ---------- storage and privacy ----------
def test_file_holds_invite_id_or_owner_never_a_token_or_hash(env):
    inv = _invite(env, env.a, "viewer", label="Lobby desk")
    _post(env, env.a, inv["token"], kind="other", note="from the desk")
    _post(env, env.a, OWNER, kind="other", note="from the owner")
    text = _file(env.a).read_text()
    assert inv["token"] not in text and OWNER not in text
    assert hashlib.sha256(inv["token"].encode()).hexdigest() not in text
    assert hashlib.sha256(OWNER.encode()).hexdigest() not in text
    recs = [r for r in _lines(env.a) if r["type"] == "report"]
    assert [r["reporter"] for r in recs] == [inv["id"], "owner"]
    for rec in recs:
        assert not any("token" in k or "sha" in k or "hash" in k for k in rec), rec


def test_reports_file_is_not_served_and_not_in_map_responses(env):
    tok = _token(env, env.a, "editor")
    _post(env, env.a, tok, kind="other", note="secret-note-xyz")
    c = env.client
    exp = 4102444800  # far future, so only the file name decides
    names = ["reports.jsonl", "..%2Freports.jsonl", "reports.jsonl%00"]
    for name in names:
        assert c.get(f"/bfiles/{env.a}/{name}").status_code == 404
        assert c.get(f"/bfiles/{env.a}/frames/{name}").status_code == 404
    signed = auth._frame_sig(env.a, "reports.jsonl", exp)
    assert c.get(f"/bfiles/{env.a}/frames/reports.jsonl?exp={exp}&sig={signed}").status_code == 404
    assert c.get(f"/bfiles/{env.a}/reports.jsonl?exp={exp}&sig={signed}").status_code == 404
    for token in (OWNER, _token(env, env.a, "viewer")):
        text = c.get(f"/api/buildings/{env.a}/map", headers=H(token)).text
        assert "secret-note-xyz" not in text and "reports" not in text
    assert "secret-note-xyz" not in c.get(f"/api/buildings/{env.a}/map", headers=H(OWNER)).text


def test_access_shows_open_reports_to_editors_and_owner_only(env):
    viewer = _token(env, env.a, "viewer")
    editor = _token(env, env.a, "editor")
    rid = _post(env, env.a, viewer, kind="other").json()["id"]
    _post(env, env.a, viewer, kind="other")
    c = env.client
    assert c.get(f"/api/buildings/{env.a}/access", headers=H(viewer)).json() == {"role": "viewer"}
    assert c.get(f"/api/buildings/{env.a}/access", headers=H(editor)).json() == \
        {"role": "editor", "open_reports": 2}
    assert c.get(f"/api/buildings/{env.a}/access", headers=H(OWNER)).json() == {"role": "owner", "open_reports": 2}
    c.patch(f"/api/buildings/{env.a}/reports/{rid}", json={"status": "resolved"}, headers=H(editor))
    assert c.get(f"/api/buildings/{env.a}/access", headers=H(editor)).json()["open_reports"] == 1


def test_torn_and_garbage_lines_are_skipped_and_appends_start_fresh(env):
    tok = _token(env, env.a, "editor")
    rid = _post(env, env.a, tok, kind="other", note="before the crash").json()["id"]
    with open(_file(env.a), "ab") as f:
        f.write(b'not json at all\n[1, 2]\n{"type": "mystery", "id": "x"}\n{"type": "report", "id": "zz')  # torn
    c = env.client
    url = f"/api/buildings/{env.a}/reports"
    rows = c.get(url, headers=H(tok), params={"status": "all"})
    assert rows.status_code == 200 and [x["id"] for x in rows.json()] == [rid]

    # a status change after the torn line is written on a fresh line, not glued onto the fragment
    assert c.patch(f"{url}/{rid}", json={"status": "resolved"}, headers=H(tok)).status_code == 200
    new_id = _post(env, env.a, tok, kind="other", note="after the crash").json()["id"]
    rows = c.get(url, headers=H(tok), params={"status": "all"}).json()
    assert {x["id"] for x in rows} == {rid, new_id}
    by_id = {x["id"]: x for x in rows}
    assert by_id[rid]["status"] == "resolved" and by_id[new_id]["status"] == "open"


# ---------- auth off ----------
def test_with_auth_off_reporter_is_null_and_it_still_works(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "AUTH_REQUIRED", False)
    monkeypatch.setattr(config, "BUILDINGS_DIR", tmp_path / "buildings")  # conftest only moves DATA_DIR
    monkeypatch.setattr(config, "REPORTS_PER_HOUR", 1)  # with no identity there is nothing to count against
    bid = store.create_building("Hall C").building_id
    bundle = store.load_map(bid)
    bundle.nodes = [Node(id="n1", label="Lobby")]
    store.save_map(bundle)
    c = TestClient(app)
    for _ in range(3):
        r = c.post(f"/api/buildings/{bid}/reports", json={"kind": "blocked", "node_id": "n1"})
        assert r.status_code == 200, r.text
    rows = c.get(f"/api/buildings/{bid}/reports").json()
    assert len(rows) == 3 and all(x["reporter"] is None and x["node_label"] == "Lobby" for x in rows)
    assert c.patch(f"/api/buildings/{bid}/reports/{rows[0]['id']}", json={"status": "resolved"}).status_code == 200
    assert c.get(f"/api/buildings/{bid}/access").json() == {"role": "owner", "open_reports": 2}
    assert all(r["reporter"] is None for r in _lines(bid) if r["type"] == "report")
