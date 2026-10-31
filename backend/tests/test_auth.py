"""Access control with AUTH_REQUIRED=1: owner key, invites, roles, signed photo URLs, brute-force limit, bootstrap.

Job submission and Gemma are stubbed out, so this runs without ffmpeg or Ollama.
"""
import hashlib
import io
import json
import logging
import re
import stat
import sys
import time
from types import SimpleNamespace

import pytest
from fastapi import Request
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from PIL import Image

from app import auth, cli, config, gemma, jobs, store
from app.main import app
from app.router import _map_out, api
from app.schemas import Edge, FrameAnalysis, Node, SuggestedMerge, VideoInfo

OWNER = "owner-test-key-0123456789abcdefghijklmnopqrstuvwxyz"
FRAME = "v1200001234_0001.jpg"


def H(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _no_gemma(*_a, **_k):
    raise gemma.GemmaUnavailable("disabled in tests")


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 24), (200, 30, 30)).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTH_REQUIRED", True)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "BUILDINGS_DIR", tmp_path / "buildings")
    monkeypatch.setattr(config, "ADMIN_TOKEN", OWNER)
    monkeypatch.setattr(config, "AUTH_MAX_FAILS", 1000)  # the brute-force test lowers this itself
    monkeypatch.setattr(gemma, "resolve_model", _no_gemma)
    monkeypatch.setattr(jobs, "JOBS", {})
    monkeypatch.setattr(auth, "_fails", {})

    submitted = []

    def fake_submit(building_id, upload_path, filename, sensors, model, front=False, start_floor=None):
        submitted.append(filename)
        upload_path.unlink(missing_ok=True)
        return jobs.Job(id="jvid", building_id=building_id, video_id="v1200001234", kind="video")

    monkeypatch.setattr(jobs, "submit_video", fake_submit)
    monkeypatch.setattr(jobs, "submit_consolidate",
                        lambda bid: jobs.Job(id="jcons", building_id=bid, video_id=None, kind="consolidate"))
    monkeypatch.setattr(jobs, "submit_rebuild",
                        lambda bid: jobs.Job(id="jreb", building_id=bid, video_id=None, kind="rebuild"))

    a = store.create_building("Hall A").building_id
    b = store.create_building("Hall B").building_id
    for bid in (a, b):
        bundle = store.load_map(bid)
        bundle.nodes = [Node(id="n1", label="Lobby", frame_id=FRAME), Node(id="n2", label="Office")]
        bundle.videos = [VideoInfo(id="v1200001234", filename="walk.mp4")]
        store.save_map(bundle)
        (store.frames_dir(bid) / FRAME).write_bytes(_jpeg())
    jobs.JOBS["jvid"] = jobs.Job(id="jvid", building_id=a, video_id="v1200001234", kind="video")
    return SimpleNamespace(client=TestClient(app), a=a, b=b, submitted=submitted)


def _invite(env, bid, role="viewer", label="test") -> dict:
    r = env.client.post(f"/api/buildings/{bid}/invites", json={"role": role, "label": label}, headers=H(OWNER))
    assert r.status_code == 200, r.text
    return r.json()


def _token(env, bid, role="viewer") -> str:
    return _invite(env, bid, role)["token"]


def _mutating(env, bid):
    """(method, url, request kwargs) for every route that changes something."""
    return [
        ("PUT", f"/api/buildings/{bid}/map", {"json": {"nodes": [], "edges": []}}),
        ("POST", f"/api/buildings/{bid}/merges/m1", {"json": {"accept": True}}),
        ("POST", f"/api/buildings/{bid}/merge", {"json": {"keep": "n1", "remove": "n2"}}),
        ("POST", f"/api/buildings/{bid}/videos", {"files": {"video": ("walk.mp4", b"x" * 10, "video/mp4")}}),
        ("PATCH", f"/api/buildings/{bid}/videos/v1200001234", {"json": {"start_floor": 1}}),
        ("DELETE", f"/api/buildings/{bid}/videos/v1200001234", {}),
        ("POST", f"/api/buildings/{bid}/consolidate", {}),
        ("POST", f"/api/buildings/{bid}/rebuild", {}),
        ("POST", f"/api/buildings/{bid}/frames/manual", {"files": {"image": ("x.jpg", _jpeg(), "image/jpeg")}}),
        ("POST", "/api/jobs/jvid/pause", {}),
        ("POST", f"/api/buildings/{bid}/invites", {"json": {"role": "viewer"}}),
        ("DELETE", f"/api/buildings/{bid}/invites/abc", {}),
        ("POST", "/api/buildings", {"json": {"name": "Sneaky"}}),
    ]


# ---------- anonymous access ----------
def test_health_stays_open(env):
    assert env.client.get("/api/health").status_code == 200


def test_every_api_route_except_health_needs_a_token(env):
    """Criterion: with AUTH_REQUIRED=1 nothing under /api is reachable anonymously, except health and the
    building list (which answers with an empty list instead of an error)."""
    seen = 0
    for route in api.routes:  # the /api router (app.routes wraps it in one entry)
        if not isinstance(route, APIRoute):
            continue
        path = re.sub(r"\{[^}]+\}", "x", route.path)
        for method in route.methods:
            if path == "/api/health":
                continue
            r = env.client.request(method, path)
            seen += 1
            if path == "/api/buildings" and method == "GET":
                assert r.status_code == 200 and r.json() == [], (method, path)
            else:
                assert r.status_code == 401, (method, path, r.status_code, r.text)
    assert seen > 20


def test_docs_and_schema_are_not_served(env):
    assert env.client.get("/openapi.json").status_code == 404
    assert env.client.get("/docs").status_code == 404


# ---------- roles on mutating and owner-only routes ----------
def test_mutating_routes_refuse_anonymous_with_401(env):
    for method, url, kwargs in _mutating(env, env.a):
        r = env.client.request(method, url, **kwargs)
        assert r.status_code == 401, (method, url, r.status_code, r.text)
    assert env.submitted == []  # the upload was refused before its body was read


def test_viewer_token_gets_403_on_mutating_routes(env):
    tok = _token(env, env.a, "viewer")
    for method, url, kwargs in _mutating(env, env.a):
        r = env.client.request(method, url, headers=H(tok), **kwargs)
        assert r.status_code == 403, (method, url, r.status_code, r.text)
    assert env.submitted == []


def test_editor_token_gets_403_on_owner_only_routes(env):
    tok = _token(env, env.a, "editor")
    checks = [
        ("POST", f"/api/buildings/{env.a}/invites", {"json": {"role": "viewer"}}),
        ("GET", f"/api/buildings/{env.a}/invites", {}),
        ("DELETE", f"/api/buildings/{env.a}/invites/abc", {}),
        ("POST", "/api/buildings", {"json": {"name": "Sneaky"}}),
    ]
    for method, url, kwargs in checks:
        r = env.client.request(method, url, headers=H(tok), **kwargs)
        assert r.status_code == 403, (method, url, r.status_code)


def test_viewer_reads_map_plans_route_and_loads_signed_photo(env):
    tok = _token(env, env.a, "viewer")
    c = env.client
    assert c.get(f"/api/buildings/{env.a}/map", headers=H(tok)).status_code == 200
    assert c.get(f"/api/buildings/{env.a}/access", headers=H(tok)).json() == {"role": "viewer"}

    r = c.post(f"/api/buildings/{env.a}/route", json={"start_node_id": "n1", "target_node_id": "n1"}, headers=H(tok))
    assert r.status_code == 200, r.text
    url = r.json()["steps"][0]["frame_url"]
    assert url.startswith(f"/bfiles/{env.a}/frames/{FRAME}?exp=") and "&sig=" in url
    # <img> tags cannot send headers: the signed URL is the access check, so no token here
    img = c.get(url)
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"


def test_editor_uploads_and_deletes_but_cannot_create_invites(env):
    tok = _token(env, env.a, "editor")
    c = env.client
    r = c.post(f"/api/buildings/{env.a}/videos", headers=H(tok),
               files={"video": ("walk.mp4", b"x" * 10, "video/mp4")})
    assert r.status_code == 200, r.text
    assert env.submitted == ["walk.mp4"]
    assert c.delete(f"/api/buildings/{env.a}/videos/v1200001234", headers=H(tok)).status_code == 200
    assert c.post(f"/api/buildings/{env.a}/invites", json={"role": "viewer"}, headers=H(tok)).status_code == 403


# ---------- invites ----------
def test_invite_response_shape_and_join_path(env):
    inv = _invite(env, env.a, "editor", label="Front desk")
    assert set(inv) == {"id", "role", "label", "created_at", "token", "join_path"}
    assert inv["role"] == "editor" and inv["label"] == "Front desk"
    assert inv["join_path"] == f"#/join?b={env.a}&t={inv['token']}"
    assert re.fullmatch(r"[A-Za-z0-9_-]{43}", inv["token"])  # secrets.token_urlsafe(32)


def test_invite_label_limited_to_80_characters(env):
    r = env.client.post(f"/api/buildings/{env.a}/invites", json={"role": "viewer", "label": "x" * 81}, headers=H(OWNER))
    assert r.status_code == 422
    r = env.client.post(f"/api/buildings/{env.a}/invites", json={"role": "owner"}, headers=H(OWNER))
    assert r.status_code == 422  # only viewer and editor can be invited


def test_list_invites_never_shows_token_or_hash(env):
    inv = _invite(env, env.a)
    r = env.client.get(f"/api/buildings/{env.a}/invites", headers=H(OWNER))
    assert r.status_code == 200
    row = r.json()[0]
    assert set(row) == {"id", "role", "label", "created_at", "revoked"}
    assert row["id"] == inv["id"] and row["revoked"] is False
    assert inv["token"] not in r.text
    assert hashlib.sha256(inv["token"].encode()).hexdigest() not in r.text


def test_revoked_invite_gets_401_immediately(env):
    inv = _invite(env, env.a, "editor")
    tok = inv["token"]
    c = env.client
    assert c.get(f"/api/buildings/{env.a}/map", headers=H(tok)).status_code == 200
    r = c.delete(f"/api/buildings/{env.a}/invites/{inv['id']}", headers=H(OWNER))
    assert r.status_code == 200 and r.json()["revoked"] is True
    assert c.get(f"/api/buildings/{env.a}/map", headers=H(tok)).status_code == 401
    assert c.post(f"/api/buildings/{env.a}/consolidate", headers=H(tok)).status_code == 401
    assert c.get("/api/buildings", headers={"X-Access-Tokens": tok}).json() == []
    assert c.delete(f"/api/buildings/{env.a}/invites/nope", headers=H(OWNER)).status_code == 404


def test_invite_is_scoped_to_its_building(env):
    tok = _token(env, env.a, "editor")
    c = env.client
    assert c.get(f"/api/buildings/{env.b}/map", headers=H(tok)).status_code == 401


def test_unknown_and_forbidden_building_answer_the_same(env):
    tok = _token(env, env.a, "viewer")
    c = env.client
    forbidden = c.get(f"/api/buildings/{env.b}/map", headers=H(tok))
    unknown = c.get("/api/buildings/no-such-hall/map", headers=H(tok))
    assert forbidden.status_code == unknown.status_code == 401
    assert forbidden.json() == unknown.json()
    assert c.get("/api/buildings/no-such-hall/map", headers=H(OWNER)).status_code == 404  # owner key: 404 as before


def test_access_endpoint_reports_role_or_401(env):
    c = env.client
    assert c.get(f"/api/buildings/{env.a}/access").status_code == 401
    assert c.get(f"/api/buildings/{env.a}/access", headers=H(OWNER)).json() == {"role": "owner", "open_reports": 0}
    assert c.get(f"/api/buildings/{env.a}/access", headers=H("made-up")).status_code == 401
    assert c.get("/api/buildings/no-such-hall/access", headers=H(OWNER)).status_code == 404


# ---------- building list ----------
def test_buildings_filtered_by_caller_with_role(env):
    c = env.client
    rows = {b["id"]: b["role"] for b in c.get("/api/buildings", headers=H(OWNER)).json()}
    assert rows == {env.a: "owner", env.b: "owner"}

    tok_a = _token(env, env.a, "viewer")
    tok_b = _token(env, env.b, "editor")
    only_a = c.get("/api/buildings", headers={"X-Access-Tokens": tok_a}).json()
    assert [(b["id"], b["role"]) for b in only_a] == [(env.a, "viewer")]
    assert set(only_a[0]) == {"id", "name", "nodes", "edges", "role"}

    both = c.get("/api/buildings", headers={"X-Access-Tokens": f"{tok_a}, {tok_b}"}).json()
    assert {b["id"]: b["role"] for b in both} == {env.a: "viewer", env.b: "editor"}


def test_buildings_list_empty_without_valid_tokens(env):
    c = env.client
    assert c.get("/api/buildings").json() == []
    assert c.get("/api/buildings", headers={"X-Access-Tokens": "nope,also-nope"}).json() == []
    r = c.get("/api/buildings", headers={"X-Access-Tokens": b"garbage,,\xc3\xa9, ,\t"})
    assert r.status_code == 200 and r.json() == []
    # the owner key sent in the wrong header is not a building token
    assert c.get("/api/buildings", headers={"X-Access-Tokens": OWNER}).json() == []


def test_list_wrong_owner_key_gets_401(env):
    """Authorization on GET /api/buildings is the owner-key check: a wrong key is 401, not an empty 200."""
    c = env.client
    r = c.get("/api/buildings", headers=H("nope"))
    assert r.status_code == 401 and r.headers.get("WWW-Authenticate") == "Bearer"
    assert "Lobby" not in r.text and env.a not in r.text
    assert c.get("/api/buildings", headers=H(OWNER + "x")).status_code == 401  # near miss
    assert c.get("/api/buildings", headers={"Authorization": f"Token {OWNER}"}).status_code == 401  # wrong scheme
    assert c.get("/api/buildings", headers={"Authorization": OWNER}).status_code == 401  # no scheme
    assert c.get("/api/buildings", headers={"Authorization": ""}).status_code == 401  # present but empty
    assert c.get("/api/buildings", headers=H(OWNER)).status_code == 200  # the right key still lists all


def test_list_invite_token_in_authorization_gets_401(env):
    """An invite token is never accepted in Authorization on this route; it goes in X-Access-Tokens."""
    c = env.client
    for role in ("viewer", "editor"):  # invites never carry the owner role
        tok = _token(env, env.a, role)
        assert c.get("/api/buildings", headers=H(tok)).status_code == 401, role
        assert c.get("/api/buildings", headers={**H(tok), "X-Access-Tokens": tok}).status_code == 401, role
        assert c.get("/api/buildings", headers={"X-Access-Tokens": tok}).json()[0]["id"] == env.a, role
    # a valid owner key next to a junk X-Access-Tokens still lists everything
    assert {b["id"] for b in c.get("/api/buildings", headers={**H(OWNER), "X-Access-Tokens": "junk"}).json()} \
        == {env.a, env.b}


def test_list_valid_owner_key_on_zero_buildings_is_empty_200(env, tmp_path, monkeypatch):
    """A brand-new server has no buildings yet; the owner key must still verify (200 with [])."""
    monkeypatch.setattr(config, "BUILDINGS_DIR", tmp_path / "no-buildings-yet")
    c = env.client
    r = c.get("/api/buildings", headers=H(OWNER))
    assert r.status_code == 200 and r.json() == []
    assert c.get("/api/buildings", headers=H("nope")).status_code == 401  # the same server still rejects a wrong key


def test_list_fresh_server_generated_owner_key_verifies_on_empty_listing(tmp_path, monkeypatch):
    """No ADMIN_TOKEN, no buildings: the generated owner key verifies via GET /api/buildings (200 []), a wrong one 401."""
    monkeypatch.setattr(config, "AUTH_REQUIRED", True)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "BUILDINGS_DIR", tmp_path / "buildings")
    monkeypatch.setattr(config, "ADMIN_TOKEN", None)
    monkeypatch.setattr(auth, "_fails", {})
    raw = auth.rotate_owner_key()
    c = TestClient(app)
    r = c.get("/api/buildings", headers=H(raw))
    assert r.status_code == 200 and r.json() == []
    assert c.get("/api/buildings", headers=H("not-the-key")).status_code == 401


def test_list_garbage_x_access_tokens_without_authorization_is_200_empty(env):
    c = env.client
    for junk in ["garbage", ",,,", " , ,", b"\xc3\xa9\xc3\xa8,\xe4\xb8\xad", b"\xff\xfe,x", "a" * 10_000,
                 ",".join(["zz"] * 60)]:
        r = c.get("/api/buildings", headers={"X-Access-Tokens": junk})
        assert r.status_code == 200 and r.json() == [], junk[:20]
    assert c.get("/api/buildings").status_code == 200 and c.get("/api/buildings").json() == []


def test_wrong_owner_key_on_list_counts_toward_the_lockout(env, monkeypatch):
    c = env.client
    monkeypatch.setattr(config, "AUTH_MAX_FAILS", 3)
    clock = {"t": 1000.0}
    monkeypatch.setattr(auth, "_now", lambda: clock["t"])
    statuses = [c.get("/api/buildings", headers=H(f"guess-{i}")).status_code for i in range(3)]
    assert statuses == [401, 401, 401]  # three failures on this route alone
    r = c.get("/api/buildings", headers=H(OWNER))
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1  # the owner key waits too
    assert c.get("/api/buildings", headers={"X-Access-Tokens": "junk"}).status_code == 429
    clock["t"] += 61
    assert c.get("/api/buildings", headers=H(OWNER)).status_code == 200


def test_x_access_tokens_read_at_most_50(env):
    tok = _token(env, env.a, "viewer")
    c = env.client
    fifty_then_valid = ",".join([f"junk{i}" for i in range(50)] + [tok])
    assert c.get("/api/buildings", headers={"X-Access-Tokens": fifty_then_valid}).json() == []
    valid_at_50 = ",".join([f"junk{i}" for i in range(49)] + [tok])
    assert [b["id"] for b in c.get("/api/buildings", headers={"X-Access-Tokens": valid_at_50}).json()] == [env.a]


# ---------- jobs ----------
def test_job_routes_check_the_jobs_building(env):
    editor_a = _token(env, env.a, "editor")
    editor_b = _token(env, env.b, "editor")
    viewer_a = _token(env, env.a, "viewer")
    c = env.client
    assert c.get("/api/jobs/jvid", headers=H(editor_a)).status_code == 200
    assert c.get("/api/jobs/jvid", headers=H(viewer_a)).status_code == 403
    assert c.get("/api/jobs/jvid", headers=H(editor_b)).status_code == 401  # job is in another building
    assert c.get("/api/jobs/jvid", headers=H(OWNER)).status_code == 200
    assert c.get("/api/jobs/missing", headers=H(OWNER)).status_code == 404
    assert c.get("/api/jobs/missing", headers=H(editor_a)).status_code == 401  # same answer as a forbidden job
    assert c.post("/api/jobs/jvid/pause", headers=H(editor_a)).status_code == 200


# ---------- signed photo URLs ----------
def _frame_url(bid, name=FRAME, exp=None, sig=None):
    exp = int(time.time()) + 600 if exp is None else exp
    sig = auth._frame_sig(bid, name, exp) if sig is None else sig
    return f"/bfiles/{bid}/frames/{name}?exp={exp}&sig={sig}"


def test_signed_photo_loads_and_bad_signatures_are_404(env):
    c = env.client
    assert c.get(_frame_url(env.a)).status_code == 200
    past = int(time.time()) - 5
    cases = {
        "expired": _frame_url(env.a, exp=past),
        "forged sig": _frame_url(env.a, sig="0" * 64),
        "missing sig": f"/bfiles/{env.a}/frames/{FRAME}?exp={int(time.time()) + 600}",
        "missing exp": f"/bfiles/{env.a}/frames/{FRAME}?sig={auth._frame_sig(env.a, FRAME, int(time.time()) + 600)}",
        "non-numeric exp": f"/bfiles/{env.a}/frames/{FRAME}?exp=soon&sig=00",
        "no query": f"/bfiles/{env.a}/frames/{FRAME}",
        "sig of another building": f"/bfiles/{env.b}/frames/{FRAME}?exp={int(time.time()) + 600}"
                                   f"&sig={auth._frame_sig(env.a, FRAME, int(time.time()) + 600)}",
        "sig of another photo": _frame_url(env.a, name="v1200001234_0002.jpg", sig=auth._frame_sig(env.a, FRAME, int(time.time()) + 600)),
    }
    for label, url in cases.items():
        assert c.get(url).status_code == 404, label


def test_signature_does_not_make_path_traversal_work(env):
    c = env.client
    exp = int(time.time()) + 600
    for name, real_name in [("%2E%2E", ".."), ("..%2Fmap.json", "../map.json"),
                            ("..%2Faccess.json", "../access.json"), (".hidden", ".hidden")]:
        url = f"/bfiles/{env.a}/frames/{name}?exp={exp}&sig={auth._frame_sig(env.a, real_name, exp)}"
        assert c.get(url).status_code == 404, name  # a valid signature for the bad name still gets 404
    for path in [f"/bfiles/{env.a}/map.json", f"/bfiles/{env.a}/access.json"]:
        assert c.get(path).status_code == 404, path


def test_frame_urls_are_built_at_response_time_not_stored(env):
    tok = _token(env, env.a, "viewer")
    env.client.post(f"/api/buildings/{env.a}/route", json={"start_node_id": "n1", "target_node_id": "n1"},
                    headers=H(tok))
    text = (store.building_dir(env.a) / "map.json").read_text()
    assert "/bfiles" not in text and "sig=" not in text


def _node(body: dict, node_id: str) -> dict:
    return next(n for n in body["nodes"] if n["id"] == node_id)


def _assert_photo_urls_signed(body: dict, bid: str) -> None:
    """Every node with a photo has a freshly signed URL for it; a node without one has null."""
    for n in body["nodes"]:
        if n["frame_id"]:
            assert n["frame_url"].startswith(f"/bfiles/{bid}/frames/{n['frame_id']}?exp="), n
            assert "&sig=" in n["frame_url"], n
        else:
            assert n["frame_url"] is None, n


def _with_frame_url_mutated(url: str) -> str:
    head, _, sig = url.rpartition("&sig=")
    return f"{head}&sig={'1' if sig[0] == '0' else '0'}{sig[1:]}"


def test_map_response_has_signed_photo_url_viewer_can_load(env):
    """GET /map as a viewer: the node's frame_url loads (200); the same URL with a changed sig does not (404)."""
    tok = _token(env, env.a, "viewer")
    c = env.client
    body = c.get(f"/api/buildings/{env.a}/map", headers=H(tok)).json()
    _assert_photo_urls_signed(body, env.a)
    url = _node(body, "n1")["frame_url"]
    assert _node(body, "n2")["frame_url"] is None  # n2 has no photo
    img = c.get(url)
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    assert c.get(_with_frame_url_mutated(url)).status_code == 404


def test_edge_photo_lists_are_signed_in_the_same_order(env):
    bundle = store.load_map(env.a)
    bundle.edges = [Edge(id="e1", from_node="n1", to_node="n2", heading_out=0.0, heading_in=0.0,
                         frames_fwd=[FRAME, "missing.jpg"], frames_rev=None)]
    store.save_map(bundle)
    tok = _token(env, env.a, "viewer")
    e = env.client.get(f"/api/buildings/{env.a}/map", headers=H(tok)).json()["edges"][0]
    assert e["frames_fwd"] == [FRAME, "missing.jpg"]  # the plain ids are unchanged
    assert e["frames_fwd_urls"][0].startswith(f"/bfiles/{env.a}/frames/{FRAME}?exp=") and "&sig=" in e["frames_fwd_urls"][0]
    assert e["frames_fwd_urls"][1].startswith(f"/bfiles/{env.a}/frames/missing.jpg?exp=")
    assert e["frames_rev_urls"] is None


def test_map_responses_sign_urls_and_map_json_never_holds_them(env):
    """Criterion 2: GET, PUT and both merge routes return signed URLs; map.json keeps none of them."""
    bundle = store.load_map(env.a)
    bundle.nodes.append(Node(id="n3", label="Stairs", type="stairs", frame_id=None))
    bundle.edges = [Edge(id="e1", from_node="n1", to_node="n2", heading_out=0.0, heading_in=0.0,
                         frames_fwd=[FRAME], frames_rev=[FRAME])]
    bundle.suggested_merges = [SuggestedMerge(id="m1", keep="n1", remove="n3", confidence=0.9, reason="test")]
    store.save_map(bundle)
    c = env.client
    viewer, editor = _token(env, env.a, "viewer"), _token(env, env.a, "editor")

    got = c.get(f"/api/buildings/{env.a}/map", headers=H(viewer))
    assert got.status_code == 200
    _assert_photo_urls_signed(got.json(), env.a)

    # the editor sends the map back as they received it, frame_url included, and forges one on n2
    body = got.json()
    nodes = body["nodes"]
    _node(body, "n2")["frame_url"] = f"/bfiles/{env.a}/frames/{FRAME}?exp=9999999999&sig=forged"
    put = c.put(f"/api/buildings/{env.a}/map", json={"nodes": nodes, "edges": body["edges"]}, headers=H(editor))
    assert put.status_code == 200, put.text
    assert "forged" not in put.text
    _assert_photo_urls_signed(put.json()["map"], env.a)

    accept = c.post(f"/api/buildings/{env.a}/merges/m1", json={"accept": True}, headers=H(editor))
    assert accept.status_code == 200, accept.text
    _assert_photo_urls_signed(accept.json(), env.a)

    manual = c.post(f"/api/buildings/{env.a}/merge", json={"keep": "n1", "remove": "n2"}, headers=H(editor))
    assert manual.status_code == 200, manual.text
    _assert_photo_urls_signed(manual.json(), env.a)

    text = (store.building_dir(env.a) / "map.json").read_text()
    assert FRAME in text  # the plain frame id is what gets stored
    assert "/bfiles" not in text and "&sig=" not in text and "?exp=" not in text and "forged" not in text
    assert "frame_url" not in text and "frames_fwd_urls" not in text and "frames_rev_urls" not in text
    stored = json.loads(text)
    assert not [k for k in _all_keys(stored) if k in {"frame_url", "frames_fwd_urls", "frames_rev_urls", "exp", "sig"}]


def _all_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _all_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _all_keys(v)


def test_save_map_drops_signed_urls_even_from_a_response_object(env):
    """Belt and braces: store.save_map never writes frame_url / frames_*_urls, even if handed a MapOut."""
    bundle = store.load_map(env.a)
    bundle.edges = [Edge(id="e1", from_node="n1", to_node="n2", heading_out=0.0, heading_in=0.0,
                         frames_fwd=[FRAME], frames_rev=[FRAME])]
    store.save_map(bundle)
    out = _map_out(store.load_map(env.a))
    assert _node(out.model_dump(), "n1")["frame_url"]  # the response object does carry the URL
    store.save_map(out)
    text = (store.building_dir(env.a) / "map.json").read_text()
    assert "frame_url" not in text and "frames_fwd_urls" not in text and "&sig=" not in text and "?exp=" not in text
    reloaded = store.load_map(env.a)
    assert reloaded.nodes[0].frame_id == FRAME and reloaded.edges[0].frames_fwd == [FRAME]


def test_put_map_ignores_frame_url_and_uses_server_urls(env):
    editor = _token(env, env.a, "editor")
    c = env.client
    body = c.get(f"/api/buildings/{env.a}/map", headers=H(editor)).json()
    _node(body, "n2")["frame_url"] = f"/bfiles/{env.a}/frames/{FRAME}?exp=9999999999&sig=forged"
    r = c.put(f"/api/buildings/{env.a}/map", json={"nodes": body["nodes"], "edges": body["edges"]}, headers=H(editor))
    assert r.status_code == 200
    out = r.json()["map"]
    assert _node(out, "n2")["frame_url"] is None  # n2 has no photo: the sent value is dropped, not echoed
    _assert_photo_urls_signed(out, env.a)


def test_manual_photo_node_comes_back_with_signed_url(env, monkeypatch):
    monkeypatch.setattr(gemma, "analyze_frame", lambda *_a, **_k: FrameAnalysis(suggested_label="Sign"))
    editor = _token(env, env.a, "editor")
    c = env.client
    r = c.post(f"/api/buildings/{env.a}/frames/manual", headers=H(editor),
               files={"image": ("sign.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 200, r.text
    node = r.json()["node"]
    assert node["frame_url"].startswith(f"/bfiles/{env.a}/frames/{node['frame_id']}?exp=") and "&sig=" in node["frame_url"]
    assert c.get(node["frame_url"]).status_code == 200
    assert "frame_url" not in (store.building_dir(env.a) / "map.json").read_text()


def test_auth_off_map_urls_are_the_plain_path(env, monkeypatch):
    monkeypatch.setattr(config, "AUTH_REQUIRED", False)
    c = env.client
    body = c.get(f"/api/buildings/{env.a}/map").json()
    plain = f"/bfiles/{env.a}/frames/{FRAME}"
    assert _node(body, "n1")["frame_url"] == plain
    assert _node(body, "n2")["frame_url"] is None
    assert store.frame_url(env.a, FRAME) == plain
    assert c.get(plain).status_code == 200  # no signature needed with access control off


# ---------- secrets never leave the server ----------
def test_no_raw_token_or_hash_in_responses_or_on_disk_map(env):
    c = env.client
    inv = _invite(env, env.a, "editor")
    tok = inv["token"]
    h = hashlib.sha256(tok.encode()).hexdigest()
    bodies = [
        c.get(f"/api/buildings/{env.a}/invites", headers=H(OWNER)).text,
        c.get(f"/api/buildings/{env.a}/map", headers=H(tok)).text,
        c.get(f"/api/buildings/{env.a}/access", headers=H(tok)).text,
        c.get("/api/buildings", headers={"X-Access-Tokens": tok}).text,
        c.get("/api/buildings", headers=H(OWNER)).text,
        c.post(f"/api/buildings/{env.a}/route", json={"start_node_id": "n1", "target_node_id": "n1"},
               headers=H(tok)).text,
        c.delete(f"/api/buildings/{env.a}/invites/{inv['id']}", headers=H(OWNER)).text,
        c.get(f"/api/buildings/{env.a}/access", headers=H(OWNER)).text,
    ]
    on_disk = (store.building_dir(env.a) / "map.json").read_text()
    for text in bodies + [on_disk]:
        assert tok not in text and h not in text and OWNER not in text
    access_on_disk = (store.building_dir(env.a) / "access.json").read_text()
    assert h in access_on_disk and tok not in access_on_disk  # the server keeps the hash only


# ---------- brute force ----------
def test_too_many_failures_get_429_until_the_window_passes(env, monkeypatch):
    c = env.client
    monkeypatch.setattr(config, "AUTH_MAX_FAILS", 3)
    clock = {"t": 1000.0}
    monkeypatch.setattr(auth, "_now", lambda: clock["t"])
    statuses = [c.get(f"/api/buildings/{env.a}/map", headers=H(f"guess-{i}")).status_code for i in range(3)]
    assert statuses == [401, 401, 401]  # three failures so far
    r = c.get(f"/api/buildings/{env.a}/map", headers=H(OWNER))
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1  # even the owner key waits
    clock["t"] += 61
    assert c.get(f"/api/buildings/{env.a}/map", headers=H("guess")).status_code == 401
    assert c.get(f"/api/buildings/{env.a}/map", headers=H(OWNER)).status_code == 200


def test_failure_limit_boundary_is_exact(env, monkeypatch):
    """AUTH_MAX_FAILS=N: attempts 1..N get 401, attempt N+1 gets 429 (README: one more gets HTTP 429)."""
    c = env.client
    monkeypatch.setattr(config, "AUTH_MAX_FAILS", 4)
    clock = {"t": 1000.0}
    monkeypatch.setattr(auth, "_now", lambda: clock["t"])
    url = f"/api/buildings/{env.a}/map"
    statuses = [c.get(url, headers=H(f"guess-{i}")).status_code for i in range(5)]
    assert statuses == [401, 401, 401, 401, 429]
    assert c.get(url, headers=H(OWNER)).status_code == 429  # the refused attempt was not counted; still locked
    clock["t"] += 61
    assert c.get(url, headers=H(OWNER)).status_code == 200


@pytest.mark.parametrize("peer", ["127.0.0.1", "::1"])
def test_behind_the_tunnel_limit_follows_cf_connecting_ip(env, monkeypatch, peer):
    """cloudflared connects from loopback, so CF-Connecting-IP names the real client and is believed."""
    monkeypatch.setattr(config, "AUTH_MAX_FAILS", 2)
    c = TestClient(app, client=(peer, 50000))

    def bad(ip):
        return c.get(f"/api/buildings/{env.a}/map", headers={**H("bad"), "CF-Connecting-IP": ip})

    for _ in range(2):
        bad("203.0.113.7")
    assert bad("203.0.113.7").status_code == 429
    assert bad("203.0.113.8").status_code == 401  # another real client behind the tunnel is not limited


def test_direct_peer_limit_ignores_cf_connecting_ip(env, monkeypatch):
    """From a non-loopback peer, CF-Connecting-IP is anyone's to set: the peer address is the key instead."""
    monkeypatch.setattr(config, "AUTH_MAX_FAILS", 2)
    c = TestClient(app, client=("198.51.100.9", 50000))
    for ip in ["203.0.113.7", "203.0.113.8"]:
        assert c.get(f"/api/buildings/{env.a}/map", headers={**H("bad"), "CF-Connecting-IP": ip}).status_code == 401
    # a fresh forged header does not start a fresh bucket for the same peer (third attempt is refused)
    assert c.get(f"/api/buildings/{env.a}/map", headers={**H("bad"), "CF-Connecting-IP": "203.0.113.10"}).status_code == 429
    other_peer = TestClient(app, client=("198.51.100.10", 50000))
    assert other_peer.get(f"/api/buildings/{env.a}/map", headers={**H("bad"), "CF-Connecting-IP": "203.0.113.7"}).status_code == 401


@pytest.mark.parametrize("peer, header, expected", [
    (("127.0.0.1", 5000), "203.0.113.7", "203.0.113.7"),
    (("::1", 5000), "203.0.113.7", "203.0.113.7"),
    (("198.51.100.9", 5000), "203.0.113.7", "198.51.100.9"),
    (("10.0.0.4", 5000), "203.0.113.7", "10.0.0.4"),
    (("127.0.0.1", 5000), "   ", "127.0.0.1"),  # a blank header is no header
    (("127.0.0.1", 5000), None, "127.0.0.1"),
    (None, None, "unknown"),  # no socket peer at all
])
def test_client_ip_believes_cf_header_only_from_loopback(peer, header, expected):
    headers = [(b"cf-connecting-ip", header.encode())] if header is not None else []
    request = Request({"type": "http", "headers": headers, "client": peer})
    assert auth.client_ip(request) == expected


# ---------- owner key bootstrap ----------
def test_owner_key_bootstrap_writes_0600_file_without_raw_key(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(config, "AUTH_REQUIRED", True)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "ADMIN_TOKEN", "")
    caplog.set_level(logging.WARNING, logger="app.auth")

    auth.startup()
    keys = [r.args[0] for r in caplog.records if r.getMessage().startswith("OWNER KEY")]
    assert len(keys) == 1, "the raw key is logged exactly once"
    raw = keys[0]

    path = tmp_path / "auth.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    text = path.read_text()
    assert raw not in text
    data = json.loads(text)
    assert data["owner_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert data["signing_secret"] and len(data["signing_secret"]) == 64

    caplog.clear()
    auth.startup()  # second start: no new key
    assert not [r for r in caplog.records if r.getMessage().startswith("OWNER KEY")]

    monkeypatch.setattr(config, "BUILDINGS_DIR", tmp_path / "buildings")
    bid = store.create_building("Hall").building_id
    assert TestClient(app).get(f"/api/buildings/{bid}/map", headers=H(raw)).status_code == 200


def test_env_owner_key_is_not_written_to_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTH_REQUIRED", True)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "ADMIN_TOKEN", OWNER)
    auth.startup()
    text = (tmp_path / "auth.json").read_text()
    assert OWNER not in text and "owner_sha256" not in text and "signing_secret" in text


SHORT_ADMIN_TOKEN = "short-admin-secret-123"  # 22 characters


def test_short_admin_token_refuses_to_start(tmp_path, monkeypatch):
    assert len(SHORT_ADMIN_TOKEN) < config.ADMIN_TOKEN_MIN_CHARS == 24
    monkeypatch.setattr(config, "AUTH_REQUIRED", True)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "ADMIN_TOKEN", SHORT_ADMIN_TOKEN)
    with pytest.raises(RuntimeError, match="at least 24") as exc:
        auth.startup()
    assert SHORT_ADMIN_TOKEN not in str(exc.value)  # the message never repeats the secret
    assert not (tmp_path / "auth.json").exists()  # refused before anything is written

    # the same app refuses to start through its own startup hook (the server does not come up)
    monkeypatch.setattr(config, "BUILDINGS_DIR", tmp_path / "buildings")
    with pytest.raises(RuntimeError, match="ADMIN_TOKEN"), TestClient(app):
        pass


def test_admin_token_length_limit_edges(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTH_REQUIRED", True)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "ADMIN_TOKEN", "x" * 24)
    auth.startup()  # exactly 24 characters is enough
    monkeypatch.setattr(config, "ADMIN_TOKEN", "x" * 23)
    with pytest.raises(RuntimeError):
        auth.startup()
    monkeypatch.setattr(config, "AUTH_REQUIRED", False)
    auth.startup()  # with access control off the length is not checked (the startup warning still shows)
    monkeypatch.setattr(config, "AUTH_REQUIRED", True)
    monkeypatch.setattr(config, "ADMIN_TOKEN", "")
    auth.startup()  # no ADMIN_TOKEN at all: a generated owner key is used instead


# ---------- auth off (local development) ----------
def test_auth_off_lets_anonymous_requests_through(env, monkeypatch):
    monkeypatch.setattr(config, "AUTH_REQUIRED", False)
    assert env.client.get(f"/api/buildings/{env.a}/map").status_code == 200
    assert env.client.get(f"/api/buildings/{env.a}/map").status_code == 200
    assert len(env.client.get("/api/buildings").json()) == 2


# ---------- command line ----------
def test_cli_invite_list_revoke(env, capsys, monkeypatch):
    c = env.client
    monkeypatch.setattr(sys, "argv", ["app.cli", "invite", env.a, "--role", "editor", "--label", "Ann"])
    cli.main()
    out = capsys.readouterr().out
    join = next(line for line in out.splitlines() if line.startswith("join path: "))
    token = join.split("&t=", 1)[1]
    invite_id = out.split()[1]
    assert join == f"join path: #/join?b={env.a}&t={token}"
    assert c.get(f"/api/buildings/{env.a}/queue", headers=H(token)).status_code == 200

    monkeypatch.setattr(sys, "argv", ["app.cli", "invites", env.a])
    cli.main()
    listing = capsys.readouterr().out
    assert invite_id in listing and "Ann" in listing and token not in listing

    monkeypatch.setattr(sys, "argv", ["app.cli", "revoke", env.a, invite_id])
    cli.main()
    assert "Revoked" in capsys.readouterr().out
    assert c.get(f"/api/buildings/{env.a}/queue", headers=H(token)).status_code == 401


def test_cli_owner_key_rotation(tmp_path, monkeypatch, capsys, caplog):
    monkeypatch.setattr(config, "AUTH_REQUIRED", True)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "BUILDINGS_DIR", tmp_path / "buildings")
    monkeypatch.setattr(config, "ADMIN_TOKEN", "")
    monkeypatch.setattr(auth, "_fails", {})
    caplog.set_level(logging.WARNING, logger="app.auth")
    auth.startup()
    old = next(r.args[0] for r in caplog.records if r.getMessage().startswith("OWNER KEY"))
    bid = store.create_building("Hall").building_id
    c = TestClient(app)
    assert c.get(f"/api/buildings/{bid}/map", headers=H(old)).status_code == 200

    monkeypatch.setattr(sys, "argv", ["app.cli", "owner-key"])
    cli.main()
    assert "Owner key: set" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["app.cli", "owner-key", "--rotate"])
    cli.main()
    new = capsys.readouterr().out.strip().splitlines()[-1]
    assert new and new != old
    assert c.get(f"/api/buildings/{bid}/map", headers=H(old)).status_code == 401
    assert c.get(f"/api/buildings/{bid}/map", headers=H(new)).status_code == 200


def test_cli_rotation_refused_when_env_key_is_set(env, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["app.cli", "owner-key", "--rotate"])
    with pytest.raises(SystemExit):
        cli.main()


def test_unknown_role_in_dependency_is_a_programming_error():
    with pytest.raises(ValueError):
        auth.require("admin")
    with pytest.raises(ValueError):
        auth.require_job("admin")
