"""Sentinel adversarial gaps for the owner/moderation work: bad sessions on every write route, owner email spelling,
signed-link abuse, public privacy, .env edge values."""
import json

import pytest
from fastapi.testclient import TestClient

from app import auth, config, envfile, jobs, store
from app.main import app
from tests.auth_helpers import owner_headers, signed_in_headers
from tests.test_owner_moderation import _building, _queue, _signed_url, _walk, queue  # noqa: F401

client = TestClient(app)

WRITE_ROUTES = [
    ("post", "/api/buildings", {"json": {"name": "Evil Hall"}}),
    ("delete", "/api/buildings/{b}", {}),
    ("post", "/api/buildings/{b}/consolidate", {}),
    ("post", "/api/buildings/{b}/rebuild", {}),
    ("get", "/api/owner/users", {}),
    ("get", "/api/owner/moderation", {}),
    ("post", "/api/owner/users/a@b.co/block", {"json": {"blocked": True}}),
    ("post", "/api/owner/videos/{b}/v1/moderation", {"json": {"action": "approve"}}),
]


def _expired_token():
    acct = store.create_account_if_missing("walker@example.com")
    tok = auth.create_session(acct.id)
    return tok


@pytest.mark.parametrize("method,path,kw", WRITE_ROUTES)
@pytest.mark.parametrize("hdr", [
    {"Authorization": "Bearer garbage"}, {"Authorization": "Bearer "}, {"Authorization": "Basic abc"},
    {"Authorization": "Bearer é中"[:0] + "Bearer x" * 50}, {"Authorization": "bearer"},
])
def test_garbage_credentials_never_pass_or_500(method, path, kw, hdr, queue):
    b = _building("Garbage Hall")
    r = getattr(client, method)(path.format(b=b), headers=hdr, **kw)
    assert r.status_code in (401, 403), (path, r.status_code)
    assert store.building_dir(b).exists()


@pytest.mark.parametrize("method,path,kw", WRITE_ROUTES)
def test_expired_session_is_refused(method, path, kw, monkeypatch, queue):
    b = _building("Expired Hall")
    h = owner_headers()
    real = auth._now()
    monkeypatch.setattr(auth, "_now", lambda: real + config.AUTH_SESSION_TTL_SEC + 10)
    r = getattr(client, method)(path.format(b=b), headers=h, **kw)
    assert r.status_code in (401, 403), (path, r.status_code)
    assert store.building_dir(b).exists()


def test_owner_email_spelling_and_whitespace(monkeypatch):
    monkeypatch.setattr(config, "OWNER_EMAILS", frozenset({"owner@example.com", "spaced@example.com"}))
    for spelling in ("OWNER@Example.COM", "  owner@example.com ", "Spaced@EXAMPLE.com"):
        h = signed_in_headers(spelling)
        assert client.get("/api/owner/users", headers=h).status_code == 200, spelling
        assert client.get("/api/auth/me", headers=h).json()["is_owner"] is True
    assert client.get("/api/owner/users", headers=signed_in_headers("owner@example.com.evil.io")).status_code == 403
    assert client.get("/api/owner/users", headers=signed_in_headers("xowner@example.com")).status_code == 403


def test_owner_cannot_be_blocked_even_with_odd_spelling():
    r = client.post("/api/owner/users/%20OWNER@Example.com/block", json={"blocked": True}, headers=owner_headers())
    assert r.status_code in (409, 422), r.status_code
    assert client.get("/api/owner/users", headers=owner_headers()).status_code == 200


def test_blocked_user_still_cannot_upload_after_case_change():
    store.create_account_if_missing("bad@example.com")
    assert client.post("/api/owner/users/BAD@example.com/block", json={"blocked": True},
                       headers=owner_headers()).status_code == 200
    h = signed_in_headers("Bad@Example.com")
    r = client.post("/api/buildings/x/videos", headers=h, files={"file": ("a.mp4", b"x", "video/mp4")})
    assert r.status_code in (403, 404)  # 404 only if the route checks the building first; never 200
    assert client.get("/api/auth/me", headers=h).json()["blocked"] is True


def test_frame_link_non_ascii_cross_walk_and_traversal_paths_never_500(queue):
    bid = _building("Link Hall")
    _walk(bid, "vaa00001", status="queued", moderation="pending", frames=1, processed=True)
    _walk(bid, "vbb00001", status="queued", moderation="pending", frames=1, processed=True)
    url = _signed_url(_queue(bid).json(), "vaa00001")
    qs = url.split("?", 1)[1]
    base = f"/api/owner/videos/{bid}"
    for p in (f"/vaa00001/frames/%E4%B8%AD%E6%96%87.jpg", f"/vaa00001/frames/..%2F..%2Fsource.mp4",
              f"/vaa00001/frames/vbb00001_0000.jpg", f"/vbb00001/frames/vaa00001_0000.jpg",
              f"/..%2F{bid}/vaa00001/frames/vaa00001_0000.jpg", "/vaa00001/frames/%00.jpg",
              "/vaa00001/frames/..%5C..%5Cx.jpg", "/vaa00001/frames/%2e%2e/%2e%2e/map.json"):
        r = client.get(f"{base}{p}?{qs}")
        assert r.status_code in (403, 404), (p, r.status_code)
    for q in ("exp=abc&sig=zz", "exp=-1&sig=", "exp=99999999999999999999999&sig=1", "exp=%E4%B8%AD&sig=%E4%B8%AD", "sig=1"):
        r = client.get(f"{base}/vaa00001/frames/vaa00001_0000.jpg?{q}")
        assert r.status_code == 403, (q, r.status_code)


def test_frame_link_signed_for_one_building_fails_in_another(queue):
    a, b = _building("Hall A"), _building("Hall B")
    _walk(a, "vcc00001", status="queued", moderation="pending", frames=1, processed=True)
    _walk(b, "vcc00001", status="queued", moderation="pending", frames=1, processed=True)
    url = _signed_url(_queue(a).json(), "vcc00001")
    assert client.get(url).status_code == 200
    assert client.get(url.replace(f"/{a}/", f"/{b}/")).status_code == 403


def _all_text(*responses):
    return " ".join(r.text for r in responses)


def test_public_endpoints_never_leak_owner_only_fields(queue):
    bid = _building("Privacy Hall")
    _walk(bid, "vpp00001", status="done", moderation="approved", email="secretwalker@example.com")
    _walk(bid, "vpp00002", status="queued", moderation="pending", email="secretwalker@example.com", processed=True)
    store.create_account_if_missing("blockedperson@example.com")
    store.set_blocked("blockedperson@example.com", True)
    owner_headers()
    client.post(f"/api/owner/videos/{bid}/vpp00001/moderation", json={"action": "hide"}, headers=owner_headers())
    rs = [client.get("/api/buildings"), client.get(f"/api/buildings/{bid}/map"),
          client.get(f"/api/buildings/{bid}/reports"), client.get("/api/health"),
          client.get(f"/api/buildings/{bid}/jobs")]
    text = _all_text(*rs)
    for needle in ("secretwalker", "uploader_email", "moderated_by", "owner@example.com", "blockedperson",
                   "\"blocked\"", "OWNER_TOKEN"):
        assert needle not in text, needle
    for p in ("/api/owner/users", "/api/owner/moderation"):
        assert client.get(p).status_code == 401


def test_me_uploads_only_lists_own_walks(queue):
    bid = _building("Own Hall")
    _walk(bid, "vmm00001", status="done", moderation="approved", email="mine@example.com")
    _walk(bid, "vmm00002", status="done", moderation="approved", email="theirs@example.com")
    mine = store.create_account_if_missing("mine@example.com")
    m = store.load_map(bid)
    for v in m.videos:
        if v.id == "vmm00001":
            v.contributor_id = mine.id
    store.save_map(m)
    r = client.get("/api/me/uploads", headers=signed_in_headers("mine@example.com"))
    assert [u["video_id"] for u in r.json()] == ["vmm00001"]
    assert "theirs" not in r.text and "uploader_email" not in r.text


def test_pending_walk_frames_are_not_served_by_name(queue):
    """A walk that is not approved keeps its frames out of the public frames folder, so /bfiles cannot serve them by name."""
    bid = _building("Leak Hall")
    names = _walk(bid, "vll00001", status="queued", moderation="pending", frames=1, processed=True)
    r = client.get(f"/bfiles/{bid}/frames/{names[0]}")
    assert r.status_code == 404


def test_delete_building_leaves_no_job_or_grant_orphans(monkeypatch):
    monkeypatch.setattr(jobs, "JOBS", {})
    monkeypatch.setattr(jobs, "_ensure_worker", lambda: None)
    bid = _building("Orphan Hall")
    acct = store.create_account_if_missing("mod@example.com")
    store.add_moderator(acct.id, bid)
    j = jobs.Job(id="jorph1", building_id=bid, video_id="v1", kind="video", status="queued")
    jobs.JOBS[j.id] = j
    assert client.delete(f"/api/buildings/{bid}", headers=owner_headers()).status_code == 204
    assert store.get_account("mod@example.com").moderator_of == []
    assert j.status == "cancelled"
    assert client.delete(f"/api/buildings/{bid}", headers=owner_headers()).status_code == 404  # twice


def test_env_values_with_equals_and_quotes():
    got = envfile.parse('A=b=c=d\nB="x=1 # not comment"\nC=\'it"s\'\nD="unterminated\nE=\nF==lead\nexport  G = "sp aced"\n=nokey\n9X=1\n')
    assert got["A"] == "b=c=d" and got["B"] == "x=1 # not comment" and got["C"] == 'it"s'
    assert "D" not in got and "E" not in got and got["F"] == "=lead" and got["G"] == "sp aced"
    assert "9X" not in got


def test_env_garbage_file_does_not_crash(tmp_path):
    p = tmp_path / "bad.env"
    p.write_bytes(b"\xff\xfe\x00garbage\nOK=1\n")
    try:
        envfile.load(p)
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"non-UTF8 .env crashes import of config: {type(e).__name__}")


def test_env_directory_or_unreadable_path_is_ignored(tmp_path):
    assert envfile.load(tmp_path) == []
