"""Permission matrix: every route that depends on who is calling, for four callers.

  public       no credentials
  contributor  a signed-in account that is not an owner and not blocked
  blocked      a signed-in account an owner has blocked
  owner        a signed-in owner account (conftest's owner list)

Expected answers: 401 for public where a sign-in is needed, 403 for contributor and blocked where the owner is needed
(blocked differs from contributor only on upload), and the normal answer for owner. Every cell runs against a fresh
building, so one check cannot change what the next one sees. No job runs: the queue is faked.
"""
import io
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import config, gemma, jobs, store
from app.main import app
from app.schemas import Edge, Node, SuggestedMerge, VideoInfo
from tests.auth_helpers import owner_headers, signed_in_headers

client = TestClient(app)
PUBLIC, CONTRIBUTOR, BLOCKED, OWNER = "public", "contributor", "blocked", "owner"
ROLES = (PUBLIC, CONTRIBUTOR, BLOCKED, OWNER)


def _headers(role: str) -> dict:
    if role == PUBLIC:
        return {}
    if role == OWNER:
        return owner_headers()
    if role == BLOCKED:
        return signed_in_headers("blocked@example.com")  # blocked in world() before any request is made
    return signed_in_headers("contributor@example.com")


@pytest.fixture(autouse=True)
def _no_pipeline(monkeypatch):
    monkeypatch.setattr(jobs, "JOBS", {})
    monkeypatch.setattr(jobs, "_ensure_worker", lambda: None)

    def fake_submit(building_id, upload_path, filename, sensors, model, front=False, start_floor=None,
                    contributor_id=None, uploader_email="unknown"):
        return jobs.Job(id="jup", building_id=building_id, video_id="v1200001234", kind="video")

    monkeypatch.setattr(jobs, "submit_video", fake_submit)

    def no_gemma(*_a, **_k):
        raise gemma.GemmaUnavailable("disabled in tests")

    monkeypatch.setattr(gemma, "resolve_model", no_gemma)
    monkeypatch.setattr(gemma, "analyze_frame", no_gemma)


def _world(tag: str) -> SimpleNamespace:
    """A building with two places, a walk on the map, a merge suggestion, a queued job and an open report."""
    bid = store.create_building(f"Matrix {tag}").building_id
    b = store.load_map(bid)
    b.nodes = [Node(id="n1", label="Lobby", video_id="vw1"), Node(id="n2", label="Room 2", video_id="vw1")]
    b.edges = [Edge(id="e1", from_node="n1", to_node="n2", heading_out=0, heading_in=0, video_id="vw1")]
    b.videos = [VideoInfo(id="vw1", filename="walk.mp4", status="done", moderation_status="approved")]
    b.suggested_merges = [SuggestedMerge(id="m1", keep="n1", remove="n2", confidence=0.5, reason="same")]
    store.save_map(b)
    store.create_account_if_missing("blocked@example.com")
    store.set_blocked("blocked@example.com", True)
    job = jobs.Job(id=f"jm{tag}", building_id=bid, video_id=None, kind="rebuild", status="queued")
    jobs.JOBS[job.id] = job
    rid = client.post(f"/api/buildings/{bid}/reports", json={"kind": "other", "note": "a note"}).json()["id"]
    return SimpleNamespace(bid=bid, job=job.id, rid=rid, tag=tag)


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 24), (90, 90, 90)).save(buf, "JPEG")
    return buf.getvalue()


def _map_body(w):
    m = store.load_map(w.bid)
    return {"nodes": [n.model_dump() for n in m.nodes], "edges": [e.model_dump() for e in m.edges]}


# name, method, path(world), request kwargs(world), answers for (public, contributor, blocked, owner)
ROWS = [
    ("list buildings", "GET", lambda w: "/api/buildings", lambda w: {}, (200, 200, 200, 200)),
    ("create building", "POST", lambda w: "/api/buildings", lambda w: {"json": {"name": f"New {w.tag}"}},
     (401, 403, 403, 200)),
    ("building map", "GET", lambda w: f"/api/buildings/{w.bid}/map", lambda w: {}, (200, 200, 200, 200)),
    ("floor map", "GET", lambda w: f"/api/buildings/{w.bid}/floors/1/map", lambda w: {}, (200, 200, 200, 200)),
    ("save map", "PUT", lambda w: f"/api/buildings/{w.bid}/map", lambda w: {"json": _map_body(w)},
     (401, 403, 403, 200)),
    ("resolve merge", "POST", lambda w: f"/api/buildings/{w.bid}/merges/m1", lambda w: {"json": {"accept": False}},
     (401, 403, 403, 200)),
    ("manual merge", "POST", lambda w: f"/api/buildings/{w.bid}/merge",
     lambda w: {"json": {"keep": "n1", "remove": "n2"}}, (401, 403, 403, 200)),
    ("upload walk", "POST", lambda w: f"/api/buildings/{w.bid}/videos",
     lambda w: {"files": {"video": ("walk.mp4", b"x" * 10, "video/mp4")}}, (401, 200, 403, 200)),
    ("edit walk floor", "PATCH", lambda w: f"/api/buildings/{w.bid}/videos/vw1", lambda w: {"json": {"start_floor": 2}},
     (401, 403, 403, 200)),
    ("delete walk", "DELETE", lambda w: f"/api/buildings/{w.bid}/videos/vw1", lambda w: {}, (401, 403, 403, 200)),
    ("link walks", "POST", lambda w: f"/api/buildings/{w.bid}/consolidate", lambda w: {}, (401, 403, 403, 200)),
    ("rebuild", "POST", lambda w: f"/api/buildings/{w.bid}/rebuild", lambda w: {}, (401, 403, 403, 200)),
    ("job pause", "POST", lambda w: f"/api/jobs/{w.job}/pause", lambda w: {}, (401, 403, 403, 200)),
    ("add photo", "POST", lambda w: f"/api/buildings/{w.bid}/frames/manual",
     lambda w: {"files": {"image": ("sign.jpg", _jpeg(), "image/jpeg")}, "data": {"label": "Sign"}},
     (401, 403, 403, 200)),
    ("delete building", "DELETE", lambda w: f"/api/buildings/{w.bid}", lambda w: {}, (401, 403, 403, 204)),
    ("file report", "POST", lambda w: f"/api/buildings/{w.bid}/reports", lambda w: {"json": {"kind": "other"}},
     (200, 200, 200, 200)),
    ("list reports", "GET", lambda w: f"/api/buildings/{w.bid}/reports", lambda w: {}, (200, 200, 200, 200)),
    ("resolve report", "PATCH", lambda w: f"/api/buildings/{w.bid}/reports/{w.rid}",
     lambda w: {"json": {"status": "resolved"}}, (401, 403, 403, 200)),
    ("who am I", "GET", lambda w: "/api/auth/me", lambda w: {}, (401, 200, 200, 200)),
    ("my uploads", "GET", lambda w: "/api/me/uploads", lambda w: {}, (401, 200, 200, 200)),
    ("moderation queue", "GET", lambda w: "/api/owner/moderation", lambda w: {}, (401, 403, 403, 200)),
    ("moderate a walk", "POST", lambda w: f"/api/owner/videos/{w.bid}/vw1/moderation",
     lambda w: {"json": {"action": "approve"}}, (401, 403, 403, 200)),
    ("user list", "GET", lambda w: "/api/owner/users", lambda w: {}, (401, 403, 403, 200)),
    ("block a user", "POST", lambda w: "/api/owner/users/someone@example.com/block",
     lambda w: {"json": {"blocked": True}}, (401, 403, 403, 200)),
    ("frame link without signature", "GET", lambda w: f"/api/owner/videos/{w.bid}/vw1/frames/vw1_0000.jpg",
     lambda w: {}, (403, 403, 403, 403)),
]


@pytest.mark.parametrize("role_index", range(len(ROLES)), ids=ROLES)
@pytest.mark.parametrize("name,method,path,kwargs,answers", ROWS, ids=[r[0] for r in ROWS])
def test_permission_matrix(name, method, path, kwargs, answers, role_index):
    role = ROLES[role_index]
    w = _world(f"{role} {name}")
    kw = dict(kwargs(w))
    headers = {**_headers(role), **kw.pop("headers", {})}
    r = client.request(method, path(w), headers=headers, **kw)
    assert r.status_code == answers[role_index], f"{role} {method} {name}: {r.status_code} {r.text[:200]}"
    if answers[role_index] >= 400:
        assert r.status_code in (401, 403), f"{role} {method} {name} should be refused by the role check"


def test_matrix_covers_every_caller_role_for_every_row():
    assert all(len(row[4]) == len(ROLES) for row in ROWS)


def test_legacy_token_counts_as_owner_on_owner_rows(monkeypatch):
    monkeypatch.setattr(config, "OWNER_TOKEN", "s3cret")
    w = _world("token legacy")
    token = {"X-Owner-Token": "s3cret"}
    assert client.get("/api/owner/users", headers=token).status_code == 200
    assert client.post(f"/api/owner/videos/{w.bid}/vw1/moderation", json={"action": "hide"},
                       headers=token).status_code == 200
    assert client.delete(f"/api/buildings/{w.bid}", headers=token).status_code == 204


def test_token_does_not_count_as_contributor_for_upload(monkeypatch):
    """Upload needs a signed-in contributor: the legacy token alone gets the route's own 401."""
    monkeypatch.setattr(config, "OWNER_TOKEN", "s3cret")
    w = _world("token upload")
    r = client.post(f"/api/buildings/{w.bid}/videos", files={"video": ("w.mp4", b"x", "video/mp4")},
                    headers={"X-Owner-Token": "s3cret"})
    assert r.status_code == 401 and r.json() == {"detail": "Sign-in required"}


def test_blocked_role_is_refused_only_on_upload_among_contributor_rows():
    """Blocked and contributor get the same answer on every row except upload (the reason for blocking)."""
    diffs = [r[0] for r in ROWS if r[4][1] != r[4][2]]
    assert diffs == ["upload walk"]


def test_public_map_never_lists_an_uploader_email(monkeypatch):
    w = _world("privacy map")
    store.create_account_if_missing("hidden-uploader@example.com")
    b = store.load_map(w.bid)
    b.videos[0].uploader_email = "hidden-uploader@example.com"
    store.save_map(b)
    assert "hidden-uploader@example.com" not in client.get(f"/api/buildings/{w.bid}/map").text
    assert "hidden-uploader@example.com" in client.get(
        "/api/owner/moderation?status=all", headers=owner_headers()).text
