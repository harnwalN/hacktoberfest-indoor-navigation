"""Contributors and blocking (owner.py, owner_router.py): the owner's user list, blocking (403 on upload), unblocking,
the contributor's own uploads view (GET /api/me/uploads), and what GET /api/auth/me says about the caller."""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import config, jobs, store
from app.main import app
from app.schemas import Node, VideoInfo
from tests.auth_helpers import owner_headers, signed_in_headers

client = TestClient(app)


@pytest.fixture
def uploads(monkeypatch):
    """Uploads go through the real route; the job it would start is recorded, not run."""
    monkeypatch.setattr(jobs, "JOBS", {})
    monkeypatch.setattr(jobs, "_ensure_worker", lambda: None)
    seen = []

    def fake_submit(building_id, upload_path, filename, sensors, model, front=False, start_floor=None,
                    contributor_id=None, uploader_email="unknown"):
        seen.append({"building_id": building_id, "contributor_id": contributor_id, "uploader_email": uploader_email})
        return jobs.Job(id="j1", building_id=building_id, video_id="v1200001234", kind="video")

    monkeypatch.setattr(jobs, "submit_video", fake_submit)
    return SimpleNamespace(seen=seen)


def _upload(bid, headers):
    return client.post(f"/api/buildings/{bid}/videos", files={"video": ("walk.mp4", b"x" * 10, "video/mp4")},
                       headers=headers)


def _building(name="Users Hall"):
    return store.create_building(name).building_id


def _add_walk(bid, vid, email, status="done", moderation="approved", created="2026-11-01T10:00:00+00:00",
              note=None):
    acct = store.create_account_if_missing(email)
    with store.lock_for(bid):
        b = store.load_map(bid)
        b.videos.append(VideoInfo(id=vid, filename=f"{vid}.mp4", status=status, contributor_id=acct.id,
                                  uploader_email=email, created_at=created, moderation_status=moderation,
                                  moderation_notes=note))
        if status == "done" and moderation == "approved":
            b.nodes.append(Node(id=f"{vid}_n", label=vid, video_id=vid))
        store.save_map(b)
    return acct


# ---------- the owner's user list ----------
def test_user_list_is_owner_only():
    assert client.get("/api/owner/users").status_code == 401
    assert client.get("/api/owner/users", headers=signed_in_headers()).status_code == 403


def test_user_list_counts_each_contributors_walks():
    bid = _building()
    _add_walk(bid, "vu000001", "a@example.com", status="done", moderation="approved")
    _add_walk(bid, "vu000002", "a@example.com", status="rejected", moderation="rejected")
    _add_walk(bid, "vu000003", "a@example.com", status="done", moderation="hidden")
    _add_walk(bid, "vu000004", "b@example.com", status="queued", moderation="pending")
    store.create_account_if_missing("idle@example.com")
    users = {u["email"]: u for u in client.get("/api/owner/users", headers=owner_headers()).json()}
    assert set(users["a@example.com"]) == {"email", "uploads", "live", "rejected", "blocked", "last_seen", "is_owner"}
    assert (users["a@example.com"]["uploads"], users["a@example.com"]["live"], users["a@example.com"]["rejected"]) == (3, 1, 2)
    assert (users["b@example.com"]["uploads"], users["b@example.com"]["live"]) == (1, 0)
    assert (users["idle@example.com"]["uploads"], users["idle@example.com"]["blocked"]) == (0, False)
    assert users["idle@example.com"]["last_seen"] is None  # never signed in to a session


def test_user_list_counts_walks_across_buildings():
    b1, b2 = _building("First Hall"), _building("Second Hall")
    _add_walk(b1, "vub00001", "multi@example.com")
    _add_walk(b2, "vub00002", "multi@example.com")
    users = {u["email"]: u for u in client.get("/api/owner/users", headers=owner_headers()).json()}
    assert users["multi@example.com"]["uploads"] == 2


def test_user_list_flags_owner_accounts_with_the_same_normalisation_as_sign_in():
    store.create_account_if_missing("plain@example.com")
    store.create_account_if_missing("OWNER@Example.com")  # same account as owner@example.com, any letter case
    users = {u["email"]: u for u in client.get("/api/owner/users", headers=owner_headers()).json()}
    assert users["owner@example.com"]["is_owner"] is True
    assert users["plain@example.com"]["is_owner"] is False
    assert all(isinstance(u["is_owner"], bool) for u in users.values())


def test_user_list_is_owner_flag_follows_owner_emails_config(monkeypatch):
    monkeypatch.setattr(config, "OWNER_EMAILS", frozenset({"plain@example.com"}))
    monkeypatch.setattr(config, "OWNER_TOKEN", "s3cret")  # owner session would now be a non-owner; use the token
    store.create_account_if_missing("plain@example.com")
    store.create_account_if_missing("owner@example.com")
    users = {u["email"]: u for u in client.get("/api/owner/users", headers={"X-Owner-Token": "s3cret"}).json()}
    assert users["plain@example.com"]["is_owner"] is True
    assert users["owner@example.com"]["is_owner"] is False


def test_last_seen_is_recorded_on_session_use():
    store.create_account_if_missing("seen@example.com")
    client.get("/api/auth/me", headers=signed_in_headers("seen@example.com"))
    users = {u["email"]: u for u in client.get("/api/owner/users", headers=owner_headers()).json()}
    assert users["seen@example.com"]["last_seen"] is not None


# ---------- blocking ----------
def test_block_is_owner_only_and_stops_uploads(uploads):
    bid = _building()
    assert client.post("/api/owner/users/x@example.com/block", json={"blocked": True}).status_code == 401
    assert client.post("/api/owner/users/x@example.com/block", json={"blocked": True},
                       headers=signed_in_headers()).status_code == 403
    r = client.post("/api/owner/users/Spammer@Example.com/block", json={"blocked": True}, headers=owner_headers())
    assert r.status_code == 200, r.text
    assert r.json()["email"] == "spammer@example.com" and r.json()["blocked"] is True
    r = _upload(bid, signed_in_headers("spammer@example.com"))
    assert r.status_code == 403 and r.json()["detail"] == "This account is blocked from uploading walks."
    assert uploads.seen == []


def test_blocked_account_can_still_sign_in_read_its_uploads_and_see_its_status(uploads):
    bid = _building()
    client.post("/api/owner/users/blocked@example.com/block", json={"blocked": True}, headers=owner_headers())
    headers = signed_in_headers("blocked@example.com")
    assert client.get("/api/me/uploads", headers=headers).status_code == 200
    me = client.get("/api/auth/me", headers=headers).json()
    assert me["is_owner"] is False and me["blocked"] is True
    assert _upload(bid, headers).status_code == 403


def test_unblock_restores_uploads(uploads):
    bid = _building()
    client.post("/api/owner/users/back@example.com/block", json={"blocked": True}, headers=owner_headers())
    client.post("/api/owner/users/back@example.com/block", json={"blocked": False}, headers=owner_headers())
    assert _upload(bid, signed_in_headers("back@example.com")).status_code == 200
    assert len(uploads.seen) == 1


def test_blocked_upload_writes_nothing(uploads):
    bid = _building()
    before = store.load_map(bid).model_dump()
    client.post("/api/owner/users/quiet@example.com/block", json={"blocked": True}, headers=owner_headers())
    assert _upload(bid, signed_in_headers("quiet@example.com")).status_code == 403
    assert store.load_map(bid).model_dump() == before


def test_owner_cannot_be_blocked():
    r = client.post("/api/owner/users/owner@example.com/block", json={"blocked": True}, headers=owner_headers())
    assert r.status_code == 409
    assert store.get_account("owner@example.com").blocked is False


def test_block_rejects_a_bad_email_and_a_bad_body():
    assert client.post("/api/owner/users/not-an-email/block", json={"blocked": True},
                       headers=owner_headers()).status_code == 422
    assert client.post("/api/owner/users/ok@example.com/block", json={"blocked": "maybe"},
                       headers=owner_headers()).status_code == 422
    assert client.post("/api/owner/users/ok@example.com/block", json={}, headers=owner_headers()).status_code == 422


def test_block_creates_the_account_for_a_first_time_spammer():
    r = client.post("/api/owner/users/never-seen@example.com/block", json={"blocked": True}, headers=owner_headers())
    assert r.status_code == 200
    assert r.json()["uploads"] == 0 and r.json()["blocked"] is True


def test_legacy_token_can_block(monkeypatch):
    monkeypatch.setattr(config, "OWNER_TOKEN", "s3cret")
    r = client.post("/api/owner/users/tok@example.com/block", json={"blocked": True},
                    headers={"X-Owner-Token": "s3cret"})
    assert r.status_code == 200 and r.json()["blocked"] is True


# ---------- the contributor's own uploads ----------
def test_my_uploads_requires_a_session():
    assert client.get("/api/me/uploads").status_code == 401
    assert client.get("/api/me/uploads", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_my_uploads_shows_only_my_walks_with_the_four_statuses():
    bid = _building("Status Hall")
    _add_walk(bid, "vmu00001", "me@example.com", status="queued", moderation="pending", created="2026-11-01T09:00:00+00:00")
    _add_walk(bid, "vmu00002", "me@example.com", status="processing", moderation="pending", created="2026-11-01T09:01:00+00:00")
    _add_walk(bid, "vmu00003", "me@example.com", status="done", moderation="approved", created="2026-11-01T09:02:00+00:00")
    _add_walk(bid, "vmu00004", "me@example.com", status="error", moderation="pending", created="2026-11-01T09:03:00+00:00")
    _add_walk(bid, "vmu00005", "me@example.com", status="rejected", moderation="rejected",
              created="2026-11-01T09:04:00+00:00", note="auto: blurry")
    _add_walk(bid, "vmu00006", "me@example.com", status="done", moderation="hidden",
              created="2026-11-01T09:05:00+00:00", note="takedown")
    _add_walk(bid, "vmu00007", "other@example.com", status="done", moderation="approved")
    rows = client.get("/api/me/uploads", headers=signed_in_headers("me@example.com")).json()
    assert [r["video_id"] for r in rows] == ["vmu00006", "vmu00005", "vmu00004", "vmu00003", "vmu00002", "vmu00001"]
    by_id = {r["video_id"]: r for r in rows}
    assert by_id["vmu00001"]["status"] == "processing" and by_id["vmu00002"]["status"] == "processing"
    assert by_id["vmu00003"]["status"] == "live"
    assert by_id["vmu00004"]["status"] == "needs_review"
    assert by_id["vmu00005"]["status"] == "rejected" and by_id["vmu00005"]["reason"] == "auto: blurry"
    assert by_id["vmu00006"]["status"] == "rejected" and by_id["vmu00006"]["reason"] == "takedown"
    assert by_id["vmu00003"]["reason"] is None
    assert set(rows[0]) == {"building_id", "building_name", "video_id", "filename", "status", "reason", "created_at"}
    assert rows[0]["building_name"] == "Status Hall" and rows[0]["building_id"] == bid


def test_my_uploads_is_empty_for_a_new_account():
    assert client.get("/api/me/uploads", headers=signed_in_headers("new@example.com")).json() == []


def test_upload_records_the_uploader_email_and_time(uploads):
    bid = _building()
    r = _upload(bid, signed_in_headers("Uploader@Example.com"))
    assert r.status_code == 200, r.text
    assert uploads.seen[0]["uploader_email"] == "uploader@example.com"
    assert uploads.seen[0]["contributor_id"] == store.get_account("uploader@example.com").id


def test_auth_me_reports_owner_status():
    assert client.get("/api/auth/me", headers=owner_headers()).json()["is_owner"] is True
    contributor = client.get("/api/auth/me", headers=signed_in_headers("plain@example.com")).json()
    assert contributor["is_owner"] is False
    assert contributor["account"]["email"] == "plain@example.com"
    assert set(contributor["account"]) == {"id", "email", "moderator_of"}


def test_backfill_gives_old_walks_their_uploader_email_and_leaves_the_rest_unknown():
    bid = _building("Backfill Hall")
    acct = store.create_account_if_missing("legacy-up@example.com")
    with store.lock_for(bid):
        b = store.load_map(bid)
        b.videos.append(VideoInfo(id="vbf00001", filename="a.mp4", status="done", contributor_id=acct.id))
        b.videos.append(VideoInfo(id="vbf00002", filename="b.mp4", status="done"))  # no contributor: stays unknown
        store.save_map(b)
    assert store.backfill_uploader_emails() == 1
    videos = {v.id: v for v in store.load_map(bid).videos}
    assert videos["vbf00001"].uploader_email == "legacy-up@example.com"
    assert videos["vbf00002"].uploader_email == "unknown"
    assert store.backfill_uploader_emails() == 0  # a second start changes nothing
