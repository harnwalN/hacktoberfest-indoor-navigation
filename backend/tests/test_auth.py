"""Magic-link sign-in (auth.py, auth_router.py, mail.py): the three routes, single-use and expiring links, hashed tokens
at rest, session gating of uploads and building creation (with the contributor recorded), reads left open, the owner
gate still in place beside it, and the legacy-walk backfill. No network: SMTP is a fake class, time is monkeypatched."""
import json
import logging
import smtplib

import pytest
from fastapi.testclient import TestClient

from app import auth, config, jobs, mail, store
from app.main import app
from tests.auth_helpers import owner_headers, signed_in_headers

client = TestClient(app)
BASE = "https://nav.example.org"


@pytest.fixture(autouse=True)
def _base_url(_temp_data_dir, monkeypatch):
    monkeypatch.setattr(config, "PUBLIC_BASE_URL", BASE)
    monkeypatch.setattr(config, "MAIL_BACKEND", "log")
    monkeypatch.setattr(config, "BUILDINGS_DIR", config.DATA_DIR / "buildings")


@pytest.fixture
def sent(monkeypatch):
    """Records what send_login_link was asked to mail, instead of mailing it."""
    calls = []
    monkeypatch.setattr(mail, "send_login_link", lambda email, link: calls.append((email, link)))
    return calls


def _token_from(sent_calls) -> str:
    link = sent_calls[-1][1]
    return link.split("token=", 1)[1]


def _request_link(email="Walker@Example.com "):
    return client.post("/api/auth/request-link", json={"email": email})


# ---------- request-link ----------
def test_request_link_is_generic_and_creates_account(sent):
    r = _request_link()
    assert r.status_code == 200
    assert r.json()["status"] == "sent"
    assert store.get_account("walker@example.com") is not None
    assert [e for e, _ in sent] == ["walker@example.com"]  # normalized before mailing


def test_request_link_same_answer_for_existing_address(sent):
    first = _request_link("a@example.com")
    again = _request_link("A@example.com")
    assert first.status_code == again.status_code == 200
    assert first.json() == again.json()
    assert len(store._read_accounts()) == 1


@pytest.mark.parametrize("bad", ["", "not-an-email", "a@b", "two@@example.com", "x" * 300])
def test_request_link_rejects_malformed_email(sent, bad):
    r = _request_link(bad)
    assert r.status_code == 422
    assert sent == []


def test_request_link_link_points_at_public_base_url(sent):
    _request_link()
    link = sent[-1][1]
    assert link.startswith(f"{BASE}/#/verify?token=")


def test_request_link_ignores_host_header_when_base_url_set(sent):
    client.post("/api/auth/request-link", json={"email": "a@example.com"}, headers={"Host": "evil.example"})
    assert sent[-1][1].startswith(f"{BASE}/#/verify?token=")


def test_request_link_without_base_url_uses_request_origin(sent, monkeypatch):
    monkeypatch.setattr(config, "PUBLIC_BASE_URL", "")
    _request_link()
    assert sent[-1][1].startswith("http://testserver/#/verify?token=")


def test_log_mode_logs_link_and_sends_nothing(caplog, monkeypatch):
    smtp_used = []
    monkeypatch.setattr(smtplib, "SMTP", lambda *a, **k: smtp_used.append(a))
    with caplog.at_level(logging.WARNING, logger="app.mail"):
        assert _request_link().status_code == 200
    assert smtp_used == []
    assert "sign-in link for walker@example.com" in caplog.text
    assert "/#/verify?token=" in caplog.text


def test_mail_failure_still_answers_200_and_logs(monkeypatch, caplog):
    def broken(email, link):
        raise mail.MailError("sending sign-in mail failed: SMTPServerDisconnected")

    monkeypatch.setattr(mail, "send_login_link", broken)
    with caplog.at_level(logging.ERROR, logger="app.auth_router"):
        r = _request_link("c@example.com")
    assert r.status_code == 200
    assert "sending sign-in mail failed: SMTPServerDisconnected" in caplog.text


# ---------- verify ----------
def test_verify_returns_session_and_account(sent):
    _request_link("d@example.com")
    r = client.post("/api/auth/verify", json={"token": _token_from(sent)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"session_token", "expires_in", "account"}
    assert body["expires_in"] == config.AUTH_SESSION_TTL_SEC
    assert body["account"]["email"] == "d@example.com"
    assert body["account"]["moderator_of"] == []
    assert len(body["session_token"]) >= 40


def test_link_is_single_use(sent):
    _request_link("e@example.com")
    token = _token_from(sent)
    assert client.post("/api/auth/verify", json={"token": token}).status_code == 200
    assert client.post("/api/auth/verify", json={"token": token}).status_code == 401


def test_expired_link_refused(sent, monkeypatch):
    _request_link("f@example.com")
    token = _token_from(sent)
    later = auth._now() + config.AUTH_LINK_TTL_SEC + 1
    monkeypatch.setattr(auth, "_now", lambda: later)
    r = client.post("/api/auth/verify", json={"token": token})
    assert r.status_code == 401


def test_link_valid_just_before_expiry(sent, monkeypatch):
    _request_link("g@example.com")
    token = _token_from(sent)
    real_now = auth._now()
    monkeypatch.setattr(auth, "_now", lambda: real_now + config.AUTH_LINK_TTL_SEC - 5)
    assert client.post("/api/auth/verify", json={"token": token}).status_code == 200


def test_unknown_and_empty_tokens_refused():
    assert client.post("/api/auth/verify", json={"token": "nope"}).status_code == 401
    assert client.post("/api/auth/verify", json={"token": ""}).status_code == 422
    assert client.post("/api/auth/verify", json={}).status_code == 422
    assert client.post("/api/auth/verify", json={"token": "t" * 500}).status_code == 422


def test_session_expires(sent, monkeypatch):
    _request_link("h@example.com")
    session = client.post("/api/auth/verify", json={"token": _token_from(sent)}).json()["session_token"]
    headers = {"Authorization": f"Bearer {session}"}
    assert client.get("/api/auth/me", headers=headers).status_code == 200
    real_now = auth._now()
    monkeypatch.setattr(auth, "_now", lambda: real_now + config.AUTH_SESSION_TTL_SEC + 1)
    assert client.get("/api/auth/me", headers=headers).status_code == 401


# ---------- stored tokens are hashed ----------
def test_raw_tokens_are_not_stored(sent):
    _request_link("i@example.com")
    link_token = _token_from(sent)  # still pending: not yet used
    body = client.post("/api/auth/verify", json={"token": link_token}).json()
    session = body["session_token"]
    text = auth.auth_path().read_text()
    assert link_token not in text  # used now, and never stored in clear
    assert session not in text
    data = json.loads(text)
    assert data["magic_links"] == []  # consumed link removed
    assert [len(r["hash"]) for r in data["sessions"]] == [64]


def test_session_is_read_from_disk_on_each_request(sent):
    """Nothing is cached in memory: a session written to data/auth.json works for the next request."""
    _request_link("j@example.com")
    session = client.post("/api/auth/verify", json={"token": _token_from(sent)}).json()["session_token"]
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {session}"}).status_code == 200


# ---------- me ----------
def test_me_with_session(sent):
    headers = signed_in_headers("k@example.com")
    r = client.get("/api/auth/me", headers=headers)
    assert r.status_code == 200
    assert r.json()["account"]["email"] == "k@example.com"


@pytest.mark.parametrize("headers", [
    {}, {"Authorization": "Bearer"}, {"Authorization": "Bearer "}, {"Authorization": "Basic abc"},
    {"Authorization": "Bearer not-a-session"}, {"X-Session-Token": "x"},
])
def test_me_without_valid_session_is_401(headers):
    r = client.get("/api/auth/me", headers=headers)
    assert r.status_code == 401
    assert r.headers.get("www-authenticate") == "Bearer"


def test_owner_token_is_not_a_session(monkeypatch):
    monkeypatch.setattr(config, "OWNER_TOKEN", "s3cret")
    assert client.get("/api/auth/me", headers={"Authorization": "Bearer s3cret"}).status_code == 401


# ---------- uploads and building creation need a session ----------
@pytest.fixture
def walks(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "JOBS", {})
    monkeypatch.setattr(jobs, "_ensure_worker", lambda: None)
    submitted = []

    def fake_submit(building_id, upload_path, filename, sensors, model, front=False, start_floor=None,
                    contributor_id=None, uploader_email="unknown"):
        submitted.append({"building_id": building_id, "contributor_id": contributor_id})
        return jobs.Job(id="j1", building_id=building_id, video_id="v1200001234", kind="video")

    monkeypatch.setattr(jobs, "submit_video", fake_submit)
    return submitted


def test_upload_without_session_is_401(walks):
    bid = store.create_building("Gate Hall").building_id
    r = client.post(f"/api/buildings/{bid}/videos", files={"video": ("w.mp4", b"x" * 10, "video/mp4")})
    assert r.status_code == 401
    assert walks == []


def test_upload_with_session_records_contributor(walks):
    bid = store.create_building("Upload Hall").building_id
    headers = signed_in_headers("walker@example.com")
    acct = store.get_account("walker@example.com")
    r = client.post(f"/api/buildings/{bid}/videos", headers=headers,
                    files={"video": ("w.mp4", b"x" * 10, "video/mp4")})
    assert r.status_code == 200, r.text
    assert walks == [{"building_id": bid, "contributor_id": acct.id}]


def test_upload_with_session_on_unknown_building_is_404(walks):
    r = client.post("/api/buildings/nope-nope/videos", headers=signed_in_headers(),
                    files={"video": ("w.mp4", b"x", "video/mp4")})
    assert r.status_code == 404
    assert walks == []


def test_create_building_without_session_is_401(tmp_path):
    before = store.list_buildings()
    r = client.post("/api/buildings", json={"name": "Nope Hall"})
    assert r.status_code == 401
    assert store.list_buildings() == before


def test_create_building_makes_creator_moderator():
    headers = owner_headers()  # creating a building is owner-only; the owner becomes its moderator
    r = client.post("/api/buildings", json={"name": "Creator Hall"}, headers=headers)
    assert r.status_code == 200, r.text
    bid = r.json()["building_id"]
    acct = store.get_account("owner@example.com")
    assert store.is_moderator(acct.id, bid)
    assert client.get("/api/auth/me", headers=headers).json()["account"]["moderator_of"] == [bid]


def test_reads_stay_open_without_session():
    store.create_building("Open Hall")
    assert client.get("/api/buildings").status_code == 200


def test_sign_in_routes_open_when_owner_gate_on(sent, monkeypatch):
    """The owner gate is additive: with OWNER_TOKEN set, sign-in still works without the owner token."""
    monkeypatch.setattr(config, "OWNER_TOKEN", "s3cret")
    assert _request_link("owner-gate@example.com").status_code == 200
    token = _token_from(sent)
    assert client.post("/api/auth/verify", json={"token": token}).status_code == 200


def test_owner_gate_lets_a_contributor_session_through_to_the_owner_check(monkeypatch):
    """With OWNER_TOKEN set, a signed-in contributor passes the gate; the route then refuses non-owners (403)."""
    monkeypatch.setattr(config, "OWNER_TOKEN", "s3cret")
    r = client.post("/api/buildings", json={"name": "Gated"}, headers=signed_in_headers())
    assert r.status_code == 403


# ---------- SMTP mode ----------
class FakeSMTP:
    instances: list = []
    fail_on_send = None

    def __init__(self, host, port, timeout=None):
        self.host, self.port, self.timeout = host, port, timeout
        self.calls = []
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def ehlo(self):
        self.calls.append("ehlo")

    def starttls(self, context=None):
        self.calls.append("starttls")

    def login(self, user, password):
        self.calls.append(("login", user, password))

    def send_message(self, msg):
        self.calls.append("send")
        self.msg = msg
        if FakeSMTP.fail_on_send:
            raise FakeSMTP.fail_on_send


@pytest.fixture
def smtp(monkeypatch):
    FakeSMTP.instances = []
    FakeSMTP.fail_on_send = None
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(config, "MAIL_BACKEND", "smtp")
    monkeypatch.setattr(config, "SMTP_HOST", "smtp.example.com")
    monkeypatch.setattr(config, "SMTP_PORT", 2525)
    monkeypatch.setattr(config, "SMTP_USER", "mailer")
    monkeypatch.setattr(config, "SMTP_PASSWORD", "hunter2")
    monkeypatch.setattr(config, "SMTP_FROM", "no-reply@example.com")
    monkeypatch.setattr(config, "SMTP_STARTTLS", True)
    return FakeSMTP


def test_smtp_mode_sends_link_with_starttls_and_login(smtp):
    r = _request_link("mail@example.com")
    assert r.status_code == 200
    conn = smtp.instances[-1]
    assert (conn.host, conn.port) == ("smtp.example.com", 2525)
    assert conn.calls == ["ehlo", "starttls", "ehlo", ("login", "mailer", "hunter2"), "send"]
    msg = conn.msg
    assert msg["To"] == "mail@example.com" and msg["From"] == "no-reply@example.com"
    body = msg.get_body().get_content()
    assert f"{BASE}/#/verify?token=" in body
    assert "15 minutes" in body


def test_smtp_mode_without_starttls_or_login(smtp, monkeypatch):
    monkeypatch.setattr(config, "SMTP_STARTTLS", False)
    monkeypatch.setattr(config, "SMTP_USER", "")
    _request_link("plain@example.com")
    assert smtp.instances[-1].calls == ["ehlo", "send"]


def test_smtp_failure_is_logged_without_password(smtp, caplog):
    smtp.fail_on_send = smtplib.SMTPRecipientsRefused({"x": (550, b"no such user")})
    with caplog.at_level(logging.ERROR, logger="app.auth_router"):
        r = _request_link("bounce@example.com")
    assert r.status_code == 200
    assert "sending sign-in mail failed: SMTPRecipientsRefused" in caplog.text
    assert "hunter2" not in caplog.text


def test_smtp_connection_error_is_mail_error(monkeypatch):
    def refuse(*a, **k):
        raise ConnectionRefusedError(111, "refused")

    monkeypatch.setattr(smtplib, "SMTP", refuse)
    monkeypatch.setattr(config, "MAIL_BACKEND", "smtp")
    monkeypatch.setattr(config, "SMTP_FROM", "no-reply@example.com")
    with pytest.raises(mail.MailError, match="ConnectionRefusedError"):
        mail.send_login_link("x@example.com", f"{BASE}/#/verify?token=t")


def test_smtp_without_from_is_mail_error(monkeypatch):
    monkeypatch.setattr(config, "MAIL_BACKEND", "smtp")
    monkeypatch.setattr(config, "SMTP_FROM", "")
    with pytest.raises(mail.MailError, match="SMTP_FROM"):
        mail.send_login_link("x@example.com", f"{BASE}/#/verify?token=t")


# ---------- legacy walks ----------
def _walk_map(bid, videos):
    b = store.load_map(bid)
    b.videos = videos
    store.save_map(b)


def test_legacy_done_walks_are_approved_once(tmp_path):
    from app.schemas import VideoInfo
    bid = store.create_building("Legacy Hall").building_id
    legacy = store.video_dir(bid, "vlegacy1")
    legacy.mkdir(parents=True)
    (legacy / "walk.json").write_text(json.dumps({"video_id": "vlegacy1", "keyframes": []}))  # no moderation record
    newer = store.video_dir(bid, "vnew0001")
    newer.mkdir(parents=True)
    (newer / "walk.json").write_text(json.dumps({"moderation": {"approved": True}}))
    _walk_map(bid, [
        VideoInfo(id="vlegacy1", filename="a.mp4", status="done"),
        VideoInfo(id="vnew0001", filename="b.mp4", status="done", moderation_status="pending"),
        VideoInfo(id="vqueued1", filename="c.mp4", status="queued"),
        VideoInfo(id="vnowalk1", filename="d.mp4", status="done"),
    ])
    assert store.approve_legacy_walks() == 2  # vlegacy1 and vnowalk1 (no walk.json at all, still published)
    got = {v.id: v.moderation_status for v in store.load_map(bid).videos}
    assert got == {"vlegacy1": "approved", "vnew0001": "pending", "vqueued1": "pending", "vnowalk1": "approved"}
    assert store.approve_legacy_walks() == 0  # idempotent


def test_legacy_backfill_leaves_rejected_walks_alone():
    from app.schemas import VideoInfo
    bid = store.create_building("Rejected Hall").building_id
    _walk_map(bid, [VideoInfo(id="vrej0001", filename="r.mp4", status="rejected", moderation_status="rejected")])
    assert store.approve_legacy_walks() == 0
    assert store.load_map(bid).videos[0].moderation_status == "rejected"
