"""Rate limit on POST /api/auth/request-link: per client IP and per email address, sliding window, in memory.
Covers: under the limit succeeds, over the limit by IP is 429, over the limit by email is 429 from another IP,
the window resets as time passes (time is monkeypatched, nothing sleeps), refused calls create no account, and the
429 answer does not reveal whether the address has an account. No network: mail is stubbed."""
import pytest
from fastapi.testclient import TestClient

from app import auth, config, mail, store
from app.main import app

BASE = "https://nav.example.org"
IP_A = "203.0.113.10"
IP_B = "198.51.100.20"


def _client(ip: str) -> TestClient:
    return TestClient(app, client=(ip, 50000))


@pytest.fixture(autouse=True)
def _setup(_temp_data_dir, monkeypatch):
    monkeypatch.setattr(config, "PUBLIC_BASE_URL", BASE)
    monkeypatch.setattr(config, "MAIL_BACKEND", "log")
    monkeypatch.setattr(config, "AUTH_LINK_RATE_LIMIT", 5)
    monkeypatch.setattr(config, "AUTH_LINK_RATE_WINDOW_SEC", 900)
    monkeypatch.setattr(mail, "send_login_link", lambda email, link: None)
    monkeypatch.setattr(auth, "_link_hits", {})
    # Fixed clock. Tests move it forward instead of sleeping.
    now = {"t": 1_000_000.0}
    monkeypatch.setattr(auth, "_now", lambda: now["t"])
    return now


def _post(ip: str, email: str):
    return _client(ip).post("/api/auth/request-link", json={"email": email})


def test_under_limit_succeeds_repeatedly(_setup):
    for i in range(5):
        assert _post(IP_A, "same@example.com").status_code == 200
    for i in range(5):
        assert _post(IP_B, f"user{i}@example.com").status_code == 200


def test_over_limit_by_ip_gets_429(_setup):
    for i in range(5):
        assert _post(IP_A, f"ip{i}@example.com").status_code == 200
    r = _post(IP_A, "sixth@example.com")
    assert r.status_code == 429
    assert r.headers["Retry-After"] == "900"
    assert "Try again in 900 seconds" in r.json()["detail"]
    # The same IP is blocked for any address, not only the ones it already used.
    assert _post(IP_A, "ip0@example.com").status_code == 429
    # Another IP is not affected.
    assert _post(IP_B, "fresh@example.com").status_code == 200


def test_over_limit_by_email_gets_429_from_other_ip(_setup):
    for _ in range(5):
        assert _post(IP_A, "target@example.com").status_code == 200
    assert _post(IP_B, "target@example.com").status_code == 429
    assert _post(IP_B, "Target@Example.com ").status_code == 429  # normalized before counting
    # Other addresses from the other IP still work.
    assert _post(IP_B, "other@example.com").status_code == 200


def test_window_resets_as_time_passes(_setup, monkeypatch):
    now = _setup
    start = now["t"]
    for i in range(5):
        assert _post(IP_A, f"w{i}@example.com").status_code == 200
    assert _post(IP_A, "w-late@example.com").status_code == 429

    now["t"] = start + 899  # one second short of the window: still blocked
    assert _post(IP_A, "w-late@example.com").status_code == 429

    now["t"] = start + 900  # the first hits have left the window
    assert _post(IP_A, "w-late@example.com").status_code == 200


def test_sliding_window_frees_one_slot_at_a_time(_setup):
    now = _setup
    start = now["t"]
    for i in range(5):
        now["t"] = start + i * 10
        assert _post(IP_A, f"s{i}@example.com").status_code == 200
    now["t"] = start + 905  # only the hit at start+0 has expired
    assert _post(IP_A, "s-new@example.com").status_code == 200
    r = _post(IP_A, "s-newer@example.com")
    assert r.status_code == 429
    assert r.headers["Retry-After"] == "5"  # hit at start+10 leaves the window at start+910


def test_refused_request_creates_no_account_and_sends_no_mail(_setup, monkeypatch):
    sent = []
    monkeypatch.setattr(mail, "send_login_link", lambda email, link: sent.append(email))
    for i in range(5):
        _post(IP_A, f"ok{i}@example.com")
    sent.clear()
    assert _post(IP_A, "blocked-new@example.com").status_code == 429
    assert store.get_account("blocked-new@example.com") is None
    assert sent == []


def test_429_does_not_reveal_existing_account(_setup):
    store.create_account_if_missing("exists@example.com")
    for _ in range(5):
        assert _post(IP_A, "exists@example.com").status_code == 200
    # Both addresses are refused for the same reason and with the same answer.
    known = _post(IP_A, "exists@example.com")
    unknown = _post(IP_A, "nobody-here@example.com")
    assert known.status_code == unknown.status_code == 429
    assert known.json() == unknown.json()
    assert known.headers["Retry-After"] == unknown.headers["Retry-After"]


def test_wait_seconds_match_oldest_hit(_setup):
    now = _setup
    start = now["t"]
    assert auth.link_request_wait(IP_A, "x@example.com") == 0
    now["t"] = start + 100
    for i in range(3):
        assert auth.link_request_wait(IP_A, f"y{i}@example.com") == 0
    # Fifth hit from the IP (at start+100) is allowed; the sixth must wait for the first hit (start+0) to expire.
    assert auth.link_request_wait(IP_A, "y3@example.com") == 0
    assert auth.link_request_wait(IP_A, "y4@example.com") == 800

