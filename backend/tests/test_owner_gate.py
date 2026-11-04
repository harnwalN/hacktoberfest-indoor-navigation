"""OWNER_TOKEN gate (main.OwnerWriteGate) with the owner rules on top.

With the token set, a map-changing /api request must carry a valid X-Owner-Token or an Authorization header. The gate
then lets it through and the route decides: owner routes need an owner session or the token, upload needs a
contributor session that is not blocked, reads and public endpoints stay open.
"""
import pytest
from fastapi.testclient import TestClient

from app import config
from app.main import app
from tests.auth_helpers import owner_headers, signed_in_headers

client = TestClient(app)
GATE_DETAIL = "Owner token or sign-in required"


@pytest.fixture
def token_set(monkeypatch):
    monkeypatch.setattr(config, "OWNER_TOKEN", "s3cret")


def test_gate_off_without_token_leaves_the_route_to_decide(monkeypatch):
    monkeypatch.setattr(config, "OWNER_TOKEN", "")
    r = client.post("/api/buildings", json={})
    assert r.status_code == 401 and r.json() == {"detail": "Owner sign-in required"}


def test_write_refused_without_token_or_session(token_set):
    assert client.post("/api/buildings", json={}).json() == {"detail": GATE_DETAIL}
    assert client.post("/api/buildings", json={}).status_code == 401
    assert client.delete("/api/buildings/x/videos/y").status_code == 401


def test_write_refused_with_wrong_token(token_set):
    r = client.post("/api/buildings", json={}, headers={"X-Owner-Token": "nope"})
    assert r.status_code == 401 and r.json() == {"detail": GATE_DETAIL}


def test_legacy_token_alone_creates_a_building(token_set):
    r = client.post("/api/buildings", json={"name": "Token Hall"}, headers={"X-Owner-Token": "s3cret"})
    assert r.status_code == 200, r.text
    assert r.json()["building_id"] == "token-hall"


def test_contributor_session_passes_the_gate_and_is_refused_by_the_route(token_set):
    r = client.post("/api/buildings", json={}, headers=signed_in_headers())
    assert r.status_code == 403 and r.json() == {"detail": "Only the owner can do this."}


def test_owner_session_passes_the_gate_without_the_token(token_set):
    r = client.post("/api/buildings", json={"name": "Owner Hall"}, headers=owner_headers())
    assert r.status_code == 200, r.text


def test_reads_and_public_endpoints_stay_open(token_set):
    assert client.get("/api/buildings").status_code != 401
    assert client.post("/api/buildings/x/route", json={}).status_code != 401
    assert client.post("/api/buildings/x/reports", json={}).status_code != 401
