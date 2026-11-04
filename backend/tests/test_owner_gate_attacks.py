import inspect
import re

import pytest
from fastapi.testclient import TestClient
from starlette.routing import Route

from app import config, main
from app.main import OwnerWriteGate, app
from app.router import api
from tests.auth_helpers import signed_in_headers

client = TestClient(app)
TOKEN = "s3cret-token"
GATE_DETAIL = "Owner token or sign-in required"
MUTATING = {"POST", "PUT", "PATCH", "DELETE"}


def _fill(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "dummy", path)


def _mutating_routes():
    # app.include_router wraps api routes in _IncludedRouter, so read them from the APIRouter itself.
    # Its route paths already carry the /api prefix.
    out = []
    for route in list(api.routes) + list(app.routes):
        if not isinstance(route, Route):
            continue
        for method in sorted((route.methods or set()) & MUTATING):
            out.append((method, route.path))
    return out


MUTATING_ROUTES = _mutating_routes()
GATED_ROUTES = [
    (m, p) for m, p in MUTATING_ROUTES
    if p.startswith("/api/") and not main._OPEN_WRITES.fullmatch(_fill(p))
]
OPEN_ROUTES = [(m, p) for m, p in MUTATING_ROUTES if (m, p) not in GATED_ROUTES]


@pytest.fixture
def token_set(monkeypatch):
    monkeypatch.setattr(config, "OWNER_TOKEN", TOKEN)


def test_route_enumeration_found_mutating_routes():
    assert len(MUTATING_ROUTES) >= 10
    assert len(GATED_ROUTES) >= 10


@pytest.mark.parametrize("method,path", GATED_ROUTES, ids=[f"{m} {p}" for m, p in GATED_ROUTES])
def test_every_gated_mutating_route_refuses_without_token_or_session(token_set, method, path):
    r = client.request(method, _fill(path), json={})
    assert r.status_code == 401, f"{method} {path} returned {r.status_code}"
    assert r.json() == {"detail": GATE_DETAIL}


@pytest.mark.parametrize("method,path", GATED_ROUTES, ids=[f"{m} {p}" for m, p in GATED_ROUTES])
def test_every_gated_mutating_route_refuses_wrong_token(token_set, method, path):
    r = client.request(method, _fill(path), json={}, headers={"X-Owner-Token": "wrong"})
    assert r.status_code == 401
    assert r.json() == {"detail": GATE_DETAIL}


@pytest.mark.parametrize("method,path", GATED_ROUTES, ids=[f"{m} {p}" for m, p in GATED_ROUTES])
def test_every_gated_mutating_route_refuses_empty_token(token_set, method, path):
    r = client.request(method, _fill(path), json={}, headers={"X-Owner-Token": ""})
    assert r.status_code == 401


@pytest.mark.parametrize("method,path", GATED_ROUTES, ids=[f"{m} {p}" for m, p in GATED_ROUTES])
def test_correct_token_passes_gate(token_set, method, path):
    """The token passes the gate. Upload is contributor-only, so a token alone still gets the route's own 401 there."""
    r = client.request(method, _fill(path), json={}, headers={"X-Owner-Token": TOKEN})
    assert r.json().get("detail") != GATE_DETAIL, f"{method} {path} refused by the gate with the correct token"


@pytest.mark.parametrize("method,path", GATED_ROUTES, ids=[f"{m} {p}" for m, p in GATED_ROUTES])
def test_any_session_passes_gate_so_the_route_decides(token_set, method, path):
    """A signed-in contributor is not stopped by the gate. The route answers: 403 for owner routes, and a normal
    validation answer for upload. Never the gate's own message."""
    r = client.request(method, _fill(path), json={}, headers=signed_in_headers())
    assert r.json().get("detail") != GATE_DETAIL, f"{method} {path} refused by the gate for a session"


def test_wrong_case_header_name_with_correct_value_is_accepted(token_set):
    # HTTP header names are case-insensitive, so x-owner-token with the right value must pass the gate.
    r = client.post("/api/buildings", json={"name": "Case Hall"}, headers={"x-OWNER-token": TOKEN})
    assert r.status_code == 200


def test_similar_header_name_is_refused(token_set):
    r = client.post("/api/buildings", json={}, headers={"X-Owner-Tokn": TOKEN})
    assert r.status_code == 401
    r = client.post("/api/buildings", json={}, headers={"Owner-Token": TOKEN})
    assert r.status_code == 401


def test_empty_token_header_refused_even_if_no_secret(token_set):
    r = client.post("/api/buildings", json={}, headers={"X-Owner-Token": ""})
    assert r.status_code == 401


def test_token_with_trailing_whitespace_refused(token_set):
    r = client.post("/api/buildings", json={}, headers={"X-Owner-Token": TOKEN + " "})
    assert r.status_code == 401
    r = client.post("/api/buildings", json={}, headers={"X-Owner-Token": TOKEN + "\t"})
    assert r.status_code == 401


def test_token_with_leading_whitespace_refused(token_set):
    r = client.post("/api/buildings", json={}, headers={"X-Owner-Token": " " + TOKEN})
    assert r.status_code == 401


def test_token_prefix_refused(token_set):
    r = client.post("/api/buildings", json={}, headers={"X-Owner-Token": TOKEN[:-1]})
    assert r.status_code == 401


def test_token_in_query_string_not_accepted(token_set):
    r = client.post(f"/api/buildings?X-Owner-Token={TOKEN}", json={})
    assert r.status_code == 401
    r = client.post(f"/api/buildings?token={TOKEN}", json={})
    assert r.status_code == 401
    r = client.post(f"/api/buildings?owner_token={TOKEN}", json={})
    assert r.status_code == 401


def test_token_in_body_or_cookie_not_accepted(token_set):
    r = client.post("/api/buildings", json={"owner_token": TOKEN})
    assert r.status_code == 401
    r = client.post("/api/buildings", json={}, cookies={"owner_token": TOKEN})
    assert r.status_code == 401


def test_token_sent_as_bearer_is_not_a_session(token_set):
    """A Bearer header passes the gate (it is a session header) but is not a session: the route refuses it."""
    r = client.post("/api/buildings", json={}, headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 401
    assert r.json() == {"detail": "Sign-in required"}


def test_get_on_mutating_paths_not_gated(token_set):
    for _method, path in GATED_ROUTES:
        r = client.get(_fill(path))
        body = r.json()
        assert not (isinstance(body, dict) and body.get("detail") == GATE_DETAIL), f"GET {path} gated"


def test_head_and_options_not_gated(token_set):
    assert client.options("/api/buildings", headers={"Origin": "http://x"}).status_code != 401
    assert client.head("/api/buildings").status_code != 401


@pytest.mark.parametrize("path", [
    "/api/route",
    "/api/buildings/x/route",
    "/api/buildings/x/reports",
])
def test_public_routes_stay_open(token_set, path):
    r = client.post(path, json={})
    assert r.status_code != 401


def test_public_routes_are_the_only_open_mutating_api_routes():
    open_api = sorted(p for _m, p in OPEN_ROUTES if p.startswith("/api/"))
    assert open_api == sorted(open_api)
    for _m, p in OPEN_ROUTES:
        if p.startswith("/api/"):
            assert main._OPEN_WRITES.fullmatch(_fill(p)), p


def test_non_api_mutating_routes_are_none():
    # The gate only inspects /api/ paths. Any mutating route outside /api/ would bypass it.
    offenders = [(m, p) for m, p in MUTATING_ROUTES if not p.startswith("/api/")]
    assert offenders == [], f"mutating routes outside /api/ bypass the gate: {offenders}"


@pytest.mark.parametrize("path", [
    "/api/buildings/x/routes",
    "/api/buildings/x/route/extra",
    "/api/route/",
    "/api/buildings/x/reports/y/z",
])
def test_lookalike_paths_do_not_match_open_list(token_set, path):
    assert not main._OPEN_WRITES.fullmatch(path)


def test_gate_uses_hmac_compare_digest():
    src = inspect.getsource(OwnerWriteGate.__call__)
    assert "hmac.compare_digest" in src
    assert not re.search(r"given\s*==|==\s*config\.OWNER_TOKEN|OWNER_TOKEN\s*==", src)


def test_gate_is_off_when_token_unset(monkeypatch):
    # Documented fail-open: an empty OWNER_TOKEN disables the gate; the route still needs an owner.
    monkeypatch.setattr(config, "OWNER_TOKEN", "")
    assert client.post("/api/buildings", json={}).json() == {"detail": "Owner sign-in required"}
