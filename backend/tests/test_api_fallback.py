"""/api paths that no route answers get a JSON 404, whether or not a built frontend (backend/static) is mounted.

The app is built here with a temporary static folder, so the result does not depend on what is in backend/static."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.auth_router import auth_api
from app.main import app as real_app
from app.main import serve_frontend
from app.owner_router import owner_api
from app.router import api


@pytest.fixture(params=["static_built", "static_absent"])
def client(request, tmp_path):
    static = tmp_path / "static"
    if request.param == "static_built":
        static.mkdir()
        (static / "index.html").write_text("<!doctype html><title>spa</title>")
    application = FastAPI()
    application.include_router(api)
    application.include_router(auth_api)
    application.include_router(owner_api)
    serve_frontend(application, static)
    return TestClient(application)


@pytest.mark.parametrize("method,path", [
    ("GET", "/api/nope"),
    ("DELETE", "/api/nope"),
    ("DELETE", "/api/buildings/..%2Fetc"),
    ("DELETE", "/api/buildings/../etc"),
    ("PUT", "/api/buildings/..%2Fetc/map"),
    ("POST", "/api"),
    ("GET", "/api/"),
])
def test_unknown_api_path_is_json_404(client, method, path):
    r = client.request(method, path)
    assert r.status_code == 404, r.text
    assert r.headers["content-type"].startswith("application/json")
    assert r.json() == {"detail": "Not Found"}


def test_known_api_path_with_wrong_method_is_json_405(client):
    r = client.delete("/api/buildings/some-id/map")
    assert r.status_code == 405, r.text
    assert r.headers["content-type"].startswith("application/json")
    assert "GET" in r.headers["allow"]
    assert r.json() == {"detail": "Method Not Allowed"}


def test_known_api_path_still_routes(client):
    r = client.get("/api/buildings")
    assert r.status_code == 200 and isinstance(r.json(), list)


def test_root_serves_built_frontend_only_when_present(tmp_path):
    built = tmp_path / "built"
    built.mkdir()
    (built / "index.html").write_text("<!doctype html><title>spa</title>")
    with_static = FastAPI()
    with_static.include_router(api)
    serve_frontend(with_static, built)
    assert "spa" in TestClient(with_static).get("/").text

    without = FastAPI()
    without.include_router(api)
    serve_frontend(without, tmp_path / "missing")
    assert "API running" in TestClient(without).get("/").text


def test_real_app_unknown_api_path_is_json_404():
    r = TestClient(real_app).delete("/api/buildings/..%2Fetc")
    assert r.status_code == 404
    assert r.json() == {"detail": "Not Found"}
