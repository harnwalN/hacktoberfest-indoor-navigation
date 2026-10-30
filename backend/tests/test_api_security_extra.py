"""Sentinel additions: HEAD on frames, CORS on 413, generated frame ids vs the /bfiles filename regex."""
import re

import pytest

from app import config, store
from app.main import _FRAME_NAME
from tests.test_api_security import _jpeg, env  # noqa: F401  (env is a fixture)


def test_frame_head_ok_and_head_on_private_404(env):
    bid, c = env["bid"], env["client"]
    (store.frames_dir(bid) / "v1200001234_0001.jpg").write_bytes(_jpeg())
    assert c.head(f"/bfiles/{bid}/frames/v1200001234_0001.jpg").status_code == 200
    assert c.head(f"/bfiles/{bid}/map.json").status_code == 404
    assert c.head(f"/bfiles/{bid}/frames/nope.jpg").status_code == 404


@pytest.mark.parametrize("name", ["v123456ab12_0001.jpg", "v000000ffff_0123.jpg", "manual_0000.jpeg",
                                  "manual_0012.png", "manual_0003.webp"])
def test_generated_frame_names_match_regex(name):
    assert _FRAME_NAME.fullmatch(name)


def test_cors_on_413_middleware_and_route(env, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 1)
    monkeypatch.setattr(config, "MAX_SENSOR_BYTES", 1000)
    h = {"Origin": "http://example.com"}
    c, bid = env["client"], env["bid"]
    r = c.post(f"/api/buildings/{bid}/videos", headers=h,
               files={"video": ("w.mp4", b"x" * (3 * 1024 * 1024), "video/mp4")})
    assert r.status_code == 413 and r.headers.get("access-control-allow-origin") == "*"
    monkeypatch.setattr(config, "MAX_SENSOR_BYTES", 100)
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 100)
    r = c.post(f"/api/buildings/{bid}/videos", headers=h,
               files={"video": ("w.mp4", b"x", "video/mp4"), "sensors": ("s.json", '{"a":"' + "y" * 300 + '"}')})
    assert r.status_code == 413 and r.headers.get("access-control-allow-origin") == "*"
