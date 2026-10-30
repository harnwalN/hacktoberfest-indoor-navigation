"""Upload limits, temp-file hygiene, video-id validation and the frames-only /bfiles route.

Runs without ffmpeg or Ollama: job submission and Gemma are stubbed out.
"""
import io
import json
import os
import shutil
import tempfile

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from app import config, gemma, jobs, router, store
from app.main import app
from app.schemas import VideoInfo


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BUILDINGS_DIR", tmp_path / "buildings")
    tmpdir = tmp_path / "tmp"
    tmpdir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmpdir))

    def no_gemma(*_a, **_k):
        raise gemma.GemmaUnavailable("disabled in tests")

    monkeypatch.setattr(gemma, "resolve_model", no_gemma)

    submitted = []

    def fake_submit(building_id, upload_path, filename, sensors, model, front=False, start_floor=None):
        dest = tmp_path / f"moved_{len(submitted)}"
        shutil.move(str(upload_path), dest)  # like the real one: the temp file is moved away
        submitted.append({"path": dest, "sensors": sensors, "filename": filename})
        return jobs.Job(id=f"j{len(submitted)}", building_id=building_id, video_id="v1200001234", kind="video")

    monkeypatch.setattr(jobs, "submit_video", fake_submit)
    bundle = store.create_building("Test Hall")
    return {"client": TestClient(app), "bid": bundle.building_id, "tmpdir": tmpdir, "submitted": submitted,
            "root": tmp_path}


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 24), (200, 30, 30)).save(buf, "JPEG")
    return buf.getvalue()


# ---------- A: upload limits + fd leak ----------
def test_video_upload_ok_and_fd_closed(env, monkeypatch):
    fds = []
    real_mkstemp = tempfile.mkstemp

    def spy(*a, **k):
        fd, name = real_mkstemp(*a, **k)
        fds.append(fd)
        return fd, name

    monkeypatch.setattr(router.tempfile, "mkstemp", spy)
    r = env["client"].post(f"/api/buildings/{env['bid']}/videos",
                           files={"video": ("walk.mp4", b"x" * 3000, "video/mp4"),
                                  "sensors": ("sensors.json", json.dumps({"a": 1}), "application/json")})
    assert r.status_code == 200, r.text
    assert r.json()["job_id"] == "j1"
    sub = env["submitted"][0]
    assert sub["path"].read_bytes() == b"x" * 3000
    assert sub["sensors"] == {"a": 1}
    assert fds and all(_closed(fd) for fd in fds)
    assert list(env["tmpdir"].iterdir()) == []


def _closed(fd: int) -> bool:
    try:
        os.fstat(fd)
    except OSError:
        return True
    return False


def test_oversized_video_413_no_temp_left(env, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 1)
    r = env["client"].post(f"/api/buildings/{env['bid']}/videos",
                           files={"video": ("walk.mp4", b"x" * (1024 * 1024 + 10), "video/mp4")})
    assert r.status_code == 413
    assert env["submitted"] == []
    assert list(env["tmpdir"].iterdir()) == []


def test_content_length_over_total_limit_rejected_early(env, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 1)
    monkeypatch.setattr(config, "MAX_SENSOR_BYTES", 1000)
    r = env["client"].post(f"/api/buildings/{env['bid']}/videos",
                           files={"video": ("walk.mp4", b"x" * (3 * 1024 * 1024), "video/mp4")})
    assert r.status_code == 413
    assert env["submitted"] == []
    assert list(env["tmpdir"].iterdir()) == []


def test_oversized_sensors_413(env, monkeypatch):
    monkeypatch.setattr(config, "MAX_SENSOR_BYTES", 100)
    r = env["client"].post(f"/api/buildings/{env['bid']}/videos",
                           files={"video": ("walk.mp4", b"x" * 10, "video/mp4"),
                                  "sensors": ("s.json", json.dumps({"k": "y" * 200}), "application/json")})
    assert r.status_code == 413
    assert env["submitted"] == []
    assert list(env["tmpdir"].iterdir()) == []


def test_bad_sensors_400_no_temp_left(env):
    r = env["client"].post(f"/api/buildings/{env['bid']}/videos",
                           files={"video": ("walk.mp4", b"x" * 10, "video/mp4"),
                                  "sensors": ("s.json", b"\xff\xfenot json", "application/json")})
    assert r.status_code == 400
    assert list(env["tmpdir"].iterdir()) == []


def test_manual_frame_ok(env):
    r = env["client"].post(f"/api/buildings/{env['bid']}/frames/manual",
                           files={"image": ("sign.html", _jpeg(), "image/jpeg")}, data={"label": "Sign"})
    assert r.status_code == 200, r.text
    frame_id = r.json()["node"]["frame_id"]
    assert frame_id == "manual_0000.jpg"  # extension comes from the decoded format, not the client filename
    assert (store.frames_dir(env["bid"]) / frame_id).is_file()


def test_manual_frame_non_image_400(env):
    r = env["client"].post(f"/api/buildings/{env['bid']}/frames/manual",
                           files={"image": ("x.jpg", b"definitely not an image", "image/jpeg")})
    assert r.status_code == 400
    assert list(store.frames_dir(env["bid"]).iterdir()) == []


def test_manual_frame_truncated_image_400(env):
    r = env["client"].post(f"/api/buildings/{env['bid']}/frames/manual",
                           files={"image": ("x.jpg", _jpeg()[:200], "image/jpeg")})
    assert r.status_code == 400
    assert list(store.frames_dir(env["bid"]).iterdir()) == []


def test_manual_frame_too_large_413(env, monkeypatch):
    monkeypatch.setattr(config, "MAX_IMAGE_BYTES", 100)
    r = env["client"].post(f"/api/buildings/{env['bid']}/frames/manual",
                           files={"image": ("x.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 413
    assert list(store.frames_dir(env["bid"]).iterdir()) == []


# ---------- B: path traversal + frames-only static route ----------
@pytest.mark.parametrize("vid", ["..", "../..", "..%2F..", "v1/../..", "v.1", "v*", ""])
def test_video_dir_rejects_unsafe_ids(vid):
    with pytest.raises(ValueError):
        store.video_dir("some-building", vid)


def test_video_dir_accepts_generated_ids():
    assert store.video_dir("b", "v1203454f2a").name == "v1203454f2a"


def test_debug_traversal_404(env):
    bdir = store.building_dir(env["bid"])
    # a track.json one level above the videos dir must not be reachable
    store.write_json_atomic(bdir / "track.json", {"t": [0], "heading": [0], "flow": [0]})
    (bdir / "videos").mkdir()  # exists once any video was uploaded; needed for videos/.. to resolve
    # HTTP clients normalise "/videos/../debug" away, but a raw request (curl --path-as-is) delivers
    # video_id=".." to the route, so call the endpoint directly.
    with pytest.raises(HTTPException) as exc:
        router.video_debug(env["bid"], "..")
    assert exc.value.status_code == 404
    for vid in ["%2E%2E", "..%2F.."]:
        r = env["client"].get(f"/api/buildings/{env['bid']}/videos/{vid}/debug")
        assert r.status_code == 404, (vid, r.text)


def test_debug_valid_id_still_works(env):
    vdir = store.video_dir(env["bid"], "v1200001234")
    store.write_json_atomic(vdir / "track.json", {"t": [0, 1], "heading": [0, 1], "flow": [0, 0]})
    r = env["client"].get(f"/api/buildings/{env['bid']}/videos/v1200001234/debug")
    assert r.status_code == 200
    assert r.json()["t"] == [0, 1]


def test_delete_video_bad_id_404_before_touching_map(env):
    # A video with an unsafe id smuggled in through PUT /map must not let DELETE rmtree outside videos/.
    bundle = store.load_map(env["bid"])
    bundle.videos.append(VideoInfo(id="..", filename="evil.mp4"))
    store.save_map(bundle)
    with pytest.raises(HTTPException) as exc:
        router.delete_video(env["bid"], "..")
    assert exc.value.status_code == 404
    assert store.building_dir(env["bid"]).is_dir()
    assert any(v.id == ".." for v in store.load_map(env["bid"]).videos)  # map untouched
    assert env["client"].delete(f"/api/buildings/{env['bid']}/videos/v.1").status_code == 404


def test_bfiles_serves_frames_only(env):
    bid = env["bid"]
    frame = store.frames_dir(bid) / "v1200001234_0001.jpg"
    frame.write_bytes(_jpeg())
    c = env["client"]
    r = c.get(f"/bfiles/{bid}/frames/v1200001234_0001.jpg")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.content == frame.read_bytes()
    assert store.frame_url(bid, "v1200001234_0001.jpg") == f"/bfiles/{bid}/frames/v1200001234_0001.jpg"

    for path in [f"/bfiles/{bid}/map.json", f"/bfiles/{bid}/gemma_cache.json",
                 f"/bfiles/{bid}/videos/v1200001234/source.mp4", f"/bfiles/{bid}/frames/missing.jpg",
                 f"/bfiles/{bid}/frames/..%2Fmap.json", f"/bfiles/{bid}/frames/%2E%2E",
                 f"/bfiles/BAD_ID/frames/v1200001234_0001.jpg", f"/bfiles/{bid}/frames/.hidden"]:
        assert c.get(path).status_code == 404, path
