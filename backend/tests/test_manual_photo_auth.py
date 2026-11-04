"""Manual photo add (POST /api/buildings/{id}/frames/manual) is owner only and goes through the same
moderation as a walk's keyframes before anything is stored.

Covers: no or bad session -> 401 with nothing written; a contributor session -> 403; a photo that passes is stored as the blurred JPEG and the new
place records contributor_id; a photo the content check rejects -> 422 with the reason and nothing stored; a check
that cannot run (face model missing, vision model error) -> 503 and nothing stored; the real blur + content check chain
with gemma._chat mocked. No network and no Ollama.
"""
import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import auth, config, gemma, moderation, store
from app.main import app
from app.moderation import ModerationVerdict
from tests.auth_helpers import owner_headers, signed_in_headers

client = TestClient(app)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BUILDINGS_DIR", tmp_path / "buildings")

    def no_gemma(*_a, **_k):
        raise gemma.GemmaUnavailable("disabled in tests")

    monkeypatch.setattr(gemma, "resolve_model", no_gemma)
    bid = store.create_building("Photo Hall").building_id
    return {"bid": bid, "headers": owner_headers()}


def _jpeg(color=(200, 30, 30), size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


def _post(bid, data=None, headers=None):
    return client.post(f"/api/buildings/{bid}/frames/manual", files={"image": ("sign.jpg", data or _jpeg(), "image/jpeg")},
                       data={"label": "Sign"}, headers=headers or {})


def _photo_files(bid):
    return sorted(p.name for p in store.frames_dir(bid).glob("manual_*"))


def _rejecting(monkeypatch, verdict):
    """Replace the moderation stub with one that returns this verdict (blur still runs as a plain copy)."""
    monkeypatch.setattr(moderation, "moderate_keyframe", lambda frame, model=None: (frame.copy(), verdict))


# ---------- 1. owner session required ----------
def test_manual_photo_without_session_is_401_and_nothing_stored(env):
    before = store.load_map(env["bid"]).model_dump()
    r = _post(env["bid"])
    assert r.status_code == 401
    assert r.json()["detail"] == "Owner sign-in required"
    assert _photo_files(env["bid"]) == []
    assert store.load_map(env["bid"]).model_dump() == before


def test_manual_photo_by_contributor_is_403_and_nothing_stored(env):
    before = store.load_map(env["bid"]).model_dump()
    r = _post(env["bid"], headers=signed_in_headers())
    assert r.status_code == 403
    assert _photo_files(env["bid"]) == []
    assert store.load_map(env["bid"]).model_dump() == before


def test_manual_photo_with_unknown_or_expired_session_is_401(env, monkeypatch):
    assert _post(env["bid"], headers={"Authorization": "Bearer not-a-real-token"}).status_code == 401
    token = auth.create_session(store.create_account_if_missing("old@example.com").id)
    monkeypatch.setattr(auth, "_now", lambda: 10**12)  # far past the session's expiry
    assert _post(env["bid"], headers={"Authorization": f"Bearer {token}"}).status_code == 401
    assert _photo_files(env["bid"]) == []


# ---------- 2. session + passing moderation ----------
def test_manual_photo_with_session_is_stored_with_contributor_id(env):
    account = store.get_account("owner@example.com")  # the owner account made by owner_headers()
    assert account is not None
    r = _post(env["bid"], headers=env["headers"])
    assert r.status_code == 200, r.text
    node = r.json()["node"]
    assert node["contributor_id"] == account.id
    assert node["frame_id"] == "manual_0000.jpg"
    assert _photo_files(env["bid"]) == ["manual_0000.jpg"]
    saved = next(n for n in store.load_map(env["bid"]).nodes if n.id == node["id"])
    assert saved.contributor_id == account.id  # on disk, not just in the response


def test_stored_photo_is_the_moderated_copy_not_the_upload(env, monkeypatch):
    """The frame on disk is whatever moderation returned (here: solid blue), never the uploaded red bytes."""
    blue = np.full((48, 64, 3), (255, 0, 0), dtype=np.uint8)  # BGR blue
    seen = []

    def blur_then_pass(frame, model=None):
        seen.append(frame.copy())
        return blue, ModerationVerdict(approved=True, reason="ok", model="test-model")

    monkeypatch.setattr(moderation, "moderate_keyframe", blur_then_pass)
    r = _post(env["bid"], headers=env["headers"])
    assert r.status_code == 200, r.text
    assert seen and np.abs(seen[0][0, 0].astype(int) - [30, 30, 200]).max() <= 10  # decoded upload (BGR red, JPEG)
    stored = cv2.imread(str(store.frames_dir(env["bid"]) / "manual_0000.jpg"))
    assert stored is not None
    assert abs(int(stored[:, :, 0].mean()) - 255) <= 8 and int(stored[:, :, 2].mean()) <= 8  # blue JPEG, no red


# ---------- 3. session + failing moderation ----------
@pytest.mark.parametrize("verdict, phrase", [
    (ModerationVerdict(approved=False, reason="inappropriate content: graphic violence", model="m"),
     "inappropriate content: graphic violence"),
    (ModerationVerdict(approved=False, reason="not a plausible indoor building photo: a selfie", model="m"),
     "not a plausible indoor building photo: a selfie"),
])
def test_rejected_by_content_check_is_422_with_reason_and_not_stored(env, monkeypatch, verdict, phrase):
    _rejecting(monkeypatch, verdict)
    before = store.load_map(env["bid"]).model_dump()
    r = _post(env["bid"], headers=env["headers"])
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert detail.startswith("Photo rejected by moderation") and phrase in detail and "not added" in detail
    assert _photo_files(env["bid"]) == []
    assert store.load_map(env["bid"]).model_dump() == before  # no place added


def test_check_that_cannot_run_is_503_and_not_stored(env, monkeypatch):
    """The vision model failed (verdict.error): fail closed, the photo is not added, and the caller is told to retry."""
    _rejecting(monkeypatch, ModerationVerdict(approved=False, reason="content check failed (rejected)",
                                              notes="vision model call failed: boom", error=True))
    before = store.load_map(env["bid"]).model_dump()
    r = _post(env["bid"], headers=env["headers"])
    assert r.status_code == 503, r.text
    assert "could not be checked" in r.json()["detail"] and "not added" in r.json()["detail"]
    assert "boom" not in r.text  # model internals are not echoed to the caller
    assert _photo_files(env["bid"]) == []
    assert store.load_map(env["bid"]).model_dump() == before


def test_face_blur_failure_is_503_and_analysis_never_runs(env, monkeypatch):
    def blur_broken(frame, model=None):
        raise moderation.FaceBlurError("face model missing")

    calls = []
    monkeypatch.setattr(moderation, "moderate_keyframe", blur_broken)
    monkeypatch.setattr(gemma, "analyze_frame", lambda *a, **k: calls.append(a))
    r = _post(env["bid"], headers=env["headers"])
    assert r.status_code == 503, r.text
    assert _photo_files(env["bid"]) == []
    assert calls == []  # Gemma never sees a photo that was not checked


# ---------- 4. the real blur and content check, with the vision model mocked ----------
@pytest.mark.real_moderation
def test_real_check_rejects_non_indoor_photo(env, monkeypatch):
    monkeypatch.setattr(gemma, "choose_model", lambda: "gemma3:4b")
    answer = moderation.ContentCheck(description="a dog in a park", indoor_building=False, inappropriate=False,
                                     reason="outdoor scene with a dog")
    monkeypatch.setattr(gemma, "_chat", lambda *a, **k: answer)
    r = _post(env["bid"], headers=env["headers"])
    assert r.status_code == 422, r.text
    assert "outdoor scene with a dog" in r.json()["detail"]
    assert _photo_files(env["bid"]) == []


@pytest.mark.real_moderation
def test_real_check_accepts_indoor_photo_and_stores_it(env, monkeypatch):
    monkeypatch.setattr(gemma, "choose_model", lambda: "gemma3:4b")
    answer = moderation.ContentCheck(description="corridor", indoor_building=True, inappropriate=False,
                                     reason="a hallway")
    monkeypatch.setattr(gemma, "_chat", lambda *a, **k: answer)
    r = _post(env["bid"], headers=env["headers"])
    assert r.status_code == 200, r.text
    assert _photo_files(env["bid"]) == ["manual_0000.jpg"]


@pytest.mark.real_moderation
def test_real_check_model_error_is_503_and_not_stored(env, monkeypatch):
    monkeypatch.setattr(gemma, "choose_model", lambda: "gemma3:4b")

    def down(*a, **k):
        raise gemma.GemmaUnavailable("ollama is not running")

    monkeypatch.setattr(gemma, "_chat", down)
    r = _post(env["bid"], headers=env["headers"])
    assert r.status_code == 503, r.text
    assert _photo_files(env["bid"]) == []


@pytest.mark.real_moderation
def test_real_face_model_missing_is_503_and_not_stored(env, monkeypatch, tmp_path):
    monkeypatch.setattr(moderation, "FACE_MODEL_PATH", tmp_path / "absent.onnx")
    monkeypatch.setattr(moderation, "_detector", None)
    r = _post(env["bid"], headers=env["headers"])
    assert r.status_code == 503, r.text
    assert _photo_files(env["bid"]) == []


def test_rejected_photo_leaves_nothing_behind_so_the_same_user_can_try_again(env, monkeypatch):
    """A 422 stores nothing, so the contributor can pick a photo again and it goes through like a first attempt."""
    _rejecting(monkeypatch, ModerationVerdict(approved=False, reason="inappropriate content: graphic violence", model="m"))
    assert _post(env["bid"], headers=env["headers"]).status_code == 422
    assert _photo_files(env["bid"]) == []
    _rejecting(monkeypatch, ModerationVerdict(approved=True, reason="a hallway with a sign", model="m"))
    r = _post(env["bid"], headers=env["headers"])
    assert r.status_code == 200, r.text
    assert r.json()["node"]["frame_id"] == "manual_0000.jpg"  # numbering restarts: the rejected try left no file
    assert _photo_files(env["bid"]) == ["manual_0000.jpg"]
