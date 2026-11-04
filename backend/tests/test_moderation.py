"""Automatic moderation of walk keyframes (app/moderation.py): face blur and the content check.

Covers: the bundled YuNet model (pinned SHA-256, loads and runs), blur on detected boxes (pixelated, input untouched,
fails closed when the model or detector is broken), the content verdict (approve only a plausible indoor photo with
no inappropriate content), fail-safe rejection on every model, API and image error, the model choice (config
through gemma.choose_model, or an explicit name), the blur-before-classify order, and the hosted path through the
real gemma._chat_hosted with hosted.generate mocked. No network and no Ollama: the Gemini transport and the Ollama
client are mocked or forbidden."""
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from app import config, gemma, hosted, moderation
from app.moderation import ContentCheck, FaceBlurError, ModerationVerdict

pytestmark = pytest.mark.real_moderation  # these tests exercise the real functions, not the conftest stub

MODEL_HOSTED = "gemini-2.5-flash-lite"
MODEL_LOCAL = "gemma3:4b"
INDOOR = ContentCheck(description="corridor with doors", indoor_building=True, inappropriate=False,
                      reason="a hallway with room doors")


@pytest.fixture(autouse=True)
def _no_ollama(monkeypatch):
    """Nothing in this file contacts Ollama. Tests that need a model answer mock gemma._chat."""
    def forbidden(*args, **kwargs):
        raise AssertionError("unmocked Ollama client call")

    monkeypatch.setattr(gemma, "_client", forbidden)


@pytest.fixture
def frame():
    """A 360x640 BGR frame: a grey wall with a high-contrast checkerboard patch at (100..180, 100..180)."""
    img = np.full((360, 640, 3), 128, dtype=np.uint8)
    yy, xx = np.mgrid[100:180, 100:180]
    img[100:180, 100:180] = np.where(((yy // 4) + (xx // 4)) % 2 == 0, 255, 0)[..., None]
    return img


class FakeChat:
    """Stands in for gemma._chat. `answer` is returned, or raised when it is an exception."""

    def __init__(self, answer=INDOOR):
        self.answer = answer
        self.calls: list[dict] = []

    def __call__(self, model, prompt, images, schema_cls, num_predict):
        self.calls.append({"model": model, "prompt": prompt, "images": images, "schema": schema_cls,
                           "num_predict": num_predict})
        if isinstance(self.answer, BaseException):
            raise self.answer
        return self.answer


def _use_chat(monkeypatch, answer=INDOOR) -> FakeChat:
    fake = FakeChat(answer)
    monkeypatch.setattr(gemma, "_chat", fake)
    monkeypatch.setattr(gemma, "choose_model", lambda: MODEL_HOSTED)
    return fake


# ---------------------------------------------------------------- face blur: model and detector
def test_bundled_model_matches_pin():
    path = Path(moderation.FACE_MODEL_PATH)
    assert path.is_file(), f"bundled face model missing: {path}"
    assert path.stat().st_size == moderation.FACE_MODEL_BYTES
    assert moderation._model_ok(path)


def test_real_detector_runs_and_leaves_input_alone(frame):
    before = frame.copy()
    out = moderation.blur_faces(frame)
    assert out.shape == frame.shape and out.dtype == np.uint8
    assert out is not frame
    assert np.array_equal(frame, before)  # the caller's frame is never changed in place


def test_blur_pixelates_the_box_and_keeps_the_rest(monkeypatch, frame):
    monkeypatch.setattr(moderation, "detect_faces", lambda img: [(100.0, 100.0, 60.0, 60.0)])
    out = moderation.blur_faces(frame)
    # The 1.3x box (about 78 px) around (130, 130): the checkerboard (std about 127) is gone inside it.
    box = out[110:150, 110:150].astype(float)
    assert box.std() < 20
    # Outside the box the pixels are untouched.
    assert np.array_equal(out[:50], frame[:50])
    assert np.array_equal(out[:, 300:], frame[:, 300:])


def test_no_faces_gives_identical_pixels(monkeypatch, frame):
    monkeypatch.setattr(moderation, "detect_faces", lambda img: [])
    out = moderation.blur_faces(frame)
    assert np.array_equal(out, frame)


def test_box_is_clamped_to_the_frame():
    x0, y0, x1, y1 = moderation._blur_box(-10.0, -5.0, 60.0, 60.0, 640, 360)
    assert (x0, y0) == (0, 0)
    x0, y0, x1, y1 = moderation._blur_box(600.0, 330.0, 60.0, 60.0, 640, 360)
    assert x1 == 640 and y1 == 360


def test_missing_model_fails_closed(monkeypatch, tmp_path, frame):
    monkeypatch.setattr(moderation, "FACE_MODEL_PATH", tmp_path / "absent.onnx")
    monkeypatch.setattr(moderation, "_detector", None)
    with pytest.raises(FaceBlurError, match="missing"):
        moderation.blur_faces(frame)


def test_wrong_model_bytes_fail_closed(monkeypatch, tmp_path, frame):
    good = Path(moderation.FACE_MODEL_PATH).read_bytes()
    wrong_size = tmp_path / "short.onnx"
    wrong_size.write_bytes(good + b"x")
    flipped = tmp_path / "flipped.onnx"  # same size, one byte changed: the SHA-256 pin must catch it
    flipped.write_bytes(good[:-1] + bytes([good[-1] ^ 0x01]))
    for bad in (wrong_size, flipped):
        monkeypatch.setattr(moderation, "_detector", None)
        monkeypatch.setattr(moderation, "FACE_MODEL_PATH", bad)
        with pytest.raises(FaceBlurError, match="pinned SHA-256"):
            moderation.blur_faces(frame)


def test_detector_error_fails_closed(monkeypatch, frame):
    class BrokenDetector:
        def setInputSize(self, size):
            pass

        def detect(self, img):
            raise cv2.error("boom")

    monkeypatch.setattr(moderation, "_get_detector", lambda: BrokenDetector())
    with pytest.raises(FaceBlurError, match="face detection failed"):
        moderation.blur_faces(frame)


@pytest.mark.parametrize("bad", [
    [[0] * 3] * 3,  # a list, not an array
    np.zeros((10, 10, 3), dtype=np.float32),  # wrong dtype
    np.zeros((10, 10), dtype=np.uint8),  # no channel axis
    np.zeros((10, 10, 4), dtype=np.uint8),  # four channels
    np.zeros((0, 10, 3), dtype=np.uint8),  # empty
])
def test_bad_frame_is_rejected_not_passed_through(bad):
    with pytest.raises(ValueError):
        moderation.blur_faces(bad)


def test_moderate_keyframe_blurs_before_classifying(monkeypatch, frame):
    blurred = np.zeros_like(frame)
    seen = []
    monkeypatch.setattr(moderation, "blur_faces", lambda img: (seen.append("blur"), blurred)[1])

    def classify(img, model=None):
        seen.append(("classify", img is blurred, model))
        return ModerationVerdict(approved=True, reason="ok", model=MODEL_HOSTED)

    monkeypatch.setattr(moderation, "classify_content", classify)
    out, verdict = moderation.moderate_keyframe(frame, model=MODEL_HOSTED)
    assert seen == ["blur", ("classify", True, MODEL_HOSTED)]
    assert out is blurred and verdict.approved


def test_moderate_keyframe_blur_failure_skips_the_model(monkeypatch, frame):
    def broken(img):
        raise FaceBlurError("no model")

    monkeypatch.setattr(moderation, "blur_faces", broken)
    monkeypatch.setattr(moderation, "classify_content", lambda *a, **k: pytest.fail("classified unblurred frame"))
    with pytest.raises(FaceBlurError):
        moderation.moderate_keyframe(frame)


# ---------------------------------------------------------------- content check: verdicts
def test_plausible_indoor_photo_is_approved(monkeypatch, frame):
    fake = _use_chat(monkeypatch)
    verdict = moderation.classify_content(frame)
    assert verdict == ModerationVerdict(approved=True, reason="a hallway with room doors", model=MODEL_HOSTED)
    assert verdict.error is False and verdict.notes == ""
    assert len(fake.calls) == 1


@pytest.mark.parametrize("indoor, inappropriate, approved", [
    (True, False, True),
    (True, True, False),
    (False, False, False),
    (False, True, False),
])
def test_only_indoor_and_appropriate_is_approved(monkeypatch, frame, indoor, inappropriate, approved):
    _use_chat(monkeypatch, ContentCheck(description="x", indoor_building=indoor, inappropriate=inappropriate,
                                        reason="because"))
    verdict = moderation.classify_content(frame)
    assert verdict.approved is approved
    assert verdict.error is False  # a judgement, not a failure
    if not approved:
        expected = "inappropriate content" if inappropriate else "not a plausible indoor building photo"
        assert verdict.reason.startswith(expected)


def test_rejection_reason_falls_back_to_description(monkeypatch, frame):
    _use_chat(monkeypatch, ContentCheck(description="a cat on a sofa", indoor_building=False, inappropriate=False,
                                        reason="  "))
    verdict = moderation.classify_content(frame)
    assert not verdict.approved and "a cat on a sofa" in verdict.reason


def test_default_model_comes_from_config(monkeypatch, frame):
    """No model argument: the model is whatever config selects (VISION_PROVIDER / HOSTED_MODEL), not a constant here."""
    monkeypatch.setattr(config, "VISION_PROVIDER", "hosted")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setattr(config, "HOSTED_MODEL", "gemini-test-model-x")
    fake = FakeChat()
    monkeypatch.setattr(gemma, "_chat", fake)
    verdict = moderation.classify_content(frame)
    assert fake.calls[0]["model"] == "gemini-test-model-x"
    assert verdict.model == "gemini-test-model-x" and verdict.approved


def test_explicit_model_wins_and_skips_the_choice(monkeypatch, frame):
    fake = FakeChat()
    monkeypatch.setattr(gemma, "_chat", fake)
    monkeypatch.setattr(gemma, "choose_model", lambda: pytest.fail("choose_model called with a model given"))
    verdict = moderation.classify_content(frame, model=MODEL_LOCAL)
    assert fake.calls[0]["model"] == MODEL_LOCAL and verdict.model == MODEL_LOCAL


def test_call_uses_the_content_prompt_schema_and_token_cap(monkeypatch, frame):
    fake = _use_chat(monkeypatch)
    moderation.classify_content(frame)
    call = fake.calls[0]
    assert call["schema"] is ContentCheck
    assert call["num_predict"] == moderation.MODERATION_NUM_PREDICT
    assert call["prompt"] == moderation.CONTENT_PROMPT
    assert len(call["images"]) == 1


# ---------------------------------------------------------------- content check: fail safe
def test_no_model_available_rejects_with_notes(monkeypatch, frame):
    fake = FakeChat()
    monkeypatch.setattr(gemma, "_chat", fake)

    def no_model():
        raise gemma.GemmaUnavailable("No Gemma model pulled. Run: ollama pull gemma3:4b")

    monkeypatch.setattr(gemma, "choose_model", no_model)
    verdict = moderation.classify_content(frame)
    assert verdict.approved is False and verdict.error is True
    assert "No Gemma model pulled" in verdict.notes
    assert verdict.model == ""
    assert fake.calls == []


def test_model_error_rejects_with_notes(monkeypatch, frame):
    fake = FakeChat(gemma.GemmaUnavailable("Model 'gemma3:4b' not found in Ollama"))
    monkeypatch.setattr(gemma, "_chat", fake)
    verdict = moderation.classify_content(frame, model=MODEL_LOCAL)
    assert verdict.approved is False and verdict.error is True
    assert verdict.reason == "content check failed (rejected)"
    assert "not found in Ollama" in verdict.notes
    assert verdict.model == MODEL_LOCAL


def test_unexpected_exception_rejects(monkeypatch, frame):
    _use_chat(monkeypatch, RuntimeError("socket exploded"))
    verdict = moderation.classify_content(frame)
    assert verdict.approved is False and verdict.error is True
    assert "RuntimeError" in verdict.notes and "socket exploded" in verdict.notes


def test_notes_are_capped(monkeypatch, frame):
    _use_chat(monkeypatch, RuntimeError("x" * 5000))
    verdict = moderation.classify_content(frame)
    assert len(verdict.notes) <= moderation.NOTES_MAX


@pytest.mark.parametrize("bad_input", [
    np.zeros((20, 20, 3), dtype=np.float32),
    np.zeros((20, 20), dtype=np.uint8),
    None,
])
def test_bad_array_rejects_without_calling_the_model(monkeypatch, bad_input):
    fake = _use_chat(monkeypatch)
    verdict = moderation.classify_content(bad_input)
    assert verdict.approved is False and verdict.error is True
    assert "could not read the image" in verdict.notes
    assert fake.calls == []


def test_missing_file_rejects_without_calling_the_model(monkeypatch, tmp_path):
    fake = _use_chat(monkeypatch)
    verdict = moderation.classify_content(tmp_path / "nope.jpg")
    assert verdict.approved is False and verdict.error is True
    assert fake.calls == []


# ---------------------------------------------------------------- image handed to the model
def test_array_is_sent_as_upright_rgb_jpeg(monkeypatch):
    fake = _use_chat(monkeypatch)
    blue_bgr = np.zeros((300, 800, 3), dtype=np.uint8)
    blue_bgr[..., 0] = 255  # BGR: pure blue
    moderation.classify_content(blue_bgr)
    jpeg = fake.calls[0]["images"][0]
    assert jpeg[:3] == b"\xff\xd8\xff"
    img = Image.open(io.BytesIO(jpeg))
    assert img.format == "JPEG"
    r, g, b = img.convert("RGB").getpixel((400, 150))
    assert b > 200 and r < 60 and g < 60  # blue, so the BGR -> RGB conversion is right
    assert max(img.size) <= config.GEMMA_IMAGE_MAX


def test_path_input_is_read_and_approved(monkeypatch, tmp_path):
    path = tmp_path / "keyframe.png"
    Image.new("RGB", (64, 48), (200, 200, 200)).save(path)
    fake = _use_chat(monkeypatch)
    verdict = moderation.classify_content(path)
    assert verdict.approved is True
    assert fake.calls[0]["images"][0][:3] == b"\xff\xd8\xff"


# ---------------------------------------------------------------- real hosted plumbing, transport mocked
def _hosted_reply(obj: dict) -> hosted.Reply:
    return hosted.Reply(text=json.dumps(obj), finish="STOP", input_tokens=10)


def test_hosted_answer_goes_through_gemma_chat_hosted(monkeypatch, frame):
    calls = []

    def generate(model, prompt, images, schema, num_predict):
        calls.append((model, num_predict, schema))
        return _hosted_reply({"description": "stairwell", "indoor_building": True, "inappropriate": False,
                              "reason": "stairs with a handrail"})

    monkeypatch.setattr(hosted, "generate", generate)
    verdict = moderation.classify_content(frame, model=MODEL_HOSTED)
    assert verdict.approved and verdict.model == MODEL_HOSTED
    assert calls[0][0] == MODEL_HOSTED and calls[0][1] == moderation.MODERATION_NUM_PREDICT
    assert calls[0][2]["type"] == "object"  # the Gemini-form schema of ContentCheck


def test_hosted_http_error_rejects_with_notes(monkeypatch, frame):
    def generate(model, prompt, images, schema, num_predict):
        raise hosted.HostedError("Gemini rejected the request (400): bad image")

    monkeypatch.setattr(hosted, "generate", generate)
    verdict = moderation.classify_content(frame, model=MODEL_HOSTED)
    assert verdict.approved is False and verdict.error is True
    assert "400" in verdict.notes


def test_hosted_unreadable_answer_twice_rejects(monkeypatch, frame):
    monkeypatch.setattr(hosted, "sleep", lambda seconds: None)
    monkeypatch.setattr(hosted, "generate", lambda *a, **k: hosted.Reply(text="not json at all", finish="STOP"))
    verdict = moderation.classify_content(frame, model=MODEL_HOSTED)
    assert verdict.approved is False and verdict.error is True
    assert "failed after a retry" in verdict.notes


def test_hosted_key_missing_rejects(monkeypatch, frame):
    monkeypatch.setattr(config, "VISION_PROVIDER", "hosted")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    monkeypatch.setattr(hosted, "generate", lambda *a, **k: pytest.fail("hosted call without a key"))
    verdict = moderation.classify_content(frame)  # default model: choose_model needs the key
    assert verdict.approved is False and verdict.error is True
    assert "GEMINI_API_KEY is not set" in verdict.notes
