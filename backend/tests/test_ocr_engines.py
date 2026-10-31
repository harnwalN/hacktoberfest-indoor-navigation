"""OCR engines (app/ocr.py): the engine selection matrix, the Gemma cache keys per engine (and that answers cached on
a Mac under the old key still hit), the RapidOCR adapter, the real RapidOCR engine (skipped unless the linux-ocr
extra is installed), grounding parity between the engines, and /api/health.

No Ollama and no hosted API here. Apple Vision and RapidOCR are mocked, except in the real-engine test."""
import importlib.machinery
import importlib.util
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from app import config, gemma, ocr
from app.gemma import GemmaCache
from app.main import app
from app.schemas import FrameAnalysis, GemmaFrame, SamePlace

BACKEND = Path(__file__).resolve().parent.parent
PV = gemma.PROMPT_VERSION
HAVE_RAPID = ocr._rapid_installed()
FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"  # macOS; other systems fall back to Pillow's default


def _font(size: int):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default(size=size)


def _jpg(path: Path, size=(1000, 600), color="white") -> Path:
    Image.new("RGB", size, color).save(path, format="JPEG")
    return path


@pytest.fixture
def engine(monkeypatch):
    """Sets the engine in use (OCR on), whatever this machine has."""
    def use(name: str):
        monkeypatch.setattr(ocr, "_ENGINE", name)
        monkeypatch.setattr(config, "OCR_ENABLED", True)
    return use


def _raw(boxes, txts, scores):
    """A RapidOCR result: boxes (N,4,2) in pixels, txts, scores (RapidOCROutput has these three)."""
    return SimpleNamespace(boxes=None if boxes is None else np.array(boxes, dtype=float), txts=txts, scores=scores)


# ---------------------------------------------------------------- A. engine selection
MATRIX = [
    # requested, platform, vision importable, rapidocr installed -> engine
    ("auto", "darwin", True, True, "vision"),        # the Mac default
    ("auto", "darwin", True, False, "vision"),
    ("auto", "darwin", False, True, "rapidocr"),     # macOS without pyobjc: RapidOCR if installed
    ("auto", "linux", False, True, "rapidocr"),
    ("auto", "linux", True, True, "rapidocr"),       # Vision is never on Linux; the platform decides
    ("auto", "linux", False, False, "none"),
    ("auto", "darwin", False, False, "none"),
    ("rapidocr", "darwin", True, True, "rapidocr"),  # an explicit engine wins over auto
    ("vision", "darwin", True, True, "vision"),
    ("vision", "linux", False, True, "none"),        # explicit but not here: none, not a silent other engine
    ("rapidocr", "linux", False, False, "none"),
    ("none", "darwin", True, True, "none"),
]


@pytest.mark.parametrize("requested,platform,vision_ok,rapid_ok,expected", MATRIX)
def test_engine_selection_matrix(requested, platform, vision_ok, rapid_ok, expected):
    eng, why = ocr.select_engine(requested, platform, vision_ok, rapid_ok)
    assert eng == expected
    assert why  # always says why


def test_select_engine_rejects_an_unknown_value():
    with pytest.raises(ValueError, match="OCR_ENGINE must be one of"):
        ocr.select_engine("tesseract", "linux", False, False)


def test_resolve_on_darwin_with_vision_picks_vision(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(ocr, "_AVAILABLE", True)
    monkeypatch.setattr(ocr, "_rapid_installed", lambda: True)
    monkeypatch.setattr(config, "OCR_ENGINE", "auto")
    assert ocr._resolve()[0] == "vision"


def test_resolve_on_linux_with_rapidocr_installed_picks_rapidocr(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(ocr, "_AVAILABLE", False)
    monkeypatch.setattr(ocr, "_rapid_installed", lambda: True)
    monkeypatch.setattr(config, "OCR_ENGINE", "auto")
    assert ocr._resolve()[0] == "rapidocr"


def test_resolve_on_linux_without_rapidocr_is_none_with_a_warning(monkeypatch, caplog):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(ocr, "_AVAILABLE", False)
    monkeypatch.setattr(config, "OCR_ENGINE", "auto")
    monkeypatch.setitem(sys.modules, "rapidocr", None)  # `import rapidocr` fails, as on a machine without it
    with caplog.at_level(logging.INFO, logger="app.ocr"):
        eng, why = ocr._resolve()
        ocr._announce(eng, why)
    assert eng == "none"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "OCR off" in warnings[0].getMessage() and "uv sync --extra linux-ocr" in warnings[0].getMessage()


def test_an_explicit_engine_wins_over_auto(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(ocr, "_AVAILABLE", True)
    monkeypatch.setattr(ocr, "_rapid_installed", lambda: True)
    monkeypatch.setattr(config, "OCR_ENGINE", "rapidocr")
    assert ocr._resolve()[0] == "rapidocr"
    monkeypatch.setattr(config, "OCR_ENGINE", "vision")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(ocr, "_AVAILABLE", False)
    assert ocr._resolve()[0] == "none"  # Apple Vision is not here, and RapidOCR is not asked for


def test_rapidocr_installed_check_needs_both_packages(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)  # rapidocr alone is not enough: onnxruntime does the work
    assert ocr._rapid_installed() is False
    monkeypatch.setitem(sys.modules, "rapidocr", None)
    assert ocr._rapid_installed() is False


def test_a_leftover_folder_is_not_an_installed_rapidocr(monkeypatch):
    """An uninstall leaves rapidocr/models behind (downloaded weights): a namespace folder, not the package."""
    def spec_of(name, origin):
        return importlib.machinery.ModuleSpec(name, None, origin=origin)

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: spec_of(name, None))
    assert ocr._rapid_installed() is False
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: spec_of(name, f"/site/{name}/__init__.py"))
    assert ocr._rapid_installed() is True


def test_explicit_engine_is_read_case_and_space_insensitively():
    code = "from app import config; print(config.OCR_ENGINE)"
    env = {k: v for k, v in os.environ.items() if k != "OCR_ENGINE"}
    default = subprocess.run([sys.executable, "-c", code], cwd=BACKEND, env=env, capture_output=True, text=True)
    assert default.returncode == 0 and default.stdout.strip() == "auto"
    env["OCR_ENGINE"] = " RapidOCR "
    spaced = subprocess.run([sys.executable, "-c", code], cwd=BACKEND, env=env, capture_output=True, text=True)
    assert spaced.returncode == 0 and spaced.stdout.strip() == "rapidocr"


def test_invalid_ocr_engine_fails_at_import_with_a_clear_message():
    env = {**os.environ, "OCR_ENGINE": "tesseract"}
    proc = subprocess.run([sys.executable, "-c", "import app.config"], cwd=BACKEND, env=env, capture_output=True,
                          text=True)
    assert proc.returncode != 0
    assert "OCR_ENGINE must be auto, vision, rapidocr or none (got 'tesseract')" in proc.stderr


def test_engine_and_available_follow_the_master_switch(engine, monkeypatch):
    engine("vision")
    assert ocr.engine() == "vision" and ocr.available() is True
    monkeypatch.setattr(config, "OCR_ENABLED", False)
    assert ocr.engine() == "none" and ocr.available() is False


def test_announce_logs_the_engine_or_why_none(caplog, monkeypatch):
    with caplog.at_level(logging.INFO, logger="app.ocr"):
        ocr._announce("rapidocr", "OCR_ENGINE=rapidocr")
        ocr._announce("none", "auto: nothing")
        monkeypatch.setattr(config, "OCR_ENABLED", False)
        ocr._announce("vision", "OCR_ENGINE=vision")
    text = [(r.levelno, r.getMessage()) for r in caplog.records]
    assert (logging.INFO, "OCR engine: rapidocr (OCR_ENGINE=rapidocr)") in text
    assert any(lvl == logging.WARNING and "OCR off (auto: nothing)" in msg for lvl, msg in text)
    assert any(lvl == logging.INFO and "OCR_ENABLED=0" in msg for lvl, msg in text)


def test_observations_dispatch_to_the_engine_in_use(engine, monkeypatch):
    monkeypatch.setattr(ocr, "_vision_observations", lambda src: "vision")
    monkeypatch.setattr(ocr, "_rapid_observations", lambda src: "rapidocr")
    engine("vision")
    assert ocr._observations("x") == "vision"
    engine("rapidocr")
    assert ocr._observations("x") == "rapidocr"
    engine("none")
    assert ocr._observations("x") == []


# ---------------------------------------------------------------- B. cache keys per engine
def test_vision_keeps_the_key_every_macos_answer_was_stored_under(engine, tmp_path):
    engine("vision")
    img = _jpg(tmp_path / "f.jpg")
    assert gemma._frame_key(img)("gemma3:4b") == f"analyze:{PV}:gemma3:4b:ocr:{gemma._sha(img)}"


def test_each_engine_and_ocr_off_have_their_own_frame_key(engine, tmp_path, monkeypatch):
    img = _jpg(tmp_path / "f.jpg")
    keys = {}
    for name in ("vision", "rapidocr"):
        engine(name)
        keys[name] = gemma._frame_key(img)("gemma3:4b")
    monkeypatch.setattr(config, "OCR_ENABLED", False)
    keys["plain"] = gemma._frame_key(img)("gemma3:4b")
    assert len(set(keys.values())) == 3
    assert keys["rapidocr"] == f"analyze:{PV}:gemma3:4b:ocr-rapidocr:{gemma._sha(img)}"
    assert keys["plain"] == f"analyze:{PV}:gemma3:4b:plain:{gemma._sha(img)}"


def test_same_place_keys_follow_the_engine_stairs_do_not(engine, tmp_path, monkeypatch):
    a, b = _jpg(tmp_path / "a.jpg"), _jpg(tmp_path / "b.jpg", color="black")
    verdict = SamePlace(same_place=True, confidence=0.9, reason="same door")
    monkeypatch.setattr(gemma, "_chat", lambda *args, **kwargs: verdict)
    monkeypatch.setattr(ocr, "read_text_with_crops", lambda p: [])  # no plaque conflict in either engine
    lo, hi = sorted((gemma._sha(a), gemma._sha(b)))
    keys = {}
    for name in ("vision", "rapidocr"):
        engine(name)
        cache = GemmaCache(tmp_path / f"{name}.jsonl")
        gemma.same_place(a, b, cache, "gemma3:4b")
        gemma.same_stairwell(a, b, cache, "gemma3:4b")
        keys[name] = set(cache.data)
    assert keys["vision"] == {f"same:{PV}:gemma3:4b:{lo}:{hi}", f"stairs:{PV}:gemma3:4b:{lo}:{hi}"}
    assert keys["rapidocr"] == {f"same:{PV}:gemma3:4b:{lo}:{hi}:ocr-rapidocr", f"stairs:{PV}:gemma3:4b:{lo}:{hi}"}


def test_answers_cached_before_this_change_still_hit_and_get_their_lines_refreshed(engine, tmp_path, monkeypatch):
    engine("vision")
    img = _jpg(tmp_path / "f.jpg")
    cache = GemmaCache(tmp_path / "c.jsonl")
    old_key = f"analyze:{PV}:gemma3:4b:ocr:{gemma._sha(img)}"  # the key a Mac stored before this change
    cache.put(old_key, FrameAnalysis(texts=["ROOM 214"], room_numbers=["214"], has_sign=True, ocr=["ROOM 214"],
                                     ocr_rev=2, grounded=True).model_dump())
    monkeypatch.setattr(ocr, "read_text_with_crops", lambda p: ["ROOM 214"])
    hit = gemma.cached_analysis(img, cache, "gemma3:4b")
    assert hit is not None and hit.room_numbers == ["214"]
    assert hit.ocr_rev == ocr.REVISION
    assert cache.get(old_key)["ocr_rev"] == ocr.REVISION  # re-read once (OCR only), kept under the same key
    engine("rapidocr")  # the same photo under another engine is a miss: its answer is grounded on other lines
    assert gemma.cached_analysis(img, cache, "gemma3:4b") is None


def test_ocr_lines_memo_is_per_engine(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(ocr, "read_text_with_crops", lambda p: calls.append(str(p)) or ["EXIT"])
    gemma._ocr_lines.cache_clear()
    img = _jpg(tmp_path / "f.jpg")
    st = img.stat()
    try:
        gemma._ocr_lines(str(img), st.st_mtime_ns, st.st_size, "vision")
        gemma._ocr_lines(str(img), st.st_mtime_ns, st.st_size, "vision")
        gemma._ocr_lines(str(img), st.st_mtime_ns, st.st_size, "rapidocr")
        assert len(calls) == 2  # one read per engine, the second vision call is memoized
    finally:
        gemma._ocr_lines.cache_clear()


# ---------------------------------------------------------------- C. RapidOCR adapter
def test_rapid_lines_have_the_same_shape_as_vision_lines(monkeypatch):
    monkeypatch.setattr(config, "OCR_MIN_CONFIDENCE_RAPIDOCR", 0.7)
    raw = _raw([[[100, 50], [300, 50], [300, 90], [100, 90]],     # ROOM 214
                [[0, 300], [250, 300], [250, 340], [0, 340]],      # low confidence: dropped
                [[500, 10], [700, 10], [700, 40], [500, 40]]],     # blank text: dropped
               ("ROOM 214", "EXIT", "  "), (0.99, 0.4, 0.99))
    lines = ocr._rapid_lines(raw, 1000)
    assert lines == [("ROOM 214", pytest.approx(0.1), pytest.approx(0.3))]
    text, x0, x1 = lines[0]
    assert isinstance(text, str) and isinstance(x0, float) and isinstance(x1, float)


def test_rapid_lines_clamp_x_to_the_image(monkeypatch):
    monkeypatch.setattr(config, "OCR_MIN_CONFIDENCE_RAPIDOCR", 0.7)
    raw = _raw([[[900, 50], [1100, 50], [1100, 90], [900, 90]], [[-20, 50], [40, 50], [40, 90], [-20, 90]]],
               ("ROOM 216", "214"), (0.95, 0.95))
    assert ocr._rapid_lines(raw, 1000) == [("ROOM 216", pytest.approx(0.9), 1.0), ("214", 0.0, pytest.approx(0.04))]


def test_rapid_lines_keep_the_per_engine_threshold(monkeypatch):
    raw = _raw([[[100, 50], [300, 50], [300, 90], [100, 90]]], ("ROOM 214",), (0.8,))
    monkeypatch.setattr(config, "OCR_MIN_CONFIDENCE", 0.1)  # Apple Vision's threshold does not apply to RapidOCR
    monkeypatch.setattr(config, "OCR_MIN_CONFIDENCE_RAPIDOCR", 0.9)
    assert ocr._rapid_lines(raw, 1000) == []
    monkeypatch.setattr(config, "OCR_MIN_CONFIDENCE_RAPIDOCR", 0.7)
    assert len(ocr._rapid_lines(raw, 1000)) == 1


def test_empty_results_and_images_are_handled(monkeypatch):
    assert ocr._rapid_lines(_raw(None, None, None), 1000) == []           # nothing detected
    assert ocr._rapid_lines(_raw(np.zeros((0, 4, 2)), (), ()), 1000) == []
    assert ocr._rapid_lines(_raw([[[1, 1], [2, 1], [2, 2], [1, 2]]], ("A",), (0.99,)), 0) == []  # zero width
    calls = []
    monkeypatch.setattr(ocr, "_rapid", lambda img: calls.append(img.size) or _raw(None, None, None))
    assert ocr._rapid_observations(Image.new("RGB", (0, 0))) == []  # empty image: the engine is not even asked
    assert ocr._rapid_observations(Image.new("RGB", (300, 200), "white")) == []  # no text on the image
    assert calls == [(300, 200)]


def test_rapid_observations_read_a_file_or_an_image_in_upright_rgb(monkeypatch, tmp_path):
    seen = []

    def fake(img):
        seen.append(img.size)
        assert img.mode == "RGB"
        return _raw([[[100, 50], [300, 50], [300, 90], [100, 90]]], ("ROOM 214",), (0.97,))

    monkeypatch.setattr(ocr, "_rapid", fake)
    expect = [("ROOM 214", pytest.approx(0.1), pytest.approx(0.3))]
    assert ocr._rapid_observations(Image.new("RGB", (1000, 600), "white")) == expect
    assert ocr._rapid_observations(_jpg(tmp_path / "f.jpg", size=(1000, 600))) == expect
    assert seen == [(1000, 600), (1000, 600)]


def test_the_engine_loads_once_and_its_calls_do_not_overlap(monkeypatch):
    loads, active, peak = [], [0], [0]
    guard = threading.Lock()

    def load():
        loads.append(1)
        time.sleep(0.05)  # widen the race: every thread arrives while the first one is still loading

        def call(img):
            with guard:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.01)
            with guard:
                active[0] -= 1
            return _raw(None, None, None)
        return call

    monkeypatch.setattr(ocr, "_rapid", None)
    monkeypatch.setattr(ocr, "_load_rapid", load)
    img = Image.new("RGB", (200, 100), "white")
    threads = [threading.Thread(target=ocr._rapid_observations, args=(img,)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(loads) == 1
    assert peak[0] == 1


def test_missing_rapidocr_raises_ocr_unavailable_and_the_next_image_tries_again(monkeypatch):
    monkeypatch.setattr(ocr, "_rapid", None)
    monkeypatch.setitem(sys.modules, "rapidocr", None)  # not installed
    with pytest.raises(ocr.OCRUnavailable, match="uv sync --extra linux-ocr"):
        ocr._rapid_observations(Image.new("RGB", (200, 100), "white"))
    assert ocr._rapid is None  # nothing half-built is kept


# ---------------------------------------------------------------- D. the real RapidOCR engine
def _plaque(path: Path) -> Path:
    img = Image.new("RGB", (900, 500), "white")
    d = ImageDraw.Draw(img)
    d.text((60, 60), "ROOM 214", font=_font(120), fill="black")
    d.text((60, 300), "EXIT", font=_font(90), fill="black")
    img.save(path)
    return path


@pytest.mark.skipif(not HAVE_RAPID, reason="RapidOCR not installed (uv sync --extra linux-ocr)")
def test_real_rapidocr_reads_a_rendered_room_plaque_and_exit_sign(engine, monkeypatch, tmp_path):
    monkeypatch.setattr(ocr, "_rapid", None)  # load the real models in this test
    engine("rapidocr")
    path = _plaque(tmp_path / "plaque.png")
    lines = ocr.read_text_with_crops(path)
    assert ocr.grounded("214", lines), lines
    assert ocr.grounded("ROOM 214", lines), lines
    assert ocr.grounded("EXIT", lines), lines
    assert ocr.grounded("215", lines) is False  # a number that is not on the sign is not read
    blank = _jpg(tmp_path / "blank.jpg")
    assert ocr.read_text_with_crops(blank) == []


# ---------------------------------------------------------------- E. grounding parity
PLATE = [("ROOM 214", 0.10, 0.30, 0.10), ("EXIT", 0.10, 0.25, 0.50)]  # text, x0, x1, y (normalized)


def _model_answer():
    """Gemma reads a room 214 that OCR shows, and a room 999 that OCR never read."""
    return GemmaFrame(texts=["ROOM 214", "ROOM 999"], room_numbers=["214", "999"], has_sign=True,
                      suggested_label="Room 214", place_type="room", description="door")


def _analyze(engine_name, monkeypatch, img):
    monkeypatch.setattr(config, "OCR_ENABLED", True)
    monkeypatch.setattr(ocr, "_ENGINE", engine_name)
    monkeypatch.setattr(gemma, "_chat", lambda model, prompt, images, schema_cls, num_predict: _model_answer())
    calls = []
    if engine_name == "vision":
        monkeypatch.setattr(ocr, "_vision_observations", lambda src: [(t, x0, x1) for t, x0, x1, _ in PLATE])
    else:
        def fake(image):  # the RapidOCR call, with the same lines at the same places in whatever size it gets
            calls.append(image.size)
            w = image.size[0]
            boxes = [[[x0 * w, y * 600], [x1 * w, y * 600], [x1 * w, y * 600 + 40], [x0 * w, y * 600 + 40]]
                     for _, x0, x1, y in PLATE]
            return _raw(boxes, tuple(t for t, *_ in PLATE), tuple(0.95 for _ in PLATE))
        monkeypatch.setattr(ocr, "_rapid", fake)
    result = gemma._compute_frame(img, "gemma3:4b")
    return result, calls


def test_unread_room_numbers_are_stripped_the_same_way_for_rapidocr_as_for_vision(monkeypatch, tmp_path):
    img = _jpg(tmp_path / "f.jpg", size=(1000, 600))
    vision, _ = _analyze("vision", monkeypatch, img)
    rapid, calls = _analyze("rapidocr", monkeypatch, img)
    assert calls, "the RapidOCR path was not used"
    assert vision.ocr == rapid.ocr == ["ROOM 214", "EXIT"]
    assert vision.room_numbers == rapid.room_numbers == ["214"]
    assert "999" not in rapid.room_numbers and "ROOM 999" not in rapid.texts
    assert vision.texts == rapid.texts == ["ROOM 214"]
    assert vision.grounded and rapid.grounded


# ---------------------------------------------------------------- F. /api/health
def test_health_reports_the_ocr_engine_and_keeps_the_other_keys(monkeypatch):
    monkeypatch.setattr(gemma, "health", lambda: {"reachable": False})  # no Ollama in tests
    monkeypatch.setattr(ocr, "_ENGINE", "rapidocr")
    monkeypatch.setattr(config, "OCR_ENABLED", True)
    body = TestClient(app).get("/api/health").json()
    assert body["ocr"] == {"engine": "rapidocr", "available": True}
    assert {"ffmpeg", "gemma", "hosted", "vision_provider"} <= set(body)
    monkeypatch.setattr(config, "OCR_ENABLED", False)
    assert TestClient(app).get("/api/health").json()["ocr"] == {"engine": "none", "available": False}
