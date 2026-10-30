"""Gemma client robustness: exact-token OCR grounding, Ollama call options/retries, and _analyze failure
semantics (per-frame errors, circuit breaker). Ollama is faked; no server is needed."""
import json
from types import SimpleNamespace

import httpx
import ollama
import pytest
from PIL import Image

from app import config, gemma, jobs, ocr, store
from app.schemas import FrameAnalysis, Keyframe

GOOD_FRAME = json.dumps({"has_sign": False, "texts": [], "room_numbers": [], "suggested_label": "Hallway",
                         "place_type": "hallway", "stairs": "none", "floor_hint": "", "description": "plain hall"})
GOOD_SAME = json.dumps({"shared_details": [], "conflicting_details": [], "same_place": True,
                        "confidence": 0.9, "reason": "same"})


def _resp(content: str, done_reason: str = "stop"):
    return SimpleNamespace(message=SimpleNamespace(content=content), done_reason=done_reason)


class FakeOllama:
    """Stands in for ollama.Client: `chat_results` are returned (or raised) in order."""

    def __init__(self, chat_results=(), show_error=None, models=("gemma3:4b",)):
        self.chat_results = list(chat_results)
        self.chat_calls: list[dict] = []
        self.show_error = show_error
        self.models = models
        self.timeouts: list = []

    def chat(self, **kw):
        self.chat_calls.append(kw)
        r = self.chat_results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def show(self, model):
        if self.show_error:
            raise self.show_error
        return {}

    def list(self):
        return SimpleNamespace(models=[SimpleNamespace(model=m) for m in self.models])


@pytest.fixture
def fake(monkeypatch):
    client = FakeOllama()

    def factory(*_a, timeout=None, **_k):
        client.timeouts.append(timeout)
        return client

    monkeypatch.setattr(gemma, "_client", factory)
    return client


# ---------------------------------------------------------------- R1 grounding
@pytest.mark.parametrize("cand, lines, ok", [
    ("121", ["ROOM 12"], False),
    ("1", ["ROOM 121"], False),
    ("12", ["ROOM 121"], False),
    ("ROOM 121", ["Room-121 STAIRS"], True),
    ("ROOM-121", ["ROOM 121"], True),
    ("Room 121", ["ROOM121"], True),
    ("121", ["Rm.121"], True),
    ("121A", ["ROOM 121A"], True),
    ("MEN'S", ["MENS"], True),
    ("ROOM 121 CHEMISTRY LAB", ["ROOM 121", "CHEMISTRY LAB"], True),  # one sign read as two OCR lines
    ("121", ["ROOM 12", "1ST FLOOR"], False),  # digits are never glued across lines
    ("", ["ROOM 121"], False),
    ("EXIT", [], False),
])
def test_grounded_matches_whole_tokens(cand, lines, ok):
    assert ocr.grounded(cand, lines) is ok


@pytest.mark.parametrize("hint, lines, kept", [
    ("1", ["ROOM 121"], False),
    ("2", ["LEVEL 2"], True),
    ("2", ["2ND FLOOR"], True),
    ("Level 3", ["FL3 ELEVATOR"], True),
    ("second floor", ["SECOND FLOOR"], True),
    ("2", ["125"], False),
    ("2", ["LEVEL 3"], False),
    ("2", ["2"], False),  # a bare number is not a floor sign
])
def test_floor_hint_needs_a_floor_word(hint, lines, kept):
    a = FrameAnalysis(floor_hint=hint, ocr=lines, place_type="hallway")
    assert (gemma._ground(a, lines).floor_hint == hint) is kept


def test_ground_drops_partial_number_matches():
    a = FrameAnalysis(has_sign=True, texts=["ROOM 121"], room_numbers=["121"], suggested_label="Room 121",
                      floor_hint="1", description="door 121")
    g = gemma._ground(a, ["ROOM 12"])
    assert (g.texts, g.room_numbers, g.has_sign, g.floor_hint) == ([], [], False, "")
    assert "121" not in g.suggested_label and "121" not in g.description


def test_prompt_version_bumped():
    assert gemma.PROMPT_VERSION == "v5"


# ---------------------------------------------------------------- R3 call options / R2 _chat errors
def test_chat_passes_keep_alive_and_options(fake):
    fake.chat_results = [_resp(GOOD_FRAME)]
    gemma._chat("gemma3:4b", "p", [b"img"], gemma.GemmaFrame, 512)
    kw = fake.chat_calls[0]
    assert kw["keep_alive"] == config.GEMMA_KEEP_ALIVE == "30m"
    assert kw["options"] == {"temperature": 0, "num_ctx": config.GEMMA_NUM_CTX, "num_predict": 512}
    assert fake.timeouts == [None]  # chat uses the long default timeout


def test_404_model_not_found_is_not_retried(fake):
    fake.chat_results = [ollama.ResponseError("model 'gemma9' not found", 404), _resp(GOOD_FRAME)]
    with pytest.raises(gemma.GemmaUnavailable, match="ollama pull gemma9"):
        gemma._chat("gemma9", "p", [], gemma.GemmaFrame, 512)
    assert len(fake.chat_calls) == 1


def test_4xx_is_not_retried(fake):
    fake.chat_results = [ollama.ResponseError("bad request", 400), _resp(GOOD_FRAME)]
    with pytest.raises(gemma.GemmaUnavailable):
        gemma._chat("m", "p", [], gemma.GemmaFrame, 512)
    assert len(fake.chat_calls) == 1


@pytest.mark.parametrize("err", [ollama.ResponseError("runner crashed", 500), ConnectionResetError("reset"),
                                 httpx.ReadTimeout("slow"), ConnectionError("refused")])
def test_transient_error_is_retried_once(fake, err):
    fake.chat_results = [err, _resp(GOOD_FRAME)]
    assert gemma._chat("m", "p", [], gemma.GemmaFrame, 512).description == "plain hall"
    assert len(fake.chat_calls) == 2


def test_transient_error_twice_gives_up(fake):
    fake.chat_results = [httpx.ReadTimeout("slow"), httpx.ReadTimeout("slow"), _resp(GOOD_FRAME)]
    with pytest.raises(gemma.GemmaUnavailable):
        gemma._chat("m", "p", [], gemma.GemmaFrame, 512)
    assert len(fake.chat_calls) == 2


def test_bad_json_once_gets_exactly_one_retry(fake):
    fake.chat_results = [_resp("{not json"), _resp(GOOD_FRAME)]
    assert gemma._chat("m", "p", [], gemma.GemmaFrame, 512).suggested_label == "Hallway"
    assert len(fake.chat_calls) == 2


def test_bad_json_twice_raises(fake):
    fake.chat_results = [_resp("{not json"), _resp('{"place_type": "spaceship"}'), _resp(GOOD_FRAME)]
    with pytest.raises(gemma.GemmaUnavailable):
        gemma._chat("m", "p", [], gemma.GemmaFrame, 512)
    assert len(fake.chat_calls) == 2


def test_truncated_output_is_not_retried(fake):
    fake.chat_results = [_resp('{"has_sign": false, "texts": ["ROO', done_reason="length"), _resp(GOOD_FRAME)]
    with pytest.raises(gemma.GemmaUnavailable, match="truncated"):
        gemma._chat("m", "p", [], gemma.GemmaFrame, 512)
    assert len(fake.chat_calls) == 1


def test_unexpected_error_is_unavailable_without_retry(fake):
    fake.chat_results = [RuntimeError("boom"), _resp(GOOD_FRAME)]
    with pytest.raises(gemma.GemmaUnavailable, match="boom"):
        gemma._chat("m", "p", [], gemma.GemmaFrame, 512)
    assert len(fake.chat_calls) == 1


def _jpg(path, color=(120, 120, 120)):
    Image.new("RGB", (64, 48), color).save(path, format="JPEG")
    return path


def test_num_predict_per_call_type(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(ocr, "available", lambda: False)
    cache = gemma.GemmaCache(tmp_path / "c.jsonl")
    a, b = _jpg(tmp_path / "a.jpg"), _jpg(tmp_path / "b.jpg", (10, 200, 10))
    fake.chat_results = [_resp(GOOD_FRAME), _resp(GOOD_SAME), _resp(GOOD_SAME)]
    gemma.analyze_frame(a, cache, model="m")
    monkeypatch.setattr(gemma, "resolve_model", lambda: "m")
    gemma.same_place(a, b, cache)
    gemma.same_stairwell(a, b, cache)
    predicts = [kw["options"]["num_predict"] for kw in fake.chat_calls]
    assert predicts == [config.GEMMA_NUM_PREDICT_ANALYZE, config.GEMMA_NUM_PREDICT_COMPARE,
                        config.GEMMA_NUM_PREDICT_COMPARE] == [512, 320, 320]
    assert all(kw["keep_alive"] == "30m" for kw in fake.chat_calls)


def test_list_and_show_use_short_timeout(fake):
    assert gemma.available_models() == ["gemma3:4b"]
    gemma.ensure_model("gemma3:4b")
    assert fake.timeouts == [config.GEMMA_LIST_TIMEOUT_SEC] * 2 == [3.0, 3.0]


def test_ensure_model_missing(fake):
    fake.show_error = ollama.ResponseError("model 'gemma3:12b' not found", 404)
    with pytest.raises(gemma.GemmaUnavailable, match="ollama pull gemma3:12b"):
        gemma.ensure_model("gemma3:12b")


def test_ensure_model_unreachable(fake):
    fake.show_error = ConnectionError("refused")
    with pytest.raises(gemma.GemmaUnavailable, match="not reachable"):
        gemma.ensure_model("gemma3:4b")


def test_keep_alive_env_parsing(monkeypatch):
    import importlib
    try:
        monkeypatch.setenv("GEMMA_KEEP_ALIVE", "-1")
        assert importlib.reload(config).GEMMA_KEEP_ALIVE == -1.0
        monkeypatch.setenv("GEMMA_KEEP_ALIVE", "1h")
        assert importlib.reload(config).GEMMA_KEEP_ALIVE == "1h"
    finally:
        monkeypatch.delenv("GEMMA_KEEP_ALIVE")
        importlib.reload(config)


# ---------------------------------------------------------------- R2 _analyze failure semantics
@pytest.fixture
def bid(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BUILDINGS_DIR", tmp_path / "buildings")
    monkeypatch.setattr(config, "GEMMA_CONCURRENCY", 1)  # deterministic call order
    monkeypatch.setattr(gemma, "resolve_model", lambda: "gemma3:4b")
    monkeypatch.setattr(gemma, "ensure_model", lambda _m: None)
    return store.create_building("Gemma Hall").building_id


def _kfs(n):
    return [Keyframe(frame_id=f"v_{i}.jpg", video_id="v", t=i, kind="heartbeat", heading=0) for i in range(n)]


def _job(bid):
    return jobs.Job(id="j", building_id=bid, video_id="v", kind="video")


def test_one_bad_frame_does_not_stop_the_others(bid, monkeypatch):
    def analyze(path, _cache, _model):
        if path.name == "v_1.jpg":
            raise OSError("cannot identify image file")
        return FrameAnalysis(description=path.name)

    monkeypatch.setattr(gemma, "analyze_frame", analyze)
    kfs, job = _kfs(4), _job(bid)
    assert jobs._analyze(job, kfs, None) is True
    assert [k.analysis is not None for k in kfs] == [True, False, True, True]
    assert jobs._frames_ok(kfs) == 3
    assert any("v_1.jpg" in line and "OSError" in line for line in job.log)


def test_breaker_trips_after_consecutive_unavailable(bid, monkeypatch):
    calls = []

    def analyze(path, _cache, _model):
        calls.append(path.name)
        raise gemma.GemmaUnavailable("timed out")

    monkeypatch.setattr(gemma, "analyze_frame", analyze)
    kfs, job = _kfs(10), _job(bid)
    assert jobs._analyze(job, kfs, None) is False  # zero successes: walk.json must not claim gemma
    assert calls == ["v_0.jpg", "v_1.jpg", "v_2.jpg"]
    assert any("in a row" in line for line in job.log)
    assert any("Skipped Gemma on 7 frame(s)" in line for line in job.log)


def test_breaker_streak_resets_on_success(bid, monkeypatch):
    calls = []

    def analyze(path, _cache, _model):
        calls.append(path.name)
        if path.name in ("v_2.jpg", "v_5.jpg"):
            return FrameAnalysis()
        raise gemma.GemmaUnavailable("flaky")

    monkeypatch.setattr(gemma, "analyze_frame", analyze)
    kfs = _kfs(10)
    assert jobs._analyze(_job(bid), kfs, None) is True
    # fail fail ok fail fail ok fail fail fail -> breaker; v_9 never called
    assert calls == [f"v_{i}.jpg" for i in range(9)]
    assert jobs._frames_ok(kfs) == 2


def test_zero_successes_returns_false(bid, monkeypatch):
    monkeypatch.setattr(gemma, "analyze_frame", lambda *_a: (_ for _ in ()).throw(OSError("bad jpg")))
    kfs, job = _kfs(2), _job(bid)
    assert jobs._analyze(job, kfs, None) is False
    assert jobs._frames_ok(kfs) == 0


def test_no_keyframes_is_still_gemma_ok(bid, monkeypatch):
    assert jobs._analyze(_job(bid), [], None) is True


def test_missing_model_fails_fast(bid, monkeypatch):
    def missing(model):
        raise gemma.GemmaUnavailable(f"Model {model!r} is not pulled in Ollama. Run: ollama pull {model}")

    monkeypatch.setattr(gemma, "ensure_model", missing)
    monkeypatch.setattr(gemma, "analyze_frame", lambda *_a: pytest.fail("must not analyse frames"))
    job = _job(bid)
    assert jobs._analyze(job, _kfs(3), "gemma3:12b") is False
    assert any("ollama pull gemma3:12b" in line for line in job.log)


def test_walk_json_records_frames_ok(bid, monkeypatch):
    from app import mapper, motion, video
    from app.schemas import VideoInfo

    with store.lock_for(bid):
        bundle = store.load_map(bid)
        bundle.videos.append(VideoInfo(id="v", filename="v.mp4", status="queued"))
        store.save_map(bundle)
    vdir = store.video_dir(bid, "v")
    vdir.mkdir(parents=True)
    monkeypatch.setattr(video, "normalize", lambda src, d: (d / "analysis.mp4", d / "hq.mp4", 3.0))
    monkeypatch.setattr(motion, "compute_track", lambda *_a, **_k: {"t": [0.0, 1.0]})
    monkeypatch.setattr(motion, "detect_events", lambda _t: [])
    monkeypatch.setattr(jobs, "select_keyframes", lambda *_a, **_k: _kfs(3))
    monkeypatch.setattr(mapper, "integrate_walk", lambda *_a, **_k: {"floor_offset": 0})
    monkeypatch.setattr(mapper, "consolidate", lambda *_a, **_k: {})

    def analyze(path, _cache, _model):
        if path.name == "v_0.jpg":
            raise OSError("bad jpg")
        return FrameAnalysis()

    monkeypatch.setattr(gemma, "analyze_frame", analyze)
    jobs._process_video(_job(bid), vdir / "source.mp4", None, None)
    walk = store.read_json(vdir / "walk.json")
    assert walk["gemma"] is True and walk["gemma_frames_ok"] == 2
