"""Hosted vision model (Google Gemini, app/hosted.py) and how it plugs into gemma.py and jobs.py.

Covers: the generateContent request (key in a header, inline images, temperature 0, JSON mode, schema subset),
answers parsed and grounded exactly like Gemma's, retries (503/429 with Retry-After, no retry on 4xx, truncation,
blocked answers), the model choice (auto / gemma / hosted, explicit model wins), one model per walk, cache keys and
offline rebuilds, the daily cost guard, and /api/health. The Gemini API is an httpx.MockTransport and Ollama is a
fake: no network, no server."""
import base64
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, get_args

import httpx
import pydantic
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import config, gemma, hosted, jobs, mapper, motion, ocr, store, video
from app.main import app
from app.schemas import (
    FrameAnalysis,
    GemmaFrame,
    Keyframe,
    PlaceType,
    SamePlace,
    VideoInfo,
)

MODEL = "gemini-2.5-flash-lite"
KEY = "test-key-0123456789"
BACKEND = Path(__file__).resolve().parent.parent
FRAME = {"texts": [], "room_numbers": [], "has_sign": False, "suggested_label": "Hallway", "place_type": "hallway",
         "stairs": "none", "floor_hint": "", "description": "plain hall"}
SAME = {"image1_sign_text": [], "image2_sign_text": [], "shared_details": ["door"], "conflicting_details": [],
        "same_place": True, "confidence": 0.9, "reason": "same door"}


# ---------------------------------------------------------------- fakes
def ok(obj=FRAME, *, finish="STOP", tokens=100, text=None):
    """A 200 generateContent answer whose text is the JSON of `obj` (or `text` verbatim)."""
    body = {"candidates": [{"content": {"role": "model", "parts": [{"text": json.dumps(obj) if text is None else text}]},
                            "finishReason": finish, "index": 0}],
            "usageMetadata": {"promptTokenCount": tokens, "candidatesTokenCount": 40, "totalTokenCount": tokens + 40},
            "modelVersion": MODEL}
    return (200, body, None)


def err(code, message="request failed", headers=None):
    return (code, {"error": {"code": code, "message": message, "status": "ERR"}}, headers)


class FakeGemini:
    """The Gemini REST API. `answers` are used in order, one per generateContent POST: a (status, body, headers)
    tuple, an httpx exception to raise, or a callable(request) -> httpx.Response. Empty: FRAME. `check` is the
    (status, body) of the model GET. `down`: every request fails to connect."""

    def __init__(self):
        self.answers: list = []
        self.posts: list[httpx.Request] = []
        self.gets: list[httpx.Request] = []
        self.check = (200, {"name": f"models/{MODEL}"})
        self.down = False
        self.sleeps: list[float] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("connection refused", request=request)
        if request.method == "GET":
            self.gets.append(request)
            status, body = self.check
            return httpx.Response(status, json=body, request=request)
        self.posts.append(request)
        item = self.answers.pop(0) if self.answers else ok()
        if isinstance(item, Exception):
            raise item
        if callable(item):
            return item(request)
        status, body, headers = item
        return httpx.Response(status, json=body, headers=headers or {}, request=request)


class FakeOllama:
    def __init__(self):
        self.chat_results: list = []
        self.chat_calls: list[dict] = []
        self.models = ["gemma3:4b"]

    def chat(self, **kw):
        self.chat_calls.append(kw)
        r = self.chat_results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def show(self, model):
        return {}

    def list(self):
        return SimpleNamespace(models=[SimpleNamespace(model=m) for m in self.models])


def _local(obj) -> SimpleNamespace:
    return SimpleNamespace(message=SimpleNamespace(content=json.dumps(obj)), done_reason="stop")


@pytest.fixture(autouse=True)
def _no_real_ocr(monkeypatch):
    monkeypatch.setattr(ocr, "available", lambda: False)


@pytest.fixture
def gemini(monkeypatch):
    fake = FakeGemini()
    transport = httpx.MockTransport(fake)
    monkeypatch.setattr(hosted, "_client", lambda timeout: httpx.Client(transport=transport))
    monkeypatch.setattr(hosted, "sleep", fake.sleeps.append)
    monkeypatch.setattr(config, "GEMINI_API_KEY", KEY)
    return fake


@pytest.fixture
def ollama(monkeypatch):
    client = FakeOllama()
    monkeypatch.setattr(gemma, "_client", lambda *_a, **_k: client)
    monkeypatch.setattr(gemma, "_resolved_model", None)
    return client


@pytest.fixture
def bid(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BUILDINGS_DIR", tmp_path / "buildings")
    monkeypatch.setattr(config, "GEMMA_CONCURRENCY", 1)
    monkeypatch.setattr(config, "HOSTED_CONCURRENCY", 1)
    monkeypatch.setattr(jobs, "JOBS", {})
    monkeypatch.setattr(jobs, "_ensure_worker", lambda: None)
    return store.create_building("Hosted Hall").building_id


def _jpg(path: Path, color=(120, 120, 120), size=(64, 48)) -> Path:
    Image.new("RGB", size, color).save(path, format="JPEG")
    return path


def _kfs(bid, vid="v", n=3):
    frames = store.frames_dir(bid)
    frames.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(n):
        _jpg(frames / f"{vid}_{i}.jpg", (20 * i, 100, 200 - 20 * i))
        out.append(Keyframe(frame_id=f"{vid}_{i}.jpg", video_id=vid, t=i, kind="heartbeat", heading=0))
    return out


def _job(bid, vid="v"):
    return jobs.Job(id="j", building_id=bid, video_id=vid, kind="video")


def _set_key(monkeypatch, value: str):
    monkeypatch.setattr(config, "GEMINI_API_KEY", value)


# ---------------------------------------------------------------- A. request and schema
def test_key_travels_in_a_header_never_in_the_url(gemini, ollama):
    gemini.answers = [ok()]
    gemma._chat(MODEL, "PROMPT", [b"img"], GemmaFrame, 512)
    req = gemini.posts[0]
    assert req.headers["x-goog-api-key"] == KEY
    assert KEY not in str(req.url) and "key" not in req.url.params
    assert req.url.path == f"/v1beta/models/{MODEL}:generateContent"
    assert ollama.chat_calls == []


def test_request_has_inline_images_temperature_zero_and_json_mode(gemini, ollama):
    gemini.answers = [ok()]
    gemma._chat(MODEL, "PROMPT", [b"jpeg-one", b"jpeg-two"], GemmaFrame, 777)
    body = json.loads(gemini.posts[0].content)
    parts = body["contents"][0]["parts"]
    assert parts[0] == {"text": "PROMPT"}
    assert [p["inline_data"]["mime_type"] for p in parts[1:]] == ["image/jpeg", "image/jpeg"]
    assert [base64.b64decode(p["inline_data"]["data"]) for p in parts[1:]] == [b"jpeg-one", b"jpeg-two"]
    gen = body["generationConfig"]
    assert gen["temperature"] == 0 and gen["maxOutputTokens"] == 777
    assert gen["responseMimeType"] == "application/json"
    assert gen["thinkingConfig"] == {"thinkingBudget": 0}  # 2.5 Flash: thinking would spend the output budget


def _walk_json(node, found):
    if isinstance(node, dict):
        found.update(node)
        for v in node.values():
            _walk_json(v, found)
    elif isinstance(node, list):
        for v in node:
            _walk_json(v, found)
    return found


@pytest.mark.parametrize("schema_cls", [GemmaFrame, SamePlace])
def test_schema_is_the_gemini_subset_with_every_field_required(schema_cls):
    schema = hosted.response_schema(gemma._schema(schema_cls))
    keys = _walk_json(schema, set())
    assert not keys & {"title", "default", "$defs", "$ref", "additionalProperties", "definitions"}
    assert schema["required"] == list(schema["properties"])
    assert schema["propertyOrdering"] == list(schema["properties"])  # evidence fields are generated first


def test_gemma_frame_schema_keeps_the_evidence_fields_first():
    schema = hosted.response_schema(gemma._schema(GemmaFrame))
    assert schema["propertyOrdering"][:2] == ["texts", "room_numbers"]
    assert schema["properties"]["texts"] == {"type": "array", "items": {"type": "string"}}


def test_place_type_enum_survives_the_conversion():
    schema = hosted.response_schema(gemma._schema(GemmaFrame))
    assert schema["properties"]["place_type"]["enum"] == list(get_args(PlaceType))
    assert schema["properties"]["place_type"]["type"] == "string"


def test_optional_and_nested_models_are_converted_without_refs():
    class Inner(pydantic.BaseModel):
        name: str

    class Outer(pydantic.BaseModel):
        inner: Inner
        note: str | None = None
        kind: Literal["a", "b"] = "a"

    schema = hosted.response_schema(gemma._schema(Outer))
    assert "$defs" not in schema and "$ref" not in json.dumps(schema)
    assert schema["properties"]["inner"]["properties"] == {"name": {"type": "string"}}
    assert schema["properties"]["note"] == {"type": "string", "nullable": True}
    assert schema["properties"]["kind"]["enum"] == ["a", "b"]


@pytest.mark.parametrize("bad", ["gemini", "gemini-2.5-flash-lite/../../x", "gemini-2.5 flash"])
def test_model_name_that_is_not_a_plain_gemini_name_makes_no_request(gemini, ollama, bad):
    with pytest.raises(gemma.GemmaUnavailable, match="not a Gemini model name"):
        gemma._chat(bad, "PROMPT", [], GemmaFrame, 512)
    assert gemini.posts == []


# ---------------------------------------------------------------- B. answers parse and are grounded
def test_valid_answers_parse_to_the_pydantic_models(gemini, ollama, tmp_path):
    a, b = _jpg(tmp_path / "a.jpg"), _jpg(tmp_path / "b.jpg", (10, 200, 10))
    gemini.answers = [ok(SAME), ok(FRAME)]
    cache = gemma.GemmaCache(tmp_path / "c.jsonl")
    verdict = gemma.same_place(a, b, cache, model=MODEL)
    assert isinstance(verdict, SamePlace) and verdict.same_place and verdict.reason == "same door"
    frame = gemma._chat(MODEL, "p", [], GemmaFrame, 512)
    assert isinstance(frame, GemmaFrame) and frame.suggested_label == "Hallway"
    assert all(k.startswith(f"same:{gemma.PROMPT_VERSION}:{MODEL}:") for k in cache.data)


def test_hosted_answer_is_grounded_exactly_like_gemma(gemini, ollama, tmp_path, monkeypatch):
    """OCR reads only 'EXIT'. The same invented room-number answer must come out identical from both providers:
    the hosted answer is not trusted more than Gemma's."""
    lines = ["EXIT"]
    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "read_text_with_crops", lambda p: list(lines))
    invented = {"texts": ["ROOM 305"], "room_numbers": ["305"], "has_sign": True, "suggested_label": "Room 305",
                "place_type": "room", "stairs": "none", "floor_hint": "3", "description": "door 305"}
    img = _jpg(tmp_path / "a.jpg")
    ollama.chat_results = [_local(invented)]
    local = gemma.analyze_frame(img, gemma.GemmaCache(tmp_path / "local.jsonl"), "gemma3:4b")
    gemini.answers = [ok(invented)]
    hosted_res = gemma.analyze_frame(img, gemma.GemmaCache(tmp_path / "hosted.jsonl"), MODEL)
    assert hosted_res.model_dump() == local.model_dump()
    assert hosted_res.grounded and hosted_res.room_numbers == [] and hosted_res.texts == []
    assert "305" not in hosted_res.suggested_label and hosted_res.floor_hint == ""


def test_hosted_grounding_keeps_a_room_that_ocr_reads(gemini, ollama, tmp_path, monkeypatch):
    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "read_text_with_crops", lambda p: ["ROOM 121", "EXIT"])
    real = {"texts": ["ROOM 121"], "room_numbers": ["121"], "has_sign": True, "suggested_label": "Room 121",
            "place_type": "room", "stairs": "none", "floor_hint": "", "description": "door 121"}
    gemini.answers = [ok(real)]
    res = gemma.analyze_frame(_jpg(tmp_path / "a.jpg"), gemma.GemmaCache(tmp_path / "c.jsonl"), MODEL)
    assert res.room_numbers == ["121"] and res.suggested_label == "Room 121" and res.has_sign


def test_cached_hosted_answer_is_retidied_on_read(gemini, ollama, tmp_path, monkeypatch):
    """A cached hosted answer is checked against the frame's OCR again on read, as Gemma's are."""
    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "read_text_with_crops", lambda p: ["EXIT"])
    img = _jpg(tmp_path / "a.jpg")
    cache = gemma.GemmaCache(tmp_path / "c.jsonl")
    key = gemma._frame_key(img)(MODEL)  # the OCR-mode key, as analyze_frame computes it
    cache.put(key, FrameAnalysis(texts=["ROOM 305"], room_numbers=["305"], has_sign=True, place_type="room",
                                 suggested_label="Room 305", description="door", grounded=True,
                                 ocr=["EXIT"], ocr_rev=ocr.REVISION).model_dump())
    res = gemma.analyze_frame(img, cache, MODEL)
    assert res.room_numbers == [] and "305" not in res.suggested_label and gemini.posts == []


# ---------------------------------------------------------------- C. retries and errors
def test_503_is_retried_once_after_one_second(gemini, ollama):
    gemini.answers = [err(503, "overloaded"), ok()]
    assert gemma._chat(MODEL, "p", [], GemmaFrame, 512).description == "plain hall"
    assert len(gemini.posts) == 2 and gemini.sleeps == [hosted.DEFAULT_WAIT_SEC]


def test_429_waits_the_retry_after_the_server_asked(gemini, ollama):
    gemini.answers = [err(429, "quota", {"Retry-After": "3"}), ok()]
    gemma._chat(MODEL, "p", [], GemmaFrame, 512)
    assert len(gemini.posts) == 2 and gemini.sleeps == [3.0]


def test_retry_after_over_the_cap_is_not_waited_for(gemini, ollama, monkeypatch):
    monkeypatch.setattr(config, "HOSTED_RETRY_MAX_SEC", 10)
    gemini.answers = [err(429, "quota", {"Retry-After": "30"}), ok()]
    with pytest.raises(gemma.GemmaUnavailable, match="HOSTED_RETRY_MAX_SEC"):
        gemma._chat(MODEL, "p", [], GemmaFrame, 512)
    assert len(gemini.posts) == 1 and gemini.sleeps == []


def test_two_server_errors_give_up_after_two_attempts(gemini, ollama):
    gemini.answers = [err(500), err(500), ok()]
    with pytest.raises(gemma.GemmaUnavailable, match="after a retry"):
        gemma._chat(MODEL, "p", [], GemmaFrame, 512)
    assert len(gemini.posts) == 2


@pytest.mark.parametrize("code", [400, 403])
def test_client_errors_are_not_retried(gemini, ollama, code):
    gemini.answers = [err(code, "bad request"), ok()]
    with pytest.raises(gemma.GemmaUnavailable, match=f"\\({code}\\)"):
        gemma._chat(MODEL, "p", [], GemmaFrame, 512)
    assert len(gemini.posts) == 1


def test_missing_model_is_a_clear_error_and_not_retried(gemini, ollama):
    gemini.answers = [err(404, "not found"), ok()]
    with pytest.raises(gemma.GemmaUnavailable, match="no model"):
        gemma._chat(MODEL, "p", [], GemmaFrame, 512)
    assert len(gemini.posts) == 1


@pytest.mark.parametrize("exc", [httpx.ConnectError("reset"), httpx.ReadTimeout("slow")])
def test_network_errors_are_retried_once(gemini, ollama, exc):
    gemini.answers = [exc, ok()]
    assert gemma._chat(MODEL, "p", [], GemmaFrame, 512).description == "plain hall"
    assert len(gemini.posts) == 2


def test_unparseable_text_is_retried_once(gemini, ollama):
    gemini.answers = [ok(text="not json at all"), ok()]
    assert gemma._chat(MODEL, "p", [], GemmaFrame, 512).description == "plain hall"
    assert len(gemini.posts) == 2


def test_non_json_http_body_is_retried_once(gemini, ollama):
    gemini.answers = [lambda req: httpx.Response(200, text="<html>oops</html>", request=req), ok()]
    assert gemma._chat(MODEL, "p", [], GemmaFrame, 512).description == "plain hall"
    assert len(gemini.posts) == 2


def test_max_tokens_is_reported_as_truncation_without_retry(gemini, ollama):
    gemini.answers = [ok(finish="MAX_TOKENS", text='{"texts": ["ROO'), ok()]
    with pytest.raises(gemma.GemmaUnavailable, match="truncated at num_predict=512"):
        gemma._chat(MODEL, "p", [], GemmaFrame, 512)
    assert len(gemini.posts) == 1


@pytest.mark.parametrize("finish", ["SAFETY", "RECITATION", "OTHER"])
def test_blocked_answer_is_unavailable_without_retry(gemini, ollama, finish):
    gemini.answers = [ok(finish=finish, text=""), ok()]
    with pytest.raises(gemma.GemmaUnavailable, match=finish):
        gemma._chat(MODEL, "p", [], GemmaFrame, 512)
    assert len(gemini.posts) == 1


def test_prompt_blocked_with_no_candidates_is_unavailable(gemini, ollama):
    gemini.answers = [(200, {"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}}, None), ok()]
    with pytest.raises(gemma.GemmaUnavailable, match="PROHIBITED_CONTENT"):
        gemma._chat(MODEL, "p", [], GemmaFrame, 512)
    assert len(gemini.posts) == 1


def test_key_is_scrubbed_from_error_messages_and_logs(gemini, ollama, caplog):
    gemini.answers = [err(403, f"API key {KEY} is not valid")]
    with caplog.at_level(logging.DEBUG), pytest.raises(gemma.GemmaUnavailable) as exc:
        gemma._chat(MODEL, "p", [], GemmaFrame, 512)
    assert KEY not in str(exc.value) and "[GEMINI_API_KEY]" in str(exc.value)
    assert KEY not in caplog.text


# ---------------------------------------------------------------- D. the model choice
def test_no_key_auto_is_the_local_model_and_no_request_is_made(gemini, ollama, monkeypatch):
    _set_key(monkeypatch, "")
    assert gemma.choose_model() == config.GEMMA_MODEL
    assert gemini.posts == [] and gemini.gets == []


def test_key_with_unreachable_hosted_api_gives_gemma(gemini, ollama):
    gemini.check = (403, {"error": {"message": "API key not valid"}})
    assert gemma.choose_model() == config.GEMMA_MODEL
    assert len(gemini.gets) == 1 and gemini.posts == []


def test_key_with_reachable_hosted_api_gives_hosted_and_checks_are_cached(gemini, ollama):
    assert gemma.choose_model() == MODEL
    assert gemma.choose_model() == MODEL
    assert len(gemini.gets) == 1  # one metadata GET for the reachability cache


def test_reachability_is_checked_again_after_the_ttl(gemini, ollama, monkeypatch):
    monkeypatch.setattr(config, "HOSTED_CHECK_TTL_SEC", 0)
    gemma.choose_model()
    gemma.choose_model()
    assert len(gemini.gets) == 2


def test_key_with_reachable_api_but_daily_limit_reached_gives_gemma(gemini, ollama, monkeypatch):
    monkeypatch.setattr(config, "HOSTED_MAX_CALLS_PER_DAY", 1)
    hosted._write_usage({"day": hosted._today(), "calls": 1, "input_tokens": 0})
    assert gemma.choose_model() == config.GEMMA_MODEL
    assert gemini.posts == []


def test_gemma_provider_never_calls_the_hosted_api(gemini, ollama, monkeypatch, bid):
    monkeypatch.setattr(config, "VISION_PROVIDER", "gemma")
    assert gemma.choose_model() == config.GEMMA_MODEL
    ollama.chat_results = [_local(FRAME)] * 3
    run = jobs._analyze_walk(_job(bid), _kfs(bid), None)
    assert run.ok and run.model == config.GEMMA_MODEL
    assert gemini.posts == [] and gemini.gets == []
    assert len(ollama.chat_calls) == 3


def test_hosted_provider_without_key_fails_clearly(gemini, ollama, monkeypatch):
    _set_key(monkeypatch, "")
    monkeypatch.setattr(config, "VISION_PROVIDER", "hosted")
    with pytest.raises(gemma.GemmaUnavailable, match="GEMINI_API_KEY is not set"):
        gemma.choose_model()
    assert ollama.chat_calls == []


def test_hosted_provider_keeps_the_hosted_model_even_when_unreachable(gemini, ollama, monkeypatch):
    monkeypatch.setattr(config, "VISION_PROVIDER", "hosted")
    gemini.check = (503, {"error": {"message": "down"}})
    assert gemma.choose_model() == MODEL  # no silent switch to Gemma
    with pytest.raises(gemma.GemmaUnavailable, match="not usable"):
        gemma.ensure_model(MODEL)


def test_invalid_vision_provider_fails_at_startup():
    env = {**os.environ, "VISION_PROVIDER": "cloud"}
    r = subprocess.run([sys.executable, "-c", "import app.config"], cwd=BACKEND, env=env, capture_output=True,
                       text=True, timeout=120, check=False)
    assert r.returncode != 0 and "VISION_PROVIDER must be auto, gemma or hosted" in r.stderr


def test_explicit_local_model_wins_over_auto_hosted(gemini, ollama, bid):
    ollama.chat_results = [_local(FRAME)] * 3
    run = jobs._analyze_walk(_job(bid), _kfs(bid), "gemma3:4b")
    assert run.model == "gemma3:4b" and run.ok
    assert gemini.posts == [] and len(ollama.chat_calls) == 3


def test_explicit_hosted_model_wins_over_gemma_provider(gemini, ollama, bid, monkeypatch):
    monkeypatch.setattr(config, "VISION_PROVIDER", "gemma")
    run = jobs._analyze_walk(_job(bid), _kfs(bid), MODEL)
    assert run.model == MODEL and run.ok and len(gemini.posts) == 3
    assert ollama.chat_calls == []


def test_auto_walk_picks_hosted_once_and_logs_it(gemini, ollama, bid):
    job = _job(bid)
    run = jobs._analyze_walk(job, _kfs(bid), None)
    assert run.model == MODEL and run.ok and run.online
    assert any(f"Gemma model: {MODEL} (hosted, Google Gemini API)" in line for line in job.log)
    assert ollama.chat_calls == []


def test_explicit_hosted_model_without_key_is_offline_with_a_clear_warning(gemini, ollama, bid, monkeypatch):
    _set_key(monkeypatch, "")
    job = _job(bid)
    run = jobs._analyze_walk(job, _kfs(bid), MODEL)
    assert not run.online and not run.ok
    assert any("hosted model gemini-2.5-flash-lite unavailable" in line and "GEMINI_API_KEY" in line
               for line in job.log)
    assert gemini.posts == []


def test_gemma_ok_checks_the_buildings_own_model(gemini, ollama, monkeypatch):
    monkeypatch.setattr(config, "HOSTED_CHECK_TTL_SEC", 0)  # no reachability cache between the two checks
    assert jobs._gemma_ok(MODEL) is True
    gemini.check = (403, {"error": {"message": "no"}})
    assert jobs._gemma_ok(MODEL) is False
    assert jobs._gemma_ok(None) is True  # no model recorded: the local check, as before


# ---------------------------------------------------------------- E. one model per walk
def test_hosted_failure_mid_walk_does_not_switch_to_gemma(gemini, ollama, bid):
    gemini.answers = [ok(), ok(), httpx.ConnectError("down"), httpx.ConnectError("down"),
                      httpx.ConnectError("down"), httpx.ConnectError("down"), httpx.ConnectError("down"),
                      httpx.ConnectError("down")]
    kfs = _kfs(bid, n=6)
    job = _job(bid)
    run = jobs._analyze_walk(job, kfs, None)
    assert run.model == MODEL and run.ok and not run.online
    assert ollama.chat_calls == []  # Gemma is never asked for this walk
    assert [k.analysis is not None for k in kfs] == [True, True, False, False, False, False]
    assert len(gemini.posts) == 8  # 2 answered, then 3 frames x (try + retry); the breaker stops the rest
    assert any("hosted model gemini-2.5-flash-lite failed 3 frames in a row" in l and "not switched to Gemma" in l
               for l in job.log)
    assert not any("Gemma model: gemma" in line for line in job.log)


def test_walk_log_counts_hosted_calls_and_input_tokens(gemini, ollama, bid):
    gemini.answers = [ok(tokens=100), ok(tokens=100), httpx.ConnectError("down"), httpx.ConnectError("down"),
                      httpx.ConnectError("down"), httpx.ConnectError("down"), httpx.ConnectError("down"),
                      httpx.ConnectError("down")]
    job = _job(bid)
    jobs._analyze_walk(job, _kfs(bid, n=6), None)
    assert any("8 call(s), ~200 input tokens since the walk started; 8/5000 calls today" in l for l in job.log)


def test_mapping_session_after_hosted_failure_reads_only_the_cache(gemini, ollama, tmp_path):
    a, b = _jpg(tmp_path / "a.jpg"), _jpg(tmp_path / "b.jpg", (10, 200, 10))
    cache = gemma.GemmaCache(tmp_path / "c.jsonl")
    gemini.down = True
    with gemma.session(MODEL, offline=False) as s:
        with pytest.raises(gemma.GemmaUnavailable):
            gemma.same_place(a, b, cache)
        assert s.offline
        with pytest.raises(gemma.NotCached):
            gemma.same_stairwell(a, b, cache)
    assert ollama.chat_calls == []


def test_hosted_walk_uses_its_own_concurrency_limit(gemini, ollama, bid, monkeypatch):
    monkeypatch.setattr(config, "GEMMA_CONCURRENCY", 2)
    monkeypatch.setattr(config, "HOSTED_CONCURRENCY", 4)
    sizes = []
    real = jobs.ThreadPoolExecutor

    class Spy(real):
        def __init__(self, max_workers=None, *a, **k):
            sizes.append(max_workers)
            super().__init__(max_workers, *a, **k)

    monkeypatch.setattr(jobs, "ThreadPoolExecutor", Spy)
    jobs._analyze_walk(_job(bid), _kfs(bid, n=1), MODEL)
    ollama.chat_results = [_local(FRAME)]
    jobs._analyze_walk(_job(bid, "w"), _kfs(bid, "w", n=1), "gemma3:4b")
    assert sizes == [4, 2]


# ---------------------------------------------------------------- F. cache keys, offline and rebuild
def test_cache_keys_carry_the_hosted_model_name(gemini, ollama, bid, tmp_path):
    kfs = _kfs(bid, n=2)
    jobs._analyze_walk(_job(bid), kfs, None)
    cache = gemma.cache_for(store.building_dir(bid))
    assert cache.data and all(k.startswith(f"analyze:{gemma.PROMPT_VERSION}:{MODEL}:") for k in cache.data)
    a, b = store.frames_dir(bid) / kfs[0].frame_id, store.frames_dir(bid) / kfs[1].frame_id
    gemini.answers = [ok(SAME)]
    gemma.same_place(a, b, cache, model=MODEL)
    assert any(k.startswith(f"same:{gemma.PROMPT_VERSION}:{MODEL}:") for k in cache.data)


def test_offline_hosted_walk_uses_its_own_cached_answers(gemini, ollama, bid, monkeypatch):
    monkeypatch.setattr(config, "HOSTED_CHECK_TTL_SEC", 0)  # the outage is seen at once, not after the cache TTL
    kfs = _kfs(bid, n=3)
    jobs._analyze_walk(_job(bid), kfs, None)  # online: fills the cache
    gemini.down = True
    fresh = _kfs(bid, n=3)
    job = _job(bid)
    run = jobs._analyze_walk(job, fresh, MODEL)  # the same walk, stored model, network gone
    assert run.ok and not run.online and run.model == MODEL
    assert any("hosted model gemini-2.5-flash-lite unavailable, using cached answers only" in l for l in job.log)
    assert all(k.analysis is not None for k in fresh)
    assert ollama.chat_calls == []


def test_offline_hosted_walk_does_not_read_local_answers(gemini, ollama, bid):
    kfs = _kfs(bid, n=2)
    frames, cache = store.frames_dir(bid), gemma.cache_for(store.building_dir(bid))
    for k in kfs:  # answers from a local model exist for these frames
        cache.put(gemma._frame_key(frames / k.frame_id)("gemma3:4b"), FrameAnalysis(description="local").model_dump())
    gemini.down = True
    run = jobs._analyze_walk(_job(bid), kfs, MODEL)
    assert not run.ok and all(k.analysis is None for k in kfs)


def _add_done_walk(bid, model, vid="v"):
    with store.lock_for(bid):
        bundle = store.load_map(bid)
        bundle.videos.append(VideoInfo(id=vid, filename=f"{vid}.mp4", status="done"))
        store.save_map(bundle)
    vdir = store.video_dir(bid, vid)
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "analysis.mp4").write_bytes(b"x")
    store.write_json_atomic(vdir / "walk.json", {"video_id": vid, "keyframes": [], "gemma": True, "model": model})
    return vdir


def test_rebuild_without_network_uses_the_cached_hosted_answers(gemini, ollama, bid, monkeypatch):
    monkeypatch.setattr(config, "HOSTED_CHECK_TTL_SEC", 0)
    jobs._analyze_walk(_job(bid), _kfs(bid, n=3), MODEL)  # the first processing, online
    _add_done_walk(bid, MODEL)
    monkeypatch.setattr(video, "normalize", lambda src, d: (d / "analysis.mp4", d / "hq.mp4", 3.0))
    monkeypatch.setattr(motion, "compute_track", lambda *_a, **_k: {"t": [0.0, 1.0]})
    monkeypatch.setattr(motion, "detect_events", lambda _t: [])
    monkeypatch.setattr(jobs, "select_keyframes", lambda *_a, **_k: _kfs(bid, n=3))
    monkeypatch.setattr(mapper, "integrate_walk", lambda *_a, **_k: {"floor_offset": 0})
    monkeypatch.setattr(mapper, "consolidate", lambda *_a, **_k: {})
    gemini.down = True  # the network is gone for the rebuild
    posts_before = len(gemini.posts)
    job = jobs.Job(id="r", building_id=bid, video_id=None, kind="rebuild")
    jobs._rebuild(job)
    assert len(gemini.posts) == posts_before and ollama.chat_calls == []
    assert any("hosted model gemini-2.5-flash-lite unavailable, using cached answers only" in l for l in job.log)
    assert store.read_json(store.video_dir(bid, "v") / "walk.json")["model"] == MODEL
    assert any("Gemma analysed 3/3 frames with gemini-2.5-flash-lite" in line for line in job.log)


def test_stored_hosted_model_is_reused_by_a_rebuild_not_rechosen(gemini, ollama, bid, monkeypatch):
    """A walk processed with the hosted model keeps it on rebuild even if auto would now pick the local one."""
    _add_done_walk(bid, MODEL)
    monkeypatch.setattr(config, "VISION_PROVIDER", "gemma")
    monkeypatch.setattr(video, "normalize", lambda src, d: (d / "analysis.mp4", d / "hq.mp4", 3.0))
    monkeypatch.setattr(motion, "compute_track", lambda *_a, **_k: {"t": [0.0, 1.0]})
    monkeypatch.setattr(motion, "detect_events", lambda _t: [])
    monkeypatch.setattr(jobs, "select_keyframes", lambda *_a, **_k: _kfs(bid, n=2))
    monkeypatch.setattr(mapper, "integrate_walk", lambda *_a, **_k: {"floor_offset": 0})
    monkeypatch.setattr(mapper, "consolidate", lambda *_a, **_k: {})
    jobs._rebuild(jobs.Job(id="r", building_id=bid, video_id=None, kind="rebuild"))
    assert len(gemini.posts) == 2 and ollama.chat_calls == []


# ---------------------------------------------------------------- G. daily cost guard
def test_daily_cap_blocks_calls(gemini, ollama, monkeypatch):
    monkeypatch.setattr(config, "HOSTED_MAX_CALLS_PER_DAY", 2)
    gemini.answers = [ok(), ok()]
    gemma._chat(MODEL, "p", [], GemmaFrame, 512)
    gemma._chat(MODEL, "p", [], GemmaFrame, 512)
    with pytest.raises(gemma.GemmaUnavailable, match="daily call limit reached \\(2/2"):
        gemma._chat(MODEL, "p", [], GemmaFrame, 512)
    assert len(gemini.posts) == 2


def test_counter_persists_in_data_dir_and_resets_each_utc_day(gemini, ollama, monkeypatch):
    monkeypatch.setattr(config, "HOSTED_MAX_CALLS_PER_DAY", 2)
    gemini.answers = [ok(tokens=50), ok(tokens=70), ok(), ok()]
    gemma._chat(MODEL, "p", [], GemmaFrame, 512)
    gemma._chat(MODEL, "p", [], GemmaFrame, 512)
    saved = json.loads((Path(config.DATA_DIR) / hosted.USAGE_FILE).read_text())
    assert saved == {"day": hosted._today(), "calls": 2, "input_tokens": 120}
    assert hosted.calls_today() == 2  # read back from the file, as after a restart
    monkeypatch.setattr(hosted, "_today", lambda: "2099-01-02")
    assert hosted.calls_today() == 0
    gemma._chat(MODEL, "p", [], GemmaFrame, 512)
    assert len(gemini.posts) == 3
    assert json.loads((Path(config.DATA_DIR) / hosted.USAGE_FILE).read_text())["day"] == "2099-01-02"


def test_auto_falls_back_to_gemma_for_new_walks_only(gemini, ollama, bid, monkeypatch):
    monkeypatch.setattr(config, "HOSTED_MAX_CALLS_PER_DAY", 1)
    hosted._write_usage({"day": hosted._today(), "calls": 1, "input_tokens": 0})
    ollama.chat_results = [_local(FRAME)] * 2
    job = _job(bid)
    run = jobs._analyze_walk(job, _kfs(bid, n=2), None)  # a new walk: Gemma
    assert run.model == config.GEMMA_MODEL and run.ok and gemini.posts == []
    # a walk that was given the hosted model does not fall back to Gemma: its calls are refused, it ends cache-only
    job2 = _job(bid, "w")
    run2 = jobs._analyze_walk(job2, _kfs(bid, "w", n=2), MODEL)
    assert not run2.ok and not run2.online and gemini.posts == []
    assert any("daily call limit" in line for line in job2.log)
    assert len(ollama.chat_calls) == 2  # Ollama was only used for the first walk


def test_unreadable_counter_stops_hosted_calls(gemini, ollama):
    Path(config.DATA_DIR).mkdir(parents=True, exist_ok=True)
    (Path(config.DATA_DIR) / hosted.USAGE_FILE).write_text("{not json")
    with pytest.raises(gemma.GemmaUnavailable, match="cannot read the Gemini call counter"):
        gemma._chat(MODEL, "p", [], GemmaFrame, 512)
    assert gemini.posts == []
    assert gemma.choose_model() == config.GEMMA_MODEL  # auto: local


def test_counter_that_cannot_be_written_fails_closed(gemini, ollama, tmp_path, monkeypatch):
    blocker = tmp_path / "a-file"
    blocker.write_text("x")
    monkeypatch.setattr(config, "DATA_DIR", blocker / "data")  # a directory cannot be created under a file
    with pytest.raises(gemma.GemmaUnavailable, match="cannot write the Gemini call counter"):
        gemma._chat(MODEL, "p", [], GemmaFrame, 512)
    assert gemini.posts == []


# ---------------------------------------------------------------- H. /api/health
def test_health_reports_hosted_status_and_never_the_key(gemini, ollama, caplog):
    with caplog.at_level(logging.DEBUG):
        r = TestClient(app).get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["hosted"] == {"configured": True, "reachable": True, "model": MODEL, "calls_today": 0,
                              "limit": config.HOSTED_MAX_CALLS_PER_DAY}
    assert body["vision_provider"] == {"setting": "auto", "auto_pick": "hosted"}
    assert KEY not in r.text and KEY not in caplog.text


def test_health_without_a_key_reports_gemma_as_the_pick(gemini, ollama, monkeypatch):
    _set_key(monkeypatch, "")
    body = TestClient(app).get("/api/health").json()
    assert body["hosted"]["configured"] is False and body["hosted"]["reachable"] is None
    assert body["vision_provider"]["auto_pick"] == "gemma"
    assert gemini.posts == [] and gemini.gets == []


def test_health_with_an_unreachable_hosted_api(gemini, ollama):
    gemini.check = (403, {"error": {"message": "API key not valid"}})
    body = TestClient(app).get("/api/health").json()
    assert body["hosted"]["reachable"] is False and body["vision_provider"]["auto_pick"] == "gemma"
    assert KEY not in json.dumps(body)


# ---------------------------------------------------------------- I. Ollama path unchanged
def test_ollama_answers_are_keyed_by_the_local_model_only(ollama, tmp_path, gemini):
    img = _jpg(tmp_path / "a.jpg")
    ollama.chat_results = [_local(FRAME)]
    gemma.analyze_frame(img, gemma.GemmaCache(tmp_path / "c.jsonl"), "gemma3:4b")
    assert ollama.chat_calls and gemini.posts == []
