"""Gemma round 2: model identity (per-walk model reaches same_place, rebuild reuses it), cache-first offline
answers, vision-capable fallback, prompt/schema v6, EXIF, crop OCR, poster filtering, floor_from_lines,
grounding gaps, the OCR room-number guard, the mapper's non-binding Gemma veto and the manual-frame route.
Ollama and OCR are faked; no server is needed."""
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import config, gemma, jobs, mapper, motion, ocr, store, video
from app.schemas import FrameAnalysis, GemmaFrame, Keyframe, MapBundle, Node, SamePlace, VideoInfo

FRAME = {"texts": [], "room_numbers": [], "has_sign": False, "suggested_label": "Hallway", "place_type": "hallway",
         "stairs": "none", "floor_hint": "", "description": "plain hall"}
SAME = {"image1_sign_text": [], "image2_sign_text": [], "shared_details": ["door"], "conflicting_details": [],
        "same_place": True, "confidence": 0.9, "reason": "same door"}


def _resp(obj) -> SimpleNamespace:
    return SimpleNamespace(message=SimpleNamespace(content=json.dumps(obj)), done_reason="stop")


class FakeOllama:
    def __init__(self):
        self.chat_results: list = []
        self.chat_calls: list[dict] = []
        self.models = ["gemma3:4b"]
        self.caps: dict = {}  # model -> capabilities (missing key: field absent)
        self.down = False

    def _check(self):
        if self.down:
            raise ConnectionError("connection refused")

    def chat(self, **kw):
        self._check()
        self.chat_calls.append(kw)
        r = self.chat_results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def show(self, model):
        self._check()
        return SimpleNamespace(capabilities=self.caps[model]) if model in self.caps else {}

    def list(self):
        self._check()
        return SimpleNamespace(models=[SimpleNamespace(model=m) for m in self.models])


@pytest.fixture
def fake(monkeypatch):
    client = FakeOllama()
    monkeypatch.setattr(gemma, "_client", lambda *_a, **_k: client)
    monkeypatch.setattr(gemma, "_resolved_model", None)
    monkeypatch.setattr(ocr, "available", lambda: False)
    return client


def _jpg(path: Path, color=(120, 120, 120), size=(64, 48)) -> Path:
    Image.new("RGB", size, color).save(path, format="JPEG")
    return path


# ---------------------------------------------------------------- A. model identity
def test_session_model_reaches_same_place_and_stairwell(fake, tmp_path):
    cache = gemma.GemmaCache(tmp_path / "c.jsonl")
    a, b = _jpg(tmp_path / "a.jpg"), _jpg(tmp_path / "b.jpg", (10, 200, 10))
    fake.chat_results = [_resp(SAME), _resp(SAME)]
    with gemma.session("gemma3:12b"):
        gemma.same_place(a, b, cache)
        gemma.same_stairwell(a, b, cache)
    assert [c["model"] for c in fake.chat_calls] == ["gemma3:12b", "gemma3:12b"]
    assert all(":gemma3:12b:" in k for k in cache.data)


def test_explicit_model_param(fake, tmp_path):
    cache = gemma.GemmaCache(tmp_path / "c.jsonl")
    a, b = _jpg(tmp_path / "a.jpg"), _jpg(tmp_path / "b.jpg", (10, 200, 10))
    fake.chat_results = [_resp(SAME)]
    gemma.same_place(a, b, cache, model="m-explicit")
    assert fake.chat_calls[0]["model"] == "m-explicit"


@pytest.fixture
def bid(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BUILDINGS_DIR", tmp_path / "buildings")
    monkeypatch.setattr(config, "GEMMA_CONCURRENCY", 1)
    monkeypatch.setattr(jobs, "JOBS", {})
    monkeypatch.setattr(jobs, "_ensure_worker", lambda: None)
    return store.create_building("Gemma Hall").building_id


def _kfs(bid, vid="v", n=3):
    frames = store.frames_dir(bid)
    frames.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(n):
        _jpg(frames / f"{vid}_{i}.jpg", (20 * i, 100, 200 - 20 * i))
        out.append(Keyframe(frame_id=f"{vid}_{i}.jpg", video_id=vid, t=i, kind="heartbeat", heading=0))
    return out


def _fake_video_stages(monkeypatch, bid, vid="v"):
    monkeypatch.setattr(video, "normalize", lambda src, d: (d / "analysis.mp4", d / "hq.mp4", 3.0))
    monkeypatch.setattr(motion, "compute_track", lambda *_a, **_k: {"t": [0.0, 1.0]})
    monkeypatch.setattr(motion, "detect_events", lambda _t: [])
    monkeypatch.setattr(jobs, "select_keyframes", lambda *_a, **_k: _kfs(bid, vid))


def _add_video(bid, vid="v", status="queued"):
    with store.lock_for(bid):
        bundle = store.load_map(bid)
        bundle.videos.append(VideoInfo(id=vid, filename=f"{vid}.mp4", status=status))
        store.save_map(bundle)
    vdir = store.video_dir(bid, vid)
    vdir.mkdir(parents=True, exist_ok=True)
    return vdir


def test_upload_model_reaches_mapping_same_place(fake, bid, monkeypatch):
    """Per-upload model -> analyze_frame AND the mapping step's same_place (via the job's gemma.session)."""
    vdir = _add_video(bid)
    _fake_video_stages(monkeypatch, bid)
    monkeypatch.setattr(gemma, "ensure_model", lambda _m: None)
    fake.chat_results = [_resp(FRAME)] * 3 + [_resp(SAME)]
    frames = store.frames_dir(bid)

    def integrate(bundle, vid, kfs, track, frames_dir, cache, use_gemma, **_k):
        assert use_gemma
        gemma.same_place(frames / kfs[0].frame_id, frames / kfs[1].frame_id, cache)  # mapper passes no model
        return {"floor_offset": 0}

    monkeypatch.setattr(mapper, "integrate_walk", integrate)
    monkeypatch.setattr(mapper, "consolidate", lambda *_a, **_k: {})
    jobs._process_video(jobs.Job(id="j", building_id=bid, video_id="v", kind="video"), vdir / "s.mp4", None,
                        "gemma3:12b")
    assert [c["model"] for c in fake.chat_calls] == ["gemma3:12b"] * 4
    assert store.read_json(vdir / "walk.json")["model"] == "gemma3:12b"
    assert store.load_map(bid).processed_with == {"model": "gemma3:12b", "prompt": "v6"}


def test_processed_with_is_the_model_used_not_a_later_resolve(bid, monkeypatch):
    vdir = _add_video(bid)
    _fake_video_stages(monkeypatch, bid)
    resolved = iter(["gemma3:4b"] + ["gemma3:12b"] * 10)  # the default changes after analysis
    monkeypatch.setattr(gemma, "resolve_model", lambda: next(resolved))
    monkeypatch.setattr(gemma, "ensure_model", lambda _m: None)
    monkeypatch.setattr(gemma, "analyze_frame", lambda p, c, m: FrameAnalysis())
    monkeypatch.setattr(mapper, "integrate_walk", lambda *_a, **_k: {"floor_offset": 0})
    monkeypatch.setattr(mapper, "consolidate", lambda *_a, **_k: {})
    jobs._process_video(jobs.Job(id="j", building_id=bid, video_id="v", kind="video"), vdir / "s.mp4", None, None)
    assert store.load_map(bid).processed_with["model"] == "gemma3:4b"
    assert store.read_json(vdir / "walk.json")["model"] == "gemma3:4b"


def _done_walk(bid, model, vid="v"):
    vdir = _add_video(bid, vid, status="done")
    (vdir / "analysis.mp4").write_bytes(b"x")
    store.write_json_atomic(vdir / "walk.json", {"video_id": vid, "keyframes": [], "gemma": True, "model": model})
    return vdir


def test_rebuild_reuses_the_stored_model(bid, monkeypatch):
    vdir = _done_walk(bid, "gemma3:12b")
    _fake_video_stages(monkeypatch, bid)
    monkeypatch.setattr(gemma, "resolve_model", lambda: "gemma3:4b")
    monkeypatch.setattr(gemma, "ensure_model", lambda _m: None)
    seen, sessions = [], []

    def analyze(path, cache, model):
        seen.append(model)
        return FrameAnalysis()

    def integrate(*_a, **_k):
        sessions.append(gemma._session.get().model)
        return {"floor_offset": 0}

    def consolidate(*_a, **_k):
        sessions.append(gemma._session.get().model)
        return {}

    monkeypatch.setattr(gemma, "analyze_frame", analyze)
    monkeypatch.setattr(mapper, "integrate_walk", integrate)
    monkeypatch.setattr(mapper, "consolidate", consolidate)
    jobs._rebuild(jobs.Job(id="r", building_id=bid, video_id=None, kind="rebuild"))
    assert seen == ["gemma3:12b"] * 3
    assert sessions == ["gemma3:12b", "gemma3:12b"]
    assert store.read_json(vdir / "walk.json")["model"] == "gemma3:12b"


def test_offline_rebuild_keeps_all_gemma_answers(fake, bid, monkeypatch):
    """Process online with 12b, then rebuild with Ollama down: every frame and same-place answer survives."""
    kfs = _kfs(bid)
    frames, cache = store.frames_dir(bid), gemma.cache_for(store.building_dir(bid))
    labelled = dict(FRAME, suggested_label="Room 9 door", place_type="landmark")
    fake.chat_results = [_resp(labelled)] * 3 + [_resp(SAME)]
    for k in kfs:
        gemma.analyze_frame(frames / k.frame_id, cache, "gemma3:12b")
    gemma.same_place(frames / kfs[0].frame_id, frames / kfs[1].frame_id, cache, model="gemma3:12b")
    n_calls = len(fake.chat_calls)

    _done_walk(bid, "gemma3:12b")
    _fake_video_stages(monkeypatch, bid)
    fake.down = True  # Ollama stopped
    verdicts = []

    def integrate(bundle, vid, kfs2, track, frames_dir, cache2, use_gemma, **_k):
        assert use_gemma and all(k.analysis and k.analysis.suggested_label == "Room 9 door" for k in kfs2)
        verdicts.append(gemma.same_place(frames / kfs2[0].frame_id, frames / kfs2[1].frame_id, cache2))
        with pytest.raises(gemma.GemmaUnavailable):  # a miss: not cached, Ollama down
            gemma.same_place(frames / kfs2[0].frame_id, frames / kfs2[2].frame_id, cache2)
        return {"floor_offset": 0}

    monkeypatch.setattr(mapper, "integrate_walk", integrate)
    monkeypatch.setattr(mapper, "consolidate", lambda *_a, **_k: {})
    job = jobs.Job(id="r", building_id=bid, video_id=None, kind="rebuild")
    jobs._rebuild(job)
    assert verdicts and verdicts[0].same_place and verdicts[0].reason == "same door"
    assert len(fake.chat_calls) == n_calls  # never reached Ollama
    assert any("cached answers only" in line for line in job.log)
    assert any("Gemma analysed 3/3 frames with gemma3:12b" in line for line in job.log)


def test_cache_first_without_explicit_model_uses_configured_model(fake, tmp_path, monkeypatch):
    cache = gemma.GemmaCache(tmp_path / "c.jsonl")
    a, b = _jpg(tmp_path / "a.jpg"), _jpg(tmp_path / "b.jpg", (10, 200, 10))
    fake.chat_results = [_resp(FRAME), _resp(SAME), _resp(SAME)]
    gemma.analyze_frame(a, cache)  # resolves gemma3:4b (= config.GEMMA_MODEL)
    gemma.same_place(a, b, cache)
    gemma.same_stairwell(a, b, cache)
    monkeypatch.setattr(gemma, "_resolved_model", None)  # e.g. after a server restart
    fake.down = True

    def boom(*_a, **_k):
        raise ConnectionError("refused")

    monkeypatch.setattr(gemma, "_client", boom)
    assert gemma.analyze_frame(a, cache).description == "plain hall"
    assert gemma.same_place(b, a, cache).reason == "same door"  # either order
    assert gemma.same_stairwell(a, b, cache).same_place
    with pytest.raises(gemma.GemmaUnavailable):
        gemma.analyze_frame(_jpg(tmp_path / "new.jpg", (1, 2, 3)), cache)


def test_offline_session_does_not_contact_ollama_after_first_failure(fake, tmp_path):
    cache = gemma.GemmaCache(tmp_path / "c.jsonl")
    a, b, c = (_jpg(tmp_path / f"{n}.jpg", (i * 50, 0, 0)) for i, n in enumerate("abc"))
    fake.chat_results = [ConnectionError("x"), ConnectionError("x")]
    with gemma.session("gemma3:4b") as s:
        with pytest.raises(gemma.GemmaUnavailable) as first:
            gemma.same_place(a, b, cache)
        assert not isinstance(first.value, gemma.NotCached) and s.offline
        with pytest.raises(gemma.NotCached):
            gemma.same_place(a, c, cache)
    assert len(fake.chat_calls) == 2  # one call, retried once; the second pair never reached Ollama
    assert gemma.session_active() is False


def test_breaker_does_not_skip_cached_frames(fake, bid, monkeypatch):
    monkeypatch.setattr(gemma, "ensure_model", lambda _m: None)
    kfs = _kfs(bid, n=6)
    frames, cache = store.frames_dir(bid), gemma.cache_for(store.building_dir(bid))
    for k in kfs[4:]:  # frames 4 and 5 were analysed before
        key = gemma._frame_key(frames / k.frame_id)("gemma3:4b")
        cache.put(key, FrameAnalysis(description=f"cached {k.frame_id}").model_dump())
    fake.chat_results = [ConnectionError("down")] * 6  # 3 frames x (1 try + 1 retry)
    job = jobs.Job(id="j", building_id=bid, video_id="v", kind="video")
    run = jobs._analyze_walk(job, kfs, "gemma3:4b")
    assert run.ok and not run.online and run.model == "gemma3:4b"
    assert [k.analysis.description if k.analysis else None for k in kfs] == \
        [None, None, None, None, "cached v_4.jpg", "cached v_5.jpg"]
    assert any("Skipped Gemma on 1 frame(s)" in line for line in job.log)


def test_ollama_down_at_start_uses_cache_for_every_frame(fake, bid, monkeypatch):
    fake.down = True
    kfs = _kfs(bid, n=2)
    frames, cache = store.frames_dir(bid), gemma.cache_for(store.building_dir(bid))
    cache.put(gemma._frame_key(frames / kfs[0].frame_id)(config.GEMMA_MODEL), FrameAnalysis(description="c").model_dump())
    run = jobs._analyze_walk(jobs.Job(id="j", building_id=bid, video_id="v", kind="video"), kfs, None)
    assert run.ok and not run.online and run.model == config.GEMMA_MODEL
    assert kfs[0].analysis.description == "c" and kfs[1].analysis is None


# ---------------------------------------------------------------- A. vision-capable fallback
def test_resolve_model_fallback_needs_vision(fake, monkeypatch):
    monkeypatch.setattr(config, "GEMMA_MODEL", "gemma3:4b")
    monkeypatch.setattr(config, "GEMMA_FALLBACK_MODEL", "gemma3:12b")
    fake.models = ["gemma3:270m", "gemma3:1b", "gemma3n:e4b"]
    fake.caps = {"gemma3:270m": ["completion"], "gemma3:1b": ["completion", "tools"],
                 "gemma3n:e4b": ["completion", "vision"]}
    assert gemma.resolve_model() == "gemma3n:e4b"


def test_resolve_model_tolerates_missing_capabilities(fake, monkeypatch):
    monkeypatch.setattr(config, "GEMMA_MODEL", "x")
    monkeypatch.setattr(config, "GEMMA_FALLBACK_MODEL", "y")
    fake.models = ["gemma3:270m", "gemma-old:7b"]
    fake.caps = {"gemma3:270m": ["completion"]}  # gemma-old: older Ollama, no capabilities field
    assert gemma.resolve_model() == "gemma-old:7b"


def test_resolve_model_no_vision_gemma(fake, monkeypatch):
    monkeypatch.setattr(config, "GEMMA_MODEL", "x")
    monkeypatch.setattr(config, "GEMMA_FALLBACK_MODEL", "y")
    fake.models = ["gemma3:270m"]
    fake.caps = {"gemma3:270m": ["completion"]}
    with pytest.raises(gemma.GemmaUnavailable):
        gemma.resolve_model()


# ---------------------------------------------------------------- B. prompts / schema / images
def test_evidence_fields_come_before_the_verdict():
    fields = list(GemmaFrame.model_fields)
    assert fields.index("texts") < fields.index("has_sign") and fields.index("room_numbers") < fields.index("has_sign")
    old = {"has_sign": True, "texts": ["EXIT"], "room_numbers": [], "suggested_label": "Exit", "place_type": "exit",
           "stairs": "none", "floor_hint": "", "description": "exit door", "ocr": ["EXIT"], "grounded": True}
    assert FrameAnalysis.model_validate(old).model_dump() == old  # same keys and values for readers
    sp = list(SamePlace.model_fields)
    assert sp.index("image1_sign_text") < sp.index("shared_details") < sp.index("same_place")
    SamePlace(same_place=True, confidence=1, reason="x")  # old construction still valid


def test_schema_sent_to_ollama_requires_every_field(fake):
    fake.chat_results = [_resp(SAME)]
    gemma._chat("m", "p", [], SamePlace, 100)
    fmt = fake.chat_calls[0]["format"]
    assert set(fmt["required"]) == set(fmt["properties"]) >= {"image1_sign_text", "shared_details", "same_place"}


def test_same_place_prompts_label_images_and_tolerate_viewpoint():
    for p in (gemma.SAME_PLACE_PROMPT, gemma.SAME_STAIRWELL_PROMPT):
        assert "Image 1" in p and "Image 2" in p and "color" in p
    assert "do NOT mean a different place" in gemma.SAME_PLACE_PROMPT
    assert "same sign text" in gemma.SAME_PLACE_PROMPT.lower()


def test_compare_sends_images_in_hash_order(fake, tmp_path):
    cache = gemma.GemmaCache(tmp_path / "c.jsonl")
    a, b = _jpg(tmp_path / "a.jpg"), _jpg(tmp_path / "b.jpg", (10, 200, 10))
    fake.chat_results = [_resp(SAME)]
    gemma.same_place(b, a, cache)
    first = Path(sorted([a, b], key=gemma._sha)[0])
    assert fake.chat_calls[0]["messages"][0]["images"][0] == gemma._encode(gemma._open(first))


def _exif_rotated(path: Path) -> Path:
    img = Image.new("RGB", (80, 40), (200, 0, 0))
    exif = Image.Exif()
    exif[0x0112] = 6  # orientation: rotate 90 CW to display
    img.save(path, format="JPEG", exif=exif.tobytes())
    return path


def test_exif_orientation_6_becomes_portrait(tmp_path):
    p = _exif_rotated(tmp_path / "phone.jpg")
    assert Image.open(p).size == (80, 40)  # stored landscape
    assert Image.open(io.BytesIO(gemma._encode(Image.open(p)))).size == (40, 80)
    full, left, right = (Image.open(io.BytesIO(v)) for v in gemma._views(p))
    assert full.size == (40, 80) and left.size[1] == right.size[1] == 80 and left.size[0] < 40


# ---------------------------------------------------------------- B. crop OCR
def test_crop_ocr_unions_lines(tmp_path, monkeypatch):
    p = _jpg(tmp_path / "wide.jpg", size=(1000, 500))
    monkeypatch.setattr(ocr, "available", lambda: True)
    calls = []

    def observations(img):
        calls.append(img.size)
        if len(calls) == 1:  # full view: misses the small plaque, half-reads another
            return [("EXIT", 0.4, 0.5), ("ROOM 12", 0.1, 0.2)]
        if len(calls) == 2:  # left crop (upscaled): the full room number, a word cut at the right edge
            return [("ROOM 121", 0.2, 0.4), ("EXIT", 0.75, 0.9), ("CHEMIS", 0.9, 1.0)]
        return [("RM 118", 0.85, 0.95), ("TRY", 0.0, 0.05)]  # right crop: small corner plaque, cut word

    monkeypatch.setattr(ocr, "_observations", observations)
    assert ocr.read_text_with_crops(p) == ["EXIT", "ROOM 121", "RM 118"]
    assert calls == [(1000, 500), (1100, 1000), (1100, 1000)]


def test_analyze_uses_crop_ocr(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "read_text_with_crops", lambda p: ["RM 118"])
    fake.chat_results = [_resp(dict(FRAME, texts=["RM 118"], room_numbers=["118"], place_type="room",
                                    suggested_label="Room 118"))]
    r = gemma.analyze_frame(_jpg(tmp_path / "a.jpg"), gemma.GemmaCache(tmp_path / "c.jsonl"), "m")
    assert r.ocr == ["RM 118"] and r.room_numbers == ["118"] and "- RM 118" in fake.chat_calls[0]["messages"][0]["content"]


# ---------------------------------------------------------------- C. poster leakage / floor hint
POSTER = ["BAKE SALE", "Oct 31 5 PM $12", "Call 555 0147", "Table 77 Hall 9"]


def test_poster_numbers_are_not_rooms():
    a = FrameAnalysis(texts=POSTER, room_numbers=["77", "9", "555"], has_sign=True,
                      suggested_label="Table 77 Hall 9", place_type="landmark", description="bake sale poster")
    g = gemma._ground(a, POSTER)
    assert g.room_numbers == []
    assert "Call 555 0147" not in g.texts and "BAKE SALE" in g.texts
    assert g.suggested_label == "Bake sale poster"


def test_room_numbers_kept_at_doors_and_on_plaques():
    room = gemma._ground(FrameAnalysis(texts=["ROOM 121"], room_numbers=["121"], place_type="room",
                                       suggested_label="Room 121"), ["ROOM 121"])
    assert room.room_numbers == ["121"] and room.suggested_label == "Room 121"
    hall = gemma._ground(FrameAnalysis(room_numbers=["214"], place_type="hallway", suggested_label="Lab 214"),
                         ["214", "CHEMISTRY LAB", "Table 77 Hall 9"])
    assert hall.room_numbers == ["214"] and hall.suggested_label == "Lab 214"
    hall2 = gemma._ground(FrameAnalysis(room_numbers=["77"], place_type="hallway", suggested_label="Table 77"),
                          ["Table 77 Hall 9"])
    assert hall2.room_numbers == [] and "77" not in hall2.suggested_label
    phone_room = gemma._ground(FrameAnalysis(room_numbers=["555"], place_type="room"), ["Call 555 0147"])
    assert phone_room.room_numbers == []


def test_long_sentences_dropped_from_texts():
    line = "Please keep this door closed at all times. Thank you!"
    g = gemma._ground(FrameAnalysis(texts=[line, "EXIT"], place_type="exit"), [line, "EXIT"])
    assert g.texts == ["EXIT"]


def test_floor_hint_derived_from_ocr_when_empty():
    g = gemma._ground(FrameAnalysis(texts=["LEVEL 2"], floor_hint="", place_type="hallway",
                                    suggested_label="Level 2"), ["LEVEL 2"])
    assert g.floor_hint == "2" and g.suggested_label == "Level 2"
    bad = gemma._ground(FrameAnalysis(floor_hint="3", place_type="hallway"), ["LEVEL 2"])
    assert bad.floor_hint == "2"  # ungrounded hint replaced by what the sign says


@pytest.mark.parametrize("lines, floor", [
    (["LEVEL 2"], "2"), (["2ND FLOOR"], "2"), (["SECOND FLOOR"], "2"), (["L3"], "3"), (["FLOOR 1 ROOM 121"], "1"),
    (["ROOM 121"], ""), (["LEVEL 2", "LEVEL 3"], ""), (["LEVEL 2", "2ND FLOOR"], "2"), (["LEVEL -1"], ""),
    (["LOWER LEVEL 1"], ""), (["LEVEL 22"], ""),  # above MAX_FLOOR
    (["Join us for the party on the 3RD FLOOR after the talk tonight"], ""), ([], ""),
])
def test_floor_from_lines(lines, floor):
    assert ocr.floor_from_lines(lines) == floor


# ---------------------------------------------------------------- C. grounding gaps
@pytest.mark.parametrize("cand, lines, ok", [
    ("121", ["1 21"], False), ("121", ["1-21"], False), ("121", ["ROOM 1 21"], False),
    ("ROOM 121", ["ROOM 1 21"], False), ("ST", ["1ST FLOOR"], False), ("ND", ["2ND"], False),
    ("A", ["121A"], False), ("121A", ["ROOM 121A"], True), ("121", ["ROOM121"], True), ("ROOM", ["ROOM121"], True),
    ("ST", ["ST JOHN"], True), ("1ST FLOOR", ["1ST FLOOR"], True), ("B12", ["B 12"], True),
])
def test_grounding_gaps(cand, lines, ok):
    assert ocr.grounded(cand, lines) is ok


@pytest.mark.parametrize("hint, lines, ok", [
    ("-1", ["1ST FLOOR"], False), ("-1", ["LEVEL -1"], False), ("1", ["LEVEL -1"], False),
    ("1", ["LEVEL −1"], False), ("B1", ["1ST FLOOR"], False), ("basement", ["1ST FLOOR"], False),
    ("1", ["BASEMENT LEVEL 1"], False), ("1", ["1ST FLOOR"], True), ("2", ["LEVEL 2"], True),
])
def test_negative_and_basement_floors(hint, lines, ok):
    assert ocr.floor_grounded(hint, lines) is ok


# ---------------------------------------------------------------- B. OCR room-number guard
def test_different_plaque_numbers_never_ask_gemma(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(ocr, "available", lambda: True)
    reads = {"a.jpg": ["ROOM 121"], "b.jpg": ["ROOM 305"], "c.jpg": ["Room-121"]}
    monkeypatch.setattr(ocr, "read_text_with_crops", lambda p: reads[Path(p).name])
    cache = gemma.GemmaCache(tmp_path / "c.jsonl")
    a, b, c = (_jpg(tmp_path / f"{n}.jpg", (i * 60, 9, 9)) for i, n in enumerate("abc"))
    v = gemma.same_place(a, b, cache, model="m")
    assert not v.same_place and gemma.ocr_conflict(v) and fake.chat_calls == []
    fake.chat_results = [_resp(SAME)]
    assert gemma.same_place(a, c, cache, model="m").same_place  # same number: Gemma decides
    assert len(fake.chat_calls) == 1
    assert not gemma.ocr_conflict(SamePlace(same_place=False, confidence=0.9, reason="wall colour"))


# ---------------------------------------------------------------- B. mapper: Gemma "False" vs strong geometry
def _match_setup(monkeypatch, tmp_path, inliers, verdict):
    monkeypatch.setattr(mapper, "orb_match", lambda a, b: (inliers, 0.0))
    monkeypatch.setattr(gemma, "same_place", lambda *_a, **_k: verdict)
    bundle = MapBundle(building_id="b", nodes=[Node(id="n001", label="Room", frame_id="n.jpg")])
    kf = Keyframe(frame_id="w.jpg", video_id="w", t=100, kind="dwell", heading=0)
    wn = mapper.WalkNode(kf=kf, t_arrive=100, t_depart=100, analysis=FrameAnalysis())
    return mapper._find_match(wn, bundle, tmp_path, None, True, set(), {}, lambda _m: None)


def test_gemma_false_does_not_veto_strong_orb(monkeypatch, tmp_path):
    m = _match_setup(monkeypatch, tmp_path, config.ORB_STRONG_INLIERS + 10,
                     SamePlace(same_place=False, confidence=0.7, reason="wall colour differs"))
    assert m is not None and m.signals["note"] == "geometry overrode Gemma" and not m.auto


def test_gemma_false_still_vetoes_weak_orb(monkeypatch, tmp_path):
    assert _match_setup(monkeypatch, tmp_path, config.ORB_WEAK_INLIERS + 5,
                        SamePlace(same_place=False, confidence=0.7, reason="different")) is None


def test_ocr_plaque_conflict_vetoes_strong_orb(monkeypatch, tmp_path):
    v = SamePlace(same_place=False, confidence=0.95, reason=f"{gemma.OCR_CONFLICT}: 121 in Image 1, 305 in Image 2")
    assert _match_setup(monkeypatch, tmp_path, config.ORB_STRONG_INLIERS + 10, v) is None


# ---------------------------------------------------------------- C. manual frame never 500s
@pytest.mark.parametrize("err", [RuntimeError("boom"), OSError("cannot identify image file"), ValueError("bad")])
def test_manual_frame_analysis_error_is_not_a_500(tmp_path, monkeypatch, err):
    from app.main import app
    monkeypatch.setattr(config, "BUILDINGS_DIR", tmp_path / "buildings")

    def broken(*_a, **_k):
        raise err

    monkeypatch.setattr(gemma, "analyze_frame", broken)
    b = store.create_building("Manual Hall").building_id
    buf = io.BytesIO()
    Image.new("RGB", (32, 24), (200, 30, 30)).save(buf, "JPEG")
    r = TestClient(app).post(f"/api/buildings/{b}/frames/manual", files={"image": ("p.jpg", buf.getvalue(), "image/jpeg")},
                             data={"label": ""})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["node"]["label"] == "Manual photo" and "warning" in body
    assert any(n["id"] == body["node"]["id"] for n in store.load_map(b).model_dump()["nodes"])
