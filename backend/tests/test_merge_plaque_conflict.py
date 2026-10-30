"""Merge safety: two places with different room / plaque numbers never merge, whatever the ORB strength
(sentinel fix round 1, cases (c)); directional range signs give no room numbers; room numbers are canonical
('RM 118' -> '118'). Ollama, ORB and OCR are faked; no server is needed."""
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from app import config, gemma, mapper, matching, ocr
from app.schemas import FrameAnalysis, Keyframe, MapBundle, Node, SamePlace, VideoInfo

S, W = config.ORB_STRONG_INLIERS, config.ORB_WEAK_INLIERS
CONFLICT = SamePlace(same_place=False, confidence=0.95, reason=f"{gemma.OCR_CONFLICT}: 121 vs 305")
TRUE = SamePlace(same_place=True, confidence=0.99, reason="same")
FALSE = SamePlace(same_place=False, confidence=0.9, reason="looks different")


def _jpg(path: Path, color=(120, 120, 120)) -> Path:
    Image.new("RGB", (64, 48), color).save(path, format="JPEG")
    return path


# ---------------------------------------------------------------- loop closure (_find_match)
def fm(monkeypatch, tmp_path, inliers, verdict, text_hit=False, node_texts=None, walk_texts=None, use_gemma=True):
    monkeypatch.setattr(ocr, "available", lambda: False)
    monkeypatch.setattr(mapper, "orb_match", lambda a, b: (inliers, 0.0))
    monkeypatch.setattr(gemma, "same_place", lambda *a, **k: verdict)
    nt = node_texts if node_texts is not None else (["ROOM 121"] if text_hit else [])
    wt = walk_texts if walk_texts is not None else (["ROOM 121"] if text_hit else [])
    n = Node(id="n001", label="Room", frame_id="n.jpg", texts=nt, sign_texts=nt, floor=1, floor_source="user")
    bundle = MapBundle(building_id="b", nodes=[n])
    kf = Keyframe(frame_id="w.jpg", video_id="w", t=100, kind="dwell", heading=0)
    wn = mapper.WalkNode(kf=kf, t_arrive=100, t_depart=100, analysis=FrameAnalysis(texts=wt))
    return mapper._find_match(wn, bundle, tmp_path, None, use_gemma, set(), {}, lambda _m: None)


@pytest.mark.parametrize("inl", [S, S + 10, int(1.5 * S), 3 * S - 1, 3 * S, 5 * S])
def test_ocr_conflict_never_merges_find_match(monkeypatch, tmp_path, inl):
    assert fm(monkeypatch, tmp_path, inl, CONFLICT, text_hit=True) is None


@pytest.mark.parametrize("inl", [S, int(1.5 * S), 3 * S, 5 * S])
def test_model_false_never_vetoes_strong_orb(monkeypatch, tmp_path, inl):
    """Regression guard: a plain Gemma 'different' (not an OCR conflict) still loses to strong geometry."""
    assert fm(monkeypatch, tmp_path, inl, FALSE) is not None


@pytest.mark.parametrize("inl", [W, 3 * S, 10 * S])
@pytest.mark.parametrize("verdict", [TRUE, None])
def test_find_match_different_room_ids_skipped(monkeypatch, tmp_path, inl, verdict):
    """Own sign texts with disjoint room numbers: never a candidate, even with Gemma 'same' and huge ORB."""
    assert fm(monkeypatch, tmp_path, inl, verdict, node_texts=["ROOM 121"], walk_texts=["ROOM 305"]) is None


def test_find_match_same_room_normalized_forms(monkeypatch, tmp_path):
    m = fm(monkeypatch, tmp_path, 3 * S, TRUE, node_texts=["ROOM-118"], walk_texts=["RM 118"])
    assert m is not None and m.auto and m.signals["text"]


def test_find_match_directional_range_is_no_room(monkeypatch, tmp_path):
    """A walk frame with only a range sign does not 'conflict' with plaque 121 (it is not a room)."""
    m = fm(monkeypatch, tmp_path, 3 * S, TRUE, node_texts=["ROOM 121"], walk_texts=["ROOMS 120-130 →"])
    assert m is not None


@pytest.mark.parametrize("inl", [S, 3 * S, 5 * S])
def test_find_match_ocr_only_conflict_without_gemma(monkeypatch, tmp_path, inl):
    """Gemma off (no verdict): OCR still reads the plaques of both photos and keeps them apart."""
    monkeypatch.setattr(mapper, "orb_match", lambda a, b: (inl, 0.0))
    for name, color in (("n.jpg", (90, 90, 90)), ("w.jpg", (200, 200, 200))):
        _jpg(tmp_path / name, color)
    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "read_text_with_crops", lambda p: ["ROOM 121"] if p.name == "n.jpg" else ["ROOM 305"])
    n = Node(id="n001", label="Room", frame_id="n.jpg", floor=1, floor_source="user")
    kf = Keyframe(frame_id="w.jpg", video_id="w", t=100, kind="dwell", heading=0)
    wn = mapper.WalkNode(kf=kf, t_arrive=100, t_depart=100, analysis=FrameAnalysis())
    assert mapper._find_match(wn, MapBundle(building_id="b", nodes=[n]), tmp_path, None, False, set(), {},
                              lambda _m: None) is None


# ---------------------------------------------------------------- consolidate
def cons(monkeypatch, tmp_path, inliers, verdict, rooms=("ROOM 121", "ROOM 305"), floors=(1, 1), gemma_on=True):
    monkeypatch.setattr(ocr, "available", lambda: False)
    monkeypatch.setattr(mapper, "orb_match", lambda a, b: (inliers, 0.0))
    calls = []

    def sp(*a, **k):
        calls.append(1)
        return verdict
    monkeypatch.setattr(gemma, "same_place", sp)
    nodes = [Node(id=f"n00{i + 1}", label=f"P{i}", frame_id=f"{i}.jpg", texts=[rooms[i]] if rooms[i] else [],
                  sign_texts=[rooms[i]] if rooms[i] else [], floor=floors[i], floor_source="user", video_id=f"v{i}")
             for i in range(2)]
    b = MapBundle(building_id="b", nodes=nodes, videos=[VideoInfo(id="v0", filename="a", status="done"),
                                                        VideoInfo(id="v1", filename="b", status="done")])
    mapper.consolidate(b, tmp_path, None, gemma_on, log=lambda *_: None)
    return b, calls


@pytest.mark.parametrize("inl", [W, S, int(1.5 * S) - 1, int(1.5 * S), 3 * S, 6 * S])
@pytest.mark.parametrize("verdict", [CONFLICT, TRUE, FALSE])
def test_consolidate_different_plaques_never_merge(monkeypatch, tmp_path, inl, verdict):
    b, _ = cons(monkeypatch, tmp_path, inl, verdict)
    assert len(b.nodes) == 2 and not b.suggested_merges, f"merged/suggested with inliers={inl}"


@pytest.mark.parametrize("inl", [int(1.5 * S), 3 * S])
def test_consolidate_different_plaques_strong_orb_gemma_off(monkeypatch, tmp_path, inl):
    b, calls = cons(monkeypatch, tmp_path, inl, CONFLICT, gemma_on=False)
    assert len(b.nodes) == 2 and not calls


@pytest.mark.parametrize("inl", [W, S])
def test_consolidate_conflict_verdict_without_sign_texts(monkeypatch, tmp_path, inl):
    """No room numbers on the nodes, but OCR (via the verdict) reads different plaques: no merge, no suggestion."""
    b, _ = cons(monkeypatch, tmp_path, inl, CONFLICT, rooms=("", ""))
    assert len(b.nodes) == 2 and not b.suggested_merges


@pytest.mark.parametrize("inl", [int(1.5 * S), 3 * S, 6 * S])
def test_consolidate_strong_orb_ocr_conflict_no_sign_texts(monkeypatch, tmp_path, inl):
    """Strong ORB skips Gemma; the OCR plaque check still runs on the photos and blocks the merge."""
    monkeypatch.setattr(mapper, "orb_match", lambda a, b: (inl, 0.0))
    monkeypatch.setattr(gemma, "same_place", lambda *a, **k: pytest.fail("Gemma asked"))
    for i, color in enumerate(((90, 90, 90), (200, 200, 200))):
        _jpg(tmp_path / f"{i}.jpg", color)
    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "read_text_with_crops", lambda p: ["ROOM 121"] if p.name == "0.jpg" else ["RM 305"])
    nodes = [Node(id=f"n00{i + 1}", label=f"P{i}", frame_id=f"{i}.jpg", floor=1, floor_source="user",
                  video_id=f"v{i}") for i in range(2)]
    b = MapBundle(building_id="b", nodes=nodes, videos=[VideoInfo(id="v0", filename="a", status="done"),
                                                        VideoInfo(id="v1", filename="b", status="done")])
    mapper.consolidate(b, tmp_path, None, True, log=lambda *_: None)
    assert len(b.nodes) == 2


def test_consolidate_false_strong_still_merges_without_plaques(monkeypatch, tmp_path):
    b, _ = cons(monkeypatch, tmp_path, int(1.5 * S), FALSE, rooms=("", ""))
    assert len(b.nodes) == 1
    b, _ = cons(monkeypatch, tmp_path, S + 5, FALSE, rooms=("", ""))
    assert len(b.nodes) == 2 and b.suggested_merges and "disagreed" in b.suggested_merges[0].reason


def test_consolidate_same_room_any_spelling_merges(monkeypatch, tmp_path):
    b, _ = cons(monkeypatch, tmp_path, 0, None, rooms=("ROOM-118", "RM 118"))
    assert len(b.nodes) == 1


def test_consolidate_range_signs_never_identity(monkeypatch, tmp_path):
    """Two different hallway spots showing 'ROOMS 120 - 130' are not 'same room 120/130'."""
    b, _ = cons(monkeypatch, tmp_path, 0, None, rooms=("ROOMS 120 - 130", "120 TO 130"))
    assert len(b.nodes) == 2


# ---------------------------------------------------------------- text helpers
@pytest.mark.parametrize("raw,want", [("RM 118", "118"), ("Room 118", "118"), ("ROOM-118", "118"), ("RM118", "118"),
                                      ("rm.118a", "118A"), ("Room 118A", "118A"), ("Room No. 118", "118"),
                                      ("B204", "B204"), ("LAB 214", "214"), ("2-105", "2-105"),
                                      ("CHEM LAB", "CHEM LAB")])
def test_room_id(raw, want):
    assert matching.room_id(raw) == want


@pytest.mark.parametrize("text,is_dir", [("ROOMS 120-130 →", True), ("120-130", True), ("120 – 130", True),
                                         ("ROOMS 120 TO 130", True), ("ROOM 121 →", True), ("ROOMS 101, 103", True),
                                         ("ROOM 121", False), ("555-0147", False), ("2-105", False), ("EXIT", False)])
def test_directional(text, is_dir):
    assert matching.directional(text) is is_dir
    if is_dir:
        assert not matching.room_ids([text]) and not ocr.plaque_like(text) and not ocr.plaque_numbers([text])


def test_room_ids_consistent_across_spellings():
    assert matching.room_ids(["ROOM-118"]) == matching.room_ids(["RM118"]) == matching.room_ids(["Room 118"]) == {"118"}
    assert matching.room_ids(["Room 118A"]) == {"118A"}


# ---------------------------------------------------------------- frame analysis (fake Ollama)
FRAME = {"texts": [], "room_numbers": [], "has_sign": False, "suggested_label": "Hallway", "place_type": "hallway",
         "stairs": "none", "floor_hint": "", "description": "plain hall"}


class FakeOllama:
    def __init__(self):
        self.chat_results: list = []

    def chat(self, **kw):
        return SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.chat_results.pop(0))),
                               done_reason="stop")


@pytest.fixture
def fake(monkeypatch):
    client = FakeOllama()
    monkeypatch.setattr(gemma, "_client", lambda *_a, **_k: client)
    monkeypatch.setattr(gemma, "_resolved_model", None)
    monkeypatch.setattr(ocr, "available", lambda: False)
    return client


@pytest.mark.parametrize("line", ["ROOMS 120-130 →", "ROOMS 120-130 •", "120-130", "ROOMS 120 TO 130"])
def test_range_sign_grounded_no_rooms_not_room(fake, tmp_path, monkeypatch, line):
    """gemma3:4b really answers place_type 'room' + room_numbers 120/130 (or 120..130) for these signs."""
    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "read_text_with_crops", lambda p: [line])
    fake.chat_results = [dict(FRAME, texts=[line], room_numbers=["120", "125", "130", "120-130"], place_type="room",
                              has_sign=True, suggested_label="Rooms 120-130",
                              description="sign for rooms 120 to 130")]
    r = gemma.analyze_frame(_jpg(tmp_path / "a.jpg"), gemma.GemmaCache(tmp_path / "c.jsonl"), "m")
    assert r.room_numbers == [] and r.place_type == "hallway" and r.texts == [line] and r.has_sign
    assert not re.match(r"rooms? \d", r.suggested_label.lower())  # not named like a room ('Rooms 120-130')
    assert "120" in r.description  # real numbers on the sign are fine in a landmark phrase


def test_directional_line_next_to_real_plaque(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "read_text_with_crops", lambda p: ["ROOMS 120-130 →", "RM 118"])
    fake.chat_results = [dict(FRAME, texts=["ROOMS 120-130 →", "RM 118"], room_numbers=["120", "RM 118"],
                              place_type="room", suggested_label="Room 118")]
    r = gemma.analyze_frame(_jpg(tmp_path / "a.jpg"), gemma.GemmaCache(tmp_path / "c.jsonl"), "m")
    assert r.room_numbers == ["118"] and r.place_type == "room" and r.suggested_label == "Room 118"


def test_hallway_arrow_sign_gives_no_room(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "read_text_with_crops", lambda p: ["ROOM 121 →"])
    fake.chat_results = [dict(FRAME, texts=["ROOM 121 →"], room_numbers=["121"])]
    r = gemma.analyze_frame(_jpg(tmp_path / "a.jpg"), gemma.GemmaCache(tmp_path / "c.jsonl"), "m")
    assert r.room_numbers == [] and r.texts == ["ROOM 121 →"]


def test_range_sign_ungrounded(fake, tmp_path):
    """OCR off: Gemma's own texts tell the numbers are only on a directional sign."""
    fake.chat_results = [dict(FRAME, texts=["ROOMS 120-130 →"], room_numbers=["120", "130"], place_type="room",
                              has_sign=True, suggested_label="Rooms 120-130")]
    r = gemma.analyze_frame(_jpg(tmp_path / "a.jpg"), gemma.GemmaCache(tmp_path / "c.jsonl"), "m")
    assert r.room_numbers == [] and r.place_type == "hallway" and r.texts == ["ROOMS 120-130 →"]


@pytest.mark.parametrize("ocr_line,gemma_rn,want,label", [
    ("RM 118", "RM 118", "118", "Room 118"), ("ROOM-118", "ROOM-118", "118", "Room 118"),
    ("Room 118", "Room 118", "118", "Room 118"), ("RM118", "118", "118", "Room 118"),
    ("ROOM 118A", "Room 118A", "118A", "Room 118A")])
def test_room_numbers_normalized_and_grounded(fake, tmp_path, monkeypatch, ocr_line, gemma_rn, want, label):
    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "read_text_with_crops", lambda p: [ocr_line])
    fake.chat_results = [dict(FRAME, texts=[ocr_line], room_numbers=[gemma_rn], place_type="room",
                              suggested_label=label)]
    r = gemma.analyze_frame(_jpg(tmp_path / "a.jpg"), gemma.GemmaCache(tmp_path / "c.jsonl"), "m")
    assert r.room_numbers == [want] and r.suggested_label == label and r.texts == [ocr_line]


def test_invented_label_rebuilt_from_normalized_number(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "read_text_with_crops", lambda p: ["RM 132"])
    fake.chat_results = [dict(FRAME, texts=["RM 132"], room_numbers=["RM 132"], place_type="room",
                              suggested_label="Room 182")]  # misread label
    r = gemma.analyze_frame(_jpg(tmp_path / "a.jpg"), gemma.GemmaCache(tmp_path / "c.jsonl"), "m")
    assert r.room_numbers == ["132"] and r.suggested_label == "Room 132"


def test_cached_old_answer_tidied_on_read(fake, tmp_path):
    """Answers cached before normalization ('RM 118', range numbers) read back canonical, without Ollama."""
    p = _jpg(tmp_path / "a.jpg")
    cache = gemma.GemmaCache(tmp_path / "c.jsonl")
    cache.put(gemma._frame_key(p)("m"), FrameAnalysis(
        texts=["RM 118", "ROOMS 120-130 →"], room_numbers=["RM 118", "120"], place_type="room",
        suggested_label="RM 118", ocr=["RM 118", "ROOMS 120-130 →"], grounded=True).model_dump())
    r = gemma.analyze_frame(p, cache, "m")
    assert r.room_numbers == ["118"] and r.place_type == "room"
    assert gemma.cached_analysis(p, cache, "m").room_numbers == ["118"]


# ---------------------------------------------------------------- text hits come from the place's own signs only
@pytest.mark.parametrize("sign", ["120-130", "ROOMS 120-130 →", "ROOMS 120 TO 130", "ROOM 121 →"])
def test_directional_sign_is_no_text_hit(monkeypatch, tmp_path, sign):
    """Two hallway spots showing the same directional sign are not 'same room number' at weak ORB."""
    assert not matching.place_tokens([sign])
    m = fm(monkeypatch, tmp_path, W, None, node_texts=[sign], walk_texts=[sign], use_gemma=False)
    assert m is None or not m.auto


@pytest.mark.parametrize("text", ["555-0147", "Call 555 0147", "(858) 534-1234", "555.0147"])
def test_phone_number_is_no_token(monkeypatch, tmp_path, text):
    assert not matching.place_tokens([text])
    m = fm(monkeypatch, tmp_path, W, None, node_texts=[text], walk_texts=[text], use_gemma=False)
    assert m is None or not m.signals["text"]


def test_phone_number_next_to_plaque_keeps_room():
    assert matching.place_tokens(["ROOM 118", "Call 555-0147"]) == {"118"}


@pytest.mark.parametrize("a,b", [("ROOM 121", "ROOM 121"), ("RM 118", "ROOM-118"), ("RM118", "Room 118")])
def test_same_real_plaque_still_text_hit(monkeypatch, tmp_path, a, b):
    m = fm(monkeypatch, tmp_path, W, None, node_texts=[a, "ROOMS 120-130 →"], walk_texts=[b], use_gemma=False)
    assert m is not None and m.auto and m.signals["text"]
