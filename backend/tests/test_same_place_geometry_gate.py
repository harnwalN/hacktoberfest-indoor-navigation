"""Gemma's 'same place' alone never merges two spots: it needs strong ORB or a shared door-plaque number
(mapper._gemma_backed), in loop closure and in both consolidation steps. Floor signs ('LEVEL2') are no identity
text, look-alike spots of one walk seconds apart stay apart, and cached frame analyses grounded on OCR lines
from an older OCR revision are re-read (the invented 'Room 110' of a 110A plaque cut at the image edge).

The three live-eval failures these cover (gemma3:4b, eval/scenarios):
  confusers  - west@16s (SEMINAR poster) loop-closed onto west@0s, 7.5 m away: Gemma 'signs are identical', 25 ORB
               inliers; then consolidation joined west@16s and west@26s (same poster, 9 m apart, 10 s apart)
  two_floor  - down@12s (turn by the 214A door) loop-closed onto the top of the stairs ('LEVEL 2'), 5.5 m away:
               Gemma 'same signs (LEVEL 2 and 2)', 22 ORB inliers
  small_loop - cw_reverse frame 10: OCR read '110' of a 110A plaque cut at the right edge; the cached answer kept
               the old OCR lines, so 'Room 110' was grounded although the current OCR drops that line
Ollama, ORB and OCR are faked; no server is needed."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from app import config, gemma, mapper, matching, ocr
from app.schemas import FrameAnalysis, Keyframe, MapBundle, Node, SamePlace, VideoInfo

S, W = config.ORB_STRONG_INLIERS, config.ORB_WEAK_INLIERS
IDENTICAL = SamePlace(same_place=True, confidence=1.0,
                      reason="The signs are identical in both images, and the overall layout is consistent.")
DIFFERENT = SamePlace(same_place=False, confidence=0.9, reason="looks different")


def _jpg(path: Path, color=(120, 120, 120)) -> Path:
    Image.new("RGB", (64, 48), color).save(path, format="JPEG")
    return path


# ---------------------------------------------------------------- loop closure (_find_match)
def find(monkeypatch, tmp_path, inliers, verdict, node_texts=(), walk_texts=(), extra_texts=(), use_gemma=True):
    monkeypatch.setattr(ocr, "available", lambda: False)
    monkeypatch.setattr(mapper, "orb_match", lambda a, b: (inliers, 0.0))
    monkeypatch.setattr(gemma, "same_place", lambda *a, **k: verdict)
    nt = list(node_texts)
    node = Node(id="n001", label="Spot", frame_id="n.jpg", texts=nt, sign_texts=nt, floor=2, floor_source="user")
    kf = Keyframe(frame_id="w.jpg", video_id="w", t=100, kind="turn_start", heading=0)
    wn = mapper.WalkNode(kf=kf, t_arrive=100, t_depart=100, analysis=FrameAnalysis(texts=list(walk_texts)),
                         extra_texts=list(extra_texts))
    return mapper._find_match(wn, MapBundle(building_id="b", nodes=[node]), tmp_path, None, use_gemma, set(), {},
                              lambda _m: None)


@pytest.mark.parametrize("inl", [W, 25, S - 1])
def test_confusers_gemma_identical_poster_spots_weak_orb_not_merged(monkeypatch, tmp_path, inl):
    """confusers west@16s vs west@0s: only poster text in common, Gemma sure, ORB below the strong threshold."""
    m = find(monkeypatch, tmp_path, inl, IDENTICAL, node_texts=["SEMINAR"], walk_texts=["SEMINAR"])
    assert m is not None and not m.auto  # left for review, not merged
    assert "Gemma alone" in m.signals["note"]


@pytest.mark.parametrize("inl", [W, 22, S - 1])
def test_two_floor_turn_by_door_vs_stair_top_level_sign_not_merged(monkeypatch, tmp_path, inl):
    """two_floor down@12s: the turn frame shows no sign itself (214A was absorbed from the glance next to it), the
    known place is the top of the stairs with a 'LEVEL 2' floor sign. Gemma 'same signs', weak ORB: not merged."""
    m = find(monkeypatch, tmp_path, inl, IDENTICAL, node_texts=["LEVEL 2"], walk_texts=[], extra_texts=["214A"])
    assert m is not None and not m.auto


@pytest.mark.parametrize("inl", [S, int(1.5 * S), 3 * S - 1])
def test_gemma_same_with_strong_orb_still_merges(monkeypatch, tmp_path, inl):
    m = find(monkeypatch, tmp_path, inl, IDENTICAL)
    assert m is not None and m.auto


@pytest.mark.parametrize("inl", [W, S - 1])
def test_gemma_same_with_shared_plaque_and_weak_orb_merges(monkeypatch, tmp_path, inl):
    m = find(monkeypatch, tmp_path, inl, IDENTICAL, node_texts=["ROOM 214A"], walk_texts=["214A"])
    assert m is not None and m.auto


def test_shared_plaque_still_needs_some_geometry(monkeypatch, tmp_path):
    m = find(monkeypatch, tmp_path, W - 1, IDENTICAL, node_texts=["ROOM 214A"], walk_texts=["214A"])
    assert m is not None and not m.auto  # the same plaque seen from far away: consolidation decides


def test_gemma_low_confidence_same_with_strong_orb_not_auto(monkeypatch, tmp_path):
    unsure = SamePlace(same_place=True, confidence=0.5, reason="maybe")
    m = find(monkeypatch, tmp_path, S, unsure)
    assert m is not None and not m.auto


@pytest.mark.parametrize("inl", [int(1.5 * S), 3 * S])
def test_gemma_different_never_vetoes_strong_orb(monkeypatch, tmp_path, inl):
    m = find(monkeypatch, tmp_path, inl, DIFFERENT)
    assert m is not None and m.auto and m.signals.get("note") == "geometry overrode Gemma"


def test_floor_sign_is_no_text_hit(monkeypatch, tmp_path):
    """'LEVEL2' (OCR glued) on both spots used to be a room-number-like text hit, and text + weak ORB merged
    without asking anyone. A floor sign hangs all along that floor's corridor."""
    m = find(monkeypatch, tmp_path, W, None, node_texts=["LEVEL2"], walk_texts=["LEVEL2"], use_gemma=False)
    assert m is not None and not m.signals["text"] and not m.auto


@pytest.mark.parametrize("text", ["LEVEL 2", "LEVEL2", "Level-2", "FL 2", "FLR2", "2ND FLOOR", "2nd Floor", "L2",
                                  "2F", "FLOOR 12", "LVL 3"])
def test_floor_sign(text):
    assert matching.floor_sign(text)
    assert matching.place_tokens([text]) == set()


@pytest.mark.parametrize("text", ["LEVEL 2 ROOM 214", "214", "ROOM 214A", "B204", "2-105", "EXIT", "ROOMS 220-230",
                                  ""])
def test_not_a_floor_sign(text):
    assert not matching.floor_sign(text)


def test_place_tokens_keep_room_numbers_next_to_floor_signs():
    assert matching.place_tokens(["LEVEL 2", "ROOM 214", "LEVEL 2 ROOM 216"]) == {"214", "216"}


# ---------------------------------------------------------------- consolidate
def _bundle(nodes):
    vids = sorted({n.video_id for n in nodes})
    return MapBundle(building_id="b", nodes=nodes,
                     videos=[VideoInfo(id=v, filename=v, status="done") for v in vids])


def _node(i, video, t=None, texts=(), type_="hallway"):
    return Node(id=f"n00{i}", label=f"P{i}", type=type_, frame_id=f"{i}.jpg", texts=list(texts),
                sign_texts=list(texts), floor=2, floor_source="user", video_id=video, t=t)


def consolidate(monkeypatch, tmp_path, nodes, inliers, verdict, sim=None):
    monkeypatch.setattr(ocr, "available", lambda: False)
    monkeypatch.setattr(mapper, "orb_match", lambda a, b: (inliers, 0.0))
    monkeypatch.setattr(gemma, "same_place", lambda *a, **k: verdict)
    if sim is not None:
        monkeypatch.setattr(mapper, "place_similarity", lambda *a, **k: dict(sim, orb=inliers))
    b = _bundle(nodes)
    mapper.consolidate(b, tmp_path, None, True, log=lambda *_: None)
    return b


@pytest.mark.parametrize("inl", [W, 25, S - 1])
def test_consolidate_appearance_gemma_alone_weak_orb_only_suggests(monkeypatch, tmp_path, inl):
    """Step 2 (appearance across walks): Gemma 'identical' with weak ORB is a suggestion, not a merge."""
    b = consolidate(monkeypatch, tmp_path, [_node(1, "v1", 16), _node(2, "v2", 0)], inl, IDENTICAL)
    assert len(b.nodes) == 2
    assert len(b.suggested_merges) == 1


@pytest.mark.parametrize("inl", [S, int(1.5 * S) - 1])
def test_consolidate_appearance_gemma_with_strong_orb_merges(monkeypatch, tmp_path, inl):
    b = consolidate(monkeypatch, tmp_path, [_node(1, "v1", 16), _node(2, "v2", 0)], inl, IDENTICAL)
    assert len(b.nodes) == 1


@pytest.mark.parametrize("inl", [int(1.5 * S), 3 * S])
def test_consolidate_appearance_strong_orb_merges_without_gemma_agreeing(monkeypatch, tmp_path, inl):
    b = consolidate(monkeypatch, tmp_path, [_node(1, "v1", 16), _node(2, "v2", 0)], inl, DIFFERENT)
    assert len(b.nodes) == 1


WEAK_SIM = {"score": 0.4, "name": 0.0, "visual": 0.6, "description": 1.0, "surroundings": 0.0,
            "agree": ["visual", "description"]}


@pytest.mark.parametrize("inl", [0, W, S - 1])
def test_consolidate_descriptors_plus_gemma_need_strong_orb(monkeypatch, tmp_path, inl):
    """Step 1: descriptors too weak to merge alone, Gemma 'identical': no merge without strong ORB."""
    b = consolidate(monkeypatch, tmp_path, [_node(1, "v1", 16), _node(2, "v2", 0)], inl, IDENTICAL, WEAK_SIM)
    assert len(b.nodes) == 2


def test_consolidate_descriptors_plus_gemma_with_strong_orb_merge(monkeypatch, tmp_path):
    b = consolidate(monkeypatch, tmp_path, [_node(1, "v1", 16), _node(2, "v2", 0)], S, IDENTICAL, WEAK_SIM)
    assert len(b.nodes) == 1


POSTER_SIM = {"score": 0.76, "name": 1.0, "visual": 0.85, "description": 1.0, "surroundings": 0.0,
              "agree": ["name", "visual", "description"]}


def test_consolidate_same_walk_look_alikes_seconds_apart_stay_apart(monkeypatch, tmp_path):
    """confusers west@16s and west@26s: the same SEMINAR poster, 9 m apart; name (the poster word), visual (51 ORB
    inliers on the poster) and description agree. The walk moved between them: never one place."""
    nodes = [_node(1, "west", 16.1, ["SEMINAR"]), _node(2, "west", 26.0, ["LEVEL 2", "SEMINAR"])]
    b = consolidate(monkeypatch, tmp_path, nodes, 51, IDENTICAL, POSTER_SIM)
    assert len(b.nodes) == 2


def test_consolidate_same_walk_far_apart_in_time_still_merges(monkeypatch, tmp_path):
    """A loop back to the same spot later in the walk is still merged by descriptors."""
    nodes = [_node(1, "west", 2.0, ["SEMINAR"]), _node(2, "west", 40.0, ["SEMINAR"])]
    b = consolidate(monkeypatch, tmp_path, nodes, 51, IDENTICAL, POSTER_SIM)
    assert len(b.nodes) == 1


def test_consolidate_same_walk_same_room_seconds_apart_still_merges(monkeypatch, tmp_path):
    """Two frames of one door (two_floor up: 'Room 114' and 'Room 114 (2)' 2 s apart): the plaque decides."""
    nodes = [_node(1, "up", 13.0, ["ROOM 114"], "room"), _node(2, "up", 15.0, ["ROOM 114"], "room")]
    b = consolidate(monkeypatch, tmp_path, nodes, 1, None)
    assert len(b.nodes) == 1


# ---------------------------------------------------------------- stale cached OCR (small_loop 'Room 110')
STALE_110 = {"texts": ["110"], "room_numbers": ["110"], "has_sign": True, "suggested_label": "Room 110",
             "place_type": "room", "stairs": "none", "floor_hint": "", "description": "Wooden door with a plaque",
             "ocr": ["110"], "grounded": True}  # as cached by an OCR revision without the image-edge filter


@pytest.fixture
def no_ollama(monkeypatch):
    def refuse(*_a, **_k):
        raise AssertionError("Ollama contacted")
    monkeypatch.setattr(gemma, "_client", refuse)
    monkeypatch.setattr(gemma, "_resolved_model", None)
    monkeypatch.setattr(ocr, "available", lambda: True)


def _cached_frame(tmp_path, value, color=(30, 60, 90)):
    frame = _jpg(tmp_path / "v_0010.jpg", color)
    cache = gemma.GemmaCache(tmp_path / "c.jsonl")
    key = gemma._frame_key(frame)(config.GEMMA_MODEL)
    cache.put(key, value)
    return frame, cache, key


def test_edge_cut_plaque_is_dropped_by_current_ocr():
    """Root cause: the plaque 110A runs off the right edge (OCR box x 0.94-0.998) and reads as '110'."""
    assert ocr._edge_cut("110", 0.942, 0.998, True, True)


@pytest.mark.parametrize("reader", ["analyze_frame", "cached_analysis"])
def test_stale_cached_ocr_is_reread_and_invented_room_dropped(no_ollama, monkeypatch, tmp_path, reader):
    frame, cache, key = _cached_frame(tmp_path, dict(STALE_110))
    reads = []
    monkeypatch.setattr(ocr, "read_text_with_crops", lambda p: reads.append(p) or [])
    a = getattr(gemma, reader)(frame, cache)
    assert reads, "OCR not re-read for an answer from an older OCR revision"
    assert a.room_numbers == [] and "110" not in a.suggested_label and "110" not in " ".join(a.texts)
    assert a.ocr == [] and a.ocr_rev == ocr.REVISION
    stored = cache.get(key)
    assert stored["ocr"] == [] and stored["ocr_rev"] == ocr.REVISION
    assert stored["room_numbers"] == ["110"]  # Gemma's raw answer is kept; only the OCR evidence is newer
    on_disk = [json.loads(line) for line in (tmp_path / "c.jsonl").read_text().splitlines()]
    assert on_disk[-1]["v"]["ocr_rev"] == ocr.REVISION


def test_current_revision_is_not_reread(no_ollama, monkeypatch, tmp_path):
    value = dict(STALE_110, texts=["110A"], room_numbers=["110A"], suggested_label="Room 110A", ocr=["110A"],
                 ocr_rev=ocr.REVISION)
    frame, cache, _ = _cached_frame(tmp_path, value, (90, 60, 30))
    monkeypatch.setattr(ocr, "read_text_with_crops", lambda p: pytest.fail("OCR re-read"))
    a = gemma.analyze_frame(frame, cache)
    assert a.room_numbers == ["110A"] and a.suggested_label == "Room 110A"


def test_stale_entry_kept_when_ocr_is_off(monkeypatch, tmp_path):
    """Without OCR (not macOS) nothing can be re-read: the cached answer is used as it is."""
    frame, cache, _ = _cached_frame(tmp_path, dict(STALE_110), (10, 200, 10))
    monkeypatch.setattr(gemma, "_resolved_model", None)
    monkeypatch.setattr(ocr, "available", lambda: False)
    monkeypatch.setattr(gemma, "_frame_key", lambda p: lambda m: f"analyze:{gemma.PROMPT_VERSION}:{m}:ocr:x")
    cache.put(f"analyze:{gemma.PROMPT_VERSION}:{config.GEMMA_MODEL}:ocr:x", dict(STALE_110))
    a = gemma.analyze_frame(frame, cache)
    assert a.ocr == ["110"] and a.ocr_rev == 0


def test_tidy_rooms_rechecks_empty_ocr_at_known_revision():
    """OCR read nothing on the frame (only the edge-cut plaque): nothing Gemma reported is grounded."""
    a = FrameAnalysis(**dict(STALE_110, ocr=[], ocr_rev=ocr.REVISION))
    gemma._tidy_rooms(a)
    assert a.room_numbers == [] and a.texts == [] and a.suggested_label != "Room 110"


def test_tidy_rooms_leaves_legacy_entry_without_lines_alone():
    """An old grounded entry that never stored its lines was already grounded when it was computed."""
    a = FrameAnalysis(texts=["EXIT"], suggested_label="Exit", place_type="exit", grounded=True)
    gemma._tidy_rooms(a)
    assert a.texts == ["EXIT"] and a.suggested_label == "Exit"
