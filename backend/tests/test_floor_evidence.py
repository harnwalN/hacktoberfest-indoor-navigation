"""Floor evidence: confirmation of room-number floors, floor signs / Gemma floor_hint, safer room parsing."""
from app import config, floors, mapper
from app.schemas import FrameAnalysis, Keyframe


def _a(rooms=(), texts=(), floor_hint="", ocr=None, grounded=True, stairs="none", place_type="hallway"):
    texts = list(rooms) + list(texts)
    return FrameAnalysis(has_sign=bool(texts), texts=texts, room_numbers=list(rooms), floor_hint=floor_hint,
                         ocr=list(texts) if ocr is None else list(ocr), grounded=grounded, stairs=stairs,
                         place_type=place_type)


def _wn(t, rooms=(), rel_floor=0, **kw):
    a = _a(rooms, **kw)
    kf = Keyframe(frame_id=f"f{t}.jpg", video_id="v", t=t, kind="scene", heading=0, analysis=a)
    return mapper.WalkNode(kf=kf, t_arrive=t, t_depart=t, analysis=a, rel_floor=rel_floor)


def _kf(t, kind, **kw):
    return Keyframe(frame_id=f"f{t}.jpg", video_id="v", t=t, kind=kind, heading=0, analysis=_a(**kw))


# ---------------------------------------------------------------- confirmation (B5a / B5b)
def test_misread_before_first_confirmed_floor_is_ignored():
    walk = [_wn(0), _wn(5), _wn(10, ["321"]), _wn(15, ["121"]), _wn(20, ["123"])]
    assert mapper.assign_floors(walk, None) == [1, 1, 1, 1, 1]


def test_single_trailing_misread_is_not_a_floor_change():
    walk = [_wn(0, ["121"]), _wn(5, ["123"]), _wn(10, ["321"])]
    assert mapper.assign_floors(walk, None) == [1, 1, 1]


def test_lone_room_number_is_not_confirmed():
    walk = [_wn(0, ["321"]), _wn(5)]
    assert mapper.assign_floors(walk, None) == [None, None]  # caller falls back to linking / default floor
    assert mapper.assign_floors(walk, 1) == [1, 1]  # and cannot override the uploader's start floor


def test_room_numbers_agree_across_stairs():
    walk = [_wn(0, ["121"]), _wn(10, rel_floor=1), _wn(20, ["221"], rel_floor=1)]
    assert mapper.assign_floors(walk, None) == [1, 2, 2]


def test_one_sign_does_not_outvote_two_rooms():
    walk = [_wn(0, ["121"]), _wn(5, ["123"]), _wn(10, texts=["2ND FLOOR"]), _wn(15, ["125"]), _wn(20, ["127"])]
    assert mapper.assign_floors(walk, None) == [1, 1, 1, 1, 1]


# ---------------------------------------------------------------- floor signs (B15)
def test_floor_hint_without_room_numbers():
    walk = [_wn(0, floor_hint="2", ocr=["2ND FLOOR"]), _wn(5)]
    assert mapper.assign_floors(walk, None) == [2, 2]


def test_floor_hint_not_backed_by_ocr_is_ignored():
    # OCR saw '125' (so the old substring grounding passes '2'), but nothing on the image states a floor
    walk = [_wn(0, floor_hint="2", ocr=["125"]), _wn(5)]
    assert mapper.assign_floors(walk, None) == [None, None]


def test_ungrounded_floor_hint_needs_agreement():
    alone = [_wn(0, floor_hint="2", grounded=False), _wn(5)]
    assert mapper.assign_floors(alone, None) == [None, None]
    agreed = [_wn(0, floor_hint="2", grounded=False), _wn(5, ["208"], grounded=False)]
    assert mapper.assign_floors(agreed, None) == [2, 2]


def test_ocr_level_sign():
    walk = [_wn(0), _wn(5, texts=["LEVEL 3"]), _wn(10)]
    assert mapper.assign_floors(walk, None) == [3, 3, 3]


def test_floor_sign_patterns():
    assert floors.sign_floor(["3RD FLOOR"]) == 3
    assert floors.sign_floor(["FLOOR 2"]) == 2
    assert floors.sign_floor(["Level 2"]) == 2
    assert floors.sign_floor(["L3"]) == 3
    assert floors.sign_floor(["SECOND FLOOR"]) == 2
    assert floors.sign_floor(["WET FLOOR"]) is None
    assert floors.sign_floor(["GROUND FLOOR"]) is None  # 0 or 1 depends on the country: no evidence
    assert floors.sign_floor(["LEVEL B1"]) is None
    assert floors.sign_floor(["FLOORS 1-3"]) is None
    assert floors.sign_floor(["FLOOR 1-3"]) is None
    assert floors.sign_floor(["FLOOR 1", "FLOOR 2"]) is None  # directory listing several floors
    assert floors.sign_floor([f"FLOOR {config.MAX_FLOOR + 1}"]) is None
    assert floors.sign_floor(["ROOM L3 LAB"]) is None  # 'L3' only counts as a whole sign


def test_floor_sign_on_frame_that_is_not_a_walk_node():
    kfs = [_kf(0, "start"), _kf(3, "heartbeat", texts=["2ND FLOOR"]), _kf(10, "dwell"), _kf(20, "end")]
    wnodes = mapper.walk_nodes(kfs)
    assert len(wnodes) == 3  # the plain hallway frame with the sign is not a node of its own
    assert mapper.assign_floors(wnodes, None, keyframes=kfs) == [2, 2, 2]


def test_floor_sign_read_on_the_stairs_is_ignored():
    kfs = [_kf(0, "start"), _kf(5, "heartbeat", texts=["FLOOR 3"], stairs="on_stairs_up"),
           _kf(8, "dwell"), _kf(15, "end", texts=["2ND FLOOR"])]
    wnodes = mapper.walk_nodes(kfs)
    assert [w.rel_floor for w in wnodes] == [0, 1, 1]
    assert mapper.assign_floors(wnodes, None, keyframes=kfs) == [1, 2, 2]


# ---------------------------------------------------------------- room-number parsing (B14)
def test_basement_room_is_not_ground_floor():
    assert mapper.room_floor(["B112"]) is None
    assert mapper.room_floor(["B-112"]) is None
    assert mapper.room_floor(["B112", "121"]) == 1


def test_hyphenated_and_four_digit_rooms(monkeypatch):
    assert mapper.room_floor(["2-105"]) is None  # building-room or floor-room: ambiguous
    assert mapper.room_floor(["858-534-1234"]) is None
    assert mapper.room_floor(["121-A"]) == 1
    assert mapper.room_floor(["1205"]) is None  # floor 12 > MAX_FLOOR (9)
    monkeypatch.setattr(config, "MAX_FLOOR", 15)
    assert mapper.room_floor(["1205"]) == 12
    assert mapper.room_floor(["0905"]) is None  # leading zero is not a floor
