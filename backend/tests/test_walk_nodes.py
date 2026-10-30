"""walk_nodes: places glanced at a few seconds apart stay apart; frames of one place still collapse.

Synthetic keyframe sequences (no video, no Gemma)."""
from app import config, mapper
from app.schemas import FrameAnalysis, Keyframe, MapBundle, Node, VideoInfo


def _kf(t, kind="scene", texts=(), rooms=(), place="hallway", label="", arrive=None, depart=None):
    tx = list(texts)
    a = FrameAnalysis(has_sign=bool(tx), texts=tx, room_numbers=list(rooms), ocr=tx, grounded=True,
                      place_type=place, suggested_label=label)
    return Keyframe(frame_id=f"f{t}.jpg", video_id="v", t=float(t), kind=kind, heading=0.0, analysis=a,
                    t_arrive=arrive, t_depart=depart)


def _glance(t, room, plaque=None, place="room"):
    return _kf(t, "glance", texts=[plaque or f"ROOM {room}"], rooms=[room], place=place, label=f"Room {room}")


def _rooms(w):
    return mapper.room_ids(w.texts)


def test_two_doors_glanced_within_the_min_gap_are_two_places():
    gap = config.NODE_MIN_GAP_SEC / 2  # used to absorb the second door, whatever its number
    nodes = mapper.walk_nodes([_kf(0, "start"), _glance(5, "112"), _glance(5 + gap, "114"), _kf(20, "end")])
    door = [w for w in nodes if w.kf.kind == "glance"]
    assert [_rooms(w) for w in door] == [{"112"}, {"114"}]


def test_different_room_numbers_never_absorb_even_without_glances():
    gap = config.NODE_MIN_GAP_SEC / 3
    nodes = mapper.walk_nodes([_kf(0, "start"), _kf(4, "dwell", ["RM 220"], ["220"], "room"),
                               _kf(4 + gap, "dwell", ["RM 224"], ["224"], "room"), _kf(20, "end")])
    assert [_rooms(w) for w in nodes if _rooms(w)] == [{"220"}, {"224"}]


def test_two_glances_seconds_apart_without_a_shared_place_stay_apart():
    # look left at an unnumbered office, then right at another one: two glances are two places
    a = _kf(5, "glance", ["OFFICE A"], place="room", label="Office A")
    b = _kf(5 + config.NODE_MIN_GAP_SEC / 2, "glance", ["OFFICE B"], place="room", label="Office B")
    nodes = mapper.walk_nodes([_kf(0, "start"), a, b, _kf(20, "end")])
    assert sum(w.kf.kind == "glance" for w in nodes) == 2


def test_consecutive_frames_of_one_place_still_collapse():
    seq = [_kf(0, "start"), _kf(4, texts=["ROOM 101"], rooms=["101"], label="Hallway with ROOM 101"),
           _kf(4.5, texts=["ROOM 101"], rooms=["101"], place="room", label="Room 101"),
           _glance(6, "101"), _kf(7, "dwell", ["ROOM 101"], ["101"], "room", "Room 101"), _kf(20, "end")]
    nodes = mapper.walk_nodes(seq)
    (door,) = [w for w in nodes if _rooms(w)]
    assert door.kf.kind == "glance" and door.glances == 1
    # the glance's own analysis wins over the corridor view that was there first
    assert door.analysis.place_type == "room" and door.analysis.suggested_label == "Room 101"


def test_absorbed_frames_do_not_chain_rooms():
    # a frame showing this door and the next one is absorbed into 101, but must not carry 103 along,
    # or 103's glance would be "the same spot" and two rooms would become one place
    seq = [_kf(0, "start"), _glance(4, "101"),
           _kf(5, texts=["ROOM 101", "RM 103"], rooms=["101", "103"], label="Hallway"),
           _glance(7, "103", "RM 103"), _kf(20, "end")]
    nodes = mapper.walk_nodes(seq)
    assert [_rooms(w) for w in nodes if _rooms(w)] == [{"101"}, {"103"}]


def test_turn_that_glanced_at_a_door_is_that_room_but_a_corner_in_view_of_one_is_not():
    seq = [_kf(0, "start"),
           _kf(10, "turn_start", arrive=10.5, depart=13), _kf(13, "turn_end", ["ROOM 105"], ["105"], label="Hallway with ROOM 105"),
           _glance(15, "105"),
           _kf(30, "turn_start", arrive=30.5, depart=33), _kf(33, "turn_end", ["ROOM 101"], ["101"], label="Hallway with ROOM 101"),
           _kf(45, "end")]
    turns = [w for w in mapper.walk_nodes(seq) if w.kf.kind == "turn_start"]
    assert [(_rooms(w), w.analysis.place_type) for w in turns] == [({"105"}, "room"), ({"101"}, "hallway")]
    assert [mapper._label(w)[1] for w in turns] == ["room", "intersection"]


def _bundle(*nodes):
    return MapBundle(building_id="t", nodes=list(nodes),
                     videos=[VideoInfo(id=v, filename=v, status="done", anchored=True) for v in ("vA", "vB")])


def _node(nid, label, room, type_, video):
    return Node(id=nid, label=label, type=type_, texts=[f"ROOM {room}"], sign_texts=[f"ROOM {room}"], floor=1,
                floor_source="rooms", video_id=video)


def test_consolidate_merges_the_room_but_not_corridor_spots_that_see_its_plaque(tmp_path):
    room = _node("n001", "Room 101", "101", "room", "vA")
    again = _node("n002", "Room 101", "101", "room", "vB")
    corner = _node("n003", "Hallway with ROOM 101", "101", "intersection", "vA")  # plaque seen from the corner
    start = _node("n004", "Hallway with ROOM 101", "101", "hallway", "vB")  # plaque seen down the hall
    b = _bundle(room, again, corner, start)
    mapper.consolidate(b, tmp_path, None, use_gemma=False, log=lambda *_: None)
    assert {n.id for n in b.nodes} == {"n001", "n003", "n004"}
