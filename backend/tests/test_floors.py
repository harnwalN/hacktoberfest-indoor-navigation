"""Floors from room numbers + stairs, and map-wide consolidation across walks."""
from pathlib import Path

from app import mapper
from app.schemas import Edge, FrameAnalysis, Keyframe, MapBundle, Node, VideoInfo


def _wn(t, rooms=(), rel_floor=0):
    a = FrameAnalysis(has_sign=bool(rooms), texts=list(rooms), room_numbers=list(rooms), grounded=True)
    kf = Keyframe(frame_id=f"f{t}.jpg", video_id="v", t=t, kind="scene", heading=0, analysis=a)
    return mapper.WalkNode(kf=kf, t_arrive=t, t_depart=t, analysis=a, rel_floor=rel_floor)


def test_room_floor_convention():
    assert mapper.room_floor(["121"]) == 1
    assert mapper.room_floor(["Room 208"]) == 2
    assert mapper.room_floor(["B112"]) == 1
    assert mapper.room_floor(["1210"]) == 12
    assert mapper.room_floor(["EXIT"]) is None


def test_floors_from_rooms_and_elevator_jump():
    # floor 1 rooms, then floor 2 rooms with no stairs seen (elevator)
    walk = [_wn(0), _wn(5, ["121"]), _wn(10, ["123"]), _wn(20), _wn(30, ["208"]), _wn(35, ["210"])]
    assert mapper.assign_floors(walk, None) == [1, 1, 1, 1, 2, 2]


def test_single_misread_is_not_a_floor_change():
    walk = [_wn(0, ["121"]), _wn(5, ["321"]), _wn(10, ["125"])]
    assert mapper.assign_floors(walk, None) == [1, 1, 1]


def test_stairs_shift_relative_floor():
    walk = [_wn(0, ["121"]), _wn(10, rel_floor=1), _wn(20, rel_floor=1)]
    assert mapper.assign_floors(walk, None) == [1, 2, 2]


def test_start_floor_used_without_evidence():
    assert mapper.assign_floors([_wn(0), _wn(5)], 3) == [3, 3]


def _bundle():
    nodes = [
        Node(id="n001", label="Room 208", texts=["208"], floor=2, floor_source="rooms", video_id="vA", frame_id=None),
        Node(id="n002", label="Hall A", floor=2, video_id="vA"),
        Node(id="n003", label="Room 208 door", texts=["208"], floor=2, floor_source="rooms", video_id="vB", frame_id=None),
        Node(id="n004", label="Hall B", floor=2, video_id="vB"),
        Node(id="n005", label="Room 208?", texts=["208"], floor=3, floor_source="user", video_id="vC"),
    ]
    edges = [
        Edge(id="e001", from_node="n002", to_node="n001", heading_out=0, heading_in=0,
             heading_out_raw=0, heading_in_raw=0, video_id="vA"),
        # walk B recorded in its own frame, rotated +90 relative to A
        Edge(id="e002", from_node="n003", to_node="n004", heading_out=90, heading_in=90,
             heading_out_raw=90, heading_in_raw=90, video_id="vB"),
    ]
    videos = [VideoInfo(id="vA", filename="a", status="done", anchored=True),
              VideoInfo(id="vB", filename="b", status="done", anchored=False),
              VideoInfo(id="vC", filename="c", status="done", anchored=True)]
    return MapBundle(building_id="t", nodes=nodes, edges=edges, videos=videos)


def test_consolidate_merges_same_room_across_walks_but_not_floors(tmp_path):
    b = _bundle()
    res = mapper.consolidate(b, tmp_path, None, use_gemma=False, log=lambda *_: None)
    ids = {n.id for n in b.nodes}
    assert "n003" not in ids  # B's Room 208 merged into A's
    assert "n005" in ids  # same number on floor 3 is a different place
    assert res["components"] == 2  # floor-2 walks joined; floor 3 separate
    assert next(v for v in b.videos if v.id == "vB").anchored


def test_guessed_walk_adopts_floor_from_named_sign(tmp_path):
    nodes = [
        Node(id="n001", label="Maker Space", texts=["MAKER SPACE", "QUALCOMM INSTITUTE"],
             sign_texts=["MAKER SPACE", "QUALCOMM INSTITUTE"], floor=2,
             floor_source="rooms", video_id="vA"),
        Node(id="n002", label="Maker Space Entrance", texts=["MAKERSPACE", "QUALCOMM"],
             sign_texts=["MAKER SPACE", "QUALCOMM"], floor=1,
             floor_source="default", video_id="vB"),
        Node(id="n003", label="Hallway", floor=1, floor_source="default", video_id="vB"),
    ]
    edges = [Edge(id="e001", from_node="n002", to_node="n003", heading_out=0, heading_in=0,
                  heading_out_raw=0, heading_in_raw=0, video_id="vB")]
    videos = [VideoInfo(id="vA", filename="a", status="done", anchored=True),
              VideoInfo(id="vB", filename="b", status="done", anchored=True)]
    b = MapBundle(building_id="t", nodes=nodes, edges=edges, videos=videos)
    mapper.consolidate(b, tmp_path, None, use_gemma=False, log=lambda *_: None)
    assert {n.id for n in b.nodes} == {"n001", "n003"}
    assert next(n for n in b.nodes if n.id == "n003").floor == 2  # whole walk moved to floor 2


def test_single_staircases_on_adjacent_floors_get_linked(tmp_path):
    nodes = [Node(id="n001", label="Stairs", type="stairs", floor=1, floor_source="rooms", video_id="vA"),
             Node(id="n002", label="Stairwell", type="stairs", floor=2, floor_source="rooms", video_id="vB")]
    b = MapBundle(building_id="t", nodes=nodes, videos=[])
    res = mapper.consolidate(b, tmp_path, None, use_gemma=False, log=lambda *_: None)
    assert res["floor_links"] == 1
    e = b.edges[0]
    assert (e.from_node, e.to_node, e.vertical, e.inferred_floor_change) == ("n001", "n002", "stairs_up", True)


def test_remove_video_keeps_shared_places():
    b = _bundle()
    mapper.consolidate(b, Path("."), None, use_gemma=False, log=lambda *_: None)  # B's 208 merged into A's
    res = mapper.remove_video(b, "vA")
    ids = {n.id for n in b.nodes}
    assert "n002" not in ids  # A-only place removed
    assert "n001" in ids  # Room 208 is still on walk B's route
    assert all(e.video_id != "vA" for e in b.edges)
    assert res["removed_places"] == 1 and not any(v.id == "vA" for v in b.videos)


def test_busy_or_partial_text_does_not_snowball():
    wall = ["WELCOME", "INNOVATORS", "CHANGEMAKERS", "ENGINEERS", "SCIENTISTS", "FOUNDERS", "ARTISTS"]
    assert not mapper.strong_name_match(wall, wall)  # wall graphic visible from everywhere
    assert not mapper.strong_name_match(["MAKERS"], ["MAKER SPACE", "QUALCOMM INSTITUTE"])
    assert mapper.strong_name_match(["MAKER SPACE"], ["MAKER SPACE BY QUALCOMM INSTITUTE"])
