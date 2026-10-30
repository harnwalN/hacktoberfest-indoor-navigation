"""Places on different floors are not merged just because they look alike (B7), floor adoption in
consolidate does not depend on pair order (B8), and merges never join two known floors (B9)."""
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import config, gemma, mapper, store
from app.main import app
from app.schemas import Edge, FrameAnalysis, Keyframe, MapBundle, Node, VideoInfo


def _quiet(*_):
    pass


# ---------------------------------------------------------------- integrate_walk (B7)
def _kf(t, kind):
    a = FrameAnalysis(grounded=True)
    return Keyframe(frame_id=f"f{int(t)}.jpg", video_id="vW", t=float(t), kind=kind, heading=0.0, analysis=a)


def _track(n=600):
    t = np.arange(n) / 10.0
    return {"t": t.tolist(), "fps": 10.0, "yaw": [0.0] * n, "fwd": [1.0] * n, "heading": [0.0] * n}


def _lobby_map():
    """Two identical-looking elevator lobbies, known (room numbers) to be on floor 3."""
    nodes = [Node(id="n001", label="Elevator lobby", type="elevator", frame_id="L3a.jpg", frame_heading=0.0,
                  floor=3, floor_source="rooms", video_id="vX", t=0),
             Node(id="n002", label="Stair landing", type="stairs", frame_id="L3b.jpg", frame_heading=0.0,
                  floor=3, floor_source="rooms", video_id="vX", t=10)]
    edges = [Edge(id="e001", from_node="n001", to_node="n002", heading_out=0, heading_in=0,
                  heading_out_raw=0, heading_in_raw=0, video_id="vX", length_m=10)]
    return MapBundle(building_id="t", nodes=nodes, edges=edges,
                     videos=[VideoInfo(id="vX", filename="x", status="done", anchored=True)])


def _fake_orb(monkeypatch, pairs: dict[str, str], inliers=70):
    """Walk frame -> map frame pairs that ORB-match with `inliers` (>= 1.5 * ORB_STRONG: auto without Gemma)."""
    links = {frozenset(p) for p in pairs.items()}

    def fake(a, b):
        return (inliers, 0.0) if frozenset((Path(a).name, Path(b).name)) in links else (0, None)

    monkeypatch.setattr(mapper, "orb_match", fake)


def _walk():
    return [_kf(0, "start"), _kf(20, "dwell"), _kf(40, "end")]


def _walk_nodes(b):
    return [n for n in b.nodes if n.video_id == "vW"]


def test_single_orb_match_with_unknown_floor_is_only_a_suggestion(monkeypatch, tmp_path):
    assert 70 >= 1.5 * config.ORB_STRONG_INLIERS  # would have auto-merged before the fix
    _fake_orb(monkeypatch, {"f0.jpg": "L3a.jpg"})
    b = _lobby_map()
    b.videos.append(VideoInfo(id="vW", filename="w", status="done"))
    res = mapper.integrate_walk(b, "vW", _walk(), _track(), tmp_path, None, False, log=_quiet)
    assert res["merged"] == 0
    mine = _walk_nodes(b)
    assert len(mine) == 3 and "n001" in {n.id for n in b.nodes}
    assert {(n.floor, n.floor_source) for n in mine} == {(config.DEFAULT_FLOOR, "default")}
    start = next(n for n in mine if n.frame_id == "f0.jpg")
    assert [(s.keep, s.remove) for s in b.suggested_merges] == [("n001", start.id)]

    # consolidation must not do it either (default floor is not 'compatible with anything' for a lone match)
    mapper.consolidate(b, tmp_path, None, use_gemma=False, log=_quiet)
    mine = _walk_nodes(b)
    assert start.id in {n.id for n in b.nodes} and "n001" in {n.id for n in b.nodes}
    assert {(n.floor, n.floor_source) for n in mine} == {(config.DEFAULT_FLOOR, "default")}
    assert len(b.suggested_merges) == 1  # still the one suggestion, not a duplicate


def test_two_agreeing_matches_set_the_walk_floor(monkeypatch, tmp_path):
    _fake_orb(monkeypatch, {"f0.jpg": "L3a.jpg", "f40.jpg": "L3b.jpg"})
    b = _lobby_map()
    res = mapper.integrate_walk(b, "vW", _walk(), _track(), tmp_path, None, False, log=_quiet)
    assert res["merged"] == 2 and res["nodes"] == 1
    ids = {n.id for n in b.nodes}
    assert {"n001", "n002"} <= ids and len(ids) == 3  # both ends merged into the floor-3 places
    middle = next(n for n in b.nodes if n.id not in {"n001", "n002"})
    assert (middle.floor, middle.floor_source) == (3, "linked")
    assert {(e.from_node, e.to_node) for e in b.edges if e.video_id == "vW"} == {("n001", middle.id),
                                                                                 (middle.id, "n002")}
    assert b.suggested_merges == []


def test_room_number_match_alone_still_sets_the_walk_floor(monkeypatch, tmp_path):
    _fake_orb(monkeypatch, {"f0.jpg": "L3a.jpg"}, inliers=config.ORB_WEAK_INLIERS)
    b = _lobby_map()
    b.nodes[0].texts = ["Room 314"]
    kfs = _walk()
    kfs[0].analysis = FrameAnalysis(grounded=True, has_sign=True, texts=["Room 314"], room_numbers=["314"])
    mapper.integrate_walk(b, "vW", kfs, _track(), tmp_path, None, False, log=_quiet)
    assert "n001" in {n.id for n in b.nodes}
    assert {(n.floor, n.floor_source) for n in _walk_nodes(b)} == {(3, "linked")}


# ---------------------------------------------------------------- consolidate order (B8)
def _b8_nodes(extra_floor2_vote=False):
    nodes = [
        Node(id="n001", label="Maker Space", sign_texts=["MAKER SPACE", "QUALCOMM INSTITUTE"],
             texts=["MAKER SPACE", "QUALCOMM INSTITUTE"], description="glass doors with logo",
             floor=1, floor_source="rooms", video_id="vA", t=0),
        Node(id="n002", label="Room 208", type="room", texts=["208"], sign_texts=["208"],
             floor=2, floor_source="rooms", video_id="vB", t=0),
        # walk vC: no floor evidence of its own (one room number does not confirm a floor)
        Node(id="n003", label="Room 208 door", type="room", texts=["208"], sign_texts=["208"],
             floor=1, floor_source="default", video_id="vC", t=0),
        Node(id="n004", label="Maker Space Entrance", sign_texts=["MAKER SPACE", "QUALCOMM"],
             texts=["MAKERSPACE", "QUALCOMM"], description="glass doors and logo",
             floor=1, floor_source="default", video_id="vC", t=5),
        Node(id="n005", label="Hallway", floor=1, floor_source="default", video_id="vC", t=10),
    ]
    if extra_floor2_vote:
        nodes += [Node(id="n006", label="Room 214", type="room", texts=["214"], sign_texts=["214"],
                       floor=2, floor_source="rooms", video_id="vB", t=10),
                  Node(id="n007", label="Room 214 door", type="room", texts=["214"], sign_texts=["214"],
                       floor=1, floor_source="default", video_id="vC", t=15)]
    return nodes


def _b8_bundle(order, extra_floor2_vote=False):
    by_id = {n.id: n for n in _b8_nodes(extra_floor2_vote)}
    nodes = [by_id[i] for i in order if i in by_id] + [n for i, n in by_id.items() if i not in order]
    edges = [Edge(id="e001", from_node="n003", to_node="n004", heading_out=0, heading_in=0,
                  heading_out_raw=0, heading_in_raw=0, video_id="vC", length_m=5),
             Edge(id="e002", from_node="n004", to_node="n005", heading_out=90, heading_in=90,
                  heading_out_raw=90, heading_in_raw=90, video_id="vC", length_m=5)]
    videos = [VideoInfo(id=v, filename=v, status="done", anchored=True) for v in ("vA", "vB", "vC")]
    return MapBundle(building_id="t", nodes=nodes, edges=edges, videos=videos)


def _outcome(b):
    return ({n.id: (n.floor, n.floor_source) for n in b.nodes},
            {frozenset((e.from_node, e.to_node)) for e in b.edges},
            {frozenset((s.keep, s.remove)) for s in b.suggested_merges},
            {v.id: v.floor_offset for v in b.videos})


ORDERS = [["n001", "n002", "n003", "n004"], ["n002", "n001", "n003", "n004"], ["n004", "n003", "n002", "n001"]]


def test_consolidate_tie_is_order_independent_and_adopts_nothing(tmp_path):
    outcomes = []
    for order in ORDERS:
        b = _b8_bundle(order)
        mapper.consolidate(b, tmp_path, None, use_gemma=False, log=_quiet)
        outcomes.append(_outcome(b))
    assert all(o == outcomes[0] for o in outcomes)
    floors, _, suggested, _ = outcomes[0]
    assert set(floors) == {"n001", "n002", "n003", "n004", "n005"}  # conflicting evidence: nothing merged
    assert {floors[i] for i in ("n003", "n004", "n005")} == {(1, "default")}
    assert suggested == {frozenset(("n001", "n004")), frozenset(("n002", "n003"))}  # left for review


def test_consolidate_majority_is_order_independent(tmp_path):
    outcomes = []
    for order in ORDERS + [["n007", "n006", "n004", "n001"]]:
        b = _b8_bundle(order, extra_floor2_vote=True)
        mapper.consolidate(b, tmp_path, None, use_gemma=False, log=_quiet)
        outcomes.append(_outcome(b))
    assert all(o == outcomes[0] for o in outcomes)
    floors, edges, _, offsets = outcomes[0]
    assert "n003" not in floors and "n007" not in floors  # two room numbers on floor 2: merged
    assert floors["n004"] == (2, "linked") and floors["n005"] == (2, "linked")  # walk moved to floor 2
    assert "n001" in floors  # the floor-1 Maker Space was outvoted: not merged across floors
    assert frozenset(("n002", "n004")) in edges and offsets["vC"] == 1


# ---------------------------------------------------------------- merge_nodes / API (B9)
def test_merge_nodes_refuses_different_known_floors():
    b = _lobby_map()
    b.nodes.append(Node(id="n009", label="Elevator lobby", floor=2, floor_source="rooms", video_id="vY"))
    before = b.model_dump()
    with pytest.raises(ValueError, match="different floors"):
        mapper.merge_nodes(b, "n001", "n009")
    assert b.model_dump() == before


def test_merge_nodes_default_floor_adopts_known_floor():
    b = _lobby_map()
    b.nodes += [Node(id="n009", label="Lobby", floor=1, floor_source="default", video_id="vY", t=0),
                Node(id="n010", label="Hall", floor=1, floor_source="default", video_id="vY", t=5)]
    b.edges.append(Edge(id="e002", from_node="n009", to_node="n010", heading_out=0, heading_in=0, video_id="vY"))
    b.videos.append(VideoInfo(id="vY", filename="y", status="done", floor_offset=1))
    mapper.merge_nodes(b, "n009", "n001")  # the guessed place survives, the known one is removed
    f = {n.id: (n.floor, n.floor_source) for n in b.nodes}
    assert "n001" not in f and f["n009"] == (3, "linked") and f["n010"] == (3, "linked")
    assert next(v for v in b.videos if v.id == "vY").floor_offset == 3


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BUILDINGS_DIR", tmp_path / "buildings")

    def no_gemma(*_a, **_k):
        raise gemma.GemmaUnavailable("disabled in tests")

    monkeypatch.setattr(gemma, "resolve_model", no_gemma)
    bid = store.create_building("Floors").building_id
    b = _lobby_map()
    b.building_id = bid
    b.nodes.append(Node(id="n009", label="Elevator lobby", frame_id="L2.jpg", floor=2, floor_source="rooms",
                        video_id="vY"))
    b.suggested_merges = [mapper.SuggestedMerge(id="m001", keep="n001", remove="n009", confidence=0.6,
                                                reason="70 matching image features")]
    store.save_map(b)
    return TestClient(app), bid


def test_manual_merge_across_floors_is_400_and_map_unchanged(client):
    c, bid = client
    before = store.load_map(bid).model_dump()
    r = c.post(f"/api/buildings/{bid}/merge", json={"keep": "n001", "remove": "n009"})
    assert r.status_code == 400, r.text
    assert "floor" in r.json()["detail"].lower()
    assert store.load_map(bid).model_dump() == before


def test_accepting_cross_floor_suggestion_is_400_and_map_unchanged(client):
    c, bid = client
    before = store.load_map(bid).model_dump()
    r = c.post(f"/api/buildings/{bid}/merges/m001", json={"accept": True})
    assert r.status_code == 400, r.text
    assert "floor" in r.json()["detail"].lower()
    assert store.load_map(bid).model_dump() == before
    r = c.post(f"/api/buildings/{bid}/merges/m001", json={"accept": False})  # rejecting still works
    assert r.status_code == 200 and r.json()["suggested_merges"] == []


# ---------------------------------------------------------------- sentinel probes
def _walk5():
    return [_kf(0, "start"), _kf(10, "dwell"), _kf(20, "dwell"), _kf(30, "dwell"), _kf(40, "end")]


def _three_floor_map():
    b = _lobby_map()
    b.nodes.append(Node(id="n003", label="Elevator lobby 2", type="elevator", frame_id="L2a.jpg", frame_heading=0.0,
                        floor=2, floor_source="rooms", video_id="vX", t=20))
    return b


@pytest.mark.parametrize("pairs", [
    {"f0.jpg": "L2a.jpg", "f10.jpg": "L3a.jpg", "f30.jpg": "L3b.jpg"},   # minority vote first
    {"f0.jpg": "L3a.jpg", "f10.jpg": "L2a.jpg", "f30.jpg": "L3b.jpg"},
    {"f0.jpg": "L3a.jpg", "f20.jpg": "L3b.jpg", "f40.jpg": "L2a.jpg"},   # minority vote last
])
def test_probe_two_votes_floor3_beat_one_vote_floor2(monkeypatch, tmp_path, pairs):
    _fake_orb(monkeypatch, pairs)
    b = _three_floor_map()
    mapper.integrate_walk(b, "vW", _walk5(), _track(), tmp_path, None, False, log=_quiet)
    ids = {n.id for n in b.nodes}
    assert {"n001", "n002", "n003"} <= ids  # nothing known got removed
    assert all(n.floor == 3 for n in _walk_nodes(b))
    # the floor-2 look-alike is not merged into the floor-3 walk
    assert not any(e for e in b.edges if e.video_id == "vW" and "n003" in (e.from_node, e.to_node))


def test_probe_same_known_place_matched_twice_counts_once(monkeypatch, tmp_path):
    _fake_orb(monkeypatch, {"f0.jpg": "L3a.jpg", "f40.jpg": "L3a.jpg"})
    b = _lobby_map()
    res = mapper.integrate_walk(b, "vW", _walk(), _track(), tmp_path, None, False, log=_quiet)
    assert res["merged"] == 0
    assert {(n.floor, n.floor_source) for n in _walk_nodes(b)} == {(config.DEFAULT_FLOOR, "default")}


def test_probe_no_room_numbers_loop_closure_between_two_default_walks(monkeypatch, tmp_path):
    _fake_orb(monkeypatch, {"f0.jpg": "A0.jpg", "f40.jpg": "A40.jpg"})
    nodes = [Node(id="n001", label="Lobby", frame_id="A0.jpg", frame_heading=0.0, floor=1, floor_source="default",
                  video_id="vX", t=0),
             Node(id="n002", label="Hall end", frame_id="A40.jpg", frame_heading=0.0, floor=1,
                  floor_source="default", video_id="vX", t=10)]
    edges = [Edge(id="e001", from_node="n001", to_node="n002", heading_out=0, heading_in=0,
                  heading_out_raw=0, heading_in_raw=0, video_id="vX", length_m=10)]
    b = MapBundle(building_id="t", nodes=nodes, edges=edges,
                  videos=[VideoInfo(id="vX", filename="x", status="done", anchored=True)])
    res = mapper.integrate_walk(b, "vW", _walk(), _track(), tmp_path, None, False, log=_quiet)
    assert res["merged"] == 2, res
    assert {"n001", "n002"} <= {n.id for n in b.nodes}


def test_probe_consolidate_default_to_default_still_merges(monkeypatch, tmp_path):
    _fake_orb(monkeypatch, {"A.jpg": "B.jpg"})
    nodes = [Node(id="n001", label="Lobby", frame_id="A.jpg", floor=1, floor_source="default", video_id="vX", t=0),
             Node(id="n002", label="Lobby 2", frame_id="B.jpg", floor=1, floor_source="default", video_id="vY", t=0)]
    b = MapBundle(building_id="t", nodes=nodes, edges=[],
                  videos=[VideoInfo(id=v, filename=v, status="done", anchored=True) for v in ("vX", "vY")])
    mapper.consolidate(b, tmp_path, None, use_gemma=False, log=_quiet)
    assert len(b.nodes) == 1


def test_probe_unknown_node_id_in_merge_endpoints(client):
    c, bid = client
    c2 = TestClient(app, raise_server_exceptions=False)
    r = c2.post(f"/api/buildings/{bid}/merge", json={"keep": "n001", "remove": "nope"})
    assert r.status_code == 400  # guarded by the id check in manual_merge
    s = store.load_map(bid)
    s.suggested_merges[0].remove = "ghost"
    store.save_map(s)
    before = {n.id for n in store.load_map(bid).nodes}
    r = c2.post(f"/api/buildings/{bid}/merges/m001", json={"accept": True})
    assert r.status_code == 409, r.status_code  # stale suggestion with a deleted node: conflict, not a 500
    assert "ghost" in r.json()["detail"] and "out of date" in r.json()["detail"]
    after = store.load_map(bid)
    assert after.suggested_merges == [] and {n.id for n in after.nodes} == before  # dropped, nothing merged
    r = c2.post(f"/api/buildings/{bid}/merges/m001", json={"accept": True})
    assert r.status_code == 404  # it is gone now


def test_merge_nodes_unknown_id_is_key_error_and_map_unchanged():
    b = _lobby_map()
    before = b.model_dump()
    for keep, remove in (("n001", "ghost"), ("ghost", "n001")):
        with pytest.raises(KeyError, match="ghost"):
            mapper.merge_nodes(b, keep, remove)
    assert b.model_dump() == before
