"""Evaluation harness (backend/eval): scorer rules on a hand-made map, and a tiny end-to-end run.

The end-to-end test renders a 2-room corridor at low resolution, ingests it through the real pipeline with
the oracle standing in for Gemma (no Ollama, OCR off) and scores it. The real-Gemma run is opt-in:
    EVAL_LIVE=1 .venv/bin/python -m pytest tests/test_eval_harness.py -k live
"""
import json
import os

import pytest

from app import config
from app.schemas import Edge, MapBundle, Node
from eval import run as eval_run
from eval.generate import generate_spec
from eval.score import score
from eval.truth import Truth


# ---------------------------------------------------------------- scorer rules
def _truth(**over) -> Truth:
    data = {
        "schema": "waypoint-eval/1", "scenario": "unit",
        "building": {"places": [
            {"id": "a", "label": "Room 101", "room": "101", "floor": 1},
            {"id": "b", "label": "Room 103", "room": "103", "floor": 1},
            {"id": "s1", "label": "Stairs", "type": "stairs", "floor": 1, "group": "A"},
            {"id": "s2", "label": "Stairs", "type": "stairs", "floor": 2, "group": "A"},
            {"id": "c", "label": "Room 201", "room": "201", "floor": 2},
        ]},
        "videos": [
            {"id": "v1", "file": "v1.mp4", "start_floor": 1, "visits": [
                {"place": "a", "t": 2.0, "kind": "look"}, {"place": "b", "t": 8.0, "kind": "look"},
                {"place": "s1", "t": 12.0}, {"place": "s2", "t": 20.0}, {"place": "c", "t": 25.0, "kind": "look"}],
             "turns": [{"t": 11.0, "delta": -90}]},
            {"id": "v2", "file": "v2.mp4", "start_floor": 1, "visits": [{"place": "a", "t": 3.0, "kind": "look"}]},
        ],
        "expect_merged": ["a"],
        "must_not_merge": [["a", "b", "different rooms"]],
        "stair_links": [["s1", "s2"]],
        "routes": [{"from": "a", "to": "c", "floor_changes": 1}],
    }
    data.update(over)
    return Truth.model_validate(data)


def _node(nid, label, floor=1, texts=(), **kw):
    return Node(id=nid, label=label, floor=floor, texts=list(texts), sign_texts=list(texts), frame_id=f"{nid}.jpg",
                **kw)


def _edge(eid, a, b, vertical="none"):
    return Edge(id=eid, from_node=a, to_node=b, heading_out=90, heading_in=90, vertical=vertical, length_m=5)


def _obs(*rows):
    return [{"t": t, "t_arrive": t - 0.5, "t_depart": t + 0.5, "node": n, "frame_id": "", "kind": "glance"}
            for t, n in rows]


def _bundle(nodes, edges):
    return MapBundle(building_id="unit", nodes=nodes, edges=edges)


def test_perfect_map_scores_full_marks():
    nodes = [_node("n1", "Room 101", texts=["ROOM 101"]), _node("n2", "Room 103", texts=["RM 103"]),
             _node("n3", "Stairs"), _node("n4", "Stairs", floor=2), _node("n5", "Room 201", floor=2, texts=["201"])]
    edges = [_edge("e1", "n1", "n2"), _edge("e2", "n2", "n3"), _edge("e3", "n3", "n4", "stairs_up"),
             _edge("e4", "n4", "n5")]
    obs = {"v1": _obs((2.0, "n1"), (8.0, "n2"), (12.0, "n3"), (20.0, "n4"), (25.0, "n5")), "v2": _obs((3.2, "n1"))}
    events = {"v1": [{"type": "turn", "start": 10.5, "end": 11.5, "delta": -84.0}], "v2": []}
    res = score(_truth(), _bundle(nodes, edges), obs, events)
    m = res["metrics"]
    assert res["hard_fail"] == []
    assert (m["rooms_found"], m["rooms_located"], m["merge_recall"], m["floor_accuracy"]) == (1.0, 1.0, 1.0, 1.0)
    assert (m["stair_links"], m["route_success"], m["turns_detected"], m["turn_error_deg"]) == (1.0, 1.0, 1.0, 6.0)
    assert m["invented_rooms"] == m["wrong_merges"] == m["missed_merges"] == m["extra_places"] == 0
    assert res["details"]["routes"][0]["floor_changes"] == 1


def test_wrong_merge_invented_room_and_missed_merge_are_caught():
    nodes = [_node("n1", "Room 101", texts=["ROOM 101"]), _node("n3", "Room 999", texts=["ROOM 999"]),
             _node("n6", "Room 101", texts=["ROOM 101"])]
    edges = [_edge("e1", "n1", "n3")]
    # v1 walks past 101 and 103 but both land in n1 (wrong merge); v2's 101 became a second place n6
    obs = {"v1": _obs((2.0, "n1"), (8.0, "n1"), (25.0, "n3")), "v2": _obs((3.0, "n6"))}
    res = score(_truth(), _bundle(nodes, edges), obs)
    m = res["metrics"]
    assert m["wrong_merges"] == 1 and res["details"]["wrong_merges"][0]["places"] == ["a", "b"]
    assert m["invented_rooms"] == 1 and res["details"]["invented"][0]["room"] == "999"
    assert m["missed_merges"] == 1 and m["merge_recall"] == 0.0
    assert set(res["hard_fail"]) == {"wrong_merges", "invented_rooms"}
    # room 103 is never claimed; n3 sits where room 201 is but claims nothing real
    assert "b" in res["details"]["missing_rooms"]
    assert m["stair_links"] == 0.0
    assert res["details"]["routes"][0]["ok"] is False


def test_cut_off_room_label_counts_as_invented():
    nodes = [_node("n1", "Room 10", texts=["ROOM 10"]), _node("n2", "Room 101 (2)", texts=["ROOM 101"])]
    res = score(_truth(), _bundle(nodes, []), {"v1": _obs((2.0, "n2")), "v2": []})
    assert [i["room"] for i in res["details"]["invented"]] == ["10"]


def test_floor_errors_misattribution_and_no_path():
    nodes = [_node("n1", "Room 103", texts=["RM 103"]), _node("n5", "Room 201", floor=1, texts=["201"])]
    obs = {"v1": _obs((2.0, "n1"), (25.0, "n5")), "v2": []}
    res = score(_truth(), _bundle(nodes, []), obs)
    d = res["details"]
    assert d["misattributed"][0]["room"] == "103" and d["misattributed"][0]["is_at"] == ["a"]
    assert {f["place"] for f in d["floor_errors"]} == {"c"}
    assert d["routes"][0]["error"] == "no path"


def test_visit_matching_respects_time_tolerance():
    nodes = [_node("n1", "Room 101", texts=["ROOM 101"])]
    obs = {"v1": _obs((5.0, "n1")), "v2": []}  # 101 was looked at t=2.0: 2.5 s outside [4.5, 5.5]
    res = score(_truth(), _bundle(nodes, []), obs)
    assert res["metrics"]["rooms_located"] == 0.0 and res["metrics"]["rooms_found"] > 0


def test_far_merge_flags_one_place_seen_at_two_distant_spots():
    truth = _truth()
    truth.videos[0].path = [[2.0, 0.0, 0.0, 1], [8.0, 6.0, 0.0, 1], [25.0, 6.0, 0.0, 2]]
    truth.videos[1].path = [[3.0, 0.5, 0.0, 1], [9.0, 20.0, 0.0, 1]]
    nodes = [_node("n1", "Room 101", texts=["ROOM 101"]), _node("n2", "Hallway")]
    # n1: v1 t=2 and v2 t=3 are 0.5 m apart (fine); n2: v1 t=8 (x=6) and v2 t=9 (x=20) are 14 m apart
    obs = {"v1": _obs((2.0, "n1"), (8.0, "n2")), "v2": _obs((3.0, "n1"), (9.0, "n2"))}
    res = score(truth, _bundle(nodes, []), obs)
    assert res["metrics"]["far_merges"] == 1
    (far,) = res["details"]["far_merges"]
    assert far["node"] == "n2" and far["distance_m"] == 14.0
    assert res["hard_fail"] == ["far_merges"]  # a gate: one place 5+ m wide is a bad merge
    assert score(_truth(), _bundle(nodes, []), obs)["metrics"]["far_merges"] is None  # no paths: not scored


def test_truth_rejects_unknown_place_ids():
    with pytest.raises(ValueError, match="unknown place"):
        _truth(routes=[{"from": "a", "to": "nowhere"}])


def test_refuses_backend_data_dir(tmp_path):
    with pytest.raises(eval_run.SetupError):
        eval_run.safe_data_dir(eval_run.BACKEND_DIR / "data" / "x")
    assert eval_run.safe_data_dir(tmp_path) == tmp_path.resolve()
    assert eval_run.main(["tiny", "--oracle", "--data-dir", str(eval_run.BACKEND_DIR / "data" / "x")]) == 2


def test_setup_errors_exit_2_without_leaking_temp_dirs(tmp_path, monkeypatch):
    made = []

    def mkdtemp(prefix=""):
        made.append(tmp_path / f"{prefix}{len(made)}")
        made[-1].mkdir()
        return str(made[-1])
    monkeypatch.setattr(eval_run.tempfile, "mkdtemp", mkdtemp)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))  # main() only sets it when unset
    assert eval_run.main(["no_such_scenario", "--oracle", "--no-save"]) == 2
    assert eval_run.main(["no_such_scenario", "--oracle", "--no-save", "--keep"]) == 2  # nothing to keep
    assert len(made) == 2 and not any(d.exists() for d in made)


# ---------------------------------------------------------------- tiny end-to-end run (oracle)
TINY = {
    "name": "tiny",
    "seed": 3,
    "size": (480, 270),
    "levels": {1: {"z": 0.0, "grid": ["..............", ".............."]}},
    "decals": [
        {"kind": "door", "id": "d101", "level": 1, "wall": [3.5, 0, "N"], "plaque": "ROOM 101", "room": "101",
         "place": {"id": "r101", "label": "Room 101"}},
        {"kind": "door", "id": "d104", "level": 1, "wall": [10.5, 2, "S"], "plaque": "RM 104", "room": "104",
         "place": {"id": "r104", "label": "Room 104"}},
        {"kind": "poster", "id": "p1", "level": 1, "wall": [7, 0, "N"], "lines": ["CALL", "555-0147"]},
    ],
    "videos": [
        {"id": "east", "start_floor": 1, "give_start_floor": True, "start": [1.0, 1.0], "heading": 90,
         "script": [{"visit": "r101"}, {"visit": "r104"}, {"go": [13.0, 1.0]}]},
        {"id": "west", "start_floor": 1, "give_start_floor": False, "start": [13.0, 1.0], "heading": 270,
         "script": [{"visit": "r104"}, {"visit": "r101"}, {"go": [1.0, 1.0]}]},
    ],
    "routes": [{"from": "r101", "to": "r104", "floor_changes": 0}],
}


@pytest.fixture
def tiny_truth(tmp_path):
    return generate_spec(TINY, tmp_path / "scenario", workers=2)


def test_tiny_scenario_end_to_end_with_oracle(tiny_truth, tmp_path, monkeypatch):
    truth = json.loads(tiny_truth.read_text())
    assert [v["place"] for v in truth["videos"][0]["visits"]] == ["r101", "r104"]
    assert ["r101", "r104", "different rooms 101 / 104"] in truth["must_not_merge"]
    monkeypatch.setattr(config, "OCR_ENABLED", False)  # portable and fast: the oracle stands in for OCR too
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(eval_run, "RESULTS_DIR", tmp_path / "results")
    code = eval_run.main([str(tiny_truth), "--oracle", "--data-dir", str(tmp_path / "data")])
    assert code == 0
    (saved,) = (tmp_path / "results").glob("*-tiny-oracle.json")
    res = json.loads(saved.read_text())
    m = res["metrics"]
    assert m["rooms_found"] == 1.0 and m["invented_rooms"] == 0 and m["wrong_merges"] == 0
    assert m["route_success"] == 1.0
    assert set(res["observations"]) == {"east", "west"}
    assert config.BUILDINGS_DIR != tmp_path / "data" / "buildings"  # restored after the run
    # a second run compares with the first
    assert eval_run.previous_result("tiny", "oracle", None, None)["metrics"] == m


@pytest.mark.skipif(os.getenv("EVAL_LIVE") != "1", reason="live Gemma run: set EVAL_LIVE=1 (needs Ollama)")
def test_live_gemma_small_loop():
    code = eval_run.main(["small_loop"])
    assert code != 2, "pipeline or setup error"
