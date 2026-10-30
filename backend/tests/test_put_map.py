"""PUT /map: server-owned fields come from disk, and the save serializes with jobs via the building lock."""
import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import config, router, store
from app.schemas import Edge, Node, SuggestedMerge, VideoInfo


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BUILDINGS_DIR", tmp_path / "buildings")
    app = FastAPI()
    app.include_router(router.api)  # bare router: no startup recovery, no static mounts
    bundle = store.create_building("Test Hall")
    bundle.nodes = [
        Node(id="n001", label="Lobby", video_id="v1"),
        Node(id="n002", label="Room 101", video_id="v1"),
        Node(id="n003", label="Stairs", video_id="v1"),
    ]
    bundle.edges = [
        Edge(id="e001", from_node="n001", to_node="n002", heading_out=0, heading_in=0, video_id="v1"),
        Edge(id="e002", from_node="n002", to_node="n003", heading_out=90, heading_in=90, video_id="v1"),
    ]
    bundle.videos = [
        VideoInfo(id="v1", filename="walk1.mp4", duration=42.0, keyframes=12, status="done",
                  start_floor=2, floor_offset=2, anchored=True),
        VideoInfo(id="v2", filename="walk2.mp4", status="processing"),
    ]
    bundle.processed_with = {"model": "gemma3:4b", "prompt": 7}
    bundle.suggested_merges = [
        SuggestedMerge(id="m1", keep="n001", remove="n003", confidence=0.6, reason="same sign"),
    ]
    store.save_map(bundle)
    return {"client": TestClient(app), "bid": bundle.building_id}


def _get(env) -> dict:
    r = env["client"].get(f"/api/buildings/{env['bid']}/map")
    assert r.status_code == 200
    return r.json()


def _put(env, body: dict):
    return env["client"].put(f"/api/buildings/{env['bid']}/map", json=body)


def test_normal_editor_save_persists(env):
    m = _get(env)
    m["nodes"] = [n | {"label": "Main Lobby"} if n["id"] == "n001" else n for n in m["nodes"]]
    m["edges"] = [e for e in m["edges"] if e["id"] != "e002"]
    r = _put(env, m)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["map"]["building_id"] == env["bid"]  # response shape unchanged
    disk = store.load_map(env["bid"])
    assert next(n for n in disk.nodes if n.id == "n001").label == "Main Lobby"
    assert [e.id for e in disk.edges] == ["e001"]


def test_stale_videos_and_processed_with_are_ignored(env):
    before = store.load_map(env["bid"])
    m = _get(env)
    m["videos"] = [
        {"id": "v1", "filename": "walk1.mp4", "status": "queued", "floor_offset": 0, "duration": 0},
        {"id": "..", "filename": "evil.mp4", "status": "queued"},
    ]
    m["processed_with"] = {"model": "stale"}
    m["name"] = "Renamed by client"
    m["building_id"] = "some-other-building"
    m["nodes"][1]["label"] = "Room 102"
    r = _put(env, m)
    assert r.status_code == 200
    disk = store.load_map(env["bid"])
    assert disk.videos == before.videos
    assert disk.processed_with == before.processed_with
    assert disk.name == before.name and disk.building_id == env["bid"]
    assert next(n for n in disk.nodes if n.id == "n002").label == "Room 102"  # the editor edit still lands
    assert r.json()["map"]["videos"] == [v.model_dump() for v in before.videos]


def test_frontend_style_body_without_server_fields(env):
    m = _get(env)
    for k in ("videos", "processed_with", "suggested_merges", "name", "dominant_axis_deg"):
        m.pop(k)
    assert _put(env, m).status_code == 200
    disk = store.load_map(env["bid"])
    assert [v.id for v in disk.videos] == ["v1", "v2"]
    assert [s.id for s in disk.suggested_merges] == ["m1"]


def test_suggested_merges_come_from_disk_and_drop_deleted_nodes(env):
    m = _get(env)
    m["suggested_merges"] = []  # stale client copy: ignored
    assert _put(env, m).status_code == 200
    assert [s.id for s in store.load_map(env["bid"]).suggested_merges] == ["m1"]

    m["nodes"] = [n for n in m["nodes"] if n["id"] != "n003"]
    assert _put(env, m).status_code == 200
    disk = store.load_map(env["bid"])
    assert disk.suggested_merges == []
    assert all("n003" not in (e.from_node, e.to_node) for e in disk.edges)


def test_unknown_building_404(env):
    m = _get(env)
    r = env["client"].put("/api/buildings/nope/map", json=m)
    assert r.status_code == 404


def test_put_waits_for_building_lock_and_keeps_job_update(env):
    """A job holding the lock finishes its save first; the PUT then applies on top without undoing it."""
    bid = env["bid"]
    m = _get(env)  # editor loaded while v2 was still processing
    m["nodes"][0]["label"] = "Edited Lobby"

    lock = store.lock_for(bid)
    lock.acquire()  # the "job" is integrating a walk
    done = threading.Event()
    result = {}

    def do_put():
        result["resp"] = _put(env, m)
        done.set()

    t = threading.Thread(target=do_put, daemon=True)
    try:
        t.start()
        assert not done.wait(0.3), "PUT must block while the building lock is held"
        # Job saves under the lock: v2 finished, processed_with updated.
        job_bundle = store.load_map(bid)
        v2 = next(v for v in job_bundle.videos if v.id == "v2")
        v2.status, v2.duration, v2.floor_offset = "done", 30.0, 1
        job_bundle.processed_with = {"model": "gemma3:12b", "prompt": 8}
        store.save_map(job_bundle)
    finally:
        lock.release()
    assert done.wait(10), "PUT should complete once the lock is released"
    t.join(5)

    assert result["resp"].status_code == 200
    disk = store.load_map(bid)
    v2 = next(v for v in disk.videos if v.id == "v2")
    assert (v2.status, v2.duration, v2.floor_offset) == ("done", 30.0, 1)  # job's update survived
    assert disk.processed_with == {"model": "gemma3:12b", "prompt": 8}
    assert next(n for n in disk.nodes if n.id == "n001").label == "Edited Lobby"  # editor's edit landed too


def test_lock_is_reentrant_for_save(env):
    """save_map re-takes lock_for(building) inside the PUT's critical section; it must be the same RLock."""
    bid = env["bid"]
    assert store.lock_for(bid) is store.lock_for(bid)
    with store.lock_for(bid):
        store.save_map(store.load_map(bid))  # would deadlock with a plain Lock


# ---- malformed / edge-case bodies ----
def _snapshot(env) -> str:
    return store.load_map(env["bid"]).model_dump_json()


def test_body_missing_nodes_and_edges_is_422_and_disk_untouched(env):
    """{"building_id": ...} alone used to validate (nodes/edges defaulted to []) and wipe the map."""
    before = _snapshot(env)
    r = _put(env, {"building_id": env["bid"]})
    assert r.status_code == 422
    locs = {tuple(e["loc"]) for e in r.json()["detail"]}
    assert ("body", "nodes") in locs and ("body", "edges") in locs
    assert _snapshot(env) == before


@pytest.mark.parametrize("missing", ["nodes", "edges"])
def test_full_body_missing_nodes_or_edges_is_422_and_disk_untouched(env, missing):
    before = _snapshot(env)
    m = _get(env)
    m.pop(missing)
    r = _put(env, m)
    assert r.status_code == 422
    assert [tuple(e["loc"]) for e in r.json()["detail"]] == [("body", missing)]
    assert _snapshot(env) == before


def test_minimal_body_nodes_and_edges_only_is_accepted(env):
    """building_id and every other MapBundle field are optional/ignored; only nodes + edges are required."""
    m = _get(env)
    r = _put(env, {"nodes": m["nodes"], "edges": m["edges"]})
    assert r.status_code == 200
    disk = store.load_map(env["bid"])
    assert len(disk.nodes) == 3 and [e.id for e in disk.edges] == ["e001", "e002"]
    assert [v.id for v in disk.videos] == ["v1", "v2"]


def test_empty_nodes_list_removes_everything_but_keeps_videos(env):
    m = _get(env)
    m["nodes"] = []
    assert _put(env, m).status_code == 200
    disk = store.load_map(env["bid"])
    assert disk.nodes == [] and disk.edges == [] and disk.suggested_merges == []
    assert len(disk.videos) == 2


def test_dangling_and_self_loop_edges_dropped(env):
    m = _get(env)
    m["edges"].append({**m["edges"][0], "id": "e900", "to_node": "ghost"})
    m["edges"].append({**m["edges"][0], "id": "e901", "to_node": m["edges"][0]["from_node"]})
    assert _put(env, m).status_code == 200
    assert [e.id for e in store.load_map(env["bid"]).edges] == ["e001", "e002"]


@pytest.mark.parametrize("bad", [{"building_id": "x", "nodes": "nope"}, {"nodes": []}, {"building_id": "x", "nodes": [{"label": "no id"}]}])
def test_invalid_body_422_and_disk_untouched(env, bad):
    r = _put(env, bad)
    assert r.status_code == 422
    assert len(store.load_map(env["bid"]).nodes) == 3


def test_concurrent_puts_and_job_saves_no_deadlock(env):
    bid = env["bid"]
    m = _get(env)
    errs = []

    def put():
        try:
            for _ in range(10):
                assert _put(env, m).status_code == 200
        except Exception as e:  # noqa
            errs.append(e)

    def job():
        try:
            for i in range(20):
                with store.lock_for(bid):
                    b = store.load_map(bid)
                    b.videos[1].duration = float(i)
                    store.save_map(b)
        except Exception as e:  # noqa
            errs.append(e)

    ts = [threading.Thread(target=f, daemon=True) for f in (put, put, job, job)]
    [t.start() for t in ts]
    [t.join(30) for t in ts]
    assert not any(t.is_alive() for t in ts), "deadlock"
    assert not errs
    assert store.load_map(bid).videos[1].id == "v2"
