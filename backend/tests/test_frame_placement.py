"""Where a walk's frames are kept, by moderation status. Public (/bfiles) only for approved walks. Held (never served) for
every other walk. Raw keyframes only in the staging folder, and only while the walk is checked. Covers the pipeline, the
owner's decisions, the signed owner links, the startup settle, and the frame urls of the debug view."""
import numpy as np
from fastapi.testclient import TestClient

from app import jobs, moderation, store
from app.main import app
from app.schemas import Edge
from tests.test_owner_moderation import (  # noqa: F401
    _building,
    _decide,
    _queue,
    _signed_url,
    _walk,
    queue,
)
from tests.test_walk_moderation import APPROVED, _process, env  # noqa: F401
from tests.test_walk_moderation import _walk as _pipeline_walk

client = TestClient(app)


def _names(folder, vid):
    """Keyframe file names of one walk in a folder (an absent folder has none)."""
    return sorted(p.name for p in folder.glob(f"{vid}_*.jpg"))


def _to_public_folder(bid, vid):
    """Put a walk's held frames back in the public folder: the layout before this change."""
    store.frames_dir(bid).mkdir(parents=True, exist_ok=True)
    for name in _names(store.held_frames_dir(bid, vid), vid):
        (store.held_frames_dir(bid, vid) / name).replace(store.frames_dir(bid) / name)


# ---------- the pipeline ----------
def test_frames_of_a_walk_under_check_are_never_public_or_held(env, monkeypatch):
    seen = []

    def moderate(frame, model=None):
        seen.append((_names(store.frames_dir(env.bid), "vchk0001"),
                     _names(store.held_frames_dir(env.bid, "vchk0001"), "vchk0001")))
        return np.zeros_like(frame), APPROVED

    monkeypatch.setattr(moderation, "moderate_keyframe", moderate)
    _pipeline_walk(env.bid, "vchk0001")
    job = _process(env.bid, "vchk0001")
    assert job.status == "done", job.log
    assert seen == [([], [])] * 2  # both raw frames were in staging while checked
    assert _names(store.frames_dir(env.bid), "vchk0001") == ["vchk0001_0000.jpg", "vchk0001_0001.jpg"]
    assert _names(store.staging_frames_dir(env.bid, "vchk0001"), "vchk0001") == []


# ---------- owner decisions ----------
def test_hide_moves_frames_out_of_the_public_folder_and_approve_brings_them_back(queue):
    bid = _building("Hide Frames Hall")
    names = _walk(bid, "vhf00001", status="done", moderation="approved", frames=2)
    public_url = f"/bfiles/{bid}/frames/{names[0]}"
    assert client.get(public_url).status_code == 200
    assert _decide(bid, "vhf00001", "hide").status_code == 200
    assert client.get(public_url).status_code == 404
    assert _names(store.held_frames_dir(bid, "vhf00001"), "vhf00001") == names
    assert _decide(bid, "vhf00001", "approve").status_code == 200
    assert client.get(public_url).status_code == 200
    assert _names(store.held_frames_dir(bid, "vhf00001"), "vhf00001") == []


def test_frame_shared_with_another_walks_place_stays_public_when_its_walk_is_hidden(queue):
    bid = _building("Shared Frame Hall")
    names_a = _walk(bid, "vsa00001", status="done", moderation="approved", frames=2)  # node on names_a[0]
    _walk(bid, "vsb00001", status="done", moderation="approved", frames=1)
    b = store.load_map(bid)
    # walk B's edge leaves from walk A's place, so that place stays on the map (re-owned by B)
    b.edges.append(Edge(id="eab1", from_node="vsa00001_n", to_node="vsb00001_n", heading_out=0, heading_in=0,
                        video_id="vsb00001", frames_fwd=[names_a[0]]))
    store.save_map(b)
    assert _decide(bid, "vsa00001", "hide").status_code == 200
    assert client.get(f"/bfiles/{bid}/frames/{names_a[0]}").status_code == 200  # still used by B's map
    assert client.get(f"/bfiles/{bid}/frames/{names_a[1]}").status_code == 404  # used by nothing else
    assert _names(store.held_frames_dir(bid, "vsa00001"), "vsa00001") == [names_a[1]]


# ---------- the owner's signed links ----------
def test_owner_signed_links_serve_frames_of_every_unapproved_status(queue):
    bid = _building("Signed Status Hall")
    _walk(bid, "vsp00001", status="queued", moderation="pending", frames=1, processed=True)
    _walk(bid, "vsh00001", status="done", moderation="hidden", frames=1)
    _walk(bid, "vsr00001", status="rejected", moderation="rejected", frames=1, processed=True)
    items = _queue(bid, status="all").json()
    for vid in ("vsp00001", "vsh00001", "vsr00001"):
        assert client.get(_signed_url(items, vid)).status_code == 200, vid
        assert client.get(f"/bfiles/{bid}/frames/{vid}_0000.jpg").status_code == 404, vid


# ---------- startup settle ----------
def test_settle_moves_frames_to_match_status_and_can_run_again(queue):
    bid = _building("Settle Hall")
    pending = _walk(bid, "vst00001", status="queued", moderation="pending", frames=2, processed=True)
    approved = _walk(bid, "vst00002", status="done", moderation="approved", frames=2)
    _to_public_folder(bid, "vst00001")  # the old layout: pending frames were public
    held_approved = store.held_frames_dir(bid, "vst00002")
    held_approved.mkdir(parents=True, exist_ok=True)
    for name in approved:  # an approved walk whose frames never reached the public folder
        (store.frames_dir(bid) / name).replace(held_approved / name)
    raw = store.staging_frames_dir(bid, "vst00001")  # left behind by an interrupted check
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "vst00001_0000.jpg").write_bytes(b"raw")

    assert jobs.settle_walk_frames() == 4
    assert client.get(f"/bfiles/{bid}/frames/{pending[0]}").status_code == 404
    assert client.get(f"/bfiles/{bid}/frames/{approved[0]}").status_code == 200
    assert not raw.exists()
    assert _names(store.held_frames_dir(bid, "vst00001"), "vst00001") == pending
    assert jobs.settle_walk_frames() == 0  # nothing left to move


def test_settle_keeps_a_hidden_walks_frame_that_the_map_still_uses(queue):
    bid = _building("Settle Shared Hall")
    names = _walk(bid, "vss00001", status="done", moderation="hidden", frames=2)
    _walk(bid, "vss00002", status="done", moderation="approved", frames=1)
    _to_public_folder(bid, "vss00001")
    b = store.load_map(bid)
    b.edges.append(Edge(id="ess1", from_node="vss00002_n", to_node="vss00001_n", heading_out=0, heading_in=0,
                        video_id="vss00002", frames_fwd=[names[0]]))  # the approved walk's edge still shows names[0]
    store.save_map(b)
    jobs.settle_walk_frames()
    assert client.get(f"/bfiles/{bid}/frames/{names[0]}").status_code == 200
    assert client.get(f"/bfiles/{bid}/frames/{names[1]}").status_code == 404


# ---------- the debug view ----------
def test_debug_view_gives_frame_urls_only_for_approved_walks(queue):
    bid = _building("Debug Hall")
    _walk(bid, "vdb00001", status="done", moderation="approved", frames=1)
    _walk(bid, "vdb00002", status="done", moderation="hidden", frames=1)
    for vid in ("vdb00001", "vdb00002"):
        store.write_json_atomic(store.video_dir(bid, vid) / "track.json",
                                {"t": [0.0, 1.0], "heading": [0.0, 0.0], "flow": [0.0, 0.0], "events": []},
                                mkdir=False)
    shown = client.get(f"/api/buildings/{bid}/videos/vdb00001/debug").json()["keyframes"][0]["url"]
    assert shown == f"/bfiles/{bid}/frames/vdb00001_0000.jpg"
    hidden = client.get(f"/api/buildings/{bid}/videos/vdb00002/debug").json()["keyframes"][0]["url"]
    assert hidden is None
