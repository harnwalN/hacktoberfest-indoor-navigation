"""Owner moderation of walks (owner.py, owner_router.py): the queue, approve / reject / hide, the map effects of each, the
signed preview links, what a public map response may show, and how owner decisions interact with the pipeline.

Walks are set up the way the pipeline leaves them (files on disk, a VideoInfo in map.json, a place on the map). The
pipeline itself is exercised with the stubs of test_walk_moderation.
"""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import auth, config, jobs, moderation, owner, store
from app.main import app
from app.schemas import Node, VideoInfo
from tests.auth_helpers import owner_headers, signed_in_headers
from tests.test_walk_moderation import (  # the pipeline stubs: face blur and content check are replaced
    APPROVED, REJECTED, Moderator, _done_walk, _process, _rebuild as run_rebuild, _video, _walk as queued_walk,
    env as walk_env,
)

client = TestClient(app)
OWNER = "owner@example.com"


@pytest.fixture
def queue(monkeypatch):
    """Job submission is recorded, not run: approve decisions that need a rebuild or a reprocess are checked by calls."""
    monkeypatch.setattr(jobs, "JOBS", {})
    monkeypatch.setattr(jobs, "_ensure_worker", lambda: None)
    calls = SimpleNamespace(rebuild=[], reprocess=[])

    def rebuild(bid):
        calls.rebuild.append(bid)
        return jobs.Job(id="rb1", building_id=bid, video_id=None, kind="rebuild")

    def reprocess(bid, vid):
        calls.reprocess.append((bid, vid))
        return jobs.Job(id="rp1", building_id=bid, video_id=vid, kind="video")

    monkeypatch.setattr(jobs, "submit_rebuild", rebuild)
    monkeypatch.setattr(jobs, "submit_reprocess", reprocess)
    return calls


def _building(name="Moderation Hall"):
    return store.create_building(name).building_id


def _walk(bid, vid, *, status="done", moderation="approved", on_map=None, frames=2, source=True,
          contributor=None, email="walker@example.com", note=None, created="2026-11-01T10:00:00+00:00",
          processed=None):
    """A walk as the pipeline leaves it: its files, its keyframes (real JPEGs in the frames or held folder), a VideoInfo and,
    when it is on the map, a place that uses its first frame. Returns the keyframe file names."""
    vdir = store.video_dir(bid, vid)
    vdir.mkdir(parents=True, exist_ok=True)
    if source:
        (vdir / "source.mp4").write_bytes(b"x")
    names = [f"{vid}_{k:04d}.jpg" for k in range(frames)]
    if processed is None:
        processed = status == "done" or moderation in ("approved", "rejected") and status != "queued"
    if processed:
        (vdir / "analysis.mp4").write_bytes(b"x")
        # Where the pipeline leaves them: approved frames are public, any other walk's frames are held (not served).
        folder = store.frames_dir(bid) if moderation == "approved" else store.held_frames_dir(bid, vid)
        folder.mkdir(parents=True, exist_ok=True)
        for name in names:
            cv2.imwrite(str(folder / name), np.full((8, 8, 3), 120, np.uint8))
        walk = {"video_id": vid, "duration": 1.0, "events": [], "gemma": False, "model": None,
                "keyframes": [{"frame_id": n, "video_id": vid, "t": float(i), "kind": "heartbeat", "heading": 0.0}
                              for i, n in enumerate(names)],
                "moderation": {"approved": moderation != "rejected", "frames": frames, "rejected": [], "model": "t"}}
        store.write_json_atomic(vdir / "walk.json", walk, mkdir=False)
    if on_map is None:
        on_map = status == "done" and moderation == "approved"
    with store.lock_for(bid):
        b = store.load_map(bid)
        b.videos.append(VideoInfo(id=vid, filename=f"{vid}.mp4", status=status, duration=1.0, keyframes=frames,
                                  moderation_status=moderation, moderation_notes=note, contributor_id=contributor,
                                  uploader_email=email if contributor else "unknown", created_at=created))
        if on_map:
            b.nodes.append(Node(id=f"{vid}_n", label=f"Place {vid}", video_id=vid, frame_id=names[0] if names else None))
        store.save_map(b)
    return names


def _video_of(bid, vid):
    return next(v for v in store.load_map(bid).videos if v.id == vid)


def _decide(bid, vid, action, headers=None, **body):
    return client.post(f"/api/owner/videos/{bid}/{vid}/moderation", json={"action": action, **body},
                       headers=owner_headers() if headers is None else headers)


def _queue(bid, headers=None, status="pending"):
    return client.get(f"/api/owner/moderation?status={status}", headers=owner_headers() if headers is None else headers)


# ---------- the queue ----------
def test_queue_refuses_anonymous_and_contributors(queue):
    bid = _building()
    _walk(bid, "vq000001", status="queued", moderation="pending")
    assert _queue(bid, headers={}).status_code == 401
    assert _queue(bid, headers=signed_in_headers()).status_code == 403


def test_queue_lists_by_status_newest_first_and_all(queue):
    bid = _building()
    _walk(bid, "vqa00001", status="queued", moderation="pending", created="2026-11-01T10:00:00+00:00")
    _walk(bid, "vqb00001", status="queued", moderation="pending", created="2026-11-02T10:00:00+00:00")
    _walk(bid, "vqc00001", status="rejected", moderation="rejected", created="2026-11-03T10:00:00+00:00",
          processed=False, note="blurry")
    pending = client.get("/api/owner/moderation", headers=owner_headers()).json()  # default: pending
    assert [i["video_id"] for i in pending] == ["vqb00001", "vqa00001"]
    rejected = _queue(bid, status="rejected").json()
    assert [i["video_id"] for i in rejected] == ["vqc00001"] and rejected[0]["reason"] == "blurry"
    assert len(_queue(bid, status="all").json()) == 3
    assert _queue(bid, status="approved").json() == []
    assert client.get("/api/owner/moderation?status=live", headers=owner_headers()).status_code == 422


def test_queue_item_shape_and_legacy_uploader(queue):
    bid = _building("Shape Hall")
    _walk(bid, "vqs00001", status="queued", moderation="pending", contributor="acct1", email="walker@example.com")
    _walk(bid, "vqs00002", status="queued", moderation="pending")  # legacy: no contributor
    items = {i["video_id"]: i for i in _queue(bid).json()}
    first = items["vqs00001"]
    assert set(first) == {"building_id", "building_name", "video_id", "filename", "uploader_email", "status", "reason",
                          "frames", "created_at"}
    assert (first["building_id"], first["building_name"], first["status"]) == (bid, "Shape Hall", "pending")
    assert first["uploader_email"] == "walker@example.com"
    assert items["vqs00002"]["uploader_email"] == "unknown"
    assert first["created_at"] == "2026-11-01T10:00:00+00:00"


def test_queue_frames_are_at_most_six_and_spread_over_the_walk(queue):
    bid = _building()
    _walk(bid, "vqf00001", status="done", moderation="approved", frames=15)
    frames = _queue(bid, status="approved").json()[0]["frames"]
    assert len(frames) == 6
    assert frames[0].endswith("vqf00001_0000.jpg") and frames[-1].endswith("vqf00001_0014.jpg")


def test_published_walk_frames_use_public_urls_and_others_use_signed_links(queue):
    bid = _building()
    _walk(bid, "vqp00001", status="done", moderation="approved", frames=2)
    _walk(bid, "vqp00002", status="queued", moderation="pending", frames=2, processed=True)
    items = {i["video_id"]: i for i in _queue(bid, status="all").json()}
    assert items["vqp00001"]["frames"][0] == f"/bfiles/{bid}/frames/vqp00001_0000.jpg"
    assert items["vqp00002"]["frames"][0].startswith(f"/api/owner/videos/{bid}/vqp00002/frames/vqp00002_0000.jpg?exp=")


# ---------- signed preview links ----------
def _signed_url(items, video_id):
    return next(u for i in items if i["video_id"] == video_id for u in i["frames"] if "?exp=" in u)


def test_signed_frame_link_serves_the_pending_frame_and_nothing_else(queue):
    bid = _building()
    _walk(bid, "vsf00001", status="queued", moderation="pending", frames=2, processed=True)
    url = _signed_url(_queue(bid).json(), "vsf00001")
    r = client.get(url)  # no session header: the link is the credential
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert client.get(url.replace("vsf00001_0000.jpg", "vsf00001_0001.jpg")).status_code == 403  # other file
    assert client.get(url.replace("/vsf00001/", "/vsf00002/")).status_code == 403  # other walk
    assert client.get(url.split("?")[0]).status_code == 403  # no signature


def test_signed_frame_link_refuses_forged_and_expired_links(queue, monkeypatch):
    bid = _building()
    _walk(bid, "vsf00002", status="queued", moderation="pending", frames=1, processed=True)
    url = _signed_url(_queue(bid).json(), "vsf00002")
    forged = url.split("sig=")[0] + "sig=" + "0" * 64
    assert client.get(forged).status_code == 403
    assert client.get(url.split("sig=")[0] + "sig=\u00e9\u4e2d").status_code == 403  # not ASCII: refused, not a 500
    monkeypatch.setattr(auth, "_now", lambda: 10**12)  # far past the link's expiry
    assert client.get(url).status_code == 403


def test_signed_frame_link_rejects_traversal_names(queue):
    bid = _building()
    _walk(bid, "vsf00003", status="queued", moderation="pending", frames=1, processed=True)
    exp, sig = _signed_url(_queue(bid).json(), "vsf00003").split("?")[1].split("&")
    # a valid signature only exists for the real name, so a traversal name fails the signature check first
    assert client.get(f"/api/owner/videos/{bid}/vsf00003/frames/..%2F..%2Fx.jpg?{exp}&{sig}").status_code in (403, 404)


def test_hidden_walk_frames_are_served_from_the_archive_to_the_owner_only(queue):
    bid = _building()
    _walk(bid, "vsf00004", status="done", moderation="approved", frames=1)
    assert client.post(f"/api/owner/videos/{bid}/vsf00004/moderation", json={"action": "hide"},
                       headers=owner_headers()).status_code == 200
    url = _signed_url(_queue(bid, status="hidden").json(), "vsf00004")
    assert client.get(url).status_code == 200  # from rejected_frames/
    assert client.get(f"/bfiles/{bid}/frames/vsf00004_0000.jpg").status_code == 404  # not public


# ---------- decisions ----------
def test_decisions_are_refused_for_anonymous_and_contributors(queue):
    bid = _building()
    _walk(bid, "vdr00001", status="queued", moderation="pending")
    assert _decide(bid, "vdr00001", "approve", headers={}).status_code == 401
    assert _decide(bid, "vdr00001", "reject", headers=signed_in_headers()).status_code == 403
    assert _video_of(bid, "vdr00001").moderation_status == "pending"


def test_approve_a_queued_walk_needs_no_job_and_records_the_owner(queue):
    bid = _building()
    _walk(bid, "vdr00002", status="queued", moderation="pending", contributor="a1", email="w@example.com")
    r = _decide(bid, "vdr00002", "approve", note="  fine  ")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "approved" and body["reason"] == "fine" and body["job_id"] is None
    v = _video_of(bid, "vdr00002")
    assert (v.moderation_status, v.moderated_by, v.moderation_notes) == ("approved", OWNER, "fine")
    assert queue.rebuild == [] and queue.reprocess == []  # its own queued job will run and see the approval


def test_reject_a_live_walk_takes_it_off_the_map_and_archives_its_frames(queue):
    bid = _building()
    names = _walk(bid, "vdr00003", status="done", moderation="approved", frames=2, on_map=True)
    other = store.frames_dir(bid) / names[0]
    assert other.is_file()
    r = _decide(bid, "vdr00003", "reject", note="shows a face")
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "rejected" and r.json()["reason"] == "shows a face"
    bundle = store.load_map(bid)
    assert not [n for n in bundle.nodes if n.video_id == "vdr00003"]
    assert _video_of(bid, "vdr00003").moderation_status == "rejected"  # still listed, with its reason
    assert not any(store.frames_dir(bid).glob("vdr00003_*.jpg"))
    archived = sorted(p.name for p in (store.video_dir(bid, "vdr00003") / "rejected_frames").glob("*.jpg"))
    assert archived == names
    assert queue.rebuild == []


def test_hide_then_approve_rebuilds_the_walk_back_onto_the_map(queue):
    bid = _building()
    _walk(bid, "vdr00004", status="done", moderation="approved", on_map=True)
    assert _decide(bid, "vdr00004", "hide").json()["status"] == "hidden"
    assert not [n for n in store.load_map(bid).nodes if n.video_id == "vdr00004"]
    r = _decide(bid, "vdr00004", "approve")
    assert r.status_code == 200 and r.json()["job_id"] == "rb1"
    assert queue.rebuild == [bid]
    assert _video_of(bid, "vdr00004").moderation_status == "approved"


def test_approve_a_walk_that_was_never_processed_queues_a_reprocess(queue):
    bid = _building()
    _walk(bid, "vdr00005", status="rejected", moderation="rejected", processed=False, note="auto: blurry")
    r = _decide(bid, "vdr00005", "approve")
    assert r.status_code == 200 and r.json()["job_id"] == "rp1"
    assert queue.reprocess == [(bid, "vdr00005")]
    assert _video_of(bid, "vdr00005").moderation_status == "approved"


def test_approve_a_failed_walk_without_source_is_409(queue):
    bid = _building()
    _walk(bid, "vdr00006", status="error", moderation="pending", source=False, processed=False)
    r = _decide(bid, "vdr00006", "approve")
    assert r.status_code == 409 and "source file" in r.json()["detail"]
    assert _video_of(bid, "vdr00006").moderation_status == "pending"  # nothing changed


def test_approve_while_processing_is_409(queue):
    bid = _building()
    _walk(bid, "vdr00007", status="processing", moderation="pending", processed=False)
    r = _decide(bid, "vdr00007", "approve")
    assert r.status_code == 409
    assert _video_of(bid, "vdr00007").moderation_status == "pending"


def test_reject_while_processing_is_allowed_and_stays_off_the_map(queue):
    bid = _building()
    _walk(bid, "vdr00008", status="processing", moderation="pending", processed=False)
    assert _decide(bid, "vdr00008", "hide").status_code == 200
    assert _video_of(bid, "vdr00008").moderation_status == "hidden"


def test_decision_errors(queue):
    bid = _building()
    _walk(bid, "vdr00009", status="queued", moderation="pending")
    assert _decide(bid, "nope0001", "approve").status_code == 404
    assert _decide("no-such-building", "vdr00009", "approve").status_code == 404
    assert _decide("../etc", "vdr00009", "approve").status_code == 404
    assert _decide(bid, "vdr00009", "delete").status_code == 422
    assert _decide(bid, "vdr00009", "reject", note="x" * 1001).status_code == 422
    assert _decide(bid, "vdr00009", "reject", note="   ").json()["reason"] is None  # blank note is no note


def test_owner_token_is_recorded_as_the_moderator(queue, monkeypatch):
    monkeypatch.setattr(config, "OWNER_TOKEN", "s3cret")
    bid = _building()
    _walk(bid, "vdr00010", status="queued", moderation="pending")
    r = client.post(f"/api/owner/videos/{bid}/vdr00010/moderation", json={"action": "reject"},
                    headers={"X-Owner-Token": "s3cret"})
    assert r.status_code == 200, r.text
    assert _video_of(bid, "vdr00010").moderated_by == "owner-token"


def test_moderation_item_shows_the_decision_to_the_next_reader(queue):
    bid = _building()
    _walk(bid, "vdr00011", status="queued", moderation="pending")
    _decide(bid, "vdr00011", "reject", note="not a building")
    item = _queue(bid, status="rejected").json()[0]
    assert item["video_id"] == "vdr00011" and item["reason"] == "not a building" and item["status"] == "rejected"


# ---------- the public map never shows an uploader's email ----------
def test_public_map_hides_uploader_and_moderator_emails(queue):
    bid = _building()
    _walk(bid, "vpm00001", status="done", moderation="approved", contributor="acct9", email="private@example.com",
          on_map=True)
    _decide(bid, "vpm00001", "hide", note="takedown")
    r = client.get(f"/api/buildings/{bid}/map")
    assert r.status_code == 200
    assert "private@example.com" not in r.text and OWNER not in r.text
    video = next(v for v in r.json()["videos"] if v["id"] == "vpm00001")
    assert "uploader_email" not in video and "moderated_by" not in video
    assert (video["moderation_status"], video["moderation_notes"]) == ("hidden", "takedown")  # the public reason stays


# ---------- owner decisions are sticky against the pipeline ----------
def test_automatic_moderation_does_not_overwrite_an_owner_decision(queue):
    bid = _building()
    _walk(bid, "vst00001", status="queued", moderation="pending")
    _decide(bid, "vst00001", "hide")
    store.set_auto_moderation(bid, "vst00001", "approved", notes=None)
    v = _video_of(bid, "vst00001")
    assert (v.moderation_status, v.moderated_by) == ("hidden", OWNER)


def test_owner_hiding_a_queued_walk_stops_it_before_any_processing(walk_env, monkeypatch):
    def no_normalize(*_a, **_k):
        raise AssertionError("a walk the owner hid must not be normalized")

    queued_walk(walk_env.bid, "vqq00001")
    _decide_direct(walk_env.bid, "vqq00001", "hide")
    monkeypatch.setattr(jobs.video, "normalize", no_normalize)
    job = _process(walk_env.bid, "vqq00001")
    assert job.status == "done", job.log
    assert job.result == {"moderation": "hidden", "on_map": False}
    assert _video(walk_env.bid, "vqq00001").status == "rejected"
    assert walk_env.integrated == []


def test_owner_approval_skips_the_content_check_but_still_blurs(walk_env, monkeypatch):
    mod = Moderator([REJECTED, REJECTED])  # the content check would reject both frames
    monkeypatch.setattr(moderation, "moderate_keyframe", mod)
    blurred = []
    monkeypatch.setattr(moderation, "blur_faces", lambda frame: blurred.append(1) or np.zeros_like(frame))
    queued_walk(walk_env.bid, "vqa00002")
    _decide_direct(walk_env.bid, "vqa00002", "approve")
    job = _process(walk_env.bid, "vqa00002")
    assert job.status == "done", job.log
    assert mod.calls == []  # not checked again
    assert len(blurred) == 2  # still blurred before anything is stored
    assert _video(walk_env.bid, "vqa00002").moderation_status == "approved"
    assert walk_env.integrated == ["vqa00002"]


def test_owner_hiding_during_processing_keeps_the_walk_off_the_map(walk_env, monkeypatch):
    class HideWhileChecking(Moderator):
        def __call__(self, frame, model=None):
            if not self.calls:  # the owner hides the walk while its first frame is being checked
                store.set_video_moderation(walk_env.bid, "vqh00001", "hidden", notes="owner",
                                           moderated_by=OWNER)
            return super().__call__(frame, model)

    monkeypatch.setattr(moderation, "moderate_keyframe", HideWhileChecking([APPROVED, APPROVED]))
    queued_walk(walk_env.bid, "vqh00001")
    job = _process(walk_env.bid, "vqh00001")
    assert job.status == "done", job.log
    assert job.result == {"moderation": "hidden", "on_map": False}
    v = _video(walk_env.bid, "vqh00001")
    assert (v.status, v.moderation_status, v.moderated_by) == ("done", "hidden", OWNER)
    assert walk_env.integrated == []
    assert not any(store.frames_dir(walk_env.bid).glob("vqh00001_*.jpg"))  # frames archived
    assert len(list((store.video_dir(walk_env.bid, "vqh00001") / "rejected_frames").glob("*.jpg"))) == 2


def test_hidden_walk_is_left_out_of_a_rebuild(walk_env, monkeypatch):
    _done_walk(walk_env.bid, "vqr00001", {"approved": True, "frames": 2, "rejected": [], "model": "t"})
    _done_walk(walk_env.bid, "vqr00002", {"approved": True, "frames": 2, "rejected": [], "model": "t"})
    _decide_direct(walk_env.bid, "vqr00001", "hide")
    monkeypatch.setattr(moderation, "moderate_keyframe", Moderator())
    job = run_rebuild(walk_env.bid)
    assert job.status == "done", job.log
    assert walk_env.integrated == ["vqr00002"]  # the approved walk is rebuilt, the hidden one is not
    assert _video(walk_env.bid, "vqr00001").moderation_status == "hidden"


# ---------- helper for the pipeline tests (no HTTP) ----------
def _decide_direct(bid, vid, action):
    """The owner's decision without the HTTP layer (owner.decide), with no job submission for hide/reject."""
    owner.decide(bid, vid, action, None, OWNER)
