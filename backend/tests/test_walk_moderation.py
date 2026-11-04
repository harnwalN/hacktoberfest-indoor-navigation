"""Walk moderation in the pipeline (jobs.py): every keyframe is face-blurred and checked before Gemma or the map see it;
a walk with any rejected frame never reaches the map (its blurred frames are archived); a blur failure fails closed;
rebuilds never re-check legacy walks, re-blur approved ones, and skip rejected ones.

Face blur and the vision model are replaced (moderation.moderate_keyframe / blur_faces): this file tests the gating,
not the models. ffmpeg, optical flow, keyframe extraction, Gemma and the map merge are faked as in test_jobs.py."""
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import config, gemma, jobs, mapper, moderation, motion, store, video
from app.main import app
from app.moderation import ModerationVerdict
from app.schemas import Keyframe, Node, VideoInfo
from tests.auth_helpers import signed_in_headers

RAW = 200  # pixel value of a raw keyframe in these tests
APPROVED = ModerationVerdict(approved=True, reason="a hallway with room doors", model="test-model")
REJECTED = ModerationVerdict(approved=False, reason="inappropriate content: nudity", model="test-model")
FAILED = ModerationVerdict(approved=False, reason="content check failed (rejected)", notes="model down",
                           model="test-model", error=True)


def _mean(path) -> float:
    img = cv2.imread(str(path))
    assert img is not None, f"not a readable image: {path}"
    return float(img.mean())


class Moderator:
    """Stands in for moderation.moderate_keyframe: returns an all-zero (blurred) frame and the verdict for that call."""

    def __init__(self, verdicts=(), raise_on=None):
        self.verdicts = list(verdicts)
        self.raise_on = raise_on
        self.calls: list[np.ndarray] = []

    def __call__(self, frame, model=None):
        n = len(self.calls)
        self.calls.append(frame.copy())
        if self.raise_on == n:
            raise moderation.FaceBlurError("face model cannot run")
        verdict = self.verdicts[n] if n < len(self.verdicts) else APPROVED
        return np.zeros_like(frame), verdict


@pytest.fixture
def env(tmp_path, monkeypatch, _temp_data_dir):
    monkeypatch.setattr(config, "BUILDINGS_DIR", tmp_path / "buildings")
    monkeypatch.setattr(jobs, "JOBS", {})
    monkeypatch.setattr(jobs, "_ensure_worker", lambda: None)
    monkeypatch.setattr(motion, "compute_track", lambda *_a, **_k: {"t": [0.0, 1.0]})
    monkeypatch.setattr(motion, "detect_events", lambda _track: [])
    monkeypatch.setattr(video, "normalize", lambda src, vdir: (vdir / "analysis.mp4", vdir / "hq.mp4", 1.0))
    monkeypatch.setattr(mapper, "consolidate", lambda *_a, **_k: {})
    bid = store.create_building("Moderation Hall").building_id
    state = SimpleNamespace(bid=bid, analyzed=[], integrated=[], blur_calls=0)

    def fake_keyframes(_hq, _track, _events, _duration, vid, frames_out, log=print):
        kfs = []
        for k in range(2):
            name = f"{vid}_{k:04d}.jpg"
            cv2.imwrite(str(frames_out / name), np.full((16, 16, 3), RAW, np.uint8))  # raw, unblurred
            kfs.append(Keyframe(frame_id=name, video_id=vid, t=float(k), kind="heartbeat", heading=0))
        return kfs

    def fake_integrate(bundle, vid, keyframes, *_a, **_k):
        state.integrated.append(vid)
        bundle.nodes.append(Node(id=f"{vid}_n", label=vid, video_id=vid, frame_id=keyframes[0].frame_id))
        return {"floor_offset": 0}

    def fake_analyze(job, keyframes, model):
        # What Gemma would read: the stored frame files, as they are at this point of the pipeline.
        state.analyzed.append({k.frame_id: _mean(store.frames_dir(bid) / k.frame_id) for k in keyframes})
        return jobs.GemmaRun(False, None, False)

    monkeypatch.setattr(jobs, "select_keyframes", fake_keyframes)
    monkeypatch.setattr(mapper, "integrate_walk", fake_integrate)
    monkeypatch.setattr(jobs, "_analyze_walk", fake_analyze)
    def no_model(*_a, **_k):
        raise gemma.GemmaUnavailable("off in tests")

    monkeypatch.setattr(gemma, "resolve_model", no_model)
    return state


def _walk(bid, vid, status="queued"):
    """A queued walk with its source file and a map entry, as the upload route leaves it."""
    vdir = store.video_dir(bid, vid)
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "source.mp4").write_bytes(b"x")
    b = store.load_map(bid)
    b.videos.append(VideoInfo(id=vid, filename=f"{vid}.mp4", status=status))
    store.save_map(b)
    return vdir


def _process(bid, vid) -> jobs.Job:
    job = jobs.Job(id=f"j{vid}", building_id=bid, video_id=vid, kind="video")
    src = store.video_dir(bid, vid) / "source.mp4"
    job.fn = lambda: jobs._process_video(job, src, None, None)
    job.status = "running"
    jobs._run(job)
    return job


def _video(bid, vid) -> VideoInfo:
    return next(v for v in store.load_map(bid).videos if v.id == vid)


def _frame_files(bid, vid) -> list[str]:
    return sorted(p.name for p in store.frames_dir(bid).glob(f"{vid}_*.jpg"))


# ---------- the pipeline ----------
def test_approved_walk_reaches_the_map(env, monkeypatch):
    monkeypatch.setattr(moderation, "moderate_keyframe", Moderator([APPROVED, APPROVED]))
    _walk(env.bid, "vok00001")
    job = _process(env.bid, "vok00001")
    assert job.status == "done", job.log
    v = _video(env.bid, "vok00001")
    assert (v.status, v.moderation_status, v.moderation_notes) == ("done", "approved", None)
    assert env.integrated == ["vok00001"]
    walk = store.read_json(store.video_dir(env.bid, "vok00001") / "walk.json")
    assert walk["moderation"]["approved"] is True and walk["moderation"]["frames"] == 2
    assert walk["moderation"]["rejected"] == []


def test_stored_frames_are_the_blurred_copies_and_gemma_reads_them(env, monkeypatch):
    mod = Moderator([APPROVED, APPROVED])
    monkeypatch.setattr(moderation, "moderate_keyframe", mod)
    _walk(env.bid, "vblur001")
    _process(env.bid, "vblur001")
    assert all(abs(float(f.mean()) - RAW) < 2 for f in mod.calls)  # given the raw frame (moderate_keyframe blurs it first)
    assert _frame_files(env.bid, "vblur001") == ["vblur001_0000.jpg", "vblur001_0001.jpg"]
    for name in _frame_files(env.bid, "vblur001"):
        assert _mean(store.frames_dir(env.bid) / name) < 5  # the blurred (zero) copy replaced the raw frame
    assert len(env.analyzed) == 1 and all(m < 5 for m in env.analyzed[0].values())  # Gemma got blurred frames


def test_rejected_frame_keeps_walk_off_the_map(env, monkeypatch):
    monkeypatch.setattr(moderation, "moderate_keyframe", Moderator([APPROVED, REJECTED]))
    _walk(env.bid, "vrej0001")
    job = _process(env.bid, "vrej0001")
    assert job.status == "done", job.log  # the job finished: the walk was judged, not broken
    assert job.result == {"moderation": "rejected", "rejected_frames": 1}
    v = _video(env.bid, "vrej0001")
    assert (v.status, v.moderation_status) == ("rejected", "rejected")
    assert "inappropriate content: nudity" in v.moderation_notes
    assert v.moderation_notes.startswith("Automatic moderation rejected 1 of 2 frame(s).")
    assert env.integrated == [] and env.analyzed == []  # no map merge, no Gemma call
    assert not [n for n in store.load_map(env.bid).nodes if n.video_id == "vrej0001"]
    assert _frame_files(env.bid, "vrej0001") == []  # out of the public frames folder ...
    archive = store.video_dir(env.bid, "vrej0001") / "rejected_frames"
    assert sorted(p.name for p in archive.glob("*.jpg")) == ["vrej0001_0000.jpg", "vrej0001_0001.jpg"]  # ... archived
    assert all(_mean(p) < 5 for p in archive.glob("*.jpg"))  # archived copies are the blurred ones
    walk = store.read_json(store.video_dir(env.bid, "vrej0001") / "walk.json")
    assert walk["moderation"]["approved"] is False
    assert [r["frame_id"] for r in walk["moderation"]["rejected"]] == ["vrej0001_0001.jpg"]


def test_one_rejected_frame_of_three_rejects_the_walk(env, monkeypatch):
    monkeypatch.setattr(moderation, "moderate_keyframe", Moderator([APPROVED, APPROVED, REJECTED]))
    orig = jobs.select_keyframes

    def three(*args, **kwargs):
        kfs = orig(*args, **kwargs)
        vid = args[4]
        cv2.imwrite(str(args[5] / f"{vid}_0002.jpg"), np.full((16, 16, 3), RAW, np.uint8))
        return kfs + [Keyframe(frame_id=f"{vid}_0002.jpg", video_id=vid, t=2.0, kind="heartbeat", heading=0)]

    monkeypatch.setattr(jobs, "select_keyframes", three)
    _walk(env.bid, "vthree01")
    _process(env.bid, "vthree01")
    assert _video(env.bid, "vthree01").moderation_status == "rejected"
    assert env.integrated == []
    assert len(list((store.video_dir(env.bid, "vthree01") / "rejected_frames").glob("*.jpg"))) == 3


def test_failed_content_check_rejects_the_walk(env, monkeypatch):
    monkeypatch.setattr(moderation, "moderate_keyframe", Moderator([FAILED, FAILED]))
    _walk(env.bid, "vfail001")
    _process(env.bid, "vfail001")
    v = _video(env.bid, "vfail001")
    assert v.moderation_status == "rejected"
    assert "content check failed" in v.moderation_notes
    assert env.integrated == []


def test_face_blur_failure_fails_closed(env, monkeypatch):
    monkeypatch.setattr(moderation, "moderate_keyframe", Moderator(raise_on=1))
    _walk(env.bid, "vblur002")
    job = _process(env.bid, "vblur002")
    assert job.status == "error" and "FaceBlurError" in job.error
    assert _frame_files(env.bid, "vblur002") == []  # the blurred first frame and the raw second one are both gone
    v = _video(env.bid, "vblur002")
    assert (v.status, v.moderation_status) == ("error", "pending")
    assert env.integrated == [] and env.analyzed == []
    assert not (store.video_dir(env.bid, "vblur002") / "walk.json").exists()


def test_unreadable_keyframe_fails_closed(env, monkeypatch):
    monkeypatch.setattr(moderation, "moderate_keyframe", Moderator())
    orig = jobs.select_keyframes

    def corrupt(*args, **kwargs):
        kfs = orig(*args, **kwargs)
        (args[5] / kfs[1].frame_id).write_bytes(b"not a jpeg")
        return kfs

    monkeypatch.setattr(jobs, "select_keyframes", corrupt)
    _walk(env.bid, "vcorr001")
    job = _process(env.bid, "vcorr001")
    assert job.status == "error" and "cannot read keyframe" in job.error
    assert _frame_files(env.bid, "vcorr001") == []


def test_moderation_module_is_called_once_per_keyframe(env, monkeypatch):
    mod = Moderator([APPROVED, APPROVED])
    monkeypatch.setattr(moderation, "moderate_keyframe", mod)
    _walk(env.bid, "vcount01")
    _process(env.bid, "vcount01")
    assert len(mod.calls) == 2


# ---------- rebuilds ----------
def _done_walk(bid, vid, moderation_record=None, frame_ids=None, status="done"):
    """A walk already on the map. moderation_record None = published before moderation existed (legacy)."""
    vdir = store.video_dir(bid, vid)
    vdir.mkdir(parents=True, exist_ok=True)
    for name in ("source.mp4", "analysis.mp4", "hq.mp4"):
        (vdir / name).write_bytes(b"x")
    ids = frame_ids or [f"{vid}_0000.jpg", f"{vid}_0001.jpg"]
    walk = {"video_id": vid, "duration": 1.0, "events": [], "gemma": False, "model": None,
            "keyframes": [{"frame_id": f, "video_id": vid, "t": 0.0, "kind": "heartbeat", "heading": 0.0}
                          for f in ids]}
    if moderation_record is not None:
        walk["moderation"] = moderation_record
    store.write_json_atomic(vdir / "walk.json", walk, mkdir=False)
    b = store.load_map(bid)
    b.videos.append(VideoInfo(id=vid, filename=f"{vid}.mp4", status=status, duration=1.0,
                              moderation_status="approved" if moderation_record is None or moderation_record.get("approved")
                              else "rejected"))
    if status == "done":
        b.nodes.append(Node(id=f"{vid}_n", label=vid, video_id=vid))
    store.save_map(b)


def _rebuild(bid) -> jobs.Job:
    job = jobs.Job(id="rb", building_id=bid, video_id=None, kind="rebuild")
    job.fn = lambda: jobs._rebuild(job)
    job.status = "running"
    jobs._run(job)
    return job


APPROVED_RECORD = {"approved": True, "frames": 2, "rejected": [], "model": "test-model"}


def test_rebuild_does_not_recheck_legacy_walks(env, monkeypatch):
    mod = Moderator()
    monkeypatch.setattr(moderation, "moderate_keyframe", mod)
    monkeypatch.setattr(moderation, "blur_faces", lambda f: pytest.fail("legacy walk must not be blurred"))
    _done_walk(env.bid, "vleg0001")  # no moderation record: published before moderation existed
    job = _rebuild(env.bid)
    assert job.status == "done", job.log
    assert mod.calls == []
    assert env.integrated == ["vleg0001"]
    assert _video(env.bid, "vleg0001").moderation_status == "approved"


def test_rebuild_reblurs_approved_walk_without_a_new_check(env, monkeypatch):
    mod = Moderator()
    monkeypatch.setattr(moderation, "moderate_keyframe", mod)
    blurred = []

    def blur(frame):
        blurred.append(frame.copy())
        return np.zeros_like(frame)

    monkeypatch.setattr(moderation, "blur_faces", blur)
    _done_walk(env.bid, "vreb0001", APPROVED_RECORD, ["vreb0001_0000.jpg", "vreb0001_0001.jpg"])
    job = _rebuild(env.bid)
    assert job.status == "done", job.log
    assert mod.calls == [] and len(blurred) == 2  # blurred again, not checked again
    for name in ("vreb0001_0000.jpg", "vreb0001_0001.jpg"):
        assert _mean(store.frames_dir(env.bid) / name) < 5  # the regenerated raw frames were blurred again
    assert env.integrated == ["vreb0001"]


def test_rebuild_checks_again_when_keyframes_differ(env, monkeypatch):
    monkeypatch.setattr(moderation, "moderate_keyframe", Moderator([APPROVED, REJECTED]))
    _done_walk(env.bid, "vdiff001", APPROVED_RECORD, ["vdiff001_0000.jpg"])  # approved set had one frame
    job = _rebuild(env.bid)
    assert job.status == "done", job.log
    assert _video(env.bid, "vdiff001").moderation_status == "rejected"
    assert env.integrated == []  # the new frame set was judged, and one frame failed


def test_rejected_walk_is_skipped_by_rebuild(env, monkeypatch):
    mod = Moderator()
    monkeypatch.setattr(moderation, "moderate_keyframe", mod)
    rejected = {"approved": False, "frames": 2, "rejected": [{"frame_id": "x", "reason": "r", "notes": ""}],
                "model": "test-model"}
    _done_walk(env.bid, "vrbj0001", rejected, status="rejected")
    job = _rebuild(env.bid)
    assert job.status == "done", job.log
    assert mod.calls == [] and env.integrated == []
    assert _video(env.bid, "vrbj0001").moderation_status == "rejected"


def test_rebuild_rejects_a_walk_that_fails_the_check(env, monkeypatch):
    monkeypatch.setattr(moderation, "moderate_keyframe", Moderator([REJECTED, APPROVED]))
    _done_walk(env.bid, "vnew0001", APPROVED_RECORD, ["vnew0001_0000.jpg"])  # approved set differs: judged again
    _done_walk(env.bid, "vleg0002")  # legacy, untouched
    job = _rebuild(env.bid)
    assert job.status == "done", job.log
    assert _video(env.bid, "vnew0001").moderation_status == "rejected"
    assert _video(env.bid, "vnew0001").status == "rejected"
    assert env.integrated == ["vleg0002"]
    walk = json.loads((store.video_dir(env.bid, "vnew0001") / "walk.json").read_text())
    assert walk["moderation"]["approved"] is False


def test_rebuild_blur_failure_fails_closed_and_keeps_map(env, monkeypatch):
    def broken(frame):
        raise moderation.FaceBlurError("face model missing")

    monkeypatch.setattr(moderation, "blur_faces", broken)
    _done_walk(env.bid, "vbrk0001", APPROVED_RECORD, ["vbrk0001_0000.jpg", "vbrk0001_0001.jpg"])
    job = _rebuild(env.bid)
    assert job.status == "error" and "FaceBlurError" in job.error
    assert _frame_files(env.bid, "vbrk0001") == []  # no unblurred frame left behind
    assert env.integrated == []


# ---------- after a rejection: the uploader can read why, and upload again ----------
def _upload_walk(client, bid, headers) -> dict:
    """The real upload route: queues the walk under the signed-in account and returns its job and video ids."""
    r = client.post(f"/api/buildings/{bid}/videos", files={"video": ("walk.mp4", b"x", "video/mp4")}, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


def test_uploader_can_read_why_a_walk_was_rejected_and_upload_again(env, monkeypatch):
    monkeypatch.setattr(moderation, "moderate_keyframe", Moderator([APPROVED, REJECTED, APPROVED, APPROVED]))
    client = TestClient(app)
    headers = signed_in_headers("walker@example.com")

    first = _upload_walk(client, env.bid, headers)
    job = jobs.get(first["job_id"])
    jobs._run(job)
    assert job.status == "done", job.log

    # GET /map lists every walk with its moderation status and notes; it needs no sign-in.
    r = client.get(f"/api/buildings/{env.bid}/map")
    assert r.status_code == 200
    v = next(v for v in r.json()["videos"] if v["id"] == first["video_id"])
    assert (v["status"], v["moderation_status"]) == ("rejected", "rejected")
    assert "inappropriate content: nudity" in v["moderation_notes"]

    # A fresh upload of the same building is a new, independent walk.
    second = _upload_walk(client, env.bid, headers)
    assert second["video_id"] != first["video_id"]
    job2 = jobs.get(second["job_id"])
    jobs._run(job2)
    assert job2.status == "done", job2.log
    videos = {v["id"]: v for v in client.get(f"/api/buildings/{env.bid}/map").json()["videos"]}
    assert videos[second["video_id"]]["moderation_status"] == "approved"
    assert videos[first["video_id"]]["moderation_status"] == "rejected"  # the rejected walk stays listed
    assert env.integrated == [second["video_id"]]
