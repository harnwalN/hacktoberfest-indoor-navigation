"""Job queue edge cases: pausing/resuming a rebuild, deleting a walk mid-run, concurrent queue reads.

ffmpeg, optical flow, keyframe extraction and Gemma are faked; jobs run synchronously via jobs._run.
"""
import threading

import pytest

from app import config, gemma, jobs, mapper, motion, store, video
from app.schemas import Keyframe, Node, VideoInfo


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BUILDINGS_DIR", tmp_path / "buildings")
    monkeypatch.setattr(jobs, "JOBS", {})
    monkeypatch.setattr(jobs, "_ensure_worker", lambda: None)

    def no_gemma(*_a, **_k):
        raise gemma.GemmaUnavailable("disabled in tests")

    def fake_keyframes(_hq, _track, _events, _duration, vid, frames_out, log=print):
        (frames_out / f"{vid}_0.jpg").write_bytes(b"jpg")
        return [Keyframe(frame_id=f"{vid}_0.jpg", video_id=vid, t=0, kind="start", heading=0)]

    def fake_integrate(bundle, vid, *_a, **_k):
        bundle.nodes.append(Node(id=f"{vid}_n", label=vid, video_id=vid))
        return {"floor_offset": 0}

    monkeypatch.setattr(gemma, "resolve_model", no_gemma)
    monkeypatch.setattr(motion, "compute_track", lambda *_a, **_k: {"t": [0.0, 1.0]})
    monkeypatch.setattr(motion, "detect_events", lambda _track: [])
    monkeypatch.setattr(jobs, "select_keyframes", fake_keyframes)
    monkeypatch.setattr(mapper, "integrate_walk", fake_integrate)
    monkeypatch.setattr(mapper, "consolidate", lambda *_a, **_k: {})
    monkeypatch.setattr(video, "normalize",
                        lambda src, vdir: (vdir / "analysis.mp4", vdir / "hq.mp4", 1.0))
    return store.create_building("Test Hall").building_id


def _add_video(bid: str, vid: str, status: str = "done") -> None:
    vdir = store.video_dir(bid, vid)
    vdir.mkdir(parents=True)
    for name in ("source.mp4", "analysis.mp4", "hq.mp4"):
        (vdir / name).write_bytes(b"x")
    with store.lock_for(bid):
        bundle = store.load_map(bid)
        bundle.videos.append(VideoInfo(id=vid, filename=f"{vid}.mp4", status=status, duration=1.0))
        if status == "done":
            bundle.nodes.append(Node(id=f"{vid}_n", label=vid, video_id=vid))
        store.save_map(bundle)


def _statuses(bid: str) -> dict:
    return {v.id: v.status for v in store.load_map(bid).videos}


def _node_videos(bid: str) -> set:
    return {n.video_id for n in store.load_map(bid).nodes}


def _rebuild_job(bid: str) -> jobs.Job:
    job = jobs.Job(id="rb", building_id=bid, video_id=None, kind="rebuild")
    job.fn = lambda: jobs._rebuild(job)
    jobs.JOBS[job.id] = job
    return job


def _video_job(bid: str, vid: str) -> jobs.Job:
    job = jobs.Job(id=f"j{vid}", building_id=bid, video_id=vid, kind="video")
    src = store.video_dir(bid, vid) / "source.mp4"
    job.fn = lambda: jobs._process_video(job, src, None, None)
    jobs.JOBS[job.id] = job
    return job


def test_pause_resume_rebuild_keeps_videos_done(env, monkeypatch):
    bid = env
    for vid in ("va", "vb", "vc"):
        _add_video(bid, vid)
    job = _rebuild_job(bid)

    real_track = motion.compute_track
    calls = []

    def pausing_track(path, *a, **k):
        calls.append(path.parent.name)
        if path.parent.name == "vb" and calls.count("vb") == 1:
            jobs.pause(job.id)  # the user hits pause while vb is being re-analyzed
        return real_track(path, *a, **k)

    monkeypatch.setattr(motion, "compute_track", pausing_track)
    job.status = "running"
    jobs._run(job)
    assert job.status == "paused", job.log
    assert job.video_id is None
    assert _statuses(bid) == {"va": "done", "vb": "done", "vc": "done"}
    assert _node_videos(bid) == {"va", "vb", "vc"}  # map untouched while paused

    jobs.resume(job.id)
    assert job.status == "queued"
    assert _statuses(bid) == {"va": "done", "vb": "done", "vc": "done"}

    job.status = "running"
    jobs._run(job)
    assert job.status == "done", job.log
    assert _statuses(bid) == {"va": "done", "vb": "done", "vc": "done"}
    assert _node_videos(bid) == {"va", "vb", "vc"}


def test_resume_done_job_leaves_video_status(env):
    bid = env
    _add_video(bid, "va")
    job = _video_job(bid, "va")
    job.status = "done"
    jobs.resume(job.id)
    assert job.status == "done"
    assert _statuses(bid) == {"va": "done"}

    rb = _rebuild_job(bid)
    rb.status = "done"
    jobs.resume(rb.id)
    assert rb.status == "done"
    assert _statuses(bid) == {"va": "done"}


def test_rebuild_error_does_not_touch_video_status(env, monkeypatch):
    bid = env
    _add_video(bid, "va")
    job = _rebuild_job(bid)

    def boom(*_a, **_k):
        raise RuntimeError("flow failed")

    monkeypatch.setattr(motion, "compute_track", boom)
    job.status = "running"
    jobs._run(job)
    assert job.status == "error"
    assert _statuses(bid) == {"va": "done"}
    assert _node_videos(bid) == {"va"}


def test_delete_video_mid_run_cancels_cleanly(env, monkeypatch):
    bid = env
    _add_video(bid, "vx", status="queued")
    job = _video_job(bid, "vx")
    real_track = motion.compute_track

    def deleting_track(*a, **k):
        jobs.delete_video(bid, "vx")  # API thread deletes the walk during optical flow
        return real_track(*a, **k)

    monkeypatch.setattr(motion, "compute_track", deleting_track)
    job.status = "running"
    jobs._run(job)
    assert job.status == "cancelled", job.log
    assert job.error is None
    assert not store.video_dir(bid, "vx").exists()
    assert not list(store.frames_dir(bid).glob("vx_*"))
    assert "vx" not in _statuses(bid)


def test_video_missing_from_map_at_mapping_cancels(env, monkeypatch):
    bid = env
    _add_video(bid, "vx", status="queued")
    job = _video_job(bid, "vx")
    real_analyze = jobs._analyze_walk

    def dropping_analyze(j, keyframes, model):
        with store.lock_for(bid):  # walk vanishes from the map before the mapping stage
            bundle = store.load_map(bid)
            bundle.videos = [v for v in bundle.videos if v.id != "vx"]
            store.save_map(bundle)
        return real_analyze(j, keyframes, model)

    monkeypatch.setattr(jobs, "_analyze_walk", dropping_analyze)
    job.status = "running"
    jobs._run(job)  # used to raise StopIteration out of next(...) -> "error"
    assert job.status == "cancelled", job.log
    assert not store.video_dir(bid, "vx").exists()
    assert not list(store.frames_dir(bid).glob("vx_*"))


def test_delete_video_during_rebuild_skips_it(env, monkeypatch):
    bid = env
    for vid in ("va", "vb"):
        _add_video(bid, vid)
    job = _rebuild_job(bid)
    real_track = motion.compute_track

    def deleting_track(path, *a, **k):
        if path.parent.name == "vb":
            jobs.delete_video(bid, "vb")
        return real_track(path, *a, **k)

    monkeypatch.setattr(motion, "compute_track", deleting_track)
    job.status = "running"
    jobs._run(job)
    assert job.status == "done", job.log
    assert _statuses(bid) == {"va": "done"}
    assert _node_videos(bid) == {"va"}
    assert not store.video_dir(bid, "vb").exists()
    assert not list(store.frames_dir(bid).glob("vb_*"))


def test_queue_while_jobs_are_added(env):
    stop = threading.Event()
    errors = []

    def add():
        for i in range(3000):
            jobs._enqueue(jobs.Job(id=f"j{i}", building_id=env, video_id=None, kind="consolidate"),
                          lambda: None, front=i % 2 == 0)
        stop.set()

    t = threading.Thread(target=add)
    t.start()
    try:
        while not stop.is_set():
            jobs.queue(env)
            jobs.queue()
    except RuntimeError as exc:  # dictionary changed size during iteration
        errors.append(exc)
    t.join()
    assert not errors
    assert len(jobs.queue(env)) == 3000
