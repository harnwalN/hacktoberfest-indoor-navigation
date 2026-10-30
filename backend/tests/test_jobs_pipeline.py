"""Real (synthetic video) pipeline: rebuild output equals the first processing; recover() re-queues."""
from app import config, gemma, jobs, store
from app.schemas import VideoInfo
from tests.synthetic import write_video


def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BUILDINGS_DIR", tmp_path / "buildings")
    monkeypatch.setattr(jobs, "JOBS", {})
    monkeypatch.setattr(jobs, "_ensure_worker", lambda: None)

    def no_gemma(*_a, **_k):
        raise gemma.GemmaUnavailable("off")

    monkeypatch.setattr(gemma, "resolve_model", no_gemma)
    return store.create_building("Synth").building_id


def _add(bid, tmp_path, vid, status="queued"):
    walk = [(3, 0)]
    for _ in range(4):
        walk += [(3, 30), (4, 0)]
    src = tmp_path / f"{vid}.mp4"
    write_video(src, walk)
    vdir = store.video_dir(bid, vid)
    vdir.mkdir(parents=True)
    (vdir / "source.mp4").write_bytes(src.read_bytes())
    b = store.load_map(bid)
    b.videos.append(VideoInfo(id=vid, filename=f"{vid}.mp4", status=status))
    store.save_map(b)
    return vdir


def test_rebuild_matches_initial_and_keeps_frames(tmp_path, monkeypatch):
    bid = _setup(tmp_path, monkeypatch)
    vdir = _add(bid, tmp_path, "vone")
    job = jobs.Job(id="p", building_id=bid, video_id="vone", kind="video")
    jobs._process_video(job, vdir / "source.mp4", None, None)
    m1 = store.load_map(bid)
    frames1 = {f.name for f in store.frames_dir(bid).glob("*.jpg")}

    rb = jobs.Job(id="r", building_id=bid, video_id=None, kind="rebuild")
    rb.fn = lambda: jobs._rebuild(rb)
    rb.status = "running"
    jobs._run(rb)
    assert rb.status == "done", rb.log
    m2 = store.load_map(bid)
    assert (len(m2.nodes), len(m2.edges)) == (len(m1.nodes), len(m1.edges))
    assert [v.status for v in m2.videos] == ["done"]
    assert frames1 <= {f.name for f in store.frames_dir(bid).glob("*.jpg")}


def test_recover_requeues_unfinished(tmp_path, monkeypatch):
    bid = _setup(tmp_path, monkeypatch)
    _add(bid, tmp_path, "vq", "queued")
    _add(bid, tmp_path, "vp", "paused")
    _add(bid, tmp_path, "vd", "done")
    jobs.recover()
    got = {j.video_id: j.status for j in jobs.JOBS.values()}
    assert got == {"vq": "queued", "vp": "paused"}
