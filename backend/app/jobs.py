"""Background processing: normalize -> motion -> keyframes -> Gemma -> mapping.

Jobs run one at a time on a single worker thread (Gemma saturates the GPU anyway), from a queue that
can be reordered: queued jobs can be paused, resumed or moved to the front, and a running job pauses
cooperatively at the next checkpoint (between stages / Gemma frames). A paused job restarts from the
beginning when resumed, which is cheap because every Gemma answer is cached; the map is only written
in the final stage, so pausing never leaves it half-updated.

Queued/paused videos are persisted in map.json (VideoInfo.status), so `recover()` re-queues them after
a server restart.
"""
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import config, gemma, mapper, motion, store, video
from .keyframes import select_keyframes
from .schemas import Keyframe, VideoInfo

STAGES = [("normalize", 0, 8), ("motion", 8, 22), ("keyframes", 22, 32), ("gemma", 32, 90), ("mapping", 90, 100)]


class JobPaused(Exception):
    pass


@dataclass
class Job:
    id: str
    building_id: str
    video_id: str | None
    kind: str
    filename: str = ""
    status: str = "queued"  # queued | running | paused | done | error
    stage: str = "queued"
    pct: float = 0.0
    log: list[str] = field(default_factory=list)
    result: dict = field(default_factory=dict)
    error: str | None = None
    created: float = field(default_factory=time.time)
    order: float = field(default_factory=time.time)  # queue position; smaller runs first
    pause_requested: bool = False
    fn: object = field(default=None, repr=False)

    def say(self, msg: str) -> None:
        self.log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def enter(self, stage: str, frac: float = 0.0) -> None:
        self.checkpoint()
        self.stage = stage
        lo, hi = next((a, b) for s, a, b in STAGES if s == stage)
        self.pct = round(lo + (hi - lo) * frac, 1)

    def checkpoint(self) -> None:
        if self.pause_requested:
            raise JobPaused()

    def public(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k not in ("fn", "pause_requested")}


JOBS: dict[str, Job] = {}
_cv = threading.Condition()
_worker: threading.Thread | None = None


def get(job_id: str) -> Job | None:
    return JOBS.get(job_id)


def queue(building_id: str | None = None) -> list[Job]:
    """Running job first, then queued/paused in run order, then recent finished jobs."""
    jobs = [j for j in JOBS.values() if building_id in (None, j.building_id)]
    rank = {"running": 0, "queued": 1, "paused": 1, "error": 2, "done": 3}
    return sorted(jobs, key=lambda j: (rank.get(j.status, 9), j.order if rank.get(j.status) == 1 else -j.created))


def _set_video_status(building_id: str, video_id: str, **kw) -> None:
    with store.lock_for(building_id):
        bundle = store.load_map(building_id)
        v = next((v for v in bundle.videos if v.id == video_id), None)
        if v:
            for k, val in kw.items():
                setattr(v, k, val)
            store.save_map(bundle)


def _enqueue(job: Job, fn, front: bool = False) -> Job:
    job.fn = fn
    if front:
        job.order = min([j.order for j in JOBS.values() if j.status in ("queued", "paused")] + [time.time()]) - 1
    with _cv:
        JOBS[job.id] = job
        _cv.notify_all()
    _ensure_worker()
    return job


def _ensure_worker() -> None:
    global _worker
    with _cv:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_worker_loop, name="pipeline", daemon=True)
            _worker.start()


def _worker_loop() -> None:
    while True:
        with _cv:
            while True:
                ready = sorted((j for j in JOBS.values() if j.status == "queued"), key=lambda j: j.order)
                if ready:
                    job = ready[0]
                    job.status = "running"
                    break
                _cv.wait()
        _run(job)


def submit_video(building_id: str, upload_path: Path, filename: str, sensors: dict | None, model: str | None,
                 front: bool = False, start_floor: int | None = None) -> Job:
    video_id = f"v{time.strftime('%H%M%S')}{uuid.uuid4().hex[:4]}"
    vdir = store.video_dir(building_id, video_id)
    vdir.mkdir(parents=True, exist_ok=True)
    src = vdir / f"source{Path(filename).suffix.lower() or '.mp4'}"
    shutil.move(str(upload_path), src)
    if sensors:
        store.write_json_atomic(vdir / "sensors.json", sensors)
    with store.lock_for(building_id):
        bundle = store.load_map(building_id)
        bundle.videos.append(VideoInfo(id=video_id, filename=filename, has_sensors=bool(sensors), start_floor=start_floor))
        store.save_map(bundle)
    job = Job(id=uuid.uuid4().hex[:10], building_id=building_id, video_id=video_id, kind="video", filename=filename)
    return _enqueue(job, lambda: _process_video(job, src, sensors, model), front)


def submit_rebuild(building_id: str) -> Job:
    job = Job(id=uuid.uuid4().hex[:10], building_id=building_id, video_id=None, kind="rebuild", filename="(rebuild)")
    return _enqueue(job, lambda: _rebuild(job))


def pause(job_id: str) -> Job:
    job = JOBS[job_id]
    with _cv:
        if job.status == "queued":
            job.status = "paused"
            job.say("Paused (waiting)")
        elif job.status == "running":
            job.pause_requested = True
            job.say("Pause requested: stopping at the next checkpoint")
    if job.video_id and job.status == "paused":
        _set_video_status(job.building_id, job.video_id, status="paused")
    return job


def resume(job_id: str, front: bool = False) -> Job:
    job = JOBS[job_id]
    with _cv:
        if job.status in ("paused", "error"):
            job.status, job.stage, job.pct, job.error = "queued", "queued", 0.0, None
            job.say("Resumed")
        if front and job.status == "queued":
            job.order = min([j.order for j in JOBS.values() if j.status in ("queued", "paused")]) - 1
        _cv.notify_all()
    if job.video_id:
        _set_video_status(job.building_id, job.video_id, status="queued")
    return job


def run_next(job_id: str) -> Job:
    """Move a job to the front; if something else is running, pause it so this one starts now."""
    job = resume(job_id, front=True)
    running = next((j for j in JOBS.values() if j.status == "running" and j.id != job_id), None)
    if running:
        pause(running.id)
        running.order = job.order + 0.5  # it resumes right after the prioritized job
        running.say("Will resume automatically after the prioritized job")
        running.auto_resume = True
    return job


def _run(job: Job) -> None:
    job.pause_requested = False
    try:
        job.fn()
        job.status, job.stage, job.pct = "done", "done", 100.0
        job.say("Done")
    except JobPaused:
        job.pause_requested = False
        auto = getattr(job, "auto_resume", False)
        job.auto_resume = False
        job.status, job.stage, job.pct = ("queued" if auto else "paused"), "queued", 0.0
        job.say("Re-queued" if auto else "Paused")
        if job.video_id:
            _set_video_status(job.building_id, job.video_id, status="queued" if auto else "paused")
    except Exception as exc:
        job.status, job.error = "error", f"{type(exc).__name__}: {exc}"
        job.say(f"ERROR {job.error}")
        traceback.print_exc()
        if job.video_id:
            _set_video_status(job.building_id, job.video_id, status="error")


def recover() -> None:
    """Re-queue videos that were queued/processing/paused when the server stopped."""
    for b in store.list_buildings():
        bundle = store.load_map(b["id"])
        for v in bundle.videos:
            if v.status not in ("queued", "processing", "paused"):
                continue
            vdir = store.video_dir(bundle.building_id, v.id)
            src = next(iter(sorted(vdir.glob("source.*"))), None)
            if src is None:
                continue
            sensors = store.read_json(vdir / "sensors.json")
            job = Job(id=uuid.uuid4().hex[:10], building_id=bundle.building_id, video_id=v.id, kind="video",
                      filename=v.filename, status="paused" if v.status == "paused" else "queued")
            job.say("Recovered after server restart")
            _enqueue(job, (lambda j, s, sn: lambda: _process_video(j, s, sn, None))(job, src, sensors))


def _analyze(job: Job, keyframes: list[Keyframe], model: str | None) -> bool:
    bdir = store.building_dir(job.building_id)
    cache = gemma.cache_for(bdir)
    frames = store.frames_dir(job.building_id)
    try:
        chosen = model or gemma.resolve_model()
        job.say(f"Gemma model: {chosen} (local, via Ollama)")
    except gemma.GemmaUnavailable as exc:
        job.say(f"WARNING Gemma unavailable, mapping from motion only: {exc}")
        return False
    def one(kf: Keyframe):
        if job.pause_requested:
            return kf, JobPaused(), 0.0
        t0 = time.time()
        try:
            return kf, gemma.analyze_frame(frames / kf.frame_id, cache, model), time.time() - t0
        except gemma.GemmaUnavailable as exc:
            return kf, exc, 0.0

    # A few requests in flight keeps the GPU busy (~25% faster with OLLAMA_NUM_PARALLEL>1).
    with ThreadPoolExecutor(max_workers=config.GEMMA_CONCURRENCY) as pool:
        for i, (kf, res, dt) in enumerate(pool.map(one, keyframes)):
            job.enter("gemma", (i + 1) / max(len(keyframes), 1))
            if isinstance(res, JobPaused):
                continue
            if isinstance(res, Exception):
                job.say(f"WARNING Gemma failed on {kf.frame_id}: {res}")
                continue
            kf.analysis = a = res
            if a.has_sign or a.place_type != "hallway" or a.stairs != "none":
                job.say(f"{kf.frame_id} [{kf.kind}] -> {a.place_type}: {a.suggested_label!r} "
                        f"texts={a.texts} stairs={a.stairs} ({dt:.1f}s)")
    job.checkpoint()
    return True


def _process_video(job: Job, src: Path, sensors: dict | None, model: str | None) -> None:
    bid, vid = job.building_id, job.video_id
    vdir = store.video_dir(bid, vid)
    _set_video_status(bid, vid, status="processing")

    job.enter("normalize"); job.say(f"Normalizing {src.name} with ffmpeg")
    analysis_path, hq_path, duration = video.normalize(src, vdir)
    job.say(f"Duration {duration:.1f}s")

    job.enter("motion"); job.say("Estimating camera motion (optical flow" + (" + phone gyro)" if sensors else ")"))
    track = motion.compute_track(analysis_path, sensors, log=job.say)
    events = motion.detect_events(track)
    store.write_json_atomic(vdir / "track.json", {**track, "events": events})
    summary = ", ".join(f"{e['type']}@{e['start']:.1f}s" + (f" {e['delta']:+.0f}deg" if e["type"] == "turn" else "")
                        for e in events if e["type"] != "dwell")
    job.say(f"Motion events: {summary or 'none'}")

    job.enter("keyframes")
    keyframes = select_keyframes(hq_path, track, events, duration, vid, store.frames_dir(bid), log=job.say)

    use_gemma = _analyze(job, keyframes, model)
    store.write_json_atomic(vdir / "walk.json", {
        "video_id": vid, "duration": duration, "keyframes": [k.model_dump() for k in keyframes],
        "events": events, "gemma": use_gemma,
    })

    job.enter("mapping"); job.say("Building graph and closing loops")
    with store.lock_for(bid):
        bundle = store.load_map(bid)
        info = next(v for v in bundle.videos if v.id == vid)
        result = mapper.integrate_walk(bundle, vid, keyframes, track, store.frames_dir(bid),
                                       gemma.cache_for(store.building_dir(bid)), use_gemma, log=job.say,
                                       start_floor=info.start_floor)
        if use_gemma:
            bundle.processed_with = {"model": model or gemma.resolve_model(), "prompt": gemma.PROMPT_VERSION}
        v = next(v for v in bundle.videos if v.id == vid)
        v.duration, v.keyframes, v.status = round(duration, 1), len(keyframes), "done"
        v.floor_offset = result.get("floor_offset") or 0
        job.say("Linking with the rest of the building map")
        result["consolidate"] = mapper.consolidate(bundle, store.frames_dir(bid), gemma.cache_for(store.building_dir(bid)),
                                                   use_gemma, log=job.say)
        store.save_map(bundle)
    job.result = result
    job.say(f"Map updated: {result}")


def _rebuild(job: Job) -> None:
    """Re-run motion/keyframes/mapping over every processed video (Gemma answers come from cache)."""
    bid = job.building_id
    with store.lock_for(bid):
        bundle = store.load_map(bid)
        bundle.nodes, bundle.edges, bundle.suggested_merges = [], [], []
        store.save_map(bundle)
    # only finished walks: queued/paused ones will be integrated when their own job runs
    videos = [v for v in bundle.videos if v.status == "done" and (store.video_dir(bid, v.id) / "analysis.mp4").exists()]
    for i, v in enumerate(videos):
        job.say(f"Reprocessing {v.filename} ({i + 1}/{len(videos)})")
        vdir = store.video_dir(bid, v.id)
        sensors = store.read_json(vdir / "sensors.json")
        job.video_id = v.id
        job.enter("motion")
        track = motion.compute_track(vdir / "analysis.mp4", sensors, log=job.say)
        events = motion.detect_events(track)
        store.write_json_atomic(vdir / "track.json", {**track, "events": events})
        job.enter("keyframes")
        keyframes = select_keyframes(vdir / "hq.mp4", track, events, v.duration or track["t"][-1], v.id,
                                     store.frames_dir(bid), log=job.say)
        use_gemma = _analyze(job, keyframes, None)
        store.write_json_atomic(vdir / "walk.json", {
            "video_id": v.id, "duration": v.duration, "keyframes": [k.model_dump() for k in keyframes],
            "events": events, "gemma": use_gemma,
        })
        job.enter("mapping")
        with store.lock_for(bid):
            bundle = store.load_map(bid)
            res = mapper.integrate_walk(bundle, v.id, keyframes, track, store.frames_dir(bid),
                                        gemma.cache_for(store.building_dir(bid)), use_gemma, log=job.say,
                                        start_floor=v.start_floor)
            info = next(x for x in bundle.videos if x.id == v.id)
            info.floor_offset = res.get("floor_offset") or 0
            store.save_map(bundle)
        job.say(f"{v.filename}: {res}")
    job.video_id = None
    _consolidate(job)


def submit_consolidate(building_id: str) -> Job:
    job = Job(id=uuid.uuid4().hex[:10], building_id=building_id, video_id=None, kind="consolidate",
              filename="(link videos)")
    return _enqueue(job, lambda: _consolidate(job), front=True)


def _consolidate(job: Job) -> None:
    bid = job.building_id
    job.enter("mapping")
    job.say("Linking all walks into one building map")
    with store.lock_for(bid):
        bundle = store.load_map(bid)
        job.result = mapper.consolidate(bundle, store.frames_dir(bid), gemma.cache_for(store.building_dir(bid)),
                                        _gemma_ok(), log=job.say)
        store.save_map(bundle)


def _gemma_ok() -> bool:
    try:
        gemma.resolve_model()
        return True
    except gemma.GemmaUnavailable:
        return False
