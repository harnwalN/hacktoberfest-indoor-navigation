"""Background processing: normalize -> motion -> keyframes -> Gemma -> mapping.

Jobs run one at a time on a single worker thread (Gemma saturates the GPU anyway), from a queue that
can be reordered: queued jobs can be paused, resumed or moved to the front, and a running job pauses
cooperatively at the next checkpoint (between stages / Gemma frames). A paused job restarts from the
beginning when resumed, which is cheap because every Gemma answer is cached; the map is only written
in the final stage (a rebuild re-analyzes every walk first, then replaces the map in one step), so
pausing never leaves it half-updated.

Queued/paused videos are persisted in map.json (VideoInfo.status), so `recover()` re-queues them after
a server restart.
"""
import shutil
import threading
import time
import traceback
import uuid
from collections import Counter
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
    cancelled: bool = False
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
        return {k: v for k, v in self.__dict__.items() if k not in ("fn", "pause_requested", "cancelled")}


JOBS: dict[str, Job] = {}
_cv = threading.Condition()
_worker: threading.Thread | None = None


def get(job_id: str) -> Job | None:
    return JOBS.get(job_id)


def queue(building_id: str | None = None) -> list[Job]:
    """Running job first, then queued/paused in run order, then recent finished jobs."""
    with _cv:
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
    with _cv:
        if front:
            job.order = min([j.order for j in JOBS.values() if j.status in ("queued", "paused")] + [time.time()]) - 1
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
    if job.kind == "video" and job.status == "paused":
        _set_video_status(job.building_id, job.video_id, status="paused")
    return job


def resume(job_id: str, front: bool = False) -> Job:
    job = JOBS[job_id]
    with _cv:
        resumed = job.status in ("paused", "error")
        if resumed:
            job.status, job.stage, job.pct, job.error = "queued", "queued", 0.0, None
            job.say("Resumed")
        if front and job.status == "queued":
            job.order = min([j.order for j in JOBS.values() if j.status in ("queued", "paused")]) - 1
        _cv.notify_all()
    if resumed and job.kind == "video":
        _set_video_status(job.building_id, job.video_id, status="queued")
    return job


def run_next(job_id: str) -> Job:
    """Move a job to the front; if something else is running, pause it so this one starts now."""
    job = resume(job_id, front=True)
    with _cv:
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
        if job.cancelled:
            return _cancel(job)
        auto = getattr(job, "auto_resume", False)
        job.auto_resume = False
        job.status, job.stage, job.pct = ("queued" if auto else "paused"), "queued", 0.0
        job.say("Re-queued" if auto else "Paused")
        if job.kind == "video":
            _set_video_status(job.building_id, job.video_id, status="queued" if auto else "paused")
    except Exception as exc:
        if job.cancelled:  # e.g. its files were deleted under it: that's the delete, not a failure
            return _cancel(job)
        job.status, job.error = "error", f"{type(exc).__name__}: {exc}"
        job.say(f"ERROR {job.error}")
        traceback.print_exc()
        if job.kind == "video":
            _set_video_status(job.building_id, job.video_id, status="error")


def _cancel(job: Job) -> None:
    job.pause_requested = False
    job.status, job.stage = "cancelled", "cancelled"
    job.say("Cancelled: video removed")
    _purge_files(job.building_id, job.video_id)  # anything written after the delete


def _video_info(job: Job, bundle, video_id: str) -> VideoInfo:
    """The walk's entry in the map; a walk deleted mid-run cancels a video job."""
    v = next((v for v in bundle.videos if v.id == video_id), None)
    if v is None:
        job.cancelled = True
        raise JobPaused()
    return v


def delete_video(building_id: str, video_id: str) -> dict:
    """Stop the video's job (if any) and remove the walk and all its data from the building."""
    with _cv:
        for j in list(JOBS.values()):
            if j.video_id == video_id and j.building_id == building_id:
                j.cancelled = True
                if j.status in ("queued", "paused"):
                    j.status, j.stage = "cancelled", "cancelled"
                    j.say("Cancelled: video removed")
                elif j.status == "running":
                    j.pause_requested = True
    with store.lock_for(building_id):
        bundle = store.load_map(building_id)
        if not any(v.id == video_id for v in bundle.videos):
            raise KeyError(video_id)
        res = mapper.remove_video(bundle, video_id)
        store.save_map(bundle)
    referenced = res.pop("referenced_frames")
    res["deleted_files"] = _purge_files(building_id, video_id, keep=referenced)
    return res


def _purge_files(building_id: str, video_id: str | None, keep: set[str] | None = None) -> int:
    if not video_id:
        return 0
    keep = keep or set()
    n = 0
    for f in store.frames_dir(building_id).glob(f"{video_id}_*.jpg"):
        if f.name not in keep:
            f.unlink(missing_ok=True)
            n += 1
    shutil.rmtree(store.video_dir(building_id, video_id), ignore_errors=True)
    return n


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


@dataclass
class GemmaRun:
    ok: bool  # Gemma answers exist for this walk (walk.json "gemma")
    model: str | None = None  # model the answers came from (walk.json "model", bundle.processed_with)
    online: bool = False  # Ollama answered; False = answers came from the cache only


def _analyze(job: Job, keyframes: list[Keyframe], model: str | None) -> bool:
    return _analyze_walk(job, keyframes, model).ok


def _analyze_walk(job: Job, keyframes: list[Keyframe], model: str | None) -> GemmaRun:
    bdir = store.building_dir(job.building_id)
    cache = gemma.cache_for(bdir)
    frames = store.frames_dir(job.building_id)
    online = True
    try:
        chosen = model or gemma.resolve_model()
        gemma.ensure_model(chosen)  # a missing model fails here once, not on every frame
        job.say(f"Gemma model: {chosen} (local, via Ollama)")
    except gemma.GemmaUnavailable as exc:
        online, chosen = False, model
        job.say(f"WARNING Gemma unavailable, using cached answers only: {exc}")
    # Circuit breaker: after GEMMA_BREAKER_FAILS GemmaUnavailable in a row (Ollama down, model unloaded,
    # broken output), stop calling Gemma for the rest of this walk instead of timing out on every frame.
    # Frames whose answer is cached still get it (cache only, Ollama is never contacted for them).
    breaker = {"streak": 0, "open": False}
    breaker_lock = threading.Lock()
    skipped = object()

    def one(kf: Keyframe):
        if job.pause_requested:
            return kf, JobPaused(), 0.0
        path = frames / kf.frame_id
        if not online or breaker["open"]:
            res = gemma.cached_analysis(path, cache, chosen)
            return kf, (skipped if res is None else res), 0.0
        t0 = time.time()
        try:
            res = gemma.analyze_frame(path, cache, chosen)
        except gemma.GemmaUnavailable as exc:
            with breaker_lock:
                breaker["streak"] += 1
                breaker["open"] = breaker["open"] or breaker["streak"] >= config.GEMMA_BREAKER_FAILS
            return kf, exc, 0.0
        except Exception as exc:  # e.g. an unreadable frame: skip just this one
            return kf, exc, 0.0
        with breaker_lock:
            breaker["streak"] = 0
        return kf, res, time.time() - t0

    ok = n_skipped = 0
    used: Counter = Counter()
    announced = False
    # A few requests in flight keeps the GPU busy (~25% faster with OLLAMA_NUM_PARALLEL>1).
    with ThreadPoolExecutor(max_workers=config.GEMMA_CONCURRENCY) as pool:
        for i, (kf, res, dt) in enumerate(pool.map(one, keyframes)):
            job.enter("gemma", (i + 1) / max(len(keyframes), 1))
            if isinstance(res, JobPaused):
                continue
            if res is skipped:
                n_skipped += 1
                continue
            if isinstance(res, gemma.GemmaUnavailable):
                job.say(f"WARNING Gemma failed on {kf.frame_id}: {res}")
                if breaker["open"] and not announced:
                    announced = True
                    job.say(f"WARNING Gemma failed {config.GEMMA_BREAKER_FAILS} frames in a row; "
                            "not calling it again for this walk (cached answers are still used)")
                continue
            if isinstance(res, Exception):
                job.say(f"WARNING Skipping {kf.frame_id}: {type(res).__name__}: {res}")
                continue
            kf.analysis = a = res
            ok += 1
            if (m := getattr(res, "_model_used", None)):
                used[m] += 1
            if a.has_sign or a.place_type != "hallway" or a.stairs != "none":
                job.say(f"{kf.frame_id} [{kf.kind}] -> {a.place_type}: {a.suggested_label!r} "
                        f"texts={a.texts} stairs={a.stairs} ({dt:.1f}s)")
    job.checkpoint()
    if n_skipped:
        job.say(f"Skipped Gemma on {n_skipped} frame(s): not cached, and Gemma unavailable or failing")
    online = online and not breaker["open"]
    if keyframes and not ok:
        job.say("WARNING Gemma analysed no frames, mapping from motion only")
        return GemmaRun(False, None, online)
    model_used = used.most_common(1)[0][0] if used else chosen
    job.say(f"Gemma analysed {ok}/{len(keyframes)} frames" + (f" with {model_used}" if model_used else ""))
    return GemmaRun(True, model_used, online)


def _frames_ok(keyframes: list[Keyframe]) -> int:
    """Keyframes Gemma analysed successfully (recorded in walk.json as gemma_frames_ok)."""
    return sum(k.analysis is not None for k in keyframes)


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
    store.write_json_atomic(vdir / "track.json", {**track, "events": events}, mkdir=False)
    summary =", ".join(f"{e['type']}@{e['start']:.1f}s" + (f" {e['delta']:+.0f}deg" if e["type"] == "turn" else "")
                        for e in events if e["type"] != "dwell")
    job.say(f"Motion events: {summary or 'none'}")

    job.enter("keyframes")
    keyframes = select_keyframes(hq_path, track, events, duration, vid, store.frames_dir(bid), log=job.say)

    run = _analyze_walk(job, keyframes, model)
    use_gemma = run.ok
    store.write_json_atomic(vdir / "walk.json", {
        "video_id": vid, "duration": duration, "keyframes": [k.model_dump() for k in keyframes],
        "events": events, "gemma": use_gemma, "gemma_frames_ok": _frames_ok(keyframes), "model": run.model,
    }, mkdir=False)

    job.enter("mapping"); job.say("Building graph and closing loops")
    # same-place / same-stairwell checks use this walk's model; if Ollama was down they answer from the cache
    with store.lock_for(bid), gemma.session(run.model or model, offline=not run.online):
        bundle = store.load_map(bid)
        info = _video_info(job, bundle, vid)
        result = mapper.integrate_walk(bundle, vid, keyframes, track, store.frames_dir(bid),
                                       gemma.cache_for(store.building_dir(bid)), use_gemma, log=job.say,
                                       start_floor=info.start_floor)
        if use_gemma and run.model:
            bundle.processed_with = {"model": run.model, "prompt": gemma.PROMPT_VERSION}
        v = _video_info(job, bundle, vid)
        v.duration, v.keyframes, v.status = round(duration, 1), len(keyframes), "done"
        v.floor_offset = result.get("floor_offset") or 0
        job.say("Linking with the rest of the building map")
        result["consolidate"] = mapper.consolidate(bundle, store.frames_dir(bid), gemma.cache_for(store.building_dir(bid)),
                                                   use_gemma, log=job.say)
        store.save_map(bundle)
    job.result = result
    job.say(f"Map updated: {result}")


def _rebuild(job: Job) -> None:
    """Re-run motion/keyframes over every processed video (Gemma answers come from cache), then rebuild the
    map from scratch in one locked step, so pausing mid-rebuild never leaves it half-built.

    job.video_id stays None: the rebuild must never change a walk's status (or be cancelled by deleting one)."""
    bid = job.building_id
    # only finished walks: queued/paused ones will be integrated when their own job runs
    videos = [v for v in store.load_map(bid).videos
              if v.status == "done" and (store.video_dir(bid, v.id) / "analysis.mp4").exists()]
    default_model = (store.load_map(bid).processed_with or {}).get("model")
    walks = []
    for i, v in enumerate(videos):
        job.say(f"Reprocessing {v.filename} ({i + 1}/{len(videos)})")
        vdir = store.video_dir(bid, v.id)
        # Re-analyse with the model the walk was processed with, so its cached answers are reused.
        stored = _stored_model(vdir) or default_model
        try:
            sensors = store.read_json(vdir / "sensors.json")
            job.enter("motion", i / len(videos))
            track = motion.compute_track(vdir / "analysis.mp4", sensors, log=job.say)
            events = motion.detect_events(track)
            store.write_json_atomic(vdir / "track.json", {**track, "events": events}, mkdir=False)
            job.enter("keyframes", i / len(videos))
            keyframes = select_keyframes(vdir / "hq.mp4", track, events, v.duration or track["t"][-1], v.id,
                                         store.frames_dir(bid), log=job.say)
            run = _analyze_walk(job, keyframes, stored)
            store.write_json_atomic(vdir / "walk.json", {
                "video_id": v.id, "duration": v.duration, "keyframes": [k.model_dump() for k in keyframes],
                "events": events, "gemma": run.ok, "gemma_frames_ok": _frames_ok(keyframes),
                "model": run.model or stored,
            }, mkdir=False)
        except JobPaused:
            raise
        except Exception:
            if vdir.exists():
                raise
            job.say(f"{v.filename} was deleted, skipping it")
            _purge_files(bid, v.id)  # keyframes written after the delete
            continue
        walks.append((v, keyframes, track, run, run.model or stored))

    job.enter("mapping")
    models = [m for *_, m in walks if m]
    online = any(w[3].online for w in walks) if walks else _gemma_ok()
    with store.lock_for(bid), gemma.session(default_model or (models[-1] if models else None), lookup=models,
                                            offline=not online):
        bundle = store.load_map(bid)
        bundle.nodes, bundle.edges, bundle.suggested_merges = [], [], []
        for v, keyframes, track, run, wmodel in walks:
            info = next((x for x in bundle.videos if x.id == v.id), None)
            if info is None:  # deleted while it was being re-analyzed
                job.say(f"{v.filename} was deleted, skipping it")
                _purge_files(bid, v.id)
                continue
            with gemma.session(wmodel):  # this walk's own same-place checks: its own model
                res = mapper.integrate_walk(bundle, v.id, keyframes, track, store.frames_dir(bid),
                                            gemma.cache_for(store.building_dir(bid)), run.ok, log=job.say,
                                            start_floor=info.start_floor)
            info.floor_offset = res.get("floor_offset") or 0
            job.say(f"{v.filename}: {res}")
        job.say("Linking all walks into one building map")
        # use_gemma=True even offline: inside the session, cached answers are used and misses are instant
        job.result = mapper.consolidate(bundle, store.frames_dir(bid), gemma.cache_for(store.building_dir(bid)),
                                        True, log=job.say)
        store.save_map(bundle)


def submit_consolidate(building_id: str) -> Job:
    job = Job(id=uuid.uuid4().hex[:10], building_id=building_id, video_id=None, kind="consolidate",
              filename="(link videos)")
    return _enqueue(job, lambda: _consolidate(job), front=True)


def _consolidate(job: Job) -> None:
    bid = job.building_id
    job.enter("mapping")
    job.say("Linking all walks into one building map")
    online = _gemma_ok()
    with store.lock_for(bid):
        bundle = store.load_map(bid)
        models = [m for v in bundle.videos if (m := _stored_model(store.video_dir(bid, v.id)))]
        with gemma.session((bundle.processed_with or {}).get("model"), lookup=models, offline=not online):
            job.result = mapper.consolidate(bundle, store.frames_dir(bid), gemma.cache_for(store.building_dir(bid)),
                                            True, log=job.say)
        store.save_map(bundle)


def _stored_model(vdir: Path) -> str | None:
    """Model a walk was processed with (walk.json), if recorded."""
    try:
        return (store.read_json(vdir / "walk.json") or {}).get("model")
    except (OSError, ValueError, AttributeError):
        return None


def _gemma_ok() -> bool:
    try:
        gemma.resolve_model()
        return True
    except gemma.GemmaUnavailable:
        return False
