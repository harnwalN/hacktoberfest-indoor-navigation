"""Background processing: normalize -> motion -> keyframes -> Gemma -> mapping.

Jobs run one at a time on a single worker thread (Gemma saturates the GPU anyway). Progress lives in
memory; every artifact the pipeline produces is written into the building bundle on disk.
"""
import shutil
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


@dataclass
class Job:
    id: str
    building_id: str
    video_id: str | None
    kind: str
    status: str = "queued"  # queued | running | done | error
    stage: str = "queued"
    pct: float = 0.0
    log: list[str] = field(default_factory=list)
    result: dict = field(default_factory=dict)
    error: str | None = None
    created: float = field(default_factory=time.time)

    def say(self, msg: str) -> None:
        self.log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def enter(self, stage: str, frac: float = 0.0) -> None:
        self.stage = stage
        lo, hi = next((a, b) for s, a, b in STAGES if s == stage)
        self.pct = round(lo + (hi - lo) * frac, 1)


JOBS: dict[str, Job] = {}
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline")


def get(job_id: str) -> Job | None:
    return JOBS.get(job_id)


def _set_video_status(building_id: str, video_id: str, **kw) -> None:
    with store.lock_for(building_id):
        bundle = store.load_map(building_id)
        v = next((v for v in bundle.videos if v.id == video_id), None)
        if v:
            for k, val in kw.items():
                setattr(v, k, val)
            store.save_map(bundle)


def submit_video(building_id: str, upload_path: Path, filename: str, sensors: dict | None, model: str | None) -> Job:
    video_id = f"v{time.strftime('%H%M%S')}{uuid.uuid4().hex[:4]}"
    vdir = store.video_dir(building_id, video_id)
    vdir.mkdir(parents=True, exist_ok=True)
    src = vdir / f"source{Path(filename).suffix.lower() or '.mp4'}"
    shutil.move(str(upload_path), src)
    if sensors:
        store.write_json_atomic(vdir / "sensors.json", sensors)
    with store.lock_for(building_id):
        bundle = store.load_map(building_id)
        bundle.videos.append(VideoInfo(id=video_id, filename=filename, has_sensors=bool(sensors)))
        store.save_map(bundle)
    job = Job(id=uuid.uuid4().hex[:10], building_id=building_id, video_id=video_id, kind="video")
    JOBS[job.id] = job
    _executor.submit(_run, job, lambda: _process_video(job, src, sensors, model))
    return job


def submit_rebuild(building_id: str) -> Job:
    job = Job(id=uuid.uuid4().hex[:10], building_id=building_id, video_id=None, kind="rebuild")
    JOBS[job.id] = job
    _executor.submit(_run, job, lambda: _rebuild(job))
    return job


def _run(job: Job, fn) -> None:
    job.status = "running"
    try:
        fn()
        job.status, job.stage, job.pct = "done", "done", 100.0
        job.say("Done")
    except Exception as exc:
        job.status, job.error = "error", f"{type(exc).__name__}: {exc}"
        job.say(f"ERROR {job.error}")
        traceback.print_exc()
        if job.video_id:
            _set_video_status(job.building_id, job.video_id, status="error")


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
        t0 = time.time()
        try:
            return kf, gemma.analyze_frame(frames / kf.frame_id, cache, model), time.time() - t0
        except gemma.GemmaUnavailable as exc:
            return kf, exc, 0.0

    # A few requests in flight keeps the GPU busy (~25% faster with OLLAMA_NUM_PARALLEL>1).
    with ThreadPoolExecutor(max_workers=config.GEMMA_CONCURRENCY) as pool:
        for i, (kf, res, dt) in enumerate(pool.map(one, keyframes)):
            job.enter("gemma", (i + 1) / max(len(keyframes), 1))
            if isinstance(res, Exception):
                job.say(f"WARNING Gemma failed on {kf.frame_id}: {res}")
                continue
            kf.analysis = a = res
            if a.has_sign or a.place_type != "hallway" or a.stairs != "none":
                job.say(f"{kf.frame_id} [{kf.kind}] -> {a.place_type}: {a.suggested_label!r} "
                        f"texts={a.texts} stairs={a.stairs} ({dt:.1f}s)")
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
        result = mapper.integrate_walk(bundle, vid, keyframes, track, store.frames_dir(bid),
                                       gemma.cache_for(store.building_dir(bid)), use_gemma, log=job.say)
        if use_gemma:
            bundle.processed_with = {"model": model or gemma.resolve_model(), "prompt": gemma.PROMPT_VERSION}
        v = next(v for v in bundle.videos if v.id == vid)
        v.duration, v.keyframes, v.status = round(duration, 1), len(keyframes), "done"
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
    videos = [v for v in bundle.videos if (store.video_dir(bid, v.id) / "analysis.mp4").exists()]
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
                                        gemma.cache_for(store.building_dir(bid)), use_gemma, log=job.say)
            store.save_map(bundle)
        job.say(f"{v.filename}: {res}")
    job.video_id = None
