"""Run the real pipeline on evaluation scenarios and score the resulting map against ground truth.

    cd backend
    .venv/bin/python -m eval.run all                          # live Gemma (GEMMA_MODEL, default gemma3:4b)
    .venv/bin/python -m eval.run two_floor --model gemma3:12b
    .venv/bin/python -m eval.run all --oracle                 # perfect perception from ground truth, no Ollama
    .venv/bin/python -m eval.run all --no-gemma               # motion only
    .venv/bin/python -m eval.run all --reuse-cache            # reuse Gemma answers from the last live run

Each scenario is ingested into a fresh building inside a temporary data dir (never backend/data), video by
video, through app.jobs exactly as an upload. The scorecard is printed and saved to
eval/results/<timestamp>-<scenario>-<mode>.json (+ .log with every job log line), and compared with the
previous result of the same scenario and mode.

Exit code: 0 ok, 1 a hard safety metric failed (wrong merge, far merge or invented room number), 2 setup/pipeline
error (including a refused --data-dir).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EVAL_DIR.parent
RESULTS_DIR = EVAL_DIR / "results"
CACHE_DIR = EVAL_DIR / ".cache"
FORBIDDEN = (BACKEND_DIR / "data").resolve()


class PipelineError(RuntimeError):
    pass


class SetupError(RuntimeError):
    """Bad arguments or environment: nothing was run (exit code 2)."""


def safe_data_dir(path: Path) -> Path:
    path = Path(path).resolve()
    if path == FORBIDDEN or FORBIDDEN in path.parents:
        raise SetupError(f"refusing to use {path}: evaluation never writes into backend/data")
    return path


# ---------------------------------------------------------------- patches (scoped to one run)
@contextmanager
def data_dir(path: Path):
    from app import config
    old = config.DATA_DIR, config.BUILDINGS_DIR
    config.DATA_DIR, config.BUILDINGS_DIR = path, path / "buildings"
    try:
        yield
    finally:
        config.DATA_DIR, config.BUILDINGS_DIR = old


@contextmanager
def gemma_off():
    from app import gemma

    def off(*_a, **_k):
        raise gemma.GemmaUnavailable("disabled by eval --no-gemma")

    saved = gemma.resolve_model, gemma.ensure_model
    gemma.resolve_model = gemma.ensure_model = off
    try:
        yield
    finally:
        gemma.resolve_model, gemma.ensure_model = saved


def _submit(bid: str, src: Path, filename: str, sensors, model, start_floor):
    """jobs.submit_video, but the job is handed to us instead of the shared worker thread (so the runner owns
    its timing, and a worker left over from elsewhere in the process cannot pick it up)."""
    from app import jobs

    def enqueue(job, fn, front=False):
        job.fn, job.status = fn, "running"
        with jobs._cv:
            jobs.JOBS[job.id] = job
        return job

    saved = jobs._enqueue
    jobs._enqueue = enqueue
    try:
        return jobs.submit_video(bid, src, filename, sensors, model, start_floor=start_floor)
    finally:
        jobs._enqueue = saved


def _run_job(job, verbose: bool) -> dict:
    """Run a submitted job to completion; return seconds per stage."""
    from app import jobs
    th = threading.Thread(target=jobs._run, args=(job,), daemon=True)
    stages: dict[str, float] = {}
    cur, since, seen = None, time.time(), 0
    th.start()
    while th.is_alive() or seen < len(job.log):
        if job.stage != cur:
            now = time.time()
            if cur is not None:
                stages[cur] = round(stages.get(cur, 0.0) + now - since, 2)
            cur, since = job.stage, now
        if verbose:
            for line in job.log[seen:]:
                print("    " + line, flush=True)
        seen = len(job.log)
        th.join(0.2)
    if cur not in (None, "done"):
        stages[cur] = round(stages.get(cur, 0.0) + time.time() - since, 2)
    return stages


# ---------------------------------------------------------------- one scenario
def run_scenario(truth_path: Path, mode: str, model: str | None, root: Path, verbose: bool = False,
                 reuse_cache: bool = False) -> dict:
    from app import gemma, store

    from .oracle import Oracle
    from .score import score
    from .trace import Tracer
    from .truth import load

    truth = load(truth_path)
    sdir = truth_path.parent
    for v in truth.videos:
        if not (sdir / v.file).exists():
            raise PipelineError(f"missing video {sdir / v.file} (run: python -m eval.generate {truth.scenario})")
    tracer = Tracer()
    oracle = Oracle(truth, sdir, tracer) if mode == "oracle" else None
    logs: dict[str, list[str]] = {}
    timings = []
    t_start = time.time()
    cache_file = CACHE_DIR / truth.scenario / f"{(model or 'default').replace(':', '_')}.gemma_cache.jsonl"
    with ExitStack() as stack:
        stack.enter_context(data_dir(root))
        stack.enter_context(tracer.installed())
        if oracle:
            stack.enter_context(oracle.installed())
        if mode == "no-gemma":
            stack.enter_context(gemma_off())
        bundle0 = store.create_building(f"eval {truth.scenario} {time.strftime('%H%M%S')}")
        bid = bundle0.building_id
        bdir = store.building_dir(bid)
        if reuse_cache and mode == "live" and cache_file.exists():
            shutil.copy(cache_file, bdir / gemma.CACHE_FILE)
        names: dict[str, str] = {}
        uploads = root / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        for v in truth.videos:
            src = sdir / v.file
            tmp = uploads / f"{bid}-{v.id}{src.suffix}"
            shutil.copy(src, tmp)
            sensors = json.loads((sdir / v.sensors).read_text()) if v.sensors else None
            start_floor = v.start_floor if v.give_start_floor else None
            job = _submit(bid, tmp, src.name, sensors, None if mode == "oracle" else model, start_floor)
            names[job.video_id] = v.id
            if oracle:
                oracle.video_names[job.video_id] = v.id
            print(f"  [{truth.scenario}] {v.id} ({v.duration or 0:.0f}s video, start floor "
                  f"{start_floor if start_floor is not None else 'not given'}) ...", flush=True)
            t0 = time.time()
            stages = _run_job(job, verbose)
            logs[v.id] = list(job.log)
            if job.status != "done":
                raise PipelineError(f"{v.id}: job {job.status}: {job.error}\n" + "\n".join(job.log[-15:]))
            wall = time.time() - t0
            kf = next((line.split("Selected ")[1].split()[0] for line in job.log if "Selected " in line), None)
            timings.append({"video": v.id, "video_s": v.duration, "wall_s": round(wall, 1), "stages": stages,
                            "keyframes": int(kf) if kf and kf.isdigit() else None})
            print(f"      done in {wall:.0f}s  " + "  ".join(f"{k} {s:.0f}s" for k, s in stages.items()), flush=True)
        bundle = store.load_map(bid)
        obs = tracer.observations(bundle, names)
        events = {name: (store.read_json(store.video_dir(bid, vid) / "track.json") or {}).get("events", [])
                  for vid, name in names.items()}
        if mode == "live":
            src_cache = bdir / gemma.CACHE_FILE
            if src_cache.exists():
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src_cache, cache_file)
    result = score(truth, bundle, obs, events)
    result.update({
        "scenario": truth.scenario, "mode": mode, "model": model if mode == "live" else None,
        "source": truth.source, "building_id": bid, "data_dir": str(root),
        "timings": {"total_wall_s": round(time.time() - t_start, 1),
                    "video_s": round(sum(v.duration or 0 for v in truth.videos), 1), "videos": timings},
        "observations": obs,
        "events": events,
        "bundle": bundle.model_dump(),  # the final map, so `python -m eval.score` can re-score later
    })
    result["_logs"] = logs
    return result


# ---------------------------------------------------------------- reporting
def _git() -> dict:
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=BACKEND_DIR, capture_output=True,
                              text=True, timeout=5).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "app"], cwd=BACKEND_DIR,
                                    capture_output=True, text=True, timeout=5).stdout.strip())
        return {"commit": head, "app_dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {}


def previous_result(scenario: str, mode: str, model: str | None, exclude: Path | None) -> dict | None:
    if not RESULTS_DIR.exists():
        return None
    for p in sorted(RESULTS_DIR.glob(f"*-{scenario}-{mode}.json"), reverse=True):
        if exclude is not None and p.resolve() == exclude.resolve():
            continue
        try:
            data = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if data.get("model") == model:
            data["_path"] = str(p)
            return data
    return None


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def compare(cur: dict, prev: dict | None) -> list[dict]:
    from .score import METRICS
    rows = []
    for m, (higher, gate) in METRICS.items():
        v = cur["metrics"].get(m)
        p = prev["metrics"].get(m) if prev else None
        flag = ""
        if prev is not None and v is not None and p is not None and v != p:
            better = (v > p) == higher
            flag = "improved" if better else "REGRESSED"
        rows.append({"metric": m, "value": v, "count": cur["counts"].get(m), "prev": p, "flag": flag, "gate": gate})
    return rows


def print_scorecard(res: dict, prev: dict | None) -> None:
    t = res["timings"]
    head = (f"\n=== {res['scenario']}  mode={res['mode']}" + (f" model={res['model']}" if res.get("model") else "")
            + f"  ({len(t['videos'])} videos, {t['video_s']:.0f}s of video, {t['total_wall_s']:.0f}s wall)")
    print(head)
    m = res["map"]
    print(f"map: {m['places']} places, {m['edges']} edges ({m['vertical_edges']} vertical), floors {m['floors']}, "
          f"{m['suggested_merges']} suggested merges")
    print(f"{'metric':<22}{'value':>8}{'':>8}  {'prev':>6}  ")
    for r in compare(res, prev):
        mark = " [GATE]" if r["gate"] else ""
        cnt = f"({r['count']})" if r["count"] else ""
        print(f"{r['metric'] + mark:<22}{_fmt(r['value']):>8}{cnt:>8}  {_fmt(r['prev']):>6}  {r['flag']}")
    d = res["details"]
    for w in d["wrong_merges"]:
        print(f"  WRONG MERGE: {w['places'][0]} + {w['places'][1]} in {', '.join(w['nodes'])} ({w['why']})")
    for i in d["invented"]:
        print(f"  INVENTED ROOM: {i['room']} on {i['node']} '{i['label']}'")
    for i in d["misattributed"]:
        print(f"  misattributed: room {i['room']} on {i['node']} '{i['label']}', which is {', '.join(i['is_at'])}")
    for pid in d["missing_rooms"]:
        print(f"  room not found: {pid}")
    for f in d.get("far_merges", []):
        print(f"  far merge: {f['node']} joins {f['a'][0]}@{f['a'][1]:.0f}s (floor {f['a'][2]}) and "
              f"{f['b'][0]}@{f['b'][1]:.0f}s (floor {f['b'][2]}), {f['distance_m']} m apart")
    for mm in d["missed_merges"]:
        print(f"  missed merge: {mm['place']} split over {', '.join(mm['nodes'])}")
    for f in d["floor_errors"]:
        print(f"  floor error: {f['place']} is on floor {f['truth']}, map says {f['map']}")
    for link in d["stair_links"]:
        if not link["ok"]:
            print(f"  stair link missing: {link['link'][0]} <-> {link['link'][1]}")
    for r in d["routes"]:
        if not r["ok"]:
            print(f"  route failed: {r['from']} -> {r['to']}: {r.get('error')}")
    bad_turns = [t for t in d["turns"] if t["detected"] is None or t["truth"] is None or t.get("error", 0) > 20]
    if bad_turns:
        print("  turns off by >20 deg / missed / spurious: " + ", ".join(
            f"{t['video']}@{t['t']:.0f}s {_fmt(t['truth'])}->{_fmt(t['detected'])}" for t in bad_turns))
    if prev:
        print(f"(compared with {Path(prev['_path']).name})")
    if res["hard_fail"]:
        print(f"HARD FAIL: {', '.join(res['hard_fail'])}")


def save(res: dict, stamp: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base = RESULTS_DIR / f"{stamp}-{res['scenario']}-{res['mode']}"
    logs = res.pop("_logs", {})
    with open(base.with_suffix(".log"), "w") as f:
        for vid, lines in logs.items():
            f.write(f"===== {vid}\n" + "\n".join(lines) + "\n")
    path = base.with_suffix(".json")
    path.write_text(json.dumps(res, indent=1, default=str) + "\n")
    return path


# ---------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="eval.run", description=__doc__.split("\n\n")[0])
    p.add_argument("scenarios", nargs="+", help="scenario names, 'all', or paths to truth.json files")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--no-gemma", action="store_true", help="motion only: Gemma (and OCR) off")
    g.add_argument("--oracle", action="store_true", help="answer Gemma questions from ground truth")
    p.add_argument("--model", help="Ollama model for live runs (default: GEMMA_MODEL)")
    p.add_argument("--data-dir", type=Path, help="data dir to ingest into (default: a fresh temp dir)")
    p.add_argument("--keep", action="store_true", help="keep the temp data dir for inspection")
    p.add_argument("--reuse-cache", action="store_true", help="seed Gemma's cache with the last live run's answers")
    p.add_argument("--no-generate", action="store_true", help="do not (re)render synthetic scenarios")
    p.add_argument("--no-save", action="store_true", help="do not write eval/results")
    p.add_argument("-v", "--verbose", action="store_true", help="print every pipeline log line")
    args = p.parse_args(argv)

    try:
        root = safe_data_dir(args.data_dir) if args.data_dir else Path(tempfile.mkdtemp(prefix="waypoint-eval-"))
    except SetupError as exc:
        print(exc, file=sys.stderr)
        return 2
    try:
        return _main(args, root)
    finally:
        # a temp dir goes unless --keep asked for it and something was ingested (setup errors leave it empty)
        ingested = root.is_dir() and any(root.iterdir())
        if not args.data_dir and not (args.keep and ingested):
            shutil.rmtree(root, ignore_errors=True)
        elif ingested:
            print(f"data dir kept: {root}")


def _main(args: argparse.Namespace, root: Path) -> int:
    os.environ.setdefault("DATA_DIR", str(root))  # before app.config is first imported
    from app import config, gemma

    from .generate import generate, scenario_dir, scenario_names

    mode = "oracle" if args.oracle else "no-gemma" if args.no_gemma else "live"
    model = (args.model or config.GEMMA_MODEL) if mode == "live" else None
    if mode == "live":
        try:
            gemma.ensure_model(model)
        except gemma.GemmaUnavailable as exc:
            print(f"Gemma is not usable: {exc}\n(start `ollama serve`, or run with --oracle / --no-gemma)")
            return 2

    targets: list[Path] = []
    for s in args.scenarios:
        if s == "all":
            names = scenario_names()
        elif s.endswith(".json"):
            targets.append(Path(s).resolve())
            continue
        else:
            names = [s]
        for n in names:
            sdir = scenario_dir(n)
            if (sdir / "spec.py").exists() and not args.no_generate:
                targets.append(generate(n))
            elif (sdir / "truth.json").exists():
                targets.append(sdir / "truth.json")
            else:
                print(f"unknown scenario {n!r}")
                return 2

    stamp = time.strftime("%Y%m%d-%H%M%S")
    git = _git()
    code = 0
    summary = []
    for truth_path in targets:
        try:
            res = run_scenario(truth_path, mode, model, root, args.verbose, args.reuse_cache)
        except PipelineError as exc:
            print(f"PIPELINE ERROR in {truth_path.parent.name}: {exc}")
            code = 2
            continue
        res["git"] = git
        res["stamp"] = stamp
        out = None if args.no_save else save(res, stamp)
        prev = previous_result(res["scenario"], mode, model, out)
        print_scorecard(res, prev)
        if out:
            print(f"saved {out.relative_to(BACKEND_DIR) if out.is_relative_to(BACKEND_DIR) else out}")
        summary.append(res)
        if res["hard_fail"] and code == 0:
            code = 1
    if len(summary) > 1:
        print("\nscenario            rooms  invent  wrongM  mergeR  floors  stairs  routes   wall")
        for r in summary:
            m = r["metrics"]
            print(f"{r['scenario']:<18}{_fmt(m['rooms_found']):>7}{m['invented_rooms']:>8}{m['wrong_merges']:>8}"
                  f"{_fmt(m['merge_recall']):>8}{_fmt(m['floor_accuracy']):>8}{_fmt(m['stair_links']):>8}"
                  f"{_fmt(m['route_success']):>8}{r['timings']['total_wall_s']:>7.0f}s")
    return code


if __name__ == "__main__":
    sys.exit(main())
