"""Batch ingest without the web UI (e.g. pre-process the whole building before the demo).

    uv run python -m app.cli ingest "Sample Building" ../IMG_3177.MOV ../IMG_3178.MOV ...
    uv run python -m app.cli rebuild sample-building
    uv run python -m app.cli route sample-building n001 n014
"""
import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

from . import jobs, routing, store


def _wait(job: jobs.Job) -> None:
    seen = 0
    while job.status in ("queued", "running"):
        for line in job.log[seen:]:
            print(line, flush=True)
        seen = len(job.log)
        time.sleep(0.5)
    for line in job.log[seen:]:
        print(line, flush=True)
    if job.status == "error":
        sys.exit(f"failed: {job.error}")


def main() -> None:
    p = argparse.ArgumentParser(prog="app.cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    ing = sub.add_parser("ingest")
    ing.add_argument("building", help="building name or existing id")
    ing.add_argument("videos", nargs="+", type=Path)
    ing.add_argument("--model")
    reb = sub.add_parser("rebuild")
    reb.add_argument("building")
    rt = sub.add_parser("route")
    rt.add_argument("building"); rt.add_argument("start"); rt.add_argument("target")
    args = p.parse_args()

    if args.cmd == "ingest":
        ids = {b["id"]: b for b in store.list_buildings()}
        bid = args.building if args.building in ids else store.slugify(args.building)
        if bid not in ids:
            bid = store.create_building(args.building).building_id
        for v in args.videos:
            tmp = Path(tempfile.mkstemp(suffix=v.suffix)[1])
            shutil.copy(v, tmp)
            sensors_path = v.with_suffix(".sensors.json")
            sensors = json.loads(sensors_path.read_text()) if sensors_path.exists() else None
            print(f"=== {v.name} -> {bid}", flush=True)
            _wait(jobs.submit_video(bid, tmp, v.name, sensors, args.model))
    elif args.cmd == "rebuild":
        _wait(jobs.submit_rebuild(args.building))
    elif args.cmd == "route":
        r = routing.route(store.load_map(args.building), args.start, args.target)
        for s in r.steps:
            print(f"{s.step_index + 1:2d}. [{s.action:>19}] {s.instruction_text}  ({s.frame_url})")


if __name__ == "__main__":
    main()
