"""Batch ingest without the web UI (e.g. pre-process the whole building before the demo).

    uv run python -m app.cli ingest "Sample Building" ../IMG_3177.MOV ../IMG_3178.MOV ...
    uv run python -m app.cli rebuild sample-building
    uv run python -m app.cli route sample-building n001 n014
    uv run python -m app.cli route sample-building n001 n014 --avoid-stairs

Access control (auth.py):
    uv run python -m app.cli owner-key [--rotate]
    uv run python -m app.cli invite sample-building --role viewer|editor [--label "Front desk"]
    uv run python -m app.cli invites sample-building
    uv run python -m app.cli revoke sample-building <invite_id>
"""
import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

from . import auth, jobs, routing, store


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


def _existing_building(building: str) -> str:
    try:
        store.load_map(building)
    except (FileNotFoundError, ValueError):
        sys.exit(f"Unknown building '{building}'")
    return building


def main() -> None:
    p = argparse.ArgumentParser(prog="app.cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    ing = sub.add_parser("ingest")
    ing.add_argument("building", help="building name or existing id")
    ing.add_argument("videos", nargs="+", type=Path)
    ing.add_argument("--model", help="gemma3:4b (local, Ollama) or gemini-2.5-flash-lite (hosted); "
                                     "default: VISION_PROVIDER decides")
    reb = sub.add_parser("rebuild")
    reb.add_argument("building")
    rt = sub.add_parser("route")
    rt.add_argument("building"); rt.add_argument("start"); rt.add_argument("target")
    rt.add_argument("--avoid-stairs", action="store_true", help="only flat walks, ramps and elevators")
    ok = sub.add_parser("owner-key", help="show where the owner key is; --rotate makes a new one")
    ok.add_argument("--rotate", action="store_true", help="replace the owner key (printed once)")
    inv = sub.add_parser("invite", help="create an invite token for one building (printed once)")
    inv.add_argument("building")
    inv.add_argument("--role", required=True, choices=["viewer", "editor"])
    inv.add_argument("--label", default="", help="who it is for (max 80 characters)")
    invs = sub.add_parser("invites", help="list a building's invites (no tokens)")
    invs.add_argument("building")
    rv = sub.add_parser("revoke", help="revoke an invite: its token stops working at once")
    rv.add_argument("building"); rv.add_argument("invite_id")
    args = p.parse_args()

    if args.cmd == "owner-key":
        if args.rotate:
            try:
                raw = auth.rotate_owner_key()
            except RuntimeError as exc:
                sys.exit(str(exc))
            print("New owner key (shown once; the old key stops working now):")
            print(raw)
            return
        st = auth.owner_key_status()
        print(f"Owner key: {'set' if st['set'] else 'not set (the server creates one on first start)'}")
        if st["source"]:
            print(f"Source: {st['source']}")
        print(f"Stored hash file: {st['path']}")
        return
    if args.cmd == "invite":
        bid = _existing_building(args.building)
        if len(args.label) > 80:
            sys.exit("label must be at most 80 characters")
        entry, token = auth.create_invite(bid, args.role, args.label.strip())
        print(f"Invite {entry['id']} ({args.role}) for {bid}; shown once:")
        print(f"join path: #/join?b={bid}&t={token}")
        return
    if args.cmd == "invites":
        rows = auth.list_invites(_existing_building(args.building))
        print(f"{'id':<14}{'role':<8}{'revoked':<9}{'created':<27}label")
        for r in rows:
            print(f"{r['id']:<14}{r['role']:<8}{str(r['revoked']).lower():<9}{r['created_at']:<27}{r['label']}")
        if not rows:
            print("(no invites)")
        return
    if args.cmd == "revoke":
        entry = auth.revoke_invite(_existing_building(args.building), args.invite_id)
        if entry is None:
            sys.exit(f"No invite '{args.invite_id}' on {args.building}")
        print(f"Revoked invite {entry['id']} ({entry['role']})")
        return

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
        try:
            r = routing.route(store.load_map(args.building), args.start, args.target, avoid_stairs=args.avoid_stairs)
        except routing.NoStepFreePath as exc:
            sys.exit(str(exc))
        for s in r.steps:
            print(f"{s.step_index + 1:2d}. [{s.action:>19}] {s.instruction_text}  ({s.frame_url})")
        print(f"Step-free: {'yes' if r.step_free else 'no, this route uses stairs'}")


if __name__ == "__main__":
    main()
