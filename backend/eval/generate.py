"""Render a synthetic scenario: MP4 walkthroughs + truth.json (+ per-frame sidecars for the oracle).

    .venv/bin/python -m eval.generate two_floor          # regenerate if the spec or generator changed
    .venv/bin/python -m eval.generate all --force

Scenario specs live in eval/scenarios/<name>/spec.py (a SPEC dict). Output:
    eval/scenarios/<name>/truth.json            ground truth (tracked in git)
    eval/scenarios/<name>/media/<video>.mp4     rendered walks (gitignored)
    eval/scenarios/<name>/media/<video>.frames.json   what each frame shows (decals, stairs), for --oracle
    eval/scenarios/<name>/media/manifest.json   fingerprint of spec + generator, to skip re-rendering
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .building import Building
from .render import Camera, Renderer
from .walk import FPS, Walker, bearing, wrap180

EVAL_DIR = Path(__file__).resolve().parent
SCENARIOS_DIR = EVAL_DIR / "scenarios"
GENERATOR_VERSION = 1
HFOV_DEG = 69.0  # app.config.LONG_SIDE_FOV_DEG default


def visits(walker: Walker, radius: float = 1.25, merge_gap: float = 4.0, max_off_deg: float = 60.0) -> list[dict]:
    """Ordered places the walk passed: one entry per pass within `radius` m of a place's standing point
    (at the moment of closest approach), or at the glance if the walker looked at it. A pass by a place with
    a sign (door, exit) only counts if the camera pointed within `max_off_deg` of that sign at some moment of
    the pass: walking away from an exit behind you is not visiting it."""
    b = walker.b
    xy = np.array([[f.pose.x, f.pose.y] for f in walker.frames])
    out = []
    for p in b.places.values():
        on = np.array([f.level == p.floor for f in walker.frames])
        d = np.hypot(xy[:, 0] - p.stand[0], xy[:, 1] - p.stand[1])
        near = on & (d <= radius)
        i = 0
        while i < len(near):
            if not near[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(near) and near[j + 1]:
                j += 1
            k = i + int(np.argmin(d[i:j + 1]))
            if p.look is None or any(
                    abs(wrap180(bearing(f.pose.x, f.pose.y, p.look[0], p.look[1]) - f.pose.yaw)) <= max_off_deg
                    for f in walker.frames[i:j + 1]):
                out.append({"place": p.id, "t": round(k / FPS, 2), "kind": "pass"})
            i = j + 1
    for t, pid in walker.looks:
        same = [v for v in out if v["place"] == pid and abs(v["t"] - t) <= merge_gap]
        if same:
            v = min(same, key=lambda v: abs(v["t"] - t))
            v["t"], v["kind"] = round(t, 2), "look"
        else:
            out.append({"place": pid, "t": round(t, 2), "kind": "look"})
    out.sort(key=lambda v: v["t"])
    merged: list[dict] = []
    for v in out:  # one place passed twice in a row a moment apart = one visit
        prev = next((m for m in reversed(merged) if m["place"] == v["place"]), None)
        if prev is not None and v["t"] - prev["t"] < merge_gap and merged[-1] is prev:
            if v["kind"] == "look":
                prev.update(v)
            continue
        merged.append(v)
    return merged


def scenario_names() -> list[str]:
    return sorted(p.parent.name for p in SCENARIOS_DIR.glob("*/spec.py"))


def scenario_dir(name: str) -> Path:
    return SCENARIOS_DIR / name


def load_spec(name: str) -> dict:
    path = scenario_dir(name) / "spec.py"
    if not path.exists():
        raise FileNotFoundError(f"no scenario {name!r} ({path})")
    mod_spec = importlib.util.spec_from_file_location(f"eval_scenario_{name}", path)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    spec = dict(mod.SPEC)
    spec.setdefault("name", name)
    return spec


def fingerprint(name: str) -> str:
    """Changes whenever the rendered media would: the spec, the renderer, the geometry or the walker."""
    h = hashlib.sha1()
    for p in [scenario_dir(name) / "spec.py", EVAL_DIR / "render.py", EVAL_DIR / "building.py", EVAL_DIR / "walk.py"]:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    h.update(str(GENERATOR_VERSION).encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- one video
def _ffmpeg_writer(path: Path, w: int, h: int, fps: int) -> subprocess.Popen:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH (brew install ffmpeg)")
    return subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
         "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
         "-threads", "1", str(path)],
        stdin=subprocess.PIPE)


def walk_video(spec: dict, vspec: dict, building: Building | None = None) -> Walker:
    b = building or Building(spec)
    w = Walker(b, vspec["start_floor"], tuple(vspec["start"]), vspec["heading"], seed=spec.get("seed", 0))
    w.run(vspec["script"])
    return w


def video_truth(spec: dict, index: int, building: Building) -> dict:
    """Truth entry of one video: runs the walker only (cheap), never renders."""
    vspec = spec["videos"][index]
    walker = walk_video(spec, vspec, building)
    return {
        "id": vspec["id"],
        "file": f"media/{vspec['id']}.mp4",
        "duration": round(len(walker.frames) / FPS, 1),
        "start_floor": vspec["start_floor"],
        "give_start_floor": bool(vspec.get("give_start_floor", False)),
        "description": vspec.get("description", ""),
        "visits": visits(walker),
        "turns": walker.turns,
        # where the walker was every 0.5 s: [t, x, y, floor] (floor null on the stairs), for far_merges
        "path": [[round(i / FPS, 1), round(f.pose.x, 2), round(f.pose.y, 2), f.level]
                 for i, f in enumerate(walker.frames) if i % (FPS // 2) == 0],
    }


def render_video(spec: dict | str, index: int, out_dir: Path) -> None:
    """Render video `index` of a scenario (spec dict or name) into out_dir: <id>.mp4 + <id>.frames.json."""
    spec = load_spec(spec) if isinstance(spec, str) else spec
    name = spec["name"]
    vspec = spec["videos"][index]
    b = Building(spec)
    walker = walk_video(spec, vspec, b)
    width, height = spec.get("size", (1280, 720))
    cam = Camera(width, height, HFOV_DEG)
    renderer = Renderer(cam, b.textures)
    decal_quads = {q.decal: q for q in b.quads if q.decal}
    mp4 = out_dir / f"{vspec['id']}.mp4"
    proc = _ffmpeg_writer(mp4, width, height, FPS)
    rng = np.random.default_rng(spec.get("seed", 0) * 1000 + index)
    frames_meta = []
    t0 = time.time()
    try:
        for i, fs in enumerate(walker.frames):
            quads = b.active_quads(fs.pose.x, fs.pose.y, fs.pose.z)
            img, zbuf = renderer.render(quads, fs.pose)
            seen = []
            for q in quads:
                if q.decal and (v := renderer.decal_view(decal_quads[q.decal], fs.pose, zbuf)):
                    seen.append([q.decal, v["h"], v["w"], v["cx"], v["cos"]])
            noise = rng.normal(0, 2.0, img.shape).astype(np.float32)  # sensor noise, so no frame is pixel-identical
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            proc.stdin.write(img.tobytes())
            frames_meta.append({"t": round(i / FPS, 2), "level": fs.level, "stairs": fs.stairs,
                                "pose": [round(fs.pose.x, 3), round(fs.pose.y, 3), round(fs.pose.z, 3),
                                         round(fs.pose.yaw % 360, 2), round(fs.pose.pitch, 2)],
                                "see": seen})
    finally:
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {mp4}")
    decals = {d.id: {"text": d.text, "kind": d.kind, "room": d.room, "wayfinding": d.wayfinding,
                     "style": d.style, "level": d.level} for d in b.decals.values()}
    (out_dir / f"{vspec['id']}.frames.json").write_text(json.dumps(
        {"fps": FPS, "size": [width, height], "decals": decals, "frames": frames_meta}, separators=(",", ":")))
    print(f"  {name}/{vspec['id']}: {len(walker.frames)} frames ({len(walker.frames) / FPS:.1f}s) rendered in "
          f"{time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- truth
def build_truth(spec: dict, videos: list[dict], b: Building) -> dict:
    places = [{"id": p.id, "label": p.label, "room": p.room, "floor": p.floor, "type": p.type,
               "x": round(p.stand[0], 2), "y": round(p.stand[1], 2), "group": p.group, "aliases": p.aliases}
              for p in b.places.values()]
    visited: dict[str, set[str]] = {}
    count: dict[str, int] = {}
    for v in videos:
        for vis in v["visits"]:
            visited.setdefault(vis["place"], set()).add(v["id"])
            count[vis["place"]] = count.get(vis["place"], 0) + 1
    by_id = {p["id"]: p for p in places}
    # Places walked past more than once should end up as one map place (a revisit / loop closure).
    expect_merged = spec.get("expect_merged") or sorted(
        pid for pid, n in count.items() if n >= 2 and by_id[pid]["type"] in ("room", "stairs", "exit", "elevator"))
    must_not = [list(x) for x in spec.get("must_not_merge", [])]
    have = {frozenset(x[:2]) for x in must_not}
    for a, c in itertools.combinations(places, 2):
        pair = frozenset((a["id"], c["id"]))
        if pair in have:
            continue
        if a["room"] and c["room"] and a["room"] != c["room"]:
            must_not.append([a["id"], c["id"], f"different rooms {a['room']} / {c['room']}"])
        elif a["floor"] != c["floor"] and a["type"] in ("stairs", "elevator") and a["type"] == c["type"]:
            must_not.append([a["id"], c["id"], f"same {a['type']} on floors {a['floor']} and {c['floor']}"])
    return {
        "schema": "waypoint-eval/1",
        "scenario": spec["name"],
        "description": spec.get("description", ""),
        "source": {"kind": "synthetic", "generator": "eval.generate", "generator_version": GENERATOR_VERSION,
                   "seed": spec.get("seed", 0), "fps": FPS, "size": list(spec.get("size", (1280, 720)))},
        "building": {
            "floors": sorted(b.levels),
            "places": places,
            "distractors": [{"id": d.id, "text": d.text, "kind": d.kind, "floor": d.level,
                             "x": round(float(d.pos[0]), 2), "y": round(float(d.pos[1]), 2)}
                            for d in b.decals.values() if not d.wayfinding],
        },
        "videos": videos,
        "expect_merged": expect_merged,
        "must_not_merge": must_not,
        "stair_links": spec.get("stair_links", []),
        "routes": spec.get("routes", []),
    }


def _compact_json(data) -> str:
    """Indented JSON with every innermost list ([t, x, y, floor], [place, place, reason]) on one line."""
    text = json.dumps(data, indent=1)
    def one_line(m: re.Match) -> str:
        return re.sub(r"\s*\n\s*", " ", m.group()).replace("[ ", "[").replace(" ]", "]")

    return re.sub(r"\[[^\[\]{}]*\]", one_line, text) + "\n"


def generate_spec(spec: dict, out_dir: Path, fp: str | None = None, force: bool = False, workers: int = 4) -> Path:
    """Render a spec's videos into out_dir/media (unless up to date with fingerprint `fp`) and write
    out_dir/truth.json. Returns the truth.json path."""
    media = out_dir / "media"
    media.mkdir(parents=True, exist_ok=True)
    manifest = media / "manifest.json"
    n = len(spec["videos"])
    fresh = (fp is not None and not force and manifest.exists()
             and json.loads(manifest.read_text()).get("fingerprint") == fp
             and all((media / f"{v['id']}.mp4").exists() and (media / f"{v['id']}.frames.json").exists()
                     for v in spec["videos"]))
    if not fresh:
        print(f"Rendering scenario {spec['name']} ({n} videos)", flush=True)
        if workers > 1 and n > 1:
            with ProcessPoolExecutor(max_workers=min(workers, n)) as pool:
                list(pool.map(render_video, [spec] * n, range(n), [media] * n))
        else:
            for i in range(n):
                render_video(spec, i, media)
        manifest.write_text(json.dumps({"fingerprint": fp, "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}))
    b = Building(spec)
    truth = build_truth(spec, [video_truth(spec, i, b) for i in range(n)], b)
    truth_path = out_dir / "truth.json"
    text = _compact_json(truth)
    if not truth_path.exists() or truth_path.read_text() != text:
        truth_path.write_text(text)
    return truth_path


def generate(name: str, force: bool = False, workers: int = 4) -> Path:
    """Render scenario `name` if its media is missing or stale, and (re)write its truth.json."""
    return generate_spec(load_spec(name), scenario_dir(name), fingerprint(name), force, workers)


def main() -> None:
    p = argparse.ArgumentParser(prog="eval.generate")
    p.add_argument("scenario", help="scenario name or 'all'")
    p.add_argument("--force", action="store_true", help="re-render even if up to date")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    names = scenario_names() if args.scenario == "all" else [args.scenario]
    for n in names:
        print(generate(n, args.force, args.workers))


if __name__ == "__main__":
    sys.exit(main())
