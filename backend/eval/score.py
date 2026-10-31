"""Score a built map against ground truth.

Inputs: a Truth (eval/truth.py), the final MapBundle, and observations: per truth video, the walk nodes of
that video with the map place each one ended up in (eval/trace.py). Truth visits are matched to walk nodes by
time; map places are matched to truth places through those visits, and also by the room numbers they claim.

Hard safety metrics (a non-zero exit in eval.run): wrong merges, far merges, invented room numbers and false key
places (an exit/entrance/restroom/elevator that is not there).
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

import networkx as nx

from app import routing
from app.mapper import KEY_FAMILY
from app.matching import room_ids
from app.schemas import MapBundle

from .truth import Truth

LOOK_TOL_S = 1.5  # a glance at a plaque may sit up to this far outside a walk node's [arrive, depart]
PASS_TOL_S = 1.0

# metric -> (higher is better?, hard gate?) ; used for the scorecard and regression comparison
METRICS = {
    "rooms_found": (True, False),
    "rooms_located": (True, False),
    "invented_rooms": (False, True),
    "misattributed_rooms": (False, False),
    "wrong_merges": (False, True),
    "merge_recall": (True, False),
    "missed_merges": (False, False),
    "places_detected": (True, False),
    "floor_accuracy": (True, False),
    "stair_links": (True, False),
    "route_success": (True, False),
    "extra_places": (False, False),
    "far_merges": (False, True),
    "false_key_places": (False, True),
    "turns_detected": (True, False),
    "turn_error_deg": (False, False),
    "spurious_turns": (False, False),
}
FAR_MERGE_M = 5.0  # one map place whose observations were this far apart (or on two floors) is a bad merge
TURN_MATCH_S = 3.0  # a detected turn this close to a true one (centre to centre) is that turn
# Key places a visitor routes to: a map place of a KEY_FAMILY type (app.mapper: exit and entrance are one family)
# claims "there is an exit/restroom/... here".
KEY_NEAR_M = 5.0  # a key place seen from up to this far away (a walk ending facing the EXIT door) is that place


def _gap(t: float, o: dict) -> float:
    lo, hi = min(o["t_arrive"], o["t"]), max(o["t_depart"], o["t"])
    return 0.0 if lo <= t <= hi else min(abs(t - lo), abs(t - hi))


_LABEL_ROOM = re.compile(r"\b(?:ROOM|RM|OFFICE|SUITE|STE|LAB)\.?\s*#?\s*([A-Z]?\d{1,4}[A-Z]?)\b")


def claimed_rooms(node) -> set[str]:
    """Room numbers a map place presents: the plaques on its own photo (as the app matches them) and its label,
    where a 'Room 10' (a cut-off 'ROOM 101') counts too: a visitor would look for a room 10."""
    label = re.sub(r"\s*\(\d+\)$", "", node.label.upper())  # 'Room 101 (2)': dedupe suffix, not a number
    out = {r.upper() for r in room_ids([label] + list(node.sign_texts or node.texts))}
    return out | set(_LABEL_ROOM.findall(label))


def associate(truth: Truth, obs: dict[str, list[dict]], claims: dict[str, set[str]] | None = None,
              spots: set[str] | None = None) -> tuple[dict[str, Counter], list[dict]]:
    """place id -> Counter(map node id), and per-visit match records.

    A walker merely passing a room (no glance) is matched to a map place only if that place claims the room's
    number: otherwise the nearby place is a junction or corridor spot that saw the plaque, not the room. Likewise
    a sign read from afar (`see`) is matched only to a map place that claims the place: its room number, or for
    any other place not a corridor spot (`spots`: hallway / intersection places), where the walker merely stood."""
    assoc: dict[str, Counter] = defaultdict(Counter)
    records = []
    rooms = {p.id: p.room.upper() for p in truth.building.places if p.room}
    for video in truth.videos:
        walk = [o for o in obs.get(video.id, []) if o.get("node")]
        for v in video.visits:
            tol = PASS_TOL_S if v.kind == "pass" else LOOK_TOL_S
            best = min(walk, key=lambda o: (_gap(v.t, o), abs(v.t - o["t"])), default=None)
            hit = best is not None and _gap(v.t, best) <= tol
            if hit and v.kind in ("pass", "see") and v.place in rooms and claims is not None:
                hit = rooms[v.place] in claims.get(best["node"], set())
            elif hit and v.kind == "see" and spots is not None:
                hit = best["node"] not in spots
            if hit:
                assoc[v.place][best["node"]] += 1
            records.append({"video": video.id, "place": v.place, "t": v.t, "kind": v.kind,
                            "node": best["node"] if hit else None})
    return assoc, records


def _wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def score_turns(truth: Truth, events: dict[str, list[dict]] | None) -> dict:
    """Detected turn events (motion.detect_events) against the true changes of walking direction."""
    rows, errs, spurious, total = [], [], 0, 0
    for v in truth.videos:
        if not v.turns or events is None or v.id not in events:
            continue
        det = [e for e in events[v.id] if e.get("type") == "turn"]
        used: set[int] = set()
        for tt in v.turns:
            total += 1
            best = None
            for i, e in enumerate(det):
                mid = (e["start"] + e["end"]) / 2
                same_way = e["delta"] * tt.delta > 0 or abs(tt.delta) >= 150  # a U-turn may go either way
                if i not in used and same_way and abs(mid - tt.t) <= TURN_MATCH_S:
                    if best is None or abs(mid - tt.t) < best[0]:
                        best = (abs(mid - tt.t), i)
            if best is None:
                rows.append({"video": v.id, "t": tt.t, "truth": tt.delta, "detected": None})
                continue
            e = det[best[1]]
            used.add(best[1])
            err = abs(_wrap180(e["delta"] - tt.delta)) if abs(tt.delta) < 150 else abs(abs(e["delta"]) - abs(tt.delta))
            errs.append(err)
            rows.append({"video": v.id, "t": tt.t, "truth": tt.delta, "detected": e["delta"], "error": round(err, 1)})
        spurious += sum(1 for i, e in enumerate(det) if i not in used)
        rows += [{"video": v.id, "t": round((e["start"] + e["end"]) / 2, 1), "truth": None, "detected": e["delta"]}
                 for i, e in enumerate(det) if i not in used]
    return {"total": total, "matched": len(errs), "spurious": spurious, "rows": rows,
            "mean_error": round(sum(errs) / len(errs), 1) if errs else None}


def _where(path: list, t: float) -> tuple[float, float, int | None] | None:
    """Walker position at time t from truth path samples [t, x, y, floor] (nearest sample)."""
    if not path:
        return None
    s = min(path, key=lambda p: abs(p[0] - t))
    return (s[1], s[2], s[3]) if abs(s[0] - t) <= 1.0 else None


def far_merges(truth: Truth, obs: dict[str, list[dict]]) -> list[dict]:
    """Map places made of observations (any walk node: starts, junctions, corridor spots too) that were more
    than FAR_MERGE_M apart or on different floors. Needs per-video truth `path`s; [] without them."""
    members: dict[str, list[tuple[str, float, tuple]]] = defaultdict(list)
    for v in truth.videos:
        for o in obs.get(v.id, []):
            if o.get("node") and (pos := _where(v.path, o["t"])) is not None and pos[2] is not None:
                members[o["node"]].append((v.id, o["t"], pos))
    out = []
    for nid, ms in members.items():
        worst = None
        for i, (va, ta, pa) in enumerate(ms):
            for vb, tb, pb in ms[i + 1:]:
                d = ((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5
                if pa[2] != pb[2] or d > FAR_MERGE_M:
                    key = (pa[2] != pb[2], d)
                    if worst is None or key > worst[0]:
                        worst = (key, {"node": nid, "a": [va, ta, pa[2]], "b": [vb, tb, pb[2]],
                                       "distance_m": round(d, 1)})
        if worst:
            out.append(worst[1])
    return out


def false_key_places(truth: Truth, bundle: MapBundle, obs: dict[str, list[dict]],
                     node_places: dict[str, set[str]]) -> list[dict]:
    """Map places typed exit/entrance/restroom/elevator with no true place of that kind behind them: matched
    neither through a visit (the place's visits landed on it) nor by position (the walker was within KEY_NEAR_M
    of such a place, on its floor, when the map place was observed; needs truth `path`s). A dead end, a door at
    a corridor end or a NO EXIT sign typed 'exit' is one."""
    places = {p.id: p for p in truth.building.places}
    out = []
    for n in bundle.nodes:
        fam = KEY_FAMILY.get(n.type)
        if fam is None:
            continue
        if any(KEY_FAMILY.get(places[pid].type) == fam for pid in node_places.get(n.id, ())):
            continue
        near = False
        for v in truth.videos:
            for o in obs.get(v.id, []):
                if o.get("node") != n.id or (pos := _where(v.path, o["t"])) is None:
                    continue
                near = near or any(KEY_FAMILY.get(p.type) == fam and p.floor == pos[2] and p.x is not None
                                   and ((p.x - pos[0]) ** 2 + (p.y - pos[1]) ** 2) ** 0.5 <= KEY_NEAR_M
                                   for p in places.values())
        if not near:
            out.append({"node": n.id, "label": n.label, "type": n.type, "floor": n.floor,
                        "places": sorted(node_places.get(n.id, ())), "video": n.video_id, "t": n.t})
    return out


def _route_nodes(pid: str, assoc, truth: Truth, bundle: MapBundle) -> list[str]:
    """Map places standing for a truth place: by time association (most visits first), then by room number."""
    cands = [n for n, _ in assoc.get(pid, Counter()).most_common()]
    room = truth.place(pid).room
    if room:
        cands += [n.id for n in bundle.nodes if room.upper() in claimed_rooms(n) and n.id not in cands]
    return cands


def score(truth: Truth, bundle: MapBundle, obs: dict[str, list[dict]],
          events: dict[str, list[dict]] | None = None) -> dict:  # noqa: C901
    """`events`: per truth video, the motion events the pipeline detected (track.json), for turn metrics."""
    nodes = {n.id: n for n in bundle.nodes}
    claims = {nid: claimed_rooms(n) for nid, n in nodes.items()}
    spots = {nid for nid, n in nodes.items() if n.type in ("hallway", "intersection") and not n.vertical}
    assoc, records = associate(truth, obs, claims, spots)
    node_places: dict[str, set[str]] = defaultdict(set)
    for pid, cnt in assoc.items():
        for nid in cnt:
            node_places[nid].add(pid)
    places = {p.id: p for p in truth.building.places}
    visited = sorted({r["place"] for r in records})
    rooms = truth.rooms  # number -> place

    # --- rooms: recall and precision
    visited_rooms = [pid for pid in visited if places[pid].room]
    found, located, missing_rooms = [], [], []
    for pid in visited_rooms:
        num = places[pid].room.upper()
        holders = [nid for nid, c in claims.items() if num in c]
        if holders:
            found.append(pid)
            if any(pid in node_places.get(nid, ()) for nid in holders):
                located.append(pid)
        else:
            missing_rooms.append(pid)
    invented, misattributed = [], []
    for nid, c in claims.items():
        for num in sorted(c):
            if num not in rooms:
                invented.append({"node": nid, "label": nodes[nid].label, "room": num})
            elif node_places.get(nid) and rooms[num] not in node_places[nid]:
                misattributed.append({"node": nid, "label": nodes[nid].label, "room": num,
                                      "is_at": sorted(node_places[nid])})

    # --- merges
    wrong = []
    for pair in truth.must_not_merge:
        a, b = pair[0], pair[1]
        shared = set(assoc.get(a, {})) & set(assoc.get(b, {}))
        if shared:
            wrong.append({"places": [a, b], "nodes": sorted(shared), "why": pair[2] if len(pair) > 2 else ""})
    merged_ok, missed = [], []
    for pid in truth.expect_merged:
        n = len(assoc.get(pid, {}))
        if n == 1:
            merged_ok.append(pid)
        elif n > 1:
            missed.append({"place": pid, "nodes": sorted(assoc[pid]), "extra": n - 1})
    missed_count = sum(m["extra"] for m in missed)
    not_found_merge = [pid for pid in truth.expect_merged if not assoc.get(pid)]

    # --- floors (every associated map place of a place must be on its floor)
    floor_ok, floor_bad = [], []
    for pid in visited:
        ns = list(assoc.get(pid, {}))
        if not ns:
            continue
        wrong_f = {nid: nodes[nid].floor for nid in ns if nodes[nid].floor != places[pid].floor}
        if wrong_f:
            floor_bad.append({"place": pid, "truth": places[pid].floor, "map": wrong_f})
        else:
            floor_ok.append(pid)

    # --- stair links: a vertical edge joining (a place of / next to) each end
    adj: dict[str, set[str]] = defaultdict(set)
    for e in bundle.edges:
        adj[e.from_node].add(e.to_node)
        adj[e.to_node].add(e.from_node)
    links = []
    for lo, hi in truth.stair_links:
        near_lo = set(assoc.get(lo, {}))
        near_hi = set(assoc.get(hi, {}))
        near_lo |= {m for n in list(near_lo) for m in adj[n]}
        near_hi |= {m for n in list(near_hi) for m in adj[n]}
        ok = any(e.vertical != "none" and ((e.from_node in near_lo and e.to_node in near_hi)
                                           or (e.to_node in near_lo and e.from_node in near_hi))
                 for e in bundle.edges)
        links.append({"link": [lo, hi], "ok": bool(ok and near_lo and near_hi)})

    # --- routes
    routes = []
    for r in truth.routes:
        src, dst = _route_nodes(r.from_, assoc, truth, bundle), _route_nodes(r.to, assoc, truth, bundle)
        rec = {"from": r.from_, "to": r.to, "expected_floor_changes": r.floor_changes, "ok": False}
        if not src or not dst:
            rec["error"] = "place not on the map: " + ", ".join(p for p, c in ((r.from_, src), (r.to, dst)) if not c)
        elif src[0] == dst[0]:
            rec["error"] = f"both ends are the same map place {src[0]} (wrong merge)"
        else:
            try:
                res = routing.route(bundle, src[0], dst[0])
                floors = [s.floor for s in res.steps]
                changes = sum(1 for a, b in zip(floors, floors[1:], strict=False) if a != b)
                rec.update(nodes=[src[0], dst[0]], steps=len(res.steps), floor_changes=changes)
                rec["ok"] = r.floor_changes is None or changes == r.floor_changes
                if not rec["ok"]:
                    rec["error"] = f"{changes} floor changes, expected {r.floor_changes}"
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                rec.update(nodes=[src[0], dst[0]], error="no path")
        routes.append(rec)

    extra = sorted(nid for nid in nodes if nid not in node_places)
    turns = score_turns(truth, events)
    far = far_merges(truth, obs)
    false_keys = false_key_places(truth, bundle, obs, node_places)
    has_paths = any(v.path for v in truth.videos)

    def frac(a, b):
        return round(a / b, 3) if b else None

    metrics = {
        "rooms_found": frac(len(found), len(visited_rooms)),
        "rooms_located": frac(len(located), len(visited_rooms)),
        "invented_rooms": len(invented),
        "misattributed_rooms": len(misattributed),
        "wrong_merges": len(wrong),
        "merge_recall": frac(len(merged_ok), len(truth.expect_merged)),
        "missed_merges": missed_count,
        "places_detected": frac(sum(1 for p in visited if assoc.get(p)), len(visited)),
        "floor_accuracy": frac(len(floor_ok), len(floor_ok) + len(floor_bad)),
        "stair_links": frac(sum(link["ok"] for link in links), len(links)),
        "route_success": frac(sum(r["ok"] for r in routes), len(routes)),
        "extra_places": len(extra),
        "far_merges": len(far) if has_paths else None,
        "false_key_places": len(false_keys),
        "turns_detected": frac(turns["matched"], turns["total"]),
        "turn_error_deg": turns["mean_error"],
        "spurious_turns": turns["spurious"] if turns["total"] else None,
    }
    counts = {
        "rooms_found": f"{len(found)}/{len(visited_rooms)}",
        "rooms_located": f"{len(located)}/{len(visited_rooms)}",
        "merge_recall": f"{len(merged_ok)}/{len(truth.expect_merged)}",
        "places_detected": f"{sum(1 for p in visited if assoc.get(p))}/{len(visited)}",
        "floor_accuracy": f"{len(floor_ok)}/{len(floor_ok) + len(floor_bad)}",
        "stair_links": f"{sum(link['ok'] for link in links)}/{len(links)}",
        "route_success": f"{sum(r['ok'] for r in routes)}/{len(routes)}",
        "turns_detected": f"{turns['matched']}/{turns['total']}",
    }
    hard = [m for m, (_, gate) in METRICS.items() if gate and metrics.get(m)]
    return {
        "metrics": metrics,
        "counts": counts,
        "hard_fail": hard,
        "map": {"places": len(nodes), "edges": len(bundle.edges), "floors": sorted({n.floor for n in nodes.values()}),
                "suggested_merges": len(bundle.suggested_merges),
                "vertical_edges": sum(e.vertical != "none" for e in bundle.edges)},
        "details": {
            "missing_rooms": missing_rooms,
            "invented": invented,
            "misattributed": misattributed,
            "wrong_merges": wrong,
            "missed_merges": missed,
            "expected_merge_not_found": not_found_merge,
            "floor_errors": floor_bad,
            "stair_links": links,
            "routes": routes,
            "turns": turns["rows"],
            "far_merges": far,
            "false_key_places": false_keys,
            "extra_places": [{"node": nid, "label": nodes[nid].label, "floor": nodes[nid].floor,
                              "video": nodes[nid].video_id, "t": nodes[nid].t} for nid in extra],
            "visits": records,
            "nodes": [{"id": n.id, "label": n.label, "type": n.type, "floor": n.floor, "floor_source": n.floor_source,
                       "rooms": sorted(claims[n.id]), "places": sorted(node_places.get(n.id, ())),
                       "video": n.video_id, "t": n.t} for n in bundle.nodes],
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Re-score saved results with the current scorer and truth.json (e.g. after changing a metric):
        .venv/bin/python -m eval.score eval/results/<stamp>-<scenario>-<mode>.json [--truth path]"""
    import argparse
    import json
    from pathlib import Path

    from .run import compare, print_scorecard, previous_result
    from .truth import load

    p = argparse.ArgumentParser(prog="eval.score")
    p.add_argument("results", nargs="+", type=Path)
    p.add_argument("--truth", type=Path, help="truth.json (default: the scenario's)")
    p.add_argument("--write", action="store_true", help="store the new scores in the result file")
    args = p.parse_args(argv)
    code = 0
    for path in args.results:
        saved = json.loads(path.read_text())
        if "bundle" not in saved:
            print(f"{path}: no saved map (results from before maps were saved); re-run the scenario")
            code = 2
            continue
        truth_path = args.truth or Path(__file__).parent / "scenarios" / saved["scenario"] / "truth.json"
        res = score(load(truth_path), MapBundle.model_validate(saved["bundle"]), saved["observations"],
                    saved.get("events"))
        for k in ("scenario", "mode", "model", "timings"):
            res[k] = saved.get(k)
        prev = previous_result(saved["scenario"], saved["mode"], saved.get("model"), path)
        print_scorecard(res, prev)
        changed = [r["metric"] for r in compare(res, saved) if r["flag"]]
        print(f"(re-scored {path.name}; changed vs saved: {', '.join(changed) or 'nothing'})")
        if args.write:
            saved.update({k: res[k] for k in ("metrics", "counts", "hard_fail", "map", "details")})
            saved["rescored"] = True
            path.write_text(json.dumps(saved, indent=1, default=str) + "\n")
        code = max(code, 1 if res["hard_fail"] else 0)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
