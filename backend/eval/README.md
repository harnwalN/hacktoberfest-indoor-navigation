# Evaluation harness

Scores the whole pipeline (motion, keyframes, Gemma + OCR, mapping, floors, routing) on walkthrough videos
with known ground truth, without filming anything. App code stays untouched: the harness drives `app.jobs`
exactly like an upload, in a temporary data dir (never `backend/data`).

```
eval/
  scenarios/<name>/spec.py      synthetic building + walk scripts (tracked)
  scenarios/<name>/truth.json   ground truth (tracked; written by the generator, or by hand for real data)
  scenarios/<name>/media/       rendered MP4s + per-frame sidecars (gitignored)
  results/                      scorecards <stamp>-<scenario>-<mode>.json + .log (gitignored)
  .cache/                       Gemma answers from the last live run, for --reuse-cache (gitignored)
  render.py building.py walk.py generate.py   synthetic renderer
  truth.py score.py trace.py oracle.py run.py   ground truth, scorer, runner
```

## Run it

From `backend/` (Ollama running with the model pulled for live runs):

```bash
.venv/bin/python -m eval.run all                       # live, GEMMA_MODEL (gemma3:4b); ~20 min for all (M3 Pro)
.venv/bin/python -m eval.run two_floor --model gemma3:12b
.venv/bin/python -m eval.run all --oracle              # perfect perception from ground truth, ~1 min, no Ollama
.venv/bin/python -m eval.run all --no-gemma            # motion only (no Gemma, no OCR)
.venv/bin/python -m eval.run all --reuse-cache         # reuse the last live run's Gemma answers (fast re-score
                                                       # after a mapping change; frames must be identical)
.venv/bin/python -m eval.run path/to/truth.json        # any scenario directory, e.g. real data
.venv/bin/python -m eval.generate all [--force]        # (re)render synthetic scenarios only
.venv/bin/python -m eval.score eval/results/<file>.json # re-score a saved run with the current scorer/truth
```

Flags: `--data-dir DIR` (ingest there and keep it), `--keep` (keep the temp dir), `--no-generate`, `--no-save`,
`-v` (print every pipeline log line).

Output: a scorecard per scenario (the saved JSON also holds the final map, per-walk observations and motion
events, so `eval.score` can re-score it later), compared with the previous result of the same scenario, mode and model
(`improved` / `REGRESSED`), plus a summary table. Exit code **0** ok, **1** a hard safety metric failed, **2**
setup or pipeline error, so `python -m eval.run all --oracle` (fast) or a live run can gate a push.

**Two baselines.** `--oracle` answers every Gemma question (frame analysis, same place, same stairwell) from
the ground truth, so it scores the map logic alone. A live run that is worse than the oracle run points at
perception (Gemma/OCR); an error that is already in the oracle run is in mapping/floors/routing.

## Metrics

| Metric | Meaning |
|---|---|
| `rooms_found` | visited rooms whose number appears on some map place (label or its own photo's plaques) |
| `rooms_located` | ... on a map place that is actually where that room is (time-matched) |
| `invented_rooms` **[gate]** | room numbers on map places that exist nowhere in the building (e.g. from a poster) |
| `misattributed_rooms` | a real room number on a map place that is somewhere else |
| `wrong_merges` **[gate]** | `must_not_merge` pairs that ended up in one map place |
| `merge_recall`, `missed_merges` | `expect_merged` places that are one map place / extra copies of them |
| `places_detected` | visited places that got any map place at the visit time |
| `floor_accuracy` | visited places whose map place(s) are on the right floor |
| `stair_links` | `stair_links` pairs joined by a stairs/elevator edge (at the places or a neighbour) |
| `route_success` | `routes` that the app can route, with the expected number of floor changes |
| `far_merges` | map places whose observations (any walk node, incl. starts and corridor spots) were > 5 m apart or on two floors, from the truth `path`s - not gated yet |
| `extra_places` | map places matching no visit (walk start/end, hallway points, duplicates) - informational |
| `turns_detected`, `turn_error_deg`, `spurious_turns` | detected turn events vs true changes of walking direction |

How places are matched: each truth visit `(video, t)` is matched to the walk node of that video whose
`[t_arrive, t_depart]` window is nearest (within 1.5 s for a `look`, 1.0 s for a `pass`). A `pass` by a room
(no glance) only counts if that map place claims the room's number; otherwise it is a corridor spot that
happened to see the plaque. `eval/trace.py`
follows each walk node to the map place it ended up in, through every later merge (it wraps
`mapper.walk_nodes`, `mapper.merge_nodes` and `jobs.select_keyframes` for the duration of a run; no app change).

## truth.json (schema `waypoint-eval/1`)

```jsonc
{
  "schema": "waypoint-eval/1",
  "scenario": "two_floor",
  "description": "...",
  "source": {"kind": "synthetic", ...},          // free-form: "habitat", "lamar", "phone", ...
  "building": {
    "floors": [1, 2],
    "places": [                                  // required
      {"id": "r110", "label": "Room 110", "room": "110", "floor": 1, "type": "room", "x": 2.65, "y": 1.0},
      {"id": "stairs_1", "label": "Stairs", "room": null, "floor": 1, "type": "stairs", "x": 14, "y": 2.5,
       "group": "A"}                             // same group on two floors = same stairwell
    ],                                           // type: room stairs elevator exit entrance hallway
                                                 //       intersection restroom landmark
    "distractors": [{"id": "p_tutor", "text": "MATH TUTORING CALL 555-0199", "kind": "poster", "floor": 2}]
  },
  "videos": [{
    "id": "up", "file": "media/up.mp4",          // relative to the scenario directory
    "duration": 57.1,
    "start_floor": 1,
    "give_start_floor": true,                    // pass it to the pipeline like an uploader would
    "sensors": null,                             // optional phone sensors.json (gyro), relative path
    "visits": [{"place": "r110", "t": 2.5, "kind": "look"},   // ordered places passed, seconds into the video
               {"place": "stairs_1", "t": 20.5, "kind": "pass"}],
    "turns": [{"t": 18.8, "delta": -90.0}],      // optional: changes of walking direction (right > 0)
    "path": [[0.0, 1.2, 1.0, 1], [0.5, 1.8, 1.0, 1]]   // optional: [t, x, y, floor|null] every ~0.5 s
  }],
  "expect_merged": ["r110", "stairs_1"],         // visited more than once: must end as ONE map place
  "must_not_merge": [["r214", "r216", "identical doors"]],   // never one map place
  "stair_links": [["stairs_1", "stairs_2"]],     // [lower, upper] must be connected vertically
  "routes": [{"from": "r110", "to": "r212", "floor_changes": 1}]
}
```

The generator fills `expect_merged` (rooms/stairs/exits visited 2+ times) and `must_not_merge` (all pairs of
different room numbers, the same stairwell on different floors, plus the spec's own pairs) automatically.

## Add a synthetic scenario

1. Copy `scenarios/small_loop/spec.py` to `scenarios/<name>/spec.py` and edit `SPEC`:
   - `levels`: `{floor: {"z": metres, "grid": [rows], "stripe": BGR, "period": metres or None}}`. One
     character per metre, north up: `#` solid, `.` corridor, `o` corridor cell open to a stair shaft, `S` stair
     shaft (same cells on every level). `period` repeats wall/floor details so two spots look identical.
   - `decals`: `door` (with `plaque`, `room`, `place`, optional `sign` above it, e.g. EXIT), `sign` (`style`:
     floor, exit, stairs, directional, name; optional `arrow`), `poster` (`lines`; same `look_key` = same
     picture). `wall: [x, y, facing]` is a point on a wall line and the direction the wall faces (into the
     corridor). A door's place stands 1 m in front of it, between door and plaque.
   - `stairs`: straight flights inside the shaft (`dir`, `across`, `start`, `flights`, `rise`, `run`, `landing`,
     `top_landing_end`, `bottom`/`top` walk endpoints). `places`: extra places (stairs, junctions).
   - `videos`: start floor/position/heading and a script of steps: `{"go": place|[x, y]}`, `{"look": place}`,
     `{"visit": place}` (go + look), `{"pause": s}`, `{"turn": deg}`, `{"face": heading}`,
     `{"stairs": id, "dir": "up"|"down"}`. Keep each video 20-60 s.
   - `routes`, `stair_links`, extra `must_not_merge`.
2. `.venv/bin/python -m eval.generate <name>` and look at a few frames (`media/<video>.mp4`).
3. `.venv/bin/python -m eval.run <name> --oracle`, then a live run. Commit `spec.py` and `truth.json`.

Rendering: a small numpy/OpenCV textured-quad renderer (`render.py`, ~0.2 s per 1280x720 frame, 10 fps,
videos rendered in parallel). Same 69 deg FOV as the pipeline assumes. Video is H.264 via ffmpeg.

## Real or Habitat/HM3D data

Nothing in the scorer depends on the synthetic generator. For any other source, make a directory with the
videos and a hand- or tool-written `truth.json`, then run `python -m eval.run path/to/truth.json`
(`--oracle` is not available: it needs the generator's per-frame sidecars).

- **Habitat-sim / HM3D**: render an agent walking a navmesh path at ~1.6 m eye height, 69 deg HFOV, 10+ fps
  (Linux/cloud; habitat-sim is unreliable on Apple Silicon). Export the agent pose per frame; place truth
  `places` from HM3D semantic annotations (doors/rooms/stairs) with their floor; `visits` are the times the
  path comes within ~1.25 m of a place's standing point; `turns` from changes of the path heading; `path`
  from the poses directly. HM3D has
  no readable room plaques, so either paste plaque textures onto doors before rendering, or judge only
  floors, merges, stair links and routes there (rooms metrics will be `-`).
- **LaMAR / ScanNet++ / phone walks**: per-frame poses give `visits`/`turns` the same way; plaques and floors
  are annotated by hand once per building. Phone gyro logs go in `sensors`.

A converter only has to write the JSON above; `eval/truth.py` validates it (unknown place ids are rejected).

## Tests

`tests/test_eval_harness.py` (main suite, ~10 s, no Ollama): scorer rules on hand-made maps, plus a tiny
2-room scenario rendered at 480x270 and run end-to-end with the oracle. The live run is opt-in:
`EVAL_LIVE=1 .venv/bin/python -m pytest tests/test_eval_harness.py -k live`.
