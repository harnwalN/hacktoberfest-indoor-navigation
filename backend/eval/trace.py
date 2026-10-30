"""Provenance for scoring: which map place each moment of each walk ended up in.

The map keeps one video id / time per place, so after merges it no longer says "walk 2 at t=14 s is place
n007". The tracer wraps three pipeline functions for the duration of a run (no app code changes):
  - mapper.walk_nodes: remembers each walk's WalkNode list (integrate_walk then sets wn.node_id),
  - mapper.merge_nodes: remembers remove -> keep, so ids can be followed to the surviving place,
  - jobs.select_keyframes: remembers frame id -> (video id, t), for the oracle.
"""
from __future__ import annotations

from contextlib import contextmanager

from app import jobs, mapper


class Tracer:
    def __init__(self):
        self.walks: dict[str, list] = {}  # app video id -> [WalkNode] (last integration wins)
        self.redirect: dict[str, str] = {}  # removed node id -> kept node id
        self.frames: dict[str, tuple[str, float]] = {}  # frame id -> (app video id, t)

    @contextmanager
    def installed(self):
        orig_wn, orig_merge, orig_sel = mapper.walk_nodes, mapper.merge_nodes, jobs.select_keyframes

        def walk_nodes(keyframes):
            res = orig_wn(keyframes)
            if keyframes:
                self.walks[keyframes[0].video_id] = res
            return res

        def merge_nodes(bundle, keep_id, remove_id, log=print):
            orig_merge(bundle, keep_id, remove_id, log=log)  # raises before changing anything: not recorded
            self.redirect[remove_id] = keep_id

        def select_keyframes(*a, **k):
            kfs = orig_sel(*a, **k)
            for kf in kfs:
                self.frames[kf.frame_id] = (kf.video_id, kf.t)
            return kfs

        mapper.walk_nodes, mapper.merge_nodes, jobs.select_keyframes = walk_nodes, merge_nodes, select_keyframes
        try:
            yield self
        finally:
            mapper.walk_nodes, mapper.merge_nodes, jobs.select_keyframes = orig_wn, orig_merge, orig_sel

    def resolve(self, node_id: str | None, alive: set[str]) -> str | None:
        seen = set()
        while node_id in self.redirect and node_id not in seen:
            seen.add(node_id)
            node_id = self.redirect[node_id]
        return node_id if node_id in alive else None

    def observations(self, bundle, video_names: dict[str, str]) -> dict[str, list[dict]]:
        """{truth video id: [{t, t_arrive, t_depart, node, frame_id, kind}]} for every walk node."""
        alive = {n.id for n in bundle.nodes}
        out: dict[str, list[dict]] = {}
        for vid, wnodes in self.walks.items():
            name = video_names.get(vid)
            if name is None:
                continue
            out[name] = [{"t": wn.kf.t, "t_arrive": wn.t_arrive, "t_depart": wn.t_depart,
                          "node": self.resolve(wn.node_id, alive), "frame_id": wn.kf.frame_id, "kind": wn.kf.kind}
                         for wn in wnodes]
        return out
