"""End-to-end pipeline on a synthetic loop walk, with Gemma disabled (motion + geometry only).

The walker turns right four times and ends up facing the same view as the start, so the last
walk node must be merged into the first one (loop closure) and the graph must contain a cycle.
"""
import networkx as nx
import pytest

from app import config, gemma, jobs, routing, store
from tests.synthetic import write_video


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BUILDINGS_DIR", tmp_path / "buildings")

    def no_gemma(*_a, **_k):
        raise gemma.GemmaUnavailable("disabled in tests")

    monkeypatch.setattr(gemma, "resolve_model", no_gemma)
    return tmp_path


def test_loop_walk_closes(isolated):
    src = isolated / "loop.mp4"
    walk = [(3, 0)]
    for _ in range(4):
        walk += [(3, 30), (4, 0)]
    write_video(src, walk)

    bundle = store.create_building("Synthetic Hall")
    job = jobs.Job(id="t", building_id=bundle.building_id, video_id=None, kind="video")
    jobs.JOBS[job.id] = job
    # run synchronously instead of on the worker thread
    upload = isolated / "upload.mp4"
    src.rename(upload)
    vid = "vtest"
    job.video_id = vid
    vdir = store.video_dir(bundle.building_id, vid)
    vdir.mkdir(parents=True)
    (vdir / "source.mp4").write_bytes(upload.read_bytes())
    from app.schemas import VideoInfo
    bundle.videos.append(VideoInfo(id=vid, filename="loop.mp4"))
    store.save_map(bundle)
    jobs._process_video(job, vdir / "source.mp4", None, None)
    print("\n".join(job.log))

    m = store.load_map(bundle.building_id)
    turns = [line for line in job.log if "turn@" in line]
    assert turns and turns[0].count("turn@") == 4
    assert any("Loop closure" in line for line in job.log), job.log
    g = routing.build_graph(m)
    assert nx.cycle_basis(g), "loop closure should create a cycle"

    # Headings snapped onto a 90-degree grid; every junction is a right turn walking forward.
    for e in m.edges:
        for h in (e.heading_out, e.heading_in):
            assert round(h - m.dominant_axis_deg) % 90 == 0, (e, m.dominant_axis_deg)
    start = m.nodes[0].id
    farthest = max(m.nodes, key=lambda n: nx.shortest_path_length(g, start, n.id)).id
    there = routing.route(m, start, farthest)
    back = routing.route(m, farthest, start)
    assert len(there.steps) >= 2 and len(back.steps) >= 2
