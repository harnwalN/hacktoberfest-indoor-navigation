from app import motion, video
from tests.synthetic import write_video


def _track(tmp_path, segments):
    src = tmp_path / "src.mp4"
    write_video(src, segments)
    analysis, _, _ = video.normalize(src, tmp_path / "norm")
    track = motion.compute_track(analysis)
    return track, motion.detect_events(track)


def test_right_then_left_turn(tmp_path):
    track, events = _track(tmp_path, [(2, 0), (3, 30), (6, 0), (3, -30), (2, 0)])
    turns = [e for e in events if e["type"] == "turn"]
    assert len(turns) == 2, events
    assert 70 <= turns[0]["delta"] <= 110
    assert -110 <= turns[1]["delta"] <= -70
    assert abs(track["heading"][-1]) < 15
    assert any(e["type"] == "dwell" for e in events)


def test_glance_is_not_a_turn(tmp_path):
    _, events = _track(tmp_path, [(2, 0), (1.5, 40), (1.5, 0), (1.5, -40), (2, 0)])
    assert [e["type"] for e in events if e["type"] in ("turn", "glance")] == ["glance"], events
