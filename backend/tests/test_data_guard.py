"""The test suite must never write to backend/data (the live data folder). These tests check the guard itself
(conftest.py): where the settings point during a test, and that the run-end comparison fails a run that changed the
folder. The run-end check is exercised with a temporary folder standing in for backend/data."""
from types import SimpleNamespace

from app import config, store
from tests import conftest as guard

REAL = guard.REAL_DATA


def test_settings_point_at_a_temporary_folder_not_the_live_data():
    for path in (config.DATA_DIR, config.BUILDINGS_DIR, store.building_dir("probe-hall")):
        assert not str(path.resolve()).startswith(str(REAL.resolve())), path


def test_the_junk_building_names_are_written_to_the_temporary_folder():
    """Regression: test_campus_field.py created buildings named like these in backend/data (see its own comments)."""
    for name in ("Campus Hall", "Listed Hall", "Legacy Campus Hall", "Engineering Hall"):
        store.create_building(name)
    assert not (REAL / "buildings" / "campus-hall").exists()
    assert not (REAL / "buildings" / "listed-hall").exists()
    assert store.list_buildings() and config.BUILDINGS_DIR.exists()


def test_snapshot_notices_created_changed_and_removed_entries(tmp_path):
    root = tmp_path / "live"
    (root / "buildings" / "a").mkdir(parents=True)
    (root / "buildings" / "a" / "map.json").write_text("{}")
    before = guard._snapshot(root)
    assert guard._snapshot(root) == before  # nothing touched: equal
    (root / "buildings" / "b").mkdir()  # created
    (root / "buildings" / "a" / "map.json").write_text('{"x": 1}')  # changed
    assert guard._snapshot(root) != before
    after_change = guard._snapshot(root)
    (root / "buildings" / "b").rmdir()  # removed
    assert guard._snapshot(root) != after_change


def test_run_end_check_fails_the_run_when_the_live_folder_changed(tmp_path, monkeypatch):
    live = tmp_path / "live"
    live.mkdir()
    monkeypatch.setattr(guard, "REAL_DATA", live)
    monkeypatch.setattr(guard, "SCRATCH", tmp_path / "scratch-gone")
    monkeypatch.setattr(guard, "_BEFORE", guard._snapshot(live))
    session = SimpleNamespace(exitstatus=0)
    (live / "junk.json").write_text("{}")  # a test wrote here
    guard.pytest_sessionfinish(session, 0)
    assert session.exitstatus == 1


def test_run_end_check_leaves_an_untouched_run_alone(tmp_path, monkeypatch):
    live = tmp_path / "live"
    live.mkdir()
    (live / "map.json").write_text("{}")
    monkeypatch.setattr(guard, "REAL_DATA", live)
    monkeypatch.setattr(guard, "SCRATCH", tmp_path / "scratch-gone")
    monkeypatch.setattr(guard, "_BEFORE", guard._snapshot(live))
    session = SimpleNamespace(exitstatus=0)
    guard.pytest_sessionfinish(session, 0)
    assert session.exitstatus == 0
