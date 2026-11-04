"""Restore test for scripts/migrate_data.py: round trip, traversal, non-empty target, corruption."""
import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "migrate_data.py"
spec = importlib.util.spec_from_file_location("migrate_data", SCRIPT)
md = importlib.util.module_from_spec(spec)
spec.loader.exec_module(md)


def _tree(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*")) if p.is_file()
    }


@pytest.fixture
def data_dir(tmp_path) -> Path:
    d = tmp_path / "src"
    (d / "buildings" / "engineering-hall" / "frames").mkdir(parents=True)
    (d / "buildings" / "engineering-hall" / "map.json").write_text('{"name": "Engineering Hall"}')
    (d / "buildings" / "engineering-hall" / "frames" / "f1.jpg").write_bytes(b"\xff\xd8" + bytes(range(256)))
    (d / "buildings" / "test-two-floor").mkdir(parents=True)
    (d / "buildings" / "test-two-floor" / "reports.jsonl").write_text('{"a": 1}\n{"a": 2}\n')
    (d / "auth.json").write_text("{}")
    (d / "hosted_usage.json").write_text('{"day": "2026-11-01", "calls": 3}')
    return d


def _make_archive(path: Path, members: list[tuple[str, bytes]]) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        for name, body in members:
            info = tarfile.TarInfo(name)
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
    return path


def test_round_trip_matches_every_hash(tmp_path, data_dir):
    archive = tmp_path / "out" / "data.tar.gz"
    assert md.export_data(data_dir, archive) == 5
    target = tmp_path / "restored"
    assert md.import_data(archive, target, force=False) == 5
    assert _tree(target) == _tree(data_dir)


def test_manifest_records_hashes_and_schema(tmp_path, data_dir):
    archive = tmp_path / "data.tar.gz"
    md.export_data(data_dir, archive)
    with tarfile.open(archive, "r:gz") as tar:
        manifest = json.loads(tar.extractfile(md.MANIFEST).read())
    assert manifest["format"] == md.FORMAT
    assert manifest["app_schema"] is None
    assert manifest["files"]["auth.json"]["sha256"] == _tree(data_dir)["auth.json"]


def test_traversal_archive_rejected(tmp_path):
    target = tmp_path / "restored"
    for bad in ("data/../../evil.txt", "/etc/evil.txt", "data/../evil.txt"):
        archive = _make_archive(tmp_path / "bad.tar.gz", [(bad, b"x")])
        with pytest.raises(md.MigrationError, match="unsafe archive member"):
            md.import_data(archive, target, force=False)
    assert not target.exists()
    assert not (tmp_path / "evil.txt").exists()


def test_non_empty_target_refused_without_force(tmp_path, data_dir):
    archive = tmp_path / "data.tar.gz"
    md.export_data(data_dir, archive)
    target = tmp_path / "live"
    target.mkdir()
    (target / "keep.txt").write_text("precious")
    with pytest.raises(md.MigrationError, match="target not empty"):
        md.import_data(archive, target, force=False)
    assert (target / "keep.txt").read_text() == "precious"


def test_force_replaces_non_empty_target(tmp_path, data_dir):
    archive = tmp_path / "data.tar.gz"
    md.export_data(data_dir, archive)
    target = tmp_path / "live"
    target.mkdir()
    (target / "stale.txt").write_text("old")
    md.import_data(archive, target, force=True)
    assert not (target / "stale.txt").exists()
    assert _tree(target) == _tree(data_dir)


def test_corrupted_file_detected_by_manifest(tmp_path, data_dir):
    archive = tmp_path / "data.tar.gz"
    md.export_data(data_dir, archive)
    # Rebuild the archive with one data file altered but the original manifest kept.
    with tarfile.open(archive, "r:gz") as tar:
        members = [(m.name, tar.extractfile(m).read() if m.isreg() else b"") for m in tar.getmembers() if m.isreg()]
    members = [(n, b"CORRUPT" if n == "data/auth.json" else b) for n, b in members]
    bad = _make_archive(tmp_path / "corrupt.tar.gz", members)
    target = tmp_path / "restored"
    with pytest.raises(md.MigrationError, match="checksum mismatch"):
        md.import_data(bad, target, force=False)
    assert not target.exists()


def test_missing_manifest_file_rejected(tmp_path, data_dir):
    archive = tmp_path / "data.tar.gz"
    md.export_data(data_dir, archive)
    with tarfile.open(archive, "r:gz") as tar:
        members = [(m.name, tar.extractfile(m).read()) for m in tar.getmembers()
                   if m.isreg() and m.name != "data/auth.json"]
    bad = _make_archive(tmp_path / "short.tar.gz", members)
    with pytest.raises(md.MigrationError, match="manifest mismatch"):
        md.import_data(bad, tmp_path / "restored", force=False)
