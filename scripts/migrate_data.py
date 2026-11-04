#!/usr/bin/env python3
"""Export or import the app's DATA_DIR as a tar.gz archive with a sha256 manifest. Stdlib only.

  python scripts/migrate_data.py export <data_dir> <archive.tar.gz>
  python scripts/migrate_data.py import <archive.tar.gz> <data_dir> [--force]

Archive layout: MANIFEST.json at the root, every data file under data/<relative path>.
The app has no schema version yet, so the manifest records "app_schema": null.
Import refuses a non-empty target unless --force, which replaces the target's contents.
"""
import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

MANIFEST = "MANIFEST.json"
PREFIX = "data/"
FORMAT = "hacktoberfest-data-archive/1"
CHUNK = 1 << 20


class MigrationError(Exception):
    pass


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def export_data(data_dir: Path, archive: Path) -> int:
    if not data_dir.is_dir():
        raise MigrationError(f"not a directory: {data_dir}")
    files, dirs = {}, []
    for root, subdirs, names in os.walk(data_dir):
        for name in subdirs + names:
            p = Path(root) / name
            if p.is_symlink():
                raise MigrationError(f"refusing symlink: {p}")
        for name in names:
            p = Path(root) / name
            if not p.is_file():
                raise MigrationError(f"refusing special file: {p}")
            rel = p.relative_to(data_dir).as_posix()
            files[rel] = {"sha256": _sha256_file(p), "size": p.stat().st_size}
        dirs += [(Path(root) / d).relative_to(data_dir).as_posix() for d in subdirs]
    manifest = {
        "format": FORMAT,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": str(data_dir.resolve()),
        "app_schema": None,
        "file_count": len(files),
        "files": dict(sorted(files.items())),
    }
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as tar:
        for d in [""] + sorted(dirs):
            tar.add(data_dir / d, arcname=(PREFIX + d).rstrip("/"), recursive=False)
        for rel in files:
            tar.add(data_dir / rel, arcname=PREFIX + rel, recursive=False)
        body = json.dumps(manifest, indent=2, sort_keys=True).encode()
        info = tarfile.TarInfo(MANIFEST)
        info.size = len(body)
        info.mtime = int(time.time())
        tar.addfile(info, io.BytesIO(body))
    return len(files)


def _safe_member_name(name: str) -> PurePosixPath:
    p = PurePosixPath(name)
    if (not name or "\\" in name or p.is_absolute() or ".." in p.parts
            or (len(name) > 1 and name[1] == ":")):
        raise MigrationError(f"unsafe archive member (absolute or '..'): {name!r}")
    return p


def _extract_verified(archive: Path, dest: Path) -> int:
    """Write every data file into dest only if its sha256 and size match the manifest."""
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        seen = set()
        for m in members:
            _safe_member_name(m.name)
            if m.isdir():
                if m.name.rstrip("/") != "data" and not m.name.startswith(PREFIX):
                    raise MigrationError(f"directory outside data/: {m.name!r}")
                continue
            if not m.isreg():
                raise MigrationError(f"unsupported member type (link or device): {m.name!r}")
            if m.name != MANIFEST and not m.name.startswith(PREFIX):
                raise MigrationError(f"member outside data/ and manifest: {m.name!r}")
            if m.name in seen:
                raise MigrationError(f"duplicate member: {m.name!r}")
            seen.add(m.name)
        if MANIFEST not in seen:
            raise MigrationError("archive has no MANIFEST.json")
        manifest = json.loads(tar.extractfile(MANIFEST).read())
        if manifest.get("format") != FORMAT:
            raise MigrationError(f"unknown archive format: {manifest.get('format')!r}")
        expected = manifest["files"]
        found = {m.name[len(PREFIX):] for m in members if m.isreg() and m.name != MANIFEST}
        if found != set(expected):
            missing, extra = set(expected) - found, found - set(expected)
            raise MigrationError(f"manifest mismatch: missing={sorted(missing)} extra={sorted(extra)}")

        for m in members:
            if m.isdir():
                (dest / m.name.rstrip("/")[len(PREFIX):]).mkdir(parents=True, exist_ok=True)
        dest_root = dest.resolve()
        for m in members:
            if not m.isreg() or m.name == MANIFEST:
                continue
            rel = m.name[len(PREFIX):]
            out = (dest / rel).resolve()
            if dest_root not in out.parents:
                raise MigrationError(f"path escapes target: {rel!r}")
            out.parent.mkdir(parents=True, exist_ok=True)
            h, size = hashlib.sha256(), 0
            with tar.extractfile(m) as src, open(out, "wb") as dst:
                for chunk in iter(lambda: src.read(CHUNK), b""):
                    h.update(chunk)
                    size += len(chunk)
                    dst.write(chunk)
            if h.hexdigest() != expected[rel]["sha256"] or size != expected[rel]["size"]:
                raise MigrationError(f"checksum mismatch (corrupt archive?): {rel}")
        return len(expected)


def import_data(archive: Path, data_dir: Path, force: bool) -> int:
    if data_dir.exists() and any(data_dir.iterdir()) and not force:
        raise MigrationError(f"target not empty: {data_dir} (pass --force to replace its contents)")
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".migrate-", dir=data_dir.parent))
    try:
        count = _extract_verified(archive, staging)
        if data_dir.exists():
            for child in data_dir.iterdir():
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        data_dir.mkdir(parents=True, exist_ok=True)
        for child in staging.iterdir():
            shutil.move(str(child), str(data_dir / child.name))
        return count
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="archive a data dir")
    e.add_argument("data_dir", type=Path)
    e.add_argument("archive", type=Path)
    i = sub.add_parser("import", help="restore an archive into a data dir")
    i.add_argument("archive", type=Path)
    i.add_argument("data_dir", type=Path)
    i.add_argument("--force", action="store_true", help="replace the contents of a non-empty target")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "export":
            n = export_data(args.data_dir, args.archive)
            print(f"exported {n} files to {args.archive}")
        else:
            n = import_data(args.archive, args.data_dir, args.force)
            print(f"imported and verified {n} files into {args.data_dir}")
    except (MigrationError, tarfile.TarError, json.JSONDecodeError, KeyError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
