"""Crowd-sourcing groundwork in store.py: contributor accounts, building campus, walk moderation fields.

Storage is JSON only (no routes here). conftest moves config.DATA_DIR to a temp dir; BUILDINGS_DIR is
computed at import time, so it is moved here too (as the other tests do).
"""
import json

import pytest
from pydantic import ValidationError

from app import config, jobs, store
from app.schemas import MapBundle, VideoInfo


@pytest.fixture(autouse=True)
def _temp_buildings(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BUILDINGS_DIR", tmp_path / "buildings")


# ---------- accounts ----------
def test_create_account_normalizes_email_and_is_idempotent():
    a = store.create_account_if_missing("  Alice@Example.COM ")
    b = store.create_account_if_missing("alice@example.com")
    assert a.email == "alice@example.com"
    assert a.id == b.id
    assert a.moderator_of == []
    assert a.created_at.endswith("+00:00")
    stored = json.loads(store.accounts_path().read_text())
    assert [r["email"] for r in stored["accounts"]] == ["alice@example.com"]


def test_create_account_gives_distinct_ids_per_email():
    a = store.create_account_if_missing("a@x.org")
    b = store.create_account_if_missing("b@x.org")
    assert a.id != b.id


@pytest.mark.parametrize("bad", ["", "   ", "no-at-sign", "a b@c.de", "a@b", "@x.org", "a@x.org@y.org", None, 42])
def test_create_account_rejects_bad_email_without_echoing_it(bad):
    with pytest.raises(ValueError) as exc:
        store.create_account_if_missing(bad)
    if isinstance(bad, str) and bad.strip():
        assert bad.strip() not in str(exc.value)
    assert not store.accounts_path().exists()  # nothing written for a rejected address


def test_get_account_by_email_any_case_and_by_id():
    acct = store.create_account_if_missing("Bob@Example.com")
    assert store.get_account("BOB@example.com").id == acct.id
    assert store.get_account(acct.id).email == "bob@example.com"


def test_get_account_unknown_or_malformed_returns_none():
    store.create_account_if_missing("bob@example.com")
    assert store.get_account("nobody@example.com") is None
    assert store.get_account("deadbeef") is None
    assert store.get_account("not an email@") is None
    assert store.get_account("") is None
    assert store.get_account(None) is None


def test_no_accounts_file_means_no_accounts():
    assert store.get_account("anyone@example.com") is None
    assert store.is_moderator("deadbeef", "some-building") is False


# ---------- moderators ----------
def test_building_creator_becomes_moderator():
    creator = store.create_account_if_missing("creator@example.com")
    other = store.create_account_if_missing("other@example.com")
    bundle = store.create_building("Price Center", creator_id=creator.id)
    assert store.is_moderator(creator.id, bundle.building_id)
    assert not store.is_moderator(other.id, bundle.building_id)
    assert store.get_account(creator.id).moderator_of == [bundle.building_id]


def test_building_without_creator_has_no_moderators():
    creator = store.create_account_if_missing("c@example.com")
    bundle = store.create_building("Solo Hall")
    assert not store.is_moderator(creator.id, bundle.building_id)


def test_create_building_with_unknown_creator_writes_nothing():
    with pytest.raises(KeyError):
        store.create_building("Ghost Hall", creator_id="0" * 32)
    assert not (config.BUILDINGS_DIR / "ghost-hall").exists()


def test_add_moderator_is_idempotent_and_validates_input():
    acct = store.create_account_if_missing("m@example.com")
    bundle = store.create_building("Library")
    store.add_moderator(acct.id, bundle.building_id)
    store.add_moderator(acct.id, bundle.building_id)
    assert store.get_account(acct.id).moderator_of == [bundle.building_id]
    with pytest.raises(KeyError):
        store.add_moderator("unknown", bundle.building_id)
    with pytest.raises(ValueError):
        store.add_moderator(acct.id, "../escape")


# ---------- campus ----------
def test_new_building_defaults_to_ucsd_campus():
    bundle = store.create_building("Geisel Library")
    assert bundle.campus == "ucsd"
    assert store.load_map(bundle.building_id).campus == "ucsd"
    summary = {b["id"]: b for b in store.list_buildings()}
    assert summary[bundle.building_id]["campus"] == "ucsd"


def test_campus_must_be_non_empty():
    with pytest.raises(ValidationError):
        MapBundle(building_id="x", campus="")
    assert MapBundle(building_id="x", campus="other").campus == "other"


def test_legacy_map_json_loads_with_defaults():
    """A map.json written before these fields existed loads with the defaults (no migration step needed)."""
    store.create_building("Old Hall")
    path = config.BUILDINGS_DIR / "old-hall" / "map.json"
    data = json.loads(path.read_text())
    data.pop("campus")
    data["videos"] = [{"id": "v1", "filename": "walk.mp4", "status": "done"}]
    path.write_text(json.dumps(data))
    bundle = store.load_map("old-hall")
    assert bundle.campus == "ucsd"
    assert bundle.videos[0].contributor_id is None
    assert bundle.videos[0].moderation_status == "pending"
    assert bundle.videos[0].moderation_notes is None


# ---------- walk contributor and moderation ----------
def _bundle_with_walk(contributor_id=None):
    bundle = store.create_building("Ridge Walk")
    bundle.videos.append(VideoInfo(id="v1", filename="walk.mp4", contributor_id=contributor_id))
    store.save_map(bundle)
    return bundle.building_id


def test_video_defaults_pending_with_no_notes():
    bid = _bundle_with_walk()
    v = store.get_video(bid, "v1")
    assert (v.contributor_id, v.moderation_status, v.moderation_notes) == (None, "pending", None)


def test_set_video_moderation_status_and_notes():
    bid = _bundle_with_walk(contributor_id="abc")
    store.set_video_moderation(bid, "v1", "rejected", notes="  too blurry  ")
    v = store.get_video(bid, "v1")
    assert v.moderation_status == "rejected"
    assert v.moderation_notes == "too blurry"
    assert v.contributor_id == "abc"  # contributor is never touched by moderation
    store.set_video_moderation(bid, "v1", "approved")  # notes left as they were
    v = store.get_video(bid, "v1")
    assert (v.moderation_status, v.moderation_notes) == ("approved", "too blurry")
    store.set_video_moderation(bid, "v1", "approved", notes="")  # empty clears the notes
    assert store.get_video(bid, "v1").moderation_notes is None


def test_set_video_moderation_rejects_bad_input_and_changes_nothing():
    bid = _bundle_with_walk()
    store.set_video_moderation(bid, "v1", "approved", notes="ok")
    with pytest.raises(ValueError):
        store.set_video_moderation(bid, "v1", "maybe")
    with pytest.raises(ValueError):
        store.set_video_moderation(bid, "v1", "rejected", notes="x" * (store.MAX_MODERATION_NOTES + 1))
    with pytest.raises(ValueError):
        store.set_video_moderation(bid, "v1", "rejected", notes=["not", "text"])
    with pytest.raises(KeyError):
        store.set_video_moderation(bid, "no-such-walk", "approved")
    v = store.get_video(bid, "v1")
    assert (v.moderation_status, v.moderation_notes) == ("approved", "ok")


def test_get_video_unknown_id_is_none_and_unknown_building_raises():
    bid = _bundle_with_walk()
    assert store.get_video(bid, "missing") is None
    with pytest.raises(FileNotFoundError):
        store.get_video("no-such-building", "v1")


def test_submit_video_records_contributor(tmp_path, monkeypatch):
    """jobs.submit_video stores contributor_id on the new walk; the job itself is not run here."""
    monkeypatch.setattr(jobs, "_enqueue", lambda job, fn, front=False: job)
    bid = store.create_building("Upload Hall").building_id
    src = tmp_path / "upload.mp4"
    src.write_bytes(b"not really a video")
    acct = store.create_account_if_missing("uploader@example.com")
    job = jobs.submit_video(bid, src, "upload.mp4", None, None, contributor_id=acct.id)
    v = store.get_video(bid, job.video_id)
    assert v.contributor_id == acct.id
    assert v.moderation_status == "pending"
