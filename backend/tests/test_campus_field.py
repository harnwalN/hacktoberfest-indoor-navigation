"""Campus field (app/schemas.py MapBundle.campus, app/store.py): new buildings default to "ucsd", the field is
exposed in list_buildings, and a map.json saved before this field existed still loads with the default (not an
error, not blank)."""
import json

from app import store
from app.schemas import DEFAULT_CAMPUS


def test_default_campus_is_ucsd():
    assert DEFAULT_CAMPUS == "ucsd"


def test_new_building_gets_default_campus():
    b = store.create_building("Campus Hall")
    assert b.campus == DEFAULT_CAMPUS
    assert store.load_map(b.building_id).campus == DEFAULT_CAMPUS


def test_list_buildings_reports_campus():
    bid = store.create_building("Listed Hall").building_id
    rows = store.list_buildings()
    row = next(r for r in rows if r["id"] == bid)
    assert row["campus"] == DEFAULT_CAMPUS


def test_map_json_without_campus_field_backfills_default():
    """A map.json written before the campus field existed has no "campus" key at all."""
    bid = store.create_building("Legacy Campus Hall").building_id
    path = store.config.BUILDINGS_DIR / bid / "map.json"
    data = json.loads(path.read_text())
    assert "campus" in data
    del data["campus"]
    path.write_text(json.dumps(data))
    reloaded = store.load_map(bid)
    assert reloaded.campus == DEFAULT_CAMPUS
