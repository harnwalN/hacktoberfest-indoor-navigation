import json

from app import gemma
from app.gemma import GemmaCache


def test_newlines_and_unicode_roundtrip(tmp_path):
    p = tmp_path / "gemma_cache.jsonl"
    c = GemmaCache(p)
    k = "a\nb\r\nc d \U0001F600 é"
    v = {"t": "line1\nline2  中文", "n": [1, None, 2.5]}
    c.put(k, v)
    assert len(p.read_bytes().split(b"\n")) == 2  # one record + trailing empty
    assert GemmaCache(p).get(k) == v


def test_empty_file(tmp_path):
    p = tmp_path / "gemma_cache.jsonl"
    p.write_bytes(b"")
    c = GemmaCache(p)
    assert c.data == {}
    c.put("a", 1)
    assert GemmaCache(p).get("a") == 1


def test_empty_jsonl_blocks_legacy_migration(tmp_path):
    (tmp_path / "gemma_cache.json").write_text(json.dumps({"old": 1}))
    (tmp_path / "gemma_cache.jsonl").write_bytes(b"")
    c = GemmaCache(tmp_path / "gemma_cache.jsonl")
    assert c.get("old") is None  # documents behaviour: legacy not migrated
    assert (tmp_path / "gemma_cache.json").exists()


def test_migration_runs_once_and_renames(tmp_path):
    (tmp_path / "gemma_cache.json").write_text(json.dumps({"a": {"x": 1}, "b": 2}))
    c = GemmaCache(tmp_path / "gemma_cache.jsonl")
    assert c.get("a") == {"x": 1} and c.get("b") == 2
    assert not (tmp_path / "gemma_cache.json").exists()
    assert (tmp_path / "gemma_cache.json.bak").exists()
    c.put("c", 3)
    c2 = GemmaCache(tmp_path / "gemma_cache.jsonl")
    assert c2.data == {"a": {"x": 1}, "b": 2, "c": 3}


def test_legacy_filename_passed_directly(tmp_path):
    (tmp_path / "gemma_cache.json").write_text(json.dumps({"a": 1}))
    c = GemmaCache(tmp_path / "gemma_cache.json")
    assert c.get("a") == 1 and c.path.suffix == ".jsonl"


def test_valid_json_wrong_shape_lines_skipped(tmp_path):
    p = tmp_path / "gemma_cache.jsonl"
    p.write_text('[1,2]\n"str"\n{"k":"a"}\n{"k":"ok","v":0}\nnull\n')
    c = GemmaCache(p)
    assert c.data == {"ok": 0}


def test_unhashable_key_line_skipped(tmp_path):
    p = tmp_path / "gemma_cache.jsonl"
    p.write_text('{"k":[1],"v":2}\n{"k":"ok","v":0}\n')
    assert GemmaCache(p).data == {"ok": 0}


def test_falsy_value_is_hit_not_miss(tmp_path):
    c = GemmaCache(tmp_path / "gemma_cache.jsonl")
    c.put("z", {})
    assert GemmaCache(tmp_path / "gemma_cache.jsonl").get("z") == {}


def test_cache_for_same_path_variants_share(tmp_path):
    (tmp_path / "sub").mkdir()
    a = gemma.cache_for(tmp_path / "sub")
    b = gemma.cache_for(tmp_path / "sub" / ".." / "sub")
    assert a is b
    assert a.path.name == "gemma_cache.jsonl"


def test_truncated_tail_without_newline_then_compaction_clean(tmp_path):
    p = tmp_path / "gemma_cache.jsonl"
    lines = "".join(json.dumps({"k": "k", "v": i}) + "\n" for i in range(300))
    p.write_text(lines + '{"k":"x","v":')
    c = GemmaCache(p)
    assert c.get("k") == 299
    c.put("y", 1)
    c2 = GemmaCache(p)
    assert c2.get("k") == 299 and c2.get("y") == 1
