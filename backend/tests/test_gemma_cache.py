import json
import threading
import time

from app import gemma
from app.gemma import GemmaCache, cache_for


def _val(i: int) -> dict:
    return {"texts": [f"Room {i}"], "confidence": 0.5}


def test_many_puts_are_fast_and_linear(tmp_path):
    c = cache_for(tmp_path)
    t0 = time.perf_counter()
    for i in range(2500):
        c.put(f"k{i:05d}", _val(i))
    size_half = c.path.stat().st_size
    for i in range(2500, 5000):
        c.put(f"k{i:05d}", _val(i))
    elapsed = time.perf_counter() - t0
    size_full = c.path.stat().st_size
    assert elapsed < 5.0, f"5000 puts took {elapsed:.2f}s"
    assert 1.9 < size_full / size_half < 2.1
    assert len(c.path.read_text().splitlines()) == 5000
    assert GemmaCache(c.path).data == c.data


def test_cache_for_shares_instance(tmp_path):
    a = cache_for(tmp_path)
    b = cache_for(tmp_path / "." )
    assert a is b
    a.put("pipeline", {"x": 1})
    b.put("manual", {"x": 2})
    fresh = GemmaCache(tmp_path / gemma.CACHE_FILE)
    assert fresh.get("pipeline") == {"x": 1}
    assert fresh.get("manual") == {"x": 2}


def test_concurrent_puts_keep_all_entries(tmp_path):
    c = cache_for(tmp_path)

    def worker(n):
        for i in range(200):
            cache_for(tmp_path).put(f"{n}:{i}", _val(i))

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    fresh = GemmaCache(c.path)
    assert len(fresh.data) == 1600


def test_legacy_json_migrates(tmp_path):
    old = {f"analyze:v4:m:plain:{i}": _val(i) for i in range(50)}
    (tmp_path / "gemma_cache.json").write_text(json.dumps(old, indent=2))
    c = cache_for(tmp_path)
    assert c.data == old
    assert (tmp_path / "gemma_cache.jsonl").exists()
    assert not (tmp_path / "gemma_cache.json").exists()
    assert json.loads((tmp_path / "gemma_cache.json.bak").read_text()) == old
    # reload reads the JSONL, not the backup
    c.put("new", {"y": 1})
    fresh = GemmaCache(tmp_path / gemma.CACHE_FILE)
    assert fresh.data == {**old, "new": {"y": 1}}


def test_legacy_ignored_when_jsonl_exists(tmp_path):
    (tmp_path / "gemma_cache.json").write_text(json.dumps({"stale": 1}))
    (tmp_path / "gemma_cache.jsonl").write_text(json.dumps({"k": "fresh", "v": 2}) + "\n")
    c = GemmaCache(tmp_path / gemma.CACHE_FILE)
    assert c.data == {"fresh": 2}
    assert (tmp_path / "gemma_cache.json").exists()  # untouched


def test_corrupt_legacy_does_not_crash(tmp_path):
    (tmp_path / "gemma_cache.json").write_text('{"a": ')
    c = GemmaCache(tmp_path / gemma.CACHE_FILE)
    assert c.data == {}
    assert (tmp_path / "gemma_cache.json").exists()  # left in place for inspection


def test_truncated_last_line_skipped_and_append_recovers(tmp_path):
    p = tmp_path / gemma.CACHE_FILE
    good = "".join(json.dumps({"k": f"k{i}", "v": i}) + "\n" for i in range(3))
    p.write_text(good + '{"k": "k3", "v": {"tru')
    c = GemmaCache(p)
    assert c.data == {"k0": 0, "k1": 1, "k2": 2}
    c.put("k4", 4)  # must not be glued onto the broken line
    assert GemmaCache(p).data == {"k0": 0, "k1": 1, "k2": 2, "k4": 4}


def test_garbage_lines_skipped(tmp_path):
    p = tmp_path / gemma.CACHE_FILE
    p.write_text('not json\n[1,2]\n{"no_k": 1}\n\n' + json.dumps({"k": "ok", "v": 1}) + "\n")
    assert GemmaCache(p).data == {"ok": 1}


def test_last_write_wins(tmp_path):
    p = tmp_path / gemma.CACHE_FILE
    c = GemmaCache(p)
    c.put("k", {"v": 1})
    c.put("k", {"v": 2})
    assert c.get("k") == {"v": 2}
    assert GemmaCache(p).get("k") == {"v": 2}


def test_compaction_on_load(tmp_path):
    p = tmp_path / gemma.CACHE_FILE
    c = GemmaCache(p)
    for i in range(300):
        c.put("same", i)
    assert len(p.read_text().splitlines()) == 300
    fresh = GemmaCache(p)
    assert fresh.data == {"same": 299}
    assert len(p.read_text().splitlines()) == 1


def test_unserializable_value_leaves_cache_untouched(tmp_path):
    c = GemmaCache(tmp_path / gemma.CACHE_FILE)
    try:
        c.put("bad", {"x": object()})
    except TypeError:
        pass
    assert c.get("bad") is None
    assert GemmaCache(c.path).data == {}
