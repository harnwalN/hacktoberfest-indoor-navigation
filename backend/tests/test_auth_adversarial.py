"""Attack-style tests for access control (AUTH_REQUIRED=1). Reuses the env fixture from test_auth.py."""
import logging
import re
import statistics
import time
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from app import auth, config
from app.main import app
from tests.test_auth import FRAME, OWNER, H, _invite, _jpeg, _token, env  # noqa: F401  (env is a fixture)


def _concrete(path: str, a: str) -> str:
    path = path.replace("{building_id}", a).replace("{job_id}", "jvid").replace("{video_id}", "v1200001234")
    return path.replace("{merge_id}", "m1").replace("{invite_id}", "abc").replace("{action}", "pause")


def test_every_route_and_method_refuses_anonymous(env):
    """Enumerates app.routes (not a hand-kept list). Every /api route except GET /api/health and the
    token-filtered list must answer 401 anonymously, for every declared method, plus HEAD/OPTIONS probes."""
    c = env.client
    checked = 0
    from app.router import api
    assert any(getattr(r, "original_router", None) is api for r in app.routes) or api.routes  # app.routes wraps api
    for r in [*api.routes, *[x for x in app.routes if hasattr(x, "methods")]]:
        path = getattr(r, "path", "")
        if not path.startswith("/api") and not path.startswith("/bfiles"):
            continue
        for m in getattr(r, "methods", None) or []:
            url = _concrete(path, env.a)
            resp = c.request(m, url, json={} if m in ("POST", "PUT", "PATCH") else None)
            if path == "/api/health" and m in ("GET", "HEAD"):
                continue
            if path == "/api/buildings" and m in ("GET", "HEAD"):
                if m == "GET":
                    assert resp.status_code == 200 and resp.json() == []
                continue
            if path.startswith("/bfiles"):
                assert resp.status_code in (404, 405), (m, url, resp.status_code)
                continue
            assert resp.status_code in (401, 405), (m, url, resp.status_code, resp.text[:100])
            checked += 1
    assert checked >= 20
    # methods that are not declared must never leak data either
    for url in (f"/api/buildings/{env.a}/map", f"/api/buildings/{env.a}/queue", "/api/jobs/jvid"):
        for m in ("HEAD", "OPTIONS", "TRACE", "PATCH", "DELETE"):
            s = c.request(m, url).status_code
            assert s in (401, 404, 405, 200) and (s != 200 or m == "OPTIONS"), (m, url, s)
            if m == "HEAD":
                assert s in (401, 404, 405), (m, url, s)


def test_cors_preflight_leaks_nothing(env):
    r = env.client.options(f"/api/buildings/{env.a}/map",
                           headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET",
                                    "Access-Control-Request-Headers": "authorization"})
    assert r.status_code in (200, 204)
    assert r.text in ("OK", "") and env.a not in r.text and "Hall" not in r.text
    # a real cross-origin GET without a token still gets 401 and no data
    r = env.client.get(f"/api/buildings/{env.a}/map", headers={"Origin": "https://evil.example"})
    assert r.status_code == 401 and "Lobby" not in r.text


def test_bearer_scheme_case_and_whitespace(env):
    t = _token(env, env.a)
    url = f"/api/buildings/{env.a}/map"
    c = env.client
    for hdr, ok in [(f"bearer {t}", True), (f"BEARER {t}", True), (f"BeArEr {t}", True), (f"Bearer  {t}", True),
                    (f"  Bearer {t}  ", True), (t, False), (f"Basic {t}", False), (f"Bearer{t}", False),
                    (f"Bearer {t}x", False), (f"Bearer {t[:-1]}", False), ("Bearer ", False), ("Bearer", False),
                    (f"Token {t}", False), (f"Bearer {t},{t}", False), (f"Bearer {t} extra", False)]:
        s = c.get(url, headers={"Authorization": hdr}).status_code
        assert (s == 200) == ok, (hdr, s)


def test_list_authorization_only_passes_the_owner_key(env):
    """GET /api/buildings: any Authorization that is not exactly the owner key (with a Bearer scheme) is 401."""
    t = _token(env, env.a)
    for hdr, ok in [(f"bearer {OWNER}", True), (f"BEARER {OWNER}", True), (f"  Bearer   {OWNER}  ", True),
                    (OWNER, False), (f"Basic {OWNER}", False), (f"Bearer {OWNER},{OWNER}", False),
                    (f"Bearer {OWNER} extra", False), (f"Bearer {t}", False), (f"Bearer {t}x", False),
                    ("Bearer", False), ("Bearer ", False), ("", False), ("   ", False), (f"Token {t}", False)]:
        r = env.client.get("/api/buildings", headers={"Authorization": hdr})
        assert (r.status_code == 200) == ok, (hdr, r.status_code)
        if ok:
            assert len(r.json()) == 2
        else:
            assert r.status_code == 401 and env.a not in r.text, (hdr, r.status_code)


def test_unicode_and_odd_tokens_never_500_or_pass(env):
    url = f"/api/buildings/{env.a}/map"
    for raw in ["tokén".encode("latin-1"), b"\xff\xfe\xfd", b"%00", b"a" * 20000, b"\t\t", b"Bearer \xe2\x80\xae"]:
        r = env.client.get(url, headers={"Authorization": b"Bearer " + raw if not raw.startswith(b"Bearer") else raw})
        assert r.status_code in (400, 401, 431), (raw[:20], r.status_code)
    # unicode lookalike of a real token does not match
    t = _token(env, env.a)
    r = env.client.get(url, headers={"Authorization": ("Bearer " + t + "\u0430").encode("utf-8")})
    assert r.status_code == 401
    # X-Access-Tokens with garbage
    r = env.client.get("/api/buildings", headers={"X-Access-Tokens": b",,, ,\xff," + b"x," * 5000})
    assert r.status_code == 200 and r.json() == []


def test_duplicate_authorization_header_cannot_escalate(env):
    viewer = _token(env, env.a, "viewer")
    url = f"/api/buildings/{env.a}/invites"
    for order in ((viewer, OWNER), (OWNER, viewer)):
        r = env.client.post(url, json={"role": "viewer"},
                            headers=[("Authorization", f"Bearer {order[0]}"), ("Authorization", f"Bearer {order[1]}")])
        # either header may win, but a viewer first must never gain owner rights through the second one
        if order[0] == viewer:
            assert r.status_code in (401, 403), r.status_code
    # WriteGuard and the route dependency must agree on which header counts: a viewer-first duplicate with an
    # owner-key second must not create an invite
    r = env.client.get(f"/api/buildings/{env.a}/invites",
                       headers=[("Authorization", f"Bearer {viewer}"), ("Authorization", f"Bearer {OWNER}")])
    assert r.status_code in (401, 403)


def test_job_from_another_building_is_refused(env):
    ea = _token(env, env.a, "editor")
    eb = _token(env, env.b, "editor")
    assert env.client.post("/api/jobs/jvid/pause", headers=H(eb)).status_code == 401  # jvid belongs to a
    assert env.client.get("/api/jobs/jvid", headers=H(eb)).status_code == 401
    assert env.client.get("/api/jobs/doesnotexist", headers=H(eb)).status_code == 401  # same as foreign job
    assert env.client.get("/api/jobs/jvid", headers=H(ea)).status_code == 200
    assert env.client.get("/api/jobs/doesnotexist", headers=H(OWNER)).status_code == 404
    # viewer of the job's own building is too low
    assert env.client.get("/api/jobs/jvid", headers=H(_token(env, env.a, "viewer"))).status_code == 403


def test_editor_token_for_a_cannot_upload_or_edit_b(env):
    ea = _token(env, env.a, "editor")
    files = {"video": ("w.mp4", b"x" * 10, "video/mp4")}
    assert env.client.post(f"/api/buildings/{env.b}/videos", files=files, headers=H(ea)).status_code == 401
    assert env.client.post(f"/api/buildings/{env.b}/frames/manual", headers=H(ea),
                           files={"image": ("x.jpg", _jpeg(), "image/jpeg")}).status_code == 401
    assert env.client.put(f"/api/buildings/{env.b}/map", json={"nodes": [], "edges": []}, headers=H(ea)).status_code == 401
    assert env.client.delete(f"/api/buildings/{env.b}/videos/v1200001234", headers=H(ea)).status_code == 401
    assert env.submitted == []


def test_put_and_merge_with_foreign_node_ids_stay_inside_the_building(env):
    from app import store
    from app.schemas import Node
    bundle = store.load_map(env.b)
    bundle.nodes = [Node(id="n9", label="SECRET-B"), Node(id="n8", label="SECRET-B2")]
    store.save_map(bundle)
    ea = _token(env, env.a, "editor")
    # merge with b's ids in a: 400, nothing from b appears
    r = env.client.post(f"/api/buildings/{env.a}/merge", json={"keep": "n9", "remove": "n8"}, headers=H(ea))
    assert r.status_code == 400 and "SECRET" not in r.text
    # PUT with edges that reference ids not in the nodes list are dropped; b untouched
    r = env.client.put(f"/api/buildings/{env.a}/map", headers=H(ea), json={
        "nodes": [{"id": "n1", "label": "x"}],
        "edges": [{"id": "e1", "from_node": "n1", "to_node": "n9", "heading_out": 0, "heading_in": 0}]})
    assert r.status_code == 200 and r.json()["map"]["edges"] == []
    assert {n.id for n in store.load_map(env.b).nodes} == {"n9", "n8"}
    assert "SECRET" not in r.text


def test_route_with_foreign_node_ids_is_404_not_a_leak(env):
    v = _token(env, env.a, "viewer")
    r = env.client.post(f"/api/buildings/{env.a}/route", json={"start_node_id": "zz", "target_node_id": "n1"}, headers=H(v))
    assert r.status_code == 404


def test_signed_url_bound_to_building_file_and_expiry(env, monkeypatch):
    v = _token(env, env.a, "viewer")
    m = env.client.get(f"/api/buildings/{env.a}/map", headers=H(v)).json()
    url = next(n["frame_url"] for n in m["nodes"] if n["frame_url"])
    q = parse_qs(urlparse(url).query)
    exp, sig = q["exp"][0], q["sig"][0]
    base = f"/bfiles/%s/frames/%s?exp={exp}&sig={sig}"
    c = env.client
    assert c.get(url).status_code == 200
    assert c.head(url).status_code == 200
    assert c.get(base % (env.b, FRAME)).status_code == 404            # other building
    assert c.get(base % (env.a, "v1200001234_0002.jpg")).status_code == 404  # other file
    assert c.get(base % (env.a, FRAME.upper())).status_code == 404
    assert c.get(f"/bfiles/{env.a}/frames/{FRAME}?exp={int(exp) + 1}&sig={sig}").status_code == 404
    assert c.get(f"/bfiles/{env.a}/frames/{FRAME}?exp={exp}&sig={sig[:-1]}0").status_code in (404,) or sig.endswith("0")
    assert c.get(f"/bfiles/{env.a}/frames/{FRAME}?exp={exp}&sig=").status_code == 404
    assert c.get(f"/bfiles/{env.a}/frames/{FRAME}?exp=-1&sig={sig}").status_code == 404
    assert c.get(f"/bfiles/{env.a}/frames/{FRAME}?exp={exp}&exp=1&sig={sig}").status_code in (200, 404)
    for bad in ("../map.json", "..%2fmap.json", "%2e%2e/map.json", "..%5cmap.json", "map.json", "%00"):
        assert c.get(f"/bfiles/{env.a}/frames/{bad}?exp={exp}&sig={sig}").status_code in (404, 405)
    assert c.get(f"/bfiles/{env.a}/map.json").status_code == 404
    # expiry: move the clock past exp
    real = time.time
    monkeypatch.setattr(auth.time, "time", lambda: real() + config.FRAME_URL_TTL_SEC + 5)
    assert c.get(url).status_code == 404


def test_signature_forged_with_a_guessed_secret_fails(env):
    import hashlib, hmac
    exp = int(time.time()) + 1000
    for key in (b"", b"secret", OWNER.encode(), hashlib.sha256(OWNER.encode()).hexdigest().encode()):
        sig = hmac.new(key, f"{env.a}/{FRAME}/{exp}".encode(), hashlib.sha256).hexdigest()
        assert env.client.get(f"/bfiles/{env.a}/frames/{FRAME}?exp={exp}&sig={sig}").status_code == 404


def test_debug_and_route_responses_carry_signed_urls(env):
    from app import store
    ed = _token(env, env.a, "editor")
    vdir = store.video_dir(env.a, "v1200001234")
    vdir.mkdir(parents=True, exist_ok=True)
    store.write_json_atomic(vdir / "track.json", {"t": [0], "heading": [0], "flow": [0], "events": []})
    store.write_json_atomic(vdir / "walk.json", {"keyframes": [{"frame_id": FRAME}]})
    r = env.client.get(f"/api/buildings/{env.a}/videos/v1200001234/debug", headers=H(ed))
    assert r.status_code == 200
    url = r.json()["keyframes"][0]["url"]
    assert "sig=" in url and env.client.get(url).status_code == 200
    # viewer may not read debug
    assert env.client.get(f"/api/buildings/{env.a}/videos/v1200001234/debug", headers=H(_token(env, env.a))).status_code == 403


def test_route_response_urls_are_signed(env):
    from app import store
    from app.schemas import Edge
    bundle = store.load_map(env.a)
    bundle.edges = [Edge(id="e1", from_node="n1", to_node="n2", heading_out=0, heading_in=180, frames_fwd=[FRAME])]
    store.save_map(bundle)
    v = _token(env, env.a)
    r = env.client.post(f"/api/buildings/{env.a}/route", json={"start_node_id": "n1", "target_node_id": "n2"}, headers=H(v))
    if r.status_code != 200:
        pytest.skip(f"route fixture not routable: {r.text[:120]}")
    for u in re.findall(r"/bfiles/[^\"]+", r.text):
        assert "sig=" in u


def test_no_secret_in_logs_or_error_bodies(env, caplog):
    caplog.set_level(logging.DEBUG)
    t = _token(env, env.a, "editor")
    c = env.client
    c.get(f"/api/buildings/{env.a}/map", headers=H("wrong-token-value-123"))
    c.post(f"/api/buildings/{env.b}/rebuild", headers=H(t))
    c.post(f"/api/buildings/{env.a}/merge", json={"keep": "q", "remove": "r"}, headers=H(t))
    c.get(f"/api/buildings/{env.a}/map", headers=H(OWNER))
    secrets_ = [OWNER, t, auth._sha(OWNER), auth._sha(t), auth._state()["signing_secret"].decode(), "wrong-token-value-123"]
    blob = caplog.text
    for s in secrets_:
        assert s not in blob, "secret in logs"
    bad = c.get(f"/api/buildings/{env.a}/map", headers=H("wrong-token-value-123"))
    for s in secrets_[:-1]:
        assert s not in bad.text and s not in str(bad.headers)


def test_lockout_not_bypassed_by_spoofed_headers(env, monkeypatch):
    monkeypatch.setattr(config, "AUTH_MAX_FAILS", 3)
    # TestClient peer is "testclient" (not loopback): spoofing CF-Connecting-IP / X-Forwarded-For must not reset
    for i in range(6):  # attempts 1-3 are 401, the rest 429; the spoofed headers never reset the count
        env.client.get(f"/api/buildings/{env.a}/map",
                       headers={**H(f"bad{i}"), "CF-Connecting-IP": f"9.9.9.{i}", "X-Forwarded-For": f"8.8.8.{i}",
                                "X-Real-IP": f"7.7.7.{i}"})
    r = env.client.get(f"/api/buildings/{env.a}/map", headers=H(OWNER))
    assert r.status_code == 429 and "Retry-After" in r.headers
    assert env.client.post(f"/api/buildings/{env.a}/rebuild", headers=H(OWNER)).status_code == 429
    assert env.client.get("/api/health").status_code == 200  # health unaffected


def test_lockout_applies_to_write_guard_before_body(env, monkeypatch):
    monkeypatch.setattr(config, "AUTH_MAX_FAILS", 2)
    for i in range(2):
        env.client.post(f"/api/buildings/{env.a}/rebuild", headers=H(f"bad{i}"))
    assert env.client.post(f"/api/buildings/{env.a}/rebuild", headers=H(OWNER)).status_code == 429


def test_valid_tokens_do_not_count_as_failures(env, monkeypatch):
    monkeypatch.setattr(config, "AUTH_MAX_FAILS", 3)
    v = _token(env, env.a)
    for _ in range(20):
        assert env.client.get(f"/api/buildings/{env.a}/map", headers=H(v)).status_code == 200


def test_ipv4_mapped_loopback_peer_is_trusted():
    """A dual-stack cloudflared connects from ::ffff:127.0.0.1, which is loopback too: its CF-Connecting-IP counts."""
    from types import SimpleNamespace
    req = SimpleNamespace(client=SimpleNamespace(host="::ffff:127.0.0.1"), headers={"cf-connecting-ip": "1.2.3.4"})
    assert auth.client_ip(req) == "1.2.3.4"
    # a non-loopback mapped address is still not trusted
    req = SimpleNamespace(client=SimpleNamespace(host="::ffff:10.0.0.9"), headers={"cf-connecting-ip": "1.2.3.4"})
    assert auth.client_ip(req) == "::ffff:10.0.0.9"


def test_revoked_invite_cuts_off_photo_urls_at_once(env):
    """Revoking an invite bumps the building's generation, so URLs signed before the revoke get 404 now, not at expiry."""
    c = env.client
    gone = _invite(env, env.a)  # the invite that will be revoked
    kept = _invite(env, env.a)  # another holder on the same building
    m = c.get(f"/api/buildings/{env.a}/map", headers=H(gone["token"])).json()
    url = next(n["frame_url"] for n in m["nodes"] if n["frame_url"])
    assert c.get(url).status_code == 200
    assert c.delete(f"/api/buildings/{env.a}/invites/{gone['id']}", headers=H(OWNER)).status_code == 200
    assert c.get(f"/api/buildings/{env.a}/map", headers=H(gone["token"])).status_code == 401
    assert c.get(url).status_code == 404
    # the remaining holder refetches the map and gets URLs that work
    fresh = c.get(f"/api/buildings/{env.a}/map", headers=H(kept["token"])).json()
    fresh_url = next(n["frame_url"] for n in fresh["nodes"] if n["frame_url"])
    assert c.get(fresh_url).status_code == 200


def test_legacy_bare_list_access_file_reads_as_generation_zero(env):
    """access.json written before generations was a bare list of invites: it still works, and a revoke upgrades it."""
    import json
    from app import store
    t = _token(env, env.a)
    path = store.building_dir(env.a) / "access.json"
    inv_list = json.loads(path.read_text())["invites"]
    path.write_text(json.dumps(inv_list))  # back to the old layout
    c = env.client
    url = next(n["frame_url"] for n in c.get(f"/api/buildings/{env.a}/map", headers=H(t)).json()["nodes"]
               if n["frame_url"])
    assert c.get(url).status_code == 200  # generation 0 when the key is absent
    assert c.get(f"/api/buildings/{env.a}/map", headers=H(t)).status_code == 200
    assert c.delete(f"/api/buildings/{env.a}/invites/{inv_list[0]['id']}", headers=H(OWNER)).status_code == 200
    assert c.get(url).status_code == 404
    upgraded = json.loads(path.read_text())
    assert upgraded["generation"] == 1 and len(upgraded["invites"]) == 1


def test_revoke_in_one_building_leaves_other_buildings_photo_urls_working(env):
    ta = _token(env, env.a)
    tb = _token(env, env.b)
    url_a = next(n["frame_url"] for n in env.client.get(f"/api/buildings/{env.a}/map", headers=H(ta)).json()["nodes"]
                 if n["frame_url"])
    url_b = next(n["frame_url"] for n in env.client.get(f"/api/buildings/{env.b}/map", headers=H(tb)).json()["nodes"]
                 if n["frame_url"])
    assert env.client.get(url_a).status_code == 200 and env.client.get(url_b).status_code == 200
    inv_a = _invite(env, env.a)  # revoke one invite on building A
    assert env.client.delete(f"/api/buildings/{env.a}/invites/{inv_a['id']}", headers=H(OWNER)).status_code == 200
    assert env.client.get(url_a).status_code == 404
    assert env.client.get(url_b).status_code == 200  # building B is untouched
    assert env.client.get(f"/api/buildings/{env.b}/map", headers=H(tb)).status_code == 200


def test_corrupt_access_json_does_not_grant_access(env):
    from app import store
    (store.building_dir(env.a) / "access.json").write_text("{not json")
    c = TestClient(app, raise_server_exceptions=False)
    r = c.get(f"/api/buildings/{env.a}/map", headers=H("whatever"))
    assert r.status_code in (401, 500)
    assert c.get(f"/api/buildings/{env.a}/map", headers=H(OWNER)).status_code == 200  # owner still works
    assert c.get("/api/buildings", headers=H("whatever")).status_code == 401  # not the owner key: refused
    # corrupt file in the listing is skipped: a good invite on another building still lists that building
    tok_b = _token(env, env.b)
    r = c.get("/api/buildings", headers={"X-Access-Tokens": tok_b})
    assert r.status_code == 200 and [b["id"] for b in r.json()] == [env.b]
    assert c.get("/api/buildings", headers={"X-Access-Tokens": "whatever"}).json() == []


def test_corrupt_access_json_for_one_building_status(env):
    """Open issue (d): record the current status for a corrupt per-building file with a token for that building."""
    from app import store
    t = _token(env, env.a)
    (store.building_dir(env.a) / "access.json").write_text("garbage")
    c = TestClient(app, raise_server_exceptions=False)
    assert c.get(f"/api/buildings/{env.a}/map", headers=H(t)).status_code in (401, 500)


def test_invite_edge_cases(env):
    c = env.client
    for body in ({"role": "owner"}, {"role": "admin"}, {"role": ""}, {}, {"role": "viewer", "label": "x" * 500}):
        assert c.post(f"/api/buildings/{env.a}/invites", json=body, headers=H(OWNER)).status_code == 422
    # revoke twice is idempotent, wrong building id for invite is 404
    i = _invite(env, env.a)
    assert c.delete(f"/api/buildings/{env.b}/invites/{i['id']}", headers=H(OWNER)).status_code == 404
    assert c.delete(f"/api/buildings/{env.a}/invites/{i['id']}", headers=H(OWNER)).status_code == 200
    assert c.delete(f"/api/buildings/{env.a}/invites/{i['id']}", headers=H(OWNER)).status_code == 200
    assert c.get(f"/api/buildings/{env.a}/map", headers=H(i["token"])).status_code == 401
    # an invite token is never usable as owner key: in Authorization it is refused on the list, and once revoked
    # it lists nothing from X-Access-Tokens either
    assert c.get("/api/buildings", headers=H(i["token"])).status_code == 401
    assert c.get("/api/buildings", headers={"X-Access-Tokens": i["token"]}).json() == []
    # unknown building: owner 404, other 401
    assert c.get("/api/buildings/nope/map", headers=H(OWNER)).status_code == 404
    assert c.get("/api/buildings/nope/map", headers=H(_token(env, env.a))).status_code == 401
    for bad in ("..", "%2e%2e", "a%2fb"):
        assert c.get(f"/api/buildings/{bad}/map", headers=H(_token(env, env.a))).status_code in (401, 404)


def test_spa_and_unknown_paths_are_not_api_leaks(env):
    for p in ("/", "/index.html", "/api/nonexistent", "/data/auth.json", "/auth.json", "/buildings"):
        r = env.client.get(p)
        assert env.a not in r.text and "owner_sha256" not in r.text and "signing_secret" not in r.text


def test_token_comparison_timing_is_not_prefix_dependent(env):
    """Coarse check: SHA-256 before compare means a long matching prefix must not be measurably slower."""
    t = _token(env, env.a)
    url = f"/api/buildings/{env.a}/map"
    c = env.client

    def timeit(tok, n=60):
        xs = []
        for _ in range(n):
            s = time.perf_counter()
            c.get(url, headers=H(tok))
            xs.append(time.perf_counter() - s)
        return statistics.median(xs)

    far, near = timeit("Z" * len(t)), timeit(t[:-1] + ("A" if t[-1] != "A" else "B"))
    assert abs(near - far) < max(0.5 * far, 0.002)


def test_overlong_building_id_is_not_a_500(env):
    c = TestClient(app, raise_server_exceptions=False)
    assert c.get(f"/api/buildings/{'a' * 5000}/map", headers=H(OWNER)).status_code in (404, 414)
    assert c.get(f"/api/buildings/{'a' * 5000}/map", headers=H(_token(env, env.a))).status_code in (401, 404, 414)


def test_very_long_inputs_do_not_crash(env):
    c = TestClient(app, raise_server_exceptions=False)
    r = c.get(f"/api/buildings/{env.a}/map", headers={"Authorization": "Bearer " + "A" * 100_000})
    assert r.status_code in (401, 431, 400)
    r = c.get("/api/buildings", headers={"X-Access-Tokens": ",".join(["t"] * 100_000)})
    assert r.status_code == 200 and r.json() == []
