"""Shared test setup.

Nothing a test writes may reach backend/data (the live data folder of a running server):
- Before the app is imported, DATA_DIR is pointed at a scratch folder and ENV_FILE is emptied, so backend/.env is not
  read either. config computes BUILDINGS_DIR at import time, so this is the only moment DATA_DIR alone is enough.
- Every test then gets its own temporary DATA_DIR and BUILDINGS_DIR (both moved together), and the owner list,
  the legacy owner token and the hosted-model key are fixed, whatever the developer's shell exports.
- A guard compares backend/data before and after the whole run. If anything under it was created, changed or removed,
  the run fails. (If a server is running from backend/data while the tests run, the guard reports its writes too.)
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

REAL_DATA = Path(__file__).resolve().parents[1] / "data"
SCRATCH = Path(tempfile.mkdtemp(prefix="indoor-nav-tests-"))
os.environ["DATA_DIR"] = str(SCRATCH)  # read by app.config at import
os.environ["ENV_FILE"] = ""  # the tests do not read backend/.env


def _snapshot(root: Path) -> dict:
    """Name, size and modification time of everything under root (empty dict when root does not exist)."""
    if not root.exists():
        return {}
    snap = {"<root>": (root.stat().st_mtime_ns,)}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            path = os.path.join(dirpath, name)
            try:
                st = os.stat(path, follow_symlinks=False)
            except OSError:
                continue
            snap[os.path.relpath(path, root)] = (st.st_size, st.st_mtime_ns)
    return snap


_BEFORE = _snapshot(REAL_DATA)

import httpx  # noqa: E402
import pytest  # noqa: E402

from app import auth, config, hosted, moderation  # noqa: E402
from app.moderation import ModerationVerdict  # noqa: E402

TEST_OWNER_EMAILS = frozenset({"owner@example.com"})


def pytest_configure(config):
    config.addinivalue_line("markers", "real_moderation: keep the real face blur and content check (no stub)")


def pytest_sessionfinish(session, exitstatus):
    after = _snapshot(REAL_DATA)
    shutil.rmtree(SCRATCH, ignore_errors=True)
    if after != _BEFORE:
        changed = sorted(set(after) ^ set(_BEFORE)) + sorted(k for k in after if k in _BEFORE and after[k] != _BEFORE[k])
        sys.stderr.write(
            "\nERROR: the test run changed backend/data (the live data folder). Tests must write only to a temporary "
            f"DATA_DIR. Changed entries: {changed[:10]}{' ...' if len(changed) > 10 else ''}\n")
        session.exitstatus = 1


@pytest.fixture(autouse=True)
def _temp_data_dir(tmp_path_factory, monkeypatch):
    data = tmp_path_factory.mktemp("data")
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "BUILDINGS_DIR", data / "buildings")
    assert not str(config.BUILDINGS_DIR.resolve()).startswith(str(REAL_DATA.resolve())), "test points at backend/data"


@pytest.fixture(autouse=True)
def _owners_and_token(monkeypatch):
    """Fixed owner list (owner@example.com) and no legacy token unless a test sets one."""
    monkeypatch.setattr(config, "OWNER_EMAILS", TEST_OWNER_EMAILS)
    monkeypatch.setattr(config, "OWNER_TOKEN", "")


@pytest.fixture(autouse=True)
def _fresh_link_rate_limit(monkeypatch):
    """The request-link limiter is in memory and module-level. Each test starts with empty counters, so one test's
    sign-in requests (all from the test client's IP) cannot trip the limit in another."""
    monkeypatch.setattr(auth, "_link_hits", {})


@pytest.fixture(autouse=True)
def _no_hosted_model(monkeypatch):
    """No test reaches the hosted Gemini API: no key (even if the developer's shell exports one), the default
    provider, empty reachability cache, and any unmocked request to the API fails the test. test_hosted.py mocks
    the transport and sets its own key."""
    def no_network(request):
        raise AssertionError(f"unmocked hosted API call: {request.method} {request.url}")

    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    monkeypatch.setattr(config, "VISION_PROVIDER", "auto")
    monkeypatch.setattr(hosted, "_checks", {})
    monkeypatch.setattr(hosted, "_client", lambda timeout: httpx.Client(transport=httpx.MockTransport(no_network)))


@pytest.fixture(autouse=True)
def _moderation_stub(request, monkeypatch):
    """Walk pipeline tests do not run the face model or the vision model. Each keyframe is blurred by a plain copy
    and approved. Tests marked real_moderation (test_moderation.py) keep the real functions."""
    if request.node.get_closest_marker("real_moderation"):
        return
    monkeypatch.setattr(moderation, "blur_faces", lambda frame: frame.copy())
    monkeypatch.setattr(moderation, "moderate_keyframe", lambda frame, model=None: (
        frame.copy(), ModerationVerdict(approved=True, reason="test stub", model="test-model")))
