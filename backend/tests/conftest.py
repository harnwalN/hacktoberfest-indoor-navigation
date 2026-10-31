"""Shared test setup.

The older API tests call the API without tokens, so access control is off for every test by default.
test_auth.py turns it back on. DATA_DIR points at a temp directory, so no owner key or signing secret is written
into backend/data during the run.
"""
import httpx
import pytest

from app import config, hosted


@pytest.fixture(autouse=True)
def _api_without_auth(tmp_path_factory, monkeypatch):
    monkeypatch.setattr(config, "AUTH_REQUIRED", False)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path_factory.mktemp("data"))


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
