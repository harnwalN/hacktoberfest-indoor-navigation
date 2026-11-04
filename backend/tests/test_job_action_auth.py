"""POST /api/jobs/{id}/{action} (pause, resume, run-next) changes the processing queue: owner only.

Who may call it: an owner session, or the legacy X-Owner-Token (when OWNER_TOKEN is set). A contributor session gets 403,
no credentials gets 401. Reads (GET /jobs/{id}, the queue) stay open. A refused request changes nothing.
"""
import pytest
from fastapi.testclient import TestClient

from app import config, jobs
from app.main import app
from tests.auth_helpers import owner_headers, signed_in_headers

client = TestClient(app)


@pytest.fixture
def queue(monkeypatch):
    monkeypatch.setattr(jobs, "JOBS", {})
    monkeypatch.setattr(jobs, "_ensure_worker", lambda: None)  # no background worker picks the job up
    job = jobs.Job(id="jq1", building_id="b", video_id=None, kind="rebuild", status="queued")  # no map on disk needed
    jobs.JOBS[job.id] = job
    return job


ACTIONS = ["pause", "resume", "run-next"]


@pytest.mark.parametrize("action", ACTIONS)
def test_job_action_without_credentials_is_401_and_changes_nothing(queue, action):
    r = client.post(f"/api/jobs/{queue.id}/{action}")
    assert r.status_code == 401, r.text
    assert r.json()["detail"] == "Owner sign-in required"
    assert jobs.JOBS[queue.id].status == "queued"


@pytest.mark.parametrize("action", ACTIONS)
def test_job_action_with_bad_session_is_401(queue, action):
    r = client.post(f"/api/jobs/{queue.id}/{action}", headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401
    assert jobs.JOBS[queue.id].status == "queued"


@pytest.mark.parametrize("action", ACTIONS)
def test_job_action_by_contributor_is_403_and_changes_nothing(queue, action):
    r = client.post(f"/api/jobs/{queue.id}/{action}", headers=signed_in_headers())
    assert r.status_code == 403, r.text
    assert jobs.JOBS[queue.id].status == "queued"


def test_unknown_job_without_credentials_is_401_not_404(queue):
    assert client.post("/api/jobs/nope/pause").status_code == 401


def test_unknown_job_by_contributor_is_403_not_404(queue):
    assert client.post("/api/jobs/nope/pause", headers=signed_in_headers()).status_code == 403


def test_job_action_with_owner_session_passes(queue):
    r = client.post(f"/api/jobs/{queue.id}/pause", headers=owner_headers())
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "paused"
    assert jobs.JOBS[queue.id].status == "paused"


def test_job_action_with_owner_session_still_validates_action_and_job(queue):
    headers = owner_headers()
    assert client.post(f"/api/jobs/{queue.id}/explode", headers=headers).status_code == 400
    assert client.post("/api/jobs/nope/pause", headers=headers).status_code == 404


def test_job_action_with_legacy_token_and_no_session_passes(queue, monkeypatch):
    monkeypatch.setattr(config, "OWNER_TOKEN", "s3cret")
    r = client.post(f"/api/jobs/{queue.id}/pause", headers={"X-Owner-Token": "s3cret"})
    assert r.status_code == 200, r.text
    assert jobs.JOBS[queue.id].status == "paused"


def test_job_action_with_wrong_token_and_no_session_is_refused_by_the_gate(queue, monkeypatch):
    monkeypatch.setattr(config, "OWNER_TOKEN", "s3cret")
    r = client.post(f"/api/jobs/{queue.id}/pause", headers={"X-Owner-Token": "nope"})
    assert r.status_code == 401 and r.json() == {"detail": "Owner token or sign-in required"}
    assert jobs.JOBS[queue.id].status == "queued"


def test_queue_reads_stay_open_without_session(queue):
    assert client.get(f"/api/jobs/{queue.id}").status_code == 200
