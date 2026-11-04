"""Mail settings in app/config.py: defaults and the smtp-needs-host check.

config validates at import time, so each case runs in a fresh interpreter with a clean mail environment.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
MAIL_VARS = ("MAIL_BACKEND", "SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "SMTP_FROM", "SMTP_STARTTLS",
             "PUBLIC_BASE_URL")
PROBE = (
    "from app import config as c; import json; "
    "print(json.dumps({'MAIL_BACKEND': c.MAIL_BACKEND, 'SMTP_HOST': c.SMTP_HOST, 'SMTP_PORT': c.SMTP_PORT, "
    "'SMTP_STARTTLS': c.SMTP_STARTTLS, 'PUBLIC_BASE_URL': c.PUBLIC_BASE_URL}))"
)


def _load(**env: str) -> subprocess.CompletedProcess:
    clean = {k: v for k, v in os.environ.items() if k not in MAIL_VARS}
    clean.update(env)
    return subprocess.run(
        [sys.executable, "-c", PROBE], cwd=BACKEND_DIR, env=clean, capture_output=True, text=True, timeout=120
    )


def test_default_is_log_mode():
    result = _load()
    assert result.returncode == 0, result.stderr
    import json

    settings = json.loads(result.stdout.strip().splitlines()[-1])
    assert settings["MAIL_BACKEND"] == "log"
    assert settings["SMTP_HOST"] == ""
    assert settings["SMTP_PORT"] == 587
    assert settings["SMTP_STARTTLS"] is True


def test_smtp_without_host_raises():
    result = _load(MAIL_BACKEND="smtp")
    assert result.returncode != 0
    assert "MAIL_BACKEND=smtp needs SMTP_HOST" in result.stderr


def test_smtp_with_blank_host_raises():
    result = _load(MAIL_BACKEND="smtp", SMTP_HOST="   ")
    assert result.returncode != 0
    assert "MAIL_BACKEND=smtp needs SMTP_HOST" in result.stderr


def test_smtp_without_public_base_url_raises():
    result = _load(MAIL_BACKEND="smtp", SMTP_HOST="smtp.example.com")
    assert result.returncode != 0
    assert "MAIL_BACKEND=smtp needs PUBLIC_BASE_URL" in result.stderr


def test_public_base_url_must_be_http_or_https():
    result = _load(PUBLIC_BASE_URL="nav.example.org")
    assert result.returncode != 0
    assert "PUBLIC_BASE_URL must start with http:// or https://" in result.stderr


def test_smtp_with_host_is_accepted():
    result = _load(MAIL_BACKEND="smtp", SMTP_HOST="smtp.example.com", SMTP_PORT="2525",
                   PUBLIC_BASE_URL="https://nav.example.org/")
    assert result.returncode == 0, result.stderr
    import json

    settings = json.loads(result.stdout.strip().splitlines()[-1])
    assert settings["MAIL_BACKEND"] == "smtp"
    assert settings["SMTP_HOST"] == "smtp.example.com"
    assert settings["SMTP_PORT"] == 2525
    assert settings["PUBLIC_BASE_URL"] == "https://nav.example.org"


def test_unknown_backend_raises():
    result = _load(MAIL_BACKEND="sendgrid")
    assert result.returncode != 0
    assert "MAIL_BACKEND must be log or smtp" in result.stderr
