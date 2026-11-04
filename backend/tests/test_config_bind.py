import os
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent


def _bind_host(extra_env: dict) -> str:
    env = {k: v for k, v in os.environ.items() if k != "BACKEND_HOST"}
    env.update(extra_env)
    out = subprocess.run(
        [sys.executable, "-c", "from app import config; print(config.BACKEND_HOST)"],
        cwd=BACKEND, env=env, capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def test_default_bind_is_loopback():
    assert _bind_host({}) == "127.0.0.1"


def test_bind_host_env_override():
    assert _bind_host({"BACKEND_HOST": "0.0.0.0"}) == "0.0.0.0"
