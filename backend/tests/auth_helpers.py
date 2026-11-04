"""Sign-in for tests: a contributor account and a session token, made directly (no HTTP round trip).

Call it only inside a test or a fixture that runs after conftest's autouse data-dir fixture, so the account and
session are written to the test's temporary DATA_DIR, never to the real one.
"""
from app import auth, store

OWNER_EMAIL = "owner@example.com"  # conftest sets config.OWNER_EMAILS to this address


def signed_in_headers(email: str = "walker@example.com") -> dict:
    account = store.create_account_if_missing(email)
    return {"Authorization": f"Bearer {auth.create_session(account.id)}"}


def owner_headers() -> dict:
    """A session of the owner account (conftest's owner list)."""
    return signed_in_headers(OWNER_EMAIL)
