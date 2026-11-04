"""Sign-in routes: magic link by email, then a session token. Mounted from main.py under /api/auth.

POST /api/auth/request-link  {email}            -> 200 {status: "sent", message}  (same answer for every address)
POST /api/auth/verify        {token}            -> 200 {session_token, expires_in, account: {id, email, moderator_of}}
GET  /api/auth/me            Authorization: Bearer <session_token> -> 200 {account: {...}, is_owner, blocked}  or 401

The link in the email opens the app at {PUBLIC_BASE_URL}/#/verify?token=... The page then POSTs the token to
/api/auth/verify. The link is not a GET endpoint on purpose: mail scanners fetch links, and a GET would use up the
single-use token before the person clicks it.
"""
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from . import auth, config, mail, store
from .schemas import Account

log = logging.getLogger(__name__)
auth_api = APIRouter(prefix="/api/auth")
LINK_PATH = "/#/verify?token="


class RequestLink(BaseModel):
    email: str = Field(max_length=store.MAX_EMAIL)


class VerifyBody(BaseModel):
    token: str = Field(min_length=1, max_length=200)


def _base_url(request: Request) -> str:
    # PUBLIC_BASE_URL wins. Without it (log mode only; smtp mode refuses to start), the request's own origin is used.
    return config.PUBLIC_BASE_URL or str(request.base_url).rstrip("/")


def _send(email: str, link_url: str) -> None:
    """Runs after the response is sent, so the answer does not depend on the mail server. Failures are logged."""
    try:
        mail.send_login_link(email, link_url)
    except mail.MailError as exc:
        log.error("%s", exc)


def _account_out(account: Account) -> dict:
    return {"id": account.id, "email": account.email, "moderator_of": account.moderator_of}


@auth_api.post("/request-link")
def request_link(body: RequestLink, request: Request, background: BackgroundTasks):
    """Create the account if this is a new address, then mail a single-use sign-in link. The answer is the same
    whether or not the address had an account."""
    try:
        email = store.normalize_email(body.email)
    except ValueError:
        raise HTTPException(422, "invalid email address")
    # Limit per IP and per address (normalized, so case or spacing tricks do not help). The 429 text is the same
    # for every address, so it does not reveal whether an account exists.
    ip = request.client.host if request.client else "unknown"
    wait = auth.link_request_wait(ip, email)
    if wait:
        raise HTTPException(429, f"Too many sign-in link requests. Try again in {wait} seconds.",
                            headers={"Retry-After": str(wait)})
    store.create_account_if_missing(email)
    token = auth.create_magic_link(email)
    background.add_task(_send, email, f"{_base_url(request)}{LINK_PATH}{token}")
    return {"status": "sent", "message": "If the address can receive mail, a sign-in link is on its way."}


@auth_api.post("/verify")
def verify(body: VerifyBody):
    """Use a sign-in link once. Returns the session token the client keeps and sends as Authorization: Bearer."""
    email = auth.consume_magic_link(body.token)
    if email is None:
        raise HTTPException(401, "This sign-in link is invalid or has expired. Request a new one.")
    account = store.create_account_if_missing(email)
    return {"session_token": auth.create_session(account.id), "expires_in": config.AUTH_SESSION_TTL_SEC,
            "account": _account_out(account)}


@auth_api.get("/me")
def me(account: Account = Depends(auth.require_session)):
    """is_owner: the account is an owner (OWNER_EMAILS). blocked: uploads are refused for this account."""
    return {"account": _account_out(account), "is_owner": auth.is_owner(account), "blocked": account.blocked}
