"""Sign-in mail: send_login_link(email, link_url). Settings are in config.py (MAIL_BACKEND, SMTP_*).

log mode: writes the link to the log and sends nothing. Local development only (config.py).
smtp mode: sends one message through SMTP_HOST with the standard library (smtplib, email); no new dependency.
Errors raise MailError with the exception type only: the password and the server's reply are not put into it.
"""
import logging
import smtplib
import ssl
from email.message import EmailMessage

from . import config

log = logging.getLogger(__name__)
SMTP_TIMEOUT_SEC = 20


class MailError(RuntimeError):
    """The sign-in mail could not be sent."""


def send_login_link(email: str, link_url: str) -> None:
    if config.MAIL_BACKEND == "log":
        log.warning("sign-in link for %s (MAIL_BACKEND=log, not emailed): %s", email, link_url)
        return
    _send_smtp(email, link_url)


def _message(email: str, link_url: str) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = "Your sign-in link"
    msg["From"] = config.SMTP_FROM
    msg["To"] = email
    msg.set_content(
        "Open this link to sign in. It works once and expires in "
        f"{config.AUTH_LINK_TTL_SEC // 60} minutes:\n\n{link_url}\n\n"
        "If you did not ask for this, you can ignore this email.\n"
    )
    return msg


def _send_smtp(email: str, link_url: str) -> None:
    if not config.SMTP_FROM:
        raise MailError("SMTP_FROM is not set")
    try:
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=SMTP_TIMEOUT_SEC) as smtp:
            smtp.ehlo()
            if config.SMTP_STARTTLS:
                smtp.starttls(context=ssl.create_default_context())
                smtp.ehlo()
            if config.SMTP_USER:
                smtp.login(config.SMTP_USER, config.SMTP_PASSWORD)
            smtp.send_message(_message(email, link_url))
    except (smtplib.SMTPException, OSError) as exc:
        raise MailError(f"sending sign-in mail failed: {type(exc).__name__}") from exc
