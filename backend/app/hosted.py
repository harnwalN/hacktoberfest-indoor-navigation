"""Hosted vision model: Google Gemini through its REST API (plain httpx, no SDK).

A walk uses this model when gemma.choose_model picks it: VISION_PROVIDER=auto with GEMINI_API_KEY set and the API
reachable, VISION_PROVIDER=hosted, or an explicit model name starting with "gemini". Its answers go through the same
prompts, the same JSON parse and the same OCR grounding as Gemma's answers, and are cached under the model name, so
answers from different models never mix. Nothing here makes a hosted answer more trusted than a Gemma one.

- Key: GEMINI_API_KEY from the environment, sent only in the x-goog-api-key header. It never goes into a URL, a log
  line, an exception message or /api/health.
- Schema: pydantic's JSON schema is reduced to the subset that responseJsonSchema accepts (response_schema below).
- Cost guard: at most HOSTED_MAX_CALLS_PER_DAY generateContent requests per UTC day, counted in
  DATA_DIR/hosted_usage.json (persisted, shared by every walk on this server). A counter that cannot be read or
  written stops hosted calls instead of letting the count drift.

This module only talks HTTP. The retry policy and the GemmaUnavailable mapping live in gemma._chat_hosted.
"""
import base64
import json
import logging
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from . import config

log = logging.getLogger(__name__)

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_MODEL_NAME = re.compile(r"gemini-[A-Za-z0-9][A-Za-z0-9.\-]*")  # the name goes into the URL path: plain names only
USAGE_FILE = "hosted_usage.json"
DEFAULT_WAIT_SEC = 1.0  # wait before the one retry when the server gave no Retry-After
NO_KEY = ("GEMINI_API_KEY is not set, so the hosted vision model cannot be used. Set the key, or use "
          "VISION_PROVIDER=gemma / --model gemma3:4b for local Gemma.")


class HostedError(RuntimeError):
    """A hosted call failed. `retry`: worth one more attempt (network, 429, 5xx, unreadable answer).
    `retry_after`: the seconds the server asked us to wait, when it said."""

    def __init__(self, message: str, *, retry: bool = False, retry_after: float | None = None):
        super().__init__(message)
        self.retry = retry
        self.retry_after = retry_after


@dataclass
class Reply:
    text: str  # the model's JSON text
    finish: str  # Gemini finishReason: "STOP" or "MAX_TOKENS" (other reasons raise HostedError)
    input_tokens: int = 0  # usageMetadata.promptTokenCount, 0 when the response has none


def is_hosted(model: str | None) -> bool:
    """Model names starting with "gemini" are hosted; everything else is a local Ollama model."""
    return bool(model) and model.startswith("gemini")


def configured() -> bool:
    return bool(config.GEMINI_API_KEY)


def require_key() -> None:
    if not configured():
        raise HostedError(NO_KEY)


def _client(timeout: float) -> httpx.Client:
    """Module-level factory (tests replace it with an httpx.MockTransport)."""
    return httpx.Client(timeout=timeout)


def sleep(seconds: float) -> None:
    """Module-level so tests can skip the wait before a retry."""
    time.sleep(seconds)


def _scrub(text: str) -> str:
    key = config.GEMINI_API_KEY
    return text.replace(key, "[GEMINI_API_KEY]") if key else text


def _model_url(model: str) -> str:
    if not _MODEL_NAME.fullmatch(model or ""):
        raise HostedError(f"{model!r} is not a Gemini model name (for example gemini-2.5-flash-lite)")
    return f"{BASE_URL}/models/{model}"


def _headers() -> dict:
    require_key()
    return {"x-goog-api-key": config.GEMINI_API_KEY}


def _error_text(resp: httpx.Response) -> str:
    try:
        msg = resp.json()["error"]["message"]
    except (ValueError, KeyError, TypeError):
        msg = resp.text
    return _scrub(str(msg))[:300]


def _retry_after(resp: httpx.Response) -> float | None:
    raw = resp.headers.get("retry-after")
    try:
        return max(0.0, float(raw)) if raw is not None else None  # seconds; an HTTP-date is ignored
    except ValueError:
        return None


# ---------------------------------------------------------------- response schema
# Keywords Gemini's responseJsonSchema accepts (its OpenAPI-style subset). Anything else (title, default, $defs,
# additionalProperties ...) is dropped; $ref is inlined.
_KEEP = {"type", "format", "description", "enum", "required", "minimum", "maximum", "minItems", "maxItems", "nullable"}


def response_schema(schema: dict) -> dict:
    """pydantic's JSON schema (gemma._schema: every field required) in the form Gemini accepts.

    Why this and not the raw schema: pydantic adds titles and defaults (not in Gemini's subset) and may emit $defs
    and anyOf-with-null for Optional fields. Here references are inlined, `anyOf [T, null]` becomes T + nullable,
    and `propertyOrdering` is set to the field order, so the evidence fields (texts, shared_details ...) are generated
    before the verdicts, as Gemma's schema does. The answer is still parsed and validated by pydantic in gemma.py."""
    defs = {**schema.get("$defs", {}), **schema.get("definitions", {})}
    return _convert(schema, defs, 0)


def _convert(node: dict, defs: dict, depth: int) -> dict:
    if depth > 20:
        raise ValueError("schema nested too deeply for Gemini")
    if "$ref" in node:
        name = node["$ref"].rsplit("/", 1)[-1]
        if name not in defs:
            raise ValueError(f"unknown schema reference {name!r}")
        return _convert({**defs[name], **{k: v for k, v in node.items() if k != "$ref"}}, defs, depth + 1)
    out: dict = {}
    if "anyOf" in node:
        kept = [o for o in node["anyOf"] if o.get("type") != "null"]
        if len(kept) == 1:  # Optional[T]: T, nullable, the field's own description kept
            merged = {**kept[0], **{k: v for k, v in node.items() if k != "anyOf"}}
            out = _convert(merged, defs, depth + 1)
            if len(kept) < len(node["anyOf"]):
                out["nullable"] = True
            return out
        if len(kept) < len(node["anyOf"]):
            out["nullable"] = True
        out["anyOf"] = [_convert(o, defs, depth + 1) for o in kept]
    for key, val in node.items():
        if key == "anyOf":
            continue
        if key == "const":  # a one-value Literal
            out["enum"] = [val]
        elif key == "properties":
            out["properties"] = {name: _convert(sub, defs, depth + 1) for name, sub in val.items()}
            out["propertyOrdering"] = list(val)
        elif key == "items" and isinstance(val, dict):
            out["items"] = _convert(val, defs, depth + 1)
        elif key == "type" and isinstance(val, list):  # ["string", "null"]
            kinds = [t for t in val if t != "null"]
            if len(kinds) < len(val):
                out["nullable"] = True
            if kinds:
                out["type"] = kinds[0]
        elif key in _KEEP:
            out[key] = val
    return out


def _thinking(model: str) -> dict | None:
    """Gemini 2.5 Flash models run with thinking off: the answer is a short JSON form, and thinking tokens would
    count against maxOutputTokens (num_predict) and could cut the JSON short."""
    return {"thinkingBudget": 0} if model.startswith("gemini-2.5-flash") else None


# ---------------------------------------------------------------- one request
def generate(model: str, prompt: str, images: list[bytes], schema: dict, num_predict: int) -> Reply:
    """One generateContent request, no retry here (gemma._chat_hosted decides). Counts against the daily limit.
    Temperature 0, JSON output in the given schema, images inline as base64 JPEG."""
    url = _model_url(model) + ":generateContent"
    headers = _headers()
    gen = {"temperature": 0, "maxOutputTokens": num_predict, "responseMimeType": "application/json",
           "responseJsonSchema": schema}
    if (thinking := _thinking(model)) is not None:
        gen["thinkingConfig"] = thinking
    parts = [{"text": prompt}] + [
        {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(img).decode("ascii")}} for img in images]
    body = {"contents": [{"role": "user", "parts": parts}], "generationConfig": gen}
    _reserve_call()
    try:
        with _client(config.HOSTED_TIMEOUT_SEC) as http:
            resp = http.post(url, json=body, headers=headers)
    except httpx.TransportError as exc:
        if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
            _remember(model, False, f"not reachable ({type(exc).__name__})")
        raise HostedError(f"Gemini not reachable: {type(exc).__name__}: {_scrub(str(exc))}", retry=True) from exc
    except httpx.HTTPError as exc:  # any other httpx failure: not worth a retry
        raise HostedError(f"Gemini request failed: {type(exc).__name__}: {_scrub(str(exc))}") from exc
    code = resp.status_code
    if code == 429 or code >= 500:
        raise HostedError(f"Gemini HTTP {code}: {_error_text(resp)}", retry=True, retry_after=_retry_after(resp))
    if code == 404:
        raise HostedError(f"Gemini has no model {model!r} for this key: check HOSTED_MODEL")
    if code != 200:
        raise HostedError(f"Gemini rejected the request ({code}): {_error_text(resp)}")
    try:
        data = resp.json()
    except ValueError as exc:
        raise HostedError("Gemini answered with something that is not JSON", retry=True) from exc
    if not isinstance(data, dict):
        raise HostedError("Gemini answered with an unexpected structure", retry=True)
    tokens = _input_tokens(data)
    _add_tokens(tokens)
    candidates = data.get("candidates") or []
    if not candidates:
        why = (data.get("promptFeedback") or {}).get("blockReason") or "no candidates"
        raise HostedError(f"Gemini returned no answer (prompt blocked: {why})")
    cand = candidates[0]
    finish = cand.get("finishReason") or "STOP"
    if finish == "MAX_TOKENS":
        return Reply("", finish, tokens)  # gemma._chat_hosted reports the truncation
    if finish != "STOP":  # SAFETY, RECITATION, OTHER ...: not used, no retry (the same frame gets the same block)
        raise HostedError(f"Gemini stopped the answer ({finish}), so it is not used")
    parts = (cand.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and not p.get("thought"))
    return Reply(text, finish, tokens)


def _input_tokens(data: dict) -> int:
    meta = data.get("usageMetadata") or {}
    try:
        return max(0, int(meta.get("promptTokenCount") or 0))
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------- reachability (cached)
_checks: dict[str, tuple[float, bool, str]] = {}  # model -> (monotonic time, usable, detail)
_checks_lock = threading.Lock()


def _remember(model: str, ok: bool, detail: str) -> None:
    with _checks_lock:
        _checks[model] = (time.monotonic(), ok, detail)


def _probe(model: str) -> tuple[bool, str]:
    try:
        url = _model_url(model)
        with _client(config.HOSTED_CHECK_TIMEOUT_SEC) as http:
            resp = http.get(url, headers=_headers())
    except HostedError as exc:
        return False, str(exc)
    except httpx.HTTPError as exc:  # no connection, timeout, ...
        return False, f"not reachable ({type(exc).__name__})"
    if resp.status_code == 200:
        return True, "ok"
    return False, f"HTTP {resp.status_code}: {_error_text(resp)}"


def status(model: str | None = None) -> tuple[bool, str]:
    """(usable, detail) for a hosted model: one GET of the model's metadata, reused for HOSTED_CHECK_TTL_SEC.
    No key: (False, ...) without a request. A failed call (connection refused) also marks the model down here."""
    model = model or config.HOSTED_MODEL
    if not configured():
        return False, NO_KEY
    with _checks_lock:
        hit = _checks.get(model)
    if hit is not None and time.monotonic() - hit[0] < config.HOSTED_CHECK_TTL_SEC:
        return hit[1], hit[2]
    ok, detail = _probe(model)
    _remember(model, ok, detail)
    if not ok:
        log.warning("hosted model %s not usable: %s", model, detail)
    return ok, detail


def check(model: str) -> None:
    """Raise HostedError unless `model` can be asked now: key set, daily calls left, API usable."""
    require_key()
    if cap_reached():
        raise HostedError(cap_message(calls_today()))
    ok, detail = status(model)
    if not ok:
        raise HostedError(f"hosted model {model!r} is not usable: {detail}")


def auto_ready(model: str | None = None) -> bool:
    """What `auto` needs before it picks the hosted model for a new walk: key, daily calls left, usable API."""
    if not configured():
        return False
    try:
        return not cap_reached() and status(model)[0]
    except HostedError as exc:  # unreadable call counter: no hosted calls
        log.warning("%s", exc)
        return False


# ---------------------------------------------------------------- daily cost guard
_usage_lock = threading.Lock()


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


def _usage_path() -> Path:
    return Path(config.DATA_DIR) / USAGE_FILE


def _read_usage() -> dict:
    """Today's counters (UTC). Caller holds _usage_lock. A missing file is an empty day; an unreadable one raises
    HostedError, because the limit cannot be checked and no hosted call may be made without it."""
    today = _today()
    path = _usage_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError) as exc:
        raise HostedError(f"cannot read the Gemini call counter {path} ({exc}); hosted calls are off until it is "
                          "fixed") from exc
    if not isinstance(data, dict) or data.get("day") != today:
        data = {}
    try:
        return {"day": today, "calls": int(data.get("calls", 0)), "input_tokens": int(data.get("input_tokens", 0))}
    except (TypeError, ValueError) as exc:
        raise HostedError(f"the Gemini call counter {path} is malformed; hosted calls are off until it is fixed") \
            from exc


def _write_usage(data: dict) -> None:
    """Atomic replace of the counter file. Caller holds _usage_lock."""
    path = _usage_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
    except OSError as exc:
        raise HostedError(f"cannot write the Gemini call counter {path} ({exc}); hosted calls are off until it is "
                          "fixed") from exc


def _reserve_call() -> None:
    """Count one request before it is sent; refuse it when the day's limit is used up."""
    with _usage_lock:
        data = _read_usage()
        if data["calls"] >= config.HOSTED_MAX_CALLS_PER_DAY:
            raise HostedError(cap_message(data["calls"]))
        data["calls"] += 1
        _write_usage(data)


def _add_tokens(n: int) -> None:
    """Approximate input tokens of an answer (best effort: a failure here must not lose the answer)."""
    if not n:
        return
    try:
        with _usage_lock:
            data = _read_usage()
            data["input_tokens"] += n
            _write_usage(data)
    except HostedError as exc:
        log.warning("input tokens not recorded: %s", exc)


def cap_message(calls: int) -> str:
    return (f"Gemini daily call limit reached ({calls}/{config.HOSTED_MAX_CALLS_PER_DAY} calls today, UTC; "
            "HOSTED_MAX_CALLS_PER_DAY): cached answers only, hosted calls resume tomorrow (UTC)")


def cap_reached() -> bool:
    return calls_today() >= config.HOSTED_MAX_CALLS_PER_DAY


def calls_today() -> int:
    with _usage_lock:
        return _read_usage()["calls"]


def usage_snapshot() -> tuple[int, int]:
    """(calls, input tokens) so far today; a walk logs the difference to its start (see usage_summary)."""
    try:
        with _usage_lock:
            data = _read_usage()
    except HostedError as exc:
        log.warning("%s", exc)
        return 0, 0
    return data["calls"], data["input_tokens"]


def usage_summary(mark: tuple[int, int]) -> str:
    calls, tokens = usage_snapshot()
    return (f"{max(calls - mark[0], 0)} call(s), ~{max(tokens - mark[1], 0)} input tokens since the walk started; "
            f"{calls}/{config.HOSTED_MAX_CALLS_PER_DAY} calls today (UTC)")


def health() -> dict:
    """The `hosted` block of /api/health. Never contains the key."""
    configured_ = configured()
    try:
        calls = calls_today()
    except HostedError as exc:
        log.warning("%s", exc)
        calls = config.HOSTED_MAX_CALLS_PER_DAY  # unreadable counter: report the limit as used
    return {"configured": configured_, "reachable": status()[0] if configured_ else None,
            "model": config.HOSTED_MODEL, "calls_today": calls, "limit": config.HOSTED_MAX_CALLS_PER_DAY}
