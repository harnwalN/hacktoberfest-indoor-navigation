"""Reads KEY=VALUE lines from backend/.env into os.environ, once, at import of config.py.

No dependency (no python-dotenv). Rules:
- a variable already set in the process environment wins over the file;
- an empty value in the file is ignored (same as leaving the line out), so a blank template line never overrides a default;
- lines starting with # and blank lines are skipped; an optional leading "export " is allowed;
- values may be quoted with ' or "; unquoted values lose a trailing " # comment".
Values are never logged. Only the names of the variables that were loaded are logged.
"""
import logging
import os
import re
from pathlib import Path

log = logging.getLogger(__name__)
_LINE = re.compile(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)")


def parse(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.fullmatch(line)
        if m is None:
            continue  # not a KEY=VALUE line: ignored, like a comment
        key, value = m.group(1), m.group(2).strip()
        if value[:1] in ("'", '"'):
            end = value.find(value[0], 1)
            if end == -1:
                continue  # unterminated quote: skip the line rather than guess
            value = value[1:end]
        else:
            value = re.sub(r"\s+#.*$", "", value).strip()
        if value:
            values[key] = value
    return values


def _decode_lines(data: bytes, name: str) -> str:
    """The file as text, line by line. A line that is not UTF-8 is skipped with a warning (its number only, never its
    content), so one stray byte in a comment does not switch off the other settings."""
    if data.startswith(b"\xef\xbb\xbf"):  # a byte order mark from some editors
        data = data[3:]
    lines = []
    for number, raw in enumerate(data.split(b"\n"), start=1):
        try:
            lines.append(raw.decode("utf-8"))
        except UnicodeDecodeError:
            log.warning("%s line %d is not UTF-8 text and is ignored", name, number)
            lines.append("")
    return "\n".join(lines)


def load(path: str | Path) -> list[str]:
    """Set the variables from the file that are not already in the environment. Returns the names set.
    An empty path or a missing file loads nothing. A file that cannot be read loads nothing, with a warning."""
    if not path:
        return []
    file = Path(path)
    if not file.is_file():
        return []
    try:
        data = file.read_bytes()
    except OSError as exc:
        log.warning("%s cannot be read (%s): no settings loaded from it", file.name, type(exc).__name__)
        return []
    loaded = []
    for key, value in parse(_decode_lines(data, file.name)).items():
        if key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    if loaded:
        # WARNING, not INFO: uvicorn shows warnings by default, and operators need to see which file was read.
        log.warning("loaded %d setting(s) from %s: %s", len(loaded), file.name, ", ".join(sorted(loaded)))
    return loaded
