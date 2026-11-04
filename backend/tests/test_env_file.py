"""backend/.env loading (app/envfile.py, read by app/config.py at import) and the owner list defaults.

The parser is checked directly. The config behaviour is checked in a child process, because config reads its settings
once, at import: the child gets its own ENV_FILE and its own environment, so nothing here touches this process."""
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from app import envfile

BACKEND = Path(__file__).resolve().parents[1]


# ---------- the parser ----------
def test_parse_reads_plain_quoted_and_export_lines():
    text = "\n".join([
        "# a comment",
        "",
        "PLAIN=value",
        "export EXPORTED=yes",
        'DOUBLE="quoted value"',
        "SINGLE='single quoted'",
        "INLINE=value   # trailing comment",
        "HASH=abc#notacomment",
        "  SPACED = spaced  ",
    ])
    assert envfile.parse(text) == {
        "PLAIN": "value", "EXPORTED": "yes", "DOUBLE": "quoted value", "SINGLE": "single quoted",
        "INLINE": "value", "HASH": "abc#notacomment", "SPACED": "spaced",
    }


def test_parse_ignores_empty_values_and_bad_lines():
    text = "EMPTY=\nBLANKQ=\"\"\nnot a setting\n1BAD=x\nOPEN=\"unterminated\nOK=1\n"
    assert envfile.parse(text) == {"OK": "1"}


def test_load_sets_only_variables_missing_from_the_environment(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("ENVFILE_TEST_NEW=from-file\nENVFILE_TEST_SET=from-file\nENVFILE_TEST_BLANK=\n")
    monkeypatch.setenv("ENVFILE_TEST_SET", "from-shell")
    monkeypatch.delenv("ENVFILE_TEST_NEW", raising=False)
    monkeypatch.delenv("ENVFILE_TEST_BLANK", raising=False)
    assert envfile.load(f) == ["ENVFILE_TEST_NEW"]
    assert os.environ["ENVFILE_TEST_NEW"] == "from-file"
    assert os.environ["ENVFILE_TEST_SET"] == "from-shell"  # the shell wins
    assert "ENVFILE_TEST_BLANK" not in os.environ  # an empty line is the same as no line
    monkeypatch.delenv("ENVFILE_TEST_NEW", raising=False)


def test_missing_or_empty_path_loads_nothing(tmp_path):
    assert envfile.load(tmp_path / "absent.env") == []
    assert envfile.load("") == []


# ---------- config, in a child process ----------
def _config_in_child(env_file_text: str, extra_env: dict, tmp_path) -> dict:
    f = tmp_path / "child.env"
    f.write_text(env_file_text)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OWNER_", "PUBLIC_", "SMTP_", "MAIL_", "AUTH_"))}
    env.update({"ENV_FILE": str(f), "DATA_DIR": str(tmp_path / "data")}, **extra_env)
    code = ("import json; from app import config as c; print(json.dumps({"
            "'owners': sorted(c.OWNER_EMAILS), 'token': c.OWNER_TOKEN, 'base': c.PUBLIC_BASE_URL, "
            "'mail': c.MAIL_BACKEND, 'ttl': c.AUTH_LINK_TTL_SEC}))")
    out = subprocess.run([sys.executable, "-c", code], cwd=BACKEND, env=env, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_config_reads_owner_token_base_url_and_mail_from_the_file(tmp_path):
    got = _config_in_child(
        "OWNER_EMAILS=Boss@Example.com, second@example.org\nOWNER_TOKEN=from-file-token\n"
        "PUBLIC_BASE_URL=https://nav.example.org/\nMAIL_BACKEND=log\nAUTH_LINK_TTL_SEC=600\n",
        {}, tmp_path)
    assert got == {"owners": ["boss@example.com", "second@example.org"], "token": "from-file-token",
                   "base": "https://nav.example.org", "mail": "log", "ttl": 600}


def test_shell_environment_wins_over_the_file(tmp_path):
    got = _config_in_child("OWNER_TOKEN=from-file\nOWNER_EMAILS=file@example.com\n",
                           {"OWNER_TOKEN": "from-shell", "OWNER_EMAILS": "shell@example.com"}, tmp_path)
    assert got["token"] == "from-shell" and got["owners"] == ["shell@example.com"]


def test_blank_owner_list_means_nobody_is_owner(tmp_path):
    got = _config_in_child("OWNER_EMAILS=\n", {}, tmp_path)
    assert got["owners"] == []
    got = _config_in_child("", {"OWNER_EMAILS": "   "}, tmp_path)
    assert got["owners"] == []


def test_owner_list_entries_must_be_email_addresses(tmp_path):
    f = tmp_path / "bad.env"
    f.write_text("OWNER_EMAILS=not-an-address\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("OWNER_")}
    env.update({"ENV_FILE": str(f), "DATA_DIR": str(tmp_path / "data")})
    out = subprocess.run([sys.executable, "-c", "import app.config"], cwd=BACKEND, env=env, capture_output=True, text=True)
    assert out.returncode != 0 and "OWNER_EMAILS must be" in out.stderr


def test_empty_env_file_setting_turns_the_file_off(tmp_path):
    got = _config_in_child("OWNER_TOKEN=should-not-load\n", {"ENV_FILE": ""}, tmp_path)
    assert got["token"] == ""


# ---------- files that are not clean UTF-8, and what the operator sees ----------
def test_non_utf8_line_is_skipped_and_the_other_settings_load(tmp_path, monkeypatch, caplog):
    f = tmp_path / ".env"
    f.write_bytes(b"# caf\xe9 note\nENVFILE_TEST_GOOD=yes\nENVFILE_TEST_BAD=\xff\xfe\n")
    for name in ("ENVFILE_TEST_GOOD", "ENVFILE_TEST_BAD"):
        monkeypatch.setenv(name, "x")  # so the teardown removes whatever load() sets
        monkeypatch.delenv(name)
    with caplog.at_level(logging.WARNING, logger="app.envfile"):
        assert envfile.load(f) == ["ENVFILE_TEST_GOOD"]
    assert os.environ["ENVFILE_TEST_GOOD"] == "yes"
    assert "ENVFILE_TEST_BAD" not in os.environ
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("line 1" in r.getMessage() for r in warnings)  # the line number, not the content
    assert any("line 3" in r.getMessage() for r in warnings)
    assert any(r.getMessage().startswith("loaded 1 setting(s)") for r in warnings)  # visible at WARNING


def test_byte_order_mark_does_not_hide_the_first_setting(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_bytes(b"\xef\xbb\xbfENVFILE_TEST_BOM=first\n")
    monkeypatch.setenv("ENVFILE_TEST_BOM", "x")
    monkeypatch.delenv("ENVFILE_TEST_BOM")
    assert envfile.load(f) == ["ENVFILE_TEST_BOM"]


def test_unreadable_file_loads_nothing_and_warns(tmp_path, monkeypatch, caplog):
    f = tmp_path / ".env"
    f.write_text("ENVFILE_TEST_UNREAD=1\n")

    def denied(self):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(envfile.Path, "read_bytes", denied)
    with caplog.at_level(logging.WARNING, logger="app.envfile"):
        assert envfile.load(f) == []
    assert "ENVFILE_TEST_UNREAD" not in os.environ
    assert any(r.levelno == logging.WARNING and "cannot be read" in r.getMessage() for r in caplog.records)
