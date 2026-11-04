#!/usr/bin/env bash
# Start the backend (FastAPI, app.main:app) on this machine. The backend serves the API and the built frontend.
#
#   scripts/start.sh            start the backend in the foreground (Ctrl-C stops it)
#   scripts/start.sh --build    run `npm ci` and `npm run build` in frontend/ first, then start
#   scripts/start.sh -h         show this help
#
# Settings: the backend reads backend/.env at startup (app/envfile.py). Variables already set in the shell win over
# the file. Environment variables HOST (default 127.0.0.1) and PORT (default 8000) set the listen address.
# Set DATA_DIR in the shell or in backend/.env to keep the data somewhere other than backend/data.
#
# The check below stops the start when backend/.env selects MAIL_BACKEND=smtp without PUBLIC_BASE_URL. The backend
# refuses that combination too, but this message says what to fix.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
ENV_FILE="${ENV_FILE:-$BACKEND_DIR/.env}"

log() { echo "[start] $*"; }
die() { echo "[start] ERROR: $*" >&2; exit 1; }

usage() {
  sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

BUILD=0
while [ $# -gt 0 ]; do
  case "$1" in
    --build) BUILD=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown option: $1" ;;
  esac
  shift
done

# Value of KEY as the backend will see it: the shell variable wins, then the last KEY=VALUE line in the env file.
# Quotes are stripped, a trailing " # comment" is dropped, and an empty value counts as unset (same as envfile.py).
effective() {
  local key="$1" current="${!1-}" line value=""
  if [ -n "$current" ]; then
    printf '%s' "$current"
    return 0
  fi
  [ -f "$ENV_FILE" ] || return 0
  line="$(grep -E "^[[:space:]]*(export[[:space:]]+)?${key}[[:space:]]*=" "$ENV_FILE" | tail -n 1 || true)"
  [ -n "$line" ] || return 0
  value="${line#*=}"
  value="$(printf '%s' "$value" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//')"
  case "$value" in
    \"*\") value="${value#\"}"; value="${value%%\"*}" ;;
    \'*\') value="${value#\'}"; value="${value%%\'*}" ;;
    *) value="$(printf '%s' "$value" | sed -E 's/[[:space:]]+#.*$//; s/[[:space:]]+$//')" ;;
  esac
  printf '%s' "$value"
}

check_mail_settings() {
  local mail smtp_host public_url
  mail="$(effective MAIL_BACKEND)"
  mail="$(printf '%s' "${mail:-log}" | tr '[:upper:]' '[:lower:]')"
  if [ "$mail" = "smtp" ]; then
    smtp_host="$(effective SMTP_HOST)"
    public_url="$(effective PUBLIC_BASE_URL)"
    [ -n "$smtp_host" ] || die "MAIL_BACKEND=smtp needs SMTP_HOST. Set SMTP_HOST in $ENV_FILE, or set MAIL_BACKEND=log for local dev."
    if [ -z "$public_url" ]; then
      die "MAIL_BACKEND=smtp needs PUBLIC_BASE_URL in $ENV_FILE (the origin users open, e.g. https://nav.example.org). Add it, or set MAIL_BACKEND=log for local dev."
    fi
  fi
}

build_frontend() {
  log "Building the frontend (npm ci, then npm run build writes into backend/static)"
  (cd "$FRONTEND_DIR" && npm ci && npm run build)
  [ -f "$BACKEND_DIR/static/index.html" ] || die "the frontend build did not produce backend/static/index.html"
}

main() {
  command -v uv >/dev/null 2>&1 || die "uv not found. Install it: brew install uv (or see https://docs.astral.sh/uv/)"
  [ -f "$BACKEND_DIR/pyproject.toml" ] || die "backend/pyproject.toml not found under $ROOT_DIR"

  if [ -f "$ENV_FILE" ]; then
    log "Reading settings from $ENV_FILE (shell variables win)"
  else
    log "No $ENV_FILE. Using defaults (MAIL_BACKEND=log). Copy backend/.env.example to backend/.env to change them."
  fi
  check_mail_settings

  if [ "$BUILD" = 1 ]; then
    build_frontend
  fi

  local host="${HOST:-127.0.0.1}" port="${PORT:-8000}"
  log "Starting the backend on $host:$port (Ctrl-C stops it)"
  cd "$BACKEND_DIR"
  exec uv run uvicorn app.main:app --host "$host" --port "$port"
}

main
