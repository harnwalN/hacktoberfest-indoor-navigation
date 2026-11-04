#!/usr/bin/env bash
# Start the indoor navigation app and share it through a Cloudflare tunnel.
#
#   scripts/share.sh            quick mode: random https://*.trycloudflare.com address (new on every run)
#   scripts/share.sh --build    also build the frontend into backend/static first
#
# Named mode (stable address on your own domain) is used when TUNNEL_NAME and TUNNEL_HOSTNAME are set,
# in the environment or in scripts/share.env. See scripts/README.md for the one-time setup.
#
# The backend listens on 127.0.0.1 only, not 0.0.0.0. The tunnel connector runs on this same Mac and
# connects to 127.0.0.1, so nothing needs to be reachable from the local network. Binding 0.0.0.0 would
# expose the server to every device on the same Wi-Fi.
#
# The tunnel address is open to anyone who has it. Keep it private and do not share it widely.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
LOG_DIR="$ROOT_DIR/logs"
ENV_FILE="$SCRIPT_DIR/share.env"

BACKEND_HOST="127.0.0.1"
BACKEND_PORT="8000"
HEALTH_URL="http://$BACKEND_HOST:$BACKEND_PORT/api/health"
LOCAL_URL="http://$BACKEND_HOST:$BACKEND_PORT"

BUILD=0
MODE="quick"
OLLAMA_PID=""
BACKEND_PID=""
TAIL_PID=""
WATCH_PID=""
TUNNEL_PID=""

log() { echo "[share] $*"; }
die() { echo "[share] ERROR: $*" >&2; exit 1; }

# Stop every child we started. Runs on any exit, including Ctrl-C (through the INT trap below).
stop_pid() {
  local pid="${1-}"
  [ -n "$pid" ] || return 0
  kill -0 "$pid" 2>/dev/null || return 0
  kill "$pid" 2>/dev/null || true
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 0.5
  done
  kill -9 "$pid" 2>/dev/null || true
}

cleanup() {
  trap - EXIT
  if [ -n "${TUNNEL_PID-}${WATCH_PID-}${BACKEND_PID-}${TAIL_PID-}${OLLAMA_PID-}" ]; then
    echo
    log "Stopping child processes started by this script"
    stop_pid "${TUNNEL_PID-}"
    stop_pid "${WATCH_PID-}"
    stop_pid "${BACKEND_PID-}"
    stop_pid "${TAIL_PID-}"
    stop_pid "${OLLAMA_PID-}"
    log "Stopped. (An Ollama you started elsewhere was not touched.)"
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

usage() {
  cat <<'EOF'
Usage: scripts/share.sh [--build]

  --build   build the frontend (npm install && npm run build) before starting
  -h        show this help

Quick mode (default): random *.trycloudflare.com address, changes on every run.
Named mode: set TUNNEL_NAME and TUNNEL_HOSTNAME in scripts/share.env or the environment.
EOF
}

parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --build) BUILD=1 ;;
      -h|--help) usage; exit 0 ;;
      *) usage >&2; die "unknown option: $1" ;;
    esac
    shift
  done
}

load_env_file() {
  [ -f "$ENV_FILE" ] || return 0
  # Variables already set in the environment win over the file.
  local tn="${TUNNEL_NAME-}" th="${TUNNEL_HOSTNAME-}" oh="${OLLAMA_HOST-}"
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
  [ -n "$tn" ] && TUNNEL_NAME="$tn"
  [ -n "$th" ] && TUNNEL_HOSTNAME="$th"
  [ -n "$oh" ] && OLLAMA_HOST="$oh"
  return 0
}

determine_mode() {
  local tn="${TUNNEL_NAME:-}" th="${TUNNEL_HOSTNAME:-}"
  if [ -n "$tn" ] || [ -n "$th" ]; then
    [ -n "$tn" ] && [ -n "$th" ] || die "named mode needs BOTH TUNNEL_NAME and TUNNEL_HOSTNAME (set one, or neither for quick mode)"
    MODE="named"
  else
    MODE="quick"
  fi
}

check_prereqs() {
  local missing=0
  check_tool() {
    if command -v "$1" >/dev/null 2>&1; then
      echo "  ok       $1"
    else
      echo "  MISSING  $1   install: $2"
      missing=1
    fi
  }
  log "Checking prerequisites"
  check_tool ffmpeg "brew install ffmpeg"
  check_tool uv "brew install uv   (or: curl -LsSf https://astral.sh/uv/install.sh | sh)"
  check_tool cloudflared "brew install cloudflared"
  check_tool ollama "brew install ollama   (or the app from https://ollama.com/download)"
  if [ "$BUILD" = 1 ]; then
    check_tool node "brew install node"
    check_tool npm "brew install node   (npm comes with node)"
  fi
  if [ "$missing" = 1 ]; then
    die "install the missing tools listed above, then run this script again"
  fi
}

build_frontend() {
  log "Building the frontend (npm run build writes into backend/static)"
  (cd "$FRONTEND_DIR" && npm install && npm run build)
  [ -f "$BACKEND_DIR/static/index.html" ] || die "the frontend build did not produce backend/static/index.html"
}

is_local_url() {
  case "$1" in
    *127.0.0.1*|*localhost*|*'[::1]'*) return 0 ;;
    *) return 1 ;;
  esac
}

ollama_up() {
  curl -fsS -m 3 "$OLLAMA_HOST/api/version" >/dev/null 2>&1
}

start_ollama() {
  if ollama_up; then
    log "Ollama is already answering at $OLLAMA_HOST (not started by this script)"
    return 0
  fi
  is_local_url "$OLLAMA_HOST" || die "Ollama is not answering at $OLLAMA_HOST, and that is not a local address, so this script will not start it"
  log "Starting ollama serve (OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL:-4}); log: logs/ollama.log"
  OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-4}" OLLAMA_HOST="$OLLAMA_HOST" \
    ollama serve >>"$LOG_DIR/ollama.log" 2>&1 &
  OLLAMA_PID=$!
  for _ in $(seq 1 60); do
    if ollama_up; then
      log "Ollama is up"
      return 0
    fi
    kill -0 "$OLLAMA_PID" 2>/dev/null || die "ollama serve exited early; see logs/ollama.log"
    sleep 1
  done
  die "Ollama did not answer within 60 seconds; see logs/ollama.log"
}

start_backend() {
  if curl -fsS -m 3 "$HEALTH_URL" >/dev/null 2>&1; then
    die "something already answers at $HEALTH_URL. Stop it first; it may be an older copy of the server with different settings"
  fi
  log "Syncing backend dependencies (uv sync)"
  (cd "$BACKEND_DIR" && uv sync)
  [ -x "$BACKEND_DIR/.venv/bin/uvicorn" ] || die "backend/.venv/bin/uvicorn not found after uv sync"

  # Show backend output in this terminal too.
  touch "$LOG_DIR/backend.log"
  tail -n 0 -F "$LOG_DIR/backend.log" &
  TAIL_PID=$!

  log "Starting the backend on $BACKEND_HOST:$BACKEND_PORT (loopback only; see the comment at the top of this file)"
  (
    cd "$BACKEND_DIR"
    PYTHONUNBUFFERED=1 exec "$BACKEND_DIR/.venv/bin/uvicorn" app.main:app --host "$BACKEND_HOST" --port "$BACKEND_PORT"
  ) >>"$LOG_DIR/backend.log" 2>&1 &
  BACKEND_PID=$!

  for _ in $(seq 1 90); do
    if curl -fsS -m 3 "$HEALTH_URL" >/dev/null 2>&1; then
      log "Backend is up: $HEALTH_URL"
      return 0
    fi
    kill -0 "$BACKEND_PID" 2>/dev/null || die "the backend exited; see logs/backend.log"
    sleep 1
  done
  die "the backend did not answer $HEALTH_URL within 90 seconds; see logs/backend.log"
}

watch_for_quick_url() {
  local url=""
  for _ in $(seq 1 120); do
    url="$(grep -Eo 'https://[A-Za-z0-9-]+\.trycloudflare\.com' "$LOG_DIR/cloudflared.log" 2>/dev/null | head -n 1 || true)"
    if [ -n "$url" ]; then
      echo
      echo "============================================================"
      echo "  Share this address: $url"
      echo "  Quick mode: it changes every time you run this script."
      echo "============================================================"
      return 0
    fi
    sleep 1
  done
}

run_quick_tunnel() {
  log "Starting a quick tunnel to $LOCAL_URL (Ctrl-C stops everything)"
  : >"$LOG_DIR/cloudflared.log"
  watch_for_quick_url &
  WATCH_PID=$!
  cloudflared tunnel --url "$LOCAL_URL" 2>&1 | tee -a "$LOG_DIR/cloudflared.log"
}

run_named_tunnel() {
  if [ ! -f "$HOME/.cloudflared/config.yml" ] && [ ! -f "$HOME/.cloudflared/config.yaml" ]; then
    echo "[share] WARNING: no ~/.cloudflared/config.yml found. Named mode needs the one-time setup in scripts/README.md" >&2
  fi
  log "Starting named tunnel '$TUNNEL_NAME'. Once it connects, the app is at https://$TUNNEL_HOSTNAME (Ctrl-C stops everything)"
  cloudflared tunnel run "$TUNNEL_NAME" 2>&1 | tee -a "$LOG_DIR/cloudflared.log"
}

main() {
  parse_args "$@"
  load_env_file
  determine_mode
  check_prereqs
  mkdir -p "$LOG_DIR"
  if [ "$BUILD" = 1 ]; then
    build_frontend
  fi
  [ -f "$BACKEND_DIR/static/index.html" ] || die "backend/static/index.html is missing. Build the frontend once: scripts/share.sh --build"
  OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
  start_ollama
  start_backend
  if [ "$MODE" = "named" ]; then
    run_named_tunnel
  else
    run_quick_tunnel
  fi
}

main "$@"
