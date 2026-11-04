#!/usr/bin/env bash
# Archive DATA_DIR to backups/data-<UTC>.tar.gz and keep the newest KEEP_N (default 7).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT/backend/data}"
BACKUP_DIR="${BACKUP_DIR:-$ROOT/backups}"
KEEP_N="${KEEP_N:-7}"
MIGRATE="$ROOT/scripts/migrate_data.py"

case "$KEEP_N" in ''|*[!0-9]*|0) echo "KEEP_N must be a whole number >= 1 (got '$KEEP_N')" >&2; exit 1 ;; esac
[ -d "$DATA_DIR" ] || { echo "DATA_DIR not found: $DATA_DIR" >&2; exit 1; }
mkdir -p "$BACKUP_DIR"
OUT="$BACKUP_DIR/data-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"

if [ -f "$MIGRATE" ]; then
  python3 "$MIGRATE" export "$DATA_DIR" "$OUT"
else
  echo "scripts/migrate_data.py not found; using plain tar" >&2
  tar -czf "$OUT" -C "$(dirname "$DATA_DIR")" "$(basename "$DATA_DIR")"
fi
echo "wrote $OUT"

ls -1 "$BACKUP_DIR"/data-*.tar.gz | sort -r | tail -n +"$((KEEP_N + 1))" | while IFS= read -r old; do
  rm -f -- "$old" && echo "pruned $old"
done
