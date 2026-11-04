# Deploy (backend container)

Status: **deploy step waiting for owner approval.** Nothing here has been deployed or pushed.

Run all commands from the repo root. Settings come from `backend/.env` (not committed).
Optional: `OCR_EXTRA=1` (RapidOCR on Linux), `DATA_HOST_DIR` (default `backend/data`), `OLLAMA_HOST`.

```sh
# Build
docker compose -f deploy/docker-compose.yml build

# Run
docker compose -f deploy/docker-compose.yml up -d
docker compose -f deploy/docker-compose.yml logs -f backend
docker compose -f deploy/docker-compose.yml down        # data stays on the host

# Health (port is published on 127.0.0.1 only)
curl -fsS http://127.0.0.1:8000/api/health
docker compose -f deploy/docker-compose.yml ps          # healthcheck: healthy / unhealthy

# Back up (keeps newest KEEP_N archives, default 7)
DATA_DIR=backend/data BACKUP_DIR=backups KEEP_N=7 deploy/backup.sh

# Restore (replaces the data folder; stop the service first)
docker compose -f deploy/docker-compose.yml down
python3 scripts/migrate_data.py import backups/data-<timestamp>.tar.gz backend/data --force
docker compose -f deploy/docker-compose.yml up -d
curl -fsS http://127.0.0.1:8000/api/health

# Deploy to a server: WAITING FOR OWNER APPROVAL. Owner command, to run later:
docker compose -f deploy/docker-compose.yml up -d --build
```

Notes:
- `gemma.available` false in health means the container cannot reach Ollama. Check `OLLAMA_HOST`.
- For a consistent backup, run `down` first, back up, then `up -d`.
- Backend listens on `127.0.0.1` by default (`BACKEND_HOST`); only the container uses `0.0.0.0`. Put a tunnel or reverse proxy in front.
- `backend/.dockerignore` keeps `.env`, `data/` and tests out of the image.
