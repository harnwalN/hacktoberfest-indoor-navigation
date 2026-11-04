# Build: docker build -f deploy/Dockerfile --build-arg OCR_EXTRA=1 backend   (or use docker-compose.yml)
FROM python:3.12-slim

# 1 = install linux-ocr extra (RapidOCR). 0 = no OCR on Linux.
ARG OCR_EXTRA=0
ARG UV_VERSION=0.8.22

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never PATH=/app/.venv/bin:$PATH DATA_DIR=/data BACKEND_HOST=0.0.0.0

COPY --from=ghcr.io/astral-sh/uv:${UV_VERSION} /uv /usr/local/bin/uv

# ffmpeg: video. libgl1/libglib2.0-0: OpenCV runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN EXTRA=""; if [ "$OCR_EXTRA" = "1" ]; then EXTRA="--extra linux-ocr"; fi \
 && uv sync --frozen --no-dev --no-install-project $EXTRA

COPY app ./app
COPY static ./static

# Non-root user. /data is the data volume.
RUN groupadd --system app \
 && useradd --system --gid app --home-dir /app --shell /usr/sbin/nologin app \
 && mkdir -p /data && chown app:app /data
USER app

VOLUME ["/data"]
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn app.main:app --host \"$BACKEND_HOST\" --port 8000"]
