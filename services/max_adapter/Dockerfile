# syntax=docker/dockerfile:1.7
FROM python:3.11-slim-bookworm AS base

RUN groupadd --gid 1001 glossa \
    && useradd --uid 1001 --gid glossa --shell /bin/bash --create-home glossa

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ── Dependencies ──────────────────────────────────────────────────────────────
FROM base AS deps
COPY libs/common /libs/common
RUN --mount=type=cache,target=/root/.cache/pip pip install /libs/common
COPY services/max_adapter/pyproject.toml .
RUN --mount=type=cache,target=/root/.cache/pip pip install ".[test]"

# ── Development ───────────────────────────────────────────────────────────────
FROM deps AS development
COPY services/max_adapter /app
USER glossa
EXPOSE 8005
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005", "--reload"]

# ── Production ────────────────────────────────────────────────────────────────
FROM deps AS production
COPY services/max_adapter /app
RUN chown -R glossa:glossa /app
USER glossa
EXPOSE 8005

ENV MAX_SESSION_BACKEND=tiered \
    MAX_RATE_LIMIT_BACKEND=redis \
    MAX_EVENT_QUEUE_WORKERS=4

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8005/health/live || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005", \
     "--workers", "1", "--loop", "uvloop"]
