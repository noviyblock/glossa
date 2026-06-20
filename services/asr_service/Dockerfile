# syntax=docker/dockerfile:1.7
FROM python:3.11-slim-bookworm AS base

RUN groupadd --gid 1001 glossa \
 && useradd --uid 1001 --gid glossa --shell /bin/bash --create-home glossa

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ── CPU deps ──────────────────────────────────────────────────────────────────
FROM base AS deps
COPY pyproject.toml .
RUN --mount=type=cache,target=/root/.cache/pip pip install .

# ── Development ───────────────────────────────────────────────────────────────
FROM deps AS development
COPY . .
EXPOSE 8002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]

# ── Production ────────────────────────────────────────────────────────────────
FROM deps AS production
COPY . .
RUN chown -R glossa:glossa /app
USER glossa
EXPOSE 8002
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8002/health/live || exit 1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002", \
     "--workers", "1", "--loop", "uvloop"]

# ── Production GPU ────────────────────────────────────────────────────────────
FROM base AS deps-gpu
COPY pyproject.toml .
RUN --mount=type=cache,target=/root/.cache/pip pip install ".[gpu]"

FROM deps-gpu AS production-gpu
COPY . .
RUN chown -R glossa:glossa /app
USER glossa
ENV ASR_DEVICE=cuda ASR_COMPUTE_TYPE=float16
EXPOSE 8002
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8002/health/live || exit 1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002", \
     "--workers", "1", "--loop", "uvloop"]
