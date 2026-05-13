# syntax=docker/dockerfile:1.7
ARG PYTHON_VERSION=3.11

# ── base ──────────────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim-bookworm AS base

RUN groupadd --gid 1001 glossa \
 && useradd --uid 1001 --gid glossa --shell /bin/bash --create-home glossa

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ── deps (CPU) ────────────────────────────────────────────────────────────────
FROM base AS deps

COPY libs/common /libs/common
RUN --mount=type=cache,target=/root/.cache/pip pip install /libs/common

COPY services/asr_service/pyproject.toml .
RUN --mount=type=cache,target=/root/.cache/pip pip install ".[silero]"

# ── development ───────────────────────────────────────────────────────────────
FROM deps AS development

RUN --mount=type=cache,target=/root/.cache/pip pip install ".[dev]"
COPY services/asr_service /app

EXPOSE 8002
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]

# ── production ────────────────────────────────────────────────────────────────
FROM deps AS production

COPY services/asr_service /app
RUN chown -R glossa:glossa /app
USER glossa

EXPOSE 8002

HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8002/health/live || exit 1

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8002", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--http", "httptools"]

# ── production-gpu ────────────────────────────────────────────────────────────
FROM base AS deps-gpu

COPY libs/common /libs/common
RUN --mount=type=cache,target=/root/.cache/pip pip install /libs/common

COPY services/asr_service/pyproject.toml .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install ".[gpu,silero]" \
    && pip install --upgrade ctranslate2

FROM deps-gpu AS production-gpu

COPY services/asr_service /app
RUN chown -R glossa:glossa /app
USER glossa

ENV ASR_DEVICE=cuda \
    ASR_COMPUTE_TYPE=float16

EXPOSE 8002

HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8002/health/live || exit 1

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8002", \
     "--workers", "1", \
     "--loop", "uvloop"]
