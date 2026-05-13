# syntax=docker/dockerfile:1.7
# ── Base: shared OS packages ──────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS base

RUN groupadd --gid 1001 glossa \
    && useradd --uid 1001 --gid glossa --shell /bin/bash --create-home glossa

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
        git \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TRANSFORMERS_CACHE=/models/hf_cache \
    HF_HOME=/models/hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/models/hf_cache

WORKDIR /app

# ── CPU deps ──────────────────────────────────────────────────────────────────
FROM base AS deps-cpu
COPY libs/common /libs/common
RUN --mount=type=cache,target=/root/.cache/pip pip install /libs/common
COPY services/nlp_service/pyproject.toml .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install ".[quantized]"

# ── GPU deps (CUDA 12.1) ──────────────────────────────────────────────────────
FROM base AS deps-gpu
COPY libs/common /libs/common
RUN --mount=type=cache,target=/root/.cache/pip pip install /libs/common
COPY services/nlp_service/pyproject.toml .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch --index-url https://download.pytorch.org/whl/cu121
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install ".[quantized,gpu]"

# ── Development ───────────────────────────────────────────────────────────────
FROM deps-cpu AS development
COPY services/nlp_service /app
USER glossa
EXPOSE 8003
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003", "--reload"]

# ── Production CPU ────────────────────────────────────────────────────────────
FROM deps-cpu AS production
COPY services/nlp_service /app
RUN chown -R glossa:glossa /app
USER glossa
EXPOSE 8003
HEALTHCHECK --interval=20s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8003/health/live || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003", \
     "--workers", "1", "--loop", "uvloop"]

# ── Production GPU ────────────────────────────────────────────────────────────
FROM deps-gpu AS production-gpu
COPY services/nlp_service /app
RUN chown -R glossa:glossa /app
USER glossa
EXPOSE 8003
ENV NLP_DEVICE=cuda
HEALTHCHECK --interval=20s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8003/health/live || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003", \
     "--workers", "1", "--loop", "uvloop"]
