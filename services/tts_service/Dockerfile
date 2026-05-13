# syntax=docker/dockerfile:1.7
# ── Base: OS packages shared by all stages ────────────────────────────────────
FROM python:3.11-slim-bookworm AS base

RUN groupadd --gid 1001 glossa \
    && useradd --uid 1001 --gid glossa --shell /bin/bash --create-home glossa

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
        libsndfile1 \
        libvorbis0a \
        libogg0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_HOME=/models/torch_cache \
    HF_HOME=/models/hf_cache

WORKDIR /app

# ── CPU deps ──────────────────────────────────────────────────────────────────
FROM base AS deps-cpu
COPY libs/common /libs/common
RUN --mount=type=cache,target=/root/.cache/pip pip install /libs/common
COPY services/tts_service/pyproject.toml .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install ".[ogg]"

# ── GPU deps (CUDA 12.1) ──────────────────────────────────────────────────────
FROM base AS deps-gpu
COPY libs/common /libs/common
RUN --mount=type=cache,target=/root/.cache/pip pip install /libs/common
COPY services/tts_service/pyproject.toml .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install ".[ogg,gpu]"

# ── Development (CPU, hot-reload) ─────────────────────────────────────────────
FROM deps-cpu AS development
COPY services/tts_service /app
USER glossa
EXPOSE 8004
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004", "--reload"]

# ── Production CPU ────────────────────────────────────────────────────────────
FROM deps-cpu AS production
COPY services/tts_service /app
RUN chown -R glossa:glossa /app
USER glossa
EXPOSE 8004
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8004/health/live || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004", \
     "--workers", "1", "--loop", "uvloop"]

# ── Production GPU ────────────────────────────────────────────────────────────
FROM deps-gpu AS production-gpu
COPY services/tts_service /app
RUN chown -R glossa:glossa /app
USER glossa
EXPOSE 8004
ENV TTS_DEVICE=cuda
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8004/health/live || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004", \
     "--workers", "1", "--loop", "uvloop"]
