# syntax=docker/dockerfile:1.7
FROM python:3.11-slim-bookworm AS base

RUN groupadd --gid 1001 glossa \
    && useradd --uid 1001 --gid glossa --shell /bin/bash --create-home glossa

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_HOME=/models/torch_cache

WORKDIR /app

# Pre-create the torch hub cache mount point owned by the non-root user, so
# Docker initialises the named volume with correct ownership on first mount
# (otherwise it defaults to root and torch.hub.load() fails with EACCES).
RUN mkdir -p /torch_cache && chown -R glossa:glossa /torch_cache

# ── CPU deps ──────────────────────────────────────────────────────────────────
FROM base AS deps-cpu
COPY pyproject.toml .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN --mount=type=cache,target=/root/.cache/pip pip install .

# ── GPU deps (CUDA 12.1) ──────────────────────────────────────────────────────
FROM base AS deps-gpu
COPY pyproject.toml .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN --mount=type=cache,target=/root/.cache/pip pip install .

# ── Development ───────────────────────────────────────────────────────────────
FROM deps-cpu AS development
COPY . .
USER glossa
EXPOSE 8004
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004", "--reload"]

# ── Production CPU ────────────────────────────────────────────────────────────
FROM deps-cpu AS production
COPY . .
RUN chown -R glossa:glossa /app
USER glossa
EXPOSE 8004
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8004/health/live || exit 1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004", \
     "--workers", "1", "--loop", "uvloop"]

# ── Production GPU ────────────────────────────────────────────────────────────
FROM deps-gpu AS production-gpu
COPY . .
RUN chown -R glossa:glossa /app
USER glossa
EXPOSE 8004
ENV TTS_DEVICE=cuda
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8004/health/live || exit 1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004", \
     "--workers", "1", "--loop", "uvloop"]
