# syntax=docker/dockerfile:1.7
FROM python:3.11-slim-bookworm AS base

RUN groupadd --gid 1001 glossa \
    && useradd --uid 1001 --gid glossa --shell /bin/bash --create-home glossa

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
        libglib2.0-0 \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

FROM base AS deps
COPY libs/common /libs/common
RUN --mount=type=cache,target=/root/.cache/pip pip install /libs/common
COPY services/cv_service/pyproject.toml .
RUN --mount=type=cache,target=/root/.cache/pip pip install .

FROM deps AS development
COPY services/cv_service /app
USER glossa
EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]

FROM deps AS production
COPY services/cv_service /app
RUN chown -R glossa:glossa /app
USER glossa
EXPOSE 8001
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health/live || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", \
     "--workers", "1", "--loop", "uvloop"]

FROM production AS production-gpu
USER root
RUN --mount=type=cache,target=/root/.cache/pip pip install onnxruntime-gpu openvino
USER glossa
