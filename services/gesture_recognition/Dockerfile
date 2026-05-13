# syntax=docker/dockerfile:1.7
ARG PYTHON_VERSION=3.11

# ── base ──────────────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── deps ──────────────────────────────────────────────────────────────────────
FROM base AS deps

COPY libs/common /libs/common
RUN pip install /libs/common

COPY services/gesture_recognition/pyproject.toml .
RUN pip install ".[mlflow]"

# ── development ───────────────────────────────────────────────────────────────
FROM deps AS development

RUN pip install ".[dev]"
COPY services/gesture_recognition/ .

EXPOSE 8004
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004", "--reload"]

# ── production ────────────────────────────────────────────────────────────────
FROM deps AS production

COPY services/gesture_recognition/ .

RUN addgroup --system glossa && adduser --system --ingroup glossa glossa \
    && chown -R glossa:glossa /app
USER glossa

EXPOSE 8004

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8004/health/live')"

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8004", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--http", "httptools"]

# ── production-gpu ────────────────────────────────────────────────────────────
FROM production AS production-gpu

USER root
RUN pip install "onnxruntime-gpu>=1.18" "openvino>=2024.1" && \
    pip uninstall -y onnxruntime || true
USER glossa

ENV GESTURE_N_ONNX_THREADS="" \
    ORT_TENSORRT_ENABLED=1
