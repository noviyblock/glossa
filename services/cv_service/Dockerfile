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
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ── CPU build ─────────────────────────────────────────────────────────────────
FROM base AS deps-cpu
COPY pyproject.toml .
RUN --mount=type=cache,target=/root/.cache/pip pip install "onnxruntime>=1.18"
RUN --mount=type=cache,target=/root/.cache/pip pip install .

# ── GPU build (CUDA 12.x) ────────────────────────────────────────────────────
# The ST-GCN classifier (gesture_classifier.py) stays on OpenVINO/CPU always
# — it's tiny (~10ms P95 already). The GPU build targets the actual
# bottleneck: rtmlib's RTMDet-nano+RTMW keypoint extraction, ~90ms/frame on
# CPU and the dominant cost in the per-frame latency budget. onnxruntime-gpu
# needs CUDA/cuDNN runtime libs; rather than an nvidia/cuda base image, they
# come from pip (nvidia-* wheels) — the container only needs the NVIDIA
# *driver* via nvidia-container-toolkit on the host (already required for
# nlp/asr/tts GPU builds), not a system CUDA toolkit install.
FROM base AS deps-gpu
COPY pyproject.toml .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "onnxruntime-gpu>=1.18" "nvidia-cudnn-cu12>=9,<10" \
        "nvidia-cuda-runtime-cu12" "nvidia-cublas-cu12"
RUN --mount=type=cache,target=/root/.cache/pip pip install .
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib

FROM deps-cpu AS development
COPY . .
USER glossa
EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]

FROM deps-cpu AS production
COPY . .
RUN chown -R glossa:glossa /app \
    && mkdir -p /home/glossa/.rtmlib \
    && chown -R glossa:glossa /home/glossa/.rtmlib
USER glossa
EXPOSE 8001
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health/live || exit 1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", \
     "--workers", "1", "--loop", "uvloop"]

FROM deps-gpu AS production-gpu
COPY . .
RUN chown -R glossa:glossa /app \
    && mkdir -p /home/glossa/.rtmlib \
    && chown -R glossa:glossa /home/glossa/.rtmlib
USER glossa
EXPOSE 8001
ENV RTMLIB_DEVICE=cuda
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health/live || exit 1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", \
     "--workers", "1", "--loop", "uvloop"]
