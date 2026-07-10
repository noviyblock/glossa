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
# `pip install .` pulls in rtmlib's OWN transitive dependency on plain
# `onnxruntime` (CPU) — it and `onnxruntime-gpu` both occupy the same
# `onnxruntime/` import path, so whichever installs LAST wins the shared
# library that's actually loaded. Installing the GPU wheel first and letting
# `pip install .` run after it silently clobbers it back to CPU (confirmed:
# CUDAExecutionProvider missing entirely from the available-providers list
# at runtime, not just failing to initialize). Fix: force-reinstall the GPU
# wheel as the LAST step, after every other dependency is resolved.
RUN --mount=type=cache,target=/root/.cache/pip pip install .
# Pinned <1.21: unbounded `onnxruntime-gpu>=1.18` resolved to a release that
# wants libcudart.so.13 (CUDA 13) at runtime, which doesn't match the CUDA 12
# nvidia-*-cu12 libs below (or the host driver, only confirmed compatible up
# to CUDA 12.1 via nlp-service's torch-cu121 build) — confirmed via
# `ImportError: libcudart.so.13: cannot open shared object file` in
# production. Keep on the CUDA 12 / cuDNN 9 line until the host driver is
# confirmed to support CUDA 13.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip uninstall -y onnxruntime; \
    pip install "nvidia-cudnn-cu12>=9,<10" "nvidia-cuda-runtime-cu12" "nvidia-cublas-cu12" && \
    pip install --force-reinstall --no-deps "onnxruntime-gpu>=1.18,<1.21"
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
