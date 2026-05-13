# Glossa — Multimodal RSL Translation Platform

Production-ready monorepo for real-time **Russian Sign Language (RSL)** bidirectional translation.
Designed to eliminate 2–4 hour interpreter wait times in medical and banking contexts.

## Architecture

```
Client (video / audio)
         │
         ▼
┌──────────────────┐
│   API Gateway    │  ← WebSocket + REST, rate limiting, CORS
│     :8000        │
└────────┬─────────┘
         │  Redis Streams (event-driven)
    ┌────┼──────────────────┐
    ▼    ▼                  ▼
┌────────┐ ┌────────┐ ┌──────────────┐
│  CV    │ │  ASR   │ │     NLP      │
│ :8001  │ │ :8002  │ │    :8003     │
│MediaP. │ │Faster  │ │ Qwen2-1.5B   │
│+ONNX/  │ │Whisper │ │ LangGraph    │
│OpenVINO│ │(base)  │ │ + RAG/Qdrant │
└────────┘ └────────┘ └──────────────┘
    │                        │
    ▼                        ▼
┌──────────┐       ┌──────────────────┐
│   TTS    │       │   MAX Adapter    │
│  :8004   │       │     :8005        │
│  Silero  │       │  HW-accelerated  │
│   v4     │       │    inference     │
└──────────┘       └──────────────────┘
              │
   ┌──────────┼──────────────┐
   │  Redis   │  Qdrant      │  OTel/Jaeger/Prometheus/Grafana
   └──────────┴──────────────┘
```

**RSL → Speech:** Camera frames → CV (MediaPipe + ONNX) → NLP (Qwen2 + RAG) → TTS (Silero) → Audio output

**Speech → RSL:** Microphone → ASR (faster-whisper) → NLP → Gloss sequence display

## Quick Start

```bash
# 1. Configure environment
cp .env.example .env

# 2. Pull model artifacts
dvc pull

# 3. Initialise Qdrant collections and seed glossary
docker compose up -d qdrant redis
python scripts/init_qdrant.py --seed

# 4. Start all services (hot-reload)
make dev

# 5. Open API docs
open http://localhost:8000/docs
```

## Services

| Service              | Port | Technology                            | Role                        |
|----------------------|------|---------------------------------------|-----------------------------|
| `api-gateway`        | 8000 | FastAPI + WebSocket                   | Entry point, orchestration  |
| `cv-service`         | 8001 | MediaPipe Holistic + ONNX/OpenVINO    | Skeleton extraction + STGCN |
| `asr-service`        | 8002 | faster-whisper base (CTranslate2)     | Speech-to-text              |
| `nlp-service`        | 8003 | Qwen2-1.5B + LangGraph + Qdrant       | Gloss translation + RAG     |
| `tts-service`        | 8004 | Silero TTS v4                         | Text-to-speech              |
| `max-adapter`        | 8005 | Modular MAX SDK                       | HW-accelerated inference    |

## Project Structure

```
glossa/
├── services/                    # Microservices (one per domain)
│   ├── api_gateway/             # WebSocket + REST gateway
│   ├── cv_service/              # MediaPipe keypoint extraction
│   ├── gesture_recognition/     # STGCN skeleton-based classifier
│   ├── asr_service/             # faster-whisper transcription
│   ├── nlp_service/             # Qwen2 + LangGraph + RAG translation
│   ├── tts_service/             # Silero TTS synthesis
│   └── max_adapter/             # MAX SDK inference bridge
├── libs/
│   ├── common/                  # Shared: schemas, messaging, telemetry, logging
│   └── max_sdk/                 # Typed MAX SDK wrapper
├── mlops/                       # ML lifecycle tooling
│   ├── pipelines/               # train_gesture, eval_gesture, eval_nlp, benchmark
│   ├── tracking/                # MLflow experiment tracker, metrics dataclasses
│   ├── registry/                # ONNX model registry + promotion gates
│   └── dataset/                 # DVC-tracked dataset versioning
├── experiments/                 # Architecture justification (diploma)
│   ├── shared/                  # SlovoDataset loader, MLflow utils
│   ├── 01_gesture_backbone/     # STGCN vs S3D vs ResNet3D
│   ├── 02_sliding_window/       # Window×stride grid search
│   ├── 03_inference_acceleration/ # PyTorch→ONNX→OpenVINO→INT8
│   ├── 04_asr_comparison/       # Whisper tiny/base/small
│   ├── 05_nlp_llm_size/         # Qwen2-0.5B/1.5B/7B
│   ├── 06_rag_ablation/         # LLM-only vs RAG-general vs RAG-domain
│   ├── 07_tts_utmos/            # Silero v4 UTMOS vs GigaTTS reference
│   ├── 08_e2e_latency/          # Full pipeline + mobile device projection
│   └── run_all.sh               # Run all 8 experiments
├── benchmarks/                  # Load & performance testing
│   ├── runners/                 # latency, stress, redis, queue runners
│   ├── profiling/               # CPU/GPU profilers (py-spy, pynvml)
│   ├── synthetic/               # WAV/keypoint workload generators
│   ├── reporters/               # HTML/JSON report builder
│   ├── locustfile.py            # Locust load test scenarios
│   └── config.py                # Device profiles, SLO thresholds
├── infra/
│   ├── docker/                  # Base Dockerfiles
│   ├── k8s/                     # Helm charts (K8s migration path)
│   ├── monitoring/              # Prometheus, Grafana, OTel Collector, Jaeger
│   ├── nginx/                   # Reverse proxy config
│   └── redis/                   # Redis Streams config
├── scripts/                     # Dev utilities
│   ├── init_qdrant.py           # Seed Qdrant with RSL glossary
│   ├── export_gesture_classifier.py  # PyTorch → ONNX export
│   ├── promote_model.py         # Registry promotion with SLO gate
│   ├── run_benchmarks.sh        # Full benchmark orchestration
│   ├── deploy.sh                # Rolling production deploy
│   └── ws_gesture_client_example.py  # WebSocket demo client
├── docker-compose.yml           # Development stack
├── docker-compose.override.yml  # Hot-reload overrides
├── docker-compose.gpu.yml       # GPU device reservations
├── docker-compose.prod.yml      # Production resource limits
├── dvc.yaml                     # Model + experiment pipeline stages
├── params.yaml                  # All hyperparameters tracked by DVC
├── pyproject.toml               # Root lint/type/test config
└── Makefile                     # Developer commands
```

## Development Commands

```bash
make help              # Show all available commands
make install           # Install dev dependencies + pre-commit hooks
make dev               # Start all services with hot-reload
make up                # Start all services (detached)
make down              # Stop all services
make build             # Build all Docker images
make test              # Run full test suite (pytest)
make lint              # Ruff + Black check
make typecheck         # mypy --strict
make check             # lint + format + typecheck combined
make gpu-up            # Start with GPU support (NVIDIA Container Toolkit)
make prod-up           # Start with production resource limits
make deploy            # Rolling production deployment
make backup            # Backup Redis, Qdrant, MLflow, Grafana
make mlflow-ui         # Open MLflow experiment tracker
make migrate-qdrant    # Re-initialise Qdrant collections
make logs-nlp-service  # Follow NLP service logs
```

## Running Experiments

Each experiment can be run in **dry-run mode** (no services or models required) or against live services.

```bash
# All 8 experiments, dry-run
bash experiments/run_all.sh --dry-run

# All 8 experiments against live services
bash experiments/run_all.sh \
    --host http://localhost:8000 \
    --slovo-root /path/to/slovo \
    --n-samples 100

# Single experiment
python -m experiments.01_gesture_backbone.run --dry-run
python -m experiments.08_e2e_latency.run --host http://localhost:8000

# Via DVC pipeline (reproduces all exp stages)
dvc repro exp_01_gesture_backbone
dvc repro exp_08_e2e_latency
```

Results are written to `experiments/results/<exp>/results.json` and logged to MLflow.

### Experiment Overview

| # | Experiment | Decision justified |
|---|------------|--------------------|
| 01 | Gesture backbone comparison | STGCN (skeleton) over S3D (video) |
| 02 | Sliding window grid search | window=30, stride=15 |
| 03 | ONNX / OpenVINO / INT8 acceleration | OpenVINO: 3.5+ PPS on CPU |
| 04 | Whisper tiny / base / small | whisper-base (WER ≤ 15%, P95 ≤ 350ms) |
| 05 | Qwen2-0.5B / 1.5B / 7B | Qwen2-1.5B (BLEU-4 ≥ 0.35, P95 ≤ 600ms) |
| 06 | RAG ablation (domain vs general vs none) | domain RAG: +27pp medical term recall |
| 07 | Silero v4 UTMOS vs GigaTTS (Sber 2024) | Silero: UTMOS 3.81, RTF 0.047, CPU-only |
| 08 | End-to-end latency + mobile projection | E2E P95 ≈ 977ms, Poco M5 passes 2000ms SLO |

## Benchmarks

```bash
# Full benchmark suite (against running services)
bash scripts/run_benchmarks.sh \
    --host http://localhost:8000 \
    --concurrency 20 \
    --duration 60

# Locust load test (UI at http://localhost:8089)
cd benchmarks && locust --host http://localhost:8000

# Just latency profiling
python -m benchmarks.runners.latency_runner --host http://localhost:8000
```

## CPU / GPU Switching

```bash
make up          # CPU (default)
make gpu-up      # GPU (requires NVIDIA Container Toolkit)
```

Device auto-detected via `CV_DEVICE`, `ASR_DEVICE`, `NLP_DEVICE`, `TTS_DEVICE` in `.env`.

## WebSocket API

```
ws://localhost:8000/api/v1/ws/translate/{mode}
```

`mode`: `rsl_to_text` | `text_to_rsl`

**Send:**
```json
{"type": "video_frame", "frame": {"keypoints": [...]}, "domain": "medical"}
{"type": "audio_chunk", "audio": "<base64 PCM>",        "domain": "banking"}
{"type": "end_session"}
```

**Receive:**
```json
{"type": "chunk",       "session_id": "...", "payload": {"text": "...", "is_final": false}}
{"type": "result",      "session_id": "...", "payload": {"text": "...", "confidence": 0.92}}
{"type": "session_end", "session_id": "...", "payload": {}}
```

## REST API

```bash
# Synchronous translation
POST /api/v1/translate
{"mode": "rsl_to_text", "gloss_sequence": "ПРИВЕТ КАК ДЕЛА", "domain": "general"}

# Health checks (Docker / K8s probes)
GET /health/live
GET /health/ready

# Prometheus metrics
GET /metrics
```

## Observability Stack

| Tool       | URL                     | Purpose               |
|------------|-------------------------|-----------------------|
| Grafana    | http://localhost:3001   | Dashboards            |
| Prometheus | http://localhost:9090   | Metrics scraping      |
| Jaeger     | http://localhost:16686  | Distributed tracing   |
| MLflow     | http://localhost:5000   | Experiment tracking   |

## Model Lifecycle (MLOps)

```bash
# Reproduce full training + eval pipeline
dvc repro

# Train gesture classifier (falls back to SlovoDataset if processed data absent)
python -m mlops.pipelines.train_gesture --params params.yaml
python -m mlops.pipelines.train_gesture --slovo-root /data/slovo  # Kaggle/YC path

# Export to ONNX
python scripts/export_gesture_classifier.py

# Promote model to production (checks SLO gates)
python scripts/promote_model.py --model gesture_classifier --run-id <run_id>

# Evaluate NLP / ASR
python -m mlops.pipelines.eval_nlp --modality nlp
python -m mlops.pipelines.eval_nlp --modality asr
```

All hyperparameters live in `params.yaml` and are versioned by DVC.

## Kubernetes Migration

Helm chart scaffolded at `infra/k8s/helm/glossa/`. Each service maps to a Deployment + Service + HPA.

```bash
# Push images (update REGISTRY in CI first)
make build && docker compose push

# Deploy
helm install glossa infra/k8s/helm/glossa/ -f infra/k8s/values.prod.yaml
```

## Tech Stack

| Layer        | Technology |
|--------------|------------|
| Runtime      | Python 3.11, FastAPI, uvicorn[uvloop] |
| CV           | MediaPipe Holistic, ONNX Runtime, OpenVINO |
| Gesture      | STGCN (skeleton-based), 75 landmarks × 3 coords |
| ASR          | faster-whisper base (CTranslate2 backend) |
| NLP          | Qwen2-1.5B-Instruct, LangGraph, Qdrant + BGE-M3 |
| TTS          | Silero TTS v4 (CPU, RTF ≈ 0.05) |
| Messaging    | Redis Streams |
| Infra        | Docker Compose, Prometheus, Grafana, OTel, Jaeger |
| MLOps        | MLflow, DVC, ONNX Model Registry |
| Quality      | Ruff, Black, mypy strict, pre-commit, pytest, Bandit |

## SLO Targets

| Metric                          | Target       | Device              |
|---------------------------------|--------------|---------------------|
| End-to-end latency (P95)        | ≤ 2 000 ms  | Poco M5 (4G)        |
| End-to-end latency (P95)        | ≤ 1 100 ms  | Realme X60 (5G)     |
| Gesture inference (P95)         | ≤ 50 ms     | CPU                 |
| ASR latency (P95)               | ≤ 350 ms    | CPU int8            |
| NLP translation (P95)           | ≤ 600 ms    | GPU / CPU           |
| TTS synthesis (P95)             | ≤ 185 ms    | CPU                 |
| Gesture accuracy (Top-1)        | ≥ 90 %      | test split          |
| ASR WER                         | ≤ 15 %      | Russian speech      |
| NLP BLEU-4                      | ≥ 0.35      | RSL gloss → Russian |
