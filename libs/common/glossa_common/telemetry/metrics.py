from prometheus_client import Counter, Gauge, Histogram

REQUEST_COUNT = Counter(
    "glossa_requests_total",
    "Total number of requests processed",
    ["service", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "glossa_request_latency_seconds",
    "Request latency in seconds",
    ["service", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

ACTIVE_WEBSOCKETS = Gauge(
    "glossa_active_websocket_connections",
    "Number of active WebSocket connections",
    ["service"],
)

TRANSLATION_LATENCY = Histogram(
    "glossa_translation_latency_seconds",
    "End-to-end translation latency",
    ["mode", "domain"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)

MODEL_INFERENCE_LATENCY = Histogram(
    "glossa_model_inference_latency_seconds",
    "Model inference latency per service",
    ["service", "model"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

STREAM_EVENTS = Counter(
    "glossa_stream_events_total",
    "Total Redis stream events processed",
    ["stream", "event_type", "status"],
)

REDIS_STREAM_LAG = Gauge(
    "glossa_redis_stream_lag",
    "Consumer group lag per stream",
    ["stream", "consumer_group"],
)

# ── MLOps model quality gauges (set by eval/benchmark pipelines) ──────────────

MODEL_QUALITY = Gauge(
    "glossa_model_quality",
    "Model evaluation metric (accuracy, f1, bleu, wer, latency_p95_ms, etc.)",
    ["model", "metric", "version", "stage"],
)

BENCHMARK_LATENCY_P95 = Gauge(
    "glossa_benchmark_latency_p95_ms",
    "Benchmark P95 latency in milliseconds",
    ["model", "backend", "batch_size"],
)

BENCHMARK_MEMORY_MB = Gauge(
    "glossa_benchmark_memory_mb",
    "Benchmark peak RSS memory in MB",
    ["model", "backend", "batch_size"],
)
