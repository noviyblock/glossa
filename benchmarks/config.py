"""Benchmark configuration — device profiles, SLOs, service endpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ── Target device profiles ────────────────────────────────────────────────────

@dataclass(frozen=True)
class DeviceProfile:
    """Budget smartphone network/compute profile for optimization targeting."""
    name: str
    cpu_cores: int
    ram_mb: int
    uplink_kbps: int       # Upload bandwidth (client → server video/audio)
    downlink_kbps: int     # Download bandwidth (server → client audio/text)
    rtt_ms: float          # Network round-trip time to server
    jitter_ms: float       # Network jitter (one sigma)
    packet_loss_pct: float = 0.0


# Budget 4G phone — common in Russian market
POCO_M5 = DeviceProfile(
    name="Poco M5 (4G)",
    cpu_cores=8,
    ram_mb=4096,
    uplink_kbps=8_000,
    downlink_kbps=25_000,
    rtt_ms=65.0,
    jitter_ms=15.0,
    packet_loss_pct=0.5,
)

# Mid-range 5G phone
REALME_X60 = DeviceProfile(
    name="Realme X60 (5G)",
    cpu_cores=8,
    ram_mb=6144,
    uplink_kbps=20_000,
    downlink_kbps=100_000,
    rtt_ms=30.0,
    jitter_ms=6.0,
    packet_loss_pct=0.1,
)

# Worst-case scenario: rural poor 4G
POOR_4G = DeviceProfile(
    name="Poor 4G (edge case)",
    cpu_cores=4,
    ram_mb=2048,
    uplink_kbps=1_500,
    downlink_kbps=4_000,
    rtt_ms=180.0,
    jitter_ms=50.0,
    packet_loss_pct=2.0,
)

DEVICE_PROFILES: list[DeviceProfile] = [POCO_M5, REALME_X60, POOR_4G]


# ── Service Level Objectives ──────────────────────────────────────────────────

@dataclass(frozen=True)
class LatencySLO:
    p50_ms: float
    p95_ms: float
    p99_ms: float
    description: str = ""

    def check(self, p50: float, p95: float, p99: float) -> tuple[bool, list[str]]:
        """Return (passed, list_of_violations)."""
        violations: list[str] = []
        if p50 > self.p50_ms:
            violations.append(f"P50 {p50:.0f}ms > SLO {self.p50_ms:.0f}ms")
        if p95 > self.p95_ms:
            violations.append(f"P95 {p95:.0f}ms > SLO {self.p95_ms:.0f}ms")
        if p99 > self.p99_ms:
            violations.append(f"P99 {p99:.0f}ms > SLO {self.p99_ms:.0f}ms")
        return len(violations) == 0, violations

    def effective(self, device: DeviceProfile) -> "LatencySLO":
        """Tighten SLOs by subtracting one-way network latency (server-side budget)."""
        half_rtt = device.rtt_ms / 2.0
        return LatencySLO(
            p50_ms=max(1.0, self.p50_ms - half_rtt),
            p95_ms=max(1.0, self.p95_ms - half_rtt - device.jitter_ms),
            p99_ms=max(1.0, self.p99_ms - half_rtt - device.jitter_ms * 2),
            description=f"{self.description} ({device.name} budget)",
        )


SLOS: dict[str, LatencySLO] = {
    "websocket_rtt":        LatencySLO(80,   150,  300,  "WebSocket round-trip"),
    "gesture_inference":    LatencySLO(30,   50,   80,   "CV gesture recognition"),
    "asr_chunk":            LatencySLO(200,  350,  600,  "ASR per 2-second audio chunk"),
    "nlp_translate":        LatencySLO(300,  600,  1000, "NLP gloss→text translation"),
    "tts_synthesize":       LatencySLO(150,  250,  400,  "TTS synthesis"),
    "end_to_end_gesture":   LatencySLO(700,  1200, 2000, "Video frame → spoken translation"),
    "end_to_end_voice":     LatencySLO(500,  900,  1500, "Audio → text translation"),
    "http_health":          LatencySLO(5,    15,   30,   "Health endpoint"),
    "redis_publish":        LatencySLO(1,    5,    10,   "Redis XADD single event"),
    "redis_consume":        LatencySLO(2,    10,   20,   "Redis XREADGROUP consumer"),
}

# Payload sizes (bytes) matching real traffic
GESTURE_FRAME_BYTES = 33 * 4 * 4    # 33 landmarks × 4 coords × float32 = 528 bytes
AUDIO_CHUNK_BYTES   = 16_000 * 2    # 1 second @ 16kHz int16 = 32 KB
TTS_OUTPUT_BYTES    = 24_000 * 2    # 1 second @ 24kHz int16 = 48 KB


# ── Service endpoints ─────────────────────────────────────────────────────────

@dataclass
class ServiceEndpoints:
    base_url: str      = "http://localhost:8000"
    cv_url: str        = "http://localhost:8001"
    asr_url: str       = "http://localhost:8002"
    nlp_url: str       = "http://localhost:8003"
    tts_url: str       = "http://localhost:8004"
    max_url: str       = "http://localhost:8005"
    redis_url: str     = "redis://localhost:6379/0"
    ws_gesture: str    = "ws://localhost:8000/api/v1/ws/translate/gesture_to_text"
    ws_voice: str      = "ws://localhost:8000/api/v1/ws/translate/speech_to_text"
    prometheus_url: str = "http://localhost:9090"

    def health(self, port: int) -> str:
        return f"http://localhost:{port}/health/live"


# ── Benchmark configuration ───────────────────────────────────────────────────

@dataclass
class BenchmarkConfig:
    endpoints: ServiceEndpoints = field(default_factory=ServiceEndpoints)

    # Sampling
    n_samples: int       = 200
    warmup_runs: int     = 20
    timeout_s: float     = 10.0
    connect_timeout_s: float = 3.0

    # Concurrency ladder for throughput curves
    concurrency_steps: list[int] = field(
        default_factory=lambda: [1, 2, 5, 10, 20, 50, 100]
    )

    # Stress test
    stress_max_users: int    = 150
    stress_spawn_rate: float = 5.0    # users/sec
    stress_duration_s: int   = 120

    # Redis benchmarks
    redis_events: int        = 10_000
    redis_pipeline_size: int = 100
    redis_streams: list[str] = field(default_factory=lambda: [
        "glossa:stream:cv:input",
        "glossa:stream:asr:input",
        "glossa:stream:nlp:input",
        "glossa:stream:tts:input",
    ])

    # Output paths
    reports_dir: Path       = Path("benchmarks/reports")
    flamegraphs_dir: Path   = Path("benchmarks/reports/flamegraphs")
    locust_html: Path       = Path("benchmarks/reports/locust_report.html")

    # Device profiles to simulate in reports
    device_profiles: list[DeviceProfile] = field(
        default_factory=lambda: DEVICE_PROFILES
    )

    def __post_init__(self) -> None:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.flamegraphs_dir.mkdir(parents=True, exist_ok=True)


_cfg: BenchmarkConfig | None = None


def get_config() -> BenchmarkConfig:
    global _cfg
    if _cfg is None:
        _cfg = BenchmarkConfig()
    return _cfg
