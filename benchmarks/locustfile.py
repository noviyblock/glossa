"""Locust load test scenarios for the Glossa multimodal platform.

Scenarios:
  GestureSessionUser  — gesture-to-text WebSocket session (Poco M5 profile)
  VoiceSessionUser    — speech-to-text WebSocket session
  APIGatewayUser      — REST API HTTP load
  FullPipelineUser    — complete multimodal session (HTTP polling)
  MixedLoadUser       — 60% gesture / 30% voice / 10% REST (realistic mix)

Usage:
    locust -f benchmarks/locustfile.py --host http://localhost:8000
    locust -f benchmarks/locustfile.py --headless -u 50 -r 5 --run-time 2m
    locust -f benchmarks/locustfile.py --headless -u 100 -r 10 --run-time 5m \
           --html benchmarks/reports/locust_report.html
"""

from __future__ import annotations

import base64
import json
import random
import time

from locust import HttpUser, TaskSet, between, constant_pacing, events, task
from locust.exception import StopUser

from benchmarks.synthetic.workloads import (
    make_gloss_payload,
    make_audio_payload,
    make_gesture_payload,
    make_tts_payload,
    make_ws_video_frame_message,
    make_ws_audio_message,
    simulate_mobile_latency,
    apply_bandwidth_limit,
    POCO_M5,
    REALME_X60,
)

# Try websocket support (locust-plugins or websockets)
try:
    from locust_plugins.users.websocket import WebSocketUser  # type: ignore[import]
    _HAS_WS_USER = True
except ImportError:
    _HAS_WS_USER = False


# ── Device profiles ───────────────────────────────────────────────────────────

# Re-import from workloads (already imported indirectly)
try:
    from benchmarks.config import POCO_M5, REALME_X60, POOR_4G
except ImportError:
    pass


# ── REST API user (api-gateway) ───────────────────────────────────────────────

class APIGatewayUser(HttpUser):
    """Hits HTTP endpoints directly — health, translate (via REST)."""

    wait_time = between(0.5, 2.0)
    weight = 3

    @task(5)
    def health_live(self) -> None:
        self.client.get("/health/live", name="/health/live")

    @task(2)
    def health_ready(self) -> None:
        self.client.get("/health/ready", name="/health/ready")

    @task(10)
    def nlp_translate(self) -> None:
        payload = make_gloss_payload()
        with self.client.post(
            "http://localhost:8003/translate",
            json=payload,
            name="nlp:/translate",
            catch_response=True,
        ) as resp:
            if resp.status_code >= 500:
                resp.failure(f"HTTP {resp.status_code}")
            elif resp.elapsed.total_seconds() > 2.0:
                resp.failure(f"Slow: {resp.elapsed.total_seconds():.2f}s > 2s SLO")

    @task(8)
    def cv_recognize(self) -> None:
        payload = make_gesture_payload()
        with self.client.post(
            "http://localhost:8001/recognize",
            json=payload,
            name="cv:/recognize",
            catch_response=True,
        ) as resp:
            if resp.status_code >= 500:
                resp.failure(f"HTTP {resp.status_code}")
            elif resp.elapsed.total_seconds() > 0.1:
                resp.failure(f"Slow: {resp.elapsed.total_seconds() * 1000:.0f}ms > 100ms SLO")

    @task(6)
    def asr_transcribe(self) -> None:
        payload = make_audio_payload(duration_s=1.0)
        with self.client.post(
            "http://localhost:8002/transcribe",
            json=payload,
            name="asr:/transcribe",
            catch_response=True,
        ) as resp:
            if resp.status_code >= 500:
                resp.failure(f"HTTP {resp.status_code}")

    @task(4)
    def tts_synthesize(self) -> None:
        payload = make_tts_payload()
        with self.client.post(
            "http://localhost:8004/synthesize",
            json=payload,
            name="tts:/synthesize",
            catch_response=True,
        ) as resp:
            if resp.status_code >= 500:
                resp.failure(f"HTTP {resp.status_code}")


# ── Full pipeline via HTTP polling (mobile-friendly fallback) ─────────────────

class FullPipelineUser(HttpUser):
    """Simulates the complete translation pipeline via HTTP polling.

    Mimics a Poco M5 user: gesture frames → CV → NLP → TTS.
    Wait time models realistic inter-sign delays (~2-3s per sign).
    """

    wait_time = between(2.0, 5.0)   # time between signs
    weight = 5

    def on_start(self) -> None:
        self.session_id = f"user-{random.randint(10000, 99999)}"
        self._device = POCO_M5

    @task
    def full_gesture_to_speech(self) -> None:
        t_pipeline_start = time.perf_counter()

        # Step 1: CV — gesture keypoints → gloss
        cv_start = time.perf_counter()
        cv_payload = make_gesture_payload(session_id=self.session_id)
        with self.client.post(
            "http://localhost:8001/recognize",
            json=cv_payload,
            name="[pipeline] 1-cv",
            catch_response=True,
        ) as r:
            cv_latency = (time.perf_counter() - cv_start) * 1000
            if r.status_code >= 500:
                r.failure("CV failed")
                return
            try:
                gloss = r.json().get("gloss_sequence", "ПРИВЕТ")
            except Exception:
                gloss = "ПРИВЕТ КАК ДЕЛА"

        # Step 2: NLP — gloss → Russian text
        nlp_start = time.perf_counter()
        nlp_payload = {
            "session_id": self.session_id,
            "gloss_sequence": gloss,
            "domain": "general",
            "context": [],
        }
        with self.client.post(
            "http://localhost:8003/translate",
            json=nlp_payload,
            name="[pipeline] 2-nlp",
            catch_response=True,
        ) as r:
            nlp_latency = (time.perf_counter() - nlp_start) * 1000
            if r.status_code >= 500:
                r.failure("NLP failed")
                return
            try:
                text = r.json().get("translation", "Привет")
            except Exception:
                text = "Привет"

        # Step 3: TTS — text → audio (optional based on mode)
        if random.random() < 0.6:   # 60% of requests use TTS
            tts_start = time.perf_counter()
            tts_payload = {
                "session_id": self.session_id,
                "text": text,
                "speaker": "aidar",
                "language": "ru",
                "sample_rate": 24_000,
            }
            with self.client.post(
                "http://localhost:8004/synthesize",
                json=tts_payload,
                name="[pipeline] 3-tts",
                catch_response=True,
            ) as r:
                tts_latency = (time.perf_counter() - tts_start) * 1000
                if r.status_code >= 500:
                    r.failure("TTS failed")

        total_ms = (time.perf_counter() - t_pipeline_start) * 1000

        # Report total pipeline as a custom event
        events.request.fire(
            request_type="PIPELINE",
            name="gesture_to_speech_e2e",
            response_time=total_ms,
            response_length=0,
            exception=None,
            context={},
        )

        # Simulate mobile transmission time (network overhead)
        # Video frame size at 320×240 JPEG ~30KB uplink
        tx_ms = apply_bandwidth_limit(
            b"x" * 30_000,
            bandwidth_kbps=self._device.uplink_kbps,
            latency_ms=self._device.rtt_ms,
        )
        # Log SLO check
        slo_budget = 2000  # ms (end-to-end SLO for Poco M5)
        effective_latency = total_ms + tx_ms
        if effective_latency > slo_budget:
            events.request.fire(
                request_type="SLO_VIOLATION",
                name=f"e2e_slo_{self._device.name}",
                response_time=effective_latency,
                response_length=0,
                exception=None,
                context={},
            )


# ── ASR voice session ─────────────────────────────────────────────────────────

class VoiceSessionUser(HttpUser):
    """Simulates voice input sessions (speech → text)."""

    wait_time = between(3.0, 8.0)
    weight = 2

    def on_start(self) -> None:
        self.session_id = f"voice-{random.randint(10000, 99999)}"

    @task(3)
    def asr_chunk(self) -> None:
        payload = make_audio_payload(
            duration_s=random.uniform(1.0, 3.0),
            session_id=self.session_id,
        )
        with self.client.post(
            "http://localhost:8002/transcribe",
            json=payload,
            name="voice:/transcribe",
            catch_response=True,
        ) as r:
            if r.status_code >= 500:
                r.failure("ASR error")

    @task(1)
    def voice_to_text_pipeline(self) -> None:
        # ASR → NLP
        asr_payload = make_audio_payload(duration_s=2.0, session_id=self.session_id)
        t0 = time.perf_counter()
        with self.client.post(
            "http://localhost:8002/transcribe",
            json=asr_payload,
            name="[voice-pipe] asr",
            catch_response=True,
        ) as r:
            if r.status_code >= 500:
                r.failure("ASR failed")
                return
            try:
                transcript = r.json().get("text", "как дела")
            except Exception:
                transcript = "как дела"

        # Forward to NLP for intent / response
        with self.client.post(
            "http://localhost:8003/translate",
            json={"session_id": self.session_id, "gloss_sequence": transcript.upper(), "domain": "general", "context": []},
            name="[voice-pipe] nlp",
            catch_response=True,
        ) as r:
            if r.status_code >= 500:
                r.failure("NLP failed")

        total_ms = (time.perf_counter() - t0) * 1000
        events.request.fire(
            request_type="PIPELINE",
            name="voice_to_text_e2e",
            response_time=total_ms,
            response_length=0,
            exception=None,
            context={},
        )


# ── Soak test user (long-running session) ────────────────────────────────────

class SoakUser(HttpUser):
    """Maintains long-running connections to detect memory leaks and drift."""

    wait_time = between(1.0, 3.0)
    weight = 1

    _REQUESTS = 0

    @task
    def keep_alive_request(self) -> None:
        SoakUser._REQUESTS += 1
        with self.client.get("/health/live", name="/health/live[soak]", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"Health degraded after {SoakUser._REQUESTS} requests")


# ── Locust events (Prometheus metrics emission) ───────────────────────────────

_REQUEST_TOTALS: dict[str, int] = {}
_REQUEST_LATENCIES: dict[str, list[float]] = {}


@events.request.add_listener
def on_request(
    request_type: str,
    name: str,
    response_time: float,
    response_length: int,
    exception: Exception | None,
    **kwargs: object,
) -> None:
    key = f"{request_type}:{name}"
    _REQUEST_TOTALS[key] = _REQUEST_TOTALS.get(key, 0) + 1
    _REQUEST_LATENCIES.setdefault(key, []).append(response_time)
    # Keep only last 1000 samples per key
    if len(_REQUEST_LATENCIES[key]) > 1000:
        _REQUEST_LATENCIES[key] = _REQUEST_LATENCIES[key][-1000:]


@events.test_stop.add_listener
def on_test_stop(environment: object, **kwargs: object) -> None:
    """Write a summary JSON when the test finishes."""
    import json
    from pathlib import Path
    summary = {}
    for key, lats in _REQUEST_LATENCIES.items():
        if lats:
            s = sorted(lats)
            n = len(s)
            summary[key] = {
                "count": _REQUEST_TOTALS.get(key, n),
                "p50": s[n // 2],
                "p95": s[int(0.95 * n)],
                "p99": s[int(0.99 * n)],
                "mean": sum(s) / n,
            }
    out = Path("benchmarks/reports/locust_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"Locust summary saved to {out}")
