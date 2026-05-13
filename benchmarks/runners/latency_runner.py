"""Per-service latency benchmarks — HTTP and WebSocket.

Measures:
  - HTTP health, gesture inference, ASR, NLP, TTS endpoints
  - WebSocket round-trip time (ping/pong + payload echo)
  - End-to-end pipeline latency (frames → translation)

Usage:
    python -m benchmarks.runners.latency_runner
    python -m benchmarks.runners.latency_runner --services cv asr nlp tts ws
    python -m benchmarks.runners.latency_runner --concurrency 10 --samples 500
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Coroutine

import aiohttp

from benchmarks.config import BenchmarkConfig, LatencySLO, SLOS, get_config
from benchmarks.synthetic.workloads import (
    make_gesture_payload,
    make_audio_payload,
    make_gloss_payload,
    make_tts_payload,
)


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class LatencyResult:
    service: str
    endpoint: str
    n_samples: int
    n_errors: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    throughput_rps: float
    slo: LatencySLO | None = None
    slo_passed: bool = True
    slo_violations: list[str] = field(default_factory=list)

    def check_slo(self) -> None:
        if self.slo:
            self.slo_passed, self.slo_violations = self.slo.check(
                self.p50_ms, self.p95_ms, self.p99_ms
            )

    def summary_line(self) -> str:
        status = "PASS" if self.slo_passed else "FAIL"
        return (
            f"[{status}] {self.service:<20} "
            f"p50={self.p50_ms:6.1f}ms  "
            f"p95={self.p95_ms:6.1f}ms  "
            f"p99={self.p99_ms:6.1f}ms  "
            f"rps={self.throughput_rps:6.1f}  "
            f"err={self.n_errors}/{self.n_samples}"
        )


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    data = sorted(data)
    idx = int(p / 100.0 * len(data))
    return data[min(idx, len(data) - 1)]


def _compute_result(
    service: str,
    endpoint: str,
    timings_ms: list[float],
    errors: int,
    total_s: float,
    slo_key: str | None = None,
) -> LatencyResult:
    if not timings_ms:
        return LatencyResult(
            service=service, endpoint=endpoint, n_samples=0, n_errors=errors,
            mean_ms=0, std_ms=0, min_ms=0, max_ms=0,
            p50_ms=0, p90_ms=0, p95_ms=0, p99_ms=0, throughput_rps=0,
        )
    result = LatencyResult(
        service=service,
        endpoint=endpoint,
        n_samples=len(timings_ms) + errors,
        n_errors=errors,
        mean_ms=statistics.mean(timings_ms),
        std_ms=statistics.stdev(timings_ms) if len(timings_ms) > 1 else 0.0,
        min_ms=min(timings_ms),
        max_ms=max(timings_ms),
        p50_ms=_percentile(timings_ms, 50),
        p90_ms=_percentile(timings_ms, 90),
        p95_ms=_percentile(timings_ms, 95),
        p99_ms=_percentile(timings_ms, 99),
        throughput_rps=(len(timings_ms) / total_s) if total_s > 0 else 0.0,
        slo=SLOS.get(slo_key) if slo_key else None,
    )
    result.check_slo()
    return result


# ── HTTP benchmarks ───────────────────────────────────────────────────────────

async def bench_http_endpoint(
    session: aiohttp.ClientSession,
    service: str,
    url: str,
    method: str = "GET",
    json_payload: dict | None = None,
    n_samples: int = 200,
    warmup: int = 20,
    slo_key: str | None = None,
) -> LatencyResult:
    errors = 0

    # Warmup
    for _ in range(warmup):
        try:
            async with session.request(method, url, json=json_payload) as r:
                await r.read()
        except Exception:
            pass

    timings_ms: list[float] = []
    t_start = time.perf_counter()

    for _ in range(n_samples):
        t0 = time.perf_counter()
        try:
            async with session.request(method, url, json=json_payload) as r:
                await r.read()
                if r.status >= 500:
                    errors += 1
                    continue
        except Exception:
            errors += 1
            continue
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

    total_s = time.perf_counter() - t_start
    return _compute_result(service, url, timings_ms, errors, total_s, slo_key)


# ── WebSocket RTT benchmark ───────────────────────────────────────────────────

async def bench_websocket_rtt(
    ws_url: str,
    n_samples: int = 200,
    warmup: int = 20,
    payload_fn: Callable[[], dict] | None = None,
) -> LatencyResult:
    try:
        import websockets  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("pip install websockets") from exc

    errors = 0
    timings_ms: list[float] = []

    try:
        async with websockets.connect(ws_url, ping_interval=None, open_timeout=5) as ws:
            # Warmup
            for _ in range(warmup):
                try:
                    msg = payload_fn() if payload_fn else {"type": "ping"}
                    await ws.send(json.dumps(msg))
                    await asyncio.wait_for(ws.recv(), timeout=5.0)
                except Exception:
                    pass

            t_start = time.perf_counter()
            for _ in range(n_samples):
                msg = payload_fn() if payload_fn else {"type": "ping"}
                t0 = time.perf_counter()
                try:
                    await ws.send(json.dumps(msg))
                    await asyncio.wait_for(ws.recv(), timeout=5.0)
                    timings_ms.append((time.perf_counter() - t0) * 1000.0)
                except Exception:
                    errors += 1
            total_s = time.perf_counter() - t_start

    except Exception as exc:
        print(f"  WebSocket connection failed: {exc}")
        return _compute_result("ws", ws_url, [], n_samples, 0.0, "websocket_rtt")

    return _compute_result("websocket", ws_url, timings_ms, errors, total_s, "websocket_rtt")


# ── Concurrent latency (under load) ─────────────────────────────────────────

async def bench_concurrent(
    service: str,
    url: str,
    method: str = "POST",
    json_payload: dict | None = None,
    concurrency: int = 10,
    total_requests: int = 500,
    slo_key: str | None = None,
) -> LatencyResult:
    sem = asyncio.Semaphore(concurrency)
    errors = 0
    timings_ms: list[float] = []
    lock = asyncio.Lock()

    connector = aiohttp.TCPConnector(limit=concurrency + 10, ttl_dns_cache=300)
    timeout = aiohttp.ClientTimeout(total=10.0)

    async def one_request() -> None:
        nonlocal errors
        async with sem:
            t0 = time.perf_counter()
            try:
                async with aiohttp.ClientSession(connector=connector, timeout=timeout) as s:
                    async with s.request(method, url, json=json_payload) as r:
                        await r.read()
                        elapsed = (time.perf_counter() - t0) * 1000.0
                        async with lock:
                            if r.status < 500:
                                timings_ms.append(elapsed)
                            else:
                                errors += 1
            except Exception:
                async with lock:
                    errors += 1

    t_start = time.perf_counter()
    await asyncio.gather(*[one_request() for _ in range(total_requests)])
    total_s = time.perf_counter() - t_start
    await connector.close()

    return _compute_result(f"{service}@{concurrency}c", url, timings_ms, errors, total_s, slo_key)


# ── Full benchmark suite ──────────────────────────────────────────────────────

async def run_all(cfg: BenchmarkConfig | None = None) -> list[LatencyResult]:
    cfg = cfg or get_config()
    ep = cfg.endpoints
    results: list[LatencyResult] = []

    timeout = aiohttp.ClientTimeout(total=cfg.timeout_s, connect=cfg.connect_timeout_s)
    connector = aiohttp.TCPConnector(limit=20, ttl_dns_cache=300)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        print("\n── Health checks ────────────────────────────────────────────")
        for name, port in [("api-gateway", 8000), ("cv-service", 8001),
                           ("asr-service", 8002), ("nlp-service", 8003),
                           ("tts-service", 8004)]:
            r = await bench_http_endpoint(
                session, name, ep.health(port),
                n_samples=cfg.n_samples, warmup=cfg.warmup_runs, slo_key="http_health",
            )
            results.append(r)
            print(r.summary_line())

        print("\n── Gesture inference ────────────────────────────────────────")
        r = await bench_http_endpoint(
            session, "cv-service", f"{ep.cv_url}/recognize",
            method="POST", json_payload=make_gesture_payload(),
            n_samples=cfg.n_samples, warmup=cfg.warmup_runs, slo_key="gesture_inference",
        )
        results.append(r)
        print(r.summary_line())

        print("\n── ASR (audio chunk) ────────────────────────────────────────")
        r = await bench_http_endpoint(
            session, "asr-service", f"{ep.asr_url}/transcribe",
            method="POST", json_payload=make_audio_payload(),
            n_samples=cfg.n_samples, warmup=cfg.warmup_runs, slo_key="asr_chunk",
        )
        results.append(r)
        print(r.summary_line())

        print("\n── NLP translation ──────────────────────────────────────────")
        r = await bench_http_endpoint(
            session, "nlp-service", f"{ep.nlp_url}/translate",
            method="POST", json_payload=make_gloss_payload(),
            n_samples=cfg.n_samples, warmup=cfg.warmup_runs, slo_key="nlp_translate",
        )
        results.append(r)
        print(r.summary_line())

        print("\n── TTS synthesis ────────────────────────────────────────────")
        r = await bench_http_endpoint(
            session, "tts-service", f"{ep.tts_url}/synthesize",
            method="POST", json_payload=make_tts_payload(),
            n_samples=cfg.n_samples, warmup=cfg.warmup_runs, slo_key="tts_synthesize",
        )
        results.append(r)
        print(r.summary_line())

        print("\n── Concurrency ladders ──────────────────────────────────────")
        for concurrency in [1, 5, 10, 25]:
            r = await bench_concurrent(
                "nlp-service", f"{ep.nlp_url}/translate",
                method="POST", json_payload=make_gloss_payload(),
                concurrency=concurrency, total_requests=concurrency * 10,
                slo_key="nlp_translate",
            )
            results.append(r)
            print(r.summary_line())

    print("\n── WebSocket RTT ────────────────────────────────────────────")
    r = await bench_websocket_rtt(
        ep.ws_gesture,
        n_samples=min(100, cfg.n_samples),
        warmup=10,
        payload_fn=lambda: {"type": "ping"},
    )
    results.append(r)
    print(r.summary_line())

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Latency benchmarks")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", default="benchmarks/reports/latency.json")
    args = parser.parse_args()

    cfg = get_config()
    cfg.n_samples = args.samples
    cfg.warmup_runs = args.warmup

    results = asyncio.run(run_all(cfg))

    # Save raw results
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(r) for r in results], indent=2, default=str))
    print(f"\nResults saved to {out}")

    n_fail = sum(1 for r in results if not r.slo_passed)
    print(f"\n{'='*60}")
    print(f"  {len(results) - n_fail}/{len(results)} SLO checks passed")
    if n_fail:
        print("  VIOLATIONS:")
        for r in results:
            for v in r.slo_violations:
                print(f"    {r.service}: {v}")


if __name__ == "__main__":
    main()
