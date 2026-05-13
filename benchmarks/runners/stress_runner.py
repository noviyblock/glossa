"""Stress testing — find throughput ceiling and degradation curve.

Ramps concurrent sessions from 1 → N, recording how latency degrades.
Identifies the saturation point (inflection in latency curve).

Usage:
    python -m benchmarks.runners.stress_runner
    python -m benchmarks.runners.stress_runner --max-users 200 --duration 120
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import AsyncIterator

import aiohttp

from benchmarks.config import BenchmarkConfig, get_config, SLOS
from benchmarks.synthetic.workloads import make_gloss_payload, make_gesture_payload


@dataclass
class StressPoint:
    concurrency: int
    target_rps: float
    actual_rps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    error_rate: float
    duration_s: float
    total_requests: int
    saturated: bool = False


@dataclass
class StressReport:
    service: str
    endpoint: str
    points: list[StressPoint] = field(default_factory=list)
    saturation_concurrency: int | None = None
    max_rps: float = 0.0
    breaking_point_concurrency: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


async def _run_sustained_load(
    url: str,
    method: str = "POST",
    payload: dict | None = None,
    concurrency: int = 10,
    duration_s: float = 30.0,
) -> StressPoint:
    """Run sustained load for `duration_s` seconds at `concurrency` level."""
    sem = asyncio.Semaphore(concurrency)
    timings_ms: list[float] = []
    errors = 0
    lock = asyncio.Lock()
    stop_event = asyncio.Event()

    connector = aiohttp.TCPConnector(limit=concurrency + 20, ttl_dns_cache=300)
    timeout = aiohttp.ClientTimeout(total=10.0)

    async def worker() -> None:
        nonlocal errors
        session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        try:
            while not stop_event.is_set():
                async with sem:
                    if stop_event.is_set():
                        break
                    t0 = time.perf_counter()
                    try:
                        async with session.request(method, url, json=payload) as r:
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
        finally:
            await session.close()

    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    t_start = time.perf_counter()

    await asyncio.sleep(duration_s)
    stop_event.set()
    await asyncio.gather(*workers, return_exceptions=True)
    await connector.close()

    total_s = time.perf_counter() - t_start
    total = len(timings_ms) + errors

    def pct(p: float) -> float:
        if not timings_ms:
            return 0.0
        s = sorted(timings_ms)
        return s[min(int(p / 100.0 * len(s)), len(s) - 1)]

    actual_rps = total / total_s if total_s > 0 else 0.0
    error_rate = errors / total if total > 0 else 0.0

    # Saturated if P95 is >3× the P95 at concurrency=1 (heuristic)
    saturated = pct(95) > SLOS.get("nlp_translate", type("", (), {"p95_ms": 99999.0})()).p95_ms * 2  # type: ignore[attr-defined]

    return StressPoint(
        concurrency=concurrency,
        target_rps=concurrency / max(pct(50) / 1000.0, 0.001),
        actual_rps=actual_rps,
        p50_ms=pct(50),
        p95_ms=pct(95),
        p99_ms=pct(99),
        error_rate=error_rate,
        duration_s=total_s,
        total_requests=total,
        saturated=saturated,
    )


async def stress_service(
    service: str,
    url: str,
    method: str = "POST",
    payload: dict | None = None,
    concurrency_steps: list[int] | None = None,
    step_duration_s: float = 30.0,
) -> StressReport:
    steps = concurrency_steps or [1, 2, 5, 10, 20, 50, 100]
    report = StressReport(service=service, endpoint=url)
    prev_p95 = None

    for c in steps:
        print(f"  → {service} @ concurrency={c}...")
        point = await _run_sustained_load(url, method, payload, c, step_duration_s)
        report.points.append(point)

        # Detect saturation: P95 doubles from previous step
        if prev_p95 and point.p95_ms > prev_p95 * 2.5 and report.saturation_concurrency is None:
            report.saturation_concurrency = c

        # Breaking point: error rate > 5%
        if point.error_rate > 0.05 and report.breaking_point_concurrency is None:
            report.breaking_point_concurrency = c
            print(f"  ⚠ Breaking point at concurrency={c} "
                  f"(error_rate={point.error_rate:.1%})")

        report.max_rps = max(report.max_rps, point.actual_rps)
        prev_p95 = point.p95_ms

        print(
            f"    p50={point.p50_ms:6.1f}ms  "
            f"p95={point.p95_ms:6.1f}ms  "
            f"rps={point.actual_rps:6.1f}  "
            f"err={point.error_rate:.1%}"
        )

        # Stop if error rate too high
        if point.error_rate > 0.20:
            print(f"  ✗ Stopping stress test at {c} concurrent users (>20% errors)")
            break

    return report


async def run_all(cfg: BenchmarkConfig | None = None) -> list[StressReport]:
    cfg = cfg or get_config()
    ep = cfg.endpoints
    steps = cfg.concurrency_steps
    duration = max(15.0, cfg.stress_duration_s / len(steps))
    reports: list[StressReport] = []

    print("\n── NLP service stress test ──────────────────────────────────")
    r = await stress_service(
        "nlp-service", f"{ep.nlp_url}/translate",
        method="POST", payload=make_gloss_payload(),
        concurrency_steps=steps, step_duration_s=duration,
    )
    reports.append(r)

    print("\n── CV service stress test ───────────────────────────────────")
    r = await stress_service(
        "cv-service", f"{ep.cv_url}/recognize",
        method="POST", payload=make_gesture_payload(),
        concurrency_steps=steps, step_duration_s=duration,
    )
    reports.append(r)

    print("\n── API Gateway stress test ──────────────────────────────────")
    r = await stress_service(
        "api-gateway", f"{ep.base_url}/health/live",
        method="GET", payload=None,
        concurrency_steps=steps, step_duration_s=duration,
    )
    reports.append(r)

    print("\n── Summary ──────────────────────────────────────────────────")
    for rep in reports:
        sat = f"saturation@{rep.saturation_concurrency}c" if rep.saturation_concurrency else "not saturated"
        print(f"  {rep.service:<25} max_rps={rep.max_rps:.1f}  {sat}")

    # Save
    out = cfg.reports_dir / "stress_results.json"
    out.write_text(json.dumps([r.to_dict() for r in reports], indent=2, default=str))
    print(f"\nStress results saved to {out}")
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-users", type=int, default=100)
    parser.add_argument("--duration", type=int, default=120, help="Total stress test duration (s)")
    args = parser.parse_args()
    cfg = get_config()
    cfg.stress_max_users = args.max_users
    cfg.stress_duration_s = args.duration
    asyncio.run(run_all(cfg))


if __name__ == "__main__":
    main()
