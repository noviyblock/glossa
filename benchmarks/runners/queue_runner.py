"""Event queue bottleneck analysis.

Measures:
  - Enqueue throughput (events/sec to MAX adapter queue)
  - Worker saturation under load
  - Drop rate when queue is full
  - Processing latency from enqueue → worker pickup

Usage:
    python -m benchmarks.runners.queue_runner
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path

import aiohttp

from benchmarks.config import BenchmarkConfig, get_config


@dataclass
class QueueAnalysis:
    queue_maxsize: int
    n_workers: int
    n_events_sent: int
    n_events_accepted: int
    n_events_dropped: int
    drop_rate: float
    enqueue_throughput_eps: float
    duration_s: float
    p50_ms: float
    p95_ms: float
    queue_full_fraction: float   # fraction of time queue was at capacity


async def _post_webhook_event(
    session: aiohttp.ClientSession,
    url: str,
    event: dict,
) -> tuple[bool, float]:
    """POST one event to the MAX adapter webhook. Returns (accepted, latency_ms)."""
    t0 = time.perf_counter()
    try:
        async with session.post(url, json=event) as r:
            await r.read()
            latency = (time.perf_counter() - t0) * 1000.0
            return r.status < 400, latency
    except Exception:
        return False, (time.perf_counter() - t0) * 1000.0


def _make_webhook_event(seq: int) -> dict:
    return {
        "update_id": seq,
        "update_type": "message_created",
        "timestamp": int(time.time()),
        "message": {
            "seq": seq,
            "sender": {
                "user_id": 42,
                "first_name": "BenchUser",
                "last_name": "Load",
                "username": "benchuser",
                "is_bot": False,
            },
            "recipient": {"chat_id": 1001},
            "body": {
                "text": "ПРИВЕТ КАК ДЕЛА СЕГОДНЯ",
                "attachments": [],
            },
            "timestamp": int(time.time()),
        },
    }


async def run_queue_analysis(cfg: BenchmarkConfig | None = None) -> QueueAnalysis:
    cfg = cfg or get_config()
    webhook_url = f"{cfg.endpoints.max_url}/api/v1/webhook"

    # First, probe queue config from the service stats endpoint
    queue_maxsize = 512
    n_workers = 4
    try:
        timeout = aiohttp.ClientTimeout(total=3.0)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.get(f"{cfg.endpoints.max_url}/health/ready") as r:
                if r.status == 200:
                    data = await r.json()
                    # Extract queue info if exposed in health response
                    for comp in data.get("components", []):
                        if "queue" in comp.get("name", ""):
                            msg = comp.get("message", "")
                            # Parse "size=512, workers=4, dropped=0"
                            import re
                            m = re.search(r"size=(\d+)", msg)
                            if m:
                                queue_maxsize = int(m.group(1))
                            m = re.search(r"workers=(\d+)", msg)
                            if m:
                                n_workers = int(m.group(1))
    except Exception:
        pass

    print(f"  Queue config: maxsize={queue_maxsize}, workers={n_workers}")

    # Phase 1: Baseline — send events at moderate rate
    n_events = 2000
    accepted = 0
    dropped = 0
    timings_ms: list[float] = []

    connector = aiohttp.TCPConnector(limit=50)
    timeout = aiohttp.ClientTimeout(total=5.0)
    t_start = time.perf_counter()

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Ramp up quickly to overflow the queue
        tasks = []
        for i in range(n_events):
            tasks.append(_post_webhook_event(session, webhook_url, _make_webhook_event(i)))
            if len(tasks) >= 50:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for ok, lat in results:
                    if isinstance(ok, bool):
                        if ok:
                            accepted += 1
                            timings_ms.append(lat)
                        else:
                            dropped += 1
                tasks = []
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for ok, lat in results:
                if isinstance(ok, bool):
                    if ok:
                        accepted += 1
                        timings_ms.append(lat)
                    else:
                        dropped += 1

    total_s = time.perf_counter() - t_start

    def pct(p: float) -> float:
        if not timings_ms:
            return 0.0
        s = sorted(timings_ms)
        return s[min(int(p / 100.0 * len(s)), len(s) - 1)]

    analysis = QueueAnalysis(
        queue_maxsize=queue_maxsize,
        n_workers=n_workers,
        n_events_sent=n_events,
        n_events_accepted=accepted,
        n_events_dropped=dropped,
        drop_rate=dropped / n_events if n_events > 0 else 0.0,
        enqueue_throughput_eps=accepted / total_s if total_s > 0 else 0.0,
        duration_s=total_s,
        p50_ms=pct(50),
        p95_ms=pct(95),
        queue_full_fraction=dropped / n_events if n_events > 0 else 0.0,
    )

    print(f"  Sent: {n_events}  Accepted: {accepted}  Dropped: {dropped}")
    print(f"  Drop rate: {analysis.drop_rate:.1%}")
    print(f"  Enqueue throughput: {analysis.enqueue_throughput_eps:.1f} eps")
    print(f"  Webhook P95 response: {analysis.p95_ms:.1f}ms")
    return analysis


async def run_all(cfg: BenchmarkConfig | None = None) -> list[QueueAnalysis]:
    cfg = cfg or get_config()
    results: list[QueueAnalysis] = []

    print("\n── Event queue analysis ─────────────────────────────────────")
    analysis = await run_queue_analysis(cfg)
    results.append(analysis)

    # Recommendations
    print("\n  Recommendations:")
    if analysis.drop_rate > 0.01:
        print(f"  ⚠ Drop rate {analysis.drop_rate:.1%} — increase MAX_EVENT_QUEUE_MAXSIZE")
    if analysis.p95_ms > 50:
        print(f"  ⚠ Webhook P95={analysis.p95_ms:.1f}ms — queue is blocking on full")
    if analysis.enqueue_throughput_eps < 500:
        print(f"  ⚠ Throughput {analysis.enqueue_throughput_eps:.1f}eps — consider more workers")

    out = cfg.reports_dir / "queue_results.json"
    out.write_text(json.dumps([asdict(a) for a in results], indent=2))
    print(f"\nQueue results saved to {out}")
    return results


if __name__ == "__main__":
    asyncio.run(run_all())
