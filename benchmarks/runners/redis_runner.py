"""Redis load testing — stream throughput, consumer lag, pipeline vs single.

Benchmarks:
  - XADD throughput (single vs pipelined)
  - XREADGROUP consumer latency
  - Consumer group lag under load
  - Key expiry and memory under sustained write load

Usage:
    python -m benchmarks.runners.redis_runner
    python -m benchmarks.runners.redis_runner --events 50000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path

from benchmarks.config import BenchmarkConfig, get_config


@dataclass
class RedisStreamResult:
    stream: str
    operation: str
    n_events: int
    n_errors: int
    duration_s: float
    throughput_eps: float   # events/sec
    p50_ms: float
    p95_ms: float
    p99_ms: float
    consumer_lag: int = 0
    memory_delta_mb: float = 0.0
    pipeline_size: int = 1

    def summary_line(self) -> str:
        return (
            f"  {self.stream:<35} {self.operation:<12} "
            f"eps={self.throughput_eps:8.1f}  "
            f"p95={self.p95_ms:6.2f}ms  "
            f"lag={self.consumer_lag}"
        )


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    return s[min(int(p / 100.0 * len(s)), len(s) - 1)]


async def _bench_xadd_single(
    client: "redis.asyncio.Redis",
    stream: str,
    n_events: int,
    group: str = "bench-group",
) -> RedisStreamResult:
    """Measure single XADD latency."""
    try:
        await client.xgroup_create(stream, group, id="0", mkstream=True)
    except Exception:
        pass

    timings_ms: list[float] = []
    errors = 0
    t_start = time.perf_counter()

    for i in range(n_events):
        payload = {
            "event_id": str(uuid.uuid4()),
            "event_type": "bench_event",
            "seq": str(i),
            "data": "x" * 128,  # ~128 byte payload
        }
        t0 = time.perf_counter()
        try:
            await client.xadd(stream, payload, maxlen=50_000, approximate=True)
            timings_ms.append((time.perf_counter() - t0) * 1000.0)
        except Exception:
            errors += 1

    total_s = time.perf_counter() - t_start
    return RedisStreamResult(
        stream=stream,
        operation="XADD",
        n_events=n_events,
        n_errors=errors,
        duration_s=total_s,
        throughput_eps=n_events / total_s if total_s > 0 else 0.0,
        p50_ms=_percentile(timings_ms, 50),
        p95_ms=_percentile(timings_ms, 95),
        p99_ms=_percentile(timings_ms, 99),
    )


async def _bench_xadd_pipeline(
    client: "redis.asyncio.Redis",
    stream: str,
    n_events: int,
    pipeline_size: int = 100,
) -> RedisStreamResult:
    """Measure pipelined XADD throughput."""
    errors = 0
    timings_ms: list[float] = []
    t_start = time.perf_counter()

    for batch_start in range(0, n_events, pipeline_size):
        batch_end = min(batch_start + pipeline_size, n_events)
        t0 = time.perf_counter()
        try:
            async with client.pipeline(transaction=False) as pipe:
                for i in range(batch_start, batch_end):
                    payload = {
                        "seq": str(i),
                        "data": "x" * 128,
                    }
                    pipe.xadd(stream, payload, maxlen=50_000, approximate=True)
                await pipe.execute()
            timings_ms.append((time.perf_counter() - t0) * 1000.0)
        except Exception:
            errors += 1

    total_s = time.perf_counter() - t_start
    return RedisStreamResult(
        stream=stream,
        operation="XADD_PIPE",
        n_events=n_events,
        n_errors=errors,
        duration_s=total_s,
        throughput_eps=n_events / total_s if total_s > 0 else 0.0,
        p50_ms=_percentile(timings_ms, 50),
        p95_ms=_percentile(timings_ms, 95),
        p99_ms=_percentile(timings_ms, 99),
        pipeline_size=pipeline_size,
    )


async def _bench_consumer_lag(
    client: "redis.asyncio.Redis",
    stream: str,
    n_events: int = 5000,
) -> RedisStreamResult:
    """Publish events, then measure how quickly a consumer can drain them."""
    group = f"bench-lag-{uuid.uuid4().hex[:8]}"
    consumer = "bench-consumer-0"

    try:
        await client.xgroup_create(stream, group, id="0", mkstream=True)
    except Exception:
        pass

    # Publish burst
    async with client.pipeline(transaction=False) as pipe:
        for i in range(n_events):
            pipe.xadd(stream, {"seq": str(i), "data": "x" * 64}, maxlen=100_000, approximate=True)
        await pipe.execute()

    # Measure drain time
    consumed = 0
    timings_ms: list[float] = []
    t_start = time.perf_counter()

    while consumed < n_events:
        t0 = time.perf_counter()
        messages = await client.xreadgroup(
            group, consumer, {stream: ">"}, count=100, block=100
        )
        if not messages:
            # No more pending
            break
        batch = messages[0][1] if messages else []
        if batch:
            msg_ids = [m[0] for m in batch]
            await client.xack(stream, group, *msg_ids)
            consumed += len(batch)
            timings_ms.append((time.perf_counter() - t0) * 1000.0)

    total_s = time.perf_counter() - t_start

    # Check remaining lag
    info = await client.xinfo_groups(stream)
    lag = 0
    for g in info:
        if g.get("name") == group.encode() or g.get("name") == group:
            lag = int(g.get("lag", 0) or g.get("pending", 0))
            break

    return RedisStreamResult(
        stream=stream,
        operation="XREADGROUP",
        n_events=consumed,
        n_errors=n_events - consumed,
        duration_s=total_s,
        throughput_eps=consumed / total_s if total_s > 0 else 0.0,
        p50_ms=_percentile(timings_ms, 50),
        p95_ms=_percentile(timings_ms, 95),
        p99_ms=_percentile(timings_ms, 99),
        consumer_lag=lag,
    )


async def _bench_concurrent_producers(
    client: "redis.asyncio.Redis",
    stream: str,
    n_producers: int = 10,
    events_per_producer: int = 1000,
) -> RedisStreamResult:
    """Simulate N concurrent service producers writing to a stream."""
    errors = 0
    timings_ms: list[float] = []
    lock = asyncio.Lock()

    async def producer(pid: int) -> None:
        nonlocal errors
        for i in range(events_per_producer):
            t0 = time.perf_counter()
            try:
                await client.xadd(
                    stream,
                    {"producer": str(pid), "seq": str(i), "data": "x" * 256},
                    maxlen=200_000, approximate=True,
                )
                async with lock:
                    timings_ms.append((time.perf_counter() - t0) * 1000.0)
            except Exception:
                async with lock:
                    errors += 1

    t_start = time.perf_counter()
    await asyncio.gather(*[producer(p) for p in range(n_producers)])
    total_s = time.perf_counter() - t_start
    total = n_producers * events_per_producer

    return RedisStreamResult(
        stream=stream,
        operation=f"XADD_{n_producers}P",
        n_events=total,
        n_errors=errors,
        duration_s=total_s,
        throughput_eps=total / total_s if total_s > 0 else 0.0,
        p50_ms=_percentile(timings_ms, 50),
        p95_ms=_percentile(timings_ms, 95),
        p99_ms=_percentile(timings_ms, 99),
    )


async def run_all(cfg: BenchmarkConfig | None = None) -> list[RedisStreamResult]:
    cfg = cfg or get_config()
    results: list[RedisStreamResult] = []

    try:
        import redis.asyncio as aioredis  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("pip install redis[hiredis]") from exc

    client = aioredis.from_url(
        cfg.endpoints.redis_url,
        encoding="utf-8",
        decode_responses=True,
        socket_connect_timeout=3,
    )

    bench_stream = "glossa:bench:stream"
    n = cfg.redis_events

    try:
        print("\n── Redis stream benchmarks ──────────────────────────────────")

        # Single XADD
        r = await _bench_xadd_single(client, bench_stream, n)
        results.append(r)
        print(r.summary_line())

        # Pipelined XADD
        r = await _bench_xadd_pipeline(client, bench_stream, n, cfg.redis_pipeline_size)
        results.append(r)
        print(r.summary_line())

        # Consumer lag drain
        print(f"  Measuring consumer lag drain ({n // 2} events)...")
        r = await _bench_consumer_lag(client, bench_stream, n // 2)
        results.append(r)
        print(r.summary_line())

        # Concurrent producers
        print("  Concurrent producers (10 × 500 events)...")
        r = await _bench_concurrent_producers(client, bench_stream, n_producers=10, events_per_producer=500)
        results.append(r)
        print(r.summary_line())

        # Real stream names
        for stream in cfg.redis_streams[:2]:
            r = await _bench_xadd_single(client, stream, min(1000, n // 10))
            results.append(r)
            print(r.summary_line())

    finally:
        # Cleanup bench stream
        try:
            await client.delete(bench_stream)
        except Exception:
            pass
        await client.aclose()

    out = cfg.reports_dir / "redis_results.json"
    out.write_text(json.dumps([asdict(r) for r in results], indent=2))
    print(f"\nRedis results saved to {out}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=10_000)
    args = parser.parse_args()
    cfg = get_config()
    cfg.redis_events = args.events
    asyncio.run(run_all(cfg))


if __name__ == "__main__":
    main()
