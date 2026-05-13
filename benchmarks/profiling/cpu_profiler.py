"""CPU profiling — py-spy flamegraphs + cProfile integration.

Captures CPU profiles of running services during load tests.

Usage:
    # Profile api-gateway during benchmark (find its PID automatically)
    python -m benchmarks.profiling.cpu_profiler --service api-gateway --duration 30

    # Profile by PID
    python -m benchmarks.profiling.cpu_profiler --pid 1234 --duration 60

    # Profile a function inline
    from benchmarks.profiling.cpu_profiler import profile_function
    with profile_function("my_func") as prof:
        my_func()
    prof.save("report.prof")
"""

from __future__ import annotations

import cProfile
import io
import json
import os
import pstats
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

from benchmarks.config import BenchmarkConfig, get_config


@dataclass
class FlameGraphResult:
    service: str
    pid: int
    output_svg: Path
    output_json: Path | None
    duration_s: float
    top_functions: list[dict[str, Any]] = field(default_factory=list)
    success: bool = True
    error: str = ""


@dataclass
class CProfileResult:
    label: str
    total_calls: int
    total_time_s: float
    top_functions: list[dict[str, Any]] = field(default_factory=list)
    output_path: Path | None = None


# ── py-spy flamegraphs ────────────────────────────────────────────────────────

def _find_service_pid(service_name: str) -> int | None:
    """Find the PID of a running Docker container or process by service name."""
    # Try Docker container PID
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Pid}}", service_name],
            capture_output=True, text=True, timeout=5,
        )
        pid = int(result.stdout.strip())
        if pid > 0:
            return pid
    except Exception:
        pass

    # Try pgrep for uvicorn processes
    try:
        result = subprocess.run(
            ["pgrep", "-f", f"uvicorn.*{service_name.replace('-', '_')}"],
            capture_output=True, text=True, timeout=5,
        )
        pids = [int(p.strip()) for p in result.stdout.splitlines() if p.strip()]
        return pids[0] if pids else None
    except Exception:
        pass

    return None


def record_flamegraph(
    service_or_pid: str | int,
    output_dir: Path,
    duration_s: int = 30,
    rate: int = 100,       # samples/sec
    native: bool = False,  # include native C extensions
) -> FlameGraphResult:
    """Run py-spy to capture a CPU flamegraph SVG of a running process."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(service_or_pid, str):
        service = service_or_pid
        pid = _find_service_pid(service)
        if pid is None:
            return FlameGraphResult(
                service=service, pid=0,
                output_svg=output_dir / "not_found.svg",
                output_json=None, duration_s=0,
                success=False, error=f"Could not find PID for {service}",
            )
    else:
        pid = service_or_pid
        service = str(pid)

    svg_path = output_dir / f"{service.replace('-', '_')}_cpu.svg"
    json_path = output_dir / f"{service.replace('-', '_')}_cpu.json"

    # Build py-spy command
    cmd = [
        "py-spy", "record",
        "--output", str(svg_path),
        "--format", "flamegraph",
        "--pid", str(pid),
        "--duration", str(duration_s),
        "--rate", str(rate),
    ]
    if native:
        cmd.append("--native")

    # Also capture JSON for top-functions analysis
    cmd_json = [
        "py-spy", "top",
        "--pid", str(pid),
        "--duration", str(min(duration_s, 10)),
        "--nonblocking",
    ]

    print(f"  Recording {duration_s}s flamegraph for {service} (PID={pid})...")

    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration_s + 30)
        if result.returncode != 0:
            return FlameGraphResult(
                service=service, pid=pid,
                output_svg=svg_path, output_json=None,
                duration_s=time.time() - t0,
                success=False, error=result.stderr[:500],
            )
    except FileNotFoundError:
        return FlameGraphResult(
            service=service, pid=pid,
            output_svg=svg_path, output_json=None,
            duration_s=0,
            success=False, error="py-spy not found — install: pip install py-spy",
        )
    except subprocess.TimeoutExpired:
        pass

    # Parse top functions from text output
    top_functions: list[dict] = []
    try:
        top_result = subprocess.run(cmd_json, capture_output=True, text=True, timeout=15)
        for line in top_result.stdout.splitlines()[:20]:
            if "%" in line:
                parts = line.strip().split()
                if parts:
                    top_functions.append({"line": line.strip()})
    except Exception:
        pass

    duration = time.time() - t0
    print(f"  Flamegraph saved: {svg_path} ({duration:.0f}s)")
    return FlameGraphResult(
        service=service, pid=pid,
        output_svg=svg_path, output_json=json_path,
        duration_s=duration,
        top_functions=top_functions,
        success=svg_path.exists(),
    )


def profile_all_services(
    cfg: BenchmarkConfig | None = None,
    duration_s: int = 30,
) -> list[FlameGraphResult]:
    """Profile all running Glossa services concurrently."""
    cfg = cfg or get_config()
    services = ["api-gateway", "cv-service", "asr-service", "nlp-service", "tts-service"]
    results: list[FlameGraphResult] = []

    import concurrent.futures

    print(f"\n── CPU flamegraphs ({duration_s}s each) ─────────────────────────")

    def profile_one(svc: str) -> FlameGraphResult:
        return record_flamegraph(svc, cfg.flamegraphs_dir, duration_s)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(profile_one, s): s for s in services}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            status = "OK" if result.success else f"ERR: {result.error}"
            print(f"  {result.service}: {status}")
            results.append(result)

    return results


# ── cProfile inline context manager ──────────────────────────────────────────

class FunctionProfile:
    def __init__(self, label: str) -> None:
        self.label = label
        self._profiler = cProfile.Profile()
        self.result: CProfileResult | None = None

    def __enter__(self) -> "FunctionProfile":
        self._profiler.enable()
        return self

    def __exit__(self, *_: Any) -> None:
        self._profiler.disable()
        stream = io.StringIO()
        ps = pstats.Stats(self._profiler, stream=stream)
        ps.sort_stats("cumulative")

        top: list[dict] = []
        ps.print_stats(20)
        for line in stream.getvalue().splitlines():
            if ".py:" in line or "built-in" in line:
                top.append({"line": line.strip()})

        stats = ps.stats or {}
        total_time = sum(v[3] for v in stats.values()) if stats else 0.0
        total_calls = sum(v[0] for v in stats.values()) if stats else 0

        self.result = CProfileResult(
            label=self.label,
            total_calls=total_calls,
            total_time_s=total_time,
            top_functions=top[:20],
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._profiler.dump_stats(str(path))
        if self.result:
            json_path = path.with_suffix(".json")
            json_path.write_text(
                json.dumps(
                    {
                        "label": self.result.label,
                        "total_calls": self.result.total_calls,
                        "total_time_s": self.result.total_time_s,
                        "top_functions": self.result.top_functions,
                    },
                    indent=2,
                )
            )


@contextmanager
def profile_function(label: str) -> Generator[FunctionProfile, None, None]:
    prof = FunctionProfile(label)
    with prof:
        yield prof


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="CPU profiler / flamegraph recorder")
    parser.add_argument("--service", default=None, help="Service name to profile")
    parser.add_argument("--pid", type=int, default=None, help="PID to profile")
    parser.add_argument("--duration", type=int, default=30, help="Profile duration (s)")
    parser.add_argument("--all", action="store_true", help="Profile all services")
    parser.add_argument("--rate", type=int, default=100, help="Samples/sec")
    args = parser.parse_args()

    cfg = get_config()

    if args.all:
        profile_all_services(cfg, args.duration)
    elif args.pid:
        r = record_flamegraph(args.pid, cfg.flamegraphs_dir, args.duration, args.rate)
        print(f"{'OK' if r.success else 'FAILED'}: {r.output_svg}")
    elif args.service:
        r = record_flamegraph(args.service, cfg.flamegraphs_dir, args.duration, args.rate)
        print(f"{'OK' if r.success else 'FAILED'}: {r.output_svg}")
    else:
        print("Specify --service, --pid, or --all")


if __name__ == "__main__":
    main()
