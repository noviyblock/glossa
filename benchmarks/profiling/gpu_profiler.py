"""GPU utilization and VRAM profiling during load tests.

Samples GPU metrics every N seconds and produces a utilization timeline.

Usage:
    python -m benchmarks.profiling.gpu_profiler --duration 60 --interval 0.5
    # Runs alongside a load test to capture GPU impact
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class GPUSample:
    timestamp: float
    gpu_index: int
    utilization_pct: float
    memory_used_mb: float
    memory_total_mb: float
    memory_free_mb: float
    temperature_c: float
    power_w: float
    sm_clock_mhz: float


@dataclass
class GPUProfile:
    duration_s: float
    interval_s: float
    gpu_count: int
    samples: list[GPUSample] = field(default_factory=list)
    available: bool = True
    error: str = ""

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {"available": self.available, "error": self.error}
        utils = [s.utilization_pct for s in self.samples]
        mems = [s.memory_used_mb for s in self.samples]
        return {
            "gpu_count": self.gpu_count,
            "duration_s": self.duration_s,
            "n_samples": len(self.samples),
            "util_mean_pct": sum(utils) / len(utils),
            "util_max_pct": max(utils),
            "util_p95_pct": sorted(utils)[int(0.95 * len(utils))],
            "mem_mean_mb": sum(mems) / len(mems),
            "mem_peak_mb": max(mems),
            "mem_total_mb": self.samples[0].memory_total_mb if self.samples else 0,
        }


def _try_nvml() -> Any | None:
    try:
        import pynvml  # type: ignore[import]
        pynvml.nvmlInit()
        return pynvml
    except Exception:
        return None


def _try_torch() -> Any | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch
    except ImportError:
        pass
    return None


def sample_gpu_once(nvml: Any | None = None) -> list[GPUSample]:
    """Capture a single GPU snapshot across all devices."""
    samples: list[GPUSample] = []
    now = time.time()

    if nvml:
        try:
            n = nvml.nvmlDeviceGetCount()
            for i in range(n):
                h = nvml.nvmlDeviceGetHandleByIndex(i)
                util = nvml.nvmlDeviceGetUtilizationRates(h)
                mem = nvml.nvmlDeviceGetMemoryInfo(h)
                temp = nvml.nvmlDeviceGetTemperature(h, nvml.NVML_TEMPERATURE_GPU)
                try:
                    power = nvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                except Exception:
                    power = 0.0
                try:
                    clock = nvml.nvmlDeviceGetClockInfo(h, nvml.NVML_CLOCK_SM)
                except Exception:
                    clock = 0
                samples.append(GPUSample(
                    timestamp=now,
                    gpu_index=i,
                    utilization_pct=float(util.gpu),
                    memory_used_mb=mem.used / (1024 ** 2),
                    memory_total_mb=mem.total / (1024 ** 2),
                    memory_free_mb=mem.free / (1024 ** 2),
                    temperature_c=float(temp),
                    power_w=power,
                    sm_clock_mhz=float(clock),
                ))
        except Exception:
            pass
        return samples

    # Fallback: torch.cuda
    torch = _try_torch()
    if torch:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_used = torch.cuda.memory_allocated(i) / (1024 ** 2)
            mem_total = props.total_memory / (1024 ** 2)
            samples.append(GPUSample(
                timestamp=now,
                gpu_index=i,
                utilization_pct=0.0,  # torch doesn't expose utilization %
                memory_used_mb=mem_used,
                memory_total_mb=mem_total,
                memory_free_mb=mem_total - mem_used,
                temperature_c=0.0,
                power_w=0.0,
                sm_clock_mhz=0.0,
            ))
    return samples


async def profile_gpu(
    duration_s: float = 60.0,
    interval_s: float = 0.5,
    output_path: Path | None = None,
) -> GPUProfile:
    """Sample GPU metrics for `duration_s` seconds at `interval_s` intervals."""
    nvml = _try_nvml()
    torch = _try_torch()

    if not nvml and not torch:
        return GPUProfile(
            duration_s=duration_s,
            interval_s=interval_s,
            gpu_count=0,
            available=False,
            error="No GPU or NVML/PyTorch not available",
        )

    gpu_count = 0
    if nvml:
        try:
            gpu_count = nvml.nvmlDeviceGetCount()
        except Exception:
            pass
    elif torch:
        gpu_count = torch.cuda.device_count()

    print(f"  Profiling {gpu_count} GPU(s) for {duration_s:.0f}s @ {interval_s}s intervals...")

    profile = GPUProfile(duration_s=duration_s, interval_s=interval_s, gpu_count=gpu_count)
    t_start = time.time()

    while time.time() - t_start < duration_s:
        samples = sample_gpu_once(nvml)
        profile.samples.extend(samples)
        await asyncio.sleep(interval_s)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({
            "summary": profile.summary(),
            "samples": [asdict(s) for s in profile.samples[:500]],  # cap at 500 rows
        }, indent=2))
        print(f"  GPU profile saved: {output_path}")

    summary = profile.summary()
    if "util_mean_pct" in summary:
        print(
            f"  GPU util: mean={summary['util_mean_pct']:.1f}%  "
            f"max={summary['util_max_pct']:.1f}%  "
            f"mem_peak={summary['mem_peak_mb']:.0f}MB"
        )

    if nvml:
        try:
            nvml.nvmlShutdown()
        except Exception:
            pass

    return profile


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="GPU profiler")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--output", default="benchmarks/reports/gpu_profile.json")
    args = parser.parse_args()
    asyncio.run(profile_gpu(args.duration, args.interval, Path(args.output)))


if __name__ == "__main__":
    main()
