"""Automated benchmarking pipeline — latency, throughput, memory across all models.

Usage:
    python -m mlops.pipelines.benchmark
    python -m mlops.pipelines.benchmark --model gesture --backends onnx openvino
    python -m mlops.pipelines.benchmark --compare-production
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

from mlops.config import MLOpsSettings, get_settings
from mlops.tracking.experiment_tracker import ExperimentTracker
from mlops.tracking.metrics import LatencyProfile, MemoryProfile, profile_latency, profile_memory


@dataclass
class BenchmarkEntry:
    model_name: str
    backend: str
    device: str
    batch_size: int
    latency: LatencyProfile
    memory: MemoryProfile
    onnx_path: str
    run_id: str = ""


def _make_gesture_inputs(batch_size: int, seq_len: int = 30) -> np.ndarray:
    return np.random.randn(batch_size, seq_len * 33 * 3).astype(np.float32)


def _bench_onnx(
    model_path: Path,
    input_fn: Any,
    batch_size: int,
    n_samples: int,
    warmup_runs: int,
) -> tuple[LatencyProfile, MemoryProfile]:
    try:
        import onnxruntime as ort  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("pip install onnxruntime") from exc

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    session = ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    data = input_fn(batch_size)

    latency = profile_latency(
        fn=lambda: session.run(None, {input_name: data}),
        n_samples=n_samples,
        warmup_runs=warmup_runs,
        batch_size=batch_size,
    )
    memory, _ = profile_memory(lambda: session.run(None, {input_name: data}))
    return latency, memory


def _bench_openvino(
    model_path: Path,
    input_fn: Any,
    batch_size: int,
    n_samples: int,
    warmup_runs: int,
) -> tuple[LatencyProfile, MemoryProfile] | None:
    try:
        from openvino.runtime import Core  # type: ignore[import]
    except ImportError:
        return None

    core = Core()
    model = core.read_model(str(model_path))
    compiled = core.compile_model(model, "CPU")
    infer_req = compiled.create_infer_request()
    data = input_fn(batch_size)

    def _infer() -> None:
        infer_req.infer({0: data})

    latency = profile_latency(_infer, n_samples=n_samples, warmup_runs=warmup_runs, batch_size=batch_size)
    memory, _ = profile_memory(_infer)
    return latency, memory


def _quantize_onnx_int8(model_path: Path) -> Path | None:
    """Return path to INT8-quantized ONNX model, or None on failure."""
    out_path = model_path.parent / (model_path.stem + "_int8.onnx")
    if out_path.exists():
        return out_path
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType  # type: ignore[import]
        quantize_dynamic(str(model_path), str(out_path), weight_type=QuantType.QInt8)
        return out_path
    except Exception:
        return None


def run_gesture_benchmarks(
    cfg: MLOpsSettings,
    tracker: ExperimentTracker,
    backends: list[str],
    batch_sizes: list[int],
) -> list[BenchmarkEntry]:
    model_path = cfg.models_path / "gesture_classifier.onnx"
    if not model_path.exists():
        print(f"[benchmark] Skipping gesture: model not found at {model_path}")
        return []

    results: list[BenchmarkEntry] = []
    params = _load_params()
    seq_len = params.get("gesture", {}).get("sequence_length", 30)

    input_fn = lambda bs: _make_gesture_inputs(bs, seq_len)  # noqa: E731

    for batch_size in batch_sizes:
        for backend in backends:
            tag = f"{backend}/batch{batch_size}"
            print(f"[benchmark] gesture {tag}")

            try:
                if backend == "onnx":
                    latency, memory = _bench_onnx(model_path, input_fn, batch_size, cfg.benchmark_n_samples, cfg.benchmark_warmup_runs)
                elif backend == "onnx_int8":
                    int8_path = _quantize_onnx_int8(model_path)
                    if int8_path is None:
                        print(f"  INT8 quantization failed, skipping")
                        continue
                    latency, memory = _bench_onnx(int8_path, input_fn, batch_size, cfg.benchmark_n_samples, cfg.benchmark_warmup_runs)
                elif backend == "openvino":
                    result = _bench_openvino(model_path, input_fn, batch_size, cfg.benchmark_n_samples, cfg.benchmark_warmup_runs)
                    if result is None:
                        print(f"  OpenVINO not available, skipping")
                        continue
                    latency, memory = result
                else:
                    print(f"  Unknown backend: {backend}")
                    continue
            except Exception as exc:
                print(f"  Error: {exc}")
                continue

            with tracker.start_run(
                experiment=cfg.benchmark_experiment,
                run_name=f"bench_gesture_{backend}_bs{batch_size}_{int(time.time())}",
                tags={
                    "model": "gesture",
                    "backend": backend,
                    "batch_size": str(batch_size),
                    "device": "CPU",
                },
            ) as run:
                tracker.log_metrics({
                    **latency.to_mlflow_dict(),
                    **memory.to_mlflow_dict(),
                })
                entry = BenchmarkEntry(
                    model_name="gesture",
                    backend=backend,
                    device="CPU",
                    batch_size=batch_size,
                    latency=latency,
                    memory=memory,
                    onnx_path=str(model_path),
                    run_id=run.info.run_id,
                )
                results.append(entry)
                print(
                    f"  p50={latency.p50_ms:.1f}ms "
                    f"p95={latency.p95_ms:.1f}ms "
                    f"rps={latency.throughput_rps:.1f} "
                    f"mem={memory.peak_rss_mb:.0f}MB"
                )

    return results


def _load_params(params_path: str = "params.yaml") -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import]
        with open(params_path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def run(
    models: list[str] | None = None,
    backends: list[str] | None = None,
    batch_sizes: list[int] | None = None,
    compare_production: bool = False,
) -> dict[str, Any]:
    cfg = get_settings()
    tracker = ExperimentTracker(cfg)
    reports_path = cfg.reports_path
    reports_path.mkdir(parents=True, exist_ok=True)

    _models = models or ["gesture"]
    _backends = backends or cfg.benchmark_backends
    _batch_sizes = batch_sizes or cfg.benchmark_batch_sizes

    all_results: dict[str, list[dict[str, Any]]] = {}

    if "gesture" in _models:
        entries = run_gesture_benchmarks(cfg, tracker, _backends, _batch_sizes)
        all_results["gesture"] = [
            {
                "backend": e.backend,
                "batch_size": e.batch_size,
                **e.latency.to_mlflow_dict(),
                **e.memory.to_mlflow_dict(),
                "run_id": e.run_id,
            }
            for e in entries
        ]

    # Write DVC metrics file
    flat_report: dict[str, Any] = {}
    for model_key, entries in all_results.items():
        for e in entries:
            prefix = f"{model_key}_{e['backend']}_bs{e['batch_size']}"
            flat_report[f"{prefix}_p95_ms"] = e.get("latency_p95_ms")
            flat_report[f"{prefix}_rps"] = e.get("throughput_rps")
            flat_report[f"{prefix}_mem_mb"] = e.get("peak_rss_mb")

    (reports_path / "benchmark_results.json").write_text(
        json.dumps({"summary": flat_report, "detail": all_results}, indent=2)
    )

    # Emit Prometheus gauges (scraped by Prometheus if this runs as a service)
    _emit_prometheus(all_results)

    print(f"Benchmark complete. Results: {reports_path / 'benchmark_results.json'}")
    return flat_report


def _emit_prometheus(results: dict[str, list[dict[str, Any]]]) -> None:
    try:
        from glossa_common.telemetry import BENCHMARK_LATENCY_P95, BENCHMARK_MEMORY_MB  # type: ignore[import]
        for model_key, entries in results.items():
            for e in entries:
                backend = e.get("backend", "unknown")
                bs = str(e.get("batch_size", 1))
                p95 = e.get("latency_p95_ms")
                mem = e.get("peak_rss_mb")
                if p95 is not None:
                    BENCHMARK_LATENCY_P95.labels(model=model_key, backend=backend, batch_size=bs).set(p95)
                if mem is not None:
                    BENCHMARK_MEMORY_MB.labels(model=model_key, backend=backend, batch_size=bs).set(mem)
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated benchmarking")
    parser.add_argument("--model", nargs="+", default=["gesture"], choices=["gesture", "nlp", "asr"])
    parser.add_argument("--backends", nargs="+", default=None)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--compare-production", action="store_true")
    args = parser.parse_args()
    run(
        models=args.model,
        backends=args.backends,
        batch_sizes=args.batch_sizes,
        compare_production=args.compare_production,
    )
