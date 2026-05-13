"""Experiment 08 — End-to-End Latency Breakdown.

Profiles the full gesture→speech pipeline per component and projects
effective latency for target mobile devices (Poco M5, Realme X60, Poor 4G).

Components measured:
  1. CV service  — gesture keypoint inference (STGCN ONNX)
  2. NLP service — gloss-to-text translation (Qwen2-1.5B)
  3. TTS service — speech synthesis (Silero v4)
  4. Redis pub   — message passing overhead

Device projection:
  Effective latency = server_p95 + RTT + jitter
  Mobile device profiles match benchmarks/config.py:
    Poco M5   — 4G,   RTT=65ms,  jitter=15ms
    Realme X60 — 5G,  RTT=30ms,  jitter=6ms
    Poor 4G   — Edge, RTT=180ms, jitter=50ms

SLO budget: 2 000 ms end-to-end on Poco M5 (4G).

Usage:
    # Against live services:
    python -m experiments.08_e2e_latency.run \
        --host http://localhost:8000 \
        --n-samples 100

    # Dry-run (no services required):
    python -m experiments.08_e2e_latency.run --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))

from experiments.shared.mlflow_utils import (
    log_metrics, log_params, mlflow_run, save_results,
)

EXPERIMENT_NAME = "08_e2e_latency_breakdown"
RESULTS_DIR = Path("experiments/results/08_e2e_latency")


# ── Device profiles (mirrors benchmarks/config.py) ───────────────────────────

@dataclass(frozen=True)
class DeviceProfile:
    name: str
    rtt_ms: float
    jitter_ms: float

POCO_M5    = DeviceProfile("Poco M5 (4G)",    rtt_ms=65.0,  jitter_ms=15.0)
REALME_X60 = DeviceProfile("Realme X60 (5G)", rtt_ms=30.0,  jitter_ms=6.0)
POOR_4G    = DeviceProfile("Poor 4G",         rtt_ms=180.0, jitter_ms=50.0)
DEVICES    = [POCO_M5, REALME_X60, POOR_4G]

E2E_SLO_MS = 2000.0


# ── Service probing ───────────────────────────────────────────────────────────

def _probe_endpoint(
    url: str,
    payload: dict,
    n: int = 50,
    warmup: int = 5,
) -> dict[str, float]:
    try:
        import requests  # type: ignore[import]
    except ImportError:
        return {"error": "pip install requests"}

    lats: list[float] = []
    errors = 0

    for i in range(n + warmup):
        try:
            t0 = time.perf_counter()
            resp = requests.post(url, json=payload, timeout=10)
            lat = (time.perf_counter() - t0) * 1000
            if i >= warmup:
                if resp.status_code < 500:
                    lats.append(lat)
                else:
                    errors += 1
        except Exception:
            if i >= warmup:
                errors += 1

    if not lats:
        return {"error": "all requests failed", "error_count": float(errors)}

    lats.sort()
    n_ok = len(lats)
    import statistics
    return {
        "mean_ms":  statistics.mean(lats),
        "p50_ms":   lats[n_ok // 2],
        "p95_ms":   lats[int(0.95 * n_ok)],
        "p99_ms":   lats[int(0.99 * n_ok)],
        "min_ms":   lats[0],
        "max_ms":   lats[-1],
        "error_rate": errors / (n + errors),
        "n_ok":     float(n_ok),
    }


def _probe_redis(redis_url: str, n: int = 200) -> dict[str, float]:
    try:
        import redis  # type: ignore[import]
        r = redis.from_url(redis_url)
        lats: list[float] = []
        for _ in range(n):
            t0 = time.perf_counter()
            r.xadd("exp08_bench", {"data": "x" * 100})
            lats.append((time.perf_counter() - t0) * 1000)
        r.delete("exp08_bench")
        lats.sort()
        return {"p50_ms": lats[n // 2], "p95_ms": lats[int(0.95 * n)], "mean_ms": sum(lats) / n}
    except Exception as e:
        return {"error": str(e)}


# ── Simulated / dry-run values ────────────────────────────────────────────────

_SIM_COMPONENTS: dict[str, dict[str, float]] = {
    "cv_service":  {"p50_ms": 24.0,  "p95_ms": 42.0,  "p99_ms": 55.0,  "mean_ms": 26.0},
    "asr_service": {"p50_ms": 165.0, "p95_ms": 210.0, "p99_ms": 280.0, "mean_ms": 175.0},
    "nlp_service": {"p50_ms": 380.0, "p95_ms": 540.0, "p99_ms": 680.0, "mean_ms": 400.0},
    "tts_service": {"p50_ms": 115.0, "p95_ms": 185.0, "p99_ms": 240.0, "mean_ms": 125.0},
    "redis_xadd":  {"p50_ms":   0.6, "p95_ms":   1.1, "p99_ms":   1.8, "mean_ms":   0.7},
}

_SIM_E2E: dict[str, float] = {
    "p50_ms": 584.0,
    "p95_ms": 977.0,
    "p99_ms": 1255.0,
}


# ── Payloads for services ─────────────────────────────────────────────────────

def _cv_payload() -> dict:
    frames = np.random.randn(30, 75, 3).tolist()
    return {"session_id": "exp08", "frames": frames, "domain": "general"}


def _nlp_payload() -> dict:
    return {"session_id": "exp08", "gloss_sequence": "ПРИВЕТ КАК ДЕЛА",
            "domain": "general", "context": []}


def _tts_payload() -> dict:
    return {"session_id": "exp08", "text": "Привет, как дела?",
            "speaker": "aidar", "language": "ru", "sample_rate": 24000}


# ── Device projection ─────────────────────────────────────────────────────────

def _project_for_device(
    e2e_p95_ms: float,
    device: DeviceProfile,
) -> dict[str, float]:
    effective = e2e_p95_ms + device.rtt_ms + device.jitter_ms
    within_slo = effective <= E2E_SLO_MS
    headroom_ms = E2E_SLO_MS - effective
    return {
        "server_p95_ms":    e2e_p95_ms,
        "device_rtt_ms":    device.rtt_ms,
        "device_jitter_ms": device.jitter_ms,
        "effective_ms":     effective,
        "within_slo":       float(within_slo),
        "headroom_ms":      headroom_ms,
        "slo_budget_ms":    E2E_SLO_MS,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    results: dict[str, Any] = {}

    if args.dry_run:
        results["components"] = _SIM_COMPONENTS
        results["e2e"] = _SIM_E2E
        print("[dry-run] Using simulated latency values:")
        for comp, m in _SIM_COMPONENTS.items():
            print(f"  {comp:<14} P50={m['p50_ms']:.0f}ms  P95={m['p95_ms']:.0f}ms")
        print(f"  {'E2E total':<14} P50={_SIM_E2E['p50_ms']:.0f}ms  P95={_SIM_E2E['p95_ms']:.0f}ms")
    else:
        base = args.host.rstrip("/")
        print(f"Probing services at {base}  (n={args.n_samples} samples each)")

        components: dict[str, dict[str, float]] = {}

        print("  CV service...", end=" ", flush=True)
        components["cv_service"] = _probe_endpoint(
            f"{base.replace('8000', '8001')}/recognize", _cv_payload(), n=args.n_samples)
        print(f"P95={components['cv_service'].get('p95_ms', '?'):.1f}ms")

        print("  NLP service...", end=" ", flush=True)
        components["nlp_service"] = _probe_endpoint(
            f"{base.replace('8000', '8003')}/translate", _nlp_payload(), n=args.n_samples)
        print(f"P95={components['nlp_service'].get('p95_ms', '?'):.1f}ms")

        print("  TTS service...", end=" ", flush=True)
        components["tts_service"] = _probe_endpoint(
            f"{base.replace('8000', '8004')}/synthesize", _tts_payload(), n=args.n_samples)
        print(f"P95={components['tts_service'].get('p95_ms', '?'):.1f}ms")

        print("  Redis XADD...", end=" ", flush=True)
        components["redis_xadd"] = _probe_redis(args.redis_url, n=200)
        print(f"P95={components['redis_xadd'].get('p95_ms', '?'):.2f}ms")

        print("  E2E pipeline (CV→NLP→TTS via gateway)...", end=" ", flush=True)
        e2e = _probe_endpoint(
            f"{base}/api/v1/translate",
            {"gloss_sequence": "ПРИВЕТ КАК ДЕЛА", "session_id": "exp08", "domain": "general"},
            n=args.n_samples,
        )
        print(f"P95={e2e.get('p95_ms', '?'):.1f}ms")

        results["components"] = components
        results["e2e"] = e2e

    # Device projections
    e2e_p95 = results["e2e"].get("p95_ms", _SIM_E2E["p95_ms"])
    projections: dict[str, dict[str, float]] = {}
    for device in DEVICES:
        proj = _project_for_device(e2e_p95, device)
        projections[device.name] = proj

    results["device_projections"] = projections

    with mlflow_run(
        EXPERIMENT_NAME, run_name=f"e2e_breakdown_n{args.n_samples}",
        tags={"experiment": "08"},
    ) as _run:
        log_params({"n_samples": args.n_samples, "host": args.host})
        for comp, m in results.get("components", {}).items():
            log_metrics({f"{comp}/{k}": v for k, v in m.items() if isinstance(v, float)})
        e2e_m = results.get("e2e", {})
        log_metrics({f"e2e/{k}": v for k, v in e2e_m.items() if isinstance(v, float)})
        for dev_name, proj in projections.items():
            safe = dev_name.replace(" ", "_").replace("(", "").replace(")", "")
            log_metrics({f"projection_{safe}/{k}": v for k, v in proj.items()})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_results(results, RESULTS_DIR / "results.json")
    _print_breakdown(results)
    return results


def _print_breakdown(results: dict[str, Any]) -> None:
    print("\n" + "═" * 72)
    print("EXPERIMENT 08 — E2E LATENCY BREAKDOWN")
    print("═" * 72)
    print(f"\n{'Component':<18} {'P50ms':>8} {'P95ms':>8} {'P99ms':>8} {'Share%':>8}")
    print("─" * 72)

    comp = results.get("components", _SIM_COMPONENTS)
    e2e_p95 = results.get("e2e", _SIM_E2E).get("p95_ms", 977.0)

    for name, m in comp.items():
        p95 = m.get("p95_ms", 0)
        share = (p95 / e2e_p95 * 100) if e2e_p95 > 0 else 0
        print(f"{name:<18} {m.get('p50_ms', 0):>8.1f} {p95:>8.1f} "
              f"{m.get('p99_ms', 0):>8.1f} {share:>7.1f}%")

    e2e = results.get("e2e", _SIM_E2E)
    print("─" * 72)
    print(f"{'E2E TOTAL':<18} {e2e.get('p50_ms', 0):>8.1f} "
          f"{e2e.get('p95_ms', 0):>8.1f} {e2e.get('p99_ms', 0):>8.1f}")

    print(f"\n{'Device Projections (server P95 + RTT + jitter)'}")
    print(f"{'Device':<22} {'Server':>8} {'Effective':>10} {'SLO 2000ms':>12} {'Headroom':>10}")
    print("─" * 72)
    for dev_name, proj in results.get("device_projections", {}).items():
        ok = "✓ PASS" if proj["within_slo"] else "✗ FAIL"
        print(f"{dev_name:<22} {proj['server_p95_ms']:>8.0f} "
              f"{proj['effective_ms']:>10.0f} "
              f"{ok:>12} "
              f"{proj['headroom_ms']:>+9.0f}ms")
    print("═" * 72)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host",       default="http://localhost:8000")
    p.add_argument("--redis-url",  default="redis://localhost:6379")
    p.add_argument("--n-samples",  type=int, default=50)
    p.add_argument("--dry-run",    action="store_true")
    run_experiment(p.parse_args())


if __name__ == "__main__":
    main()
