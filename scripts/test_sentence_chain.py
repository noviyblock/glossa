"""Offline sentence-chain test: does a sequence of real, isolated gestures
assemble into a coherent sentence end to end -- no camera needed.

Context (punch-list plan item 3): the sentence-accumulation layer (#26) and
T9-style disambiguation endpoint (#27) have only ever been exercised one
gesture at a time (unit tests) -- never chained. This script builds a
synthetic "sentence" from N real per-gesture keypoint windows pulled from
data/gestures/processed_64_200/val (the same val split used for this
session's confusion-matrix eval), classifies each with the real ONNX/
OpenVINO classifier, and drives them through the REAL Orchestrator's
accumulation/flush logic via a fake CV-service HTTP client (same pattern as
services/api_gateway/tests/test_orchestrator_sentence.py) -- except the NLP
leg of that fake client is wired to the REAL locally-loaded Translator
(models/qwen2_merged_qwen2_1.5b is present in this environment, confirmed
before writing this script), not a canned string.

Each val-split sample is already a (64, 75, 3) resampled single-gesture
window -- exactly the shape GestureSegmenter._finish_segment() hands to
Normalizer+GestureClassifier in production -- so this deliberately does NOT
exercise GestureSegmenter's onset/offset state machine (that's already
covered by services/cv_service/tests/test_gesture_segmenter.py against
synthetic frames at a known, controlled activity scale; features.npy's
hip/shoulder-normalized coordinates are NOT in the same scale as raw
extractor output GestureSegmenter expects, so re-deriving segmentation
thresholds for this data would be a separate, unvalidated exercise -- see
punch-list plan item 3's own note on this). What's new here is everything
downstream of segmentation: classification -> accumulation -> disambiguation.

z-scoring reuses the exact validated pattern from this session's
real_confusion_matrix.py (features.npy is hip/shoulder-normalized but NOT
z-scored; calling Normalizer.__call__ here would incorrectly redo the
hip/shoulder step on already-normalized data).

Run: python scripts/test_sentence_chain.py [--n 6] [--seed 0] [--skip-nlp]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Windows consoles default stdout to cp1252, which can't encode Cyrillic
# gloss names -- this script's whole point is printing them.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]


def _load_service_module(service_dir: Path, module_name: str):
    """Import `module_name` from a flat-file service directory, with that
    service's own config.py bound as `sys.modules['config']` for the
    duration of the import. Each service (cv_service/api_gateway/
    nlp_service) has its OWN config.py -- importing more than one service's
    modules in the same process needs this staging, or the second service's
    `from config import X` would silently resolve against whichever
    config.py got cached under the shared name "config" first. Once a
    module executes its top-level `from config import X`, the value is
    copied into that module's own namespace -- safe to swap `config` out
    again immediately after.
    """
    # Each service also defines its own module-level Prometheus Histogram
    # under the SAME metric name (e.g. glossa_model_inference_latency_seconds
    # in both gesture_classifier.py and translator.py), registered against
    # prometheus_client's global default registry -- harmless in production
    # (separate processes, separate registries) but a hard ValueError when
    # two services' modules load in one process, as this script deliberately
    # does. Histogram()'s registry default is bound once at prometheus_client
    # import time (a plain default-argument value, not a live module lookup)
    # -- reassigning prometheus_client.REGISTRY doesn't reach it, so the fix
    # has to mutate the actual registry object in place instead. This script
    # never scrapes metrics, so dropping prior services' registrations here
    # is fine.
    from prometheus_client import REGISTRY as _prom_registry
    for _collector in list(_prom_registry._collector_to_names):
        _prom_registry.unregister(_collector)

    sys.modules.pop("config", None)
    sys.path.insert(0, str(service_dir))
    try:
        import config as _cfg  # noqa: F401  -- import binds sys.modules['config']
        module = __import__(module_name)
        return module
    finally:
        sys.path.remove(str(service_dir))
        sys.modules.pop("config", None)


def _pick_samples(n: int, seed: int) -> list[tuple[np.ndarray, str]]:
    val_dir = ROOT / "data" / "gestures" / "processed_64_200" / "val"
    features = np.load(val_dir / "features.npy")
    labels = np.load(val_dir / "labels.npy")
    class_names = json.loads((val_dir.parent / "class_names.json").read_text(encoding="utf-8"))

    rng = np.random.default_rng(seed)
    unique_labels = np.unique(labels)
    chosen_labels = rng.choice(unique_labels, size=min(n, len(unique_labels)), replace=False)

    samples = []
    for label in chosen_labels:
        idx = int(np.flatnonzero(labels == label)[0])
        samples.append((features[idx].astype(np.float32), class_names[int(label)]))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-nlp", action="store_true")
    args = parser.parse_args()

    print(f"Picking {args.n} real gesture windows from processed_64_200/val (seed={args.seed})...")
    samples = _pick_samples(args.n, args.seed)
    true_sequence = [name for _, name in samples]
    print("Ground truth gloss sequence:", true_sequence)

    # ── Stage 1: cv_service -- classify each window with the real classifier ──
    cv_dir = ROOT / "services" / "cv_service"
    gesture_classifier_mod = _load_service_module(cv_dir, "gesture_classifier")

    stats = np.load(ROOT / "models" / "norm_stats.npz")
    mean, std = stats["mean"][0], stats["std"][0]
    std = np.where(std < 1e-6, 1.0, std)

    classifier = gesture_classifier_mod.GestureClassifier(
        ov_xml_path=str(ROOT / "models" / "stgcn_topk_int8" / "stgcn_topk_int8.xml"),
        onnx_path=str(ROOT / "models" / "gesture_classifier_mobile.onnx"),
        class_map_path=str(ROOT / "data" / "idx_to_class.json"),
    )

    positions: list[list[dict[str, Any]]] = []
    top1_sequence: list[str] = []
    for window, true_name in samples:
        norm_win = ((window - mean) / std).astype(np.float32)
        top3 = classifier.predict_top3(norm_win)
        positions.append([dict(r) for r in top3])
        top1_sequence.append(top3[0]["gloss"])
        marker = "OK" if top3[0]["gloss"] == true_name else "MISS"
        print(f"  true={true_name!r:20s} top1={top3[0]['gloss']!r:20s} "
              f"prob={top3[0]['prob']:.2f}  [{marker}]")

    n_correct = sum(1 for t, p in zip(true_sequence, top1_sequence) if t == p)
    print(f"\nPer-gesture top1 accuracy on this sample: {n_correct}/{len(samples)}")

    # ── Stage 2: api_gateway -- real Orchestrator accumulation/flush logic ──
    gw_dir = ROOT / "services" / "api_gateway"
    orchestrator_mod = _load_service_module(gw_dir, "orchestrator")
    gw_config_mod = _load_service_module(gw_dir, "config")  # for URL constants only

    translator = None
    if not args.skip_nlp:
        nlp_dir = ROOT / "services" / "nlp_service"
        print("\nLoading local NLP model (models/qwen2_merged_qwen2_1.5b)...")
        translator_mod = _load_service_module(nlp_dir, "translator")
        translator = translator_mod.Translator(model_path=str(ROOT / "models" / "qwen2_merged_qwen2_1.5b"))

    class _FakeResponse:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return self._payload

    class _FakeRedis:
        def __init__(self) -> None:
            self._store: dict[str, str] = {}

        async def setex(self, key: str, ttl: int, value: str) -> None:
            self._store[key] = value

        async def get(self, key: str) -> str | None:
            return self._store.get(key)

        async def delete(self, key: str) -> None:
            self._store.pop(key, None)

        async def xadd(self, *a: Any, **kw: Any) -> None:
            pass

    cv_url = gw_config_mod.CV_SERVICE_URL
    nlp_url = gw_config_mod.NLP_SERVICE_URL
    remaining_frames = list(positions)

    class _ChainHTTPClient:
        """CV leg returns the pre-classified glosses in order (one per
        process_frame call, mirroring one completed gesture per frame here
        since segmentation is out of scope -- see module docstring). NLP
        leg invokes the real local Translator in-process instead of a
        canned string."""

        async def post(self, url: str, json: dict[str, Any] | None = None) -> _FakeResponse:
            if url == f"{cv_url}/process_frame":
                glosses = remaining_frames.pop(0)
                return _FakeResponse({
                    "glosses": glosses, "person_detected": True,
                    "gesture_active": False, "preview": False,
                })
            if url == f"{nlp_url}/translate_sequence_topk":
                if translator is None:
                    return _FakeResponse({"translation": " ".join(p[0]["gloss"] for p in json["positions"])})
                text = translator.translate_sequence_topk(json["positions"], json.get("context"))
                return _FakeResponse({"translation": text})
            raise AssertionError(f"unexpected call: POST {url}")

    orch = orchestrator_mod.Orchestrator(redis=_FakeRedis(), http=_ChainHTTPClient())

    import asyncio

    async def _run_chain() -> str:
        result: dict[str, Any] = {}
        for _ in positions:
            result = await orch.process_frame("test-sentence-chain", "fake-frame-b64")
        # Force-flush whatever's left buffered (fewer than MAX_SENTENCE_GLOSSES,
        # no pause elapsed in this synthetic run) so the test doesn't depend on
        # wall-clock timing for its final assertion.
        if not result.get("translation"):
            flush = await orch.flush_pending_sentence("test-sentence-chain")
            return flush["translation"]
        return result["translation"]

    print("\nRunning through Orchestrator.process_frame() (real accumulation/flush logic)...")
    sentence = asyncio.run(_run_chain())

    print(f"\nRaw top1 sequence:    {' '.join(top1_sequence)}")
    print(f"Assembled sentence:   {sentence!r}")
    if args.skip_nlp:
        print("(--skip-nlp: this is the joined-top1 fallback, not real disambiguation)")


if __name__ == "__main__":
    main()
