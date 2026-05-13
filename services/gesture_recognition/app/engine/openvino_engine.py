"""OpenVINO inference engine — optimal for Intel CPU/iGPU/NPU.

OpenVINO advantages over plain ONNX Runtime on Intel hardware:
  - Automatic INT8 calibration via Post-Training Optimization Toolkit
  - Multi-stream inference via PERFORMANCE_HINT = THROUGHPUT
  - NPU support on Intel Core Ultra (Meteor Lake+)
  - Static shape compilation for maximum speed

Device plugins (spec.extra["device"]):
  "CPU"   — Intel CPU with AVX2/AVX-512 kernels
  "GPU"   — Intel iGPU (Gen9+)
  "NPU"   — Intel Neural Processing Unit (available via AUTO)
  "AUTO"  — Automatic best-device selection at runtime

Throughput vs Latency modes:
  For streaming (low-latency), use PERFORMANCE_HINT = LATENCY.
  For batch processing, use PERFORMANCE_HINT = THROUGHPUT.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from glossa_common.logging import get_logger

from ..domain.entities import InferenceDevice, ModelSpec
from .base import BaseInferenceEngine

logger = get_logger(__name__)

_DEFAULT_CONFIG: dict[str, Any] = {
    "PERFORMANCE_HINT": "LATENCY",          # minimize single-request latency
    "NUM_STREAMS": "1",                     # 1 stream for latency mode
    "INFERENCE_PRECISION_HINT": "f32",      # switch to f16 for iGPU
}


class OpenVINOInferenceEngine(BaseInferenceEngine):
    """OpenVINO inference engine with async infer request pool."""

    def __init__(
        self,
        spec: ModelSpec,
        labels: list[str],
        feature_dim: int = 258,
        n_threads: int | None = None,
    ) -> None:
        super().__init__(spec, labels, feature_dim, n_threads)
        self._compiled_model: object = None
        self._input_key: object = None
        self._output_key: object = None

    async def _load(self) -> None:
        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        from openvino.runtime import Core, AsyncInferQueue

        device = self._spec.extra.get("device", "CPU")
        perf_hint = self._spec.extra.get("performance_hint", "LATENCY")

        config = {**_DEFAULT_CONFIG, "PERFORMANCE_HINT": perf_hint}

        # Throughput mode → more streams
        if perf_hint == "THROUGHPUT":
            config["NUM_STREAMS"] = "AUTO"

        core = Core()
        model = core.read_model(str(self._spec.path))

        # Reshape to static for maximum speed (if window_size known)
        batch_size = self._spec.extra.get("batch_size", 1)
        seq_len = self._spec.extra.get("sequence_length", 30)
        try:
            model.reshape({model.input(0).get_any_name(): [batch_size, seq_len, self._feature_dim]})
            logger.info("openvino_model_reshaped", shape=[batch_size, seq_len, self._feature_dim])
        except Exception:
            logger.debug("openvino_using_dynamic_shape", model_id=self.model_id)

        self._compiled_model = core.compile_model(model, device, config)
        self._input_key = self._compiled_model.input(0)
        self._output_key = self._compiled_model.output(0)

        logger.info(
            "openvino_engine_loaded",
            model_id=self.model_id,
            device=device,
            performance_hint=perf_hint,
            model_path=str(self._spec.path),
        )
        self._ready = True

    def _run_inference_sync(self, batch: np.ndarray) -> np.ndarray:
        compiled = self._compiled_model
        infer_req = compiled.create_infer_request()  # type: ignore[union-attr]
        infer_req.infer({self._input_key: batch.astype(np.float32)})
        logits: np.ndarray = infer_req.get_output_tensor(0).data.copy()

        if logits.ndim == 3:
            logits = logits.squeeze(-1) if logits.shape[-1] == 1 else logits.squeeze(1)

        return logits

    def get_supported_properties(self) -> dict:
        if not self._ready or self._compiled_model is None:
            return {}
        try:
            return dict(self._compiled_model.get_property("SUPPORTED_PROPERTIES"))  # type: ignore[arg-type]
        except Exception:
            return {}


class OpenVINOEngineFactory:
    @staticmethod
    async def create(
        spec: ModelSpec,
        labels: list[str],
        feature_dim: int = 258,
        n_threads: int | None = None,
    ) -> OpenVINOInferenceEngine:
        engine = OpenVINOInferenceEngine(
            spec=spec, labels=labels, feature_dim=feature_dim, n_threads=n_threads
        )
        await engine._load()
        return engine
