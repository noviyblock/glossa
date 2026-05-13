"""ONNX Runtime inference engine — CPU-optimized with optional CUDA.

CPU tuning strategy:
  - ORT_ENABLE_ALL graph optimization
  - Arena memory allocator to minimize malloc calls
  - intra_op_num_threads = physical cores (not hyperthreads)
  - Disable global thread pool to prevent contention
  - Memory pattern + CPU memory arena

INT8 / quantized model support:
  ONNX quantized models (.quant.onnx) are auto-detected by filename suffix.
  Dynamic quantization can be applied on-the-fly to float32 models.

Providers priority:
  CPU only:  ["CPUExecutionProvider"]
  GPU:       ["CUDAExecutionProvider", "CPUExecutionProvider"]
  OpenVINO:  ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import numpy as np

from glossa_common.logging import get_logger

from ..domain.entities import InferenceDevice, ModelSpec
from .base import BaseInferenceEngine

logger = get_logger(__name__)

# ── CPU provider options ───────────────────────────────────────────────────────
_CPU_PROVIDER_OPTIONS: dict[str, Any] = {
    "arena_extend_strategy": "kSameAsRequested",   # predictable memory growth
}


def _build_session_options(n_threads: int | None = None) -> object:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL   # no intra-op parallelism issues

    n_phys = n_threads or max(1, (os.cpu_count() or 4) // 2)  # physical cores
    opts.intra_op_num_threads = n_phys
    opts.inter_op_num_threads = 2

    opts.enable_mem_pattern = True
    opts.enable_cpu_mem_arena = True
    opts.add_session_config_entry("session.disable_prepacking", "0")
    opts.add_session_config_entry("session.use_env_allocators", "1")

    return opts


def _detect_providers(spec: ModelSpec) -> list[str]:
    device = spec.extra.get("device", "CPU").upper()
    if device == "GPU":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "OPENVINO":
        return ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class ONNXInferenceEngine(BaseInferenceEngine):
    """ONNX Runtime engine with CPU memory optimization and INT8 support."""

    def __init__(
        self,
        spec: ModelSpec,
        labels: list[str],
        feature_dim: int = 258,
        n_threads: int | None = None,
    ) -> None:
        super().__init__(spec, labels, feature_dim, n_threads)
        self._input_name: str = ""
        self._output_name: str = ""
        self._input_shape: tuple[int, ...] = ()
        self._dynamic_axes: list[int] = []

    async def _load(self) -> None:
        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        import onnxruntime as ort

        model_path = str(self._spec.path)

        # Apply dynamic quantization if requested on a float32 model
        if (
            self._spec.quantization.value in ("int8",)
            and not model_path.endswith(".quant.onnx")
        ):
            model_path = self._maybe_quantize(model_path)

        opts = _build_session_options(self._n_threads)
        providers = _detect_providers(self._spec)

        self._session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)

        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()

        self._input_name = inputs[0].name
        self._output_name = outputs[0].name
        self._input_shape = tuple(inputs[0].shape)

        # Detect dynamic axes (where shape dim is None or string)
        self._dynamic_axes = [
            i for i, d in enumerate(inputs[0].shape) if not isinstance(d, int)
        ]

        actual_providers = self._session.get_providers()
        logger.info(
            "onnx_engine_loaded",
            model_id=self.model_id,
            model_path=model_path,
            providers=actual_providers,
            input_shape=self._input_shape,
            quantization=self._spec.quantization,
        )
        self._ready = True

    def _maybe_quantize(self, model_path: str) -> str:
        """Apply ONNX dynamic quantization → save as .quant.onnx beside source."""
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quant_path = model_path.replace(".onnx", ".quant.onnx")
        if Path(quant_path).exists():
            logger.info("using_cached_quantized_model", path=quant_path)
            return quant_path

        logger.info("applying_dynamic_quantization", source=model_path, target=quant_path)
        quantize_dynamic(
            model_input=model_path,
            model_output=quant_path,
            weight_type=QuantType.QInt8,
            optimize_model=True,
        )
        logger.info("quantization_complete", target=quant_path)
        return quant_path

    def _run_inference_sync(self, batch: np.ndarray) -> np.ndarray:
        """Synchronous inference on [B, T, D] → [B, C]."""
        import onnxruntime as ort

        session: ort.InferenceSession = self._session  # type: ignore[assignment]

        # Preprocess: squeeze/reshape if model expects specific format
        feeds = {self._input_name: batch.astype(np.float32)}
        outputs = session.run([self._output_name], feeds)
        logits: np.ndarray = outputs[0]

        # Some models output [B, 1, C] or [B, C, 1] — normalize to [B, C]
        if logits.ndim == 3:
            logits = logits.squeeze(axis=-1) if logits.shape[-1] == 1 else logits.squeeze(axis=1)

        return logits

    def get_providers(self) -> list[str]:
        if self._session is None:
            return []
        return self._session.get_providers()  # type: ignore[union-attr]

    def get_model_metadata(self) -> dict[str, str]:
        if self._session is None:
            return {}
        meta = self._session.get_modelmeta()  # type: ignore[union-attr]
        return dict(meta.custom_metadata_map)


class ONNXEngineFactory:
    """Factory to create ONNXInferenceEngine instances from ModelSpec."""

    @staticmethod
    async def create(
        spec: ModelSpec,
        labels: list[str],
        feature_dim: int = 258,
        n_threads: int | None = None,
    ) -> ONNXInferenceEngine:
        engine = ONNXInferenceEngine(
            spec=spec, labels=labels, feature_dim=feature_dim, n_threads=n_threads
        )
        await engine._load()
        return engine
