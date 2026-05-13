from __future__ import annotations

import asyncio
from functools import cached_property
from pathlib import Path

import numpy as np

from glossa_common.logging import get_logger
from glossa_common.telemetry import MODEL_INFERENCE_LATENCY

from ..domain.pipeline import GestureClassifier

logger = get_logger(__name__)


class ONNXGestureClassifier(GestureClassifier):
    """ONNX Runtime backend — supports CPU and CUDA execution providers."""

    def __init__(
        self,
        model_path: str,
        device: str = "CPU",
        class_labels: list[str] | None = None,
    ) -> None:
        self._model_path = Path(model_path)
        self._device = device
        self._class_labels = class_labels or []
        self._session: object | None = None
        self._loop = asyncio.get_event_loop()

    def _build_session(self) -> object:
        import onnxruntime as ort

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self._device.upper() == "GPU"
            else ["CPUExecutionProvider"]
        )
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 2

        session = ort.InferenceSession(
            str(self._model_path), sess_options=opts, providers=providers
        )
        logger.info(
            "onnx_session_created",
            model=self._model_path.name,
            providers=session.get_providers(),
        )
        return session

    async def warmup(self) -> None:
        if self._session is None:
            self._session = await asyncio.to_thread(self._build_session)

        dummy = np.zeros((1, 30, 1629), dtype=np.float32)  # [batch, seq, features]
        await self.predict(dummy[0])
        logger.info("onnx_warmup_done", model=self._model_path.name)

    async def predict(self, feature_sequence: np.ndarray) -> tuple[list[str], list[float]]:
        if self._session is None:
            self._session = await asyncio.to_thread(self._build_session)

        with MODEL_INFERENCE_LATENCY.labels(service="cv-service", model="onnx").time():
            result = await asyncio.to_thread(self._run_inference, feature_sequence)
        return result

    def _run_inference(self, feature_sequence: np.ndarray) -> tuple[list[str], list[float]]:
        import onnxruntime as ort

        session: ort.InferenceSession = self._session  # type: ignore[assignment]
        input_name = session.get_inputs()[0].name
        batch = feature_sequence[np.newaxis, ...]  # [1, T, D]

        outputs = session.run(None, {input_name: batch})
        logits: np.ndarray = outputs[0][0]  # [num_classes]

        probs = _softmax(logits)
        top_idx = int(np.argmax(probs))
        label = self._class_labels[top_idx] if self._class_labels else str(top_idx)
        return [label], [float(probs[top_idx])]


class OpenVINOGestureClassifier(GestureClassifier):
    """OpenVINO backend — optimal for Intel CPU/iGPU."""

    def __init__(self, model_path: str, device: str = "CPU") -> None:
        self._model_path = Path(model_path)
        self._device = device
        self._compiled: object | None = None

    def _build_model(self) -> object:
        from openvino.runtime import Core

        core = Core()
        model = core.read_model(str(self._model_path))
        compiled = core.compile_model(model, self._device)
        logger.info("openvino_model_loaded", model=self._model_path.name, device=self._device)
        return compiled

    async def warmup(self) -> None:
        if self._compiled is None:
            self._compiled = await asyncio.to_thread(self._build_model)
        dummy = np.zeros((1, 30, 1629), dtype=np.float32)
        await self.predict(dummy[0])
        logger.info("openvino_warmup_done", model=self._model_path.name)

    async def predict(self, feature_sequence: np.ndarray) -> tuple[list[str], list[float]]:
        if self._compiled is None:
            self._compiled = await asyncio.to_thread(self._build_model)

        with MODEL_INFERENCE_LATENCY.labels(service="cv-service", model="openvino").time():
            result = await asyncio.to_thread(self._run_inference, feature_sequence)
        return result

    def _run_inference(self, feature_sequence: np.ndarray) -> tuple[list[str], list[float]]:
        batch = feature_sequence[np.newaxis, ...]
        infer_request = self._compiled.create_infer_request()  # type: ignore[union-attr]
        infer_request.infer({0: batch})
        logits: np.ndarray = infer_request.get_output_tensor(0).data[0]
        probs = _softmax(logits)
        top_idx = int(np.argmax(probs))
        return [str(top_idx)], [float(probs[top_idx])]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def build_classifier(backend: str, model_path: str, device: str) -> GestureClassifier:
    if backend == "openvino":
        return OpenVINOGestureClassifier(model_path=model_path, device=device)
    return ONNXGestureClassifier(model_path=model_path, device=device)
