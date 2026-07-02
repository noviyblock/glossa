from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TypedDict

import numpy as np
from prometheus_client import Histogram

from config import CLASS_MAP_PATH, ONNX_MOBILE_PATH, OV_XML_PATH, TTA_JITTER_SIGMA, TTA_VARIANTS

logger = logging.getLogger(__name__)

TOP_K = 3

MODEL_INFERENCE_LATENCY = Histogram(
    "glossa_model_inference_latency_seconds", "Model inference latency per service",
    ["service", "model"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)


class GlossResult(TypedDict):
    gloss: str
    prob: float


class GestureClassifier:
    """OpenVINO INT8 with ONNX Runtime fallback.

    Input:  (1, T, J, C) = (1, 64, 75, 3) float32  — T must equal WINDOW_SIZE
    Output: softmax scores of shape (1, num_classes)
    """

    def __init__(
        self,
        ov_xml_path: str = OV_XML_PATH,
        onnx_path: str = ONNX_MOBILE_PATH,
        class_map_path: str = CLASS_MAP_PATH,
    ) -> None:
        self._idx_to_class = self._load_class_map(class_map_path)
        self._session, self._backend = self._load_model(ov_xml_path, onnx_path)

    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_class_map(path: str) -> dict[int, str]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return {int(k): v for k, v in data.items()}

    @staticmethod
    def _load_model(ov_xml: str, onnx_path: str):
        try:
            try:
                from openvino import Core  # type: ignore  # OpenVINO >= 2024: flat namespace
            except ImportError:
                from openvino.runtime import Core  # type: ignore  # OpenVINO < 2024
            core = Core()
            model = core.read_model(ov_xml)
            compiled = core.compile_model(model, "CPU")
            logger.info("GestureClassifier: OpenVINO INT8 loaded from %s", ov_xml)
            return compiled, "openvino"
        except Exception as ov_err:
            logger.warning("OpenVINO unavailable (%s), falling back to ONNX", ov_err)

        import onnxruntime as ort  # type: ignore
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        session = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
        logger.info("GestureClassifier: ONNX Runtime loaded from %s", onnx_path)
        return session, "onnx"

    # ------------------------------------------------------------------ #

    def predict_top3(self, window: np.ndarray) -> list[GlossResult]:
        """Run inference on a (T, J, C) window → top-3 gloss results."""
        return self._top3_from_probs(self._predict_probs(window))

    def predict_top3_tta(
        self, window: np.ndarray, n_variants: int = TTA_VARIANTS, jitter_sigma: float = TTA_JITTER_SIGMA,
    ) -> list[GlossResult]:
        """Test-time augmentation: average softmax over the window plus
        `n_variants` gaussian-jittered copies, sigma matching the exact
        spatial_jitter augmentation the model was trained with (colab_
        glossa_01a's AugmentedGestureDataset) — the model has specifically
        learned to be invariant to noise of this shape, so averaging
        predictions over several draws of it is a well-grounded way to
        reduce variance from any single (possibly unlucky) resampling of
        the input, unlike arbitrary alternative resampling schemes.
        """
        probs = self._predict_probs(window)
        rng = np.random.default_rng()
        for _ in range(max(n_variants - 1, 0)):
            jittered = window + rng.normal(0, jitter_sigma, size=window.shape).astype(np.float32)
            probs = probs + self._predict_probs(jittered)
        probs /= max(n_variants, 1)
        return self._top3_from_probs(probs)

    def _predict_probs(self, window: np.ndarray) -> np.ndarray:
        """Run inference on one (T, J, C) window → softmax probability vector."""
        inp = window[np.newaxis].astype(np.float32)  # (1, T, J, C)

        start = time.perf_counter()
        if self._backend == "openvino":
            output = next(iter(self._session.outputs))
            logits = self._session([inp])[output]   # (1, num_classes)
        else:
            input_name = self._session.get_inputs()[0].name
            logits = self._session.run(None, {input_name: inp})[0]  # (1, num_classes)
        MODEL_INFERENCE_LATENCY.labels(service="cv-service", model=self._backend).observe(time.perf_counter() - start)

        logits0 = np.nan_to_num(logits[0], nan=0.0, posinf=1e4, neginf=-1e4)
        return self._softmax(logits0)

    def _top3_from_probs(self, probs: np.ndarray) -> list[GlossResult]:
        top_idx = np.argsort(probs)[::-1][:TOP_K]
        return [
            GlossResult(gloss=self._idx_to_class.get(int(i), str(i)), prob=float(probs[i]))
            for i in top_idx
        ]

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()
