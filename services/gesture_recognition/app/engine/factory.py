"""Engine factory — creates the correct engine from a ModelSpec.

Composition:
  ModelSpec.backend == ONNX      → ONNXInferenceEngine
  ModelSpec.backend == OPENVINO  → OpenVINOInferenceEngine

  ModelSpec.architecture == STGCN      → STGCNAdapter(inner)
  ModelSpec.architecture == POSEFORMER → PoseFormerAdapter(inner)
  ModelSpec.architecture == CUSTOM     → inner engine directly

This factory is the single wiring point for engine + adapter composition.
"""

from __future__ import annotations

from glossa_common.logging import get_logger

from ..domain.entities import ModelArchitecture, ModelBackend, ModelSpec
from ..domain.interfaces import InferenceEnginePort
from .onnx_engine import ONNXInferenceEngine
from .openvino_engine import OpenVINOInferenceEngine
from .poseformer_adapter import PoseFormerAdapter
from .stgcn_adapter import STGCNAdapter

logger = get_logger(__name__)


async def create_engine(
    spec: ModelSpec,
    labels: list[str],
    feature_dim: int = 258,
    n_threads: int | None = None,
) -> InferenceEnginePort:
    """Build and load the appropriate engine+adapter for the given spec."""

    # 1. Build the inner hardware engine
    if spec.backend == ModelBackend.ONNX:
        inner = ONNXInferenceEngine(spec, labels, feature_dim, n_threads)
    elif spec.backend == ModelBackend.OPENVINO:
        inner = OpenVINOInferenceEngine(spec, labels, feature_dim, n_threads)
    else:
        raise ValueError(f"Unsupported backend: {spec.backend!r}")

    # 2. Wrap with architecture-specific adapter
    engine: InferenceEnginePort
    if spec.architecture == ModelArchitecture.STGCN:
        engine = STGCNAdapter(inner_engine=inner, spec=spec, labels=labels, feature_dim=feature_dim)
    elif spec.architecture == ModelArchitecture.POSEFORMER:
        engine = PoseFormerAdapter(inner_engine=inner, spec=spec, labels=labels, feature_dim=feature_dim)
    else:
        # MLSTM or CUSTOM: use inner engine directly (expects [B, T, D])
        engine = inner

    # 3. Load (triggers actual session creation, warmup deferred to service layer)
    await engine._load()

    logger.info(
        "engine_created",
        model_id=spec.id,
        architecture=spec.architecture,
        backend=spec.backend,
        quantization=spec.quantization,
        device=spec.extra.get("device", "CPU"),
    )
    return engine
