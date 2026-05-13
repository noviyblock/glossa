"""ONNX model loader — file discovery, metadata parsing, on-the-fly quantization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from glossa_common.logging import get_logger

from ..domain.entities import (
    ModelArchitecture,
    ModelBackend,
    ModelInputSpec,
    ModelOutputSpec,
    ModelSpec,
    QuantizationType,
)

logger = get_logger(__name__)

_META_FILENAME = "model_meta.json"


class ModelLoaderError(RuntimeError):
    pass


class ModelLoader:
    """Scans a model directory and constructs ModelSpec instances.

    Expected directory layout::

        models/
          stgcn-v2-int8/
            model.onnx          (or model_fp32.onnx, model_int8.onnx)
            labels.txt
            model_meta.json     (optional — overrides auto-detection)
          poseformer-v1/
            model.onnx
            labels.txt
    """

    def __init__(self, models_root: Path, labels_fallback: Path | None = None) -> None:
        self._root = models_root
        self._labels_fallback = labels_fallback

    # ── Public API ────────────────────────────────────────────────────────────

    def load_spec(self, model_dir: Path) -> ModelSpec:
        """Build a ModelSpec from a single model directory."""
        if not model_dir.is_dir():
            raise ModelLoaderError(f"Not a directory: {model_dir}")

        meta = self._read_meta(model_dir)
        onnx_path = self._find_onnx(model_dir, meta)
        labels_path = self._find_labels(model_dir, meta)

        model_id = meta.get("id", model_dir.name)
        architecture = ModelArchitecture(meta.get("architecture", ModelArchitecture.CUSTOM))
        backend = ModelBackend(meta.get("backend", ModelBackend.ONNX))
        quantization = QuantizationType(meta.get("quantization", QuantizationType.FLOAT32))

        input_spec = self._parse_input_spec(meta.get("input_spec"))
        output_spec = self._parse_output_spec(meta.get("output_spec"))

        spec = ModelSpec(
            id=model_id,
            name=meta.get("name", model_id),
            architecture=architecture,
            backend=backend,
            quantization=quantization,
            path=onnx_path,
            labels_path=labels_path,
            version=meta.get("version", "1.0.0"),
            description=meta.get("description", ""),
            input_spec=input_spec,
            output_spec=output_spec,
            extra=meta.get("extra", {}),
        )

        logger.info(
            "model_loader_spec_built",
            model_id=model_id,
            path=str(onnx_path),
            quantization=quantization,
            backend=backend,
        )
        return spec

    def scan_all(self) -> list[ModelSpec]:
        """Discover and load all model specs under models_root."""
        if not self._root.exists():
            logger.warning("model_loader_root_missing", root=str(self._root))
            return []

        specs: list[ModelSpec] = []
        for candidate in sorted(self._root.iterdir()):
            if not candidate.is_dir():
                continue
            try:
                spec = self.load_spec(candidate)
                specs.append(spec)
            except (ModelLoaderError, ValueError) as exc:
                logger.warning(
                    "model_loader_skip_dir",
                    dir=candidate.name,
                    reason=str(exc),
                )

        logger.info("model_loader_scan_complete", found=len(specs), root=str(self._root))
        return specs

    def quantize_to_int8(self, spec: ModelSpec, output_dir: Path | None = None) -> ModelSpec:
        """Quantize a FLOAT32 ONNX model to INT8 and return an updated spec.

        Uses onnxruntime.quantization.quantize_dynamic (CPU-friendly, no calibration data).
        """
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic  # type: ignore[import]
        except ImportError as exc:
            raise ModelLoaderError("onnxruntime-tools not installed: pip install onnxruntime") from exc

        if spec.quantization != QuantizationType.FLOAT32:
            raise ModelLoaderError(
                f"Can only quantize FLOAT32 models, got {spec.quantization}"
            )

        dest_dir = output_dir or spec.path.parent
        out_path = dest_dir / f"{spec.path.stem}_int8.onnx"

        if out_path.exists():
            logger.info("quantization_cached", output=str(out_path))
        else:
            logger.info("quantization_start", input=str(spec.path), output=str(out_path))
            quantize_dynamic(
                str(spec.path),
                str(out_path),
                weight_type=QuantType.QInt8,
                optimize_model=True,
            )
            logger.info("quantization_complete", output=str(out_path))

        import dataclasses
        return dataclasses.replace(
            spec,
            id=f"{spec.id}-int8",
            path=out_path,
            quantization=QuantizationType.INT8,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _read_meta(self, model_dir: Path) -> dict[str, Any]:
        meta_path = model_dir / _META_FILENAME
        if meta_path.exists():
            try:
                return json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ModelLoaderError(f"Invalid JSON in {meta_path}: {exc}") from exc
        return {}

    def _find_onnx(self, model_dir: Path, meta: dict[str, Any]) -> Path:
        if "path" in meta:
            p = model_dir / meta["path"]
            if p.exists():
                return p
            raise ModelLoaderError(f"model_meta.json 'path' points to missing file: {p}")

        # Priority: int8 > fp16 > default
        for candidate in ("model_int8.onnx", "model_fp16.onnx", "model.onnx"):
            p = model_dir / candidate
            if p.exists():
                return p

        raise ModelLoaderError(f"No ONNX file found in {model_dir}")

    def _find_labels(self, model_dir: Path, meta: dict[str, Any]) -> Path:
        if "labels_path" in meta:
            p = model_dir / meta["labels_path"]
            if p.exists():
                return p

        for candidate in ("labels.txt", "glosses.txt", "classes.txt"):
            p = model_dir / candidate
            if p.exists():
                return p

        if self._labels_fallback and self._labels_fallback.exists():
            logger.debug("model_loader_using_fallback_labels", path=str(self._labels_fallback))
            return self._labels_fallback

        raise ModelLoaderError(f"No labels file found in {model_dir}")

    @staticmethod
    def _parse_input_spec(raw: dict | None) -> ModelInputSpec | None:
        if not raw:
            return None
        return ModelInputSpec(
            name=raw["name"],
            shape=tuple(raw["shape"]),
            dtype=raw.get("dtype", "float32"),
        )

    @staticmethod
    def _parse_output_spec(raw: dict | None) -> ModelOutputSpec | None:
        if not raw:
            return None
        return ModelOutputSpec(
            name=raw["name"],
            shape=tuple(raw["shape"]),
            dtype=raw.get("dtype", "float32"),
        )

    @staticmethod
    def compute_checksum(path: Path) -> str:
        """SHA-256 of the model file (used for cache invalidation)."""
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
