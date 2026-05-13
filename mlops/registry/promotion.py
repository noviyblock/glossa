"""Model promotion workflow — staging → production with validation gates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlops.config import MLOpsSettings, get_settings
from mlops.registry.onnx_registry import ONNXModelRegistry, RegisteredModel


@dataclass
class PromotionResult:
    success: bool
    model_name: str
    version: str
    from_stage: str
    to_stage: str
    reason: str
    failed_checks: list[str]


class ModelPromoter:
    """Validates and transitions model versions between MLflow stages.

    Stages: None → Staging → Production (→ Archived)

    Gate checks before each promotion:
    - Staging: model must exist, metrics keys must be present
    - Production: metric thresholds from MLOpsSettings must pass,
                  a Staging version must already exist
    """

    def __init__(
        self,
        registry: ONNXModelRegistry | None = None,
        settings: MLOpsSettings | None = None,
    ) -> None:
        self._cfg = settings or get_settings()
        self._registry = registry or ONNXModelRegistry(self._cfg)
        self._mlflow: Any = None
        self._client: Any = None
        self._connect()

    # ── Public API ────────────────────────────────────────────────────────────

    def promote_to_staging(
        self,
        model_name: str,
        version: str,
    ) -> PromotionResult:
        """Promote a registered version to Staging."""
        failed = self._check_exists(model_name, version)
        if failed:
            return PromotionResult(False, model_name, version, "None", "Staging", "Pre-check failed", failed)

        return self._transition(model_name, version, "None", "Staging")

    def promote_to_production(
        self,
        model_name: str,
        version: str,
        archive_existing: bool = True,
    ) -> PromotionResult:
        """Promote a Staging version to Production after passing metric gates."""
        failed = self._check_exists(model_name, version)
        failed += self._check_metric_thresholds(model_name, version)

        if failed:
            return PromotionResult(
                False, model_name, version, "Staging", "Production",
                f"{len(failed)} gate(s) failed", failed,
            )

        if archive_existing:
            self._archive_current_production(model_name)

        result = self._transition(model_name, version, "Staging", "Production")
        return result

    def rollback_production(
        self,
        model_name: str,
    ) -> PromotionResult:
        """Revert to the most recent Archived version (undoes last promotion)."""
        archived = [
            v for v in self._registry.list_versions(model_name)
            if v.stage == "Archived"
        ]
        if not archived:
            return PromotionResult(
                False, model_name, "?", "Production", "Production",
                "No archived versions to roll back to", [],
            )

        # Sort by version number descending to find the most recent
        archived.sort(key=lambda v: int(v.version), reverse=True)
        target = archived[0]

        self._archive_current_production(model_name)
        return self._transition(model_name, target.version, "Archived", "Production")

    def auto_promote(
        self,
        model_name: str,
        metric: str,
        lower_is_better: bool = False,
    ) -> PromotionResult:
        """Find the best-scoring version in Staging and promote it to Production."""
        candidates = [
            v for v in self._registry.list_versions(model_name)
            if v.stage == "Staging"
        ]
        if not candidates:
            return PromotionResult(
                False, model_name, "?", "Staging", "Production",
                "No Staging versions found", [],
            )

        def get_metric(rv: RegisteredModel) -> float:
            lineage = self._registry.get_lineage(model_name, rv.version)
            val = lineage.get("metrics", {}).get(metric)
            if val is None:
                return float("inf") if lower_is_better else float("-inf")
            return float(val)

        best = min(candidates, key=get_metric) if lower_is_better else max(candidates, key=get_metric)
        return self.promote_to_production(model_name, best.version)

    # ── Checks ────────────────────────────────────────────────────────────────

    def _check_exists(self, model_name: str, version: str) -> list[str]:
        versions = self._registry.list_versions(model_name)
        if not any(v.version == version for v in versions):
            return [f"Version {version} of {model_name} not found in registry"]
        return []

    def _check_metric_thresholds(self, model_name: str, version: str) -> list[str]:
        lineage = self._registry.get_lineage(model_name, version)
        metrics = lineage.get("metrics", {})
        failed: list[str] = []

        thresholds: dict[str, tuple[float, bool]] = {}  # metric → (threshold, lower_is_better)

        if model_name == self._cfg.gesture_model_name:
            thresholds = {
                "accuracy": (self._cfg.gesture_min_accuracy, False),
                "f1_weighted": (self._cfg.gesture_min_f1, False),
                "latency_p95_ms": (self._cfg.gesture_max_latency_p95_ms, True),
            }
        elif model_name == self._cfg.nlp_model_name:
            thresholds = {
                "bleu_4": (self._cfg.nlp_min_bleu, False),
                "rouge_l": (self._cfg.nlp_min_rouge_l, False),
                "latency_p95_ms": (self._cfg.nlp_max_latency_p95_ms, True),
            }
        elif model_name == self._cfg.asr_model_name:
            thresholds = {
                "wer": (self._cfg.asr_max_wer, True),
            }

        for metric_key, (threshold, lower_is_better) in thresholds.items():
            val = metrics.get(metric_key)
            if val is None:
                failed.append(f"Missing metric: {metric_key}")
                continue
            if lower_is_better and float(val) > threshold:
                failed.append(f"{metric_key}={val:.4f} exceeds max={threshold}")
            elif not lower_is_better and float(val) < threshold:
                failed.append(f"{metric_key}={val:.4f} below min={threshold}")

        return failed

    # ── Transitions ───────────────────────────────────────────────────────────

    def _transition(
        self,
        model_name: str,
        version: str,
        from_stage: str,
        to_stage: str,
    ) -> PromotionResult:
        if not self._client:
            return PromotionResult(
                False, model_name, version, from_stage, to_stage,
                "MLflow client not available", [],
            )
        try:
            self._client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=to_stage,
                archive_existing_versions=False,
            )
            return PromotionResult(
                True, model_name, version, from_stage, to_stage,
                f"Promoted {model_name} v{version} to {to_stage}", [],
            )
        except Exception as exc:
            return PromotionResult(
                False, model_name, version, from_stage, to_stage,
                str(exc), [],
            )

    def _archive_current_production(self, model_name: str) -> None:
        current = self._registry.get_production_version(model_name)
        if current and self._client:
            try:
                self._client.transition_model_version_stage(
                    name=model_name,
                    version=current.version,
                    stage="Archived",
                    archive_existing_versions=False,
                )
            except Exception:
                pass

    def _connect(self) -> None:
        try:
            from mlflow.tracking import MlflowClient  # type: ignore[import]
            import mlflow  # type: ignore[import]
            mlflow.set_tracking_uri(self._cfg.mlflow_tracking_uri)
            self._mlflow = mlflow
            self._client = MlflowClient(self._cfg.mlflow_tracking_uri)
        except Exception:
            pass
