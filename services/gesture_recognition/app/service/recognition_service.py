"""Recognition service — main orchestration layer.

Pipeline:
  feature_sequence → sliding window → batch_processor → post-processing → fallback
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from glossa_common.logging import get_logger

from ..domain.entities import (
    GlossCandidate,
    InferenceDevice,
    InferenceRequest,
    InferenceResult,
    TopKResult,
)
from ..domain.interfaces import FallbackHeuristicsPort, ModelRegistryPort
from ..domain.sliding_window import SlidingWindowManager, deduplicate_predictions, filter_low_confidence
from .batch_processor import BatchProcessor

logger = get_logger(__name__)


@dataclass
class RecognitionConfig:
    top_k: int = 5
    confidence_threshold: float = 0.35
    dedup_window: int = 3
    use_fallback: bool = True
    session_id: str = ""


@dataclass
class RecognitionOutput:
    glosses: list[str]
    confidences: list[float]
    is_fallback: bool
    inference_ms: float
    queue_wait_ms: float
    model_id: str
    session_id: str = ""
    candidates: list[TopKResult] = field(default_factory=list)


class GestureRecognitionService:
    """Orchestrates gesture recognition from raw feature sequences.

    Maintains per-session sliding window state via a shared SlidingWindowManager.
    Falls back to heuristics when model confidence is below threshold.
    """

    def __init__(
        self,
        registry: ModelRegistryPort,
        batch_processor: BatchProcessor,
        fallback: FallbackHeuristicsPort,
        sliding_window: SlidingWindowManager,
        config: RecognitionConfig | None = None,
    ) -> None:
        self._registry = registry
        self._batch = batch_processor
        self._fallback = fallback
        self._window_mgr = sliding_window
        self._cfg = config or RecognitionConfig()

    async def recognize(
        self,
        feature_sequence: np.ndarray,
        session_id: str = "",
        top_k: int | None = None,
        confidence_threshold: float | None = None,
    ) -> RecognitionOutput:
        """Run full recognition pipeline on a feature sequence [T, D].

        Returns a RecognitionOutput with ranked gloss candidates.
        """
        top_k = top_k or self._cfg.top_k
        threshold = confidence_threshold or self._cfg.confidence_threshold
        session_id = session_id or self._cfg.session_id

        t_submit = time.perf_counter()

        # Push into sliding window — may produce multiple emission windows
        windows = self._window_mgr.push(feature_sequence)
        if not windows:
            # Not enough frames yet — return empty
            return RecognitionOutput(
                glosses=[],
                confidences=[],
                is_fallback=False,
                inference_ms=0.0,
                queue_wait_ms=0.0,
                model_id=self._active_model_id(),
                session_id=session_id,
            )

        # Run inference for each window
        all_results: list[TopKResult] = []
        total_inference_ms = 0.0

        for window_seq, win_start, win_end in windows:
            req = InferenceRequest(
                feature_sequence=window_seq,
                top_k=top_k,
                confidence_threshold=threshold,
                session_id=session_id,
            )
            result: InferenceResult = await self._batch.submit(req)
            total_inference_ms += result.inference_ms
            all_results.extend(result.top_k)

        # Post-process
        filtered = filter_low_confidence(all_results, min_confidence=threshold)
        deduped = deduplicate_predictions(filtered, window=self._cfg.dedup_window)

        is_fallback = False
        if not deduped and self._cfg.use_fallback:
            fb_result = self._fallback.predict(
                feature_sequence,
                window_start=0,
                window_end=len(feature_sequence) - 1,
            )
            deduped = [fb_result] if fb_result.candidates else []
            is_fallback = True
            logger.debug("recognition_fallback_used", session_id=session_id)
        elif deduped:
            best_conf = max(
                (c.confidence for r in deduped for c in r.candidates if r.candidates),
                default=0.0,
            )
            if self._fallback.should_fallback(best_conf) and self._cfg.use_fallback:
                fb_result = self._fallback.predict(feature_sequence)
                deduped.append(fb_result)
                is_fallback = True

        glosses, confidences = self._flatten(deduped, top_k)
        queue_wait_ms = (time.perf_counter() - t_submit) * 1000.0 - total_inference_ms

        logger.info(
            "recognition_complete",
            session_id=session_id,
            n_glosses=len(glosses),
            is_fallback=is_fallback,
            inference_ms=round(total_inference_ms, 2),
        )

        return RecognitionOutput(
            glosses=glosses,
            confidences=confidences,
            is_fallback=is_fallback,
            inference_ms=total_inference_ms,
            queue_wait_ms=queue_wait_ms,
            model_id=self._active_model_id(),
            session_id=session_id,
            candidates=deduped,
        )

    def reset_session(self, session_id: str | None = None) -> None:
        """Clear sliding window state for a session."""
        self._window_mgr.reset()
        logger.debug("recognition_session_reset", session_id=session_id)

    def flush(self) -> list[TopKResult]:
        """Flush any buffered frames and return partial results."""
        return self._window_mgr.flush()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _active_model_id(self) -> str:
        try:
            engine = self._registry.get_active()
            return engine.model_id
        except RuntimeError:
            return "none"

    @staticmethod
    def _flatten(
        results: list[TopKResult], top_k: int
    ) -> tuple[list[str], list[float]]:
        """Collect best candidates across all result windows, deduplicated by gloss."""
        seen: dict[str, float] = {}
        for result in results:
            for cand in result.candidates:
                if cand.gloss not in seen or cand.confidence > seen[cand.gloss]:
                    seen[cand.gloss] = cand.confidence

        ranked = sorted(seen.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        if not ranked:
            return [], []
        glosses, confidences = zip(*ranked)
        return list(glosses), list(confidences)
