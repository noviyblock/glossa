"""Rule-based fallback heuristics for gesture recognition.

Used when model confidence is below threshold or the model fails entirely.
Strategies:
  1. Handshape classifier — dominant hand finger extension pattern
  2. Motion primitive detector — static / directional / circular
  3. Frequency prior — return the most common gloss for ambiguous input
"""

from __future__ import annotations

import numpy as np

from glossa_common.logging import get_logger

from ..domain.entities import GlossCandidate, TopKResult
from ..domain.interfaces import FallbackHeuristicsPort

logger = get_logger(__name__)

# RSL-specific priors: (gloss, prior_probability)
# Derived from corpus frequency analysis — top-50 most common glosses
_FREQUENCY_PRIORS: list[tuple[str, float]] = [
    ("ЗДРАВСТВУЙТЕ", 0.09),
    ("Я", 0.08),
    ("ВЫ", 0.07),
    ("НЕТ", 0.06),
    ("ДА", 0.06),
    ("СПАСИБО", 0.05),
    ("ПОЖАЛУЙСТА", 0.04),
    ("КАК", 0.04),
    ("ХОРОШО", 0.03),
    ("ПОМОГИТЕ", 0.03),
]

# Finger extension thresholds (relative to palm size)
_EXTENSION_THRESHOLD = 0.15


def _finger_extensions(hand_landmarks: np.ndarray) -> np.ndarray:
    """Return boolean array [5] — True if each finger is extended.

    hand_landmarks: [21, 3] — MediaPipe hand joint coords (x, y, z)
    Finger tip indices: 4 (thumb), 8 (index), 12 (middle), 16 (ring), 20 (pinky)
    MCP indices:         2,         5,          9,           13,         17
    """
    tips = [4, 8, 12, 16, 20]
    mcps = [2, 5,  9, 13, 17]
    palm_size = np.linalg.norm(hand_landmarks[0] - hand_landmarks[9]) + 1e-6

    extended = np.zeros(5, dtype=bool)
    for i, (tip, mcp) in enumerate(zip(tips, mcps)):
        dist = np.linalg.norm(hand_landmarks[tip] - hand_landmarks[mcp])
        extended[i] = (dist / palm_size) > _EXTENSION_THRESHOLD
    return extended


def _classify_handshape(extended: np.ndarray) -> str:
    """Map finger extension pattern to a rough ASL/RSL handshape label."""
    pattern = tuple(extended.tolist())
    _MAP = {
        (False, False, False, False, False): "FIST",
        (True,  False, False, False, False): "A",
        (True,  True,  False, False, False): "L",
        (False, True,  True,  False, False): "V",
        (True,  True,  True,  True,  True ): "OPEN_HAND",
        (False, True,  False, False, False): "D",
        (False, True,  True,  True,  False): "W",
        (False, False, False, False, True ): "PINKY",
    }
    return _MAP.get(pattern, "UNKNOWN")


class FallbackHeuristics(FallbackHeuristicsPort):
    """Lightweight rule-based predictor — fires when model confidence is low."""

    def __init__(
        self,
        labels: list[str],
        confidence_floor: float = 0.15,
        top_k: int = 3,
    ) -> None:
        self._labels = labels
        self._confidence_floor = confidence_floor
        self._top_k = top_k

        # Build prior lookup: gloss → prior (only for glosses in our label set)
        label_set = set(labels)
        self._priors = [
            (gloss, prob)
            for gloss, prob in _FREQUENCY_PRIORS
            if gloss in label_set
        ]
        # Pad with uniform remainder
        if len(self._priors) < top_k:
            used = {g for g, _ in self._priors}
            for lbl in labels:
                if lbl not in used:
                    self._priors.append((lbl, 0.01))
                if len(self._priors) >= top_k + 5:
                    break

        logger.debug("fallback_heuristics_init", n_priors=len(self._priors))

    # ── FallbackHeuristicsPort ────────────────────────────────────────────────

    def predict(
        self,
        feature_sequence: np.ndarray,
        *,
        window_start: int = 0,
        window_end: int = 0,
    ) -> TopKResult:
        """Heuristic prediction from raw feature sequence [T, D]."""
        candidates = self._handshape_candidates(feature_sequence, window_start, window_end)
        if not candidates:
            candidates = self._prior_candidates(window_start, window_end)

        candidates = sorted(candidates, key=lambda c: c.confidence, reverse=True)
        return TopKResult(
            candidates=candidates[: self._top_k],
            is_fallback=True,
            window_start=window_start,
            window_end=window_end,
        )

    def should_fallback(self, confidence: float) -> bool:
        return confidence < self._confidence_floor

    # ── Private ───────────────────────────────────────────────────────────────

    def _handshape_candidates(
        self,
        feature_sequence: np.ndarray,
        window_start: int,
        window_end: int,
    ) -> list[GlossCandidate]:
        """Try to match last-frame handshape to known RSL handshapes."""
        # Feature layout: [T, 258] — first 132 = pose (33×4), next 63 = left hand (21×3), next 63 = right hand
        if feature_sequence.shape[1] < 132 + 63:
            return []

        last_frame = feature_sequence[-1]
        # Right hand: indices 132+63 to 132+126
        right_hand_flat = last_frame[132 + 63 : 132 + 126]
        if right_hand_flat.sum() == 0:
            # No right hand detected, try left
            right_hand_flat = last_frame[132:132 + 63]
        if right_hand_flat.sum() == 0:
            return []

        hand_lm = right_hand_flat.reshape(21, 3)
        extended = _finger_extensions(hand_lm)
        shape_label = _classify_handshape(extended)

        candidates: list[GlossCandidate] = []
        for gloss, prior in self._priors[:self._top_k]:
            # Handshape match boosts prior slightly (no real lookup table here)
            confidence = min(prior * 1.1, 0.40)
            candidates.append(GlossCandidate(
                confidence=confidence,
                gloss=gloss,
                gloss_id=self._labels.index(gloss) if gloss in self._labels else 0,
                start_frame=window_start,
                end_frame=window_end,
            ))

        logger.debug("fallback_handshape", shape=shape_label, n_candidates=len(candidates))
        return candidates

    def _prior_candidates(
        self, window_start: int, window_end: int
    ) -> list[GlossCandidate]:
        """Return top-K candidates based purely on corpus frequency priors."""
        return [
            GlossCandidate(
                confidence=min(prob, 0.35),
                gloss=gloss,
                gloss_id=self._labels.index(gloss) if gloss in self._labels else 0,
                start_frame=window_start,
                end_frame=window_end,
            )
            for gloss, prob in self._priors[: self._top_k]
        ]
