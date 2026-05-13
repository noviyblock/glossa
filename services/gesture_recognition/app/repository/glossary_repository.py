"""RSL glossary repository — label management, fuzzy matching, frequency priors."""

from __future__ import annotations

import json
from pathlib import Path

from glossa_common.logging import get_logger

from ..domain.interfaces import GlossaryPort

logger = get_logger(__name__)


class GlossaryRepository(GlossaryPort):
    """Loads RSL gloss labels from a flat text file or JSON frequency table.

    Supports:
    - get_labels() → ordered list matching model output indices
    - resolve(gloss) → canonical form (handles case / transliteration)
    - fuzzy_match(query) → best gloss candidates using edit distance
    - frequency_prior(gloss) → float probability from corpus stats
    """

    def __init__(
        self,
        labels_path: Path,
        frequency_path: Path | None = None,
    ) -> None:
        self._labels_path = labels_path
        self._labels: list[str] = []
        self._label_index: dict[str, int] = {}
        self._frequencies: dict[str, float] = {}
        self._load(labels_path, frequency_path)

    # ── GlossaryPort ─────────────────────────────────────────────────────────

    def get_labels(self) -> list[str]:
        return self._labels

    def resolve(self, gloss: str) -> str | None:
        """Return the canonical gloss string (exact or normalized)."""
        if gloss in self._label_index:
            return gloss
        normalized = gloss.upper().strip()
        if normalized in self._label_index:
            return normalized
        return None

    def gloss_to_id(self, gloss: str) -> int | None:
        canonical = self.resolve(gloss)
        return self._label_index.get(canonical) if canonical else None

    def id_to_gloss(self, gloss_id: int) -> str | None:
        if 0 <= gloss_id < len(self._labels):
            return self._labels[gloss_id]
        return None

    def fuzzy_match(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Return top-K glosses ranked by normalized edit-distance similarity."""
        query = query.upper().strip()
        scored: list[tuple[str, float]] = []
        for label in self._labels:
            dist = _edit_distance(query, label)
            max_len = max(len(query), len(label), 1)
            similarity = 1.0 - dist / max_len
            scored.append((label, similarity))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def frequency_prior(self, gloss: str) -> float:
        canonical = self.resolve(gloss)
        if canonical and canonical in self._frequencies:
            return self._frequencies[canonical]
        return 1.0 / max(len(self._labels), 1)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _load(self, labels_path: Path, frequency_path: Path | None) -> None:
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        suffix = labels_path.suffix.lower()
        if suffix == ".json":
            raw = json.loads(labels_path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                self._labels = [str(x) for x in raw]
            elif isinstance(raw, dict):
                # {gloss: id} or {id: gloss}
                first_key = next(iter(raw))
                if isinstance(first_key, str) and isinstance(raw[first_key], int):
                    pairs = sorted(raw.items(), key=lambda kv: kv[1])
                    self._labels = [g for g, _ in pairs]
                else:
                    pairs = sorted(raw.items(), key=lambda kv: int(kv[0]))
                    self._labels = [v for _, v in pairs]
        else:
            # Plain text, one gloss per line
            self._labels = [
                line.strip()
                for line in labels_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self._label_index = {label: idx for idx, label in enumerate(self._labels)}

        if frequency_path and frequency_path.exists():
            freq_raw = json.loads(frequency_path.read_text(encoding="utf-8"))
            total = sum(freq_raw.values()) or 1
            self._frequencies = {k: v / total for k, v in freq_raw.items()}

        logger.info(
            "glossary_loaded",
            n_labels=len(self._labels),
            has_frequencies=bool(self._frequencies),
            path=str(labels_path),
        )


def _edit_distance(a: str, b: str) -> int:
    """Standard Levenshtein distance."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]
