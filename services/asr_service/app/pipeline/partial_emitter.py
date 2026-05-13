"""Partial transcript emitter — accumulates segments and emits partial updates.

Strategy:
- Emit a partial after each new decoded segment
- Deduplicate identical consecutive partials
- Gate on minimum text change (avoid spam on silence)
- Track cumulative text per session
"""

from __future__ import annotations

import time

from ..domain.entities import PartialTranscript, TranscriptSegment


class PartialEmitter:
    """Per-session state machine for partial transcript emission."""

    def __init__(
        self,
        session_id: str,
        request_id: str,
        language: str = "ru",
        min_delta_chars: int = 3,
    ) -> None:
        self._session_id = session_id
        self._request_id = request_id
        self._language = language
        self._min_delta = min_delta_chars
        self._segments: list[TranscriptSegment] = []
        self._last_emitted_text: str = ""
        self._chunk_seq: int = 0

    def push_segment(self, seg: TranscriptSegment) -> PartialTranscript | None:
        """Add a newly decoded segment. Returns a PartialTranscript if worth emitting."""
        self._segments.append(seg)
        full_text = self._build_text()

        if not full_text.strip():
            return None

        delta = len(full_text) - len(self._last_emitted_text)
        if delta < self._min_delta and full_text == self._last_emitted_text:
            return None

        self._last_emitted_text = full_text
        self._chunk_seq += 1

        return PartialTranscript(
            session_id=self._session_id,
            request_id=self._request_id,
            text=full_text,
            language=self._language,
            segments=list(self._segments),
            chunk_seq=self._chunk_seq,
            emitted_at=time.perf_counter(),
        )

    def build_final(
        self,
        language: str | None = None,
        language_prob: float = 1.0,
        audio_duration_s: float = 0.0,
    ) -> tuple[str, list[TranscriptSegment]]:
        """Return the accumulated text and segments for final transcript construction."""
        return self._build_text(), list(self._segments)

    def reset(self) -> None:
        self._segments = []
        self._last_emitted_text = ""
        self._chunk_seq = 0

    def _build_text(self) -> str:
        return " ".join(s.text for s in self._segments if s.text).strip()
