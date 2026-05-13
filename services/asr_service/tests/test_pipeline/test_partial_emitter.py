"""Tests for PartialEmitter."""

import pytest

from app.domain.entities import TranscriptSegment
from app.pipeline.partial_emitter import PartialEmitter


def _seg(text: str, start: float = 0.0, end: float = 1.0) -> TranscriptSegment:
    return TranscriptSegment(text=text, start=start, end=end, avg_logprob=-0.2, no_speech_prob=0.01)


class TestPartialEmitter:
    def setup_method(self):
        self.emitter = PartialEmitter(
            session_id="test-session",
            request_id="req-1",
            language="ru",
            min_delta_chars=3,
        )

    def test_emits_on_first_segment(self):
        partial = self.emitter.push_segment(_seg("Привет"))
        assert partial is not None
        assert partial.text == "Привет"

    def test_no_emit_on_empty_text(self):
        partial = self.emitter.push_segment(_seg(""))
        assert partial is None

    def test_accumulates_multiple_segments(self):
        self.emitter.push_segment(_seg("Здравствуйте"))
        partial = self.emitter.push_segment(_seg("доктор"))
        assert partial is not None
        assert "Здравствуйте" in partial.text
        assert "доктор" in partial.text

    def test_no_duplicate_emit(self):
        self.emitter.push_segment(_seg("Привет"))
        # Pushing same text again — delta = 0, should suppress
        partial = self.emitter.push_segment(_seg("Привет"))
        assert partial is None

    def test_chunk_seq_increments(self):
        p1 = self.emitter.push_segment(_seg("Один"))
        p2 = self.emitter.push_segment(_seg("Два"))
        assert p2.chunk_seq > p1.chunk_seq

    def test_reset_clears_state(self):
        self.emitter.push_segment(_seg("Привет"))
        self.emitter.reset()
        partial = self.emitter.push_segment(_seg("Привет"))
        assert partial is not None
        assert partial.chunk_seq == 1

    def test_build_final_returns_all_text(self):
        self.emitter.push_segment(_seg("Здравствуйте"))
        self.emitter.push_segment(_seg("пациент"))
        text, segments = self.emitter.build_final()
        assert "Здравствуйте" in text
        assert "пациент" in text
        assert len(segments) == 2
