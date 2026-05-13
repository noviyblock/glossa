from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import Field

from ..schemas.base import GlossaModel


class EventType(StrEnum):
    # CV pipeline
    CV_FRAMES_READY = "cv.frames_ready"
    CV_RECOGNITION_DONE = "cv.recognition_done"

    # ASR pipeline
    ASR_AUDIO_READY = "asr.audio_ready"
    ASR_TRANSCRIPTION_DONE = "asr.transcription_done"

    # NLP pipeline
    NLP_TEXT_READY = "nlp.text_ready"
    NLP_TRANSLATION_DONE = "nlp.translation_done"
    NLP_STREAM_CHUNK = "nlp.stream_chunk"

    # TTS pipeline
    TTS_TEXT_READY = "tts.text_ready"
    TTS_SYNTHESIS_DONE = "tts.synthesis_done"
    TTS_STREAM_CHUNK = "tts.stream_chunk"

    # Session lifecycle
    SESSION_STARTED = "session.started"
    SESSION_COMPLETED = "session.completed"
    SESSION_FAILED = "session.failed"


# Stream name registry
STREAMS = {
    EventType.CV_FRAMES_READY: "glossa:stream:cv:input",
    EventType.CV_RECOGNITION_DONE: "glossa:stream:cv:output",
    EventType.ASR_AUDIO_READY: "glossa:stream:asr:input",
    EventType.ASR_TRANSCRIPTION_DONE: "glossa:stream:asr:output",
    EventType.NLP_TEXT_READY: "glossa:stream:nlp:input",
    EventType.NLP_TRANSLATION_DONE: "glossa:stream:nlp:output",
    EventType.NLP_STREAM_CHUNK: "glossa:stream:nlp:chunks",
    EventType.TTS_TEXT_READY: "glossa:stream:tts:input",
    EventType.TTS_SYNTHESIS_DONE: "glossa:stream:tts:output",
    EventType.TTS_STREAM_CHUNK: "glossa:stream:tts:chunks",
    EventType.SESSION_STARTED: "glossa:stream:sessions",
    EventType.SESSION_COMPLETED: "glossa:stream:sessions",
    EventType.SESSION_FAILED: "glossa:stream:sessions",
}


class GlossaEvent(GlossaModel):
    event_id: UUID = Field(default_factory=uuid4)
    event_type: EventType
    session_id: UUID
    producer: str
    payload: dict[str, object] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    correlation_id: UUID | None = None

    def stream_name(self) -> str:
        return STREAMS.get(self.event_type, "glossa:stream:unknown")
