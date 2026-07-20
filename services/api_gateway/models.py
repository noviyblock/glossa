from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Incoming WebSocket messages (client → gateway) ────────────────────────── #

class WsVideoFrame(BaseModel):
    type: Literal["video_frame"]
    session_id: str
    frame: str                          # base64 JPEG

class WsAudioChunk(BaseModel):
    type: Literal["audio_chunk"]
    session_id: str
    audio: str                          # base64 WAV/PCM

class WsEndSession(BaseModel):
    type: Literal["end_session"]
    session_id: str

class WsFlushSentence(BaseModel):
    type: Literal["flush_sentence"]
    session_id: str

class WsDeleteLastGesture(BaseModel):
    type: Literal["delete_last_gesture"]
    session_id: str


# ── Outgoing WebSocket messages (gateway → client) ────────────────────────── #

class WsGloss(BaseModel):
    type: Literal["gloss"] = "gloss"
    session_id: str
    payload: dict[str, Any]             # {"glosses": [...], "confidence": float}

class WsChunk(BaseModel):
    type: Literal["chunk"] = "chunk"
    session_id: str
    payload: dict[str, Any]             # {"text": str, "is_final": bool}

class WsResult(BaseModel):
    type: Literal["result"] = "result"
    session_id: str
    payload: dict[str, Any]             # {"text": str, "confidence": float}

class WsAudio(BaseModel):
    type: Literal["audio"] = "audio"
    session_id: str
    payload: dict[str, Any]             # {"wav": base64}

class WsVideo(BaseModel):
    type: Literal["video"] = "video"
    session_id: str
    payload: dict[str, Any]             # {"video": base64 | None}

class WsPendingSentence(BaseModel):
    type: Literal["pending_sentence"] = "pending_sentence"
    session_id: str
    payload: dict[str, Any]             # {"positions": [[{"gloss","prob"},...],...]}

class WsPeerMessage(BaseModel):
    """Relayed to the *other* participant in a two-party call (see
    call_manager.py) — same shape regardless of which direction produced
    it: {"text": str, "audio": base64|None, "video": base64|None}."""
    type: Literal["peer_message"] = "peer_message"
    session_id: str
    payload: dict[str, Any]

class WsError(BaseModel):
    type: Literal["error"] = "error"
    session_id: str
    payload: dict[str, Any]             # {"message": str}


# ── REST schemas ─────────────────────────────────────────────────────────────  #

class TranslateRequest(BaseModel):
    mode: Literal["rsl_to_text", "text_to_rsl"]
    # rsl_to_text: supply gloss_sequence
    gloss_sequence: str | None = None
    # text_to_rsl: supply text (Russian)
    text: str | None = None
    session_id: str = Field(default="")

class TranslateResponse(BaseModel):
    translation: str
    glosses: list[dict] | None = None
    audio_wav: str | None = None        # base64 WAV (text_to_rsl only)
    video_mp4: str | None = None        # base64 MP4, sign clips (text_to_rsl only)
    skeleton_sequences: list[dict] | None = None  # [{"gloss","frames"}, ...] (text_to_rsl only)
    latency_ms: float
    cached: bool = False


# ── Two-party call (see call_manager.py) ────────────────────────────────────  #

class CallCreateRequest(BaseModel):
    session_id: str = Field(default="")

class CallCreateResponse(BaseModel):
    call_id: str
    session_id: str

class CallJoinRequest(BaseModel):
    session_id: str = Field(default="")

class CallJoinResponse(BaseModel):
    call_id: str
    session_id: str
