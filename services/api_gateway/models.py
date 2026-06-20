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
    latency_ms: float
    cached: bool = False
