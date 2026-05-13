"""WebSocket protocol codec for the ASR service.

Message types:
  Client → Server:
    0x01  AUDIO_CHUNK   binary PCM/WAV payload
    0x02  CONTROL       JSON config (language, sample_rate, encoding)
    0x03  PING          heartbeat request (seq in payload)
    0x04  END_OF_STREAM signal to flush and finalize
    0x05  RECONNECT     JSON {reconnect_token, session_id}

  Server → Client:
    0x10  PARTIAL       JSON partial transcript
    0x11  FINAL         JSON final transcript
    0x12  PONG          heartbeat response (echoes seq)
    0x13  ERROR         JSON error details
    0x14  SESSION_INFO  JSON session metadata + reconnect_token
    0x15  BACKPRESSURE  JSON {reason, queue_depth, suggested_action}

Header (8 bytes): [type:1][flags:1][seq:2][length:4]  (big-endian)
Flags: 0x01 = IS_JSON
"""

from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

_HEADER_FMT  = "!BBHi"    # type(1) flags(1) seq(2) length(4) — 8 bytes
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)

IS_JSON_FLAG = 0x01


class MsgType(IntEnum):
    # Client → Server
    AUDIO_CHUNK    = 0x01
    CONTROL        = 0x02
    PING           = 0x03
    END_OF_STREAM  = 0x04
    RECONNECT      = 0x05
    # Server → Client
    PARTIAL        = 0x10
    FINAL          = 0x11
    PONG           = 0x12
    ERROR          = 0x13
    SESSION_INFO   = 0x14
    BACKPRESSURE   = 0x15


class ProtocolError(ValueError):
    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ParsedFrame:
    msg_type: MsgType
    flags: int
    seq: int
    payload: bytes

    @property
    def is_json(self) -> bool:
        return bool(self.flags & IS_JSON_FLAG)

    def json(self) -> dict[str, Any]:
        return json.loads(self.payload)


# ── Encode ────────────────────────────────────────────────────────────────────

def encode_frame(msg_type: MsgType, payload: bytes, seq: int = 0, is_json: bool = False) -> bytes:
    flags = IS_JSON_FLAG if is_json else 0
    header = struct.pack(_HEADER_FMT, int(msg_type), flags, seq & 0xFFFF, len(payload))
    return header + payload


def encode_json(msg_type: MsgType, data: dict, seq: int = 0) -> bytes:
    return encode_frame(msg_type, json.dumps(data, ensure_ascii=False).encode(), seq=seq, is_json=True)


def encode_pong(seq: int) -> bytes:
    return encode_frame(MsgType.PONG, b"", seq=seq)


def encode_session_info(session_id: str, reconnect_token: str, language: str) -> bytes:
    return encode_json(MsgType.SESSION_INFO, {
        "session_id": session_id,
        "reconnect_token": reconnect_token,
        "language": language,
        "server_time": time.time(),
    })


def encode_partial(text: str, language: str, seq: int, chunk_seq: int) -> bytes:
    return encode_json(MsgType.PARTIAL, {
        "type": "partial",
        "text": text,
        "language": language,
        "chunk_seq": chunk_seq,
        "ts": time.time(),
    }, seq=seq)


def encode_final(
    text: str,
    language: str,
    language_probability: float,
    seq: int,
    inference_ms: float,
    total_latency_ms: float,
    corrected_text: str = "",
    medical_terms: list[str] | None = None,
    words: list[dict] | None = None,
) -> bytes:
    return encode_json(MsgType.FINAL, {
        "type": "final",
        "text": text,
        "corrected_text": corrected_text or text,
        "language": language,
        "language_probability": language_probability,
        "medical_terms": medical_terms or [],
        "words": words or [],
        "inference_ms": round(inference_ms, 2),
        "total_latency_ms": round(total_latency_ms, 2),
        "ts": time.time(),
    }, seq=seq)


def encode_error(code: str, message: str, seq: int = 0) -> bytes:
    return encode_json(MsgType.ERROR, {"code": code, "message": message}, seq=seq)


def encode_backpressure(reason: str, queue_depth: int, suggested_action: str = "slow_down") -> bytes:
    return encode_json(MsgType.BACKPRESSURE, {
        "reason": reason,
        "queue_depth": queue_depth,
        "suggested_action": suggested_action,
    })


# ── Decode ────────────────────────────────────────────────────────────────────

def decode_frame(data: bytes) -> ParsedFrame:
    if len(data) < _HEADER_SIZE:
        raise ProtocolError(f"Frame too short: {len(data)} < {_HEADER_SIZE}", "FRAME_TOO_SHORT")

    msg_type_raw, flags, seq, length = struct.unpack_from(_HEADER_FMT, data)

    try:
        msg_type = MsgType(msg_type_raw)
    except ValueError:
        raise ProtocolError(f"Unknown message type: 0x{msg_type_raw:02X}", "UNKNOWN_MSG_TYPE")

    payload_start = _HEADER_SIZE
    payload_end   = payload_start + length
    if len(data) < payload_end:
        raise ProtocolError(
            f"Payload truncated: expected {length}, got {len(data) - _HEADER_SIZE}",
            "PAYLOAD_TRUNCATED",
        )

    return ParsedFrame(
        msg_type=msg_type,
        flags=flags,
        seq=seq,
        payload=data[payload_start:payload_end],
    )


def decode_control(payload: bytes) -> dict[str, Any]:
    return json.loads(payload)


HEADER_SIZE = _HEADER_SIZE
