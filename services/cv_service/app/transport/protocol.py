"""Binary + JSON wire protocol for gesture streaming.

Binary frame layout (big-endian):
┌────────┬────────┬──────────┬─────────────────┬────────────┬───────────┐
│ type   │ flags  │ seq_num  │ timestamp_ms     │ payload_len│  payload  │
│  1B    │  1B    │   4B     │      8B          │    4B      │   N bytes │
└────────┴────────┴──────────┴─────────────────┴────────────┴───────────┘
Total header: 18 bytes

Keypoints binary payload (client → server):
  pose:            33 × [x, y, z, vis]  float32  → 528 B
  left_hand_flag:  1B  (0x00 absent / 0x01 present)
  left_hand:       21 × [x, y, z]       float32  → 252 B  (if flagged)
  right_hand_flag: 1B
  right_hand:      21 × [x, y, z]       float32  → 252 B  (if flagged)
Min payload: 530 B (no hands)  Max: 1034 B (both hands)
"""

from __future__ import annotations

import struct
import time
from enum import IntEnum
from typing import Final

import numpy as np
import orjson
from pydantic import BaseModel, ConfigDict, Field

# ── Header ───────────────────────────────────────────────────────────────────
HEADER_FMT: Final = "!BBIQII"           # type flags seq ts_ms pad payload_len
HEADER_SIZE: Final = struct.calcsize(HEADER_FMT)  # 22 bytes

_POSE_N: Final = 33
_HAND_N: Final = 21
_POSE_FLOATS: Final = _POSE_N * 4      # x y z vis
_HAND_FLOATS: Final = _HAND_N * 3     # x y z


# ── Message types ─────────────────────────────────────────────────────────────
class MsgType(IntEnum):
    # Client → Server
    KEYFRAME        = 0x01
    PING            = 0x02
    SESSION_RESUME  = 0x03
    CONTROL         = 0x04   # runtime config update (fps, window_size)

    # Server → Client
    PREDICTION      = 0x10
    PONG            = 0x11
    BACKPRESSURE    = 0x12   # client must slow down
    ERROR           = 0x13
    SESSION_CREATED = 0x14
    ACK             = 0x15   # frame acknowledged


class FrameFlags(IntEnum):
    NONE        = 0x00
    IS_JSON     = 0x01   # payload is JSON, not binary
    COMPRESSED  = 0x02   # reserved for future zstd


# ── Pydantic wire models ──────────────────────────────────────────────────────
class KeyframePayload(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    seq: int
    ts_ms: float
    pose: np.ndarray          # shape (33, 4)
    left_hand: np.ndarray | None   # shape (21, 3) or None
    right_hand: np.ndarray | None  # shape (21, 3) or None


class PredictionPayload(BaseModel):
    seq: int
    gloss: str
    confidence: float = Field(ge=0.0, le=1.0)
    start_frame: int
    end_frame: int
    inference_ms: float
    queue_wait_ms: float


class SessionCreatedPayload(BaseModel):
    session_id: str
    reconnect_token: str
    window_size: int
    stride: int
    max_fps: float


class BackpressurePayload(BaseModel):
    reason: str
    suggested_fps: float
    queue_depth: int


class ErrorPayload(BaseModel):
    code: str
    message: str


class ControlPayload(BaseModel):
    max_fps: float | None = None
    window_size: int | None = None
    stride: int | None = None


# ── Encoder ───────────────────────────────────────────────────────────────────
def encode_frame(
    msg_type: MsgType,
    payload: bytes,
    seq: int = 0,
    flags: int = FrameFlags.NONE,
) -> bytes:
    ts_ms = int(time.time() * 1000)
    header = struct.pack(HEADER_FMT, msg_type, flags, seq, ts_ms, 0, len(payload))
    return header + payload


def encode_json_frame(msg_type: MsgType, data: dict, seq: int = 0) -> bytes:
    payload = orjson.dumps(data)
    return encode_frame(msg_type, payload, seq, flags=FrameFlags.IS_JSON)


def encode_pong(seq: int) -> bytes:
    return encode_frame(MsgType.PONG, b"", seq)


def encode_ack(seq: int) -> bytes:
    return encode_frame(MsgType.ACK, b"", seq)


def encode_backpressure(bp: BackpressurePayload, seq: int = 0) -> bytes:
    return encode_json_frame(MsgType.BACKPRESSURE, bp.model_dump(), seq)


def encode_prediction(pred: PredictionPayload, seq: int = 0) -> bytes:
    return encode_json_frame(MsgType.PREDICTION, pred.model_dump(), seq)


def encode_session_created(s: SessionCreatedPayload) -> bytes:
    return encode_json_frame(MsgType.SESSION_CREATED, s.model_dump())


def encode_error(code: str, message: str) -> bytes:
    e = ErrorPayload(code=code, message=message)
    return encode_json_frame(MsgType.ERROR, e.model_dump())


# ── Decoder ───────────────────────────────────────────────────────────────────
class ParsedFrame:
    __slots__ = ("msg_type", "flags", "seq", "ts_ms", "payload")

    def __init__(
        self,
        msg_type: MsgType,
        flags: int,
        seq: int,
        ts_ms: int,
        payload: bytes,
    ) -> None:
        self.msg_type = msg_type
        self.flags = flags
        self.seq = seq
        self.ts_ms = ts_ms
        self.payload = payload

    @property
    def is_json(self) -> bool:
        return bool(self.flags & FrameFlags.IS_JSON)

    def json(self) -> dict:
        return orjson.loads(self.payload)


class ProtocolError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def decode_frame(data: bytes) -> ParsedFrame:
    if len(data) < HEADER_SIZE:
        raise ProtocolError("FRAME_TOO_SHORT", f"Need {HEADER_SIZE}B header, got {len(data)}B")

    type_b, flags, seq, ts_ms, _pad, payload_len = struct.unpack_from(HEADER_FMT, data)

    if len(data) < HEADER_SIZE + payload_len:
        raise ProtocolError(
            "PAYLOAD_TRUNCATED",
            f"Expected {payload_len}B payload, got {len(data) - HEADER_SIZE}B",
        )

    try:
        msg_type = MsgType(type_b)
    except ValueError:
        raise ProtocolError("UNKNOWN_MSG_TYPE", f"Unknown message type: 0x{type_b:02X}")

    payload = data[HEADER_SIZE : HEADER_SIZE + payload_len]
    return ParsedFrame(msg_type, flags, seq, ts_ms, payload)


def decode_keyframe_binary(payload: bytes) -> KeyframePayload:
    """Decode binary keypoints payload into numpy arrays."""
    offset = 0

    if len(payload) < _POSE_FLOATS * 4 + 2:
        raise ProtocolError("KEYFRAME_TOO_SHORT", "Keyframe payload too small for pose data")

    pose_flat = np.frombuffer(payload[offset : offset + _POSE_FLOATS * 4], dtype=">f4").astype(np.float32)
    pose = pose_flat.reshape(_POSE_N, 4)
    offset += _POSE_FLOATS * 4

    left_hand: np.ndarray | None = None
    lh_flag = payload[offset]
    offset += 1
    if lh_flag:
        lh_flat = np.frombuffer(payload[offset : offset + _HAND_FLOATS * 4], dtype=">f4").astype(np.float32)
        left_hand = lh_flat.reshape(_HAND_N, 3)
        offset += _HAND_FLOATS * 4

    right_hand: np.ndarray | None = None
    rh_flag = payload[offset]
    offset += 1
    if rh_flag:
        rh_flat = np.frombuffer(payload[offset : offset + _HAND_FLOATS * 4], dtype=">f4").astype(np.float32)
        right_hand = rh_flat.reshape(_HAND_N, 3)

    # seq and ts pulled from header — we use dummy here; caller merges with ParsedFrame
    return KeyframePayload(seq=0, ts_ms=0.0, pose=pose, left_hand=left_hand, right_hand=right_hand)


def decode_keyframe_json(data: dict) -> KeyframePayload:
    """Decode JSON keypoints payload (fallback for web clients)."""
    pose = np.array(data["pose"], dtype=np.float32)          # (33, 4)
    lh_raw = data.get("left_hand")
    rh_raw = data.get("right_hand")
    left_hand = np.array(lh_raw, dtype=np.float32) if lh_raw else None
    right_hand = np.array(rh_raw, dtype=np.float32) if rh_raw else None
    return KeyframePayload(
        seq=data.get("seq", 0),
        ts_ms=data.get("ts", 0.0),
        pose=pose,
        left_hand=left_hand,
        right_hand=right_hand,
    )
