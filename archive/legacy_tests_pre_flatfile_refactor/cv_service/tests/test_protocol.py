"""Unit tests for the binary/JSON wire protocol codec."""

import struct

import numpy as np
import pytest

from app.transport.protocol import (
    HEADER_SIZE,
    BackpressurePayload,
    FrameFlags,
    MsgType,
    ParsedFrame,
    PredictionPayload,
    ProtocolError,
    decode_frame,
    decode_keyframe_binary,
    decode_keyframe_json,
    encode_ack,
    encode_backpressure,
    encode_frame,
    encode_json_frame,
    encode_pong,
    encode_prediction,
)

_POSE_N = 33
_HAND_N = 21


def _make_pose_bytes() -> bytes:
    """33 × 4 float32, big-endian."""
    arr = np.zeros((_POSE_N, 4), dtype=np.float32)
    arr[0] = [0.5, 0.3, 0.1, 0.9]
    return arr.astype(">f4").tobytes()


def _make_hand_bytes() -> bytes:
    """21 × 3 float32, big-endian."""
    arr = np.zeros((_HAND_N, 3), dtype=np.float32)
    return arr.astype(">f4").tobytes()


# ── encode / decode round-trip ────────────────────────────────────────────────

def test_encode_decode_roundtrip_empty_payload():
    raw = encode_frame(MsgType.PING, b"")
    frame = decode_frame(raw)
    assert frame.msg_type == MsgType.PING
    assert frame.payload == b""


def test_encode_pong_contains_seq():
    raw = encode_pong(seq=42)
    frame = decode_frame(raw)
    assert frame.msg_type == MsgType.PONG
    assert frame.seq == 42


def test_encode_ack_roundtrip():
    raw = encode_ack(seq=99)
    frame = decode_frame(raw)
    assert frame.msg_type == MsgType.ACK
    assert frame.seq == 99


def test_encode_json_frame_sets_flag():
    raw = encode_json_frame(MsgType.CONTROL, {"max_fps": 15.0}, seq=1)
    frame = decode_frame(raw)
    assert frame.is_json
    data = frame.json()
    assert data["max_fps"] == 15.0


def test_encode_prediction_roundtrip():
    pred = PredictionPayload(
        seq=5, gloss="ВРАЧ", confidence=0.92,
        start_frame=0, end_frame=29, inference_ms=45.2, queue_wait_ms=3.1
    )
    raw = encode_prediction(pred, seq=5)
    frame = decode_frame(raw)
    assert frame.msg_type == MsgType.PREDICTION
    data = frame.json()
    assert data["gloss"] == "ВРАЧ"
    assert data["confidence"] == pytest.approx(0.92, abs=1e-6)


def test_encode_backpressure_roundtrip():
    bp = BackpressurePayload(reason="fps_limit_exceeded", suggested_fps=15.0, queue_depth=32)
    raw = encode_backpressure(bp)
    frame = decode_frame(raw)
    assert frame.msg_type == MsgType.BACKPRESSURE
    data = frame.json()
    assert data["suggested_fps"] == 15.0
    assert data["queue_depth"] == 32


# ── Keyframe binary decoder ───────────────────────────────────────────────────

def test_decode_keyframe_binary_no_hands():
    pose_bytes = _make_pose_bytes()
    payload = pose_bytes + bytes([0x00, 0x00])  # no left/right hand
    kf = decode_keyframe_binary(payload)
    assert kf.pose.shape == (_POSE_N, 4)
    assert kf.left_hand is None
    assert kf.right_hand is None
    assert kf.pose[0, 0] == pytest.approx(0.5, abs=1e-5)


def test_decode_keyframe_binary_both_hands():
    pose_bytes = _make_pose_bytes()
    hand_bytes = _make_hand_bytes()
    payload = (
        pose_bytes
        + bytes([0x01]) + hand_bytes   # left hand present
        + bytes([0x01]) + hand_bytes   # right hand present
    )
    kf = decode_keyframe_binary(payload)
    assert kf.pose.shape == (_POSE_N, 4)
    assert kf.left_hand is not None and kf.left_hand.shape == (_HAND_N, 3)
    assert kf.right_hand is not None and kf.right_hand.shape == (_HAND_N, 3)


def test_decode_keyframe_json_roundtrip():
    pose = [[0.1 * i, 0.2 * i, 0.3 * i, 0.9] for i in range(33)]
    lh = [[0.0, 0.0, 0.0] for _ in range(21)]
    data = {"type": "keyframe", "seq": 7, "ts": 1234.5, "pose": pose, "left_hand": lh}
    kf = decode_keyframe_json(data)
    assert kf.pose.shape == (33, 4)
    assert kf.left_hand is not None
    assert kf.seq == 7
    assert kf.ts_ms == pytest.approx(1234.5)


# ── Error handling ─────────────────────────────────────────────────────────────

def test_decode_too_short_raises():
    with pytest.raises(ProtocolError) as exc_info:
        decode_frame(b"\x01\x00\x00")
    assert exc_info.value.code == "FRAME_TOO_SHORT"


def test_decode_unknown_type_raises():
    raw = encode_frame(MsgType.PING, b"")
    # Corrupt the type byte
    raw = bytes([0xFF]) + raw[1:]
    with pytest.raises(ProtocolError) as exc_info:
        decode_frame(raw)
    assert exc_info.value.code == "UNKNOWN_MSG_TYPE"


def test_decode_truncated_payload_raises():
    payload = b"hello world"
    raw = encode_frame(MsgType.CONTROL, payload)
    # Truncate after header
    truncated = raw[:HEADER_SIZE + 3]
    with pytest.raises(ProtocolError) as exc_info:
        decode_frame(truncated)
    assert exc_info.value.code == "PAYLOAD_TRUNCATED"
