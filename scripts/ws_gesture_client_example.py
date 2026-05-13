#!/usr/bin/env python3
"""Example Python client for the Glossa gesture WebSocket endpoint.

Demonstrates:
- Binary frame encoding
- Session creation and reconnect token storage
- FPS-throttled keypoint streaming
- Handling BACKPRESSURE, PREDICTION, and PONG frames
- Graceful reconnect on disconnect

Usage:
    python scripts/ws_gesture_client_example.py --url ws://localhost:8001/api/v1/ws/gesture
"""

from __future__ import annotations

import argparse
import asyncio
import struct
import time
from pathlib import Path

import numpy as np

# Reuse the protocol module from the cv_service
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "cv_service"))

from app.transport.protocol import (
    HEADER_FMT,
    HEADER_SIZE,
    MsgType,
    FrameFlags,
    encode_frame,
    decode_frame,
    encode_pong,
)

import orjson

# ── Keyframe encoder ──────────────────────────────────────────────────────────

def encode_keyframe(
    pose: np.ndarray,          # (33, 4) float32
    left_hand: np.ndarray | None,   # (21, 3) or None
    right_hand: np.ndarray | None,  # (21, 3) or None
    seq: int = 0,
) -> bytes:
    """Encode a keyframe as compact binary (big-endian float32)."""
    parts = [pose.astype(">f4").tobytes()]

    if left_hand is not None:
        parts += [bytes([0x01]), left_hand.astype(">f4").tobytes()]
    else:
        parts.append(bytes([0x00]))

    if right_hand is not None:
        parts += [bytes([0x01]), right_hand.astype(">f4").tobytes()]
    else:
        parts.append(bytes([0x00]))

    payload = b"".join(parts)
    return encode_frame(MsgType.KEYFRAME, payload, seq=seq)


def fake_pose() -> np.ndarray:
    """Generate a random pose for testing."""
    arr = np.random.rand(33, 4).astype(np.float32)
    arr[:, :3] = np.clip(arr[:, :3], 0.0, 1.0)  # x,y,z in [0,1]
    arr[:, 3] = np.random.uniform(0.8, 1.0, 33)  # visibility
    return arr


def fake_hand() -> np.ndarray:
    return np.random.rand(21, 3).astype(np.float32)


# ── Client ────────────────────────────────────────────────────────────────────

class GestureClient:
    def __init__(self, url: str, target_fps: float = 30.0) -> None:
        self._url = url
        self._fps = target_fps
        self._frame_interval = 1.0 / target_fps
        self._seq = 0
        self._session_id: str | None = None
        self._reconnect_token: str | None = None
        self._predictions: list[dict] = []

    async def run(self, duration_seconds: float = 10.0) -> None:
        try:
            import websockets
        except ImportError:
            print("Install websockets: pip install websockets")
            return

        print(f"Connecting to {self._url} ...")
        async with websockets.connect(self._url, max_size=10 * 1024 * 1024) as ws:
            print("Connected.")

            # Receive SESSION_CREATED
            raw = await ws.recv()
            frame = decode_frame(raw)
            if frame.msg_type == MsgType.SESSION_CREATED:
                info = frame.json()
                self._session_id = info["session_id"]
                self._reconnect_token = info["reconnect_token"]
                print(f"Session created: {self._session_id}")
                print(f"Reconnect token: {self._reconnect_token}")

            # Start receiver task
            recv_task = asyncio.create_task(self._receiver(ws))

            # Stream keyframes for duration_seconds
            deadline = time.monotonic() + duration_seconds
            while time.monotonic() < deadline:
                t0 = time.monotonic()
                self._seq += 1

                pose = fake_pose()
                lh = fake_hand() if np.random.rand() > 0.3 else None
                rh = fake_hand() if np.random.rand() > 0.3 else None

                keyframe = encode_keyframe(pose, lh, rh, seq=self._seq)
                await ws.send(keyframe)

                # Throttle to target FPS
                elapsed = time.monotonic() - t0
                sleep_for = max(0.0, self._frame_interval - elapsed)
                await asyncio.sleep(sleep_for)

            recv_task.cancel()
            print(f"\nStreaming done. {self._seq} frames sent.")
            print(f"Predictions received: {len(self._predictions)}")
            for p in self._predictions[-5:]:
                print(f"  [{p['start_frame']}–{p['end_frame']}] "
                      f"{p['gloss']} ({p['confidence']:.0%}) "
                      f"infer={p['inference_ms']:.1f}ms wait={p['queue_wait_ms']:.1f}ms")

    async def _receiver(self, ws) -> None:
        async for raw in ws:
            if isinstance(raw, str):
                continue
            try:
                frame = decode_frame(raw)
            except Exception as exc:
                print(f"[RECV] decode error: {exc}")
                continue

            match frame.msg_type:
                case MsgType.PREDICTION:
                    pred = frame.json()
                    self._predictions.append(pred)
                    print(f"[PREDICTION] {pred['gloss']} conf={pred['confidence']:.0%} "
                          f"infer={pred['inference_ms']:.1f}ms")

                case MsgType.PING:
                    await ws.send(encode_pong(frame.seq))

                case MsgType.BACKPRESSURE:
                    bp = frame.json()
                    print(f"[BACKPRESSURE] {bp['reason']} → slow to {bp['suggested_fps']:.0f}fps "
                          f"queue_depth={bp['queue_depth']}")
                    self._fps = bp["suggested_fps"]
                    self._frame_interval = 1.0 / self._fps

                case MsgType.ACK:
                    pass  # normal flow

                case MsgType.ERROR:
                    err = frame.json()
                    print(f"[ERROR] {err['code']}: {err['message']}")

                case _:
                    print(f"[RECV] unknown type: {frame.msg_type}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:8001/api/v1/ws/gesture")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=10.0, help="Streaming duration (seconds)")
    args = parser.parse_args()

    client = GestureClient(url=args.url, target_fps=args.fps)
    asyncio.run(client.run(duration_seconds=args.duration))


if __name__ == "__main__":
    main()
