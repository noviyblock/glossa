"""Audio encoding — PCM float32 → WAV / OGG bytes.

WAV: always available (stdlib wave module).
OGG: requires soundfile + libsndfile + OGG/Vorbis codec.
     Falls back gracefully to WAV if soundfile is unavailable.

Streaming WAV: synthesize_to_wav_stream() yields length-prefixed WAV chunks
suitable for chunked HTTP responses or WebSocket frames.
"""

from __future__ import annotations

import io
import struct
import wave

import numpy as np

from glossa_common.logging import get_logger

from ..domain.entities import AudioFormat
from ..domain.interfaces import AudioEncoderPort

logger = get_logger(__name__)

_OGG_AVAILABLE: bool | None = None   # lazy detection


def _check_ogg() -> bool:
    global _OGG_AVAILABLE
    if _OGG_AVAILABLE is None:
        try:
            import soundfile  # noqa: F401
            _OGG_AVAILABLE = True
        except ImportError:
            logger.warning("soundfile_not_available_ogg_disabled")
            _OGG_AVAILABLE = False
    return _OGG_AVAILABLE


class AudioEncoder(AudioEncoderPort):
    """Thread-safe (stateless) audio encoder."""

    def encode(self, pcm: np.ndarray, sample_rate: int, fmt: AudioFormat) -> bytes:
        if fmt == AudioFormat.OGG:
            if _check_ogg():
                return self.pcm_to_ogg(pcm, sample_rate)
            logger.warning("ogg_requested_but_unavailable_falling_back_to_wav")
        return self.pcm_to_wav(pcm, sample_rate)

    def pcm_to_wav(self, pcm: np.ndarray, sample_rate: int) -> bytes:
        """Float32 [-1, 1] → 16-bit PCM WAV bytes."""
        pcm_clipped = np.clip(pcm, -1.0, 1.0)
        pcm16 = (pcm_clipped * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)   # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16.tobytes())
        return buf.getvalue()

    def pcm_to_ogg(self, pcm: np.ndarray, sample_rate: int) -> bytes:
        """Float32 → OGG Vorbis bytes via soundfile."""
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, pcm.astype(np.float32), sample_rate, format="OGG", subtype="VORBIS")
        buf.seek(0)
        return buf.read()

    def pcm_to_raw_s16le(self, pcm: np.ndarray) -> bytes:
        """Float32 → raw 16-bit little-endian PCM (no header)."""
        return (np.clip(pcm, -1.0, 1.0) * 32767).astype("<i2").tobytes()

    def audio_duration(self, pcm: np.ndarray, sample_rate: int) -> float:
        """Duration in seconds."""
        return len(pcm) / sample_rate if sample_rate > 0 else 0.0


# ── Framing helpers for streaming ─────────────────────────────────────────────

# Binary frame layout for HTTP chunked and WebSocket transport:
#   [type:1][flags:1][seq:2][length:4][payload:N]  = 8-byte header
_FRAME_FMT = "!BBHi"
_FRAME_HEADER_SIZE = struct.calcsize(_FRAME_FMT)

assert _FRAME_HEADER_SIZE == 8


class FrameType:
    AUDIO = 0x01
    FINAL = 0x02
    ERROR = 0x03
    PING  = 0x04
    PONG  = 0x05


class FrameFlags:
    NONE     = 0x00
    LAST_CHUNK = 0x01
    FROM_CACHE = 0x02


def pack_frame(
    payload: bytes,
    frame_type: int = FrameType.AUDIO,
    flags: int = FrameFlags.NONE,
    seq: int = 0,
) -> bytes:
    header = struct.pack(_FRAME_FMT, frame_type, flags, seq & 0xFFFF, len(payload))
    return header + payload


def unpack_frame_header(data: bytes) -> tuple[int, int, int, int]:
    """Return (type, flags, seq, payload_length)."""
    if len(data) < _FRAME_HEADER_SIZE:
        raise ValueError(f"Frame header too short: {len(data)} < {_FRAME_HEADER_SIZE}")
    return struct.unpack(_FRAME_FMT, data[:_FRAME_HEADER_SIZE])
