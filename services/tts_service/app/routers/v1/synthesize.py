"""REST synthesis endpoints."""

from __future__ import annotations

import base64
import struct
import time

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import Response, StreamingResponse

from ...audio.encoder import FrameFlags, FrameType, pack_frame
from ...domain.entities import AudioFormat, SynthesisRequest
from ...schemas.request import SynthesizeRequest, SynthesizeStreamRequest
from ...schemas.response import SynthesizeResponse, VoiceListResponse, VoiceProfileResponse

router = APIRouter()


def _get_pipeline(request: Request):
    return request.app.state.pipeline


# ── Full synthesis → JSON (base64 audio) ─────────────────────────────────────

@router.post(
    "/synthesize",
    response_model=SynthesizeResponse,
    summary="Synthesize text — returns base64-encoded audio in JSON",
)
async def synthesize(body: SynthesizeRequest, request: Request) -> SynthesizeResponse:
    pipeline = _get_pipeline(request)
    req = SynthesisRequest(
        text=body.text,
        voice_id=body.voice_id,
        language=body.language,
        sample_rate=body.sample_rate,
        format=body.format,
        put_accent=body.put_accent,
        put_yo=body.put_yo,
        session_id=body.session_id,
        priority=body.priority,
    )
    try:
        result = await pipeline.synthesize(req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {exc}") from exc

    return SynthesizeResponse(
        session_id=result.session_id,
        request_id=result.request_id,
        audio_b64=base64.b64encode(result.audio_bytes).decode(),
        format=result.format,
        sample_rate=result.sample_rate,
        duration_s=result.duration_s,
        char_count=result.char_count,
        voice_id=result.voice_id,
        from_cache=result.from_cache,
        inference_ms=round(result.inference_ms, 2),
        total_ms=round(result.total_ms, 2),
    )


# ── Full synthesis → binary (Content-Type: audio/wav or audio/ogg) ───────────

@router.post(
    "/synthesize/audio",
    summary="Synthesize text — returns raw audio bytes",
    responses={
        200: {
            "content": {"audio/wav": {}, "audio/ogg": {}},
            "description": "Raw encoded audio",
        }
    },
)
async def synthesize_audio(body: SynthesizeRequest, request: Request) -> Response:
    pipeline = _get_pipeline(request)
    req = SynthesisRequest(
        text=body.text,
        voice_id=body.voice_id,
        language=body.language,
        sample_rate=body.sample_rate,
        format=body.format,
        put_accent=body.put_accent,
        put_yo=body.put_yo,
        session_id=body.session_id,
        priority=body.priority,
    )
    try:
        result = await pipeline.synthesize(req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    media_type = "audio/ogg" if result.format == AudioFormat.OGG else "audio/wav"
    return Response(
        content=result.audio_bytes,
        media_type=media_type,
        headers={
            "X-Duration-S": str(result.duration_s),
            "X-Inference-Ms": str(round(result.inference_ms, 1)),
            "X-From-Cache": str(result.from_cache).lower(),
        },
    )


# ── Chunked streaming → length-prefixed WAV chunks ───────────────────────────

@router.post(
    "/synthesize/stream",
    summary="Stream audio chunks (sentence-by-sentence, length-prefixed binary frames)",
)
async def synthesize_stream(
    body: SynthesizeStreamRequest, request: Request
) -> StreamingResponse:
    pipeline = _get_pipeline(request)
    req = SynthesisRequest(
        text=body.text,
        voice_id=body.voice_id,
        language=body.language,
        sample_rate=body.sample_rate,
        format=body.format,
        put_accent=body.put_accent,
        put_yo=body.put_yo,
        session_id=body.session_id,
        chunk_size_chars=body.chunk_size_chars,
    )

    async def _generate():
        seq = 0
        async for chunk in pipeline.synthesize_streaming(req):
            flags = FrameFlags.LAST_CHUNK if chunk.is_final else FrameFlags.NONE
            frame = pack_frame(
                payload=chunk.audio_bytes,
                frame_type=FrameType.AUDIO,
                flags=flags,
                seq=seq,
            )
            yield frame
            seq += 1

    return StreamingResponse(
        _generate(),
        media_type="application/octet-stream",
        headers={"X-Frame-Format": "TTS-FRAME-v1"},
    )


# ── Voice catalogue ───────────────────────────────────────────────────────────

@router.get(
    "/voices",
    response_model=VoiceListResponse,
    summary="List available voice profiles",
)
async def list_voices(request: Request) -> VoiceListResponse:
    pipeline = _get_pipeline(request)
    voices = pipeline._synth.list_voices()
    return VoiceListResponse(
        voices=[
            VoiceProfileResponse(
                id=v.id,
                display_name=v.display_name,
                language=v.language,
                gender=v.gender,
                engine=v.engine,
                sample_rate=v.sample_rate,
            )
            for v in voices
        ]
    )
