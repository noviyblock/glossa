"""Service orchestrator.

Pipelines
─────────
RSL → Text:
  frame (base64) → CV /process_frame → glosses
                 → NLP /translate_topk → Russian text

Text → RSL:
  audio (base64) → ASR /transcribe → Russian text
                 → NLP /translate_reverse → RSL gloss sequence
                 → TTS /synthesize → WAV

Session state is stored in Redis with a 5-minute TTL.
Redis Streams (cv:results, asr:results, nlp:results) are also maintained so
independent consumers can subscribe to pipeline events.
"""
from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

import redis.asyncio as aioredis

from config import (
    ASR_SERVICE_URL,
    CV_SERVICE_URL,
    MAX_SENTENCE_GLOSSES,
    NLP_SERVICE_URL,
    SENTENCE_PAUSE_SECONDS,
    SESSION_TTL,
    TTS_SERVICE_URL,
)

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)

# Redis Stream names
_STREAM_CV_RESULTS  = "cv:results"
_STREAM_ASR_RESULTS = "asr:results"
_STREAM_NLP_RESULTS = "nlp:results"
_CONSUMER_GROUP     = "api-gateway"
_CONSUMER_NAME      = "gateway-0"


class Orchestrator:
    def __init__(self, redis: aioredis.Redis, http: httpx.AsyncClient) -> None:
        self._redis = redis
        self._http  = http

    # ── Startup ──────────────────────────────────────────────────────────── #

    async def setup_streams(self) -> None:
        """Create consumer groups for all result streams (idempotent)."""
        for stream in (_STREAM_CV_RESULTS, _STREAM_ASR_RESULTS, _STREAM_NLP_RESULTS):
            try:
                await self._redis.xgroup_create(stream, _CONSUMER_GROUP, id="$", mkstream=True)
                logger.info("Consumer group created: stream=%s group=%s", stream, _CONSUMER_GROUP)
            except aioredis.ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    logger.warning("xgroup_create %s: %s", stream, e)

    # ── Session state ─────────────────────────────────────────────────────── #

    async def _set_session(self, session_id: str, data: dict) -> None:
        await self._redis.setex(
            f"gw:session:{session_id}", SESSION_TTL, json.dumps(data)
        )

    async def _get_session(self, session_id: str) -> dict | None:
        raw = await self._redis.get(f"gw:session:{session_id}")
        return json.loads(raw) if raw else None

    async def delete_session(self, session_id: str) -> None:
        await self._redis.delete(f"gw:session:{session_id}")

    # ── Pipeline: RSL → Text ──────────────────────────────────────────────── #

    async def process_frame(self, session_id: str, frame_b64: str) -> dict:
        """Send one video frame through CV and return structured result.

        Completed gestures are NOT translated one at a time. Instead they're
        buffered per session (`pending_positions`, one top-3 candidate list
        per gesture) until a sentence boundary is detected — either a pause
        of SENTENCE_PAUSE_SECONDS since the last gesture, or the buffer
        reaching MAX_SENTENCE_GLOSSES — at which point the whole buffered
        sequence is translated in one LLM call via /translate_sequence_topk.
        This both cuts LLM calls (one per sentence, not per gesture) and
        gives the LLM the full sentence to disambiguate against, T9-style.

        Returns:
            {
                "glosses": [{"gloss": str, "prob": float}, ...],
                "translation": str,   # non-empty only on the frame that flushed a sentence
                "confidence": float,
            }
        """
        t0 = time.perf_counter()

        # 1. CV service — get top-3 glosses
        try:
            cv_resp = await self._http.post(
                f"{CV_SERVICE_URL}/process_frame",
                json={"data": frame_b64, "session_id": session_id},
            )
            cv_resp.raise_for_status()
            cv_data = cv_resp.json()
        except Exception as exc:
            logger.error("CV service error session=%s: %s", session_id, exc)
            raise RuntimeError(f"CV service unavailable: {exc}") from exc

        glosses: list[dict] = cv_data.get("glosses", [])
        confidence = glosses[0]["prob"] if glosses else 0.0
        keypoints: list | None = cv_data.get("keypoints")
        person_detected: bool = cv_data.get("person_detected", False)
        gesture_active: bool = cv_data.get("gesture_active", False)
        is_preview: bool = cv_data.get("preview", False)

        # Preview classifications (early, provisional — segment keeps
        # accumulating) skip translation and session persistence entirely:
        # the point is fast raw-gloss feedback, not a final answer. Running
        # the LLM on every preview would add seconds of latency right before
        # the real result and could clobber last_translation/history with an
        # empty/provisional value.
        if is_preview:
            return {"glosses": glosses, "translation": "", "confidence": confidence,
                    "keypoints": keypoints, "person_detected": person_detected,
                    "gesture_active": gesture_active, "preview": True}

        if glosses:
            await self._redis.xadd(
                _STREAM_CV_RESULTS,
                {"payload": json.dumps({"session_id": session_id, "glosses": glosses})},
                maxlen=500, approximate=True,
            )

        # Skip buffering when confidence is too low — usually means no
        # person in frame or random noise classified by the model.
        _MIN_CONFIDENCE = 0.15  # lowered for diagnostics; raise to 0.35+ in production
        _HISTORY_TURNS  = 2      # recent translated sentences given to the LLM as context

        prior_session = await self._get_session(session_id) or {}
        history: list[str] = list(prior_session.get("history", []))
        pending_positions: list[list[dict]] = list(prior_session.get("pending_positions", []))
        last_gesture_ts: float = prior_session.get("last_gesture_ts", 0.0)

        now = time.time()
        translation = ""

        if glosses and confidence >= _MIN_CONFIDENCE:
            # A gesture just completed — buffer its top-3 candidates rather
            # than translating it in isolation.
            pending_positions.append(glosses)
            last_gesture_ts = now
            if len(pending_positions) >= MAX_SENTENCE_GLOSSES:
                translation = await self._flush_sentence(session_id, pending_positions, history)
                pending_positions = []
        elif pending_positions and (now - last_gesture_ts) >= SENTENCE_PAUSE_SECONDS:
            # No new gesture this frame, but the pause since the last one is
            # long enough to treat the buffer as a finished sentence. This
            # branch is what actually flushes most sentences in practice,
            # since it's checked on every frame (not just gesture completions).
            translation = await self._flush_sentence(session_id, pending_positions, history)
            pending_positions = []

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        if glosses or translation:
            logger.info("rsl_to_text latency=%.1fms session=%s flushed=%s",
                        latency_ms, session_id, bool(translation))

        # Persist session state, including a short rolling history for the
        # next call's context (only append non-empty, genuinely new turns).
        if translation and (not history or history[-1] != translation):
            history = (history + [translation])[-_HISTORY_TURNS:]

        await self._set_session(session_id, {
            "mode": "rsl_to_text",
            "last_translation": translation or prior_session.get("last_translation", ""),
            "last_glosses": glosses or prior_session.get("last_glosses", []),
            "latency_ms": latency_ms, "history": history,
            "pending_positions": pending_positions, "last_gesture_ts": last_gesture_ts,
        })

        return {"glosses": glosses, "translation": translation, "confidence": confidence,
                "keypoints": keypoints, "person_detected": person_detected,
                "gesture_active": gesture_active, "preview": False}

    async def _flush_sentence(
        self, session_id: str, pending_positions: list[list[dict]], history: list[str],
    ) -> str:
        """Translate a full buffered gesture sequence in one LLM call."""
        try:
            nlp_resp = await self._http.post(
                f"{NLP_SERVICE_URL}/translate_sequence_topk",
                json={"positions": pending_positions, "context": history},
            )
            nlp_resp.raise_for_status()
            translation = nlp_resp.json().get("translation", "")
        except Exception as exc:
            logger.error("NLP sequence translate error session=%s: %s", session_id, exc)
            translation = " ".join(p[0]["gloss"] for p in pending_positions if p)

        await self._redis.xadd(
            _STREAM_NLP_RESULTS,
            {"payload": json.dumps({"session_id": session_id, "translation": translation})},
            maxlen=500, approximate=True,
        )
        return translation

    # ── Pipeline: Text → RSL ──────────────────────────────────────────────── #

    async def process_audio(self, session_id: str, audio_b64: str) -> dict:
        """Send audio through ASR → NLP-reverse → TTS.

        Returns:
            {
                "text": str,          # recognised Russian text
                "gloss_sequence": str,  # RSL gloss sequence
                "wav_b64": str,       # base64 WAV from TTS
                "video_b64": str | None,  # base64 MP4 sign clips (None if no glosses matched)
            }
        """
        t0 = time.perf_counter()

        # 1. ASR service — speech to text
        try:
            asr_resp = await self._http.post(
                f"{ASR_SERVICE_URL}/transcribe",
                json={"data": audio_b64, "session_id": session_id},
            )
            asr_resp.raise_for_status()
            asr_data = asr_resp.json()
        except Exception as exc:
            logger.error("ASR service error session=%s: %s", session_id, exc)
            raise RuntimeError(f"ASR service unavailable: {exc}") from exc

        russian_text: str = asr_data.get("text", "").strip()

        # Publish to asr:results stream
        await self._redis.xadd(
            _STREAM_ASR_RESULTS,
            {"payload": json.dumps({"session_id": session_id, "text": russian_text})},
            maxlen=500, approximate=True,
        )

        # 2. NLP service — Russian text → RSL gloss sequence (reverse translation)
        gloss_sequence = russian_text  # fallback: echo the text
        if russian_text:
            try:
                nlp_resp = await self._http.post(
                    f"{NLP_SERVICE_URL}/translate_reverse",
                    json={"text": russian_text, "session_id": session_id},
                )
                nlp_resp.raise_for_status()
                gloss_sequence = nlp_resp.json().get("translation", russian_text)
            except Exception as exc:
                logger.warning("NLP reverse error session=%s: %s — falling back to raw text", session_id, exc)

        # Publish to nlp:results stream
        await self._redis.xadd(
            _STREAM_NLP_RESULTS,
            {"payload": json.dumps({"session_id": session_id, "gloss_sequence": gloss_sequence})},
            maxlen=500, approximate=True,
        )

        # 3. TTS service — synthesize Russian text to audio
        wav_b64 = ""
        if russian_text:
            try:
                tts_resp = await self._http.post(
                    f"{TTS_SERVICE_URL}/synthesize",
                    json={"text": russian_text, "speaker": "xenia"},
                )
                tts_resp.raise_for_status()
                wav_b64 = tts_resp.json().get("audio", "")
            except Exception as exc:
                logger.warning("TTS service error session=%s: %s", session_id, exc)

        # 4. TTS service — render gloss_sequence as concatenated sign clips.
        # None (not "") when no glosses matched a clip — distinguishable from
        # "not attempted" so the client can tell "no visual available" apart
        # from a service error.
        video_b64: str | None = None
        if gloss_sequence:
            try:
                video_resp = await self._http.post(
                    f"{TTS_SERVICE_URL}/sign_video",
                    json={"gloss_sequence": gloss_sequence},
                )
                video_resp.raise_for_status()
                video_b64 = video_resp.json().get("video")
            except Exception as exc:
                logger.warning("TTS sign_video error session=%s: %s", session_id, exc)

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.info("text_to_rsl latency=%.1fms session=%s", latency_ms, session_id)

        await self._set_session(session_id, {
            "mode": "text_to_rsl", "last_text": russian_text,
            "last_gloss_sequence": gloss_sequence, "latency_ms": latency_ms,
        })

        return {"text": russian_text, "gloss_sequence": gloss_sequence,
                "wav_b64": wav_b64, "video_b64": video_b64}

    # ── Synchronous REST orchestration ────────────────────────────────────── #

    async def translate_sync(
        self,
        mode: str,
        *,
        gloss_sequence: str | None = None,
        text: str | None = None,
        session_id: str = "",
    ) -> dict:
        """Synchronous REST translation (no streaming)."""
        t0 = time.perf_counter()

        if mode == "rsl_to_text":
            if not gloss_sequence:
                raise ValueError("gloss_sequence required for rsl_to_text")
            nlp_resp = await self._http.post(
                f"{NLP_SERVICE_URL}/translate",
                json={"gloss_sequence": gloss_sequence, "session_id": session_id},
            )
            nlp_resp.raise_for_status()
            translation = nlp_resp.json().get("translation", "")
            return {
                "translation": translation,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            }

        elif mode == "text_to_rsl":
            if not text:
                raise ValueError("text required for text_to_rsl")
            # Reverse: Russian text → RSL glosses
            nlp_resp = await self._http.post(
                f"{NLP_SERVICE_URL}/translate_reverse",
                json={"text": text, "session_id": session_id},
            )
            nlp_resp.raise_for_status()
            gloss_seq = nlp_resp.json().get("translation", "")
            # Also synthesize audio
            tts_resp = await self._http.post(
                f"{TTS_SERVICE_URL}/synthesize",
                json={"text": text, "speaker": "xenia"},
            )
            tts_resp.raise_for_status()
            # And render the gloss sequence as concatenated sign clips —
            # best-effort, None if nothing matched (see SignVideoAssembler).
            video_b64: str | None = None
            if gloss_seq:
                try:
                    video_resp = await self._http.post(
                        f"{TTS_SERVICE_URL}/sign_video",
                        json={"gloss_sequence": gloss_seq},
                    )
                    video_resp.raise_for_status()
                    video_b64 = video_resp.json().get("video")
                except Exception as exc:
                    logger.warning("TTS sign_video error session=%s: %s", session_id, exc)
            return {
                "translation": gloss_seq,
                "audio_wav": tts_resp.json().get("audio", ""),
                "video_mp4": video_b64,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            }

        raise ValueError(f"Unknown mode: {mode}")

    # ── Service health checks ──────────────────────────────────────────────── #

    async def check_services(self) -> dict[str, str]:
        services = {
            "cv":  f"{CV_SERVICE_URL}/health/live",
            "asr": f"{ASR_SERVICE_URL}/health/live",
            "nlp": f"{NLP_SERVICE_URL}/health/live",
            "tts": f"{TTS_SERVICE_URL}/health/live",
        }
        statuses: dict[str, str] = {}
        for name, url in services.items():
            try:
                r = await self._http.get(url, timeout=3.0)
                statuses[name] = "healthy" if r.status_code == 200 else "degraded"
            except Exception:
                statuses[name] = "unreachable"
        return statuses
