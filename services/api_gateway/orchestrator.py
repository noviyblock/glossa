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

# Skip buffering when confidence is too low — usually means no person in
# frame or random noise classified by the model.
_MIN_CONFIDENCE = 0.15  # lowered for diagnostics; raise to 0.35+ in production
_HISTORY_TURNS  = 2      # recent translated sentences given to the LLM as context


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

        Auto-flush is a fallback, not the only way a sentence ends — the
        client can also trigger flush_pending_sentence()/delete_last_pending()
        explicitly (see api_gateway/main.py's "flush_sentence"/
        "delete_last_gesture" WS messages) so a user who notices a
        misrecognized gesture isn't stuck waiting for the pause timer.

        Returns:
            {
                "glosses": [{"gloss": str, "prob": float}, ...],
                "translation": str,   # non-empty only on the frame that flushed a sentence
                "confidence": float,
                "pending_positions": list[list[dict]],  # only present when the buffer changed this frame
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

        prior_session = await self._get_session(session_id) or {}
        history: list[str] = list(prior_session.get("history", []))
        pending_positions: list[list[dict]] = list(prior_session.get("pending_positions", []))
        last_gesture_ts: float = prior_session.get("last_gesture_ts", 0.0)

        now = time.time()
        translation = ""
        pending_changed = False

        if glosses and confidence >= _MIN_CONFIDENCE:
            # A gesture just completed — buffer its top-3 candidates rather
            # than translating it in isolation.
            pending_positions.append(glosses)
            last_gesture_ts = now
            pending_changed = True
            if len(pending_positions) >= MAX_SENTENCE_GLOSSES:
                translation = await self._flush_sentence(session_id, pending_positions, history)
                pending_positions = []
        elif pending_positions and (now - last_gesture_ts) >= SENTENCE_PAUSE_SECONDS:
            # No new gesture this frame, but the pause since the last one is
            # long enough to treat the buffer as a finished sentence. This is
            # the FALLBACK path — most useful when the user isn't actively
            # watching/correcting the buffer; a client that IS watching can
            # pre-empt this via flush_pending_sentence()/delete_last_pending().
            translation = await self._flush_sentence(session_id, pending_positions, history)
            pending_positions = []
            pending_changed = True

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        if glosses or translation:
            logger.info("rsl_to_text latency=%.1fms session=%s flushed=%s",
                        latency_ms, session_id, bool(translation))

        # Persist session state, including a short rolling history for the
        # next call's context (only append non-empty, genuinely new turns).
        if translation and (not history or history[-1] != translation):
            history = [*history, translation][-_HISTORY_TURNS:]

        await self._set_session(session_id, {
            "mode": "rsl_to_text",
            "last_translation": translation or prior_session.get("last_translation", ""),
            "last_glosses": glosses or prior_session.get("last_glosses", []),
            "latency_ms": latency_ms, "history": history,
            "pending_positions": pending_positions, "last_gesture_ts": last_gesture_ts,
        })

        result = {"glosses": glosses, "translation": translation, "confidence": confidence,
                  "keypoints": keypoints, "person_detected": person_detected,
                  "gesture_active": gesture_active, "preview": False}
        if pending_changed:
            result["pending_positions"] = pending_positions
        return result

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

    async def flush_pending_sentence(self, session_id: str) -> dict:
        """Explicit client-triggered "send now" — flushes pending_positions
        immediately, bypassing SENTENCE_PAUSE_SECONDS. No-op (empty
        translation) if the buffer is already empty.

        Returns: {"translation": str, "pending_positions": []}
        """
        session = await self._get_session(session_id) or {}
        pending_positions: list[list[dict]] = list(session.get("pending_positions", []))
        history: list[str] = list(session.get("history", []))

        if not pending_positions:
            return {"translation": "", "pending_positions": []}

        translation = await self._flush_sentence(session_id, pending_positions, history)
        if translation and (not history or history[-1] != translation):
            history = [*history, translation][-_HISTORY_TURNS:]

        await self._set_session(session_id, {
            **session,
            "last_translation": translation,
            "history": history,
            "pending_positions": [],
        })
        return {"translation": translation, "pending_positions": []}

    # ── Downstream calls with built-in graceful degradation ─────────────────
    #
    # Shared by both the WS path (process_audio) and the REST path
    # (translate_sync) for the same text_to_rsl direction — previously each
    # path duplicated (WS) or skipped (REST) this fallback logic, so REST
    # would raise a hard 500/502/504 on the whole request whenever NLP or
    # TTS hiccuped instead of degrading like WS already did.

    async def _translate_reverse(self, session_id: str, text: str) -> str:
        """Russian text -> RSL gloss sequence via NLP /translate_reverse.

        Falls back to echoing the input text on any failure — never raises.
        """
        if not text:
            return text
        try:
            resp = await self._http.post(
                f"{NLP_SERVICE_URL}/translate_reverse",
                json={"text": text, "session_id": session_id},
            )
            resp.raise_for_status()
            return resp.json().get("translation", text)
        except Exception as exc:
            logger.warning("NLP reverse error session=%s: %s — falling back to raw text", session_id, exc)
            return text

    async def _synthesize_audio(self, session_id: str, text: str) -> str:
        """Russian text -> base64 WAV via TTS /synthesize.

        Returns "" on any failure (including empty input) — never raises.
        """
        if not text:
            return ""
        try:
            resp = await self._http.post(
                f"{TTS_SERVICE_URL}/synthesize",
                json={"text": text, "speaker": "xenia"},
            )
            resp.raise_for_status()
            return resp.json().get("audio", "")
        except Exception as exc:
            logger.warning("TTS service error session=%s: %s", session_id, exc)
            return ""

    async def _render_sign_video(self, session_id: str, gloss_sequence: str) -> str | None:
        """gloss_sequence -> base64 MP4 via TTS /sign_video.

        None covers both "nothing matched a clip" and "the call failed" —
        both mean the same thing to the caller: no visual available. Never
        raises.
        """
        if not gloss_sequence:
            return None
        try:
            resp = await self._http.post(
                f"{TTS_SERVICE_URL}/sign_video",
                json={"gloss_sequence": gloss_sequence},
            )
            resp.raise_for_status()
            return resp.json().get("video")
        except Exception as exc:
            logger.warning("TTS sign_video error session=%s: %s", session_id, exc)
            return None

    async def _render_skeleton_sequence(self, session_id: str, gloss_sequence: str) -> list[dict] | None:
        """gloss_sequence -> per-gloss skeleton keypoint sequences via TTS
        /skeleton_sequence — alternative to _render_sign_video for clients
        that play back a skeleton (see clients/web's _skeletonPanel)
        instead of / alongside the concatenated-clip video. None covers
        both "nothing matched" and "the call failed", same as
        _render_sign_video. Never raises.
        """
        if not gloss_sequence:
            return None
        try:
            resp = await self._http.post(
                f"{TTS_SERVICE_URL}/skeleton_sequence",
                json={"gloss_sequence": gloss_sequence},
            )
            resp.raise_for_status()
            sequences = resp.json().get("sequences")
            return sequences or None
        except Exception as exc:
            logger.warning("TTS skeleton_sequence error session=%s: %s", session_id, exc)
            return None

    async def delete_last_pending(self, session_id: str) -> list[list[dict]]:
        """User-initiated correction: drop the most recently buffered
        gesture (e.g. it was misrecognized). Returns the updated buffer so
        the caller can echo it back to the client."""
        session = await self._get_session(session_id) or {}
        pending_positions: list[list[dict]] = list(session.get("pending_positions", []))
        if pending_positions:
            pending_positions.pop()
        await self._set_session(session_id, {**session, "pending_positions": pending_positions})
        return pending_positions

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
        gloss_sequence = await self._translate_reverse(session_id, russian_text)

        # Publish to nlp:results stream
        await self._redis.xadd(
            _STREAM_NLP_RESULTS,
            {"payload": json.dumps({"session_id": session_id, "gloss_sequence": gloss_sequence})},
            maxlen=500, approximate=True,
        )

        # 3. TTS service — synthesize Russian text to audio
        wav_b64 = await self._synthesize_audio(session_id, russian_text)

        # 4. TTS service — render gloss_sequence as concatenated sign clips.
        video_b64 = await self._render_sign_video(session_id, gloss_sequence)

        # 5. TTS service — render gloss_sequence as skeleton keypoint
        # sequences (see clients/web's _skeletonPanel playback).
        skeleton_sequences = await self._render_skeleton_sequence(session_id, gloss_sequence)

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.info("text_to_rsl latency=%.1fms session=%s", latency_ms, session_id)

        await self._set_session(session_id, {
            "mode": "text_to_rsl", "last_text": russian_text,
            "last_gloss_sequence": gloss_sequence, "latency_ms": latency_ms,
        })

        return {"text": russian_text, "gloss_sequence": gloss_sequence,
                "wav_b64": wav_b64, "video_b64": video_b64,
                "skeleton_sequences": skeleton_sequences}

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
            # Reverse: Russian text → RSL glosses. Same graceful-degradation
            # helpers as the WS path (process_audio) — NLP/TTS being down no
            # longer fails the whole REST request, it just returns whatever
            # partial content succeeded (echoed text, empty audio, no video).
            gloss_seq = await self._translate_reverse(session_id, text)
            wav_b64 = await self._synthesize_audio(session_id, text)
            video_b64 = await self._render_sign_video(session_id, gloss_seq)
            skeleton_sequences = await self._render_skeleton_sequence(session_id, gloss_seq)
            return {
                "translation": gloss_seq,
                "audio_wav": wav_b64,
                "video_mp4": video_b64,
                "skeleton_sequences": skeleton_sequences,
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
