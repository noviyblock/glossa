from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path

from config import CLASS_TO_IDX_PATH, GLOSS_CLIPS_DIR

logger = logging.getLogger(__name__)

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


class SignVideoAssembler:
    """gloss_sequence -> MP4 by concatenating pre-cut reference clips
    (notebooks/colab_glossa_05_gloss_clips.ipynb, one clip per class, all
    exported to the same codec/resolution/fps) via ffmpeg's concat demuxer —
    stream-copy, no re-encode.

    Matching is normalized (case/punctuation-insensitive) and greedy
    longest-phrase-first, because class names in class_to_idx.json are not
    uniform single uppercase words (e.g. "2 часа", "Добро пожаловать!") while
    translate_reverse's LLM output is NOT currently constrained to that exact
    vocabulary — this is a best-effort bridge, not a guarantee of a match.
    Full reliability needs translate_reverse itself grounded in the known
    200-class vocabulary (see the RAG-glossary follow-up).
    """

    def __init__(self, clips_dir: str = GLOSS_CLIPS_DIR, class_map_path: str = CLASS_TO_IDX_PATH) -> None:
        self._clips_dir = Path(clips_dir)
        raw_map: dict[str, int] = json.loads(Path(class_map_path).read_text(encoding="utf-8"))
        self._norm_to_idx: dict[str, int] = {self._normalize(k): v for k, v in raw_map.items()}
        self._max_words = max((len(k.split()) for k in raw_map), default=1)
        available = sum(1 for idx in raw_map.values() if (self._clips_dir / f"{idx}.mp4").exists())
        logger.info(
            "SignVideoAssembler: %d/%d classes have a clip on disk (clips_dir=%s)",
            available, len(raw_map), self._clips_dir,
        )

    @staticmethod
    def _normalize(s: str) -> str:
        return _PUNCT_RE.sub("", s).strip().upper()

    # ------------------------------------------------------------------ #

    def build(self, gloss_sequence: str) -> bytes | None:
        """Return MP4 bytes for the gloss sequence, or None if none of the
        (space-separated) glosses matched a clip on disk."""
        tokens = gloss_sequence.strip().split()
        clip_paths: list[Path] = []
        missing: list[str] = []

        i = 0
        while i < len(tokens):
            matched = False
            max_span = min(self._max_words, len(tokens) - i)
            for span in range(max_span, 0, -1):
                phrase = self._normalize(" ".join(tokens[i:i + span]))
                idx = self._norm_to_idx.get(phrase)
                if idx is None:
                    continue
                path = self._clips_dir / f"{idx}.mp4"
                if not path.exists():
                    continue
                clip_paths.append(path)
                i += span
                matched = True
                break
            if not matched:
                missing.append(tokens[i])
                i += 1

        if missing:
            logger.warning("sign_video: no clip for tokens %r (skipped)", missing)
        if not clip_paths:
            return None
        return self._concat(clip_paths)

    @staticmethod
    def _concat(clip_paths: list[Path]) -> bytes:
        with tempfile.TemporaryDirectory() as tmp:
            list_file = Path(tmp) / "concat.txt"
            list_file.write_text(
                "\n".join(f"file '{p.resolve()}'" for p in clip_paths), encoding="utf-8",
            )
            out_path = Path(tmp) / "out.mp4"
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", str(list_file),
                "-c", "copy",
                str(out_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg concat failed: {result.stderr[-500:]}")
            return out_path.read_bytes()
