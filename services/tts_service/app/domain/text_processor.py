"""Text preprocessing for TTS — normalization, sentence splitting, validation."""

from __future__ import annotations

import re
import unicodedata

# ── Abbreviation protection ───────────────────────────────────────────────────
# These are common Russian abbreviations where the period does NOT end a sentence.
_RU_ABBREVS = {
    "д-р", "д-ра", "проф", "акад", "ул", "пр", "пл", "г", "обл", "р-н",
    "тел", "факс", "стр", "стр-е", "корп", "оф", "кв", "эт", "пом",
    "им", "т", "д", "и", "б", "с", "ст", "пос", "пгт",
    "руб", "коп", "грн", "долл", "евро",
    "рис", "табл", "ст-я",
    "www", "http", "https",
}

_ABBREV_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in sorted(_RU_ABBREVS, key=len, reverse=True)) + r")\.",
    re.IGNORECASE,
)

# Ellipsis marker (must not split on ...)
_ELLIPSIS_PATTERN = re.compile(r"\.{2,}")

# Decimal numbers (1.5, 3.14 — must not split)
_DECIMAL_PATTERN = re.compile(r"\d+\.\d+")

# Sentence boundary: period/!? followed by space + uppercase (or end of string)
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+(?=[А-ЯA-Z\"\'\(«])")

# SSML-like tag stripping
_TAG_PATTERN = re.compile(r"<[^>]+>")

# Multiple spaces
_MULTI_SPACE = re.compile(r" {2,}")

# Max chars per TTS chunk (Silero degrades on very long inputs)
_MAX_CHUNK_CHARS = 200
_MIN_CHUNK_CHARS = 10


class TextPreprocessor:
    """Normalise and split text into TTS-ready sentence chunks."""

    def normalize(self, text: str) -> str:
        # Strip SSML/HTML tags
        text = _TAG_PATTERN.sub(" ", text)
        # Normalize unicode
        text = unicodedata.normalize("NFC", text)
        # Fix typographic quotes to straight
        text = text.replace("«", '"').replace("»", '"').replace("“", '"').replace("”", '"')
        # Normalize dashes
        text = text.replace("—", " — ").replace("–", "–")
        # Collapse whitespace
        text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        text = _MULTI_SPACE.sub(" ", text).strip()
        return text

    def split_sentences(self, text: str) -> list[str]:
        """Split normalized text into synthesizable sentence chunks."""
        text = self.normalize(text)
        if not text:
            return []

        # Protect abbreviations and decimals from splitting
        protected = self._protect(text)

        # Split on sentence boundaries
        parts = _SENTENCE_END.split(protected)

        # Restore protected tokens and clean up
        sentences: list[str] = []
        for part in parts:
            restored = self._restore(part).strip()
            if not restored:
                continue
            # Long sentences get further split at comma/semicolon boundaries
            if len(restored) > _MAX_CHUNK_CHARS:
                sentences.extend(self._split_long(restored))
            else:
                sentences.append(restored)

        return [s for s in sentences if len(s) >= _MIN_CHUNK_CHARS]

    def validate(self, text: str) -> tuple[bool, str]:
        """Return (is_valid, reason). Empty or too-long text is rejected."""
        stripped = text.strip()
        if not stripped:
            return False, "empty_text"
        if len(stripped) > 5000:
            return False, "text_too_long"
        return True, ""

    # ── Internal ──────────────────────────────────────────────────────────────

    _PLACEHOLDER_ELLIPSIS = "\x01ELLIP\x01"
    _PLACEHOLDER_DECIMAL  = "\x02DEC\x02"
    _PLACEHOLDER_ABBREV   = "\x03ABB\x03"

    def _protect(self, text: str) -> str:
        text = _ELLIPSIS_PATTERN.sub(self._PLACEHOLDER_ELLIPSIS, text)
        text = _DECIMAL_PATTERN.sub(lambda m: m.group().replace(".", self._PLACEHOLDER_DECIMAL), text)
        text = _ABBREV_PATTERN.sub(lambda m: m.group()[:-1] + self._PLACEHOLDER_ABBREV, text)
        return text

    def _restore(self, text: str) -> str:
        text = text.replace(self._PLACEHOLDER_ELLIPSIS, "...")
        text = text.replace(self._PLACEHOLDER_DECIMAL, ".")
        text = text.replace(self._PLACEHOLDER_ABBREV, ".")
        return text

    def _split_long(self, text: str) -> list[str]:
        """Split on comma/semicolon/dash while keeping chunks ≤ MAX_CHUNK_CHARS."""
        parts = re.split(r"(?<=[,;])\s+", text)
        chunks: list[str] = []
        buf = ""
        for part in parts:
            candidate = (buf + ", " + part).strip() if buf else part
            if len(candidate) <= _MAX_CHUNK_CHARS:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = part
        if buf:
            chunks.append(buf)
        return chunks or [text[:_MAX_CHUNK_CHARS]]
