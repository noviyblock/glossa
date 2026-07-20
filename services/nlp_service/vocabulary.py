from __future__ import annotations

import json
import re
from pathlib import Path

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


class GlossVocabulary:
    """The 200-word RSL gloss vocabulary translate_reverse must stay
    within (see punch-list follow-up: "RAG-глоссарий для домена в
    translate_reverse"). Loaded once at Translator startup, used two ways:

    1. `prompt_list()` — the full word list embedded directly in the
       reverse-translation system prompt. 200 short phrases fit easily in
       a small model's context window, so there's no need for real
       embedding-based retrieval over a corpus this size — the "retrieval
       corpus" for this RAG-lite IS the whole vocabulary, every time.
    2. `constrain()` — a deterministic post-filter applied to whatever the
       LLM actually outputs, dropping anything that doesn't literally
       match a vocabulary entry. A prompt instruction alone isn't a
       guarantee, especially from a 1.5B model — this is the real
       hallucination guard, not the prompt wording.

    Matching (normalize + greedy longest-phrase-first) mirrors
    SignVideoAssembler/SkeletonSequenceProvider in tts_service for
    consistency, since all three ultimately key off the same
    class_to_idx.json vocabulary -- duplicated, not shared, each lives in
    a different service.
    """

    def __init__(self, class_map_path: str) -> None:
        raw: dict[str, int] = json.loads(Path(class_map_path).read_text(encoding="utf-8"))
        self._words: list[str] = sorted(raw.keys())
        self._norm_to_word: dict[str, str] = {self._normalize(w): w for w in self._words}
        self._max_words = max((len(k.split()) for k in self._norm_to_word), default=1)

    @staticmethod
    def _normalize(s: str) -> str:
        return _PUNCT_RE.sub("", s).strip().lower()

    @property
    def words(self) -> list[str]:
        return self._words

    def prompt_list(self) -> str:
        return "\n".join(self._words)

    def constrain(self, llm_output: str) -> str:
        """Keep only tokens/phrases from llm_output that literally match a
        vocabulary entry, in order, uppercased (matching the gloss-token
        convention the rest of the pipeline expects)."""
        tokens = llm_output.strip().split()
        kept: list[str] = []
        i = 0
        while i < len(tokens):
            matched = False
            max_span = min(self._max_words, len(tokens) - i)
            for span in range(max_span, 0, -1):
                phrase = self._normalize(" ".join(tokens[i:i + span]))
                word = self._norm_to_word.get(phrase)
                if word is None:
                    continue
                kept.append(word.upper())
                i += span
                matched = True
                break
            if not matched:
                i += 1
        return " ".join(kept)
