from __future__ import annotations

import json
import re
from pathlib import Path

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


class GlossVocabulary:
    """The 200-word RSL gloss vocabulary translate_reverse must stay
    within (see punch-list follow-up: "RAG-глоссарий для домена в
    translate_reverse"). Loaded once at Translator startup, used two ways:

    1. `prompt_list()` — the full word list, for callers that genuinely
       want everything (kept for tests/introspection).
    2. `candidates()` — a cheap lexical pre-filter: vocabulary entries
       plausibly related to a given input text (exact phrase/substring
       match, or a shared-prefix fuzzy match per word to survive Russian
       inflection). Used to shrink the reverse-translation prompt down
       from all 200 entries to just the handful actually worth showing
       the model — shorter prompt (faster prefill on CPU) and a small
       model has an easier time complying with "pick from this list" when
       the list is 10-30 items instead of 200.
    3. `constrain()` — a deterministic post-filter applied to whatever the
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

    def candidates(self, text: str, max_candidates: int = 30, min_shared_prefix: int = 5) -> list[str]:
        """Vocabulary entries plausibly relevant to `text`. Two match
        modes, either is enough:

        - exact phrase: the whole normalized vocab entry appears verbatim
          in the normalized input (catches fixed multi-word idioms like
          "С днем рождения" used as-is).
        - per-word prefix fuzzy match: an input word and a vocab word
          share a long-enough prefix (catches simple inflection, e.g.
          "бабочку" vs "бабочка" -- share "бабочк" -- without a real
          morphology library).

        Falls back to the full list if nothing matches at all, so the
        model still gets a chance rather than an empty prompt.
        """
        # split on any non-word run (not just _PUNCT_RE) so hyphenated
        # compounds like "галстук-бабочку" become two input words instead
        # of merging into one -- matters here even though constrain()'s
        # normalize is fine leaving gloss-side punctuation alone.
        norm_text = " ".join(re.split(r"[^\w]+", text.lower(), flags=re.UNICODE)).strip()
        input_words = [w for w in norm_text.split() if len(w) >= 3]
        matched: list[str] = []
        for word in self._words:
            norm_word = self._normalize(word)
            if not norm_word:
                continue
            if norm_word in norm_text:
                matched.append(word)
                continue
            vocab_tokens = [t for t in norm_word.split() if len(t) >= 3]
            if any(
                self._shares_prefix(iw, vt, min_shared_prefix)
                for iw in input_words for vt in vocab_tokens
            ):
                matched.append(word)
        return matched[:max_candidates] if matched else self._words

    @staticmethod
    def _shares_prefix(a: str, b: str, min_shared: int) -> bool:
        shared = 0
        for ca, cb in zip(a, b):
            if ca != cb:
                break
            shared += 1
        return shared >= min(min_shared, len(a), len(b))

    def constrain(self, llm_output: str) -> str:
        """Keep only tokens/phrases from llm_output that literally match a
        vocabulary entry, in order, uppercased (matching the gloss-token
        convention the rest of the pipeline expects).

        Real regression: for a multi-word answer the model sometimes
        ignores the "через пробел" instruction and separates words with
        "|" or a newline instead (observed: 'Золото|корона|крест' for
        candidates ЗОЛОТО/КОРОНА/КРЕСТ -- all three are valid vocabulary
        words, but plain `.split()` treated the whole pipe-joined string
        as one token, which matched nothing, so the answer silently came
        back empty despite being "right"). Splitting on any of the
        delimiters a model plausibly uses instead of a space -- pipe,
        comma, semicolon, slash, middle dot, newline -- makes this
        tolerant of that formatting drift without weakening the actual
        vocabulary check that follows.
        """
        tokens = [t for t in re.split(r"[\s|,;/·•]+", llm_output.strip()) if t]
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
