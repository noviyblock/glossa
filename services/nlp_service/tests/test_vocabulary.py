"""Unit tests for GlossVocabulary: the deterministic guardrail behind
translate_reverse's constrained prompt (see punch-list follow-up:
"RAG-глоссарий для домена в translate_reverse"). Regression cases are the
exact real failures reported: "бабочка" -> "бабочка приближается",
"с днем рождения" -> "верный добрый друг".

Run: pytest services/nlp_service/tests/ -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.pop("config", None)

from vocabulary import GlossVocabulary  # noqa: E402


def _make_vocab(tmp_path: Path, words: dict[str, int]) -> GlossVocabulary:
    path = tmp_path / "class_to_idx.json"
    path.write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
    return GlossVocabulary(str(path))


def test_constrain_keeps_valid_multiword_and_single_word_glosses(tmp_path: Path) -> None:
    vocab = _make_vocab(tmp_path, {"С днем рождения": 0, "бабочка": 1})

    result = vocab.constrain("С ДНЕМ РОЖДЕНИЯ БАБОЧКА")

    assert result == "С ДНЕМ РОЖДЕНИЯ БАБОЧКА"


def test_constrain_drops_hallucinated_words_not_in_vocabulary(tmp_path: Path) -> None:
    """Real regression: model output 'бабочка приближается' for input
    'бабочка' -- only 'бабочка' is an actual gloss."""
    vocab = _make_vocab(tmp_path, {"бабочка": 0})

    result = vocab.constrain("БАБОЧКА ПРИБЛИЖАЕТСЯ")

    assert result == "БАБОЧКА"


def test_constrain_drops_entirely_unrelated_hallucination(tmp_path: Path) -> None:
    """Real regression: model output 'верный добрый друг' for input
    'с днем рождения' -- none of those three words is the actual gloss
    (which is the 3-word phrase 'с днем рождения', not present here at
    all)."""
    vocab = _make_vocab(tmp_path, {"с днем рождения": 0, "друг": 1})

    result = vocab.constrain("ВЕРНЫЙ ДОБРЫЙ ДРУГ")

    assert result == "ДРУГ"


def test_constrain_greedy_prefers_longest_matching_phrase(tmp_path: Path) -> None:
    vocab = _make_vocab(tmp_path, {"время": 0, "время по гринвичу": 1})

    result = vocab.constrain("ВРЕМЯ ПО ГРИНВИЧУ")

    assert result == "ВРЕМЯ ПО ГРИНВИЧУ"


def test_constrain_empty_input_returns_empty(tmp_path: Path) -> None:
    vocab = _make_vocab(tmp_path, {"привет": 0})

    assert vocab.constrain("") == ""
    assert vocab.constrain("НЕСУЩЕСТВУЮЩЕЕСЛОВО") == ""


def test_constrain_tolerates_pipe_separated_output(tmp_path: Path) -> None:
    """Real regression: model output 'Золото|корона|крест' for a 3-word
    answer where all three words are valid vocabulary entries -- plain
    whitespace .split() treated the whole string as one unmatched token,
    so the answer came back empty despite being correct."""
    vocab = _make_vocab(tmp_path, {"золото": 0, "корона": 1, "крест": 2})

    result = vocab.constrain("Золото|корона|крест")

    assert result == "ЗОЛОТО КОРОНА КРЕСТ"


def test_constrain_tolerates_newline_separated_output(tmp_path: Path) -> None:
    vocab = _make_vocab(tmp_path, {"золото": 0, "корона": 1})

    result = vocab.constrain("золото\nкорона")

    assert result == "ЗОЛОТО КОРОНА"


def test_constrain_is_case_and_punctuation_insensitive(tmp_path: Path) -> None:
    vocab = _make_vocab(tmp_path, {"С днем рождения!": 0})

    result = vocab.constrain("с ДНЕМ рождения")

    assert result == "С ДНЕМ РОЖДЕНИЯ!"  # canonical spelling from the vocab, uppercased


def test_prompt_list_contains_every_word_one_per_line(tmp_path: Path) -> None:
    vocab = _make_vocab(tmp_path, {"привет": 0, "пока": 1})

    listing = vocab.prompt_list()

    assert "привет" in listing.splitlines()
    assert "пока" in listing.splitlines()


def test_candidates_matches_fixed_phrase_and_inflected_word(tmp_path: Path) -> None:
    """Real production case: 'С Днем Рождения! Надень галстук-бабочку' ->
    the 200-word prompt made the model paraphrase instead of selecting
    (translate_reverse latency=19745.0ms, output matched no vocabulary
    entry). candidates() should retrieve both relevant entries from a
    larger vocabulary, exact-phrase for the idiom and prefix-fuzzy for the
    inflected "бабочку" -> "бабочка"."""
    vocab = _make_vocab(tmp_path, {
        "С днем рождения": 0, "бабочка": 1, "футбол": 2, "смех": 3, "белка": 4,
    })

    result = vocab.candidates("С Днем Рождения! Надень галстук-бабочку")

    assert "С днем рождения" in result
    assert "бабочка" in result
    assert "футбол" not in result
    assert "смех" not in result


def test_candidates_ignores_short_word_false_positives(tmp_path: Path) -> None:
    """A 2-letter vocab token like 'на' inside a longer phrase must not
    fuzzy-match every input word that happens to start with those same two
    letters (e.g. 'надень') -- would defeat the point of shrinking the
    candidate list."""
    vocab = _make_vocab(tmp_path, {"позвонить на сервис": 0, "бабочка": 1})

    result = vocab.candidates("Надень галстук-бабочку")

    assert "позвонить на сервис" not in result
    assert "бабочка" in result


def test_candidates_falls_back_to_full_list_when_nothing_matches(tmp_path: Path) -> None:
    vocab = _make_vocab(tmp_path, {"футбол": 0, "смех": 1})

    result = vocab.candidates("совершенно не связанный текст")

    assert result == vocab.words


def test_candidates_respects_max_candidates(tmp_path: Path) -> None:
    vocab = _make_vocab(tmp_path, {f"тест{i}": i for i in range(10)})

    result = vocab.candidates("тест0 тест1 тест2", max_candidates=2)

    assert len(result) == 2
