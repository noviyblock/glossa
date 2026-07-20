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


def test_constrain_is_case_and_punctuation_insensitive(tmp_path: Path) -> None:
    vocab = _make_vocab(tmp_path, {"С днем рождения!": 0})

    result = vocab.constrain("с ДНЕМ рождения")

    assert result == "С ДНЕМ РОЖДЕНИЯ!"  # canonical spelling from the vocab, uppercased


def test_prompt_list_contains_every_word_one_per_line(tmp_path: Path) -> None:
    vocab = _make_vocab(tmp_path, {"привет": 0, "пока": 1})

    listing = vocab.prompt_list()

    assert "привет" in listing.splitlines()
    assert "пока" in listing.splitlines()
