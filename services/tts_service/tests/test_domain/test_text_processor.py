"""Tests for TextPreprocessor — normalization and sentence splitting."""

from __future__ import annotations

import pytest

from app.domain.text_processor import TextPreprocessor


@pytest.fixture
def pre() -> TextPreprocessor:
    return TextPreprocessor()


# ── normalize ─────────────────────────────────────────────────────────────────

def test_normalize_strips_html_tags(pre):
    result = pre.normalize("<b>Текст</b> и <i>курсив</i>")
    assert "<" not in result
    assert "Текст" in result
    assert "курсив" in result


def test_normalize_collapses_whitespace(pre):
    result = pre.normalize("Один   два\tтри\nчетыре")
    assert "  " not in result
    assert result == "Один два три четыре"


def test_normalize_fixes_typographic_quotes(pre):
    result = pre.normalize("«Привет» и "мир"")
    assert "«" not in result
    assert "»" not in result


def test_normalize_empty_string(pre):
    assert pre.normalize("") == ""
    assert pre.normalize("   ") == ""


def test_normalize_returns_stripped(pre):
    assert pre.normalize("  Привет  ") == "Привет"


# ── split_sentences ───────────────────────────────────────────────────────────

def test_split_single_sentence(pre):
    sentences = pre.split_sentences("Добрый день.")
    assert len(sentences) == 1
    assert sentences[0] == "Добрый день."


def test_split_multiple_sentences(pre):
    text = "Добрый день. Как дела? Хорошо!"
    sentences = pre.split_sentences(text)
    assert len(sentences) == 3


def test_split_does_not_break_on_abbreviation(pre):
    text = "Проф. Иванов работает в ул. Ленина."
    sentences = pre.split_sentences(text)
    # Should not split "ул. Ленина" — abbreviation protection
    assert len(sentences) == 1


def test_split_does_not_break_decimal_numbers(pre):
    text = "Цена составляет 3.14 рубля."
    sentences = pre.split_sentences(text)
    assert len(sentences) == 1
    assert "3.14" in sentences[0]


def test_split_empty_text_returns_empty(pre):
    assert pre.split_sentences("") == []
    assert pre.split_sentences("   ") == []


def test_split_long_sentence_at_comma(pre):
    # Sentence over 200 chars should be split at comma boundaries
    long_text = "Это очень длинное предложение, которое содержит множество слов и запятых, и оно должно быть разбито на несколько частей, потому что одна синтезирующая система не может обработать слишком длинный текст за один раз."
    sentences = pre.split_sentences(long_text)
    # All chunks should be within max char limit
    for s in sentences:
        assert len(s) <= 220  # small buffer above max


def test_split_strips_min_length_chunks(pre):
    # Very short fragments should be filtered out
    sentences = pre.split_sentences("А. Б. В.")
    # Fragments "А", "Б", "В" are < MIN_CHUNK_CHARS so filtered
    assert all(len(s) >= 10 for s in sentences)


# ── validate ──────────────────────────────────────────────────────────────────

def test_validate_valid_text(pre):
    ok, reason = pre.validate("Привет мир.")
    assert ok is True
    assert reason == ""


def test_validate_empty_text(pre):
    ok, reason = pre.validate("")
    assert ok is False
    assert reason == "empty_text"


def test_validate_whitespace_only(pre):
    ok, reason = pre.validate("   ")
    assert ok is False
    assert reason == "empty_text"


def test_validate_too_long(pre):
    ok, reason = pre.validate("А" * 5001)
    assert ok is False
    assert reason == "text_too_long"


def test_validate_exactly_max_length(pre):
    ok, _ = pre.validate("А" * 5000)
    assert ok is True
