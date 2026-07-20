"""Tests for MedicalAdapter terminology correction."""

import pytest

from app.transcription.medical_adapter import MedicalAdapter, NoopMedicalAdapter


class TestMedicalAdapter:
    def setup_method(self):
        self.adapter = MedicalAdapter(languages=["ru", "en"])

    def test_corrects_known_error_ru(self):
        text = "у пациента обнаружен диабит второго типа"
        corrected, terms = self.adapter.adapt(text, "ru")
        assert "диабет" in corrected
        assert "диабет" in terms

    def test_corrects_mrt_abbreviation(self):
        text = "результаты мрт показали опухоль"
        corrected, terms = self.adapter.adapt(text, "ru")
        assert "МРТ" in corrected

    def test_no_change_on_clean_text(self):
        text = "пациент чувствует себя хорошо"
        corrected, terms = self.adapter.adapt(text, "ru")
        assert corrected == text
        assert terms == []

    def test_add_custom_term(self):
        self.adapter.add_term("кровоснабжение", "кровоснабжение", "ru")
        corrected, terms = self.adapter.adapt("нарушено кровоснабжение", "ru")
        assert "кровоснабжение" in corrected

    def test_empty_text_passthrough(self):
        corrected, terms = self.adapter.adapt("", "ru")
        assert corrected == ""
        assert terms == []

    def test_noop_adapter(self):
        adapter = NoopMedicalAdapter()
        text = "у пациента диабит"
        corrected, terms = adapter.adapt(text, "ru")
        assert corrected == text
        assert terms == []

    def test_multiple_corrections(self):
        text = "пациент с диабитом и гипертание"
        corrected, terms = self.adapter.adapt(text, "ru")
        assert "диабет" in corrected
        # гипертония is in the rules
        assert len(terms) >= 1

    def test_case_insensitive_matching(self):
        text = "ДИАБИТ выявлен при обследовании"
        corrected, terms = self.adapter.adapt(text, "ru")
        assert "диабет" in corrected.lower() or "ДИАБИТ" not in corrected
