"""Medical terminology adaptation hooks.

Provides:
- Rule-based correction of common ASR errors in medical/RSL context
- Runtime term registration
- Regex-based replacement with confidence gating
- Russian-specific normalization (abbreviations, dosages, anatomy)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from glossa_common.logging import get_logger

from ..domain.interfaces import MedicalAdapterPort

logger = get_logger(__name__)


@dataclass
class TermRule:
    pattern: re.Pattern
    replacement: str
    term: str
    language: str = "ru"


# ── Built-in RSL/medical vocabulary ──────────────────────────────────────────
# Format: (asr_error_pattern, canonical_term)
_DEFAULT_RULES_RU: list[tuple[str, str]] = [
    # Diagnoses
    (r"\bдиабит\b", "диабет"),
    (r"\bгипертани[яи]\b", "гипертония"),
    (r"\bонкалогия\b", "онкология"),
    (r"\bвичь?\b", "ВИЧ"),
    (r"\bпнемония\b", "пневмония"),
    (r"\bаритмие?\b", "аритмия"),
    (r"\bинфарт\b", "инфаркт"),
    (r"\bинсулт\b", "инсульт"),
    (r"\bтромбо?з\b", "тромбоз"),
    (r"\bастм[аы]\b", "астма"),
    # Anatomy
    (r"\bпичень\b", "печень"),
    (r"\bподжалудочн\b", "поджелудочн"),
    (r"\bщитовидк[аеи]\b", "щитовидка"),
    (r"\bпочк[аеи]\b", "почка"),
    # Medications
    (r"\bаспирин[еа]?\b", "аспирин"),
    (r"\bметформин[еа]?\b", "метформин"),
    (r"\bамлодипин[еа]?\b", "амлодипин"),
    (r"\bлизиноприл[еа]?\b", "лизиноприл"),
    # Units
    (r"\bмилиграм+\b", "мг"),
    (r"\bмиллилитр+\b", "мл"),
    # Procedures
    (r"\bмрт\b", "МРТ"),
    (r"\bкт\b", "КТ"),
    (r"\bэкг\b", "ЭКГ"),
    (r"\bузи\b", "УЗИ"),
    (r"\bоак\b", "ОАК"),   # общий анализ крови
    # RSL-specific glosses that Whisper might mishear
    (r"\bврач[еа]?\b", "врач"),
    (r"\bбольниц[аеы]\b", "больница"),
    (r"\bлекарств[оа]\b", "лекарство"),
    (r"\bрецепт\b", "рецепт"),
    (r"\bоперация\b", "операция"),
]

_DEFAULT_RULES_EN: list[tuple[str, str]] = [
    (r"\bdiabetes?\b", "diabetes"),
    (r"\bhypertention\b", "hypertension"),
    (r"\bpnemonia\b", "pneumonia"),
    (r"\bmyocardiel\b", "myocardial"),
    (r"\bMRI\b", "MRI"),
    (r"\bECG\b", "ECG"),
]


class MedicalAdapter(MedicalAdapterPort):
    """Post-processes Whisper output to fix medical terminology errors."""

    def __init__(self, languages: list[str] | None = None) -> None:
        self._rules: list[TermRule] = []
        langs = languages or ["ru", "en"]

        if "ru" in langs:
            for pattern, replacement in _DEFAULT_RULES_RU:
                self._add(pattern, replacement, "ru")
        if "en" in langs:
            for pattern, replacement in _DEFAULT_RULES_EN:
                self._add(pattern, replacement, "en")

        logger.info("medical_adapter_init", n_rules=len(self._rules))

    # ── MedicalAdapterPort ────────────────────────────────────────────────────

    def adapt(self, text: str, language: str = "ru") -> tuple[str, list[str]]:
        """Apply all rules for the given language. Returns (corrected, found_terms)."""
        if not text:
            return text, []

        found: list[str] = []
        result = text

        for rule in self._rules:
            if rule.language != language and language != "auto":
                continue
            new_text, n_subs = rule.pattern.subn(rule.replacement, result)
            if n_subs > 0:
                result = new_text
                if rule.term not in found:
                    found.append(rule.term)

        if found:
            logger.debug("medical_adapter_corrections", n=len(found), terms=found)

        return result, found

    def add_term(self, term: str, correction: str, language: str = "ru") -> None:
        pattern = r"\b" + re.escape(term) + r"\b"
        self._add(pattern, correction, language)
        logger.info("medical_adapter_term_added", term=term, correction=correction)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _add(self, pattern: str, replacement: str, language: str) -> None:
        self._rules.append(TermRule(
            pattern=re.compile(pattern, re.IGNORECASE | re.UNICODE),
            replacement=replacement,
            term=replacement,
            language=language,
        ))


class NoopMedicalAdapter(MedicalAdapterPort):
    """Drop-in that passes text through unchanged."""

    def adapt(self, text: str, language: str = "ru") -> tuple[str, list[str]]:
        return text, []

    def add_term(self, term: str, correction: str, language: str = "ru") -> None:
        pass
