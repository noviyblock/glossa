"""Unit tests for GestureClassifier's pure-logic pieces: class-map loading,
top-k formation, and the missing-index fallback -- deliberately does NOT
instantiate a real GestureClassifier (that loads an actual OpenVINO/ONNX
model from disk in __init__, too heavy for a unit test; see punch-list plan,
item 6, which explicitly scopes asr/tts-style heavy-model tests out of this
pass). Static/pure methods are called directly; instance methods that only
touch self._idx_to_class use object.__new__() to skip __init__ entirely.

Run: pytest services/cv_service/tests/ -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# See test_gesture_segmenter.py for why this pop is needed -- gesture_classifier
# imports from config.py too, and another service's tests may have already
# cached a different config.py under sys.modules['config'] in this process.
sys.modules.pop("config", None)

from gesture_classifier import GestureClassifier  # noqa: E402


def _make_classifier_with_class_map(idx_to_class: dict[int, str]) -> GestureClassifier:
    """Instance with only _idx_to_class set -- skips __init__/_load_model."""
    clf = object.__new__(GestureClassifier)
    clf._idx_to_class = idx_to_class
    return clf


def test_load_class_map_parses_string_keys_to_int(tmp_path: Path) -> None:
    path = tmp_path / "idx_to_class.json"
    path.write_text(
        json.dumps({"0": "привет", "5": "пока", "12": "спасибо"}, ensure_ascii=False),
        encoding="utf-8",
    )

    result = GestureClassifier._load_class_map(str(path))

    assert result == {0: "привет", 5: "пока", 12: "спасибо"}
    assert all(isinstance(k, int) for k in result)


def test_top3_from_probs_returns_highest_first() -> None:
    clf = _make_classifier_with_class_map({0: "a", 1: "b", 2: "c", 3: "d"})
    probs = np.array([0.1, 0.6, 0.05, 0.25], dtype=np.float32)

    top3 = clf._top3_from_probs(probs)

    assert [r["gloss"] for r in top3] == ["b", "d", "a"]
    assert top3[0]["prob"] == pytest.approx(0.6)
    # descending by prob
    assert top3[0]["prob"] >= top3[1]["prob"] >= top3[2]["prob"]


def test_top3_from_probs_falls_back_to_str_index_when_unmapped() -> None:
    """A class index missing from idx_to_class.json (e.g. stale/edited
    vocab file) must not crash inference -- falls back to the raw index
    as a string rather than raising KeyError."""
    clf = _make_classifier_with_class_map({0: "a"})  # index 1 deliberately missing
    probs = np.array([0.2, 0.8], dtype=np.float32)

    top3 = clf._top3_from_probs(probs)

    assert top3[0]["gloss"] == "1"  # str(1), not a KeyError
    assert top3[1]["gloss"] == "a"


def test_top3_from_probs_respects_top_k_even_with_more_classes() -> None:
    clf = _make_classifier_with_class_map({i: str(i) for i in range(10)})
    probs = np.array([0.05] * 10, dtype=np.float32)
    probs[7] = 0.55

    top3 = clf._top3_from_probs(probs)

    assert len(top3) == 3
    assert top3[0]["gloss"] == "7"


def test_softmax_sums_to_one_and_preserves_order() -> None:
    x = np.array([1.0, 3.0, 2.0], dtype=np.float32)

    probs = GestureClassifier._softmax(x)

    assert probs.sum() == pytest.approx(1.0, abs=1e-6)
    assert probs[1] > probs[2] > probs[0]  # order preserved (3 > 2 > 1)


def test_softmax_is_numerically_stable_on_large_logits() -> None:
    """Subtracting max before exp() must avoid overflow on realistic
    (post nan_to_num clamp) logit magnitudes."""
    x = np.array([1e4, -1e4, 0.0], dtype=np.float32)

    probs = GestureClassifier._softmax(x)

    assert np.all(np.isfinite(probs))
    assert probs.sum() == pytest.approx(1.0, abs=1e-3)
    assert probs[0] > probs[1]
