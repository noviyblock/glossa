"""Unit tests for SkeletonSequenceProvider: normalize+greedy-match (same
convention as SignVideoAssembler) and the coordinate remap into [0,1]
screen-space. Uses a small synthetic processed_64_200-shaped dataset in
tmp_path instead of the real (DVC-tracked, multi-GB) dataset -- keeps this
fast and independent of `dvc pull` having run.

Run: pytest services/tts_service/tests/ -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.pop("config", None)

from skeleton import SkeletonSequenceProvider  # noqa: E402


def _make_dataset(root: Path, class_names: list[str], labels: list[int]) -> None:
    (root / "val").mkdir(parents=True)
    (root / "class_names.json").write_text(json.dumps(class_names, ensure_ascii=False), encoding="utf-8")
    n, t, j = len(labels), 4, 75
    features = np.zeros((n, t, j, 3), dtype=np.float32)
    for i in range(n):
        # A few real (nonzero, confident) points spread over a known range,
        # so the bbox-fit remap has something non-degenerate to work with.
        features[i, :, 0, :] = [10.0, 20.0, 1.0]
        features[i, :, 1, :] = [-5.0, 15.0, 1.0]
        features[i, :, 2, :] = [10.0, -5.0, 0.8]
    np.save(root / "val" / "features.npy", features)
    np.save(root / "val" / "labels.npy", np.array(labels, dtype=np.int64))


def test_get_matches_single_word_gloss(tmp_path: Path) -> None:
    _make_dataset(tmp_path, ["привет", "пока"], labels=[0, 1])
    provider = SkeletonSequenceProvider(processed_dir=str(tmp_path), split="val")

    result = provider.get("привет")

    assert len(result) == 1
    assert result[0]["gloss"] == "привет"
    assert len(result[0]["frames"]) == 4  # T
    assert len(result[0]["frames"][0]) == 75  # J


def test_get_greedy_matches_longest_multiword_phrase_first(tmp_path: Path) -> None:
    _make_dataset(tmp_path, ["время", "время по гринвичу"], labels=[0, 1])
    provider = SkeletonSequenceProvider(processed_dir=str(tmp_path), split="val")

    result = provider.get("время по гринвичу")

    assert len(result) == 1
    assert result[0]["gloss"] == "время по гринвичу"


def test_get_skips_unmatched_tokens_without_failing(tmp_path: Path) -> None:
    _make_dataset(tmp_path, ["привет"], labels=[0])
    provider = SkeletonSequenceProvider(processed_dir=str(tmp_path), split="val")

    result = provider.get("привет несуществующееслово")

    assert len(result) == 1
    assert result[0]["gloss"] == "привет"


def test_get_case_and_punctuation_insensitive_match(tmp_path: Path) -> None:
    _make_dataset(tmp_path, ["привет!"], labels=[0])
    provider = SkeletonSequenceProvider(processed_dir=str(tmp_path), split="val")

    result = provider.get("ПРИВЕТ")

    assert len(result) == 1


def test_coordinates_remapped_into_unit_range(tmp_path: Path) -> None:
    _make_dataset(tmp_path, ["привет"], labels=[0])
    provider = SkeletonSequenceProvider(processed_dir=str(tmp_path), split="val")

    result = provider.get("привет")
    frames = np.array(result[0]["frames"])

    confident = frames[:, :, 2] > 0
    xy = frames[:, :, :2][confident]
    assert xy.min() >= 0.0
    assert xy.max() <= 1.0
    # 10% margin on both sides means the extreme points shouldn't sit
    # exactly on 0 or 1.
    assert xy.min() > 0.05
    assert xy.max() < 0.95


def test_zero_confidence_points_stay_at_origin(tmp_path: Path) -> None:
    """Points that were (0,0,0) -- the padding/missing-joint convention --
    must stay exactly (0,0,*) after remapping, so the client's existing
    "skip if x==0 and y==0" rendering logic keeps hiding them."""
    _make_dataset(tmp_path, ["привет"], labels=[0])
    provider = SkeletonSequenceProvider(processed_dir=str(tmp_path), split="val")

    result = provider.get("привет")
    frames = np.array(result[0]["frames"])

    # Indices 3.. were never set in _make_dataset -- still (0,0,0).
    assert np.all(frames[:, 3:, 0] == 0.0)
    assert np.all(frames[:, 3:, 1] == 0.0)


def test_get_returns_empty_list_for_no_matches(tmp_path: Path) -> None:
    _make_dataset(tmp_path, ["привет"], labels=[0])
    provider = SkeletonSequenceProvider(processed_dir=str(tmp_path), split="val")

    result = provider.get("несуществующееслово")

    assert result == []
