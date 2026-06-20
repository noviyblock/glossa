"""Slovo RSL dataset loader (SberDevices, 2022).

Dataset: https://github.com/hukenovs/slovo
Kaggle:  kaggle datasets download -d kapitanov/slovo

Expected layout after unzip:
  data/slovo/
    annotations.csv
    train/  <class_name>/<video_id>.mp4
    test/   <class_name>/<video_id>.mp4
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class SlovoSample:
    video_path: Path
    label: int
    label_name: str
    start_frame: int
    end_frame: int
    split: str  # "train" | "test"


class SlovoDataset:
    """Iterable loader for the Slovo RSL gesture dataset.

    Handles the actual Kaggle CSV schema:
        attachment_id, text, user_id, height, width, length, train, begin, end
    where `train` is a bool (True = train split, False = val/test).
    Videos are searched in: <root>/videos/, <root>/train/, <root>/test/

    Pass label_mapping_path pointing to the label_mapping.json produced by
    experiment 00 to guarantee that class IDs are identical across all stages
    (EDA → preprocessing → training → evaluation).
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "all",
        exclude_no_event: bool = True,
        label_mapping_path: str | Path | None = None,
    ):
        self.root = Path(root)
        self.split = split
        self.exclude_no_event = exclude_no_event
        self._label_mapping_path = Path(label_mapping_path) if label_mapping_path else None
        self._samples: list[SlovoSample] = []
        self._label_map: dict[str, int] = {}
        self._load()

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_split(row: dict) -> str:
        """Derive split string from either 'split' col or 'train' bool col."""
        if "split" in row and row["split"]:
            return row["split"]
        train_val = row.get("train", "True")
        if isinstance(train_val, bool):
            return "train" if train_val else "val"
        return "train" if str(train_val).lower() in ("true", "1", "yes") else "val"

    def _find_video(self, row: dict, split: str) -> Path:
        """Try multiple path conventions used by Slovo on Kaggle and locally.

        The CSV encodes non-train samples as split="val", but on Kaggle the
        physical directory is named "test/", so we try both names.
        """
        aid = row["attachment_id"]
        text = row["text"]
        alt = "test" if split == "val" else split
        candidates = [
            self.root / "videos" / f"{aid}.mp4",
            self.root / split / text / f"{aid}.mp4",
            self.root / alt   / text / f"{aid}.mp4",
            self.root / split / f"{aid}.mp4",
            self.root / alt   / f"{aid}.mp4",
            self.root / text  / f"{aid}.mp4",
            self.root / f"{aid}.mp4",
        ]
        for p in candidates:
            if p.exists():
                return p
        return candidates[0]  # best guess even if missing

    def _load(self) -> None:
        import json as _json

        ann = self._find_annotations()
        classes: list[str] = []
        raw: list[dict] = []

        with open(ann, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row["text"]
                if self.exclude_no_event and text == "no_event":
                    continue
                s = self._parse_split(row)
                if self.split != "all" and s != self.split:
                    continue
                if text not in classes:
                    classes.append(text)
                raw.append({**row, "_split": s})

        # Use external mapping (from exp 00 EDA) when available so that
        # class IDs are consistent across preprocessing, training and evaluation.
        if self._label_mapping_path and self._label_mapping_path.exists():
            with open(self._label_mapping_path, encoding="utf-8") as f:
                mapping = _json.load(f)
            self._label_map = {str(k): int(v) for k, v in mapping["label_to_id"].items()}
        else:
            classes.sort()
            self._label_map = {name: i for i, name in enumerate(classes)}

        for row in raw:
            s = row["_split"]
            text = row["text"]
            if text not in self._label_map:
                continue  # class absent from external mapping — skip
            self._samples.append(SlovoSample(
                video_path=self._find_video(row, s),
                label=self._label_map[text],
                label_name=text,
                start_frame=int(row.get("begin", 0) or 0),
                end_frame=int(row.get("end", -1) or -1),
                split=s,
            ))

    def _find_annotations(self) -> Path:
        for candidate in [
            self.root / "annotations.csv",
            self.root / "slovo" / "annotations.csv",
            self.root / "SLOVO" / "annotations.csv",
        ]:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"annotations.csv not found under {self.root}. "
            "Run: kaggle datasets download -d kapitanov/slovo"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> SlovoSample:
        return self._samples[idx]

    def __iter__(self) -> Iterator[SlovoSample]:
        return iter(self._samples)

    @property
    def num_classes(self) -> int:
        return len(self._label_map)

    @property
    def label_map(self) -> dict[str, int]:
        return self._label_map

    @property
    def inv_label_map(self) -> dict[int, str]:
        return {v: k for k, v in self._label_map.items()}

    @property
    def samples(self) -> list[SlovoSample]:
        return self._samples

    def by_split(self, split: str) -> list[SlovoSample]:
        return [s for s in self._samples if s.split == split]

    # ── Frame extraction ──────────────────────────────────────────────────────

    def load_frames(
        self,
        sample: SlovoSample,
        target_len: int = 30,
        size: tuple[int, int] = (224, 224),
        normalize: bool = True,
    ) -> np.ndarray:
        """Load uniformly-sampled RGB frames → float32 (T, H, W, 3)."""
        try:
            import cv2
        except ImportError as e:
            raise ImportError("pip install opencv-python") from e

        cap = cv2.VideoCapture(str(sample.video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        start = max(0, sample.start_frame)
        end = sample.end_frame if sample.end_frame > 0 else total

        indices = np.linspace(start, max(start, end - 1), target_len, dtype=int)
        frames: list[np.ndarray] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok:
                frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), size)
                frames.append(frame)
        cap.release()

        if not frames:
            return np.zeros((target_len, *size, 3), dtype=np.float32)

        arr = np.stack(frames, axis=0).astype(np.float32)
        if len(arr) < target_len:
            pad = np.zeros((target_len - len(arr), *size, 3), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)

        if normalize:
            arr /= 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            arr = (arr - mean) / std

        return arr  # (T, H, W, 3)

    def load_keypoints_mediapipe(
        self,
        sample: SlovoSample,
        target_len: int = 64,
    ) -> np.ndarray:
        """Extract MediaPipe Holistic keypoints → float32 (T, 75, 3).

        Returns concatenated pose (33) + left_hand (21) + right_hand (21) landmarks.
        """
        try:
            import cv2
            import mediapipe as mp  # type: ignore[import]
        except ImportError as e:
            raise ImportError("pip install mediapipe opencv-python") from e

        holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        cap = cv2.VideoCapture(str(sample.video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        start = max(0, sample.start_frame)
        end = sample.end_frame if sample.end_frame > 0 else total

        indices = np.linspace(start, max(start, end - 1), target_len, dtype=int)
        sequences: list[np.ndarray] = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                sequences.append(np.zeros((75, 3), dtype=np.float32))
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(rgb)

            def _lm(landmarks, n: int) -> np.ndarray:
                if landmarks is None:
                    return np.zeros((n, 3), dtype=np.float32)
                return np.array([[l.x, l.y, l.z] for l in landmarks.landmark], dtype=np.float32)

            pose  = _lm(result.pose_landmarks, 33)
            lhand = _lm(result.left_hand_landmarks, 21)
            rhand = _lm(result.right_hand_landmarks, 21)
            sequences.append(np.concatenate([pose, lhand, rhand], axis=0))  # (75, 3)

        cap.release()
        holistic.close()

        if not sequences:
            return np.zeros((target_len, 75, 3), dtype=np.float32)

        arr = np.stack(sequences, axis=0)  # (T, 75, 3)
        if len(arr) < target_len:
            pad = np.zeros((target_len - len(arr), 75, 3), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        return arr


def horizontal_flip_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """Flip skeleton keypoints horizontally (swap left/right hands).

    Args:
        keypoints: (T, 75, 3) array — pose(33) + lhand(21) + rhand(21)

    Returns:
        Flipped array (T, 75, 3) with x-coordinate mirrored and hands swapped.
    """
    kp = keypoints.copy()
    kp[:, :, 0] = 1.0 - kp[:, :, 0]  # mirror x

    # Swap left (33:54) and right (54:75) hand blocks
    kp[:, 33:54, :], kp[:, 54:75, :] = (
        keypoints[:, 54:75, :].copy(),
        keypoints[:, 33:54, :].copy(),
    )
    return kp
