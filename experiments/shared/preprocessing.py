"""Gesture keypoint preprocessing transforms for the Slovo RSL dataset.

All transforms operate on numpy arrays of shape (T, J, C):
    T = time steps (frames)
    J = joints  — 75 for DWPose (COCO-WholeBody → 75-point remap)
    C = coordinates — 3 (x, y, score), normalised to [0, 1]

DWPose joint layout (see services/cv_service/keypoint_extractor.py):
    [0:17]  — COCO body (17 joints)
    [17:23] — COCO feet (6 joints)
    [23:33] — duplicated key joints (shoulders/hips/knees/ankles), padding to 33
    [33:54] — left hand (21 joints)
    [54:75] — right hand (21 joints)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

POSE_SLICE  = slice(0, 33)
LHAND_SLICE = slice(33, 54)
RHAND_SLICE = slice(54, 75)

# Upper-body joints used in RSL (shoulders=5,6 elbows=7,8 wrists=9,10)
SIGNING_JOINTS = list(range(5, 11)) + list(range(33, 75))


@dataclass
class PreprocessConfig:
    target_len: int = 30
    normalize: bool = True
    impute_missing: bool = True
    missing_threshold: float = 0.02  # coords near 0 → likely missing landmark
    length_mode: str = "interpolate"  # "interpolate" | "pad_crop"
    # Augmentation (train only)
    flip_prob: float = 0.5
    scale_range: tuple[float, float] = (0.85, 1.15)
    noise_std: float = 0.01
    time_warp: bool = True


# ── Missing data imputation ───────────────────────────────────────────────────

def impute_missing(kp: np.ndarray, threshold: float = 0.02) -> np.ndarray:
    """Linear-interpolate joints that are near-zero across all axes (not tracked).

    DWPose sets undetected landmarks to (0, 0, 0). We detect these frames
    per joint and replace them via linear interpolation from valid neighbours.
    If the joint is never visible, leave as zeros.
    """
    kp = kp.copy()
    T, J, C = kp.shape
    t = np.arange(T, dtype=float)
    for j in range(J):
        missing = np.all(np.abs(kp[:, j, :]) < threshold, axis=-1)
        if not np.any(missing) or np.all(missing):
            continue
        valid = ~missing
        for c in range(C):
            kp[missing, j, c] = np.interp(t[missing], t[valid], kp[valid, j, c])
    return kp


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalize_keypoints(kp: np.ndarray) -> np.ndarray:
    """Pose-anchored scale-invariant normalisation.

    Steps:
      1. Translate so that the mid-hip point is at the origin.
      2. Scale by mean shoulder width (makes the representation resolution-
         and distance-invariant — crucial because signers appear at varying
         distances from the camera).
      3. Z-score the depth (z) channel across the whole sequence.
    """
    kp = kp.copy()

    # 1. Hip-centre translation (joints 11=left hip, 12=right hip in COCO)
    if kp.shape[1] > 12:
        hip_mid = (kp[:, 11, :2] + kp[:, 12, :2]) / 2.0  # (T, 2)
        kp[:, :, :2] -= hip_mid[:, np.newaxis, :]

    # 2. Shoulder-width scaling (joints 5=left shoulder, 6=right shoulder)
    if kp.shape[1] > 6:
        sh_width = np.mean(
            np.linalg.norm(kp[:, 5, :2] - kp[:, 6, :2], axis=-1)
        )
        if sh_width > 1e-6:
            kp[:, :, :2] /= sh_width

    # 3. Z-channel standardisation
    z = kp[:, :, 2]
    std = z.std()
    kp[:, :, 2] = (z - z.mean()) / (std + 1e-6)

    return kp.astype(np.float32)


# ── Length standardisation ────────────────────────────────────────────────────

def standardize_length(
    kp: np.ndarray,
    target_len: int,
    mode: str = "interpolate",
) -> np.ndarray:
    """Resize (T, J, C) → (target_len, J, C).

    interpolate: linear resampling — preserves motion dynamics.
    pad_crop:    centre-crop if too long, zero-pad if too short.
    """
    T = kp.shape[0]
    if T == target_len:
        return kp

    if mode == "interpolate":
        t_old = np.linspace(0.0, 1.0, T)
        t_new = np.linspace(0.0, 1.0, target_len)
        J, C = kp.shape[1], kp.shape[2]
        out = np.empty((target_len, J, C), dtype=kp.dtype)
        for j in range(J):
            for c in range(C):
                out[:, j, c] = np.interp(t_new, t_old, kp[:, j, c])
        return out

    if mode == "pad_crop":
        if T >= target_len:
            start = (T - target_len) // 2
            return kp[start : start + target_len]
        pad = np.zeros((target_len - T, *kp.shape[1:]), dtype=kp.dtype)
        return np.concatenate([kp, pad], axis=0)

    raise ValueError(f"Unknown length mode: {mode!r}")


# ── Augmentation ──────────────────────────────────────────────────────────────

def augment(
    kp: np.ndarray,
    cfg: PreprocessConfig,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply stochastic augmentations for training robustness.

    Horizontal flip: mirrors x, swaps left/right hand blocks.
    Scale jitter:    random uniform scale of the xy plane.
    Gaussian noise:  small per-coordinate perturbation.
    Time warp:       non-uniform temporal resampling (simulates speed variation).
    """
    rng = rng or np.random.default_rng()
    kp = kp.copy()

    if rng.random() < cfg.flip_prob:
        kp[:, :, 0] = -kp[:, :, 0]  # mirror x
        kp[:, LHAND_SLICE, :], kp[:, RHAND_SLICE, :] = (
            kp[:, RHAND_SLICE, :].copy(),
            kp[:, LHAND_SLICE, :].copy(),
        )

    scale = rng.uniform(*cfg.scale_range)
    kp[:, :, :2] *= scale

    if cfg.noise_std > 0:
        kp += rng.normal(0.0, cfg.noise_std, kp.shape).astype(np.float32)

    if cfg.time_warp:
        T = kp.shape[0]
        speed = rng.uniform(0.8, 1.2, T)
        cum = np.cumsum(speed)
        cum = (cum - cum[0]) / (cum[-1] - cum[0] + 1e-9)
        t_new = np.linspace(0.0, 1.0, T)
        J, C = kp.shape[1], kp.shape[2]
        out = np.empty_like(kp)
        for j in range(J):
            for c in range(C):
                out[:, j, c] = np.interp(t_new, cum, kp[:, j, c])
        kp = out

    return kp.astype(np.float32)


# ── Full pipeline ─────────────────────────────────────────────────────────────

def preprocess_sample(
    kp: np.ndarray,
    cfg: PreprocessConfig,
    training: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply the complete preprocessing pipeline to one (T, J, C) array."""
    if cfg.impute_missing:
        kp = impute_missing(kp, cfg.missing_threshold)
    kp = standardize_length(kp, cfg.target_len, cfg.length_mode)
    if cfg.normalize:
        kp = normalize_keypoints(kp)
    if training:
        kp = augment(kp, cfg, rng)
    return kp.astype(np.float32)


# ── Stratified split ──────────────────────────────────────────────────────────

class StratifiedSplitter:
    """Split a labelled dataset into train / val / test preserving class ratios.

    For classes with < 3 samples all go to train (avoids empty splits on
    long-tail classes — common in Slovo's 1000-class distribution).
    """

    def __init__(
        self,
        train: float = 0.80,
        val: float = 0.10,
        seed: int = 42,
    ) -> None:
        assert 0 < train < 1 and 0 < val < 1 and train + val < 1
        self.train = train
        self.val = val
        self.test = round(1.0 - train - val, 10)
        self._rng = np.random.default_rng(seed)

    def split(
        self, labels: list[int]
    ) -> tuple[list[int], list[int], list[int]]:
        """Return (train_idx, val_idx, test_idx)."""
        from collections import defaultdict

        buckets: dict[int, list[int]] = defaultdict(list)
        for i, lbl in enumerate(labels):
            buckets[lbl].append(i)

        train_idx, val_idx, test_idx = [], [], []
        for idxs in buckets.values():
            perm = list(self._rng.permutation(idxs))
            n = len(perm)
            if n < 3:
                train_idx.extend(perm)
                continue
            n_tr = max(1, int(n * self.train))
            n_va = max(1, int(n * self.val))
            train_idx.extend(perm[:n_tr])
            val_idx.extend(perm[n_tr : n_tr + n_va])
            test_idx.extend(perm[n_tr + n_va :])

        return sorted(train_idx), sorted(val_idx), sorted(test_idx)
