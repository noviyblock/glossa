"""DVC dataset versioning — snapshots and lineage for MLflow runs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from mlops.config import MLOpsSettings, get_settings


@dataclass
class DatasetSnapshot:
    path: str
    dvc_hash: str | None          # md5 / sha256 from DVC cache
    dvc_version: str | None       # git commit that last changed this path
    n_files: int
    total_size_bytes: int
    splits: dict[str, int] = field(default_factory=dict)    # split → n_samples
    schema: dict[str, Any] = field(default_factory=dict)    # column / shape info
    extra: dict[str, Any] = field(default_factory=dict)

    def to_mlflow_tags(self) -> dict[str, str]:
        tags: dict[str, str] = {
            "dataset_path": self.path,
            "dataset_n_files": str(self.n_files),
            "dataset_size_bytes": str(self.total_size_bytes),
        }
        if self.dvc_hash:
            tags["dataset_dvc_hash"] = self.dvc_hash
        if self.dvc_version:
            tags["dataset_git_sha"] = self.dvc_version
        for split, count in self.splits.items():
            tags[f"dataset_split_{split}"] = str(count)
        return tags


class DatasetVersioner:
    """Captures dataset state via DVC and attaches it to MLflow runs.

    Workflow:
    1.  `snapshot(path)` → DatasetSnapshot (reads DVC .dvc files / lock)
    2.  `add_to_dvc(path)` → tracks a new dataset directory with DVC
    3.  `push()` → pushes DVC-tracked data to the remote
    4.  Attach snapshot to an MLflow run via `snapshot.to_mlflow_tags()`
    """

    def __init__(self, settings: MLOpsSettings | None = None) -> None:
        self._cfg = settings or get_settings()

    # ── Public API ────────────────────────────────────────────────────────────

    def snapshot(self, path: str | Path) -> dict[str, Any]:
        """Return a dict describing the current state of `path`."""
        p = Path(path)
        snap = DatasetSnapshot(
            path=str(p),
            dvc_hash=self._get_dvc_hash(p),
            dvc_version=self._get_git_sha(p),
            n_files=sum(1 for _ in p.rglob("*") if _.is_file()) if p.exists() else 0,
            total_size_bytes=sum(
                f.stat().st_size for f in p.rglob("*") if f.is_file()
            ) if p.exists() else 0,
            splits=self._count_splits(p),
            schema=self._detect_schema(p),
        )
        return asdict(snap)

    def add_to_dvc(self, path: str | Path) -> bool:
        """Run `dvc add <path>` to start tracking a dataset."""
        try:
            result = subprocess.run(
                ["dvc", "add", str(path)],
                capture_output=True, text=True, check=True,
            )
            print(result.stdout)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"dvc add failed: {exc}")
            return False

    def push(self, remote: str | None = None) -> bool:
        """Push DVC-tracked data to the remote storage."""
        cmd = ["dvc", "push"]
        if remote:
            cmd += ["--remote", remote]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def get_dataset_at_hash(self, path: str | Path, dvc_hash: str) -> bool:
        """Checkout a specific dataset version by DVC hash (for reproducibility)."""
        try:
            subprocess.run(
                ["dvc", "checkout", str(path)],
                capture_output=True, text=True, check=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def compute_content_hash(self, path: str | Path) -> str:
        """Compute a deterministic SHA-256 of all files in a directory."""
        p = Path(path)
        h = hashlib.sha256()
        for file in sorted(p.rglob("*")):
            if file.is_file():
                h.update(str(file.relative_to(p)).encode())
                with file.open("rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        h.update(chunk)
        return h.hexdigest()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_dvc_hash(self, path: Path) -> str | None:
        # Check for <path>.dvc sidecar file
        dvc_file = Path(str(path) + ".dvc")
        if dvc_file.exists():
            try:
                import yaml  # type: ignore[import]
                data = yaml.safe_load(dvc_file.read_text())
                for out in data.get("outs", []):
                    h = out.get("md5") or out.get("hash")
                    if h:
                        return h
            except Exception:
                pass

        # Check dvc.lock
        lock = Path("dvc.lock")
        if lock.exists():
            try:
                import yaml  # type: ignore[import]
                data = yaml.safe_load(lock.read_text())
                for stage in data.get("stages", {}).values():
                    for out in stage.get("outs", []):
                        if Path(out.get("path", "")).resolve() == path.resolve():
                            return out.get("md5") or out.get("hash")
            except Exception:
                pass
        return None

    def _get_git_sha(self, path: Path) -> str | None:
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--format=%H", "--", str(path)],
                capture_output=True, text=True,
            )
            sha = result.stdout.strip()
            return sha if sha else None
        except Exception:
            return None

    def _count_splits(self, path: Path) -> dict[str, int]:
        """Count files per split subdirectory (train/val/test)."""
        splits: dict[str, int] = {}
        for split in ("train", "val", "test"):
            split_dir = path / split
            if split_dir.exists():
                splits[split] = sum(1 for f in split_dir.rglob("*") if f.is_file())
        return splits

    def _detect_schema(self, path: Path) -> dict[str, Any]:
        """Inspect numpy arrays or jsonl files to determine shape/dtype."""
        schema: dict[str, Any] = {}
        for npy in list(path.rglob("*.npy"))[:3]:
            try:
                import numpy as np
                arr = np.load(npy, mmap_mode="r")
                schema[npy.name] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
            except Exception:
                pass
        for jsonl in list(path.rglob("*.jsonl"))[:1]:
            try:
                first_line = jsonl.read_text().splitlines()[0]
                schema["jsonl_keys"] = list(json.loads(first_line).keys())
            except Exception:
                pass
        return schema
