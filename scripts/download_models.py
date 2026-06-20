#!/usr/bin/env python3
"""
Download Glossa model files from Google Drive.

Usage:
    python scripts/download_models.py          # download all
    python scripts/download_models.py --list   # print file inventory
    python scripts/download_models.py gesture nlp   # download named groups only

Prerequisites:
    pip install gdown

Google Drive IDs must be set in the MODELS dict below (or via env vars).
For large files / shared drives use gdown --fuzzy or folder IDs.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ── Model registry ────────────────────────────────────────────────────────────
# Fill in the Drive file/folder IDs before distributing.
# Format: "GDRIVE_ID" for single files, "folder:GDRIVE_ID" for folders.
MODELS: dict[str, dict] = {
    "gesture": {
        "description": "ST-GCN gesture classifier (ONNX mobile + OpenVINO INT8)",
        "files": [
            {
                "name": "gesture_classifier_mobile.onnx",
                "dest": "models/gesture_classifier_mobile.onnx",
                "id": os.getenv("GDRIVE_ONNX_MOBILE_ID", ""),
            },
            {
                "name": "stgcn_topk_int8/ (OpenVINO folder)",
                "dest": "models/stgcn_topk_int8",
                "id": os.getenv("GDRIVE_OV_FOLDER_ID", ""),
                "folder": True,
            },
            {
                "name": "norm_stats.npz",
                "dest": "models/norm_stats.npz",
                "id": os.getenv("GDRIVE_NORM_STATS_ID", ""),
            },
        ],
    },
    "nlp": {
        "description": "Qwen2-1.5B merged fine-tune (≈ 3.2 GB)",
        "files": [
            {
                "name": "qwen2_merged_qwen2_1.5b/ (model folder)",
                "dest": "models/qwen2_merged_qwen2_1.5b",
                "id": os.getenv("GDRIVE_QWN2_FOLDER_ID", ""),
                "folder": True,
            },
        ],
    },
    "data": {
        "description": "idx_to_class.json class map for Slovo",
        "files": [
            {
                "name": "idx_to_class.json",
                "dest": "data/idx_to_class.json",
                "id": os.getenv("GDRIVE_CLASS_MAP_ID", ""),
            },
        ],
    },
}

ROOT = Path(__file__).parent.parent


def _gdown(file_id: str, dest: Path, *, folder: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["gdown", "--fuzzy"]
    if folder:
        cmd += ["--folder", f"https://drive.google.com/drive/folders/{file_id}", "-O", str(dest.parent)]
    else:
        cmd += [f"https://drive.google.com/uc?id={file_id}", "-O", str(dest)]
    print(f"  ↓  {dest}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  ✗  gdown failed for {dest.name} — check the Drive ID or permissions")


def download_group(name: str, group: dict) -> None:
    print(f"\n[{name}] {group['description']}")
    for f in group["files"]:
        file_id = f["id"]
        dest = ROOT / f["dest"]
        if not file_id:
            print(f"  ⚠  No Drive ID configured for {f['name']} — skipping")
            print(f"     Set env var or edit MODELS['{name}'] in this script")
            continue
        if dest.exists():
            print(f"  ✓  {dest.relative_to(ROOT)} already exists — skipping")
            continue
        _gdown(file_id, dest, folder=f.get("folder", False))


def list_models() -> None:
    print("Glossa model inventory\n" + "=" * 40)
    for group_name, group in MODELS.items():
        print(f"\n[{group_name}] {group['description']}")
        for f in group["files"]:
            dest = ROOT / f["dest"]
            status = "✓ present" if dest.exists() else "✗ missing"
            print(f"  {status:12s} {f['dest']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("groups", nargs="*", help="Groups to download (default: all)")
    parser.add_argument("--list", action="store_true", help="Print model status and exit")
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    try:
        subprocess.run(["gdown", "--version"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("gdown not found. Install with: pip install gdown")
        sys.exit(1)

    groups = args.groups or list(MODELS.keys())
    unknown = set(groups) - set(MODELS.keys())
    if unknown:
        print(f"Unknown groups: {unknown}. Available: {list(MODELS.keys())}")
        sys.exit(1)

    for name in groups:
        download_group(name, MODELS[name])

    print("\nDone. Run `make health` to verify services can find the models.")


if __name__ == "__main__":
    main()
