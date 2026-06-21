#!/usr/bin/env python3
"""Generate class_to_idx.json and idx_to_class.json from slovo_glosses.txt."""

import argparse
import json
import sys
from pathlib import Path

# Force UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def generate_class_map(input_path: Path, output_dir: Path) -> int:
    glosses = [
        line.strip()
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    glosses_sorted = sorted(glosses)

    class_to_idx = {gloss: idx for idx, gloss in enumerate(glosses_sorted)}
    idx_to_class = {str(idx): gloss for idx, gloss in enumerate(glosses_sorted)}

    output_dir.mkdir(parents=True, exist_ok=True)

    out_c2i = output_dir / "class_to_idx.json"
    out_i2c = output_dir / "idx_to_class.json"

    out_c2i.write_text(json.dumps(class_to_idx, ensure_ascii=False, indent=2), encoding="utf-8")
    out_i2c.write_text(json.dumps(idx_to_class, ensure_ascii=False, indent=2), encoding="utf-8")

    n = len(glosses_sorted)
    print(f"Создан словарь: {n} классов → {out_c2i}")
    print(f"Создан обратный словарь: {n} классов → {out_i2c}")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/slovo_glosses.txt",
        help="Путь к файлу со списком глосс (одна на строку)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/",
        help="Директория для сохранения JSON-файлов",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Файл не найден: {input_path}")

    generate_class_map(input_path, Path(args.output_dir))


if __name__ == "__main__":
    main()
