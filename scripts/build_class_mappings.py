"""Регенерирует data/class_to_idx.json и data/idx_to_class.json под топ-200
классов — заменяет старый маппинг на 1000 классов (полный корпус Slovo).

Порядок индексов ДОЛЖЕН совпадать с тем, что видела модель во время обучения.
В colab_glossa_00_preprocess_dwpose_200.ipynb (ячейка 4 "Label mapping")
индексы строятся так:
    unique_labels = sorted(df['text'].unique())   # после фильтра топ-200
    label_to_id   = {label: idx for idx, label in enumerate(unique_labels)}

data/selected_classes_200.json уже хранит ровно этот отфильтрованный список
классов, и он уже отсортирован (alphabetical_first_200) — поэтому
sorted(classes) воспроизводит идентичный порядок без доступа к датасету.

Форматы (под потребителей):
  data/class_to_idx.json  — {label: idx}        (используется colab_glossa_03)
  data/idx_to_class.json  — {str(idx): label}    (используется cv_service,
                                                    GestureClassifier._load_class_map)

Запуск: python scripts/build_class_mappings.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
SELECTED_PATH = ROOT / "data" / "selected_classes_200.json"
CLASS_TO_IDX_OUT = ROOT / "data" / "class_to_idx.json"
IDX_TO_CLASS_OUT = ROOT / "data" / "idx_to_class.json"


def main() -> None:
    if not SELECTED_PATH.exists():
        raise FileNotFoundError(f"Не найден {SELECTED_PATH}")

    selected = json.loads(SELECTED_PATH.read_text(encoding="utf-8"))
    classes = selected["classes"]
    unique_labels = sorted(classes)
    assert unique_labels == classes, (
        "selected_classes_200.json не отсортирован — порядок индексов разойдётся "
        "с тем, что видела модель при обучении (sorted(df['text'].unique()))."
    )

    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {str(idx): label for label, idx in label_to_id.items()}

    CLASS_TO_IDX_OUT.write_text(
        json.dumps(label_to_id, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    IDX_TO_CLASS_OUT.write_text(
        json.dumps(idx_to_label, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Классов: {len(label_to_id)}")
    print(f"Сохранено: {CLASS_TO_IDX_OUT}")
    print(f"Сохранено: {IDX_TO_CLASS_OUT}")
    print("\nПримеры:")
    for label, idx in list(label_to_id.items())[:5]:
        print(f"  {idx:3d}  {label!r}")


if __name__ == "__main__":
    main()
