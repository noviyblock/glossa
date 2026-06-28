"""Добавляет SCENARIO-переключатель в
notebooks/colab_glossa_03_export_inference_topk.ipynb и исправляет три
пути, ставшие неверными после scenario-aware патчей colab_glossa_02:

- ADAPTER_DIR  раньше: f'{MODELS_DIR}/qwen2_lora_qwen2_1.5b_slovo_synth'
                теперь: f'{MODELS_DIR}/qwen2_lora_{SCENARIO}_qwen2_1.5b_slovo_synth'
- VAL_CSV      раньше: f'{DATA_DIR}/rsl_gloss_val_slovo_synth.csv'
                теперь: f'{DATA_DIR}/rsl_gloss_val_slovo_synth_{SCENARIO}.csv'
- NUM_CLASSES  раньше: хардкод 1000
                теперь: 200 для top200, 1000 для full
- WINDOW_SIZE  раньше: хардкод 32 (несовместимо с T=64 моделью из colab_01a)
                теперь: 64

CLASS_MAP_PATH/MERGED_DIR/ONNX_*_PATH/NORM_STATS_PATH/OV_DIR оставлены без
суффикса сценария — они совпадают с путями, которые читает docker-compose
(cv-service/nlp-service), а production одновременно обслуживает только один
сценарий.

Запуск: python scripts/patch_colab_03_scenario.py
"""

from __future__ import annotations

import nbformat

NB_PATH = "notebooks/colab_glossa_03_export_inference_topk.ipynb"


def patch(nb: nbformat.NotebookNode) -> None:
    cells = nb.cells

    # ── Вставка ячейки SCENARIO перед установкой зависимостей ───────────────
    scenario_md = nbformat.v4.new_markdown_cell(
        "## Ячейка 0.5 — Переключатель сценария (top200 / full corpus)"
    )
    scenario_code = nbformat.v4.new_code_cell(
        """\
# Переключатель сценария — должен совпадать со SCENARIO, с которым были
# обучены ST-GCN (colab_01a) и Qwen2-LoRA (colab_02), которые этот ноутбук
# экспортирует в production-артефакты.
SCENARIO = 'top200'  # 'top200' | 'full'

assert SCENARIO in ('top200', 'full'), f'Неизвестный сценарий: {SCENARIO}'
print(f'Сценарий: {SCENARIO}')
"""
    )
    cells.insert(0, scenario_code)
    cells.insert(0, scenario_md)

    # ── Ячейка путей (теперь индекс смещён на +2) ───────────────────────────
    c = cells[6]
    assert "ADAPTER_DIR         = f'{MODELS_DIR}/qwen2_lora_qwen2_1.5b_slovo_synth'" in c.source

    c.source = (
        c.source
        .replace(
            "ADAPTER_DIR         = f'{MODELS_DIR}/qwen2_lora_qwen2_1.5b_slovo_synth'  # ← из colab_02",
            "ADAPTER_DIR         = f'{MODELS_DIR}/qwen2_lora_{SCENARIO}_qwen2_1.5b_slovo_synth'  # ← из colab_02",
        )
        .replace(
            "VAL_CSV             = f'{DATA_DIR}/rsl_gloss_val_slovo_synth.csv'          # ← из colab_02",
            "VAL_CSV             = f'{DATA_DIR}/rsl_gloss_val_slovo_synth_{SCENARIO}.csv'  # ← из colab_02",
        )
        .replace(
            "WINDOW_SIZE         = 32",
            "WINDOW_SIZE         = 64  # T=64, см. colab_01a",
        )
        .replace(
            "NUM_CLASSES         = 1000",
            "NUM_CLASSES         = 200 if SCENARIO == 'top200' else 1000",
        )
    )
    c.outputs = []
    c.execution_count = None

    # ── Ячейка merge_and_unload (теперь индекс +2): ADAPTER_DIR с суффиксом ─
    c = cells[15]
    assert "ADAPTER_DIR = f'{MODELS_DIR}/qwen2_lora_qwen2_1.5b_slovo_synth'" in c.source
    c.source = c.source.replace(
        "ADAPTER_DIR = f'{MODELS_DIR}/qwen2_lora_qwen2_1.5b_slovo_synth'",
        "ADAPTER_DIR = f'{MODELS_DIR}/qwen2_lora_{SCENARIO}_qwen2_1.5b_slovo_synth'",
    )
    c.outputs = []
    c.execution_count = None


def main() -> None:
    nb = nbformat.read(NB_PATH, as_version=4)
    nb.nbformat_minor = 5
    patch(nb)
    nbformat.validator.normalize(nb)
    nbformat.validate(nb)
    nbformat.write(nb, NB_PATH)
    print(f"Patched: {NB_PATH} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
