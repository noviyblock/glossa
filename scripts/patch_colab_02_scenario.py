"""Добавляет SCENARIO-переключатель (top200 / full) в
notebooks/colab_glossa_02_finetune_qwen2_taiga_gemini.ipynb, как в
colab_glossa_01a_train_stgcn_64_200.ipynb.

Проблема: ноутбук строит датасет gloss->text из ПОЛНОГО словаря Slovo
(~1000 классов, без фильтра) и из захардкоженных доменных словарей, где
только ~13% слов (29 из 229) пересекаются с топ-200 классами, которые
реально умеет распознавать/показывать cv-service в сценарии top200.
В результате NLP-модель учится переводить (и для text->RSL — генерировать)
глоссы, которых cv-service никогда не произведёт.

Правки:
- Ячейка SCENARIO (новая, после установки зависимостей): 'top200' | 'full',
  определяет путь к selected_classes_200.json и суффиксы путей/имён ранов.
- Ячейка 2 (Drive/Secrets): ADAPTER_DIR получает суффикс _{SCENARIO}.
- Ячейка 3 (CFG): experiment_name включает SCENARIO.
- Ячейка 4 (словари глосс): SLOVO_GLOSSES и DOMAIN_GLOSSES фильтруются по
  пересечению с selected_classes_200.json для top200; для доменов с <5
  словами после фильтрации — fallback на общий пул (с явным предупреждением,
  без тихого падения).
- Ячейка 6 (извлечение словаря из Slovo annotations.csv): тот же фильтр
  применяется к all_glosses.
- Ячейка 7, 9 (кэши/CSV датасета): пути получают суффикс _{SCENARIO}.
- Ячейки 12, 13, 14 (LoRA/обучение/оценка): MODEL_PATH получает суффикс
  _{SCENARIO}, чтобы адаптеры top200 и full не перезатирали друг друга.

Запуск: python scripts/patch_colab_02_scenario.py
"""

from __future__ import annotations

import nbformat

NB_PATH = "notebooks/colab_glossa_02_finetune_qwen2_taiga_gemini.ipynb"


def patch(nb: nbformat.NotebookNode) -> None:
    cells = nb.cells

    # ── Вставка ячеек SCENARIO после ячейки 1 (установка зависимостей) ──────
    scenario_md = nbformat.v4.new_markdown_cell(
        "## Ячейка 1.5 — Переключатель сценария (top200 / full corpus)"
    )
    scenario_code = nbformat.v4.new_code_cell(
        """\
# Переключатель сценария — два параллельных эксперимента (как в
# colab_glossa_01a_train_stgcn_64_200.ipynb):
#   'top200' — словарь глосс ограничен топ-200 классами, которые реально
#              умеет распознавать/показывать cv-service. Без этого
#              ограничения LLM учится на глоссах, которых cv-service
#              никогда не произведёт (и для text->RSL может их сгенерировать).
#   'full'   — полный словарь Slovo (~1000 классов), без фильтра.
SCENARIO = 'top200'  # 'top200' | 'full'

SELECTED_CLASSES_PATH = '/content/drive/MyDrive/glossa/data/selected_classes_200.json'

assert SCENARIO in ('top200', 'full'), f'Неизвестный сценарий: {SCENARIO}'
print(f'Сценарий: {SCENARIO}')
"""
    )
    cells.insert(2, scenario_code)
    cells.insert(2, scenario_md)

    # ── Ячейка 2 (теперь индекс 4): Drive/Secrets — ADAPTER_DIR с суффиксом ─
    c = cells[4]
    assert "ADAPTER_DIR" in c.source
    c.source = c.source.replace(
        "ADAPTER_DIR         = f'{DRIVE_ROOT}/models/qwen2_lora'",
        "ADAPTER_DIR         = f'{DRIVE_ROOT}/models/qwen2_lora_{SCENARIO}'",
    )
    c.outputs = []
    c.execution_count = None

    # ── Ячейка 3 (теперь 5): CFG — experiment_name включает SCENARIO ────────
    c = cells[5]
    assert "experiment_name" in c.source
    c.source = c.source.replace(
        "'experiment_name':     '05_nlp_llm_size',",
        "'experiment_name':     f'05_nlp_llm_size_{SCENARIO}',",
    )
    c.outputs = []
    c.execution_count = None

    # ── Ячейка 4 (теперь 6): словари глосс — фильтр по топ-200 ──────────────
    c = cells[6]
    assert "SLOVO_GLOSSES = [" in c.source
    filter_block = """\

# ── Фильтр словаря под SCENARIO ───────────────────────────────────────────────
# top200: оставляем только глоссы из selected_classes_200.json — это всё,
# что cv-service в этом сценарии способен распознать/показать. Без фильтра
# модель тренируется на глоссах, которых cv-service никогда не произведёт.
if SCENARIO == 'top200':
    from pathlib import Path as _Path
    if not _Path(SELECTED_CLASSES_PATH).exists():
        raise FileNotFoundError(
            f'Не найден {SELECTED_CLASSES_PATH} — нужен для фильтрации словаря '
            'в сценарии top200. Запустите colab_glossa_00a_select_classes.ipynb.'
        )
    with open(SELECTED_CLASSES_PATH, encoding='utf-8') as f:
        _selected = json.load(f)
    TOP200_SET = {w.lower() for w in _selected['classes']}

    _before = len(SLOVO_GLOSSES)
    SLOVO_GLOSSES = [g for g in SLOVO_GLOSSES if g.lower() in TOP200_SET]
    print(f'[top200 filter] SLOVO_GLOSSES: {_before} -> {len(SLOVO_GLOSSES)}')

    _MIN_DOMAIN_WORDS = 5
    _global_pool = list(TOP200_SET)
    for domain, words in list(DOMAIN_GLOSSES.items()):
        filtered = [w for w in words if w.lower() in TOP200_SET]
        if len(filtered) < _MIN_DOMAIN_WORDS:
            print(
                f'[top200 filter] Домен {domain!r}: {len(filtered)}/{len(words)} слов '
                f'после фильтра (< {_MIN_DOMAIN_WORDS}) — добираем из общего пула топ-200.'
            )
            needed = _MIN_DOMAIN_WORDS * 3 - len(filtered)
            extra = [w for w in _global_pool if w not in filtered][:max(needed, 0)]
            filtered = filtered + extra
        else:
            print(f'[top200 filter] Домен {domain!r}: {len(filtered)}/{len(words)} слов после фильтра.')
        DOMAIN_GLOSSES[domain] = filtered
else:
    TOP200_SET = None
    print('[full] Фильтр словаря не применяется — используется полный словарь Slovo.')
"""
    c.source = c.source.rstrip("\n") + "\n" + filter_block
    c.outputs = []
    c.execution_count = None

    # ── Ячейка 6 (теперь 8): извлечение all_glosses из annotations.csv ──────
    c = cells[8]
    assert "all_glosses = [g for g in all_glosses if len(g) <= 20]" in c.source
    c.source = c.source.replace(
        "all_glosses = [g for g in all_glosses if len(g) <= 20]\n",
        "all_glosses = [g for g in all_glosses if len(g) <= 20]\n\n"
        "if SCENARIO == 'top200':\n"
        "    _before = len(all_glosses)\n"
        "    all_glosses = [g for g in all_glosses if g.lower() in TOP200_SET]\n"
        "    print(f'[top200 filter] all_glosses: {_before} -> {len(all_glosses)}')\n",
    )
    c.outputs = []
    c.execution_count = None

    # ── Ячейка 7 (теперь 9): кэш переводов Slovo — суффикс сценария ─────────
    c = cells[9]
    assert "SLOVO_PAIRS_CACHE = f'{DATA_DIR}/slovo_pairs_cache.json'" in c.source
    c.source = c.source.replace(
        "SLOVO_PAIRS_CACHE = f'{DATA_DIR}/slovo_pairs_cache.json'",
        "SLOVO_PAIRS_CACHE = f'{DATA_DIR}/slovo_pairs_cache_{SCENARIO}.json'",
    )
    c.source = c.source.replace(
        "pd.DataFrame(slovo_pairs).to_csv(f'{DATA_DIR}/slovo_pairs_generated.csv', index=False)",
        "pd.DataFrame(slovo_pairs).to_csv(f'{DATA_DIR}/slovo_pairs_generated_{SCENARIO}.csv', index=False)",
    )
    c.outputs = []
    c.execution_count = None

    # ── Ячейка 8 (теперь 10): синтетика по доменам — кэш с суффиксом ────────
    c = cells[10]
    assert "SYNTHETIC_CACHE = f'{DATA_DIR}/synthetic_pairs_cache.json'" in c.source
    c.source = c.source.replace(
        "SYNTHETIC_CACHE = f'{DATA_DIR}/synthetic_pairs_cache.json'",
        "SYNTHETIC_CACHE = f'{DATA_DIR}/synthetic_pairs_cache_{SCENARIO}.json'",
    )
    c.outputs = []
    c.execution_count = None

    # ── Ячейка 9 (теперь 11): объединённый датасет — пути с суффиксом ───────
    c = cells[11]
    assert "DATASET_CSV = f'{DATA_DIR}/rsl_gloss_dataset_slovo_synth.csv'" in c.source
    c.source = (
        c.source
        .replace(
            "DATASET_CSV = f'{DATA_DIR}/rsl_gloss_dataset_slovo_synth.csv'",
            "DATASET_CSV = f'{DATA_DIR}/rsl_gloss_dataset_slovo_synth_{SCENARIO}.csv'",
        )
        .replace(
            "TRAIN_CSV = f'{DATA_DIR}/rsl_gloss_train_slovo_synth.csv'",
            "TRAIN_CSV = f'{DATA_DIR}/rsl_gloss_train_slovo_synth_{SCENARIO}.csv'",
        )
        .replace(
            "VAL_CSV = f'{DATA_DIR}/rsl_gloss_val_slovo_synth.csv'",
            "VAL_CSV = f'{DATA_DIR}/rsl_gloss_val_slovo_synth_{SCENARIO}.csv'",
        )
    )
    c.outputs = []
    c.execution_count = None

    # ── Ячейка 12 (теперь 14): LoRA/SFTTrainer — output_dir с суффиксом ─────
    c = cells[14]
    assert "output_dir=ADAPTER_DIR + '_qwen2_1.5b_slovo_synth'," in c.source
    c.source = c.source.replace(
        "output_dir=ADAPTER_DIR + '_qwen2_1.5b_slovo_synth',",
        "output_dir=ADAPTER_DIR + '_qwen2_1.5b_slovo_synth',  # ADAPTER_DIR уже содержит _{SCENARIO}",
    )
    c.outputs = []
    c.execution_count = None

    # ── Ячейка 13 (теперь 15): обучение — MODEL_PATH/банеры со сценарием ────
    c = cells[15]
    assert "MODEL_PATH = f'{ADAPTER_DIR}_qwen2_1.5b_slovo_synth'" in c.source
    c.source = c.source.replace(
        "print('ОБУЧЕНИЕ Qwen2-1.5B (Slovo + Синтетика)')",
        "print(f'ОБУЧЕНИЕ Qwen2-1.5B (Slovo + Синтетика, сценарий: {SCENARIO})')",
    )
    c.source = c.source.replace(
        "'dataset_source': 'slovo_30_synthetic_70',",
        "'dataset_source': f'slovo_30_synthetic_70_{SCENARIO}',",
    )
    c.outputs = []
    c.execution_count = None

    # ── Ячейка 14 (теперь 16): оценка — баннер со сценарием ─────────────────
    c = cells[16]
    assert "MODEL_PATH = f'{ADAPTER_DIR}_qwen2_1.5b_slovo_synth'" in c.source
    c.source = c.source.replace(
        "print('ОЦЕНКА КАЧЕСТВА МОДЕЛИ Qwen2-1.5B (Slovo + Синтетика)')",
        "print(f'ОЦЕНКА КАЧЕСТВА МОДЕЛИ Qwen2-1.5B (Slovo + Синтетика, сценарий: {SCENARIO})')",
    )
    c.outputs = []
    c.execution_count = None


def main() -> None:
    nb = nbformat.read(NB_PATH, as_version=4)
    nb.nbformat_minor = 5  # allow cell 'id' field (new cells get one from nbformat.v4.new_*_cell)
    patch(nb)
    nbformat.validator.normalize(nb)
    nbformat.validate(nb)
    nbformat.write(nb, NB_PATH)
    print(f"Patched: {NB_PATH} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
