"""Убирает хардкод доменных словарей из
notebooks/colab_glossa_02_finetune_qwen2_taiga_gemini.ipynb и переключает их
на загрузку из предпосчитанных файлов (data/domain_glosses_full.json,
data/domain_glosses_200.json — см. scripts/build_domain_glosses.py),
построенных из реальной таксономии доменов Slovo, а не из придуманных слов.

Правки:
- Ячейка Drive/Secrets: DOMAIN_GLOSSES_JSON теперь зависит от SCENARIO
  (_full.json для 'full', _200.json для 'top200') вместо фиксированного
  domain_glosses.json.
- Ячейка загрузки словарей: вместо DEFAULT_DOMAIN_GLOSSES (захардкоженные
  14 доменов с произвольными словами) — обязательная загрузка из файла;
  если файла нет на Drive — явная ошибка с инструкцией. Старый блок
  top200-фильтра с добором слов для доменов удалён — больше не нужен,
  т.к. data/domain_glosses_200.json уже предпосчитан под топ-200.

Запуск: python scripts/patch_colab_02_domain_glosses.py
"""

from __future__ import annotations

import nbformat

NB_PATH = "notebooks/colab_glossa_02_finetune_qwen2_taiga_gemini.ipynb"


def patch(nb: nbformat.NotebookNode) -> None:
    cells = nb.cells

    # ── Ячейка Drive/Secrets: путь к domain_glosses зависит от SCENARIO ──────
    c = cells[4]
    assert "DOMAIN_GLOSSES_JSON = f'{DRIVE_ROOT}/data/domain_glosses.json'" in c.source
    c.source = c.source.replace(
        "DOMAIN_GLOSSES_JSON = f'{DRIVE_ROOT}/data/domain_glosses.json'",
        "DOMAIN_GLOSSES_JSON = f'{DRIVE_ROOT}/data/domain_glosses_'"
        "f'{\"200\" if SCENARIO == \"top200\" else \"full\"}.json'",
    )
    c.outputs = []
    c.execution_count = None

    # ── Ячейка загрузки словарей: убираем хардкод DEFAULT_DOMAIN_GLOSSES ─────
    c = cells[6]
    assert "DEFAULT_DOMAIN_GLOSSES = {" in c.source

    old_block_start = c.source.index("# ── domain_glosses.json")
    old_block_end = c.source.index(
        "# ── Фильтр словаря под SCENARIO"
    )
    head = c.source[:old_block_start]

    new_domain_block = """\
# ── domain_glosses.json (зависит от SCENARIO, см. scripts/build_domain_glosses.py) ──
# Файл строится локально из реальной таксономии доменов Slovo (НЕ
# захардкожен здесь): python scripts/build_domain_glosses.py, затем
# domain_glosses_full.json / domain_glosses_200.json копируются в
# {DRIVE_ROOT}/data/.
if not Path(DOMAIN_GLOSSES_JSON).exists():
    raise FileNotFoundError(
        f'Не найден {DOMAIN_GLOSSES_JSON}.\\n'
        'Запустите locally: python scripts/build_domain_glosses.py\\n'
        'и скопируйте data/domain_glosses_full.json и data/domain_glosses_200.json '
        f'в {DRIVE_ROOT}/data/ на Google Drive.'
    )

with open(DOMAIN_GLOSSES_JSON, encoding='utf-8') as f:
    DOMAIN_GLOSSES = json.load(f)

print(f'[domain_glosses] Сценарий: {SCENARIO} -> {DOMAIN_GLOSSES_JSON}')
print(f'[domain_glosses] Загружено доменов: {len(DOMAIN_GLOSSES)}')
for domain, words in list(DOMAIN_GLOSSES.items())[:5]:
    print(f'  {domain}: {len(words)} глосс')
if len(DOMAIN_GLOSSES) > 5:
    print(f'  ... и ещё {len(DOMAIN_GLOSSES) - 5} доменов')

"""

    new_filter_block = """\
# ── Фильтр SLOVO_GLOSSES под SCENARIO ─────────────────────────────────────────
# top200: оставляем только глоссы из selected_classes_200.json — это всё,
# что cv-service в этом сценарии способен распознать/показать. DOMAIN_GLOSSES
# уже предпосчитан под нужный сценарий (см. блок выше), фильтровать его
# здесь повторно не нужно.
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
else:
    TOP200_SET = None
    print('[full] Фильтр SLOVO_GLOSSES не применяется — используется полный словарь Slovo.')
"""

    c.source = head + new_domain_block + new_filter_block
    c.outputs = []
    c.execution_count = None


def main() -> None:
    nb = nbformat.read(NB_PATH, as_version=4)
    patch(nb)
    nbformat.validator.normalize(nb)
    nbformat.validate(nb)
    nbformat.write(nb, NB_PATH)
    print(f"Patched: {NB_PATH} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
