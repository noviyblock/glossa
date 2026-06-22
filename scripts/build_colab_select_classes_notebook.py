"""Генерация notebooks/colab_glossa_00a_select_classes.ipynb.

Воспроизводимый (Colab Secrets, как colab_glossa_00) выбор топ-200 классов
жестов из корпуса Slovo. Не требует GPU — чистый pandas, занимает минуты.
Результат совместим с фильтром, зашитым в colab_glossa_00_preprocess_dwpose.ipynb
(одинаковая логика отбора: исключить no_event и однобуквенный дактиль,
взять первые N слов по алфавиту).

Запуск (локально, чтобы пересобрать .ipynb после правок):
    python scripts/build_colab_select_classes_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).parents[1] / "notebooks" / "colab_glossa_00a_select_classes.ipynb"

_CELL_INSTALL = """\
# ── 1. Установка зависимостей ──────────────────────────────────────────────────
# Лёгкий ноутбук: нужен только pandas + kagglehub, GPU не требуется.
import subprocess, sys

def pip(*args):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + list(args), check=True)

pip('kagglehub', 'pandas')
print('✅ Установка завершена')
"""

_CELL_SECRETS = """\
# ── 2. Kaggle credentials + скачивание annotations.csv ────────────────────────
import os
from google.colab import userdata

os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
os.environ['KAGGLE_KEY']      = userdata.get('KAGGLE_KEY')
print(f"Kaggle user: {os.environ['KAGGLE_USERNAME']}")

import kagglehub
from pathlib import Path

DATASET_PATH = kagglehub.dataset_download('kapitanov/slovo')
print(f'Датасет: {DATASET_PATH}')

ANNOT_CSV = list(Path(DATASET_PATH).rglob('annotations.csv'))[0]
print(f'annotations.csv: {ANNOT_CSV}')
"""

_CELL_SELECT = """\
# ── 3. Выбор топ-N классов ─────────────────────────────────────────────────────
# Критерий (детерминированный, без читерства):
#   1. Исключаем no_event (служебный класс "нет жеста").
#   2. Исключаем однобуквенные классы — это дактильный алфавит (35 знаков),
#      отдельная задача (побуквенное письмо), не совпадающая по семантике
#      со словами-жестами.
#   3. Из оставшихся ~965 словесных классов берём первые N по алфавиту —
#      детерминированный, не подогнанный под результат выбор.
#
# ВАЖНО: в полном корпусе Slovo почти все классы (кроме no_event) имеют
# РОВНО 20 записей (15 train / 5 val) — сокращение числа классов не даёт
# больше данных на класс, оно снижает число разделяющих границ задачи
# (1000-way/15-shot -> N-way/15-shot).

import pandas as pd

N_CLASSES = 200  # поменяй здесь, если нужно другое число

df = pd.read_csv(ANNOT_CSV, sep='\\t', on_bad_lines='skip')
df = df[df['text'] != 'no_event']

all_classes = sorted(df['text'].unique())
dactyl = [c for c in all_classes if len(c) == 1]
words = sorted(c for c in all_classes if len(c) > 1)

print(f'Всего классов (без no_event): {len(all_classes)}')
print(f'  из них дактиль (1 символ):  {len(dactyl)}')
print(f'  словесных жестов:           {len(words)}')

selected = words[:N_CLASSES]
assert len(selected) == N_CLASSES, f'Ожидали {N_CLASSES}, получили {len(selected)}'

print(f'\\nОтобрано: {len(selected)} классов')
print('Первые 10:', selected[:10])
"""

_CELL_VALIDATE = """\
# ── 4. Проверка покрытия выборками ────────────────────────────────────────────
sub = df[df['text'].isin(selected)]
sub_train = sub[sub['train'] == True]
sub_val   = sub[sub['train'] == False]

print(f'Видео в подвыборке: {len(sub)} (train={len(sub_train)}, val={len(sub_val)})')
print(f'Ожидаемо: {N_CLASSES} x 20 = {N_CLASSES * 20}')

per_class = sub['text'].value_counts()
print(f'\\nМин/макс записей на класс: {per_class.min()} / {per_class.max()}')
assert per_class.min() >= 15, 'Какой-то класс имеет аномально мало записей — проверь датасет'

print('\\n✅ Все классы покрыты ожидаемым количеством записей')
"""

_CELL_SAVE = """\
# ── 5. Сохранение selected_classes_N.json ─────────────────────────────────────
import json
from pathlib import Path

out = {
    'selection_method': f'alphabetical_first_{N_CLASSES}_excluding_dactyl',
    'excluded': ['no_event', f'single-character dactyl alphabet signs ({len(dactyl)} classes)'],
    'num_classes': len(selected),
    'classes': selected,
}

OUT_PATH = Path(f'/content/selected_classes_{N_CLASSES}.json')
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print(f'✅ Сохранено: {OUT_PATH}')
print('\\nСкачай файл и закоммить в репозиторий как data/selected_classes_200.json')
print('(на ветке feature/top200-classes-t64), либо просто сверь содержимое —')
print('если N_CLASSES=200 и метод не менялся, результат побайтово совпадает')
print('с уже закоммиченным data/selected_classes_200.json.')
"""

_CELL_DOWNLOAD = """\
# ── 6. Скачивание ──────────────────────────────────────────────────────────────
from google.colab import files
files.download(str(OUT_PATH))
"""

CELLS = [
    ("markdown", (
        "# Glossa — Выбор подмножества классов жестов (топ-N)\n"
        "**Среда:** Google Colab (CPU достаточно, GPU не нужен)  \n"
        "**Время:** ~1-2 мин  \n"
        "**Результат:** `selected_classes_200.json` — список N classов, "
        "совместимый с фильтром в `colab_glossa_00_preprocess_dwpose.ipynb`\n\n"
        "## Зачем\n"
        "Полный корпус Slovo — 1000 классов x 15 train-видео/класс. Это "
        "экстремальный few-shot режим: ST-GCN, обученный на всех 1000 классах, "
        "даёт top-1 accuracy ~38% при выраженном переобучении (train acc 78.6% "
        "vs val acc 38.2%). Сокращение словаря до N=200 снижает число "
        "разделяющих границ задачи при том же объёме данных на класс — "
        "контролируемый эксперимент, а не подгонка результата.\n\n"
        "## Перед запуском — Colab Secrets (🔑)\n"
        "- `KAGGLE_USERNAME` — логин Kaggle\n"
        "- `KAGGLE_KEY` — API-ключ из kaggle.json\n"
    )),
    ("markdown", "## Ячейка 1 — Установка зависимостей"),
    ("code", _CELL_INSTALL),
    ("markdown", "## Ячейка 2 — Kaggle credentials + annotations.csv"),
    ("code", _CELL_SECRETS),
    ("markdown", "## Ячейка 3 — Выбор топ-N классов"),
    ("code", _CELL_SELECT),
    ("markdown", "## Ячейка 4 — Проверка покрытия"),
    ("code", _CELL_VALIDATE),
    ("markdown", "## Ячейка 5 — Сохранение JSON"),
    ("code", _CELL_SAVE),
    ("markdown", "## Ячейка 6 — Скачивание"),
    ("code", _CELL_DOWNLOAD),
]


def build_notebook() -> dict:
    cells = []
    for kind, content in CELLS:
        cell = {
            "cell_type": kind,
            "metadata": {},
            "source": content.splitlines(keepends=True),
        }
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    nb = build_notebook()
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Notebook written: {NB_PATH}")
