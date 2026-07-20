"""Генерирует notebooks/colab_glossa_05_gloss_clips.ipynb — собирает по одному
эталонному клипу жеста на каждый из 200 классов, чтобы направление
текст→РЖЯ могло отдавать реальное видео вместо голой строки глосс.

Почему это нужно (контекст для будущих читателей):
Пайплайн текст→РЖЯ (ASR → NLP /translate_reverse → TTS) сейчас отдаёт
глухому пользователю только текстовую строку "ПРИВЕТ КАК ДЕЛА" — бесполезно,
если он не читает по-русски бегло или ждёт именно жестового ответа. Полный
синтез аватара — отдельный ML-проект; самое дешёвое и при этом реально
рабочее решение — склеить готовые эталонные клипы жестов из того же Slovo
корпуса, которым обучалась и дообучалась (RTMW) модель распознавания.

Как и colab_glossa_04, ноутбук переиспользует тот же доступ к Slovo через
kagglehub и ту же логику поиска видео по attachment_id/begin/end из
data/annotations.csv — никаких новых записей, никакого нового датасета.

Дизайн-решения (обсуждены и подтверждены пользователем перед реализацией):
- 1 клип на класс (не несколько) — проще пайплайн, меньше веса в
  models/gloss_clips на VM. Можно поднять CLIPS_PER_CLASS позже, если
  понадобится вариативность.
- Жёсткая нарезка (concat demuxer, без ре-энкода) на стороне tts_service —
  НЕ crossfade. Поэтому здесь стандартизируем кодек/разрешение/fps у ВСЕХ
  200 клипов на экспорте, чтобы concat demuxer мог их просто склеивать
  без пересборки на лету.
- Имена файлов — по числовому индексу класса (`{idx}.mp4`), не по русскому
  названию глоссы: class_to_idx.json/idx_to_class.json остаются единственным
  источником истины для маппинга текст↔индекс, а числовые имена файлов не
  создают проблем с путями/URL-энкодингом кириллицы.

Запуск: python scripts/generate_colab_05_gloss_clips.py
"""
from __future__ import annotations

import nbformat as nbf

NB_PATH = "notebooks/colab_glossa_05_gloss_clips.ipynb"


def md(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(src)


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src)


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells: list[nbf.NotebookNode] = []

    # ── 0. Заголовок и контекст ──────────────────────────────────────────── #
    cells.append(md(
"""# colab_glossa_05 — Сбор эталонных видео-клипов жестов (gloss → видео)

**Зачем этот ноутбук.** Направление текст→РЖЯ (`ASR → NLP /translate_reverse
→ TTS`) сейчас отдаёт глухому пользователю только текстовую строку глосс
(`ПРИВЕТ КАК ДЕЛА`) — реального видимого жестового ответа нет. Этот ноутбук
не делает синтез аватара — он просто вырезает по одному эталонному клипу
жеста на каждый из 200 классов из того же Slovo-корпуса, которым обучалась
и дообучалась (RTMW) модель распознавания, чтобы `tts_service` мог склеивать
их в ответ жёсткой нарезкой (concat demuxer, см. `services/tts_service/video.py`).

**Никаких новых видео снимать не нужно** — используется тот же доступ к
Slovo через `kagglehub` и та же логика поиска видео по
`attachment_id`/`begin`/`end` из `data/annotations.csv`, что и в
`colab_glossa_04_finetune_rtmw`.

**Почему стандартизируем кодек/разрешение/fps на экспорте:** склейка на
проде — `concat demuxer` (`ffmpeg -f concat`), без ре-энкода — самый
быстрый и надёжный способ, но требует, чтобы у всех входных клипов
совпадали параметры потока. Поэтому здесь КАЖДЫЙ клип перекодируется в
единый формат (h264, 480px по ширине, 25fps, yuv420p, без звука) один раз,
а не на лету при каждом запросе пользователя.

**Почему имена файлов — числовые (`{idx}.mp4`), а не по тексту глоссы:**
`data/class_to_idx.json`/`data/idx_to_class.json` остаются единственным
источником истины для маппинга слово↔индекс; числовые имена не создают
проблем с кодировкой кириллицы в путях и URL."""
    ))

    # ── 1. Setup ──────────────────────────────────────────────────────────── #
    cells.append(md("## Ячейка 1 — Окружение и пути"))
    cells.append(code(
"""from google.colab import drive
drive.mount('/content/drive')

DRIVE_ROOT = '/content/drive/MyDrive/glossa'   # поправьте под свою структуру
OUT_DIR    = f'{DRIVE_ROOT}/data/gloss_clips_export'

import os
os.makedirs(OUT_DIR, exist_ok=True)
print('OUT_DIR:', OUT_DIR)"""
    ))

    cells.append(md("## Ячейка 2 — Зависимости"))
    cells.append(code(
"""!pip install -q kagglehub pandas

# Colab поставляется с ffmpeg предустановленным — просто проверяем.
!ffmpeg -version | head -1"""
    ))

    # ── 2. Скачивание Slovo + выборка по 1 клипу на класс ───────────────────── #
    cells.append(md(
"""## Ячейка 3 — Скачивание Slovo + выборка 1 клипа на класс

Та же логика, что в `colab_glossa_04` (Ячейка 3): скачиваем корпус целиком
через `kagglehub`, фильтруем аннотации по 200 классам из `class_to_idx.json`,
сэмплируем `CLIPS_PER_CLASS=1` клип на класс (фиксированный `SEED` — тот же
клип при повторном запуске)."""
    ))
    cells.append(code(
"""import os
from google.colab import userdata

os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
os.environ['KAGGLE_KEY']      = userdata.get('KAGGLE_KEY')

import kagglehub, time
from pathlib import Path
print('Скачиваем датасет Slovo RSL (или берём из кэша, если уже скачан)...')
t0 = time.time()
DATASET_PATH = kagglehub.dataset_download('kapitanov/slovo')
print(f'Готово за {(time.time()-t0)/60:.1f} мин. Путь: {DATASET_PATH}')

import json
import pandas as pd

SLOVO_ROOT = Path(DATASET_PATH)
ANNOT_CSV  = list(SLOVO_ROOT.rglob('annotations.csv'))[0]
df = pd.read_csv(ANNOT_CSV, sep='\\t', on_bad_lines='skip')
df = df[df['text'] != 'no_event'].reset_index(drop=True)

class_to_idx = json.loads(Path(f'{DRIVE_ROOT}/data/class_to_idx.json').read_text(encoding='utf-8'))
print(f'{len(class_to_idx)} классов в маппинге (сценарий top200)')

df = df[df['text'].isin(class_to_idx)].reset_index(drop=True)
print(f'Аннотаций после фильтра по 200 классам: {len(df)}')


def find_video(attachment_id: str) -> str | None:
    for split in ['train', 'test', 'val']:
        p = SLOVO_ROOT / 'slovo' / split / f'{attachment_id}.mp4'
        if p.exists():
            return str(p)
    return None


CLIPS_PER_CLASS = 1   # обсуждено и подтверждено — см. docstring модуля-генератора
SEED = 42

sampled = (
    df.groupby('text', group_keys=False)
      .apply(lambda g: g.sample(min(len(g), CLIPS_PER_CLASS), random_state=SEED))
      .reset_index(drop=True)
)
print(f'Выбрано {len(sampled)} клипов ({sampled["text"].nunique()} / {len(class_to_idx)} классов покрыто)')

missing_classes = sorted(set(class_to_idx) - set(sampled['text']))
if missing_classes:
    print(f'\\n⚠️  {len(missing_classes)} классов без аннотаций в Slovo вообще '
          f'(не найдётся клип, tts_service должен уметь это переживать):')
    print(missing_classes)"""
    ))

    # ── 3. Нарезка + перекодирование в единый формат ────────────────────────── #
    cells.append(md(
"""## Ячейка 4 — Нарезка (begin/end) + перекодирование в единый формат

Каждый исходный клип обрезается по `begin`/`end` (как и при извлечении
keypoints в `colab_glossa_00`/`colab_glossa_04` — Slovo-видео может содержать
не только целевой жест) и перекодируется в ЕДИНЫЙ формат: h264, 480px по
ширине (высота — по аспекту, чётная), 25fps, yuv420p, без звука. Имя файла —
числовой индекс класса, под который `tts_service/video.py` его и будет
искать в проде."""
    ))
    cells.append(code(
"""import subprocess
from tqdm.notebook import tqdm

TARGET_WIDTH = 480
TARGET_FPS   = 25

ok, failed = [], []

for row in tqdm(sampled.itertuples(), total=len(sampled), desc='Нарезка/перекодирование'):
    vpath = find_video(row.attachment_id)
    cls_idx = class_to_idx[row.text]
    out_path = f'{OUT_DIR}/{cls_idx}.mp4'

    if vpath is None:
        failed.append((row.text, cls_idx, 'video not found in Slovo dump'))
        continue

    begin, end = int(row.begin), int(row.end)
    if end <= begin:
        failed.append((row.text, cls_idx, f'invalid begin/end ({begin}, {end})'))
        continue

    vf = (
        f"select='between(n\\\\,{begin}\\\\,{end})',"
        f"setpts=PTS-STARTPTS,"
        f"scale={TARGET_WIDTH}:-2"
    )
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-i', vpath,
        '-vf', vf,
        '-an',
        '-r', str(TARGET_FPS),
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-vsync', '0',
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(out_path):
        failed.append((row.text, cls_idx, result.stderr[-300:]))
        continue
    ok.append((row.text, cls_idx))

print(f'\\nГотово: {len(ok)} клипов экспортировано, {len(failed)} провалилось.')
if failed:
    print('\\nПровалившиеся классы:')
    for text, idx, reason in failed:
        print(f'  [{idx}] {text!r}: {reason}')"""
    ))

    # ── 4. Проверка + упаковка ──────────────────────────────────────────────── #
    cells.append(md(
"""## Ячейка 5 — Проверка покрытия + упаковка для выгрузки на VM

Итоговое покрытие 200 классов (аннотация отсутствовала ИЛИ нарезка/перекодирование
провалились — в обоих случаях `tts_service` должен просто пропускать такой
индекс при склейке, не падать) и zip-архив для скачивания."""
    ))
    cells.append(code(
"""exported_idx = {int(Path(p).stem) for p in Path(OUT_DIR).glob('*.mp4')}
all_idx = set(class_to_idx.values())
covered = len(exported_idx & all_idx)
print(f'Покрытие: {covered}/{len(all_idx)} классов имеют экспортированный клип')
if covered < len(all_idx):
    still_missing = sorted(all_idx - exported_idx)
    print(f'Классы без клипа (индексы): {still_missing}')

import shutil
ZIP_PATH = f'{DRIVE_ROOT}/data/gloss_clips_export.zip'
shutil.make_archive(ZIP_PATH.removesuffix('.zip'), 'zip', OUT_DIR)
size_mb = os.path.getsize(ZIP_PATH) / 1e6
print(f'\\nАрхив готов: {ZIP_PATH} ({size_mb:.1f} МБ)')"""
    ))

    # ── 5. Промоушен на VM ───────────────────────────────────────────────────── #
    cells.append(md(
"""## Ячейка 6 (справочная, выполняется НЕ здесь) — Промоушен на VM

```bash
# 1. Скачать gloss_clips_export.zip из Google Drive на VM, распаковать:
mkdir -p /opt/glossa/models/gloss_clips
unzip gloss_clips_export.zip -d /opt/glossa/models/gloss_clips

# 2. DVC-трек (тот же паттерн, что остальные активы в models/):
cd /opt/glossa
dvc add models/gloss_clips
git add models/gloss_clips.dvc models/.gitignore
git commit -m "feat: add gloss->video reference clips (colab_glossa_05)"
dvc push

# 3. Перезапуск tts-service не требуется отдельно монтировать новый volume —
#    ./models:/models:ro уже смонтирован (см. docker-compose.yml x-model-volumes),
#    достаточно docker compose restart tts-service после появления файлов.
```"""
    ))

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    return nb


def main() -> None:
    nb = build()
    nbf.validate(nb)
    with open(NB_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Written: {NB_PATH} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
