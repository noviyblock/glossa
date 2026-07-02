"""Генерирует notebooks/colab_glossa_04_finetune_rtmw.ipynb — дообучение
(domain adaptation) ST-GCN на keypoints, извлечённых ЖИВЫМ пайплайном
(rtmlib/RTMW), а не офлайн-пайплайном (DWPose), которым обучалась текущая
модель.

Почему это нужно (контекст для будущих читателей):
Модель никогда не видела ни одного примера, извлечённого через rtmlib —
только через controlnet_aux.dwpose.DwposeDetector (OpenPose-JSON формат).
cv_service/keypoint_extractor.py воспроизводит тот же порядок точек через
_COCO17_TO_TRAINING_ORDER, но это лишь подгонка порядка индексов, не
распределения: RTMDet-nano/RTMW имеют свой bias в детекции/позе, отличный
от YOLOX-L/DWPose. Новых видео снимать не нужно — ноутбук скачивает те же
исходные клипы Slovo через kagglehub (тот же доступ, что colab_glossa_00) и
просто извлекает их через RTMW вместо DWPose, так что "новый" набор
отличается от старого именно экстрактором, а не человеком/камерой/светом.
Даже небольшой (5-10 клипов/класс) такой набор, подмешанный в дообучение
поверх текущих весов, должен закрыть этот разрыв лучше любых инженерных
патчей на инференсе.

Запуск: python scripts/generate_colab_04_finetune_rtmw.py
"""
from __future__ import annotations

import nbformat as nbf

NB_PATH = "notebooks/colab_glossa_04_finetune_rtmw.ipynb"


def md(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(src)


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src)


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells: list[nbf.NotebookNode] = []

    # ── 0. Заголовок и контекст ──────────────────────────────────────────── #
    cells.append(md(
"""# colab_glossa_04 — Дообучение ST-GCN на RTMW-фичах (domain adaptation)

**Зачем этот ноутбук.** Действующая модель (`stgcn_best.pt` / `stgcn_final.pt`,
экспортированная в `gesture_classifier_mobile.onnx` + `stgcn_topk_int8`)
обучалась ИСКЛЮЧИТЕЛЬНО на keypoints, извлечённых через
`controlnet_aux.dwpose.DwposeDetector` (offline-пайплайн, `colab_glossa_00`).
Продакшен (`cv_service`) с недавних пор извлекает keypoints через
`rtmlib` (RTMDet-nano + RTMW) — другую пару моделей с другим detection/pose
bias. Мы уже воспроизвели тот же ПОРЯДОК точек на инференсе
(`_COCO17_TO_TRAINING_ORDER` в `keypoint_extractor.py`), но порядок — не то
же самое, что распределение: сама геометрия/точность/шум RTMW отличается от
DWPose, и модель никогда не видела ни одного такого примера.

**Подход — не полный retrain, а fine-tune (domain adaptation):**
1. Скачать Slovo через `kagglehub` (те же credentials/доступ, что
   `colab_glossa_00`) и сэмплировать несколько клипов на класс из
   `annotations.csv` — **никаких новых записей не требуется**, берём те же
   исходные видео, что уже размечены под сценарий `top200`.
2. Извлечь keypoints из этих клипов ТЕМ ЖЕ кодом, что использует продакшен
   (`rtmlib.Wholebody` + идентичный ремап/нормализация) — так «новый» набор
   отличается от старого именно экстрактором (DWPose → RTMW), а не человеком
   в кадре, это и есть источник разрыва, который мы закрываем.
3. Пересчитать BatchNorm running-статистику модели на новых данных
   (дёшево, без градиентов — часто даёт заметный прирост само по себе).
4. Дообучить с низким LR на смеси {новые RTMW-клипы (с повышенным весом) +
   сэмпл старых DWPose-клипов (для регуляризации, чтобы не забыть остальные
   198 классов, которых в новых данных нет)}.
5. Сравнить accuracy до/после на старой (DWPose) val-выборке — не должно
   заметно просесть — и на новой (RTMW) val-выборке — должно вырасти.
6. Экспортировать в ONNX/OpenVINO под НОВЫМИ именами файлов — не
   перезаписывать продакшен-артефакты автоматически, руками промотировать
   после проверки.

**Важная отдельная находка (не чинится в этом ноутбуке — см. последнюю
ячейку):** графовая структура (`load_skeleton_graph()` в `colab_glossa_01a`)
похоже построена в предположении ИСТИННОГО COCO-17 порядка точек
(рёбра вида `(5,7)` = "плечо-локоть" под COCO-нумерацией), а реальные
фичи (`features.npy`) — в OpenPose-переставленном порядке (индекс 5 = левый
локоть, а не плечо). Это структурный prior графовой свёртки, а не баг
инференса — почитать и подумать отдельно, вне scope этого ноутбука."""
    ))

    # ── 1. Setup ──────────────────────────────────────────────────────────── #
    cells.append(md("## Ячейка 1 — Окружение и пути"))
    cells.append(code(
"""from google.colab import drive
drive.mount('/content/drive')

DRIVE_ROOT  = '/content/drive/MyDrive/glossa'   # поправьте под свою структуру
MODELS_DIR  = f'{DRIVE_ROOT}/models'
DATA_DIR    = f'{DRIVE_ROOT}/data/gestures/processed_64_200'   # существующий DWPose-датасет
OUT_DIR     = f'{DRIVE_ROOT}/data/gestures/rtmw_finetune_out'

import os
os.makedirs(OUT_DIR, exist_ok=True)
print('DATA_DIR (старый DWPose):', DATA_DIR)"""
    ))

    cells.append(md("## Ячейка 2 — Установка зависимостей"))
    cells.append(code(
"""!pip install -q rtmlib onnxruntime openvino torch torchvision scikit-learn mlflow tqdm opencv-python-headless kagglehub

import torch, numpy as np, json, cv2
from pathlib import Path
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"""
    ))

    # ── 2. Скачивание Slovo + выборка клипов (тот же доступ, что colab_glossa_00) ── #
    cells.append(md(
"""## Ячейка 3 — Скачивание Slovo + выборка клипов для дообучения

**Не нужно записывать новые видео.** Используем ТЕ ЖЕ исходные клипы Slovo,
что и `colab_glossa_00` (те же Kaggle credentials, тот же `annotations.csv`),
просто извлекаем их через RTMW вместо DWPose — так «новый» набор отражает
реальные жесты датасета, а не самодельные, и разрыв домена закрывается
именно между экстракторами (DWPose vs RTMW), а не между людьми/камерами.

Сэмплируем `CLIPS_PER_CLASS` случайных клипов на каждый из 200 классов
(сценарий `top200`, тот же `class_to_idx.json`, что видит продакшен)."""
    ))
    cells.append(code(
"""import os
from google.colab import userdata

os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
os.environ['KAGGLE_KEY']      = userdata.get('KAGGLE_KEY')

import kagglehub, time
print('Скачиваем датасет Slovo RSL (или берём из кэша, если уже скачан)...')
t0 = time.time()
DATASET_PATH = kagglehub.dataset_download('kapitanov/slovo')
print(f'Готово за {(time.time()-t0)/60:.1f} мин. Путь: {DATASET_PATH}')

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


CLIPS_PER_CLASS = 8  # начните с малого — даже 3-5 на класс уже помогает
SEED = 42

sampled = (
    df.groupby('text', group_keys=False)
      .apply(lambda g: g.sample(min(len(g), CLIPS_PER_CLASS), random_state=SEED))
      .reset_index(drop=True)
)
print(f'Выбрано {len(sampled)} клипов ({sampled["text"].nunique()} классов, '
      f'до {CLIPS_PER_CLASS} на класс)')"""
    ))

    cells.append(md(
"""## Ячейка 4 — Извлечение keypoints через rtmlib (тот же код, что в продакшене)

Копия логики `services/cv_service/keypoint_extractor.py` — намеренно
скопирована, а не импортирована, чтобы ноутбук был самодостаточным в Colab
(там нет доступа к репозиторию). **Если поменяете
`_COCO17_TO_TRAINING_ORDER` / `_EXTRA_JOINT_SRC` / confidence-порог зануления
в `keypoint_extractor.py` — обновите и здесь**, иначе фичи разъедутся с
продакшеном.

`extract_video_keypoints` принимает `begin`/`end` (кадры), как и
`process_video()` в `colab_glossa_00` — Slovo-видео могут содержать не
только целевой жест, обрезаем строго так же, как офлайн-пайплайн."""
    ))

    cells.append(code(
"""from rtmlib import Wholebody

# ── Идентично keypoint_extractor.py — держать в синхроне ──────────────────── #
_COCO17_TO_TRAINING_ORDER = [0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
_EXTRA_JOINT_SRC = [5, 6, 11, 12, 13, 14, 15, 16, 5, 6]
_LOW_CONF_ZERO_THRESHOLD = 0.3  # см. keypoint_extractor.py::_zero_low_confidence_joints

def remap_coco133_to_75(kp133: np.ndarray) -> np.ndarray:
    body17 = kp133[0:17][_COCO17_TO_TRAINING_ORDER]
    feet6 = kp133[17:23]
    extra10 = body17[_EXTRA_JOINT_SRC]
    pose33 = np.concatenate([body17, feet6, extra10], axis=0)
    left_hand = kp133[91:112]
    right_hand = kp133[112:133]
    kp75 = np.concatenate([pose33, left_hand, right_hand], axis=0).astype(np.float32)
    # зануление низкоуверенных точек — см. Ячейку fix в keypoint_extractor.py
    low_conf = kp75[:, 2] < _LOW_CONF_ZERO_THRESHOLD
    kp75[low_conf] = 0.0
    return kp75


_wholebody = Wholebody(mode='lightweight', backend='onnxruntime', device='cpu', to_openpose=False)

def extract_video_keypoints(video_path: str, begin: int | None = None, end: int | None = None) -> np.ndarray:
    \"\"\"Возвращает (T_raw, 75, 3) — по одному кадру в диапазоне [begin, end]
    (весь клип, если begin/end не заданы), как process_video() в colab_glossa_00.\"\"\"
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    b = max(0, begin) if begin is not None else 0
    e = min(total - 1, end) if end is not None else total - 1

    frames_kp = []
    for idx in range(b, e + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frames_kp.append(np.zeros((75, 3), dtype=np.float32))
            continue
        h, w = frame.shape[:2]
        keypoints, scores = _wholebody(frame)
        if keypoints is None or len(keypoints) == 0:
            frames_kp.append(np.zeros((75, 3), dtype=np.float32))
            continue
        best = int(np.argmax([s.mean() for s in scores])) if len(keypoints) > 1 else 0
        kp = np.asarray(keypoints[best], dtype=np.float32)
        sc = np.asarray(scores[best], dtype=np.float32)
        kp[:, 0] = np.clip(kp[:, 0] / w, 0.0, 1.0)
        kp[:, 1] = np.clip(kp[:, 1] / h, 0.0, 1.0)
        kp133 = np.stack([kp[:, 0], kp[:, 1], sc], axis=1)
        frames_kp.append(remap_coco133_to_75(kp133))
    cap.release()
    return np.stack(frames_kp, axis=0) if frames_kp else np.zeros((0, 75, 3), dtype=np.float32)


def normalize_keypoints(kp: np.ndarray) -> np.ndarray:
    \"\"\"Hip-center + shoulder-scale — идентично colab_glossa_00 / normalizer.py._translate_scale_invariant.\"\"\"
    HIP_IDX, SHOU_IDX = [11, 12], [5, 6]
    xy = kp[:, :, :2].copy()
    hips_center = np.median(xy[:, HIP_IDX, :], axis=1, keepdims=True)
    xy -= hips_center
    shoulder_dist = np.linalg.norm(xy[:, SHOU_IDX[0], :] - xy[:, SHOU_IDX[1], :], axis=1)
    shoulder_dist = np.maximum(shoulder_dist, 1e-6).reshape(-1, 1, 1)
    xy /= shoulder_dist
    out = kp.copy()
    out[:, :, :2] = xy
    return out


def resample_to(window: np.ndarray, target_T: int = 64) -> np.ndarray:
    T = window.shape[0]
    if T == target_T:
        return window
    src_idx = np.linspace(0, T - 1, target_T)
    lo = np.floor(src_idx).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    alpha = (src_idx - lo)[:, None, None]
    return ((1 - alpha) * window[lo] + alpha * window[hi]).astype(np.float32)

print('Функции извлечения готовы.')"""
    ))

    cells.append(md("## Ячейка 5 — Прогон выбранных клипов через RTMW"))
    cells.append(code(
"""from tqdm.notebook import tqdm

new_X, new_y, new_paths = [], [], []
skipped = []

for row in tqdm(sampled.itertuples(), total=len(sampled), desc='Извлечение RTMW'):
    vpath = find_video(row.attachment_id)
    if vpath is None:
        skipped.append(row.attachment_id)
        continue
    cls_idx = class_to_idx[row.text]
    raw_kp = extract_video_keypoints(vpath, int(row.begin), int(row.end))
    if raw_kp.shape[0] < 8:
        skipped.append(row.attachment_id)
        continue
    normed = normalize_keypoints(raw_kp)
    resampled = resample_to(normed, target_T=64)
    new_X.append(resampled.astype(np.float16))
    new_y.append(cls_idx)
    new_paths.append(vpath)

new_X = np.stack(new_X, axis=0) if new_X else np.zeros((0, 64, 75, 3), dtype=np.float16)
new_y = np.array(new_y, dtype=np.int64)
print(f'Извлечено {len(new_X)} клипов, классов: {len(set(new_y.tolist()))}')
print(f'Пропущено (видео не найдено/слишком короткое): {len(skipped)}')

np.save(f'{OUT_DIR}/new_X_raw.npy', new_X)      # ещё БЕЗ z-score — только hip/shoulder-норм
np.save(f'{OUT_DIR}/new_y.npy', new_y)"""
    ))

    # ── 3. Train/val split нового набора ────────────────────────────────────── #
    cells.append(md(
"""## Ячейка 6 — Train/val split нового набора

При малом числе клипов на класс (5-10) — holdout 1-2 клипа на класс для
валидации, остальное в train. Если на класс всего 1 клип — он идёт только
в train (нельзя оценить точность по этому классу, но дообучение всё равно
полезно)."""
    ))
    cells.append(code(
"""from collections import defaultdict
import random
random.seed(42)

by_class = defaultdict(list)
for i, cls in enumerate(new_y.tolist()):
    by_class[cls].append(i)

train_idx, val_idx = [], []
for cls, idxs in by_class.items():
    random.shuffle(idxs)
    if len(idxs) >= 3:
        val_idx.extend(idxs[:1])       # 1 клип на класс в val, если их >=3
        train_idx.extend(idxs[1:])
    else:
        train_idx.extend(idxs)          # слишком мало — всё в train

new_X_train, new_y_train = new_X[train_idx], new_y[train_idx]
new_X_val,   new_y_val   = new_X[val_idx],   new_y[val_idx]
print(f'new train={len(new_X_train)}  new val={len(new_X_val)}')"""
    ))

    # ── 4. Загрузка существующей модели + старого датасета ─────────────────── #
    cells.append(md(
"""## Ячейка 7 — Загрузка текущего чекпоинта и старого (DWPose) датасета

Нужны определения `STGCN`/`GraphConv`/`TCN`/`STGCNBlock`/`load_skeleton_graph`
из `colab_glossa_01a_train_stgcn_64_200.ipynb` — выполните эти ячейки того
ноутбука перед этим (или скопируйте класс сюда — граф и архитектура должны
быть идентичны чекпоинту, иначе `load_state_dict` не сработает)."""
    ))
    cells.append(code(
"""# --- ЗАМЕНИТЕ на фактическое содержимое ячеек STGCN/GraphConv/TCN/STGCNBlock/
#     load_skeleton_graph из colab_glossa_01a_train_stgcn_64_200.ipynb, либо
#     выполните тот ноутбук в этой же runtime перед этой ячейкой. ---
assert 'STGCN' in dir(), 'Определите класс STGCN (см. colab_glossa_01a) перед продолжением'

CKPT_PATH = f'{MODELS_DIR}/stgcn_final.pt'
NUM_CLASSES, NUM_NODES, SEQ_LEN = 200, 75, 64

model = STGCN(num_classes=NUM_CLASSES, num_nodes=NUM_NODES, mobile=False).to(DEVICE)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
print(f'Загружен чекпоинт: {CKPT_PATH}')

# Старый (DWPose) датасет — для регуляризации при дообучении + контроль регрессии
X_train_old = np.load(f'{DATA_DIR}/train/features.npy', mmap_mode='r')
y_train_old = np.load(f'{DATA_DIR}/train/labels.npy')
X_val_old   = np.load(f'{DATA_DIR}/val/features.npy')
y_val_old   = np.load(f'{DATA_DIR}/val/labels.npy')

norm_stats = np.load(f'{MODELS_DIR}/norm_stats.npz')
mean, std = norm_stats['mean'], norm_stats['std']
std = np.where(std < 1e-6, 1.0, std)
print(f'Старый train={len(X_train_old)}  Старый val={len(X_val_old)}')"""
    ))

    # ── 5. BN recalibration (дёшево, без градиентов) ───────────────────────── #
    cells.append(md(
"""## Ячейка 8 — Recalibration BatchNorm (шаг 0, до полноценного fine-tune)

`STGCN.data_bn` — это `nn.BatchNorm1d`, чья running-статистика (mean/var)
"заморожена" под распределение DWPose-фичей. Простой и очень дешёвый первый
шаг domain adaptation: прогнать новые RTMW-клипы через модель в
`model.train()` режиме **без backprop** — только обновить running-статистику
BatchNorm. Иногда сам по себе даёт заметный прирост, до всякого fine-tune
градиентами. Пропустите эту ячейку, если новых клипов совсем мало (< 20-30) —
на таком объёме recalibration может, наоборот, ухудшить статистику."""
    ))
    cells.append(code(
"""if len(new_X_train) >= 20:
    new_X_train_z = ((new_X_train.astype(np.float32) - mean[0]) / std[0]).astype(np.float32)
    recalib_tensor = torch.tensor(new_X_train_z, dtype=torch.float32)

    model.train()
    with torch.no_grad():
        for i in range(0, len(recalib_tensor), 16):
            model(recalib_tensor[i:i+16].to(DEVICE))
    model.eval()
    print(f'BatchNorm recalibrated на {len(recalib_tensor)} новых сэмплах.')
else:
    print(f'Пропущено — только {len(new_X_train)} новых train-сэмплов (< 20), риск ухудшить статистику.')"""
    ))

    # ── 6. Fine-tune ─────────────────────────────────────────────────────────── #
    cells.append(md(
"""## Ячейка 9 — Fine-tune с низким LR на смеси старых + новых данных

- LR на 1-2 порядка ниже, чем при исходном обучении (`CFG['lr']=1e-3` →
  здесь `1e-4` или ниже) — чтобы не "перезаписать" то, что модель уже
  выучила на 200 классах, а лишь подстроить под новое распределение.
- Новые RTMW-клипы **умышленно oversampled** (повторены N раз в эпохе) —
  иначе на фоне тысяч старых DWPose-примеров их градиентный вклад
  теряется.
- Малое число эпох (`FT_EPOCHS`) — это fine-tune, не обучение с нуля."""
    ))
    cells.append(code(
"""FT_LR = 1e-4
FT_EPOCHS = 15
FT_BATCH_SIZE = 32
OVERSAMPLE_NEW = 8   # каждый новый клип повторяется N раз за эпоху
OLD_SUBSAMPLE = 4000  # сколько старых сэмплов подмешать для регуляризации (None = все)

X_train_old_z = ((np.array(X_train_old).astype(np.float32) - mean[0]) / std[0]).astype(np.float32)
if OLD_SUBSAMPLE is not None and len(X_train_old_z) > OLD_SUBSAMPLE:
    keep = np.random.default_rng(42).choice(len(X_train_old_z), OLD_SUBSAMPLE, replace=False)
    X_train_old_z, y_train_old_sub = X_train_old_z[keep], y_train_old[keep]
else:
    y_train_old_sub = y_train_old

new_X_train_z = ((new_X_train.astype(np.float32) - mean[0]) / std[0]).astype(np.float32)
new_X_train_z_rep = np.repeat(new_X_train_z, OVERSAMPLE_NEW, axis=0)
new_y_train_rep = np.repeat(new_y_train, OVERSAMPLE_NEW, axis=0)

X_mix = np.concatenate([X_train_old_z, new_X_train_z_rep], axis=0)
y_mix = np.concatenate([y_train_old_sub, new_y_train_rep], axis=0)
print(f'Смешанный fine-tune датасет: {len(X_mix)} сэмплов ({len(new_X_train_z_rep)} новых с oversample x{OVERSAMPLE_NEW})')

ft_dataset = torch.utils.data.TensorDataset(
    torch.tensor(X_mix, dtype=torch.float32), torch.tensor(y_mix, dtype=torch.long))
ft_loader = torch.utils.data.DataLoader(ft_dataset, batch_size=FT_BATCH_SIZE, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=FT_LR, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

X_val_old_z = ((np.array(X_val_old).astype(np.float32) - mean[0]) / std[0]).astype(np.float32)
val_old_tensor = torch.tensor(X_val_old_z, dtype=torch.float32)
val_old_labels = torch.tensor(y_val_old, dtype=torch.long)

new_X_val_z = ((new_X_val.astype(np.float32) - mean[0]) / std[0]).astype(np.float32)
val_new_tensor = torch.tensor(new_X_val_z, dtype=torch.float32)
val_new_labels = torch.tensor(new_y_val, dtype=torch.long)

def eval_acc(x, y):
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(x), 64):
            preds.append(model(x[i:i+64].to(DEVICE)).argmax(1).cpu())
        preds = torch.cat(preds)
    return (preds == y).float().mean().item() if len(y) else float('nan')

print(f'ДО fine-tune: old_val_acc={eval_acc(val_old_tensor, val_old_labels):.4f}  new_val_acc={eval_acc(val_new_tensor, val_new_labels):.4f}')

best_state, best_new_acc = None, -1.0
for epoch in range(1, FT_EPOCHS + 1):
    model.train()
    for xb, yb in ft_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    old_acc = eval_acc(val_old_tensor, val_old_labels)
    new_acc = eval_acc(val_new_tensor, val_new_labels)
    print(f'epoch {epoch:2d}  loss={loss.item():.4f}  old_val_acc={old_acc:.4f}  new_val_acc={new_acc:.4f}')

    if new_acc >= best_new_acc and old_acc >= eval_acc(val_old_tensor, val_old_labels) - 0.05:
        best_new_acc = new_acc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

if best_state is not None:
    model.load_state_dict(best_state)
    print(f'\\nЛучший чекпоинт по new_val_acc={best_new_acc:.4f} восстановлен.')"""
    ))

    cells.append(md(
"""## Ячейка 10 — Итоговое сравнение до/после

**Что смотреть:** `old_val_acc` не должен заметно просесть (> 3-5 пунктов —
сигнал переобучения на маленьком новом наборе, стоит уменьшить `FT_EPOCHS`
или `FT_LR`, или увеличить `OLD_SUBSAMPLE`). `new_val_acc` должен вырасти —
если нет, скорее всего новых данных пока мало для сигнала, либо проблема
не в domain gap, а в чём-то ещё (см. офлайн-замер точности отдельно)."""
    ))
    cells.append(code(
"""print(f'ПОСЛЕ fine-tune: old_val_acc={eval_acc(val_old_tensor, val_old_labels):.4f}  new_val_acc={eval_acc(val_new_tensor, val_new_labels):.4f}')"""
    ))

    # ── 7. Экспорт ────────────────────────────────────────────────────────── #
    cells.append(md(
"""## Ячейка 11 — Экспорт (НЕ перезаписывает продакшен-файлы)

Сохраняет под новыми именами (`*_rtmw_ft.*`). После проверки на реальной
камере — переименовать/скопировать вручную поверх продакшен-путей
(`gesture_classifier_mobile.onnx`, `stgcn_topk_int8/`), как в
`colab_glossa_03_export_inference_topk.ipynb`."""
    ))
    cells.append(code(
"""FT_OUT = f'{MODELS_DIR}/finetuned_rtmw'
os.makedirs(FT_OUT, exist_ok=True)

torch.save(model.state_dict(), f'{FT_OUT}/stgcn_rtmw_ft.pt')

model.eval()
dummy = torch.randn(1, SEQ_LEN, NUM_NODES, 3).to(DEVICE)
torch.onnx.export(
    model, dummy, f'{FT_OUT}/gesture_classifier_rtmw_ft.onnx',
    input_names=['keypoints'], output_names=['logits'],
    dynamic_axes={'keypoints': {0: 'batch_size'}, 'logits': {0: 'batch_size'}},
    export_params=True, opset_version=18,
)
print(f'Сохранено: {FT_OUT}/stgcn_rtmw_ft.pt, {FT_OUT}/gesture_classifier_rtmw_ft.onnx')
print('Для OpenVINO INT8: прогнать через тот же quantization pipeline, что colab_glossa_03')
print('(mo/ovc convert + NNCF calibration на смешанном val-наборе old+new).')"""
    ))

    # ── 8. Отдельная находка про граф ────────────────────────────────────── #
    cells.append(md(
"""## Ячейка 12 (справочная, код не выполнять здесь) — Находка про граф скелета

`load_skeleton_graph()` в `colab_glossa_01a` строит рёбра графовой свёртки
похоже в предположении **истинного COCO-17** порядка индексов 0-16
(`(5,7)` трактуется как "левое плечо — левый локоть"), но реальные фичи
(`features.npy`) — в OpenPose-переставленном порядке, где индекс 5 = левый
локоть, а не плечо (см. `keypoint_extractor.py::_COCO17_TO_TRAINING_ORDER`
и её комментарий). Это НЕ баг инференса — это фиксированный
структурный prior внутри самой архитектуры модели, действующий одинаково
на train и inference, так что явного краша/NaN он не даёт — модель просто
обучается поверх "перепутанной" топологии графа, что, вероятно, ограничивает
достижимую точность (baseline val_acc в `colab_glossa_01a` = 0.4770).

**Почему не чиним прямо здесь:** веса `GraphConv` уже натренированы под
ТЕКУЩУЮ (пусть и мисматчную) топологию. Подмена графа на корректный "на лету"
в fine-tune с малым числом эпох/данных не даст модели времени
переобучиться под новую топологию — это, по сути, задача для ПОЛНОГО
retrain с нуля на исправленном графе, отдельная инициатива, а не патч.

Если решите делать полный retrain с исправленным графом — вот сопоставление
для `load_skeleton_graph()`: рёбра нужно переписать в терминах
_TRAINING_-порядка (`nose=0, Rsho=1, Relb=2, Rwri=3, Lsho=4, Lelb=5, Lwri=6,
Rhip=7, Rkne=8, Rank=9, Lhip=10, Lkne=11, Lank=12, Reye=13, Leye=14, Rear=15,
Lear=16`), а не COCO-порядка, которым сейчас написаны `COCO_EDGES`."""
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
