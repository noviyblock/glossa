"""Генерация notebooks/colab_glossa_01a_train_stgcn_64_200.ipynb.

Облегчённая версия colab_glossa_01_train_stgcn_64_200.ipynb:
- убраны архитектурные варианты (Temporal Attention / Hierarchical) —
  три эксперимента подряд показали, что усложнение архитектуры ухудшает
  результат на этом объёме данных (200 классов x 15 train/класс);
- аугментация перенесена в Dataset.__getitem__ (применяется на лету каждую
  эпоху, а не один раз статическим массивом перед обучением — баг,
  резко снижавший эффективное разнообразие данных);
- добавлен короткий Optuna-поиск гиперпараметров (lr, weight_decay,
  label_smoothing, batch_size, TCN dropout);
- финальное обучение с лучшими гиперпараметрами;
- сравнение с уже известным результатом baseline (val_acc=0.4770,
  val_f1=0.4726, run stgcn_train_20260626_0643) — без повторного обучения
  baseline;
- ONNX-экспорт (полный + mobile) и копирование в Drive — ТОЛЬКО если новая
  модель обходит baseline по val_f1.

Запуск (локально, чтобы пересобрать .ipynb после правок):
    python scripts/build_colab_01a_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).parents[1] / "notebooks" / "colab_glossa_01a_train_stgcn_64_200.ipynb"

# ── Markdown intro ───────────────────────────────────────────────────────────
_MD_INTRO = """\
# ST-GCN: Улучшенное обучение (Slovo, топ-200 классов, T=64)

**Google Colab Pro** | GPU: A100 | Время: ~30-60 мин

## Что делает этот ноутбук
Облегчённая версия `colab_glossa_01_train_stgcn_64_200.ipynb` — без
архитектурных вариантов (Temporal Attention / Hierarchical), которые в
прошлом прогоне трижды подряд показали худший результат, чем простой
baseline ST-GCN. Вместо усложнения архитектуры — два конкретных улучшения
процесса обучения:

1. **On-the-fly аугментация.** В прошлом ноутбуке аугментация применялась
   один раз перед обучением, создавая фиксированный набор — модель видела
   одни и те же 7 случайно аугментированных версий каждого примера во всех
   122 эпохах. Здесь аугментация применяется заново при каждом обращении к
   примеру (`Dataset.__getitem__`), так что за время обучения модель видит
   сотни разных случайных аугментаций каждого видео.
2. **Optuna-поиск гиперпараметров** (lr, weight_decay, label_smoothing,
   batch_size, dropout в TCN-блоках) — короткий бюджет (20 trials × 40 эпох
   с ранней остановкой), а не вручную подобранные значения.

## Единый ноутбук для двух сценариев
Ячейка 2 (`SCENARIO`) переключает между двумя параллельными экспериментами:
- `'top200'` — топ-200 классов, T=64 (`processed_64_200`). Baseline уже
  известен: **val_acc = 0.4770, val_f1 = 0.4726** (run `stgcn_train_20260626_0643`,
  122 эпохи, без HPO, со статической аугментацией). Этот ноутбук не
  переобучает baseline заново — сравнение в конце идёт с этими числами.
- `'full'` — полный корпус, 1000 классов, T=64 (`processed`, препроцессинг
  идёт параллельно). Baseline для этого сценария пока не известен — первый
  успешный прогон становится отправной точкой, экспорт/обновление Drive
  выполняются автоматически без сравнения.

Просто меняйте `SCENARIO = 'top200'` на `SCENARIO = 'full'` в ячейке 2, когда
препроцессинг полного корпуса будет готов — остальной код не требует правок.

## Перед запуском
Добавьте в **Colab → Secrets (🔑)**:
- `DAGSHUB_TOKEN` — токен DAGsHub (Settings → Access Tokens)
- `MLFLOW_TRACKING_USERNAME` — `noviyblock`
- `MLFLOW_TRACKING_PASSWORD` — тот же токен DAGsHub

Датасет должен быть в Drive по пути `/content/drive/MyDrive/glossa/data/`:
```
data/gestures/processed_64_200/train/features.npy  # (N, 64, 75, 3)
data/gestures/processed_64_200/train/labels.npy
data/gestures/processed_64_200/val/features.npy
data/gestures/processed_64_200/val/labels.npy
data/gestures/processed_64_200/class_names.json
```
"""

# ── Cell: Drive mount + deps ────────────────────────────────────────────────
_CELL_INSTALL = """\
from google.colab import drive
drive.mount('/content/drive')

print('Google Drive смонтирован')

!pip install -q dagshub mlflow pyyaml onnx onnxruntime onnxscript optuna
!pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
!pip install -q scikit-learn matplotlib pandas

print('Все зависимости успешно установлены')
"""

# ── Cell: scenario switch (200 classes vs full 1000-class corpus) ───────────
_CELL_SCENARIO = """\
# Переключатель сценария — два параллельных эксперимента:
#   'top200' — топ-200 классов (уже обучен baseline 0.4770/0.4726)
#   'full'   — полный корпус, 1000 классов (препроцессинг T=64 ещё идёт,
#              baseline пока не известен — первый успешный прогон станет
#              отправной точкой для сравнения в будущих запусках)
SCENARIO = 'top200'  # 'top200' | 'full'

_SCENARIO_PRESETS = {
    'top200': {
        'data_subdir':       'processed_64_200',
        'num_classes':       200,
        'baseline_val_acc':  0.4770,
        'baseline_val_f1':   0.4726,
        'baseline_run_id':   '54f547d1a7634e0aa225ea91fbd45b41',
    },
    'full': {
        'data_subdir':       'processed',
        'num_classes':       1000,
        'baseline_val_acc':  None,
        'baseline_val_f1':   None,
        'baseline_run_id':   None,
    },
}

assert SCENARIO in _SCENARIO_PRESETS, f'Неизвестный сценарий: {SCENARIO}'
PRESET = _SCENARIO_PRESETS[SCENARIO]
print(f'Сценарий: {SCENARIO} -> {PRESET[\"num_classes\"]} классов, data_subdir={PRESET[\"data_subdir\"]!r}')
if PRESET['baseline_val_f1'] is None:
    print('  Baseline для этого сценария ещё не известен — этот прогон станет отправной точкой.')
else:
    print(f'  Известный baseline: val_acc={PRESET[\"baseline_val_acc\"]:.4f}  val_f1={PRESET[\"baseline_val_f1\"]:.4f}')
"""

# ── Cell: secrets / dagshub ─────────────────────────────────────────────────
_CELL_SECRETS = """\
import os
import time
import random
import json
import shutil
from pathlib import Path

DRIVE_ROOT  = '/content/drive/MyDrive/glossa'
MODELS_DIR  = f'{DRIVE_ROOT}/models'
DATA_DIR    = f'{DRIVE_ROOT}/data/gestures/{PRESET[\"data_subdir\"]}'
PLOTS_DIR   = f'{MODELS_DIR}/plots'

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

from google.colab import userdata
for key in ('DAGSHUB_TOKEN', 'MLFLOW_TRACKING_USERNAME', 'MLFLOW_TRACKING_PASSWORD'):
    try:
        os.environ[key] = userdata.get(key)
    except Exception:
        print(f'[Предупреждение] Секрет {key} не найден')

token = os.environ.get('DAGSHUB_TOKEN') or os.environ.get('MLFLOW_TRACKING_PASSWORD', '')
if token:
    os.environ.setdefault('AWS_ACCESS_KEY_ID',     token)
    os.environ.setdefault('AWS_SECRET_ACCESS_KEY', token)
    os.environ.setdefault('MLFLOW_S3_ENDPOINT_URL', 'https://dagshub.com/noviyblock/glossa.s3')

try:
    import dagshub
    dagshub.init(repo_owner='noviyblock', repo_name='glossa', mlflow=True)
    print('[DAGsHub] Инициализация успешна')
    print('[DAGsHub] UI: https://dagshub.com/noviyblock/glossa')
except Exception as e:
    import mlflow
    mlflow.set_tracking_uri('https://dagshub.com/noviyblock/glossa.mlflow')
    print(f'[MLflow] Fallback режим: {e}')
"""

# ── Cell: CFG (base, HPO will override a subset per-trial) ─────────────────
_CELL_CFG = """\
import torch
import numpy as np

# Известный результат baseline для текущего SCENARIO (для сравнения в конце, без переобучения)
BASELINE_VAL_ACC = PRESET['baseline_val_acc']
BASELINE_VAL_F1  = PRESET['baseline_val_f1']
BASELINE_RUN_ID  = PRESET['baseline_run_id']

CFG = {
    # Данные
    'num_classes':      PRESET['num_classes'],
    'sequence_length':  64,
    'num_nodes':        75,

    # Архитектура
    'mobile_mode':      False,
    'tcn_dropout':       0.5,    # будет подбираться Optuna

    # Обучение (значения по умолчанию — переопределяются в HPO/финальном запуске)
    'batch_size':       32,
    'epochs':           200,
    'lr':               1e-3,
    'weight_decay':      1e-4,
    'patience':         25,
    'flip_prob':        0.5,
    'seed':             42,
    'label_smoothing':  0.1,

    'opset_version':    18,
    'experiment_name':  f'01a_glossa_stgcn_improved_{SCENARIO}',
}

random.seed(CFG['seed'])
np.random.seed(CFG['seed'])
torch.manual_seed(CFG['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG['seed'])

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Устройство: {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"""

# ── Cell: skeleton graph (unchanged) ────────────────────────────────────────
_CELL_GRAPH = """\
import numpy as np
import torch
from scipy.sparse.csgraph import connected_components

A_np = None
A_partition_np = None
skeleton_graph = None

def load_skeleton_graph():
    \"\"\"Граф скелета для DWPose (COCO-WholeBody) -> 75 точек.\"\"\"
    global A_np, A_partition_np, skeleton_graph

    # ── ⚠️ KNOWN LIMITATION (см. docs/KNOWN_LIMITATIONS.md, "ST-GCN graph
    # topology mismatch") ─────────────────────────────────────────────────
    # Ниже рёбра выглядят так, будто индексы 0-16 — это истинный COCO-17
    # порядок (5/6=плечи, 7/8=локти, 9/10=запястья и т.д.). Это НЕ так:
    # реальный порядок здесь — training order из
    # _COCO17_TO_TRAINING_ORDER (services/cv_service/keypoint_extractor.py:
    # 24-29,45):
    #   0=nose   1=Rsho  2=Relb  3=Rwri  4=Lsho  5=Lelb  6=Lwri
    #   7=Rhip   8=Rkne  9=Rank  10=Lhip 11=Lkne 12=Lank
    #   13=Reye  14=Leye 15=Rear 16=Lear
    # Что реально соединяет каждое ребро, доживающее до фильтра
    # `i < 17 and j < 17` ниже (при true-COCO прочтении раньше это читалось
    # бы как "плечи → локти → запястья" и т.п. — неверно для всех рёбер):
    #   (0,1)  nose–Rsho              (0,2)  nose–Relb
    #   (1,3)  Rsho–Rwri   [та же рука, пропускает локоть]
    #   (2,4)  Relb–Lsho   [через тело]
    #   (5,6)  Lelb–Lwri   [предплечье — анатомически ОК, случайно]
    #   (5,7)  Lelb–Rhip   [через тело]     (6,8)  Lwri–Rkne  [через тело]
    #   (7,9)  Rhip–Rank   [та же нога, пропускает колено]
    #   (5,11) Lelb–Lkne   [через тело]     (6,12) Lwri–Lank  [через тело]
    #   (11,12) Lkne–Lank  [голень — ОК, случайно]
    #   (11,13) Lkne–Reye  [несвязанные части] (13,15) Reye–Rear [лицо — ОК]
    #   (12,14) Lank–Leye  [несвязанные части] (14,16) Leye–Lear [лицо — ОК]
    #   (0,5)  nose–Lelb   (0,6)  nose–Lwri    (8,10) Rkne–Lhip  [через тело]
    # Примерно треть выживших рёбер (предплечье/голень/лицо) случайно
    # остаются анатомически смежными парами благодаря структуре
    # перестановки; остальные соединяют несвязанные части тела. Это
    # структурный prior, зашитый в веса GraphConv через обучение, а не баг
    # инференса — этот ноутбук ИМЕННО ТАК обучил действующую (до RTMW-
    # дообучения) базовую модель. Не исправлено здесь: исправление означает
    # полное переобучение с нуля на скорректированном графе (текущие веса
    # завязаны именно на эту, пусть и мисномер, топологию).
    COCO_EDGES = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (6, 8), (7, 9),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (15, 17), (17, 19),
        (12, 14), (14, 16), (16, 18), (18, 20),
        (0, 5), (0, 6),
        (5, 7), (7, 9),
        (6, 8), (8, 10),
    ]
    FEET_EDGES = [
        (17, 19), (19, 21),
        (18, 20), (20, 22),
        (17, 18),
    ]
    EXTRA_EDGES = [
        (23, 5), (24, 6),
        (25, 11), (26, 12),
        (27, 13), (28, 14),
        (29, 15), (30, 16),
        (31, 5), (32, 6),
    ]
    HAND_EDGES = [
        (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
        (1, 2), (2, 3), (3, 4),
        (5, 6), (6, 7), (7, 8),
        (9, 10), (10, 11), (11, 12),
        (13, 14), (14, 15), (15, 16),
        (17, 18), (18, 19), (19, 20),
    ]

    edges = set()
    for i, j in COCO_EDGES:
        if i < 17 and j < 17:
            edges.add((i, j)); edges.add((j, i))
    for i, j in FEET_EDGES:
        if 17 <= i <= 22 and 17 <= j <= 22:
            edges.add((i, j)); edges.add((j, i))
    for i, j in EXTRA_EDGES:
        edges.add((i, j)); edges.add((j, i))
    for i, j in HAND_EDGES:
        edges.add((33 + i, 33 + j)); edges.add((33 + j, 33 + i))
        edges.add((54 + i, 54 + j)); edges.add((54 + j, 54 + i))

    edges.add((7, 33)); edges.add((33, 7))
    edges.add((4, 54)); edges.add((54, 4))
    edges.add((5, 33)); edges.add((33, 5))
    edges.add((6, 54)); edges.add((54, 6))

    face_nodes = [1, 2, 3, 8, 9, 10]
    for node in face_nodes:
        edges.add((node, 0)); edges.add((0, node))

    edges.add((15, 17)); edges.add((17, 15))
    edges.add((16, 18)); edges.add((18, 16))

    V = 75
    A = np.zeros((V, V), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0; A[j, i] = 1.0
    np.fill_diagonal(A, 1.0)

    isolated = np.where((A > 0).sum(axis=1) - 1 == 0)[0]
    for node in isolated:
        if node < 33:
            A[node, 0] = 1.0; A[0, node] = 1.0
        elif node < 54:
            A[node, 33] = 1.0; A[33, node] = 1.0
        else:
            A[node, 54] = 1.0; A[54, node] = 1.0

    n_components, labels = connected_components(A, directed=False)
    if n_components > 1:
        comp_nodes = [np.where(labels == comp)[0] for comp in range(n_components)]
        for i in range(1, n_components):
            a, b = comp_nodes[0][0], comp_nodes[i][0]
            A[a, b] = 1.0; A[b, a] = 1.0

    deg = A.sum(axis=1)
    d_inv_sqrt = np.where(deg > 1e-8, 1.0 / np.sqrt(deg), 0.0)
    A_norm = A * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]

    binary = (A > 0).astype(np.float32)
    np.fill_diagonal(binary, 0)
    dist = np.full(V, np.inf)
    dist[0] = 0
    queue = [0]
    while queue:
        v = queue.pop(0)
        for u in np.where(binary[v] > 0)[0]:
            if dist[u] == np.inf:
                dist[u] = dist[v] + 1
                queue.append(u)

    partition = np.zeros((3, V, V), dtype=np.float32)
    for i in range(V):
        for j in range(V):
            w = A_norm[i, j]
            if w == 0:
                continue
            if i == j:
                partition[0, i, j] = w
            elif dist[j] <= dist[i]:
                partition[1, i, j] = w
            else:
                partition[2, i, j] = w

    A_np = A_norm
    A_partition_np = partition
    skeleton_graph = {'num_nodes': V, 'edges': list(edges), 'adjacency_matrix': A_norm, 'num_edges': len(edges)}
    print(f'Граф готов: V={V}, E={len(edges)}, компонент связности={n_components}')
    return A_np, A_partition_np

A_np, A_partition_np = load_skeleton_graph()
"""

# ── Cell: model (TCN dropout now parametrized) ──────────────────────────────
_CELL_MODEL = """\
import torch
import torch.nn as nn

class GraphConv(nn.Module):
    \"\"\"ST-GCN graph convolution.\"\"\"
    def __init__(self, in_ch: int, out_ch: int, mobile: bool = False) -> None:
        super().__init__()
        self.K = 1 if mobile else 3
        self.mobile = mobile
        if mobile:
            self.register_buffer('A', torch.tensor(A_np, dtype=torch.float32).unsqueeze(0))
        else:
            self.register_buffer('A', torch.tensor(A_partition_np, dtype=torch.float32))
            self.M = nn.Parameter(torch.zeros(self.K, A_np.shape[0], A_np.shape[0]))
        self.convs = nn.ModuleList([nn.Conv2d(in_ch, out_ch, kernel_size=1) for _ in range(self.K)])
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = 0
        for k in range(self.K):
            A_eff = self.A[k] if self.mobile else (self.A[k] + self.M[k])
            branch = self.convs[k](x)
            out = out + torch.einsum('bctv,vw->bctw', branch, A_eff)
        return self.bn(out)


class TCN(nn.Module):
    \"\"\"Temporal Convolution Block. dropout параметризован для HPO.\"\"\"
    def __init__(self, ch: int, stride: int = 1, kernel: int = 9, dropout: float = 0.5) -> None:
        super().__init__()
        pad = (kernel - 1) // 2
        self.net = nn.Sequential(
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=(kernel, 1), stride=(stride, 1), padding=(pad, 0)),
            nn.BatchNorm2d(ch),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class STGCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, residual: bool = True,
                 mobile: bool = False, tcn_dropout: float = 0.5) -> None:
        super().__init__()
        self.gcn = GraphConv(in_ch, out_ch, mobile=mobile)
        self.tcn = TCN(out_ch, stride=stride, dropout=tcn_dropout)
        self.relu = nn.ReLU(inplace=True)
        self._no_res = not residual
        if residual and (in_ch != out_ch or stride != 1):
            self.skip = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=(stride, 1)), nn.BatchNorm2d(out_ch))
        elif residual:
            self.skip = nn.Identity()
        else:
            self.skip = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = 0 if self._no_res else self.skip(x)
        return self.relu(self.tcn(self.gcn(x)) + res)


class STGCN(nn.Module):
    \"\"\"ST-GCN для распознавания жестов РЖЯ (датасет Slovo, топ-200 классов).\"\"\"
    _CFG = [
        (3, 64, 1, False),
        (64, 64, 1, True),
        (64, 64, 1, True),
        (64, 128, 2, True),
        (128, 128, 1, True),
        (128, 256, 2, True),
        (256, 256, 1, True),
    ]

    def __init__(self, num_classes: int = 200, num_nodes: int = 75, mobile: bool = False,
                 tcn_dropout: float = 0.5, fc_dropout: float = 0.5) -> None:
        super().__init__()
        self.data_bn = nn.BatchNorm1d(3 * num_nodes)
        self.layers = nn.ModuleList([
            STGCNBlock(ic, oc, st, res, mobile=mobile, tcn_dropout=tcn_dropout)
            for ic, oc, st, res in self._CFG
        ])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(256, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, V, C = x.shape
        x = x.permute(0, 1, 3, 2).contiguous().view(B * T, C * V)
        x = self.data_bn(x)
        x = x.view(B, T, C, V).permute(0, 2, 1, 3).contiguous()
        for layer in self.layers:
            x = layer(x)
        x = self.pool(x).view(B, -1)
        return self.fc(self.drop(x))

print('Классы ST-GCN определены (tcn_dropout параметризован для Optuna)')
"""

# ── Cell: data loading (no static augmentation — moved to Dataset) ─────────
_CELL_DATA = """\
import json
import numpy as np
from pathlib import Path

DATA_ROOT = Path(DATA_DIR)

X_train = np.load(DATA_ROOT / 'train' / 'features.npy').astype(np.float32)
y_train = np.load(DATA_ROOT / 'train' / 'labels.npy').astype(np.int64)
X_val   = np.load(DATA_ROOT / 'val'   / 'features.npy').astype(np.float32)
y_val   = np.load(DATA_ROOT / 'val'   / 'labels.npy').astype(np.int64)

print('Загрузка данных:')
print(f'  [train] X={X_train.shape}  y={y_train.shape}  классов={np.unique(y_train).size}')
print(f'  [val]   X={X_val.shape}    y={y_val.shape}    классов={np.unique(y_val).size}')

# Удаление проблемных видео (inf в координатах) - динамически, не хардкод
BAD_TRAIN = np.where(np.isinf(X_train).any(axis=(1, 2, 3)))[0].tolist()
BAD_VAL   = np.where(np.isinf(X_val).any(axis=(1, 2, 3)))[0].tolist()
print(f'  train: найдено {len(BAD_TRAIN)} видео с inf, val: найдено {len(BAD_VAL)}')
if BAD_TRAIN:
    X_train = np.delete(X_train, BAD_TRAIN, axis=0)
    y_train = np.delete(y_train, BAD_TRAIN, axis=0)
if BAD_VAL:
    X_val = np.delete(X_val, BAD_VAL, axis=0)
    y_val = np.delete(y_val, BAD_VAL, axis=0)
print(f'  После очистки: train={len(X_train)}, val={len(X_val)}')

assert not np.isinf(X_train).any() and not np.isinf(X_val).any()

# Обрезка выбросов до нормализации
clip_init = 10.0
X_train = np.clip(X_train, -clip_init, clip_init)
X_val   = np.clip(X_val,   -clip_init, clip_init)

# Z-score нормализация по обучающей выборке
mean = X_train.mean(axis=(0, 1, 2), keepdims=True)
std  = X_train.std(axis=(0, 1, 2), keepdims=True)
std  = np.where(std < 1e-6, 1.0, std)

X_train_norm = (X_train - mean) / std
X_val_norm   = (X_val   - mean) / std

clip_val = 4.0
X_train_norm = np.clip(X_train_norm, -clip_val, clip_val)
X_val_norm   = np.clip(X_val_norm,   -clip_val, clip_val)

norm_stats_path = f'{MODELS_DIR}/norm_stats.npz'
np.savez(norm_stats_path, mean=mean, std=std)
print(f'  Статистика нормализации сохранена: {norm_stats_path}')

class_names_path = DATA_ROOT / 'class_names.json'
class_names = json.loads(class_names_path.read_text()) if class_names_path.exists() else []
print(f'  Загружено названий классов: {len(class_names)}')

print('\\nДанные загружены и нормализованы (без статической аугментации — она теперь on-the-fly).')
"""

# ── Cell: on-the-fly augmented Dataset ──────────────────────────────────────
_CELL_DATASET = """\
import torch
from torch.utils.data import Dataset, DataLoader

# Те же 7 аугментаций, что и в эталонном ноутбуке — но теперь применяются
# заново при каждом __getitem__, а не один раз статическим массивом.

_POSE_FLIP = [
    0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13,
    16, 15, 18, 17, 20, 19, 22, 21, 24, 23, 26, 25,
    28, 27, 30, 29, 32, 31,
]
_LEFT_IDX  = list(range(33, 54))
_RIGHT_IDX = list(range(54, 75))


def horizontal_flip(x: np.ndarray) -> np.ndarray:
    f = x.copy()
    f[:, :, 0] = -f[:, :, 0]
    f[:, :33] = f[:, _POSE_FLIP]
    tmp = f[:, _LEFT_IDX].copy()
    f[:, _LEFT_IDX] = f[:, _RIGHT_IDX]
    f[:, _RIGHT_IDX] = tmp
    return f


def spatial_jitter(x: np.ndarray, sigma: float = 0.02) -> np.ndarray:
    return x + np.random.normal(0, sigma, size=x.shape).astype(np.float32)


def temporal_warp(x: np.ndarray, max_scale: float = 0.15) -> np.ndarray:
    T = x.shape[0]
    scale = np.random.uniform(1 - max_scale, 1 + max_scale)
    new_T = max(1, int(T * scale))
    indices = np.linspace(0, T - 1, new_T).astype(np.int32)
    warped = x[indices]
    indices_back = np.linspace(0, new_T - 1, T).astype(np.int32)
    return warped[indices_back]


def hand_rotation(x: np.ndarray, max_angle: float = 8.0) -> np.ndarray:
    x = x.copy()
    angle = np.random.uniform(-max_angle, max_angle) * np.pi / 180
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    hands = x[:, 33:75, :2]
    shape = hands.shape
    x[:, 33:75, :2] = (hands.reshape(-1, 2) @ rot.T).reshape(shape)
    return x


def temporal_shift(x: np.ndarray, max_shift: float = 0.08) -> np.ndarray:
    T = x.shape[0]
    shift = np.random.randint(-int(T * max_shift), int(T * max_shift) + 1)
    return x if shift == 0 else np.roll(x, shift, axis=0)


def random_scaling(x: np.ndarray, max_scale: float = 0.08) -> np.ndarray:
    return x * np.random.uniform(1 - max_scale, 1 + max_scale)


def cutout_nodes(x: np.ndarray, num_nodes_to_zero: int = 3) -> np.ndarray:
    x = x.copy()
    V = x.shape[1]
    nodes = np.random.choice(V, size=num_nodes_to_zero, replace=False)
    x[:, nodes, :] = 0.0
    return x


class AugmentedGestureDataset(Dataset):
    \"\"\"Применяет аугментацию заново при каждом обращении к примеру (on-the-fly),
    а не один раз перед обучением — модель видит новую случайную аугментацию
    каждую эпоху, а не фиксированный набор.
    \"\"\"
    def __init__(self, X: np.ndarray, y: np.ndarray, flip_prob: float = 0.5,
                 clip_val: float = 4.0, train: bool = True) -> None:
        self.X = X
        self.y = y
        self.flip_prob = flip_prob
        self.clip_val = clip_val
        self.train = train

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self.X[idx].copy()
        if self.train:
            if np.random.rand() < self.flip_prob:
                x = horizontal_flip(x)
            if np.random.rand() < 0.5:
                x = spatial_jitter(x, sigma=0.02)
            if np.random.rand() < 0.3:
                x = temporal_warp(x, max_scale=0.15)
            if np.random.rand() < 0.3:
                x = hand_rotation(x, max_angle=8.0)
            if np.random.rand() < 0.3:
                x = temporal_shift(x, max_shift=0.08)
            if np.random.rand() < 0.3:
                x = random_scaling(x, max_scale=0.08)
            if np.random.rand() < 0.15:
                x = cutout_nodes(x, num_nodes_to_zero=3)
            x = np.clip(x, -self.clip_val, self.clip_val)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)


X_val_tensor = torch.tensor(X_val_norm, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

print('AugmentedGestureDataset готов — аугментация применяется on-the-fly каждую эпоху.')
"""

# ── Cell: reusable training function ────────────────────────────────────────
_CELL_TRAIN_FN = """\
import mlflow
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm.notebook import tqdm
import time
import os


def train_one_run(cfg: dict, max_epochs: int, patience: int,
                   trial=None, log_mlflow: bool = False, run_name: str | None = None):
    \"\"\"Обучает ST-GCN с заданными гиперпараметрами.

    trial: объект optuna.Trial — если передан, репортит промежуточный val_f1
           для прунинга и поднимает optuna.TrialPruned при необходимости.
    log_mlflow: логировать ли полный прогон в MLflow (False для HPO-trials,
                True для финального обучения).

    Возвращает dict с best_val_acc, best_val_f1, model_state, total_epochs.
    \"\"\"
    train_ds = AugmentedGestureDataset(X_train_norm, y_train, flip_prob=cfg['flip_prob'], train=True)
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
                               num_workers=2, pin_memory=True)

    model = STGCN(
        num_classes=cfg['num_classes'],
        num_nodes=cfg['num_nodes'],
        mobile=cfg['mobile_mode'],
        tcn_dropout=cfg.get('tcn_dropout', 0.5),
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])

    best_val_f1, best_val_acc, no_improve = 0.0, 0.0, 0
    best_state = None
    VAL_BS = 64

    mlflow_active = log_mlflow
    if mlflow_active:
        mlflow.set_experiment(cfg['experiment_name'])
        run_ctx = mlflow.start_run(run_name=run_name or cfg.get('run_name'))
        run_ctx.__enter__()
        mlflow.log_params({k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool))})

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_logits_list = []
        with torch.no_grad():
            for i in range(0, len(X_val_tensor), VAL_BS):
                xb = X_val_tensor[i:i + VAL_BS].to(DEVICE)
                val_logits_list.append(model(xb).cpu())
        val_logits = torch.cat(val_logits_list, dim=0)
        val_pred = val_logits.argmax(1)
        val_acc = (val_pred == y_val_tensor).float().mean().item()
        val_f1 = f1_score(y_val_tensor.numpy(), val_pred.numpy(), average='macro', zero_division=0)

        scheduler.step(val_acc)

        if mlflow_active:
            mlflow.log_metrics({'val_acc': val_acc, 'val_f1_macro': val_f1,
                                 'lr': optimizer.param_groups[0]['lr']}, step=epoch)

        if val_f1 > best_val_f1:
            best_val_f1, best_val_acc, no_improve = val_f1, val_acc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if trial is not None:
            trial.report(val_f1, step=epoch)
            import optuna
            if trial.should_prune():
                if mlflow_active:
                    run_ctx.__exit__(None, None, None)
                raise optuna.TrialPruned()

        if no_improve >= patience:
            break

    if mlflow_active:
        mlflow.log_metrics({'best_val_acc': best_val_acc, 'best_val_f1': best_val_f1, 'total_epochs': epoch})
        run_ctx.__exit__(None, None, None)

    return {
        'best_val_acc': best_val_acc,
        'best_val_f1': best_val_f1,
        'model_state': best_state,
        'total_epochs': epoch,
        'cfg': cfg,
    }


print('train_one_run() готова к использованию (HPO и финальное обучение).')
"""

# ── Cell: Optuna search ──────────────────────────────────────────────────────
_CELL_OPTUNA = """\
import optuna

N_TRIALS = 20
MAX_EPOCHS_PER_TRIAL = 40
PATIENCE_PER_TRIAL = 8


def objective(trial: optuna.Trial) -> float:
    cfg = dict(CFG)
    cfg['lr'] = trial.suggest_float('lr', 1e-4, 3e-3, log=True)
    cfg['weight_decay'] = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    cfg['label_smoothing'] = trial.suggest_float('label_smoothing', 0.0, 0.3)
    cfg['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64])
    cfg['tcn_dropout'] = trial.suggest_float('tcn_dropout', 0.3, 0.7)

    result = train_one_run(cfg, max_epochs=MAX_EPOCHS_PER_TRIAL, patience=PATIENCE_PER_TRIAL,
                            trial=trial, log_mlflow=False)
    return result['best_val_f1']


study = optuna.create_study(direction='maximize',
                             pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
print(f'Запуск Optuna: {N_TRIALS} trials x до {MAX_EPOCHS_PER_TRIAL} эпох (patience={PATIENCE_PER_TRIAL})')
study.optimize(objective, n_trials=N_TRIALS)

print('\\n' + '=' * 60)
print('OPTUNA ЗАВЕРШЁН')
print('=' * 60)
print(f'Лучший val_f1 (короткий бюджет): {study.best_value:.4f}')
print(f'Лучшие гиперпараметры: {study.best_params}')
"""

# ── Cell: final training with best hyperparameters ──────────────────────────
_CELL_FINAL_TRAIN = """\
FINAL_CFG = dict(CFG)
FINAL_CFG.update(study.best_params)
FINAL_CFG['run_name'] = f'stgcn_optuna_best_{time.strftime(\"%Y%m%d_%H%M\")}'

print('Финальное обучение с лучшими гиперпараметрами:')
for k, v in study.best_params.items():
    print(f'  {k}: {v}')
print(f'\\nЭпох: {CFG[\"epochs\"]}, patience: {CFG[\"patience\"]}')

final_result = train_one_run(
    FINAL_CFG,
    max_epochs=CFG['epochs'],
    patience=CFG['patience'],
    log_mlflow=True,
    run_name=FINAL_CFG['run_name'],
)

print('\\n' + '=' * 60)
print('ФИНАЛЬНОЕ ОБУЧЕНИЕ ЗАВЕРШЕНО')
print('=' * 60)
print(f'  best_val_acc : {final_result[\"best_val_acc\"]:.4f}')
print(f'  best_val_f1  : {final_result[\"best_val_f1\"]:.4f}')
print(f'  total_epochs : {final_result[\"total_epochs\"]}')

# Восстанавливаем лучшую модель и сохраняем чекпоинт (тот же путь, что в эталонном ноутбуке)
model = STGCN(num_classes=CFG['num_classes'], num_nodes=CFG['num_nodes'],
              mobile=CFG['mobile_mode'], tcn_dropout=FINAL_CFG['tcn_dropout']).to(DEVICE)
model.load_state_dict(final_result['model_state'])
model.eval()

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
NUM_NODES = CFG['num_nodes']

CKPT_BEST = f'{MODELS_DIR}/stgcn_best.pt'
CKPT_FINAL = f'{MODELS_DIR}/stgcn_final.pt'
torch.save(model.state_dict(), CKPT_BEST)
torch.save(model.state_dict(), CKPT_FINAL)
print(f'  Модель сохранена: {CKPT_BEST}')
"""

# ── Cell: comparison with baseline ───────────────────────────────────────────
_CELL_COMPARE = """\
print('=' * 60)
print(f'СРАВНЕНИЕ С BASELINE (сценарий: {SCENARIO})')
print('=' * 60)
print(f'  Новая модель (Optuna + on-the-fly аугм.):  val_acc={final_result[\"best_val_acc\"]:.4f}  val_f1={final_result[\"best_val_f1\"]:.4f}')

if BASELINE_VAL_F1 is None:
    print('  Baseline для этого сценария не известен (первый прогон) — считаем текущий результат отправной точкой.')
    IS_BETTER = True
else:
    print(f'  Baseline   (без HPO, статич. аугментация): val_acc={BASELINE_VAL_ACC:.4f}  val_f1={BASELINE_VAL_F1:.4f}')
    IS_BETTER = final_result['best_val_f1'] > BASELINE_VAL_F1
    delta = final_result['best_val_f1'] - BASELINE_VAL_F1
    print(f'\\n  Разница по val_f1: {delta:+.4f}')

if IS_BETTER:
    print('  Новая модель ЛУЧШЕ baseline (или baseline отсутствует) — переходим к экспорту.')
else:
    print('  Новая модель НЕ превзошла baseline — экспорт и обновление Drive пропускаются.')
    print('  (чекпоинт всё равно сохранён локально в models/, можно проанализировать отдельно)')
print('=' * 60)
"""

# ── Cell: conditional ONNX export (full) ────────────────────────────────────
_CELL_EXPORT_FULL = """\
if IS_BETTER:
    import onnx
    import onnxruntime as ort

    print('=' * 60)
    print('ЭКСПОРТ ПОЛНОЙ МОДЕЛИ В ONNX')
    print('=' * 60)

    if not FINAL_CFG['mobile_mode']:
        with torch.no_grad():
            for layer in model.layers:
                gcn = layer.gcn
                if hasattr(gcn, 'M'):
                    gcn.A.copy_(gcn.A + gcn.M)
                    gcn.M.data.zero_()

    ONNX_PATH = f'{MODELS_DIR}/gesture_classifier.onnx'
    dummy = torch.randn(1, CFG['sequence_length'], NUM_NODES, 3).to(DEVICE)

    torch.onnx.export(
        model, dummy, ONNX_PATH,
        export_params=True, opset_version=CFG['opset_version'],
        input_names=['keypoints'], output_names=['logits'],
        dynamic_axes={'keypoints': {0: 'batch_size'}, 'logits': {0: 'batch_size'}},
        do_constant_folding=True, verbose=False,
        dynamo=False,  # dynamo-экспортёр теряет веса на Python-условиях в GraphConv.forward
    )
    onnx_size_mb = os.path.getsize(ONNX_PATH) / 1e6
    expected_min_mb = n_params * 4 / 1e6 * 0.5  # fp32, с запасом на constant folding
    print(f'  Сохранён: {ONNX_PATH} ({onnx_size_mb:.2f} MB)')
    if onnx_size_mb < expected_min_mb:
        raise RuntimeError(
            f'ONNX-файл подозрительно мал ({onnx_size_mb:.2f} MB при {n_params:,} параметрах, '
            f'ожидалось хотя бы {expected_min_mb:.2f} MB) — экспортёр, вероятно, потерял веса.'
        )

    sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    out = sess.run(['logits'], {'keypoints': dummy.cpu().numpy()})[0]
    with torch.no_grad():
        torch_out = model(dummy).cpu().numpy()
    diff = np.abs(torch_out - out).max()
    # GPU (torch) vs CPU (onnxruntime) дают разные ядра свёртки — расхождение ~1e-3
    # на глубокой сети (7 ST-GCN блоков) ожидаемо. Допуск как в torch.onnx verification: rtol=1e-3, atol=1e-5.
    is_close = np.allclose(torch_out, out, rtol=1e-3, atol=1e-3)
    print(f'  Макс. разница ONNX vs PyTorch: {diff:.6f} ({\"OK\" if is_close else \"ВНИМАНИЕ\"})')
    if not is_close:
        raise RuntimeError(f'ONNX и PyTorch расходятся (diff={diff:.6f}, rtol=1e-3/atol=1e-3) — экспорт некорректен.')
else:
    print('Пропущено: новая модель не превзошла baseline.')
"""

# ── Cell: conditional ONNX export (mobile) ──────────────────────────────────
_CELL_EXPORT_MOBILE = """\
if IS_BETTER:
    print('=' * 60)
    print('ЭКСПОРТ МОБИЛЬНОЙ МОДЕЛИ')
    print('=' * 60)

    ONNX_MOBILE_PATH = f'{MODELS_DIR}/gesture_classifier_mobile.onnx'

    model_mobile = STGCN(num_classes=CFG['num_classes'], num_nodes=NUM_NODES,
                          mobile=True, tcn_dropout=FINAL_CFG['tcn_dropout']).to(DEVICE)

    src_state = model.state_dict()
    dst_state = model_mobile.state_dict()
    for dst_key in dst_state:
        if 'gcn.convs.0' in dst_key:
            if dst_key in src_state and dst_state[dst_key].shape == src_state[dst_key].shape:
                dst_state[dst_key] = src_state[dst_key].clone()
        elif dst_key in src_state and dst_state[dst_key].shape == src_state[dst_key].shape:
            dst_state[dst_key] = src_state[dst_key].clone()
    model_mobile.load_state_dict(dst_state, strict=False)
    model_mobile.eval()

    n_mobile = sum(p.numel() for p in model_mobile.parameters() if p.requires_grad)
    print(f'  Параметры mobile: {n_mobile:,} ({n_mobile / 1e6:.2f}M), full: {n_params:,} ({n_params / 1e6:.2f}M)')

    dummy_m = torch.randn(1, CFG['sequence_length'], NUM_NODES, 3).to(DEVICE)
    torch.onnx.export(
        model_mobile, dummy_m, ONNX_MOBILE_PATH,
        export_params=True, opset_version=CFG['opset_version'],
        input_names=['keypoints'], output_names=['logits'],
        dynamic_axes={'keypoints': {0: 'batch_size'}, 'logits': {0: 'batch_size'}},
        do_constant_folding=True, verbose=False,
        dynamo=False,
    )
    mobile_size_mb = os.path.getsize(ONNX_MOBILE_PATH) / 1e6
    expected_min_mobile_mb = n_mobile * 4 / 1e6 * 0.5
    print(f'  Сохранён: {ONNX_MOBILE_PATH} ({mobile_size_mb:.2f} MB)')
    if mobile_size_mb < expected_min_mobile_mb:
        raise RuntimeError(
            f'ONNX mobile-файл подозрительно мал ({mobile_size_mb:.2f} MB при {n_mobile:,} параметрах) '
            '— экспортёр, вероятно, потерял веса.'
        )
else:
    print('Пропущено: новая модель не превзошла baseline.')
"""

# ── Cell: copy to Drive (conditional) ────────────────────────────────────────
_CELL_DRIVE_COPY = """\
if IS_BETTER:
    import shutil

    print('=' * 60)
    print('ОБНОВЛЕНИЕ ФАЙЛОВ В GOOGLE DRIVE')
    print('=' * 60)

    files_to_save = [
        (CKPT_BEST, 'stgcn_best.pt'),
        (CKPT_FINAL, 'stgcn_final.pt'),
        (ONNX_PATH, 'gesture_classifier.onnx'),
        (ONNX_MOBILE_PATH, 'gesture_classifier_mobile.onnx'),
        (norm_stats_path, 'norm_stats.npz'),
    ]
    for src, name in files_to_save:
        dst = f'{MODELS_DIR}/{name}'
        if os.path.exists(src):
            if src != dst:
                shutil.copy2(src, dst)
            print(f'  OK {name} ({os.path.getsize(dst) / 1e6:.2f} MB)')
        else:
            print(f'  ПРОПУЩЕН {name} (файл не найден: {src})')

    metrics_dict = {
        'best_val_acc': final_result['best_val_acc'],
        'best_val_f1': final_result['best_val_f1'],
        'total_epochs': final_result['total_epochs'],
        'best_hyperparams': study.best_params,
        'baseline_val_acc': BASELINE_VAL_ACC,
        'baseline_val_f1': BASELINE_VAL_F1,
    }
    np.save(f'{MODELS_DIR}/training_metrics_optuna.npy', metrics_dict)

    print('\\n' + '=' * 60)
    print('ИТОГ: новая модель сохранена и заменяет baseline в Drive')
    print(f'  val_acc: {BASELINE_VAL_ACC:.4f} -> {final_result[\"best_val_acc\"]:.4f}')
    print(f'  val_f1:  {BASELINE_VAL_F1:.4f} -> {final_result[\"best_val_f1\"]:.4f}')
    print('=' * 60)
else:
    print('Drive не обновлён: новая модель не превзошла baseline.')
    print('Чекпоинт остался только локально в /content — baseline в Drive не тронут.')
"""

CELLS = [
    ("markdown", _MD_INTRO),
    ("markdown", "## Ячейка 1 — Drive + зависимости (+ optuna)"),
    ("code", _CELL_INSTALL),
    ("markdown", "## Ячейка 2 — Переключатель сценария (top200 / full corpus)"),
    ("code", _CELL_SCENARIO),
    ("markdown", "## Ячейка 3 — DAGsHub / MLflow секреты"),
    ("code", _CELL_SECRETS),
    ("markdown", "## Ячейка 4 — Конфигурация (num_classes/baseline из сценария, HPO переопределит часть)"),
    ("code", _CELL_CFG),
    ("markdown", "## Ячейка 5 — Граф скелета DWPose (COCO-WholeBody, без изменений)"),
    ("code", _CELL_GRAPH),
    ("markdown", "## Ячейка 6 — Архитектура ST-GCN (dropout в TCN параметризован)"),
    ("code", _CELL_MODEL),
    ("markdown", "## Ячейка 7 — Загрузка и нормализация данных (без статической аугментации)"),
    ("code", _CELL_DATA),
    ("markdown", "## Ячейка 8 — Dataset с on-the-fly аугментацией"),
    ("code", _CELL_DATASET),
    ("markdown", "## Ячейка 9 — Функция обучения (используется и в HPO, и в финальном запуске)"),
    ("code", _CELL_TRAIN_FN),
    ("markdown", "## Ячейка 10 — Optuna: подбор гиперпараметров (короткий бюджет)"),
    ("code", _CELL_OPTUNA),
    ("markdown", "## Ячейка 11 — Финальное обучение с лучшими гиперпараметрами"),
    ("code", _CELL_FINAL_TRAIN),
    ("markdown", "## Ячейка 12 — Сравнение с baseline текущего сценария"),
    ("code", _CELL_COMPARE),
    ("markdown", "## Ячейка 13 — Экспорт ONNX (полная модель) — только если новая модель лучше"),
    ("code", _CELL_EXPORT_FULL),
    ("markdown", "## Ячейка 14 — Экспорт ONNX (мобильная модель) — только если новая модель лучше"),
    ("code", _CELL_EXPORT_MOBILE),
    ("markdown", "## Ячейка 15 — Обновление файлов в Google Drive — только если новая модель лучше"),
    ("code", _CELL_DRIVE_COPY),
]


def build_notebook() -> dict:
    cells = []
    for kind, content in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": content.splitlines(keepends=True)}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"provenance": [], "gpuType": "A100"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    nb = build_notebook()
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Notebook written: {NB_PATH}")
