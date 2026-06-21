"""Генерация ноутбука experiments/diploma_diagrams.ipynb.

Ноутбук рисует схемы, которых не хватает в дипломе (Приложение А и
раздел 2.2): архитектура микросервисов, потоки данных, конвейер DWPose,
алгоритм скользящего окна. Использует только matplotlib — без graphviz
и сетевых зависимостей.

Запуск:
    python scripts/build_diploma_diagrams_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).parents[1] / "experiments" / "diploma_diagrams.ipynb"

_SETUP = '''\
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MplPath
from pathlib import Path
import os

for _root in [Path.cwd(), Path.cwd().parent]:
    if (_root / "dvc.yaml").exists():
        PROJECT_ROOT = _root
        break
else:
    PROJECT_ROOT = Path.cwd()

OUT_DIR = PROJECT_ROOT / "models" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Диаграммы будут сохранены в: {OUT_DIR}")

def box(ax, xy, w, h, text, fc="#dbeafe", ec="#1d4ed8", fontsize=9.5):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                        fc=fc, ec=ec, lw=1.4)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, wrap=True)
    return (x, y, w, h)

def arrow(ax, p1, p2, text="", color="#374151", style="-|>", curve=0.0, fontsize=8.5):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, color=color, lw=1.3,
                         connectionstyle=f"arc3,rad={curve}",
                         mutation_scale=14, shrinkA=2, shrinkB=2)
    ax.add_patch(a)
    if text:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 + 0.15
        ax.text(mx, my, text, ha="center", va="bottom", fontsize=fontsize, color=color)
'''

_CELL_ARCH = '''\
fig, ax = plt.subplots(figsize=(11, 7))
ax.set_xlim(0, 11)
ax.set_ylim(0, 8)
ax.axis("off")
ax.set_title("Архитектура микросервисов Glossa (docker-compose)", fontsize=13, pad=12)

client = box(ax, (0.3, 6.3), 2.0, 1.0, "Клиент\\n(Flutter Web/Mobile)", fc="#fef3c7", ec="#b45309")
gw     = box(ax, (4.5, 6.3), 2.4, 1.0, "API Gateway\\n:8000\\nFastAPI + WS", fc="#dbeafe", ec="#1d4ed8")

cv  = box(ax, (0.3, 4.3), 2.2, 1.1, "CV Service\\n:8001\\nDWPose + ST-GCN", fc="#dcfce7", ec="#15803d")
asr = box(ax, (2.9, 4.3), 2.2, 1.1, "ASR Service\\n:8002\\nfaster-whisper", fc="#dcfce7", ec="#15803d")
nlp = box(ax, (5.5, 4.3), 2.2, 1.1, "NLP Service\\n:8003\\nQwen2-1.5B LoRA", fc="#dcfce7", ec="#15803d")
tts = box(ax, (8.1, 4.3), 2.2, 1.1, "TTS Service\\n:8004\\nSilero v4", fc="#dcfce7", ec="#15803d")

redis = box(ax, (4.5, 2.3), 2.4, 1.0, "Redis :6379\\nStreams + sessions", fc="#fee2e2", ec="#b91c1c")
mlflow = box(ax, (0.3, 2.3), 2.2, 1.0, "MLflow :5000\\n(SQLite)", fc="#ede9fe", ec="#6d28d9")
mon = box(ax, (8.1, 2.3), 2.2, 1.0, "Prometheus :9090\\nGrafana :3001", fc="#ede9fe", ec="#6d28d9")
qdrant = box(ax, (5.6, 0.3), 2.2, 1.0, "Qdrant :6333\\n(опционально, RAG)", fc="#f3f4f6", ec="#6b7280")

arrow(ax, (1.3, 6.3), (4.7, 7.0), "WebSocket /\\nREST")
arrow(ax, (5.7, 6.3), (1.4, 5.4), "video_frame")
arrow(ax, (5.7, 6.3), (4.0, 5.4), "audio_chunk")
arrow(ax, (6.7, 6.3), (6.6, 5.4), "translate")
arrow(ax, (7.7, 6.3), (9.2, 5.4), "synthesize")
arrow(ax, (1.4, 4.3), (5.0, 3.3), curve=-0.15)
arrow(ax, (4.0, 4.3), (5.2, 3.3), curve=-0.1)
arrow(ax, (6.6, 4.3), (5.9, 3.3), curve=0.1)
arrow(ax, (9.2, 4.3), (6.8, 3.3), curve=0.15)
arrow(ax, (6.6, 4.3), (6.6, 1.3), "RAG-запрос\\n(если включён)", color="#9ca3af")

fig.tight_layout()
out = OUT_DIR / "architecture_diagram.png"
fig.savefig(out, dpi=180)
print("Сохранено:", out)
plt.show()
'''

_CELL_FLOW_RSL = '''\
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.set_xlim(0, 11)
ax.set_ylim(0, 2.4)
ax.axis("off")
ax.set_title("Поток данных РЖЯ → речь", fontsize=13, pad=10)

steps = [
    ("Webcam\\n(кадры)", "#fef3c7", "#b45309"),
    ("API Gateway\\n(WS)", "#dbeafe", "#1d4ed8"),
    ("CV Service\\nDWPose+ST-GCN", "#dcfce7", "#15803d"),
    ("Redis\\ncv:results", "#fee2e2", "#b91c1c"),
    ("API Gateway", "#dbeafe", "#1d4ed8"),
    ("NLP Service\\n(глоссы→текст)", "#dcfce7", "#15803d"),
    ("TTS Service\\n(текст→аудио)", "#dcfce7", "#15803d"),
    ("Client\\n(аудио)", "#fef3c7", "#b45309"),
]
w, gap = 1.15, 0.27
x = 0.2
for label, fc, ec in steps:
    box(ax, (x, 0.7), w, 1.0, label, fc=fc, ec=ec, fontsize=8.3)
    x += w + gap
x = 0.2
for _ in range(len(steps) - 1):
    arrow(ax, (x + w, 1.2), (x + w + gap, 1.2))
    x += w + gap

fig.tight_layout()
out = OUT_DIR / "dataflow_rsl_to_text.png"
fig.savefig(out, dpi=180)
print("Сохранено:", out)
plt.show()
'''

_CELL_FLOW_TTR = '''\
fig, ax = plt.subplots(figsize=(9.5, 3.6))
ax.set_xlim(0, 9.5)
ax.set_ylim(0, 2.4)
ax.axis("off")
ax.set_title("Поток данных речь → РЖЯ", fontsize=13, pad=10)

steps = [
    ("Microphone\\n(аудио)", "#fef3c7", "#b45309"),
    ("API Gateway\\n(WS)", "#dbeafe", "#1d4ed8"),
    ("ASR Service\\nfaster-whisper", "#dcfce7", "#15803d"),
    ("API Gateway", "#dbeafe", "#1d4ed8"),
    ("NLP Service\\n(текст→глоссы)", "#dcfce7", "#15803d"),
    ("Client\\n(глоссы)", "#fef3c7", "#b45309"),
]
w, gap = 1.3, 0.3
x = 0.2
for label, fc, ec in steps:
    box(ax, (x, 0.7), w, 1.0, label, fc=fc, ec=ec, fontsize=8.5)
    x += w + gap
x = 0.2
for _ in range(len(steps) - 1):
    arrow(ax, (x + w, 1.2), (x + w + gap, 1.2))
    x += w + gap

fig.tight_layout()
out = OUT_DIR / "dataflow_text_to_rsl.png"
fig.savefig(out, dpi=180)
print("Сохранено:", out)
plt.show()
'''

_CELL_DWPOSE = '''\
fig, ax = plt.subplots(figsize=(10.5, 3.4))
ax.set_xlim(0, 10.5)
ax.set_ylim(0, 2.6)
ax.axis("off")
ax.set_title("Конвейер DWPose: от кадра до 75-точечного скелета", fontsize=13, pad=10)

box(ax, (0.2, 0.8), 1.7, 1.0, "Видеокадр\\n(BGR)", fc="#fef3c7", ec="#b45309")
box(ax, (2.3, 0.8), 1.9, 1.0, "YOLOX\\nдетектор\\n640×640", fc="#dbeafe", ec="#1d4ed8")
box(ax, (4.6, 0.8), 1.9, 1.0, "RTMPose\\n(top-down)\\n288×384", fc="#dcfce7", ec="#15803d")
box(ax, (6.9, 0.8), 2.0, 1.0, "133 точки\\nCOCO-WholeBody", fc="#ede9fe", ec="#6d28d9")
box(ax, (9.1, 0.8), 1.2, 1.0, "75 точек\\n(remap)", fc="#fee2e2", ec="#b91c1c", fontsize=8.5)

for x1, x2 in [(1.9, 2.3), (4.2, 4.6), (6.5, 6.9), (8.9, 9.1)]:
    arrow(ax, (x1, 1.3), (x2, 1.3))

ax.text(5.25, 2.3, "bbox человека → афинный crop", ha="center", fontsize=8.5, color="#374151")
ax.text(9.7, 1.95, "0–16 body, 17–22 feet,\\n23–32 extra, 33–53 LH,\\n54–74 RH",
        ha="center", fontsize=7.3, color="#374151")

fig.tight_layout()
out = OUT_DIR / "dwpose_pipeline.png"
fig.savefig(out, dpi=180)
print("Сохранено:", out)
plt.show()
'''

_CELL_WINDOW = '''\
import numpy as np

fig, ax = plt.subplots(figsize=(10, 3.2))
n_frames = 70
W, S = 32, 15
ax.set_xlim(-1, n_frames + 1)
ax.set_ylim(-0.5, 4.5)
ax.set_xlabel("Номер кадра видеопотока")
ax.set_yticks([])
ax.set_title(f"Алгоритм скользящего окна (W={W}, S={S})", fontsize=13, pad=10)

starts = list(range(0, n_frames - W, S))[:4]
colors = ["#1d4ed8", "#15803d", "#b45309", "#b91c1c"]
for i, (start, color) in enumerate(zip(starts, colors)):
    y = 3.5 - i * 1.0
    ax.add_patch(plt.Rectangle((start, y - 0.3), W, 0.6, fc=color, ec="none", alpha=0.35))
    ax.plot([start, start + W], [y, y], color=color, lw=2)
    ax.text(start + W + 1, y, f"окно {i+1}: [{start}, {start+W})", va="center",
            fontsize=8.5, color=color)

ax.plot(range(n_frames), [-0.2] * n_frames, color="#9ca3af", lw=1)
ax.text(n_frames / 2, -0.45, "поток кадров (25 FPS)", ha="center", fontsize=8.5, color="#6b7280")

fig.tight_layout()
out = OUT_DIR / "sliding_window_diagram.png"
fig.savefig(out, dpi=180)
print("Сохранено:", out)
plt.show()
'''

_CELL_MLOPS = '''\
fig, ax = plt.subplots(figsize=(11, 3.2))
ax.set_xlim(0, 11)
ax.set_ylim(0, 2.2)
ax.axis("off")
ax.set_title("DVC-пайплайн (dvc.yaml): стадии предобработки и обучения", fontsize=13, pad=10)

stages = [
    "data/raw\\n(Slovo)",
    "preprocess\\n(DWPose extract)",
    "train_gesture\\n(ST-GCN, 4 варианта)",
    "export_gesture\\n_classifier (ONNX)",
    "eval_gesture\\n(метрики)",
    "MLflow / DagsHub\\n(трекинг)",
]
w, gap = 1.55, 0.25
x = 0.1
for s in stages:
    box(ax, (x, 0.6), w, 1.0, s, fc="#dbeafe", ec="#1d4ed8", fontsize=8.2)
    x += w + gap
x = 0.1
for _ in range(len(stages) - 1):
    arrow(ax, (x + w, 1.1), (x + w + gap, 1.1))
    x += w + gap

fig.tight_layout()
out = OUT_DIR / "mlops_pipeline.png"
fig.savefig(out, dpi=180)
print("Сохранено:", out)
plt.show()
'''

CELLS = [
    ("markdown", "# Диаграммы для диплома Glossa\n\nГенерирует схемы, отсутствующие в виде изображений: архитектура, потоки данных, конвейер DWPose, скользящее окно, DVC-пайплайн. Все файлы сохраняются в `models/plots/`."),
    ("code", _SETUP),
    ("markdown", "## Рисунок А.1 — Архитектура микросервисов"),
    ("code", _CELL_ARCH),
    ("markdown", "## Рисунок А.2 — Поток данных РЖЯ → речь"),
    ("code", _CELL_FLOW_RSL),
    ("markdown", "## Рисунок А.3 — Поток данных речь → РЖЯ"),
    ("code", _CELL_FLOW_TTR),
    ("markdown", "## Рисунок 2.1 — Конвейер DWPose (раздел 2.2)"),
    ("code", _CELL_DWPOSE),
    ("markdown", "## Рисунок 2.2 — Алгоритм скользящего окна (раздел 2.2)"),
    ("code", _CELL_WINDOW),
    ("markdown", "## Рисунок 2.3 — DVC-пайплайн (раздел 2.6)"),
    ("code", _CELL_MLOPS),
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
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    nb = build_notebook()
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Ноутбук создан: {NB_PATH}")
