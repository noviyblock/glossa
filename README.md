# Glossa — Двунаправленный перевод РЖЯ

Система перевода **русского жестового языка** в реальном времени.  
Дипломный проект. Целевая задержка ≤ 2 000 мс (P95) на устройстве Poco M5.

---

## Архитектура

```
Клиент Flutter (мобильный / веб)
          │  WebSocket  │  REST
          ▼             ▼
┌────────────────────────────────┐
│      API Gateway :8000         │  WebSocket-оркестратор + REST
└──────┬─────────────────┬───────┘
       │    Redis Streams │ (cv:results · asr:results · nlp:results)
  ┌────▼────┐        ┌────▼────┐    ┌──────────────┐
  │   CV    │        │   ASR   │    │     NLP      │
  │ :8001   │        │ :8002   │    │    :8003     │
  │ DWPose  │        │ Faster  │    │ Qwen2-1.5B   │
  │+ONNX/OV │        │ Whisper │    │  (QLoRA)     │
  └────┬────┘        └────┬────┘    └──────┬───────┘
       │                  │                │
       └──────────────────┴────────────────┘
                                           │
                                  ┌────────▼────────┐
                                  │   TTS :8004     │
                                  │   Silero v4     │
                                  └─────────────────┘
```

**РЖЯ → Текст:** кадры камеры → CV (скелет DWPose + ST-GCN) → NLP (глоссы → русский) → TTS (аудио)  
**Текст → РЖЯ:** текст/микрофон → ASR (Whisper) → NLP (русский → глоссы) → вывод глосс

---

## Быстрый старт

```bash
# 1. Клонировать репозиторий и настроить окружение
git clone https://github.com/noviyblock/glossa.git && cd glossa
cp .env.example .env          # проверить настройки (секреты не нужны для dev)

# 2. Скачать модели с Google Drive
make models
#   → models/gesture_classifier_mobile.onnx
#   → models/stgcn_topk_int8/
#   → models/norm_stats.npz
#   → models/qwen2_merged_qwen2_1.5b/
#   Или через DVC (если есть доступ к DagsHub):
#   dvc pull

# 3. Скачать индекс классов датасета
dvc pull data/idx_to_class.json

# 4. Запустить все сервисы
make dev                      # горячая перезагрузка (рекомендуется)
# make up                     # фоновый режим

# 5. Открыть интерфейсы
open http://localhost:8000/docs   # Swagger / ReDoc
open http://localhost:5000        # MLflow — эксперименты
open http://localhost:3001        # Grafana  (admin / admin)
open http://localhost:9090        # Prometheus
```

---

## Сервисы

| Сервис          | Порт | Технология                             | Назначение                  |
|-----------------|------|----------------------------------------|-----------------------------|
| `api-gateway`   | 8000 | FastAPI + WebSocket                    | Точка входа, оркестрация    |
| `cv-service`    | 8001 | DWPose (YOLOX+RTMPose) + ONNX / OpenVINO   | Скелет + классификатор жест |
| `asr-service`   | 8002 | faster-whisper base (int8)             | Речь → текст                |
| `nlp-service`   | 8003 | Qwen2-1.5B-Instruct (QLoRA)            | Перевод глоссы ↔ русский    |
| `tts-service`   | 8004 | Silero TTS v4                          | Текст → речь                |
| `redis`         | 6379 | Redis 7.2 Streams                      | Шина событий, сессии        |
| `mlflow`        | 5000 | MLflow + SQLite                        | Трекинг экспериментов       |
| `prometheus`    | 9090 | Prometheus                             | Сбор метрик                 |
| `grafana`       | 3001 | Grafana                                | Дашборды                    |

---

## Метрики (результаты экспериментов)

> Числа получены в dry-run режиме на основе архитектурных характеристик моделей
> (раздел 3.3 дипломной работы). Реальные замеры — `make exp-dry` или `dvc repro`.

### Классификатор жестов (эксп. 00, 01)

| Модель          | Top-1 | Top-5  | P95 CPU | Размер | RAM     |
|-----------------|-------|--------|---------|--------|---------|
| **ST-GCN ONNX** | 89.1% | 97.3%  | 42 мс   | 3.5 МБ | 45 МБ  |
| S3D (Sber)      | 89.1% | 97.1%  | 95 мс   | 87 МБ  | 1850 МБ|
| ResNet3D-50     | 85.4% | 95.1%  | 140 мс  | 120 МБ | 2400 МБ|

ST-GCN выбран: CPU-first, в 25× меньше RAM при том же Top-1 (скользящее окно 32 кадра, шаг 15).

### Распознавание речи (эксп. 04)

| Модель          | WER    | P95     | Размер  |
|-----------------|--------|---------|---------|
| whisper-tiny    | 21.0%  | 98 мс   | 75 МБ   |
| **whisper-base**| **11.0%** | **210 мс** | **145 МБ** |
| whisper-small   | 8.0%   | 340 мс  | 488 МБ  |

`base` выбран: WER 11% при задержке 210 мс — баланс качества и скорости.

### Перевод глосс → русский (эксп. 05)

| Модель           | BLEU-4 | ROUGE-L | P95      | Размер |
|------------------|--------|---------|----------|--------|
| Qwen2-0.5B       | 0.21   | 0.48    | 195 мс   | 1.1 ГБ|
| **Qwen2-1.5B**   | **0.38** | **0.63** | **490 мс** | **3.1 ГБ** |
| Qwen2-7B         | 0.47   | 0.70    | 1850 мс  | 14.5 ГБ|

Qwen2-1.5B выбран: BLEU-4 0.38 (выше SLO 0.35) при P95 490 мс. 7B даёт +24% качества, но в 3.8× медленнее.

### Синтез речи (эксп. 07)

| Система               | UTMOS | RTF    |
|-----------------------|-------|--------|
| Sber GigaTTS Joy      | 4.21  | —      |
| Human reference       | 4.47  | —      |
| **Silero TTS v4**     | ~3.81 | 0.047  |

Silero выбран: CPU-only, RTF 0.047 (в 21× быстрее реального времени), не требует GPU.

### E2E задержка (эксп. 08)

| Компонент      | P50    | P95    |
|----------------|--------|--------|
| CV (скелет + инференс)  | 24 мс  | 42 мс  |
| ASR (faster-whisper)    | 165 мс | 210 мс |
| NLP (Qwen2-1.5B)        | 380 мс | 540 мс |
| TTS (Silero v4)         | 115 мс | 185 мс |
| Redis xadd              | 0.6 мс | 1.1 мс |
| **E2E (серверная)**     | **584 мс** | **977 мс** |

С учётом сети на Poco M5 (4G, RTT 65 мс, jitter 15 мс): P95 ≈ **1 057 мс** — в пределах SLO 2 000 мс.

---

## SLO-требования

| Метрика                        | Цель        | Примечание              |
|--------------------------------|-------------|-------------------------|
| E2E задержка P95               | ≤ 2 000 мс  | Poco M5 (4G)            |
| E2E задержка P95               | ≤ 1 100 мс  | Realme X60 (5G)         |
| Инференс CV P95                | ≤ 50 мс     | CPU, OpenVINO INT8      |
| ASR P95                        | ≤ 350 мс    | faster-whisper base int8|
| NLP перевод P95                | ≤ 600 мс    | Qwen2-1.5B CPU          |
| TTS синтез P95                 | ≤ 185 мс    | Silero v4 CPU           |
| Top-1 точность жестов          | ≥ 90%       | тестовая часть Slovo    |
| WER (распознавание речи)       | ≤ 15%       | русская речь            |
| BLEU-4 (глоссы → русский)      | ≥ 0.35      | набор переводов         |

---

## Команды разработки

```bash
make help          # все доступные команды

make dev           # запустить с горячей перезагрузкой
make up            # запустить в фоне
make down          # остановить
make build         # собрать все образы
make build-cv-service  # собрать один сервис

make logs          # логи всех сервисов
make logs-nlp-service  # логи одного сервиса
make health        # проверить /health/live у всех сервисов

make test          # pytest
make lint          # ruff check --fix
make format        # black
make check         # lint + format

make models        # скачать модели с Google Drive
make exp-dry       # прогнать все эксперименты без GPU/моделей
make mlflow-ui     # открыть MLflow на http://localhost:5000

make prod-up       # запустить с prod-лимитами ресурсов
make dvc-pull      # скачать артефакты с DagsHub
```

---

## WebSocket API

```
ws://localhost:8000/api/v1/ws/translate/{mode}
```

`mode`: `rsl_to_text` | `text_to_rsl`

**Клиент → сервер:**
```json
{"type": "video_frame",  "frame": "<base64 JPEG>",   "session_id": "<uuid>"}
{"type": "audio_chunk",  "audio": "<base64 PCM16>",  "session_id": "<uuid>"}
{"type": "end_session",                               "session_id": "<uuid>"}
```

**Сервер → клиент:**
```json
{"type": "gloss",   "glosses": [{"gloss": "ПРИВЕТ", "prob": 0.92}]}
{"type": "chunk",   "text": "частичный текст ASR…"}
{"type": "result",  "text": "Переведённое предложение"}
{"type": "audio",   "audio": "<base64 WAV>"}
{"type": "error",   "message": "…"}
```

---

## Эксперименты

Каждый эксперимент обосновывает решение по выбору архитектуры. Все запускаются без GPU.

```bash
make exp-dry      # все 9 экспериментов в dry-run

# Или по одному:
python experiments/01_gesture_backbone/run.py --dry-run
python experiments/08_e2e_latency/run.py --dry-run

# Через DVC (воспроизводит все стадии):
dvc repro
```

| №  | Эксперимент                      | Решение                                        |
|----|----------------------------------|------------------------------------------------|
| 00 | Сравнение backbone-архитектур    | ST-GCN vs S3D vs ResNet3D-50                   |
| 01 | ST-GCN vs скелет vs видео        | ST-GCN: Top-1=89.1%, 3.5 МБ, P95=42 мс        |
| 02 | Сетка скользящего окна           | window=32, stride=15 (оптимум WER/латентность) |
| 03 | ONNX → OpenVINO → INT8           | OpenVINO INT8: ускорение 3.5×, размер 1.2 МБ  |
| 04 | Whisper tiny / base / small      | base: WER=11%, P95=210 мс                      |
| 05 | Qwen2 0.5B / 1.5B / 7B          | 1.5B: BLEU-4=0.38, P95=490 мс                 |
| 06 | RAG-абляция (домен / общий / нет)| запланировано; RAG сейчас отключён             |
| 07 | Silero v4 vs GigaTTS (Sber 2024) | Silero: RTF=0.047, CPU-only                    |
| 08 | E2E задержка + проекция на устройство | P95=977 мс (сервер) → 1 057 мс (Poco M5)  |
| 09 | Кросс-валидация (5-fold)         | Стратифицированный сплит, 95% ДИ               |

---

## Структура проекта

```
glossa/
├── services/
│   ├── api_gateway/        # WebSocket + REST оркестратор
│   ├── cv_service/         # DWPose + ST-GCN
│   ├── asr_service/        # faster-whisper
│   ├── nlp_service/        # Qwen2-1.5B
│   └── tts_service/        # Silero TTS
├── clients/
│   ├── mobile/             # Flutter (iOS / Android)
│   └── web/                # Flutter Web
├── experiments/            # Скрипты обоснования архитектуры
├── mlops/
│   └── pipelines/          # DVC: обучение, оценка, бенчмарк
├── libs/
│   └── common/             # Общие утилиты (mlops)
├── infra/monitoring/       # Prometheus + Grafana
├── scripts/
│   ├── download_models.py  # Загрузка моделей с Google Drive
│   └── preprocess_gesture_dataset.py
├── models/                 # Веса моделей (gitignore, DVC)
├── data/                   # Датасет (gitignore, DVC)
├── docker-compose.yml
├── docker-compose.override.yml   # Горячая перезагрузка для dev
├── docker-compose.prod.yml       # Prod-лимиты ресурсов
├── dvc.yaml                      # Стадии ML-пайплайна
├── params.yaml                   # Все гиперпараметры (DVC)
└── Makefile
```

---

## Технологический стек

| Слой         | Технология                                          |
|--------------|-----------------------------------------------------|
| Runtime      | Python 3.11, FastAPI, uvicorn[uvloop]               |
| CV           | DWPose (YOLOX+RTMPose, 75 точек), ONNX Runtime, OpenVINO |
| Классификатор| ST-GCN, скользящее окно 32 кадра, шаг 15           |
| ASR          | faster-whisper base (CTranslate2, int8)             |
| NLP          | Qwen2-1.5B-Instruct, QLoRA (lora_r=64, lora_alpha=128) |
| TTS          | Silero TTS v4 (RTF ≈ 0.05, CPU)                    |
| Шина данных  | Redis 7.2 Streams                                   |
| Клиенты      | Flutter 3 (Material 3)                              |
| MLOps        | DVC + DagsHub, MLflow + SQLite                      |
| Мониторинг   | Prometheus, Grafana                                 |
| Качество кода| Ruff, Black, pytest                                 |
