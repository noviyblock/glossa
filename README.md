# Glossa — двунаправленный перевод русского жестового языка

Система перевода **русского жестового языка (РЖЯ)** в реальном времени: с
камеры/загруженного видео — в русский текст и озвучку, и обратно — из
текста/речи — в последовательность жестов (видео эталонных клипов или
skeleton-анимация). Поддерживается режим двустороннего видеозвонка между
двумя участниками. Дипломный проект.
---
- [[Презентация проекта:](docs\Презентация проекта.pdf)](https://github.com/noviyblock/glossa/blob/f2482027aa4b361e8d8c0536782a5cdcde29a948/docs/%D0%9F%D1%80%D0%B5%D0%B7%D0%B5%D0%BD%D1%82%D0%B0%D1%86%D0%B8%D1%8F%20%D0%BF%D1%80%D0%BE%D0%B5%D0%BA%D1%82%D0%B0.pdf)
---

## Архитектура

```
                    Flutter Web Client
             │  WebSocket (/ws/translate/{mode}) │  REST
             ▼                                    ▼
    ┌───────────────────────────────────────────────────┐
    │                 api-gateway :8000                  │
    │  оркестрация, буфер предложения, история диалога,  │
    │  relay между участниками звонка                    │
    └───┬──────────────┬──────────────┬──────────────┬───┘
        │              │              │              │
   ┌────▼────┐    ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
   │   CV    │    │   NLP   │    │   TTS   │    │   ASR   │
   │  :8001  │    │  :8003  │    │  :8004  │    │  :8002  │
   │ rtmlib  │    │ Qwen2-  │    │ Silero  │    │ faster- │
   │(RTMDet+ │    │ 1.5B,   │    │ TTS +   │    │ whisper │
   │ RTMPose)│    │ 4-bit   │    │ gloss-  │    │         │
   │ + ST-GCN│    │         │    │ video/  │    │         │
   │         │    │         │    │ skeleton│    │         │
   └─────────┘    └─────────┘    └─────────┘    └─────────┘
        │              │              │
        └──────────────┴──────────────┴──── Redis (сессии, очереди результатов)
```

![Общая архитектура](docs/img/1.%20%D0%9E%D0%B1%D1%89%D0%B0%D1%8F%20%D0%B0%D1%80%D1%85%D0%B8%D1%82%D0%B5%D0%BA%D1%82%D1%83%D1%80%D0%B0.png)

**РЖЯ → Текст:** камера/видео → CV (skeleton → автоматическая сегментация
жеста по амплитуде движения → ST-GCN классификатор, top-3) → буфер
предложения → NLP (T9-подобная дизамбигуация последовательности жестов с
учётом контекста диалога → русское предложение) → TTS (озвучка).

![Пайплайн РЖЯ → Текст](docs/img/2.%20%D0%9F%D0%B0%D0%B9%D0%BF%D0%BB%D0%B0%D0%B9%D0%BD%20%D0%A0%D0%96%D0%AF%20%E2%86%92%20%D0%A2%D0%B5%D0%BA%D1%81%D1%82.png)

**Текст → РЖЯ:** текст или голос (→ ASR) → NLP (перевод, строго
ограниченный реальным 200-словным словарём жестов — детерминированный
пост-фильтр отбрасывает всё, чего нет в словаре) → TTS (склейка эталонных
видео-клипов или skeleton-анимация для вывода клиенту).

![Пайплайн Текст → РЖЯ](docs/img/3.%20%D0%9F%D0%B0%D0%B9%D0%BF%D0%BB%D0%B0%D0%B9%D0%BD%20%D0%A2%D0%B5%D0%BA%D1%81%D1%82%20%E2%86%92%20%D0%A0%D0%96%D0%AF.png)

**Двусторонний звонок:** два участника привязываются друг к другу по
короткому коду; финальный перевод каждой стороны транслируется другой в
реальном времени поверх тех же двух однонаправленных пайплайнов.

![Relay двустороннего звонка](docs/img/6.%20Relay%20%D0%B4%D0%B2%D1%83%D1%81%D1%82%D0%BE%D1%80%D0%BE%D0%BD%D0%BD%D0%B5%D0%B3%D0%BE%20%D0%B7%D0%B2%D0%BE%D0%BD%D0%BA%D0%B0.png)

GPU-развёртывание (`docker-compose.gpu.yml`) поддержано для всех четырёх
ML-сервисов (CV/NLP/TTS/ASR) и подтверждено на реальном VM-стенде — даёт
кратный выигрыш в задержке относительно CPU (например, обратный перевод
текст→жесты: ~20 c на CPU → ~1.5–2 c на GPU).

---

## Быстрый старт

```bash
# 1. Клонировать репозиторий и настроить окружение
git clone https://github.com/noviyblock/glossa.git && cd glossa
cp .env.example .env

# 2. Скачать модели и датасет
make models          # модели с Google Drive
dvc pull              # или через DVC/DagsHub — модели + data/idx_to_class.json

# 3. Запустить (CPU)
make dev              # горячая перезагрузка (docker-compose.override.yml)
# make up              # фоновый режим

# 3'. Или на GPU-хосте (CV/NLP/TTS/ASR на CUDA):
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build

# 4. Открыть интерфейсы
open http://localhost:8000/docs   # Swagger / ReDoc
open http://localhost:5000        # MLflow — эксперименты
open http://localhost:3001        # Grafana  (admin / admin)
open http://localhost:9090        # Prometheus
```

---

## Сервисы

| Сервис          | Порт | Технология                                   | Возможности                                                                                     |
|-----------------|------|-----------------------------------------------|---------------------------------------------------------------------------------------------------|
| `api-gateway`   | 8000 | FastAPI + WebSocket                           | Точка входа, буферизация и накопление предложения из нескольких жестов, общая для обоих направлений история диалога, изоляция состояния между сессиями/режимами, relay для видеозвонка |
| `cv-service`    | 8001 | rtmlib (RTMDet+RTMPose) + ST-GCN (ONNX/OpenVINO) | Извлечение skeleton (тело + обе кисти), автоматическая сегментация непрерывного видео на отдельные жесты (state machine по амплитуде движения, без ручной кнопки), классификация (200 классов, top-3, TTA), устойчивость к кратковременной потере трекинга рук |
| `nlp-service`   | 8003 | Qwen2-1.5B, LoRA-дообучение, 4-bit inference  | Дизамбигуация последовательности распознанных жестов в связное русское предложение с учётом контекста; обратный перевод текста в жесты, строго ограниченный реальным словарём (RAG-lite: лексический предфильтр кандидатов + few-shot + детерминированный пост-фильтр) |
| `tts-service`   | 8004 | Silero TTS + сборка видео/skeleton            | Озвучка распознанного текста; сборка клипа из эталонных видео на каждую глоссу (ffmpeg) либо покадровая skeleton-анимация — без необходимости иметь видео на каждое слово |
| `asr-service`   | 8002 | faster-whisper                                | Голосовой ввод для слышащего собеседника (альтернатива печати текста)                             |
| `redis`         | 6379 | Redis 7.2                                     | Состояние сессий, очереди результатов, пары звонков                                               |
| `mlflow`        | 5000 | MLflow + SQLite                               | Трекинг экспериментов                                                                             |
| `prometheus`    | 9090 | Prometheus                                    | Сбор метрик всех сервисов                                                                         |
| `grafana`       | 3001 | Grafana                                       | Дашборды                                                                                           |

**cv-service — сегментация жеста и классификатор:**

![Конечный автомат сегментации жеста](docs/img/4.%20%D0%9A%D0%BE%D0%BD%D0%B5%D1%87%D0%BD%D1%8B%D0%B9%20%D0%B0%D0%B2%D1%82%D0%BE%D0%BC%D0%B0%D1%82%20%D1%81%D0%B5%D0%B3%D0%BC%D0%B5%D0%BD%D1%82%D0%B0%D1%86%D0%B8%D0%B8%20%D0%B6%D0%B5%D1%81%D1%82%D0%B0.png)

![Раскладка keypoint-скелета](docs/img/9.%20%D0%A0%D0%B0%D1%81%D0%BA%D0%BB%D0%B0%D0%B4%D0%BA%D0%B0%20keypoint-%D1%81%D0%BA%D0%B5%D0%BB%D0%B5%D1%82%D0%B0.png)

**nlp-service — накопление предложения:**

![Накопление предложения + LLM-дизамбигуация](docs/img/5.%20%D0%9D%D0%B0%D0%BA%D0%BE%D0%BF%D0%BB%D0%B5%D0%BD%D0%B8%D0%B5%20%D0%BF%D1%80%D0%B5%D0%B4%D0%BB%D0%BE%D0%B6%D0%B5%D0%BD%D0%B8%D1%8F%20%2B%20LLM-%D0%B4%D0%B8%D0%B7%D0%B0%D0%BC%D0%B1%D0%B8%D0%B3%D1%83%D0%B0%D1%86%D0%B8%D1%8F.png)

---

## Реальные показатели (боевое развёртывание)

В отличие от dry-run-оценок ниже (раздел «Эксперименты»), это числа,
подтверждённые на реальном VM-стенде с реальной моделью и реальными
пользовательскими сессиями:

| Показатель                                             | Значение                          |
|----------------------------------------------------------|------------------------------------|
| Top-1 accuracy ST-GCN (200 классов, held-out val, n=1000) | ~60% (случайный baseline — 0.5%)   |
| Обратный перевод текст→жесты (NLP), CPU                  | ~7–20 c (в зависимости от длины промпта) |
| Обратный перевод текст→жесты (NLP), GPU                  | ~1.5–2 c                           |
| Дизамбигуация последовательности жестов (NLP), GPU        | ~1.5–2 c                           |
| Пропускная способность CV-инференса на боевом GPU-стенде  | ~5 кадров/с (наблюдаемо, не пиковая мощность конкретной модели GPU) |
| Тестовое покрытие backend                                 | 68+ юнит-тестов (см. `make test`)  |

---

## SLO / целевые показатели

Ниже — целевые ориентиры для проектирования (не результаты измерений на
конкретном мобильном устройстве):

| Метрика                        | Цель        |
|---------------------------------|-------------|
| E2E задержка P95 (сервер)      | ≤ 2 000 мс  |
| Инференс CV P95 (GPU)          | ≤ 50 мс     |
| ASR P95                        | ≤ 350 мс    |
| NLP перевод P95 (GPU)          | ≤ 600 мс    |
| TTS синтез P95                 | ≤ 185 мс    |

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
make exp-dry       # прогнать все эксперименты без GPU/моделей (dry-run)
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
{"type": "video_frame",         "frame": "<base64 JPEG>",  "session_id": "<uuid>"}
{"type": "audio_chunk",         "audio": "<base64 PCM>",   "session_id": "<uuid>"}
{"type": "flush_sentence",      "session_id": "<uuid>"}
{"type": "delete_last_gesture", "session_id": "<uuid>"}
{"type": "end_session",         "session_id": "<uuid>"}
```

**Сервер → клиент:**
```json
{"type": "gloss",            "payload": {"glosses": [...], "gesture_active": true, "preview": false}}
{"type": "pending_sentence",  "payload": {"positions": [...]}}
{"type": "chunk",             "payload": {"text": "...", "is_final": false}}
{"type": "result",            "payload": {"text": "..."}}
{"type": "audio",             "payload": {"wav": "<base64 WAV>"}}
{"type": "video",             "payload": {"video": "<base64 MP4>"}}
{"type": "skeleton",          "payload": {"sequences": [...]}}
{"type": "peer_message",      "payload": {...}}
{"type": "error",             "payload": {"message": "..."}}
```

REST: `POST /api/v1/translate` (text_to_rsl без WS), `POST /api/v1/asr`
(речь → текст → тот же путь text_to_rsl), `POST /api/v1/call/create`,
`POST /api/v1/call/{call_id}/join` (двусторонний звонок).

![Sequence-диаграмма WebSocket-обмена](docs/img/10.%20Sequence-%D0%B4%D0%B8%D0%B0%D0%B3%D1%80%D0%B0%D0%BC%D0%BC%D0%B0%20WebSocket-%D0%BE%D0%B1%D0%BC%D0%B5%D0%BD%D0%B0.png)

---

## CI/CD и качество

GitHub Actions, четыре независимых workflow:

- **`ci.yml`** — lint (ruff/black/mypy), security lint (bandit), тестовый
  прогон (`pytest libs/ services/` с покрытием), матричная сборка и
  публикация Docker-образов всех пяти сервисов в GHCR.
- **`security.yml`** — CodeQL SAST, Bandit SAST, Trivy (сканирование
  файловой системы и собранных образов), dependency review, TruffleHog
  (поиск утечек секретов).
- **`cd.yml`** — деплой.
- **`mlops.yml`** — проверки ML-пайплайна.

Каждый сервис — многостадийный Docker-образ с отдельными CPU- и
GPU-таргетами (`production` / `production-gpu`).

![CI/CD пайплайн](docs/img/8.%20CICD%20%D0%BF%D0%B0%D0%B9%D0%BF%D0%BB%D0%B0%D0%B9%D0%BD.png)

---

## Эксперименты

Каждый эксперимент обосновывает решение по выбору архитектуры и запускается
без GPU в dry-run режиме (оценка по архитектурным характеристикам моделей,
не полный реальный прогон). Реальные измерения — `dvc repro` или ручной
запуск конкретного скрипта.

```bash
make exp-dry      # все 9 экспериментов в dry-run

# Или по одному:
python experiments/01_gesture_backbone/run.py --dry-run
python experiments/08_e2e_latency/run.py --dry-run

# Через DVC (воспроизводит все стадии):
dvc repro
```

| №  | Эксперимент                           | Вопрос                                          |
|----|-----------------------------------------|--------------------------------------------------|
| 01 | Сравнение backbone-архитектур           | ST-GCN vs S3D vs ResNet3D-50                     |
| 02 | Сетка скользящего окна                  | размер окна / шаг                                |
| 03 | Ускорение инференса                     | ONNX → OpenVINO → INT8                           |
| 04 | Сравнение ASR                           | Whisper tiny / base / small                      |
| 05 | Размер NLP-модели                       | Qwen2 0.5B / 1.5B / 7B                           |
| 06 | RAG-абляция                             | словарное ограничение домена / общий режим / нет |
| 07 | Сравнение TTS                           | Silero vs альтернативы (UTMOS)                   |
| 08 | E2E задержка                            | проекция на целевое устройство                   |
| 09 | Кросс-валидация                         | устойчивость метрик (5-fold)                     |

![ML-пайплайн данные → модель → деплой](docs/img/7.%20ML-%D0%BF%D0%B0%D0%B9%D0%BF%D0%BB%D0%B0%D0%B9%D0%BD%20%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D0%B5%20%E2%86%92%20%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D1%8C%20%E2%86%92%20%D0%B4%D0%B5%D0%BF%D0%BB%D0%BE%D0%B9.png)

---

## Устройство cv-service и классификатора ST-GCN

Модули cv-service и путь одного кадра через них:

![Компонентная схема cv-service](docs/img/A1.%20%D0%9A%D0%BE%D0%BC%D0%BF%D0%BE%D0%BD%D0%B5%D0%BD%D1%82%D0%BD%D0%B0%D1%8F%20%D1%81%D1%85%D0%B5%D0%BC%D0%B0%20cv-service.png)

![Sequence-диаграмма одного кадра](docs/img/A2.%20Sequence-%D0%B4%D0%B8%D0%B0%D0%B3%D1%80%D0%B0%D0%BC%D0%BC%D0%B0%20%D0%BE%D0%B4%D0%BD%D0%BE%D0%B3%D0%BE%20%D0%BA%D0%B0%D0%B4%D1%80%D0%B0.png)

![Детальный конечный автомат GestureSegmenter](docs/img/A3.%20%D0%94%D0%B5%D1%82%D0%B0%D0%BB%D1%8C%D0%BD%D1%8B%D0%B9%20%D0%BA%D0%BE%D0%BD%D0%B5%D1%87%D0%BD%D1%8B%D0%B9%20%D0%B0%D0%B2%D1%82%D0%BE%D0%BC%D0%B0%D1%82%20GestureSegmenter.png)

ST-GCN (Spatio-Temporal Graph Convolutional Network) — 10 блоков
graph+temporal convolution (см. `mlops/pipelines/train_gesture.py`), вход
64 кадра × 75 точек × 3 канала, каналы 3→64→...→128→...→256, голова
`Linear(256 → 200)`:

![Полная сеть — вход → 10 ST-GCN блоков → голова](docs/img/B1.%20%D0%9F%D0%BE%D0%BB%D0%BD%D0%B0%D1%8F%20%D1%81%D0%B5%D1%82%D1%8C%20%D0%B2%D1%85%D0%BE%D0%B4%20%E2%86%92%2010%20ST-GCN%20%D0%B1%D0%BB%D0%BE%D0%BA%D0%BE%D0%B2%20%E2%86%92%20%D0%B3%D0%BE%D0%BB%D0%BE%D0%B2%D0%B0.png)

![Внутренняя структура одного ST-GCN блока](docs/img/B2.%20%D0%92%D0%BD%D1%83%D1%82%D1%80%D0%B5%D0%BD%D0%BD%D1%8F%D1%8F%20%D1%81%D1%82%D1%80%D1%83%D0%BA%D1%82%D1%83%D1%80%D0%B0%20%D0%BE%D0%B4%D0%BD%D0%BE%D0%B3%D0%BE%20ST-GCN%20%D0%B1%D0%BB%D0%BE%D0%BA%D0%B0.png)

![Топология графа (75 узлов, группы рёбер)](docs/img/B3.%20%D0%A2%D0%BE%D0%BF%D0%BE%D0%BB%D0%BE%D0%B3%D0%B8%D1%8F%20%D0%B3%D1%80%D0%B0%D1%84%D0%B0%20%2875%20%D1%83%D0%B7%D0%BB%D0%BE%D0%B2%2C%20%D0%B3%D1%80%D1%83%D0%BF%D0%BF%D1%8B%20%D1%80%D1%91%D0%B1%D0%B5%D1%80%29.png)

---

## Структура проекта

```
glossa/
├── services/
│   ├── api_gateway/        # WebSocket + REST оркестратор
│   ├── cv_service/         # rtmlib + ST-GCN, сегментация жестов
│   ├── asr_service/        # faster-whisper
│   ├── nlp_service/        # Qwen2-1.5B, словарное ограничение
│   └── tts_service/        # Silero TTS, сборка видео/skeleton
├── clients/
│   ├── mobile/             # Flutter (iOS / Android)
│   └── web/                # Flutter Web
├── experiments/            # Скрипты обоснования архитектуры (01-09)
├── mlops/
│   └── pipelines/          # DVC: обучение, оценка, бенчмарк
├── libs/
│   └── common/             # Общие утилиты (mlops)
├── infra/monitoring/       # Prometheus + Grafana
├── scripts/                # Загрузка моделей, препроцессинг, офлайн-оценка
├── docs/                   # Диплом (ВКР, отзыв, презентация, антиплагиат)
├── models/                 # Веса моделей (gitignore, DVC)
├── data/                   # Датасет (gitignore, DVC)
├── docker-compose.yml
├── docker-compose.gpu.yml        # GPU-оверлей (CV/NLP/TTS/ASR на CUDA)
├── docker-compose.override.yml   # Горячая перезагрузка для dev
├── docker-compose.prod.yml       # Prod-лимиты ресурсов
├── dvc.yaml                      # Стадии ML-пайплайна
├── params.yaml                   # Все гиперпараметры (DVC)
└── Makefile
```

---

## Технологический стек

| Слой         | Технология                                                    |
|--------------|-----------------------------------------------------------------|
| Runtime      | Python 3.11, FastAPI, uvicorn                                   |
| CV           | rtmlib (RTMDet + RTMPose), ONNX Runtime, OpenVINO                |
| Классификатор| ST-GCN, скользящее окно 64 кадра                                 |
| ASR          | faster-whisper (CTranslate2)                                    |
| NLP          | Qwen2-1.5B, LoRA-дообучение (merge), bitsandbytes 4-bit inference |
| TTS          | Silero TTS                                                       |
| Шина данных  | Redis 7.2                                                        |
| Клиенты      | Flutter 3 (Material 3)                                           |
| MLOps        | DVC + DagsHub, MLflow + SQLite                                   |
| Мониторинг   | Prometheus, Grafana                                              |
| CI/CD        | GitHub Actions (lint, test, build, security, deploy)             |
| Качество кода| Ruff, Black, MyPy, Bandit, pytest                                |

---

## Документы (Диплом)

Пакет документов дипломной работы — в [`docs/`](docs/):

- [ВКР](docs/%D0%92%D0%9A%D0%A0_%D0%A7%D0%B5%D1%80%D0%BD%D1%8B%D1%85%20%D0%90.%D0%92.pdf)
- [Отзыв](docs/%D0%9E%D1%82%D0%B7%D1%8B%D0%B2_%D0%92%D0%9A%D0%A0_%D0%A7%D0%B5%D1%80%D0%BD%D1%8B%D1%85%20%D0%90.%D0%92.pdf)
- [Презентация ВКР](docs/%D0%9F%D1%80%D0%B5%D0%B7%D0%B5%D0%BD%D1%82%D0%B0%D1%86%D0%B8%D1%8F_%D0%92%D0%9A%D0%A0_%D0%A7%D0%B5%D1%80%D0%BD%D1%8B%D1%85%20%D0%90.%D0%92.pdf)
- [Антиплагиат](docs/%D0%90%D0%BD%D1%82%D0%B8%D0%BF%D0%BB%D0%B0%D0%B3%D0%B8%D0%B0%D1%82_%D0%92%D0%9A%D0%A0_%D0%A7%D0%B5%D1%80%D0%BD%D1%8B%D1%85%20%D0%90.%D0%92.pdf)
