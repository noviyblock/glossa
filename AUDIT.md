# Glossa — аудит кода (не README)

Дата: 2026-07-18. Метод: чтение исходников, конфигов, git-истории. Каждый
пункт — конкретные файлы/строки на момент аудита, не общие впечатления.
Этот файл **не добавлен в git** (см. `.gitignore`, если нужно — добавьте
вручную).

---

## 1. Карта сервисов (по факту кода)

### Реальные сервисы и их состав файлов

| Сервис | Файлы (кроме `main.py`/`config.py`/`Dockerfile`/`pyproject.toml`) |
|---|---|
| `api_gateway` | `orchestrator.py`, `models.py`, `ws_handler.py` |
| `cv_service` | `gesture_classifier.py`, `gesture_segmenter.py`, `keypoint_extractor.py`, `keypoint_smoother.py`, `normalizer.py`, `sliding_window.py` |
| `asr_service` | `transcriber.py` |
| `nlp_service` | `translator.py`, `cache.py` |
| `tts_service` | `synthesizer.py`, `video.py` |

Ни в одном сервисе нет пакета `app/` — важно для раздела 6 (тесты).

### Реальные роуты/WS

- **cv_service** (`services/cv_service/main.py`): `POST /process_frame` (:117), `WS /ws` (:199, легаси, не вызывается api_gateway — использует отдельный `SlidingWindowBuffer`, не `GestureSegmenter`), `GET /health/live`, `/health/ready`, `/metrics`.
- **asr_service** (`services/asr_service/main.py`): `POST /transcribe` (:89), `WS /ws` (:112), health/metrics.
- **nlp_service** (`services/nlp_service/main.py`): `POST /translate`, `/translate_topk`, `/translate_sequence_topk`, `/translate_reverse`, health/metrics.
- **tts_service** (`services/tts_service/main.py`): `POST /synthesize`, `/sign_video`, `GET /voices`, health/metrics.
- **api_gateway** (`services/api_gateway/main.py`): `POST /api/v1/translate` (:119), `WS /api/v1/ws/translate/{mode}` (:164), health/metrics.

### Версии зависимостей / потенциальные конфликты

- **Нет ни одного lock-файла** (`poetry.lock`/`uv.lock`/`requirements-freeze.txt`) ни в одном из 5 сервисов — все `pyproject.toml` используют `>=`, без верхних границ (кроме `services/cv_service/Dockerfile` — там `onnxruntime-gpu>=1.18,<1.21` запиновали руками после реального падения; **CPU-стадия того же файла — `pip install "onnxruntime>=1.18"` — верхней границы не имеет до сих пор**, тот же риск не устранён для CPU-сборки).
- `torch`/`torchaudio` (nlp_service, tts_service) устанавливаются напрямую в Dockerfile через `--index-url .../cu121` или `/cpu` **без указания версии** (`services/nlp_service/Dockerfile:25,32`, `services/tts_service/Dockerfile:29,36`) — при пересборке в разное время получите разные версии torch без какого-либо сигнала об этом в git diff.
- `openvino>=2024.1` (cv_service), `transformers>=4.41` (nlp_service) — тоже без верхней границы.
- Именно отсутствие верхних границ было корневой причиной трёх реальных падений в этой сессии (см. раздел 4) — паттерн повторяемый, не устранён системно, устранены только конкретные уже случившиеся инциденты.

### Хардкод

- `services/cv_service/keypoint_extractor.py:45-46` — `_COCO17_TO_TRAINING_ORDER`, `_EXTRA_JOINT_SRC`: хардкод индексов, но это уместно (жёстко привязано к обученной модели, комментарий объясняет откуда взялось).
- `services/api_gateway/config.py:4-7` — URL остальных сервисов хардкожены на Docker Compose service names (`http://cv-service:8001` и т.д.) через `os.getenv` с этим значением по умолчанию — работает только внутри данного docker-compose-стека, K8s/другая оркестрация потребует явного окружения (это нормально для текущего масштаба, но не готово к переносу без правок).
- Все `GESTURE_*`/`SMOOTH_*`/`TTA_*`/`SESSION_*` — не хардкод, идут через `os.getenv` с дефолтами.
- **Не проверено**: `clients/web/lib/config.dart` (адрес backend в Flutter-клиенте) — не открывал в рамках этого аудита. Нужно: прочитать этот файл, если важно знать, хардкожен ли prod-адрес.

### `docker-compose.yml` vs `docker-compose.gpu.yml`

- GPU-оверрайд трогает `cv-service`, `asr-service`, `nlp-service`, `tts-service` — не трогает `api-gateway`, `redis`, `prometheus`, `grafana`, `mlflow` (корректно, им GPU не нужен).
- Для всех 4 сервисов реально есть `production-gpu` target в соответствующем `Dockerfile` (не просто заглушка) — подтверждено чтением: `services/asr_service/Dockerfile:42-56` (`pip install ".[gpu]"`, `ASR_DEVICE=cuda`), `services/nlp_service/Dockerfile` (torch cu121), `services/tts_service/Dockerfile` (torch/torchaudio cu121), `services/cv_service/Dockerfile` (onnxruntime-gpu + nvidia-*-cu12 пакеты).
- `deploy.resources.reservations.devices` блок с `driver: nvidia` повторён 4 раза дословно (`docker-compose.gpu.yml:16-22, 30-36, 43-49, 56-62`) — не YAML-anchor'ен, чисто DRY-замечание, не баг.

---

## 2. Путь данных end-to-end

### РЖЯ → текст

Кадр (JPEG, base64) → WS `"video_frame"` (`api_gateway/main.py:203`) → `orchestrator.process_frame` (`orchestrator.py:83`) → `POST cv-service/process_frame` (httpx, общий таймаут `HTTP_TIMEOUT=120s` из `GATEWAY_HTTP_TIMEOUT`, `api_gateway/config.py:10` — единый таймаут на ВСЕ downstream-вызовы, нет per-call override) → cv_service извлекает keypoints + классифицирует → JSON-ответ → gloss буферизуется в Redis (`_set_session`, `orchestrator.py:75`, `setex` с `SESSION_TTL=300s`) → по паузе/лимиту — `POST nlp-service/translate_sequence_topk` → перевод → отдаётся клиенту как WS `"result"`/`"chunk"`.

**Обработка ошибок по каждому переходу:**

| Переход | try/except | Поведение при ошибке |
|---|---|---|
| gateway → cv-service (`orchestrator.py:117-126`) | есть | `raise RuntimeError` → перехватывается в `main.py` WS-хендлере → клиенту `{"type":"error"}`, сессия не рвётся |
| gateway → nlp `_flush_sentence` (`orchestrator.py:210-217`) | есть | графическая деградация — join топ-1 глосс как fallback-текст |
| gateway → tts `/synthesize` в WS-пути `process_audio` (`orchestrator.py:324-332`) | есть | `wav_b64=""`, лог warning, не падает |
| gateway → tts `/sign_video` везде (`orchestrator.py:340-348`, `:408-416`) | есть | `video_b64=None`, лог warning |
| **gateway → nlp `/translate_reverse` в REST-пути `translate_sync` (`orchestrator.py:392-396`)** | **нет** | необработанное исключение httpx поднимается наверх |
| **gateway → tts `/synthesize` в REST-пути `translate_sync` (`orchestrator.py:399-403`)** | **нет** | то же |
| `_redis.xadd`/`_redis.setex` — ВСЕ вызовы (`orchestrator.py:75,147,221,295,315`) | нет нигде | недоступность Redis не обработана явно в этих местах |
| cv_service, ошибка внутри rtmlib (`keypoint_extractor.py:157-163`) | есть | возвращает нулевой массив keypoints — неотличимо от "человека нет в кадре" по коду вызывающей стороны |

⚠️ **РАСХОЖДЕНИЕ**: WS-путь (`process_audio`, строки 279-357) последовательно оборачивает каждый downstream-вызов и деградирует мягко; REST-путь (`translate_sync`, строки 388-422) для того же направления (text→RSL) НЕ оборачивает `/translate_reverse` и `/synthesize` — при недоступности любого из них весь REST-запрос падает целиком (перехватывается уже на уровне `api_gateway/main.py`'s `httpx.TimeoutException`/`httpx.HTTPStatusError` → 504/502, но это curенее и грубее, чем soft-degrade в WS-пути для той же логической операции).

**Нигде во всей системе нет retry** (ни `tenacity`, ни ручных циклов повтора) — единичный сетевой сбой = потерянный кадр/ошибка клиенту, не повтор.

### Текст → РЖЯ

Аудио (base64) → WS `"audio_chunk"` → `orchestrator.process_audio` → `POST asr-service/transcribe` (обёрнут, `orchestrator.py:281-290`, при ошибке `RuntimeError`) → `POST nlp-service/translate_reverse` (обёрнут, `:304-311`, при ошибке — эхо исходного текста как fallback) → параллельно `POST tts-service/synthesize` + `/sign_video` (оба обёрнуты в этом пути) → клиенту WS `"chunk"`/`"result"`/`"audio"`/`"video"`.

---

## 3. ST-GCN / keypoints — ⚠️ ГЛАВНОЕ РАСХОЖДЕНИЕ (приоритетное расследование)

**Порядок keypoints, который реально отдаёт rtmlib** (`services/cv_service/keypoint_extractor.py:24-29`, дословно из комментария):

```
training body17 = [nose, Rsho, Relb, Rwri, Lsho, Lelb, Lwri,
                    Rhip, Rkne, Rank, Lhip, Lkne, Lank, Reye, Leye, Rear, Lear]
```

То есть индекс 1 = правое плечо, 2 = правый локоть, 3 = правое запястье,
4 = левое плечо, 5 = **левый локоть**, 6 = **левое запястье** и т.д. Это
результат применения `_COCO17_TO_TRAINING_ORDER = [0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]`
(`keypoint_extractor.py:45`) к истинному COCO-17 порядку — переиндексация
делается именно здесь, `_remap_coco133_to_75()` (`keypoint_extractor.py:69-78`).

**Граф, который строит ST-GCN (та же логика в двух местах):**

- `mlops/pipelines/train_gesture.py:41-49` (`_COCO_EDGES`)
- `scripts/build_colab_01a_notebook.py:234-238` (`COCO_EDGES` внутри `load_skeleton_graph()`) — **идентичный список рёбер**, это генератор блокнота `colab_glossa_01a_train_stgcn_64_200.ipynb`, который по всей этой сессии фигурирует как реальный тренировочный ноутбук базовой (до RTMW-дообучения) модели.

Рёбра `(5, 6), (5, 7), (6, 8), (7, 9)` в обоих файлах помечены комментарием
`# shoulders → elbows → wrists` — то есть граф построен в предположении, что
индекс 5 = плечо, 6 = плечо, 7 = локоть и т.д. (**истинная COCO-17
семантика**). Но по факту (см. выше) индекс 5 в реальных данных — это
**левый локоть**, а не плечо; индекс 6 — **левое запястье**, а не плечо.

**Итог**: ребро, которое код называет "плечо↔плечо" или "плечо↔локоть",
на реальных тренировочных данных физически соединяет "левый локоть↔левое
запястье" — граф-свёртка учится на анатомически перепутанной топологии.
Это не баг инференса (на train и на inference — одна и та же перепутанная,
но *консистентная* топология, поэтому модель вообще что-то предсказывает),
а структурный prior архитектуры, ограничивающий потолок точности. Исправление
требует полного переобучения с нуля на исправленном графе — веса `GraphConv`
уже завязаны на текущую (пусть и мисномер) структуру.

Это уже документировано (не мной впервые) в самом ноутбуке
`colab_glossa_04_finetune_rtmw.ipynb` (ячейка 12, справочная) — но не
исправлено нигде по коду, включая обе найденные копии графа.

---

## 4. Реальное состояние GPU-пути

### Кто реально использует GPU (по Dockerfile/коду, не по факту наличия файла)

| Сервис | GPU-стадия реальна? | Device-selection в коде |
|---|---|---|
| cv_service | да (`Dockerfile` deps-gpu: onnxruntime-gpu + nvidia-curand/cufft/cudnn/cublas-cu12) | `keypoint_extractor.py:125-134` передаёт `device` строкой прямо в `rtmlib.Wholebody(device=...)` |
| asr_service | да (`Dockerfile:43-56`, `pip install ".[gpu]"`) | не проверялось детально в этом проходе — `transcriber.py` не читался |
| nlp_service | да (torch cu121 в deps-gpu) | `translator.py:88,103,107` — `device_map` прямо в `AutoModelForCausalLM.from_pretrained(device_map=...)`, дефолт в коде `DEVICE_MAP=os.getenv("NLP_DEVICE","auto")` (`nlp_service/config.py:5`), но **в `docker-compose.yml:202` явно захардкожено `NLP_DEVICE: cpu`** для базовой (не-GPU) сборки |
| tts_service | да (torch/torchaudio cu121) | не проверялось детально |

### История падений на CPU — git-подтверждённая хронология

1. `10d7cee` (2026-06-29) — **более раннее, ДО этой сессии**: "cv-service production-gpu image failed to start (onnxruntime-gpu requires libcudart.so.13, incompatible with CUDA 12.2 host); GestureClassifier never uses GPU anyway, drop the unused package" — на этом этапе GPU для cv-service был просто **отключён** (пакет убран).
2. `3c209fe` (2026-07-11) — в этой сессии: GPU для cv-service переподключён заново ("wire real GPU build"), обнаружено, что `production-gpu` был мёртвой заглушкой (`CV_DEVICE`/`CV_MODEL_BACKEND` — переменные, не существующие в коде).
3. `329492a` (2026-07-11) — живой лог на VM показал `CUDAExecutionProvider` отсутствует в списке провайдеров: rtmlib тянет свою зависимость `onnxruntime` (CPU), которая переустанавливается ПОСЛЕ `onnxruntime-gpu` и тихо его затирает.
4. `6894b1b` (2026-07-11) — фикс #3 сработал, но всплыла `libcudart.so.13` — неограниченная версия `onnxruntime-gpu` подтянула CUDA 13 рантайм, несовместимый с установленными `nvidia-*-cu12`.
5. `7fbf62f` (2026-07-13) — версия зафиксирована на CUDA 12, но всплыла `libcurand.so.10` — `cudart+cublas+cudnn` оказались неполным набором зависимостей CUDA EP.

**Текущее состояние на момент последнего живого теста в этой сессии** (не git, а лог с реальной VM): контейнер снова упал на `libcurand.so.10` — потому что образ не был пересобран после коммита `7fbf62f` (`docker compose up -d`, без `build`). **GPU-путь cv-service ни разу не был подтверждён работающим end-to-end на реальной VM** — только теоретически исправлен в коде, финальная проверка не прислана обратно.

**NLP latency, обнаруженная в этой сессии**: `translate_sequence_topk` занимает **~9.6-10 секунд** на flush предложения (реальные логи api-gateway, `orchestrator.py:351` `rsl_to_text latency=...`), при том что `docker-compose.yml:202` жёстко ставит `NLP_DEVICE: cpu` для nlp-service, и в тестовой сессии GPU-профиль применялся ТОЛЬКО к `cv-service` (`docker compose ... up -d cv-service`) — nlp-service, скорее всего, всё это время работал на CPU, что и объясняет 10-секундную задержку. **Не подтверждено окончательно** — нужно: лог старта `nlp-service` (`grep -i "device\|cuda"`), не присланный на момент этого аудита.

---

## 5. Словарь и fallback

- 200 классов: `data/idx_to_class.json` (индекс→глосса) и `data/class_to_idx.json` (глосса→индекс). Загружаются в `services/cv_service/gesture_classifier.py` (`CLASS_MAP_PATH`) и `services/tts_service/video.py` (`CLASS_TO_IDX_PATH`).
- **NLP-генерация глосс не ограничена словарём никак**: `services/nlp_service/translator.py`, `REVERSE_SYSTEM_PROMPT` — свободная генерация с few-shot примерами в промпте, ни grammar/constrained decoding, ни post-hoc валидации токенов против `class_to_idx.json` в коде нет.
- **gloss→видео matching**: `services/tts_service/video.py`, класс `SignVideoAssembler` — нормализация (upper-case + удаление пунктуации) и жадное сопоставление по убыванию длины фразы против ключей `class_to_idx.json`. Несовпавшие токены — `logger.warning`, пропускаются. Если не совпал НИ один токен — `build()` возвращает `None` → `/sign_video` отдаёт `{"video": null}` → клиент просто не показывает видео. Исключений/дефолтного клипа нет — деградация тихая и предсказуемая.
- **Bukva**: ни одного файла/ветки/notebook-упоминания в репозитории не найдено (`grep -rniE "bukva|dactyl|дактил"` по всему репо, кроме `.venv`) — только обсуждение в этом чате, ничего не закоммичено.
- Неожиданная находка: `data/selected_classes_200.json:2-5` явно документирует, что при формировании сценария из 200 классов **35 дактильных (посимвольных) классов уже существовали в корпусе Slovo и были сознательно исключены** (`"excluded": [..., "single-character dactyl alphabet signs (35 classes)"]`). Это потенциально более дешёвый источник данных для дактильного fallback, чем интеграция Bukva с нуля — доступ к Slovo уже прошит во все ноутбуки этой сессии.

---

## 6. Тесты и валидация

### ⚠️ РАСХОЖДЕНИЕ — весь видимый набор тестов нерабочий

14 тестовых файлов в `services/{cv,asr,nlp,tts}_service/tests/` (у `api_gateway`
тестов нет вообще) импортируют из пакета `app.*`:

- `services/nlp_service/tests/conftest.py:10` — `from app.domain.entities import ...`
- `services/nlp_service/tests/test_graphs/test_orchestrator.py:10,18` — `from app.domain.entities import ...`, `from app.graphs.orchestrator import NLPOrchestrator`
- `services/cv_service/tests/test_inference_queue.py:9,10,11` — `from app.domain.pipeline import GestureClassifier`, `from app.domain.sliding_window import ...`, `from app.inference.queue import InferenceQueue`

Пакет `app/` **нигде не существует** в текущем дереве репозитория (проверено
`find . -iname app -type d` — пусто). Причина видна в git log: коммит
`0424891` (2026-06-20) — "refactor: flat-file services, remove stale code...
**Replace services/\*/app/ with flat-file modules**" удалил `app/`, но не
удалил тесты, которые от него зависели. Все 14 файлов упадут на первой же
строке импорта (`ModuleNotFoundError: No module named 'app'`) при попытке
`pytest`. **Рабочих автотестов на текущий код нет ни одного.**

Тесты описывают архитектуру, которой не существует в проде: multi-agent
NLP-оркестратор на LangGraph (`intent_classifier`, `safety_filter`,
`medical_simplifier`, `banking_simplifier`, `rag_retriever`,
`hallucination_checker` — ни один из этих модулей не существует в реальном
`nlp_service/translator.py`), `InferenceQueue` с backpressure для cv_service
(реальный `cv_service/main.py` — простые `dict`, без очереди).

### A/B-методология (`scripts/ab_compare_backends.py`) — реальна и рабочая

Подтверждено живым прогоном в этой сессии (не только чтением): загружает
OpenVINO INT8 (`core.compile_model`) и полный ONNX
(`onnxruntime.InferenceSession`) отдельно, гоняет оба на одних и тех же
предзагруженных `(T,75,3)` фичах из `.npy` (аргументы `--features`/`--labels`,
дефолт — путь на несуществующий на тот момент `processed_64_200`, сейчас
DVC-трекнут), применяет один и тот же `norm_stats.npz` z-score к обоим
бэкендам перед подачей, сравнивает `argmax` с истинной меткой, репортит
accuracy/mean confidence/latency + до 20 примеров расхождений.

`scripts/measure_accuracy.py` — отдельный скрипт (два режима: сырые клипы
через `KeypointExtractor`+`cv2`, либо готовые `.npy` keypoints), считает
top1/top1-with-TTA/top3 + near-miss анализ. **Не запускался в рамках этой
сессии/аудита** (нет `rtmlib`/`cv2` в локальном окружении) — методология
проверена чтением, не исполнением.

---

## 7. Лицензии и данные

- `LICENSE` в корне репозитория (:1-21) — **MIT**, `Copyright (c) 2026 noviy_block`. Покрывает только собственный код Glossa.
- **Ни одного упоминания лицензии Slovo или Bukva нигде в репозитории** — проверено `grep -rniE "license|лицензи|CC BY|non-commercial|некоммерч"` по `data/*.dvc`, `data/*.json`, `models/*.dvc`, `README.md` — 0 совпадений.
- Slovo используется в двух принципиально разных ролях:
  1. Обучающие данные (keypoints, никогда не покидают пайплайн офлайн-обработки) — стандартное исследовательское использование.
  2. **Видео-клипы теперь редистрибутируются конечным пользователям** как часть продукта — `models/gloss_clips/` (DVC-трекнуто), отдаётся через `tts_service` `/sign_video` любому клиенту приложения. Это принципиально другой характер использования (распространение как часть сервиса, а не исследование), и в репозитории нет никакого следа проверки, разрешает ли реальная лицензия Slovo на Kaggle такое использование.
- Bukva — не интегрирован (см. раздел 5), вопрос лицензии пока не актуален практически, но встанет при реализации задачи дактильного fallback.

---

## 8. Мониторинг

### Что реально экспортируется (проверено чтением кода каждого сервиса)

- Общее для всех 5 сервисов: `glossa_requests_total` (Counter, `service`/`endpoint`/`status_code`) + `glossa_request_latency_seconds` (Histogram, `service`/`endpoint`) — единообразный `_prometheus_middleware` в каждом `main.py`.
- `glossa_model_inference_latency_seconds` (Histogram, `service`/`model`) — подтверждено определено и вызывается (`.observe()`) в `services/cv_service/gesture_classifier.py` и `services/nlp_service/translator.py`. **Не найдено** в `services/asr_service/transcriber.py` или `services/tts_service/synthesizer.py` (`grep` по обоим файлам — 0 совпадений) — латентность модели ASR/TTS не видна отдельной метрикой, только внутри общего `glossa_request_latency_seconds`.
- Только в `api_gateway/main.py`: `glossa_translation_latency_seconds` (Histogram, `mode`/`domain`) и `glossa_active_websocket_connections` (Gauge, `service`).

### ⚠️ РАСХОЖДЕНИЕ — дашборд ссылается на несуществующую метрику

`infra/monitoring/grafana/dashboards/glossa-overview.json` — 6 панелей, одна
из них **"Redis Stream Lag"** с запросом `glossa_redis_stream_lag`. Эта
метрика **не определена нигде** ни в одном сервисе (`grep -rn
"redis_stream_lag" services/` — 0 совпадений). Панель всегда будет
показывать "No data".

Ни одна из 6 панелей не покрывает новые фичи этой сессии: latency flush
предложения (видна только как текстовая строка в логах `rsl_to_text
latency=...`, не как отдельный Prometheus-ряд), время генерации
`sign_video`, частоту class collapse.

Ещё два дашборда — `mlops.json`, `load_testing.json` — **не открывались в
рамках этого прохода**. Нужно: свериться, реальны ли их панели, тем же
способом, что и `glossa-overview.json`.

Кроме Prometheus/Grafana, в `infra/monitoring/` есть конфиги Loki,
Promtail и `otel-collector.yml` — но ни в одном сервисе нет импорта
`opentelemetry` (`grep -rn "opentelemetry" services/` — 0 совпадений). Эта
часть observability-стека, похоже, провижионится инфраструктурно, но
никогда не подключена к приложению — тот же паттерн, что и мёртвые тесты
в разделе 6 (инфраструктура/скаффолдинг создан, но не доведён до
использования в реальном коде).

---

## 9. Auth / multi-user

- `session_id` генерируется **исключительно на клиенте**: `clients/web/lib/pages/home_page.dart:81` — `final _sessionId = const Uuid().v4();`, заново при каждой загрузке страницы. Сервер никогда не валидирует, не подписывает и не аутентифицирует это значение — просто использует как ключ словаря/Redis.
- Изоляция между пользователями — исключительно через уникальность `session_id` как ключа. Никакой проверки владения/секретности — кто угодно, знающий строку `session_id`, может читать/писать состояние этой сессии.
- `grep -rniE "TODO.*auth|FIXME.*auth|api[_-]?key|jwt|bearer"` по всем `main.py`/`config.py` сервисов и Flutter-клиенту — **0 совпадений**. Ни закомментированных заготовок, ни TODO под auth нет вообще — это не "не доделано", это никогда не начиналось.
- `.env.example` не содержит ни одной auth-related переменной (нет `JWT_SECRET`, `API_KEY` и т.п.) — подтверждает то же самое с другой стороны.
- CORS в `services/api_gateway/main.py:48-53` — `allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]` — полностью открыт, любой источник может дёргать API.

---

## ⚠️ Дополнительная находка вне исходного плана: `.env.example` устарел

`.env.example` (корень репо) не синхронизирован с реальными дефолтами кода:

- `WINDOW_SIZE=32`, `WINDOW_STRIDE=15` — реальный прод сейчас `WINDOW_SIZE=64`
  (`services/cv_service/config.py:3`, `docker-compose.yml:129`). Модель имеет
  фиксированную форму входа `T=64` — при попытке скопировать `.env.example`
  в `.env` буквально получите внутреннее рассогласование с тем, что реально
  задеплоено (этот же T=32/T=64 конфликт уже один раз проявился в этой сессии
  как явная ошибка формы тензора при прогоне `ab_compare_backends.py` против
  не того val-сета).
- Использует непрефиксованные имена (`DEVICE`, `COMPUTE_TYPE`) там, где
  `services/asr_service/config.py` реально читает `ASR_DEVICE`/`ASR_COMPUTE_TYPE`
  — копирование `.env.example` как есть тихо не применит эти настройки.
- Не содержит примерно 20 переменных, добавленных в этой сессии:
  `SENTENCE_PAUSE_SECONDS`, `MAX_SENTENCE_GLOSSES`, `SESSION_CLEANUP_INTERVAL`,
  `SESSION_IDLE_TTL`, `HAND_LOW_CONF_ZERO_THRESHOLD`, все `GESTURE_*`,
  `GLOSS_CLIPS_DIR`, `CLASS_TO_IDX_PATH`, `RTMLIB_MODE`/`RTMLIB_DEVICE`, все
  `TTA_*`/`SMOOTH_*`. Судя по содержимому, файл не обновлялся с очень раннего
  этапа проекта.

---

## Нашёл, но не смог объяснить / требует вашего решения

1. **`mlops/`, `experiments/`, `services/*/tests/` — что с ними делать?**
   Это не мелкий мусор: `mlops/` содержит полноценный (읽абельный, с реальной
   логикой) альтернативный training-pipeline с MLflow-трекингом, ONNX-registry,
   promotion-логикой; `experiments/` — 9 пронумерованных "ablation"-экспериментов
   с notebook + `run.py` + результатами. Но `experiments/01_gesture_backbone/run.py:142`
   содержит комментарий **`# ── Simulated results for dry-run (match diploma
   section 3.3) ──`**, а закоммиченные `experiments/results/01_gesture_backbone/results.json`
   показывают `top1_accuracy: 0.891` **одинаковым** и для STGCN_ONNX, и для
   S3D_Sber (два разных архитектурных семейства с бит-в-бит идентичной
   точностью — статистически такого не бывает у реально измеренных экспериментов).
   Это похоже на **заранее подогнанные под текст диплома синтетические цифры**,
   а не измерения. Совпадает ли что-то из `experiments/results/*.json` с тем,
   что реально попадёт в текст защиты? Если да — эти цифры нужно либо
   перепроверить реальным прогоном, либо явно не выдавать за измеренные.
   Отдельно: `services/*/tests/` (14 файлов) ссылаются на архитектуру
   (`app.graphs.orchestrator`, LangGraph multi-agent NLP, `InferenceQueue`),
   которой в проде никогда не было в текущем виде — оставить как исторический
   артефакт, удалить, или это была реальная более ранняя версия, к которой вы
   хотите вернуться?

2. **NLP реально на GPU или CPU прямо сейчас?** Логи 10-секундного flush
   сильно намекают на CPU, `docker-compose.yml:202` жёстко фиксирует
   `NLP_DEVICE: cpu` в базовом файле — но без свежего лога старта
   `nlp-service` я не могу утверждать однозначно.

3. **Лицензия Slovo на редистрибуцию клипов** — юридический вопрос, не
   технический; код тут ничего не подскажет, нужно смотреть реальные условия
   датасета на Kaggle.

4. **`.env.example`** — обновить сейчас или отложить (файл явно никто не
   поддерживал долгое время, но это не блокирует текущий деплой, раз
   `docker-compose.yml` не читает `.env` для большинства уже захардкоженных
   там значений).
