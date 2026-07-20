# Единый прогон: коммит/пуш здесь + инструкция для VM

Дата: 2026-07-18. Не добавлен в git (сам этот файл — инструкция, не отчёт с
данными). Объединяет и заменяет по шагам исполнения `RTMLIB_MODE_BENCHMARK.md`
и `SEGMENTER_CALIBRATION.md` (те остаются как справочный контекст/обоснование,
но порядок действий — здесь, одним заходом).

---

## 1. Что закоммитить и запушить здесь

Я НЕ выполняю коммит/пуш сам — ниже команды для вас (или скажите "давай"
и я выполню их сам). Проверил `git status`: dirty-файлы шире, чем только
сегодняшняя работа — некоторые изменения лежат некоммиченными ещё с
предыдущих задач этой сессии (TTL-очистка сессий в `main.py`, комментарий
в `scripts/build_colab_01a_notebook.py`). Разделяю на то, что **обязательно
нужно для этого прогона**, и то, что просто рядом лежит.

### Обязательно для VM-прогона (без этого команды из раздела 3-4 не сработают)

```bash
git add services/cv_service/config.py services/cv_service/main.py \
        docker-compose.yml scripts/measure_accuracy.py \
        data/gloss_clips_labels.csv
git commit -m "feat: DIAG_LOG_INTERVAL calibration knob + extraction-latency timing in measure_accuracy.py

Needed for the segmenter threshold calibration run and rtmlib
lightweight/balanced benchmark — DIAG log was undersampling short
segments (~30-frame cadence vs 20-150 frame segments), and
measure_accuracy.py previously only timed the classifier, not the
rtmlib extraction step the benchmark is actually about.

main.py's diff also carries the earlier TTL session-cleanup addition
(same file, unrelated but already-completed, uncommitted work)."
git push origin feature/top200-classes-t64
```

`main.py`'s diff includes the earlier TTL-очистка session state (задача уже
отмечена completed, но не была закоммичена) — не разделял её от
DIAG_LOG_INTERVAL хирургически построчно, обе правки в одном файле и обе
безопасны/уже проверены по отдельности.

### Рядом лежит, но НЕ обязательно для этого конкретного прогона

- `scripts/analyze_segmenter_calibration.py` — **не нужен на VM**, он
  анализирует логи ПОСЛЕ того, как вы их пришлёте, я запущу его здесь.
  Коммитить можно (это готовый инструмент), просто не блокер для VM-шагов.
- `docs/KNOWN_LIMITATIONS.md` (+ вся папка `docs/`) — чистая документация,
  не влияет на рантайм. Отдельная, уже завершённая задача.
- `services/cv_service/tests/` → `archive/...` (git mv), `.github/workflows/ci.yml`,
  `Makefile` — отдельная завершённая задача (GAPS_ARCHIVED.md), не влияет
  на cv-service контейнер.
- `services/api_gateway/orchestrator.py` + `services/api_gateway/tests/`,
  `services/nlp_service/translator.py`, `notebooks/colab_glossa_01a_...ipynb`,
  `mlops/pipelines/train_gesture.py` — другие завершённые задачи этой сессии,
  тоже некоммичены, но не пересекаются с сегодняшним прогоном.

Эти три группы можно закоммитить отдельными коммитами позже (или сейчас,
если хотите разом) — не включаю их в команду выше, чтобы не смешивать
разнородные изменения в одном коммите и не блокировать сегодняшний прогон
их дозакоммичиванием.

### НЕ коммитить

`AUDIT.md`, `FAKE_RESULTS_AUDIT.md`, `GAPS_ARCHIVED.md`,
`DIAGNOSIS_CLASS_COLLAPSE.md`, `RTMLIB_MODE_BENCHMARK.md`,
`SEGMENTER_CALIBRATION.md`, этот файл (`VM_RUN_PROTOCOL.md`) —
каждая из этих задач явно просила не добавлять отчёт в git.
`models/_backup_pre_rtmw_20260704_2242/` — незнакомый мне бэкап, не трогаю,
не знаю его происхождения/размера, не мой вызов решать.

---

## 2. На VM: подтянуть код и пересобрать образ

```bash
cd /path/to/glossa
git pull                      # подтягивает коммит из шага 1
docker compose build cv-service
docker compose up -d cv-service   # именно up -d, не restart -- иначе
                                   # новые/изменённые ENV из docker-compose.yml
                                   # не подхватятся существующим контейнером
docker compose logs --tail=20 cv-service   # убедиться, что стартовал чисто
```

---

## 3. Единый контролируемый прогон (закрывает 3 вопроса разом)

### 3.1 Подготовка (до включения камеры)

1. Составьте список **15-20 жестов** из словаря 200 классов
   (`data/idx_to_class.json`), для каждого заранее решите скорость показа:
   быстро / нормально / медленно — вперемешку, не блоками (иначе скорость
   и порядок показа переменные, а сравнение по скорости легко спутать с
   дрейфом внимания/усталостью за время прогона).
2. Запишите CSV `ground_truth.csv`:
   ```
   order,gloss,speed
   1,"6 часов",normal
   2,"7:45",fast
   3,"MakDonalds",slow
   ...
   ```
   Порядок строк = порядок реального показа, один в один.

### 3.2 Включить плотное логирование ТОЛЬКО на время прогона

```bash
DIAG_LOG_INTERVAL=1 docker compose up -d cv-service
```
(благодаря правке в docker-compose.yml из шага 1 — раньше эта переменная
была хардкод-константой 30 внутри `main.py`, теперь читается из окружения
с дефолтом 30, так что это безопасно и обратимо одной командой.)

### 3.3 Прогон

Покажите жесты в камеру **строго в зафиксированном порядке**, с паузой
между каждым (дайте сегментатору полностью вернуться в IDLE — не начинайте
следующий жест, пока предыдущий явно не завершился, иначе эпизоды могут
слиться и порядковое сопоставление с CSV собьётся).

### 3.4 Снять лог и вернуть интервал обратно

```bash
docker compose logs cv-service --since <время начала прогона> > controlled_run.log
docker compose up -d cv-service   # без DIAG_LOG_INTERVAL в окружении -- вернётся дефолт 30
```

(Проще и надёжнее: `docker compose logs cv-service > controlled_run.log`
без `--since`, если это был единственный/последний прогон сессии — тогда
не нужно ловить точное время начала.)

---

## 4. rtmlib lightweight vs balanced (камера не нужна, можно параллельно/после)

Команды и обоснование — в `RTMLIB_MODE_BENCHMARK.md`, без изменений (уже
готовы, используют `data/gloss_clips_labels.csv` из этого же коммита):

```bash
docker compose run --rm -e RTMLIB_MODE=lightweight \
  -v "$(pwd)/scripts:/repo/scripts:ro" \
  -v "$(pwd)/services/cv_service:/repo/services/cv_service:ro" \
  --entrypoint python cv-service /repo/scripts/measure_accuracy.py \
  --clips-dir /models/gloss_clips --labels-csv /data/gloss_clips_labels.csv \
  --norm-stats /models/norm_stats.npz --ov-xml /models/stgcn_topk_int8/stgcn_topk_int8.xml \
  --onnx /models/gesture_classifier_mobile.onnx --class-map /data/idx_to_class.json \
  2>&1 | tee lightweight_bench.log

docker compose run --rm -e RTMLIB_MODE=balanced \
  -v "$(pwd)/scripts:/repo/scripts:ro" \
  -v "$(pwd)/services/cv_service:/repo/services/cv_service:ro" \
  --entrypoint python cv-service /repo/scripts/measure_accuracy.py \
  --clips-dir /models/gloss_clips --labels-csv /data/gloss_clips_labels.csv \
  --norm-stats /models/norm_stats.npz --ov-xml /models/stgcn_topk_int8/stgcn_topk_int8.xml \
  --onnx /models/gesture_classifier_mobile.onnx --class-map /data/idx_to_class.json \
  2>&1 | tee balanced_bench.log
```

Не трогает живой cv-service (отдельный одноразовый контейнер) — можно
запускать до/после/во время раздела 3, не мешает контролируемому прогону.

---

## 5. Что прислать обратно

1. `ground_truth.csv` (раздел 3.1)
2. `controlled_run.log` (раздел 3.4) — полный, не обрезанный
3. `lightweight_bench.log`, `balanced_bench.log` (раздел 4) — если успеете

Если время поджимает — по вашей же расстановке приоритетов: **разделы
3.1-3.4 важнее раздела 4** (один прогон закрывает 3 вопроса, bench — один
и без подозрения на баг, просто неизвестный trade-off).

---

## 6. Что я сделаю с этим по возвращении

```bash
python scripts/analyze_segmenter_calibration.py controlled_run.log --ground-truth ground_truth.csv
```
даёт: таблицу жест/скорость/segment_len/verdict, распределение segment_len
по correct/incorrect/discarded, last_activity-ряд на каждый эпизод (теперь
плотный благодаря `DIAG_LOG_INTERVAL=1`).

Дополнительно, раз в CSV есть колонка `speed` — проверю гипотезу про
forearm-length-scale (`docs/KNOWN_LIMITATIONS.md`, раздел "Normalizer
hip-center/shoulder-scale index mismatch"): коррелирует ли `speed=fast` с
более высокой долей `incorrect`/`discarded`, чем `speed=slow` — это прямая,
пусть и предварительная (n=15-20, не production-scale) проверка гипотезы
"быстрое сгибание руки → больше шума в масштабном референсе → больше
ошибок", а не просто эвристика по частоте классов.

Если корреляций (по segment_len ИЛИ по speed) не найдётся — прямо напишу
"перекалибровка threshold'ов не объясняет проблему" / "гипотеза про
forearm-length-scale не подтвердилась на этих данных", а не буду менять
`GESTURE_ONSET_THRESHOLD`/`GESTURE_OFFSET_THRESHOLD`/`normalizer.py` без
доказанного эффекта.
