# rtmlib `lightweight` vs `balanced` — benchmark handoff

Дата: 2026-07-18. Не добавлен в git.

## РЕЗУЛЬТАТ (обновлено 2026-07-18, прогон на glossa-vm-demo)

Оба режима прогнаны на всех 200 клипах `models/gloss_clips/` (185
оценено, 15 пропущено как слишком короткие):

| | top-1 (single) | top-1 (TTA) | top-3 (TTA) | extraction/frame |
|---|---|---|---|---|
| lightweight | 0.9514 | 0.9514 | 0.9892 | 40.75ms |
| balanced    | 0.9459 | 0.9405 | 0.9946 | 132.75ms |

95%-доверительные интервалы по accuracy (биномиальные, n=185):
- lightweight: [0.920, 0.982]
- balanced: [0.913, 0.979]

Интервалы сильно перекрываются — разница в accuracy в пределах шума на
этой выборке. Разница в latency — НЕ шум (усреднено по 7321 кадру,
и архитектурно ожидаемо: `balanced` реально грузит более крупные модели —
`yolox_m` вместо `yolox_tiny`, `rtmw-dw-x-l` вместо `rtmw-dw-l-m`).

**Вердикт: `balanced` не стоит того на CPU** — платит гарантированные
+3.26x к extraction-latency за прирост accuracy, который даже статистически
не отличим от текущего дефолта. Комментарий в `config.py:13-20` обновлён
этими цифрами и датой, "NOT benchmarked" убрано. Дефолт (`lightweight`)
НЕ менял — вердикт как раз "не менять".

Не проверено этим прогоном: влияет ли `balanced` на dropout keypoints рук
именно на живом зашумлённом видео (этот бенчмарк — на чистых
предзаписанных клипах Slovo, где severe dropout, видимый в живых логах
сессии, не так выражен). Учитывая, что `balanced` не помог даже на чистом
входе — маловероятно, что поможет на зашумлённом, но это экстраполяция,
не измеренный факт.

Задача `#20` в трекере задач была помечена `completed` без реального
замера (см. "Побочная находка" ниже) — теперь `completed` подтверждён
настоящими цифрами.

## Что заблокировано здесь

Нужен реальный `rtmlib` с загрузкой ONNX-весов (RTMDet-m/RTMPose-m для
`balanced` — их ещё ни разу не скачивали, `RTMLIB_MODE` всегда был
`lightweight` в проде) и работающий `cv-service` контейнер. В этой
песочнице нет ни `rtmlib`, ни `cv2`, ни доступа к VM — выполнить сам не
могу. Ниже — всё, что можно было подготовить локально, плюс точные
команды для прогона на VM.

## Что подготовлено локально

1. **`data/gloss_clips_labels.csv`** — сгенерирован из
   `data/idx_to_class.json` (200 строк, `filename,gloss`, например
   `0.mp4,2 часа`). Даёт `measure_accuracy.py` эталонные метки для всех 200
   клипов `models/gloss_clips/` (по одному на класс, реальные Slovo-клипы,
   уже DVC-tracked с задачи gloss→video). Это n=200 — выше минимума
   15-20/режим из задачи, и это НЕ "на глаз по одному жесту".

   ⚠️ Ограничение: `gloss_clips` — по одному клипу на класс, т.е. это
   тест **между классами**, не повторные съёмки одного жеста разными
   дублями. Для устойчивости `balanced` к вариациям одного и того же
   жеста (что тоже важно) этого недостаточно — но для сравнения
   accuracy/latency между режимами на одинаковом входе годится: оба
   режима видят ИДЕНТИЧНЫЕ 200 клипов, разница в цифрах будет чисто от
   `RTMLIB_MODE`, не от вариативности набора.

2. **`scripts/measure_accuracy.py`** — доработан: раньше замерял
   latency только классификатора (ST-GCN, который от `RTMLIB_MODE` не
   зависит вообще — significant miss для этой задачи, поскольку разница
   `lightweight`/`balanced` целиком в extraction-шаге, не в классификации).
   Добавлен таймер вокруг `extractor.extract()` (сама rtmlib forward pass),
   репортится как `Avg extraction latency/frame` вместе с
   `rtmlib mode=... device=...` — теперь один прогон скрипта даёт и
   accuracy (top1/top3, с TTA и без), и latency extraction-шага. Проверено
   `py_compile` — не проверено end-to-end (нет rtmlib здесь).

## Команды для VM (обе — свежий одноразовый контейнер через `docker compose run`,
## НЕ трогает уже работающий прод cv-service, можно гонять без даунтайма)

```bash
cd /path/to/glossa   # корень репозитория на VM

# Прогон 1: lightweight (текущий дефолт)
docker compose run --rm \
  -e RTMLIB_MODE=lightweight \
  -v "$(pwd)/scripts:/repo/scripts:ro" \
  -v "$(pwd)/services/cv_service:/repo/services/cv_service:ro" \
  --entrypoint python cv-service /repo/scripts/measure_accuracy.py \
  --clips-dir /models/gloss_clips \
  --labels-csv /data/gloss_clips_labels.csv \
  --norm-stats /models/norm_stats.npz \
  --ov-xml /models/stgcn_topk_int8/stgcn_topk_int8.xml \
  --onnx /models/gesture_classifier_mobile.onnx \
  --class-map /data/idx_to_class.json \
  2>&1 | tee lightweight_bench.log

# Прогон 2: balanced (ПЕРВЫЙ раз скачает RTMDet-m/RTMPose-m веса — нужен
# сетевой доступ у контейнера; вес закэшируется в ./rtmlib_cache, второй
# прогон будет быстрее стартовать)
docker compose run --rm \
  -e RTMLIB_MODE=balanced \
  -v "$(pwd)/scripts:/repo/scripts:ro" \
  -v "$(pwd)/services/cv_service:/repo/services/cv_service:ro" \
  --entrypoint python cv-service /repo/scripts/measure_accuracy.py \
  --clips-dir /models/gloss_clips \
  --labels-csv /data/gloss_clips_labels.csv \
  --norm-stats /models/norm_stats.npz \
  --ov-xml /models/stgcn_topk_int8/stgcn_topk_int8.xml \
  --onnx /models/gesture_classifier_mobile.onnx \
  --class-map /data/idx_to_class.json \
  2>&1 | tee balanced_bench.log
```

Обе переменные (TTA, threshold'ы) остаются константными между прогонами —
скрипт использует `cfg.TTA_ENABLED` и остальные значения из
`services/cv_service/config.py`/окружения контейнера как есть, единственное,
что меняется между прогонами — `RTMLIB_MODE`.

Если позже нужен и GPU-путь (задача упоминает "если GPU-путь уже
стабилен") — то же самое, но `--entrypoint python cv-service` заменить на
образ `production-gpu` таргет (`docker compose -f ... build cv-service
--target production-gpu` уже должен существовать по Dockerfile) и добавить
`--gpus all -e RTMLIB_DEVICE=cuda`. Пока GPU-путь для cv-service
(задача #28) не подтверждён рабочим end-to-end в этой сессии — не включаю
эту команду как готовую, только как заметку на будущее.

## Что нужно прислать обратно

Оба лога (`lightweight_bench.log`, `balanced_bench.log`) целиком — из
"=== Results (...) ===" секции нужны: top-1/top-3 accuracy (с TTA и без),
`Avg extraction latency/frame`, и, если есть, список mistakes (для
проверки "near-miss vs FAR OFF" паттерна отдельно по режимам).

## Что сделаю по возвращении цифр (не додумываю сейчас)

- Если `balanced` даёт значимый прирост accuracy при приемлемой latency —
  явно предложу смену дефолта в `docker-compose.yml`/`config.py` (строки
  108-109 и `config.py:21`) как отдельное решение, с цифрами, НЕ поменяю
  сам.
- Если разница в пределах шума (n=200, но каждый класс представлен ОДНИМ
  клипом — оценю значимость через простой биномиальный интервал, не "на
  глаз") — обновлю комментарий `config.py:13-20`, уберу
  "NOT benchmarked", впишу реальные цифры и дату замера.

## Побочная находка

Задача `#20` в трекере задач помечена `completed` ("Evaluate rtmlib mode
presets for speed/quality tradeoff"), но комментарий в `config.py:17-20`
прямым текстом говорит "NOT benchmarked in this environment" и обе
переменные (`RTMLIB_MODE`) не тронуты с дефолта. ⚠️ РАСХОЖДЕНИЕ — либо
задача закрыта без реального замера, либо замер был, но не отражён ни в
коде, ни в артефактах репозитория. Эта задача, по сути, и есть повторное
закрытие #20 — с реальными цифрами в этот раз, если/когда они появятся.
