# GPU-путь: протокол проверки на реальной VM

Дата: 2026-07-20. Не добавлен в git (инструкция, не отчёт с данными — тот же
паттерн, что `VM_RUN_PROTOCOL.md`).

## Зачем

Задача #28 ("Wire real GPU build for cv-service rtmlib extraction") числится
pending с самого начала работы над GPU-путём. Код и конфигурация на месте
(`services/cv_service/Dockerfile`'s `deps-gpu`/`production-gpu` таргеты,
`docker-compose.gpu.yml`, device-placement логирование в `main.py:85-100`) —
но ни разу не подтверждено рабочим на реальном GPU, потому что в этой
песочнице нет CUDA-устройства (`CUDAExecutionProvider` отсутствует у
`onnxruntime`, проверено ранее в сессии). Этот документ — то, что нужно
прогнать на VM с реальной видеокартой, и что мне прислать обратно.

## 1. Собрать и поднять GPU-профиль

```bash
cd /path/to/glossa
git pull   # подтягивает коммит "fix(gpu): stop forcing RTMLIB_MODE=balanced..."
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build cv-service
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d cv-service
docker compose logs --tail=50 cv-service
```

## 2. Что искать в логах (успех/провал видно сразу)

Ищите строку `onnxruntime available providers: [...]` (`main.py:94`):

- **Успех**: `CUDAExecutionProvider` есть в списке, и **нет** следующей за
  ней строки `RTMLIB_DEVICE=cuda requested but CUDAExecutionProvider is NOT
  in onnxruntime's available providers`.
- **Провал**: список провайдеров содержит только `CPUExecutionProvider`
  (и, возможно, `AzureExecutionProvider` — так же, как в этой песочнице) —
  значит GPU-сборка де-факто не даёт CUDA рантайму, несмотря на
  `RTMLIB_DEVICE=cuda`. В `Dockerfile`'s `deps-gpu`-таргете уже
  задокументированы три реальных прошлых сбоя этого рода (порядок установки
  onnxruntime-gpu/CPU, версия CUDA13/12 vs `onnxruntime-gpu<1.21`,
  `LD_LIBRARY_PATH` для cudnn/cuda_runtime/cublas/curand/cufft) — если
  провал, для начала перепроверить именно эти три места, не гадать заново.

## 3. Латентность CPU vs GPU на одинаковом наборе клипов

`scripts/measure_accuracy.py` уже тайминг extraction-шага отдельно от
классификатора (`ONNXExecutionProvider` для классификатора всегда CPU
намеренно, см. `gesture_classifier.py:72` — сравнение имеет смысл только
для rtmlib-экстракции, не для классификации):

`measure_accuracy.py` и `services/cv_service` не скопированы в образ (образ
собирает только `/app` из `services/cv_service` на момент сборки — `scripts/`
туда вообще не входит), поэтому нужны volume mounts, иначе `python: can't
open file '/repo/scripts/measure_accuracy.py'` (уже наступили на это в этой
сессии — см. `VM_RUN_PROTOCOL.md`, где эти mounts уже были, а здесь я их
по невнимательности не скопировал в первой версии этого файла):

```bash
# CPU baseline (обычный docker-compose.yml, без GPU-профиля)
docker compose run --rm \
  -v "$(pwd)/scripts:/repo/scripts:ro" \
  -v "$(pwd)/services/cv_service:/repo/services/cv_service:ro" \
  --entrypoint python cv-service /repo/scripts/measure_accuracy.py \
  --clips-dir /models/gloss_clips --labels-csv /data/gloss_clips_labels.csv \
  --norm-stats /models/norm_stats.npz --ov-xml /models/stgcn_topk_int8/stgcn_topk_int8.xml \
  --onnx /models/gesture_classifier_mobile.onnx --class-map /data/idx_to_class.json \
  2>&1 | tee cpu_bench.log

# GPU (профиль из шага 1 уже поднят)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm \
  -v "$(pwd)/scripts:/repo/scripts:ro" \
  -v "$(pwd)/services/cv_service:/repo/services/cv_service:ro" \
  --entrypoint python cv-service /repo/scripts/measure_accuracy.py \
  --clips-dir /models/gloss_clips --labels-csv /data/gloss_clips_labels.csv \
  --norm-stats /models/norm_stats.npz --ov-xml /models/stgcn_topk_int8/stgcn_topk_int8.xml \
  --onnx /models/gesture_classifier_mobile.onnx --class-map /data/idx_to_class.json \
  2>&1 | tee gpu_bench.log
```

Точность (accuracy) между двумя прогонами меняться не должна вообще —
GPU меняет только скорость экстракции keypoints, не то, что экстрактор
находит. Если accuracy разошлась — это отдельная находка (баг), не
GPU/CPU-эффект, и стоит сообщить отдельно, не списывать на "погрешность".

## 4. Что прислать обратно

1. Полный вывод `docker compose logs cv-service` из шага 1 (не обрезанный —
   особенно строку про `available providers`).
2. `cpu_bench.log`, `gpu_bench.log` из шага 3.

## 5. Что я сделаю по возвращении

Сравню `extraction_latency_ms` (или как называется поле в выводе
`measure_accuracy.py` — сверю по факту получения логов) между
`cpu_bench.log`/`gpu_bench.log`, подтвержу совпадение accuracy между ними, и
закрою задачу #28 как подтверждённую — либо, если providers-строка покажет
провал, распишу, какой из трёх задокументированных в Dockerfile сценариев
сбоя воспроизвёлся.
