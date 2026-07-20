# Перемещённые артефакты дорефакторной архитектуры

Дата: 2026-07-18. Итог задачи по архивации нерабочих тестов и
неподключённого observability-конфига. Не добавлен в git (как `AUDIT.md`
и `FAKE_RESULTS_AUDIT.md`).

## Что перемещено (`git mv`, история сохранена)

| Откуда | Куда |
|---|---|
| `services/cv_service/tests/` | `archive/legacy_tests_pre_flatfile_refactor/cv_service/tests/` |
| `services/asr_service/tests/` | `archive/legacy_tests_pre_flatfile_refactor/asr_service/tests/` |
| `services/nlp_service/tests/` | `archive/legacy_tests_pre_flatfile_refactor/nlp_service/tests/` |
| `services/tts_service/tests/` | `archive/legacy_tests_pre_flatfile_refactor/tts_service/tests/` |
| `infra/monitoring/otel-collector.yml` | `archive/legacy_tests_pre_flatfile_refactor/otel-collector.yml` |

Структура внутри каждого `<service>/tests/` не менялась — 1:1 копия
исходного дерева (`test_domain/`, `test_pipeline/`, `test_agents/` и т.д.
со своими `__init__.py`), только сменился корень. `README.md` в корне
архива объясняет происхождение (коммит `0424891`) и зачем сохранено, а не
удалено.

## Почему `otel-collector.yml` — тоже в архив, а не подключён

Решение по развилке из задачи: **архивировать, не подключать**. Причины:

1. `otel-collector.yml` **не был указан ни в одном docker-compose файле**
   (`docker-compose.yml`, `docker-compose.gpu.yml`, `docker-compose.prod.yml`,
   `docker-compose.override.yml`) как сервис — контейнер otel-collector
   никогда не запускался в текущем стеке, не только код его не использовал.
2. Единственные ссылки на него в репозитории — тоже дорефакторные
   артефакты: `scripts/deploy.sh:60` (`$COMPOSE up -d redis qdrant
   otel-collector prometheus grafana loki promtail jaeger mlflow nginx`) и
   `libs/common/glossa_common/config/base.py:25`
   (`otel_exporter_otlp_endpoint`). `scripts/deploy.sh` сам по себе
   ссылается на `qdrant`/`jaeger`/`nginx` — сервисы, которых тоже нет в
   текущем docker-compose — это отдельный, ещё один нерабочий деплой-скрипт
   от старой архитектуры, не тот процесс, которым реально пользовались в
   этой сессии (реальный деплой — ручные `docker compose` команды на VM
   через SSH). Не трогал `scripts/deploy.sh` — вне заявленного объёма этой
   задачи, но **отмечаю отдельно**: возможно, стоит архивировать и его тем
   же способом при случае.
3. Реальная OTel-инструментация уже написана — `libs/common/glossa_common/
   telemetry/otel.py` и `metrics.py` — но весь `libs/common` (пакет
   `glossa_common`) не импортируется НИ ОДНИМ текущим flat-file сервисом
   (`grep -rn "glossa_common" services/` — 0 совпадений). Это значит
   "подключить минимальную инструментацию" на деле означало бы не написать
   несколько строк, а реанимировать целый неиспользуемый общий пакет —
   заметно больший объём работы, чем предполагает формулировка задачи
   "если время есть". Оставляю как находку для отдельного решения, не
   тяну в этот заход.

## CI/Makefile — поправлено, чтобы не ссылались на старый путь

- **`.github/workflows/ci.yml`** (шаг "Run tests", строка ~113) — реально
  запускал `pytest libs/ services/ ...`, что до перемещения падало на
  `ModuleNotFoundError: No module named 'app'` при сборе тестов. Путь
  `services/` не переименован (это не ссылка на конкретно старый путь, а
  общий скан), но добавлен комментарий, объясняющий, что тестов там больше
  нет и почему — шаг теперь будет падать на "no tests collected" вместо
  ошибки импорта, **сознательно оставлено падающим**, а не пропатчено
  `|| true` — тестов реально нет нигде в репозитории (ни в `libs/`, ни
  теперь в `services/`), маскировать это было бы нечестно.
- **`Makefile`** (`test:` цель) — ссылался на `tests/` верхнего уровня,
  которого в репозитории никогда не было (отдельно от архивации, эта
  ссылка была битой изначально) — убран, оставлен `services/` с тем же
  комментарием, что и в CI.
- **`.github/workflows/mlops.yml:76`** (`pytest mlops/tests/`) — **найдено,
  но не тронуто**: директории `mlops/tests/` не существует в репозитории
  вообще, не связано с архивацией `services/*/app/`-тестов и не входило в
  заявленный объём этой задачи. Отдельная битая ссылка, требует отдельного
  решения.

## Проверка: ничего активного не задето

- `grep -rn "app\.domain\|app\.graphs\|app\.inference"` за пределами
  архива — 0 совпадений (ни один живой модуль сервисов не импортировал
  архивированные тесты или то, что они тестировали).
- Ни один `docker-compose*.yml` не ссылался на `infra/monitoring/
  otel-collector.yml` по пути — перемещение не требует правки
  compose-файлов.
