# Архив: тесты и observability-конфиг под дорефакторную архитектуру

Здесь лежат тесты (`cv_service/tests`, `asr_service/tests`, `nlp_service/tests`,
`tts_service/tests`) и `otel-collector.yml`, написанные под архитектуру
`services/*/app/` (domain-driven, с LangGraph multi-agent NLP-оркестратором —
`intent_classifier`/`safety_filter`/`medical_simplifier`/`banking_simplifier`/
`rag_retriever`/`hallucination_checker` — и `InferenceQueue` с backpressure
для cv_service), которая была удалена коммитом `0424891`
("refactor: flat-file services, remove stale code... Replace services/*/app/
with flat-file modules", 2026-06-20) в пользу текущих плоских модулей
(`main.py`/`config.py`/`translator.py` и т.д.). Тесты импортировали из
пакета `app.*`, который тем коммитом удалён и не воссоздан — на текущем коде
они падают на первой же строке импорта (`ModuleNotFoundError: No module
named 'app'`), поэтому перемещены сюда, а не удалены: если решите вернуться
к multi-agent NLP-оркестрации или к очереди инференса с backpressure для
cv_service, здесь есть готовый референс того, как это уже было спроектировано
и протестировано один раз, прежде чем архитектуру упростили. `otel-collector.yml`
добавлен тем же порядком, что и тесты — он провижионился в
`infra/monitoring/`, но ни один из текущих flat-file сервисов не импортирует
`opentelemetry` (реальная OTel-инструментация существует только в
неиспользуемом `libs/common/glossa_common/telemetry/`, на который тоже
никто не ссылается) — конфиг был мёртвым scaffolding'ом без потребителя,
а не забытой, но рабочей частью системы.
