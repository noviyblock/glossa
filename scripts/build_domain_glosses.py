"""Строит data/domain_glosses_full.json и data/domain_glosses_200.json
из реальной таксономии доменов Slovo (вместо захардкоженных в ноутбуке
произвольных русских слов).

Источник: data/domains_raw.txt — сырой текст таксономии (вставлен
пользователем, скопирован из интерфейса категорий датасета). Формат:
    <Название домена>
    Свернуть
    <слово1>
    <слово2>
    ...
    <Название следующего домена>
    ...

full.json   — домены как есть (для сценария 'full', 1000 классов).
200.json    — каждый домен пересечён с data/selected_classes_200.json
              (сценарий 'top200'); домены, где пересечение пустое,
              исключаются (с явным предупреждением в stdout — без
              тихого выпадения данных).

Запуск: python scripts/build_domain_glosses.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
RAW_PATH = ROOT / "data" / "domains_raw.txt"
FULL_OUT = ROOT / "data" / "domain_glosses_full.json"
TOP200_OUT = ROOT / "data" / "domain_glosses_200.json"
SELECTED_200_PATH = ROOT / "data" / "selected_classes_200.json"

# Названия доменов — фиксированная таксономия (используется только для
# разбора сырого текста на границы доменов, сами слова не хардкодятся).
DOMAIN_HEADERS = [
    "Алфавит",
    "Абстрактные понятия",
    "Время и даты",
    "Действия и состояние",
    "Деньги и финансы",
    "Дом и быт",
    "Еда и напитки",
    "Закон и порядок",
    "Здоровье и медицина",
    "Инструменты и материалы",
    "Искусство и культура",
    "Коммуникация и речь",
    "Космос и астрономия",
    "Личность и характер",
    "Мифология и религия",
    "Музыка и звуки",
    "Мысли и идеи",
    "Образование и наука",
    "Одежда и аксессуары",
    "Ощущения и восприятие",
    "Природа и животные",
    "Путешествия и транспорт",
    "Работа и карьера",
    "Спорт и физическая активность",
    "Строительство и архитектура",
    "Технологии и гаджеты",
    "Форма и размер",
    "Цвет и свет",
    "Человек и части тела",
    "Числовые величины",
    "Эмоции и чувства",
    "Другое",
]


def parse_domains(raw_text: str) -> dict[str, list[str]]:
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    headers = set(DOMAIN_HEADERS)
    domains: dict[str, list[str]] = {}
    current = None
    for line in lines:
        if line in headers:
            current = line
            domains.setdefault(current, [])
            continue
        if line == "Свернуть":
            continue
        if current is None:
            continue
        domains[current].append(line)

    # Дедуп с сохранением порядка (текст содержит дублирующиеся подсписки)
    for domain, words in domains.items():
        seen = set()
        deduped = []
        for w in words:
            key = w.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(w)
        domains[domain] = deduped

    missing = headers - domains.keys()
    if missing:
        print(f"[warn] Домены без слов в исходном тексте: {sorted(missing)}")
    return domains


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Не найден {RAW_PATH}. Положите туда сырой текст таксономии доменов."
        )
    raw_text = RAW_PATH.read_text(encoding="utf-8")
    domains_full = parse_domains(raw_text)

    total_words = sum(len(v) for v in domains_full.values())
    print(f"Доменов: {len(domains_full)}, всего слов: {total_words}")
    for d, words in domains_full.items():
        print(f"  {d}: {len(words)} слов")

    FULL_OUT.write_text(
        json.dumps(domains_full, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nСохранено: {FULL_OUT}")

    # ── top200: пересечение каждого домена с selected_classes_200.json ──────
    if not SELECTED_200_PATH.exists():
        print(f"[warn] {SELECTED_200_PATH} не найден — domain_glosses_200.json не создан.")
        return

    selected = json.loads(SELECTED_200_PATH.read_text(encoding="utf-8"))
    top200_set = {w.lower() for w in selected["classes"]}

    domains_200: dict[str, list[str]] = {}
    dropped = []
    for domain, words in domains_full.items():
        filtered = [w for w in words if w.lower() in top200_set]
        if filtered:
            domains_200[domain] = filtered
        else:
            dropped.append(domain)

    print(f"\n[top200] Доменов с пересечением: {len(domains_200)} / {len(domains_full)}")
    if dropped:
        print(f"[top200] Исключены (пустое пересечение): {dropped}")
    for d, words in domains_200.items():
        print(f"  {d}: {len(words)} слов")

    matched_total = sum(len(v) for v in domains_200.values())
    print(f"[top200] Всего слов после фильтра: {matched_total} / {len(top200_set)} классов покрыто")

    TOP200_OUT.write_text(
        json.dumps(domains_200, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Сохранено: {TOP200_OUT}")


if __name__ == "__main__":
    main()
