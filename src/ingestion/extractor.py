"""Гибридное извлечение требований.

1. Детерминированно: документ режется на элементы (пункты списков, строки таблиц,
   блоки кода, абзацы). Ничего не теряется: каждый содержательный элемент получает ID.
   Тип определяется эвристикой по заголовкам разделов.
2. LLM-уточнение: модель подтверждает/исправляет тип каждого элемента пакетами.
   Текст элемента LLM не переписывает, поэтому цитаты остаются дословными.
"""
from __future__ import annotations

import logging
import re
from collections import Counter

from src.analysis.llm_client import LLMClient, LLMError
from src.analysis import prompts
from src.models.schemas import (
    CONTEXT_TYPES, ClassificationResponse, ParsedDocument, Requirement, ReqType,
)

log = logging.getLogger(__name__)

_FENCE = re.compile(r"^\s*(```|~~~)")
_BULLET = re.compile(r"^\s*(?:[*+-]|\d+[.)])\s+(.*\S)\s*$")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_HR = re.compile(r"^\s*([-*_])\1{2,}\s*$")
_DOC_ID = re.compile(r"^[\s|*+>-]*\**\s*((?:[A-ZА-ЯЁ]{1,6}[-–‑]\d+(?:\.\d+)*))\s*[.:)]?\**(?:\s|\||$)")
_MARKUP = re.compile(r"[*`]|^\s*[-+]\s+|^\s*\d+[.)]\s+")

# Ключевые слова заголовков → тип. Порядок важен («нефункциональные» раньше «функциональные»).
_SECTION_RULES: list[tuple[tuple[str, ...], ReqType]] = [
    (("нефункциональн", "non-functional", "nonfunctional", "quality attribute"), ReqType.NON_FUNCTIONAL),
    (("функциональн", "functional requirement"), ReqType.FUNCTIONAL),
    (("критерии прием", "критерии приём", "acceptance"), ReqType.ACCEPTANCE),
    (("ревью требований", "чек-лист", "checklist", "блок согласующих", "согласующих", "approval", "sign-off"), ReqType.TEMPLATE),
    (("общие сведения", "паспорт", "document control", "revision history", "история изменений"), ReqType.METADATA),
    (("цели", "goal", "objective", "назначение"), ReqType.GOAL),
    (("ui", "дизайн", "интерфейс", "экран"), ReqType.UI),
    (("bpmn", "sequence", "процесс", "сценари", "workflow"), ReqType.PROCESS),
    (("архитектур", "architecture", "схема"), ReqType.ARCHITECTURE),
    (("данным", "модель данных", "data requirement", "data model"), ReqType.DATA),
    (("интеграц", "integration"), ReqType.INTEGRATION),
    (("тестирован", "testing"), ReqType.TESTING),
    (("rest api", "api", "параметр", "ответ", "статус", "логика работы метода", "особые правила", "request", "response"), ReqType.API),
    (("ограничени", "constraint"), ReqType.CONSTRAINT),
]


# --------------------------------------------------------------------------- #
# Нарезка на элементы
# --------------------------------------------------------------------------- #
def _plain(text: str) -> str:
    return re.sub(r"\s+", " ", _MARKUP.sub("", text)).strip()


def _shorten(text: str, limit: int = 140) -> str:
    if _TABLE_ROW.match(text.split("\n")[0]):
        text = text.strip().strip("|").replace("|", " · ")
    p = _plain(text)
    return p if len(p) <= limit else p[: limit - 1].rstrip() + "…"


def heuristic_type(doc: ParsedDocument, line: int) -> ReqType:
    """Тип по ближайшему (самому глубокому) заголовку, чьё название содержит ключевое слово."""
    chain = _heading_chain(doc, line)
    for title in reversed(chain):
        low = title.lower()
        for keywords, rtype in _SECTION_RULES:
            if any(_kw_in(k, low) for k in keywords):
                return rtype
    return ReqType.UNCLASSIFIED


def _kw_in(keyword: str, low_title: str) -> bool:
    if len(keyword) <= 3:   # «ui», «api» — только как отдельное слово
        return re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", low_title) is not None
    return keyword in low_title


def _heading_chain(doc: ParsedDocument, line: int) -> list[str]:
    return doc.section_path(line).split(" › ")


def _is_block_start(line: str) -> bool:
    s = line.strip()
    return (not s or bool(_FENCE.match(line)) or bool(_TABLE_ROW.match(line)) or bool(_DOC_ID.match(line))
            or bool(_BULLET.match(line)) or bool(_HR.match(line)))


def split_into_units(doc: ParsedDocument) -> list[Requirement]:
    """Детерминированная нарезка. report_id присваивается позднее, здесь — временный."""
    lines = doc.lines
    heading_lines = {h.line for h in doc.headings}
    units: list[Requirement] = []
    table_header: str = ""
    i, n = 0, len(lines)

    def add(text: str, start: int, end: int, context: str = "") -> None:
        text = text.strip()
        if not _plain(text):
            return
        units.append(Requirement(
            report_id="", doc_id=_find_doc_id(text),
            type=ReqType.CONTEXT if len(_plain(text)) <= 5 else heuristic_type(doc, start), text=text,
            short=_shorten(text), section=doc.section_path(start), line_start=start, line_end=end, context=context,
        ))

    while i < n:
        no, line = i + 1, lines[i]
        s = line.strip()
        if not s or no in heading_lines or _HR.match(line):
            table_header = ""
            i += 1
            continue
        if _FENCE.match(line):                      # блок кода (JSON, curl, схемы)
            j = i + 1
            while j < n and not _FENCE.match(lines[j]):
                j += 1
            body = "\n".join(lines[i + 1:j])
            add(body, no + 1, max(no + 1, j), context="блок кода")
            i = j + 1
            continue
        if _TABLE_ROW.match(line):                  # строка таблицы
            nxt = lines[i + 1] if i + 1 < n else ""
            if _TABLE_SEP.match(line):
                pass
            elif _TABLE_SEP.match(nxt) and not (i > 0 and _TABLE_ROW.match(lines[i - 1])):
                table_header = _plain(s.strip("|").replace("|", " | "))   # заголовок таблицы — контекст
            else:
                cells = [c.strip() for c in s.strip().strip("|").split("|")]
                if any(_plain(c) for c in cells):
                    add(s, no, no, context=f"колонки таблицы: {table_header}" if table_header else "строка таблицы")
            i += 1
            continue
        m = _BULLET.match(line)
        if m:                                       # пункт списка (+ продолжения с отступом)
            j = i + 1
            while j < n and lines[j].strip() and lines[j].startswith(("  ", "\t")) and not _BULLET.match(lines[j]) \
                    and not _TABLE_ROW.match(lines[j]) and not _FENCE.match(lines[j]) and (j + 1) not in heading_lines:
                j += 1
            add(_strip_marker("\n".join(x.rstrip() for x in lines[i:j])), no, j)
            i = j
            continue
        # абзац: склеиваем строки, пока нет жёсткого переноса/нового блока/нового жирного ярлыка
        j = i + 1
        while (j < n and not lines[j - 1].endswith("  ") and not _is_block_start(lines[j])
               and not lines[j].lstrip().startswith("**") and (j + 1) not in heading_lines):
            j += 1
        add("\n".join(x.strip() for x in lines[i:j]), no, j, context="")
        i = j
    return units


def _strip_marker(text: str) -> str:
    first, *rest = text.split("\n")
    first = re.sub(r"^\s*(?:[*+-]|\d+[.)])\s+", "", first)
    return "\n".join([first.strip(), *[r.strip() for r in rest]])


def _find_doc_id(text: str) -> str | None:
    m = _DOC_ID.match(text)
    return m.group(1).replace("–", "-").replace("‑", "-") if m else None


# --------------------------------------------------------------------------- #
# Нумерация и LLM-уточнение
# --------------------------------------------------------------------------- #
def assign_report_ids(units: list[Requirement]) -> list[Requirement]:
    width = max(3, len(str(len(units))))
    for k, u in enumerate(units, start=1):
        u.report_id = f"REQ-{k:0{width}d}"
    return units


_KEEP_TYPES = {ReqType.FUNCTIONAL, ReqType.NON_FUNCTIONAL, ReqType.ACCEPTANCE, ReqType.DATA, ReqType.INTEGRATION}


def merge_classification(unit: Requirement, llm_type: ReqType) -> ReqType:
    """Не даём LLM «потерять» явные требования: они не переводятся в контекстные типы."""
    protected = unit.doc_id is not None or unit.type in _KEEP_TYPES
    if protected and llm_type in CONTEXT_TYPES:
        return unit.type
    return llm_type


def classify_with_llm(units: list[Requirement], client: LLMClient, batch_size: int = 40) -> list[str]:
    """Уточняет типы пакетами. Возвращает список предупреждений (сбои не фатальны)."""
    warnings: list[str] = []
    for start in range(0, len(units), batch_size):
        batch = units[start:start + batch_size]
        by_id = {u.report_id: u for u in batch}
        try:
            resp = client.generate(system=prompts.CLASSIFY_SYSTEM, prompt=prompts.classify_prompt(batch),
                                   schema=ClassificationResponse)
        except LLMError as exc:
            msg = (f"Классификация элементов {batch[0].report_id}–{batch[-1].report_id} через LLM не выполнена "
                   f"({exc}); использована эвристика по заголовкам разделов.")
            log.error(msg)
            warnings.append(msg)
            continue
        counts = Counter(item.unit_id for item in resp.items)
        unexpected = counts.keys() - by_id.keys()
        if unexpected:
            warnings.append("Модель вернула ID за пределами пакета: " + ", ".join(sorted(unexpected)) + ".")
        valid = {item.unit_id: item.type for item in resp.items if counts[item.unit_id] == 1}
        for unit in batch:
            if unit.report_id not in valid:
                warnings.append(f"{unit.report_id}: отсутствует однозначная классификация модели. "
                                "Сохранён тип по структуре документа.")
                continue
            new_type = merge_classification(unit, valid[unit.report_id])
            if new_type != unit.type:
                log.debug("%s: %s → %s", unit.report_id, unit.type.value, new_type.value)
            unit.type = new_type
            unit.classified_by = "heuristic+llm"
    return warnings


def extract_requirements(doc: ParsedDocument, client: LLMClient | None = None) -> tuple[list[Requirement], list[str]]:
    units = assign_report_ids(split_into_units(doc))
    warnings: list[str] = []
    if client is not None and units:
        warnings = classify_with_llm(units, client)
    seen: dict[str, str] = {}
    for u in units:
        if u.doc_id:
            if u.doc_id in seen:
                warnings.append(f"Идентификатор «{u.doc_id}» встречается повторно ({seen[u.doc_id]} и {u.report_id}).")
            else:
                seen[u.doc_id] = u.report_id
    return units, warnings
