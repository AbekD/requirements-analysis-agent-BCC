"""Анализ набора требований целиком: противоречия, дубли, несогласованность частей, полнота.

Большие документы режутся на чанки «фокус», но каждый чанк получает сжатый список
ВСЕХ остальных элементов — так конфликты между чанками не теряются.
"""
from __future__ import annotations

import logging

from src.analysis import prompts
from src.analysis.llm_client import LLMClient, LLMError
from src.models.schemas import (
    CompletenessFinding, CompletenessResponse, CrossIssue, CrossResponse, ParsedDocument, Requirement,
)
from src.validation.validator import (
    DocIndex, RefResolver, dedupe_cross_issues, validate_completeness_item, validate_cross_issue,
)

log = logging.getLogger(__name__)

CROSS_BUDGET_CHARS = 40_000
COMPLETENESS_BUDGET_CHARS = 60_000


def build_chunks(requirements: list[Requirement], budget: int = CROSS_BUDGET_CHARS) -> list[list[Requirement]]:
    """Группы подряд идущих требований, чей полный текст укладывается в бюджет."""
    chunks: list[list[Requirement]] = []
    current: list[Requirement] = []
    size = 0
    for r in requirements:
        cost = len(r.text) + len(r.section) + 80
        if current and size + cost > budget:
            chunks.append(current)
            current, size = [], 0
        current.append(r)
        size += cost
    if current:
        chunks.append(current)
    return chunks


def run_cross_analysis(client: LLMClient, doc: ParsedDocument, index: DocIndex, resolver: RefResolver,
                       requirements: list[Requirement], language: str = "ru",
                       budget: int = CROSS_BUDGET_CHARS) -> tuple[list[CrossIssue], bool, list[str]]:
    """Возвращает (замечания, полнота_анализа, ошибки)."""
    chunks = build_chunks(requirements, budget)
    raw_issues: list[CrossIssue] = []
    errors: list[str] = []
    for k, chunk in enumerate(chunks, start=1):
        ids = {r.report_id for r in chunk}
        others = [r for r in requirements if r.report_id not in ids]
        info = f"Chunk {k} of {len(chunks)}." if len(chunks) > 1 else ""
        try:
            resp = client.generate(system=prompts.cross_system(language),
                                   prompt=prompts.cross_prompt(doc.name, chunk, others, info), schema=CrossResponse)
        except LLMError as exc:
            msg = f"Анализ набора требований (часть {k}/{len(chunks)}) не выполнен: {exc}"
            log.error(msg)
            errors.append(msg)
            continue
        for raw in resp.issues:
            raw_issues.append(validate_cross_issue(raw, f"X-tmp-{len(raw_issues)}", index, resolver))
    return dedupe_cross_issues(raw_issues), not errors, errors


def run_completeness(client: LLMClient, doc: ParsedDocument, index: DocIndex, resolver: RefResolver,
                     requirements: list[Requirement], language: str = "ru",
                     budget: int = COMPLETENESS_BUDGET_CHARS) -> tuple[list[CompletenessFinding], bool, list[str]]:
    text = doc.text
    is_full = len(text) <= budget
    digest = "" if is_full else "\n".join(prompts.requirement_block(r, full=False, limit=140) for r in requirements)
    try:
        resp = client.generate(system=prompts.completeness_system(language),
                               prompt=prompts.completeness_prompt(doc.name, text[:budget], is_full, digest),
                               schema=CompletenessResponse)
    except LLMError as exc:
        msg = f"Анализ полноты спецификации не выполнен: {exc}"
        log.error(msg)
        return [], False, [msg]
    findings = [validate_completeness_item(i, index, resolver) for i in resp.items]
    return findings, True, []
