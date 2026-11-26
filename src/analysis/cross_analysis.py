"""Анализ набора требований целиком: противоречия, дубли, несогласованность частей, полнота.

Большие документы режутся на чанки «фокус», но каждый чанк получает сжатый список
ВСЕХ остальных элементов — так конфликты между чанками не теряются.
"""
from __future__ import annotations

import logging
import re

from src.analysis import prompts
from src.analysis.acceptance import check_acceptance, check_explicit_rules
from src.analysis.llm_client import LLMClient, LLMError
from src.models.schemas import (
    Certainty, CompletenessFinding, CompletenessItemLLM, CompletenessResponse, CompletenessStatus, CrossIssue,
    CrossResponse, IssueCategory, OmissionDecision, OmissionReviewResponse, ParsedDocument, Requirement, Severity,
    TWO_SOURCE_CATEGORIES,
)
from src.validation.validator import (
    DocIndex, RefResolver, dedupe_cross_issues, false_arithmetic_issue, validate_completeness_item, validate_cross_issue,
)

log = logging.getLogger(__name__)

CROSS_BUDGET_CHARS = 40_000
COMPLETENESS_BUDGET_CHARS = 60_000
_CONFLICT_CLAIM = re.compile(r"противореч|конфликт|contradict|conflict", re.IGNORECASE)


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
    deterministic = check_acceptance(requirements, language) + check_explicit_rules(requirements, language)
    raw_issues = [
        validate_cross_issue(raw, f"X-tmp-{i}", index, resolver)
        for i, raw in enumerate(deterministic)
    ]
    checked_findings = "\n".join(
        f"{', '.join(issue.related_ids)}: {issue.explanation}" for issue in raw_issues
    )
    errors: list[str] = []
    for k, chunk in enumerate(chunks, start=1):
        ids = {r.report_id for r in chunk}
        others = [r for r in requirements if r.report_id not in ids]
        info = f"Chunk {k} of {len(chunks)}." if len(chunks) > 1 else ""
        try:
            resp = client.generate(system=prompts.cross_system(language),
                                   prompt=prompts.cross_prompt(doc.name, chunk, others, info, checked_findings),
                                   schema=CrossResponse)
        except LLMError as exc:
            msg = f"Анализ набора требований (часть {k}/{len(chunks)}) не выполнен: {exc}"
            log.error(msg)
            errors.append(msg)
            continue
        for raw in resp.issues:
            issue = validate_cross_issue(raw, f"X-tmp-{len(raw_issues)}", index, resolver)
            if not false_arithmetic_issue(raw, issue, resolver):
                raw_issues.append(issue)
    suspects = [issue for issue in raw_issues if issue.category in TWO_SOURCE_CATEGORIES
                and issue.certainty == Certainty.POTENTIAL and any(not e.verified for e in issue.evidence)
                and any(e.verified for e in issue.evidence)]
    if len(chunks) == 1 and suspects:
        verified = [issue for issue in raw_issues if issue.certainty == Certainty.CONFIRMED]
        existing = "\n".join(f"{', '.join(issue.related_ids)}: {issue.explanation}" for issue in verified)
        candidates = "\n".join(f"- {issue.explanation}" for issue in suspects)
        retry_prompt = (
            prompts.cross_prompt(doc.name, requirements, [], checked_findings=existing)
            + "\n\n=== RECHECK UNVERIFIED CANDIDATES ===\n" + candidates
            + "\nRe-evaluate these candidates against the full text above. The candidate descriptions may contain "
              "bad IDs or categories. Cite both actual sources using the IDs directly above the quotes. Compare "
              "what happens for an input between different boundaries. Return only newly verified issues. "
              "If a claim has no two supporting texts, omit it."
        )
        try:
            resp = client.generate(system=prompts.cross_system(language), prompt=retry_prompt, schema=CrossResponse)
        except LLMError as exc:
            log.warning("Не удалось перепроверить цитаты межтребовательных дефектов: %s", exc)
        else:
            recovered = [
                validate_cross_issue(raw, f"X-recheck-{i}", index, resolver)
                for i, raw in enumerate(resp.issues)
            ]
            raw_issues = _reconcile_recheck(raw_issues, suspects, recovered)
    return dedupe_cross_issues(raw_issues), not errors, errors


def _reconcile_recheck(issues: list[CrossIssue], suspects: list[CrossIssue],
                       recovered: list[CrossIssue]) -> list[CrossIssue]:
    confirmed = [issue for issue in issues if issue.certainty == Certainty.CONFIRMED]
    known_pairs = {frozenset(issue.related_ids) for issue in confirmed if issue.category in TWO_SOURCE_CATEGORIES}
    accepted = []
    for issue in recovered:
        pair = frozenset(issue.related_ids)
        if (issue.certainty == Certainty.CONFIRMED and issue.category in TWO_SOURCE_CATEGORIES
                and len(pair) == 2 and pair not in known_pairs):
            accepted.append(issue)
            known_pairs.add(pair)
    # Успешная перепроверка заменяет записи с неполным доказательством.
    return [issue for issue in issues if issue not in suspects] + accepted


def run_completeness(client: LLMClient, doc: ParsedDocument, index: DocIndex, resolver: RefResolver,
                     requirements: list[Requirement], language: str = "ru",
                     budget: int = COMPLETENESS_BUDGET_CHARS,
                     cross_issues: list[CrossIssue] | None = None) -> tuple[list[CompletenessFinding], bool, list[str]]:
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
    _link_covered_conflicts(findings, cross_issues or [], resolver, language)
    _verify_omissions(client, findings, doc, index, resolver, requirements, language, budget)
    return findings, True, []


def _link_covered_conflicts(findings: list[CompletenessFinding], cross_issues: list[CrossIssue],
                            resolver: RefResolver, language: str) -> None:
    """Область описана, если модель сослалась только на уже зарегистрированный конфликт."""
    for item in findings:
        if item.certainty is None or not _CONFLICT_CLAIM.search(item.explanation):
            continue
        refs = resolver.mentioned_ids(item.explanation)
        for ev in item.evidence:
            if ev.verified and ev.line_start is not None and ev.line_end is not None:
                req = resolver.requirement_at(ev.line_start, ev.line_end)
                if req is not None:
                    refs.add(req.report_id)
        matches = [
            x for x in cross_issues if x.certainty == Certainty.CONFIRMED
            and x.category in TWO_SOURCE_CATEGORIES and x.category != IssueCategory.DUPLICATE
            and len(x.related_ids) > 1 and set(x.related_ids) <= refs
        ]
        if not matches or refs != {rid for match in matches for rid in match.related_ids}:
            continue
        # Смешанное замечание о конфликте и отдельном пропуске ещё требует проверки.
        if re.search(r"отсутств|не указан|не описан|не определ[её]н|\bmissing\b|not specified", item.explanation, re.I):
            continue
        item.status = CompletenessStatus.COVERED
        item.certainty = None
        item.severity = Severity.NONE
        item.explanation = (
            ("The rules are specified. See the conflict analysis: " if language == "en"
             else "Правила описаны. Противоречие разобрано в разделе 4.1: ")
            + ", ".join(match.issue_id for match in matches) + "."
        )
        item.recommendation = ""


def _verify_omissions(client: LLMClient, findings: list[CompletenessFinding], doc: ParsedDocument,
                      index: DocIndex, resolver: RefResolver, requirements: list[Requirement],
                      language: str, budget: int) -> None:
    candidates = [item for item in findings if item.status == CompletenessStatus.CONFIRMED_OMISSION]
    if not candidates:
        return
    anchors = "\n".join(prompts.requirement_block(r, full=False, limit=0) for r in requirements)
    decisions = {}
    if len(doc.text) + len(anchors) <= budget:
        alleged = "\n".join(f"- {item.area.value}: {item.explanation}" for item in candidates)
        try:
            response = client.generate(
                system=prompts.omission_review_system(language),
                prompt=f"=== FULL SOURCE ===\n{doc.text}\n\n=== REPORT IDS AND LINES ===\n{anchors}"
                       f"\n\n=== PROPOSED OMISSIONS ===\n{alleged}",
                schema=OmissionReviewResponse,
            )
        except LLMError as exc:
            log.warning("Не удалось перепроверить заявленные пропуски: %s", exc)
        else:
            for area in {item.area for item in response.items}:
                matches = [item for item in response.items if item.area == area]
                if len(matches) == 1:
                    decisions[area] = matches[0]
    for item in candidates:
        review = decisions.get(item.area)
        if review and review.decision == OmissionDecision.SUPPORTED:
            continue
        if review and review.decision == OmissionDecision.COVERED:
            verified = validate_completeness_item(CompletenessItemLLM(
                area=item.area, status=CompletenessStatus.COVERED, explanation=review.explanation,
                evidence=review.evidence,
            ), index, resolver)
            if verified.evidence and all(ev.verified for ev in verified.evidence):
                item.status = CompletenessStatus.COVERED
                item.severity = Severity.NONE
                item.certainty = None
                item.explanation = verified.explanation
                item.evidence = verified.evidence
                item.recommendation = ""
                item.notes = verified.notes
                continue
        item.status = CompletenessStatus.POTENTIAL_OMISSION
        item.certainty = Certainty.POTENTIAL
        item.severity = Severity.LOW
        item.notes.append("Независимая проверка не подтвердила отсутствие сценария или обязательного условия."
                          if language == "ru" else "The second pass did not confirm this omission.")
