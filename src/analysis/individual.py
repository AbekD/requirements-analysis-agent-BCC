"""Индивидуальная проверка требований с пакетным вызовом и перепроверкой пропусков."""
from __future__ import annotations

import logging
import re
from collections import Counter

from src.analysis import prompts
from src.analysis.llm_client import LLMClient, LLMError
from src.models.schemas import (
    AnalysisResult, Certainty, Criterion, IndividualAnalysis, IndividualBatchResponse, IndividualResponse,
    LocalOmissionReviewResponse, OmissionDecision, ParsedDocument, Requirement, Severity, Status,
    TWO_SOURCE_CATEGORIES,
)
from src.validation.validator import DocIndex, RefResolver, validate_evidence, validate_individual

log = logging.getLogger(__name__)
_RELATIONSHIP = re.compile(r"противореч|конфликт|дубл|несогласован|несоответств|conflict|contradict|duplicat|inconsisten", re.I)
_COVERED_ELSEWHERE = re.compile(
    r"(?:хотя|однако|но)\s+.{0,120}\b(?:указан\w*|описан\w*|предусмотрен\w*)|"
    r"(?:although|but)\s+.{0,120}\b(?:defined|described|specified)\b", re.I,
)


def link_cross_review(result: AnalysisResult, resolver: RefResolver, language: str) -> int:
    """Заменяет повтор межтребовательной проблемы ссылкой на её проверенное основание."""
    pairs = [
        (issue.issue_id, set(issue.related_ids)) for issue in result.cross_issues
        if issue.certainty == Certainty.CONFIRMED and issue.category in TWO_SOURCE_CATEGORIES
        and len(issue.related_ids) == 2
    ]
    linked = 0
    for analysis in result.individual:
        for assessment in analysis.assessments:
            if assessment.status not in (Status.FAIL, Status.UNCERTAIN):
                continue
            mentioned = resolver.mentioned_ids(assessment.explanation) - {analysis.req_id}
            if not mentioned:
                continue
            if (assessment.criterion == Criterion.COMPLETENESS
                    and _COVERED_ELSEWHERE.search(assessment.explanation)):
                assessment.status = Status.UNCERTAIN
                assessment.severity = Severity.LOW
                assessment.certainty = Certainty.POTENTIAL
                assessment.notes.append(
                    "The response cites coverage elsewhere. Confirm whether any part is still missing."
                    if language == "en" else "Модель указала, что правило описано в другом пункте. "
                    "Нужно уточнить, осталась ли действительно непокрытая часть."
                )
                continue
            if assessment.criterion not in (Criterion.CORRECTNESS, Criterion.CONSISTENCY, Criterion.NECESSITY,
                                             Criterion.UNAMBIGUITY, Criterion.VERIFIABILITY):
                continue
            if not _RELATIONSHIP.search(assessment.explanation):
                continue
            matches = [issue_id for issue_id, ids in pairs if ids <= mentioned | {analysis.req_id}]
            if not matches:
                assessment.status = Status.UNCERTAIN
                assessment.severity = Severity.LOW
                assessment.certainty = Certainty.POTENTIAL
                continue
            assessment.status = Status.NOT_APPLICABLE
            assessment.severity = Severity.NONE
            assessment.certainty = None
            assessment.evidence = []
            assessment.suggestion = ""
            assessment.notes = []
            assessment.explanation = (
                ("See the cross-requirement review: " if language == "en" else "Сопоставление требований приведено в разделе 4: ")
                + ", ".join(matches) + "."
            )
            linked += 1
    return linked


def build_context(doc: ParsedDocument, req: Requirement, max_chars: int = prompts.MAX_CONTEXT_CHARS) -> tuple[str, bool]:
    """Весь документ, если помещается; иначе оглавление + раздел требования. Возвращает (текст, это_весь_документ)."""
    full = doc.text
    if len(full) <= max_chars:
        return full, True
    outline = "\n".join(f"{'  ' * (h.level - 1)}- {h.title} (строка {h.line})" for h in doc.headings)
    starts = [h.line for h in doc.headings if h.line <= req.line_start]
    nexts = [h.line for h in doc.headings if h.line > req.line_start]
    first = starts[-1] if starts else 1
    last = (nexts[0] - 1) if nexts else len(doc.lines)
    section = "\n".join(doc.lines[first - 1:last])
    text = f"[ОГЛАВЛЕНИЕ]\n{outline}\n\n[РАЗДЕЛ ТРЕБОВАНИЯ, строки {first}-{last}]\n{section}"
    return text[:max_chars], False


def analyze_requirement(client: LLMClient, doc: ParsedDocument, index: DocIndex, req: Requirement,
                        language: str = "ru") -> IndividualAnalysis:
    context, is_full = build_context(doc, req)
    try:
        resp = client.generate(
            system=prompts.individual_system(language),
            prompt=prompts.individual_prompt(req, doc.name, context, is_full),
            schema=IndividualResponse,
        )
    except LLMError as exc:
        log.error("%s: анализ не выполнен: %s", req.report_id, exc)
        return IndividualAnalysis(req_id=req.report_id, ok=False, error=str(exc))
    return validate_individual(resp, req, index)


def analyze_batch(client: LLMClient, doc: ParsedDocument, index: DocIndex, requirements: list[Requirement],
                  language: str, resolver: RefResolver | None = None) -> list[IndividualAnalysis]:
    if len(requirements) == 1 or len(doc.text) > prompts.MAX_CONTEXT_CHARS:
        return [analyze_requirement(client, doc, index, req, language) for req in requirements]
    responses = {}
    try:
        batch = client.generate(
            system=prompts.individual_system(language, batch=True),
            prompt=prompts.individual_batch_prompt(doc.text, requirements),
            schema=IndividualBatchResponse,
        )
    except LLMError as exc:
        log.warning("Пакетная проверка недоступна, требования будут проверены отдельно: %s", exc)
    else:
        expected = {req.report_id for req in requirements}
        identified = []
        for item in batch.items:
            rid = item.req_id.strip()
            if rid not in expected and resolver is not None:
                req = resolver.resolve(rid)
                if req is not None:
                    rid = req.report_id
            if rid in expected:
                identified.append((rid, item.response))
        counts = Counter(rid for rid, _ in identified)
        responses = {rid: response for rid, response in identified if counts[rid] == 1}
        missing = expected - responses.keys()
        if missing:
            log.info("Повторная индивидуальная проверка ID: %s", ", ".join(sorted(missing)))
    return [
        validate_individual(responses[req.report_id], req, index)
        if req.report_id in responses else analyze_requirement(client, doc, index, req, language)
        for req in requirements
    ]


def verify_local_omissions(client: LLMClient, doc: ParsedDocument, index: DocIndex, resolver: RefResolver,
                           result: AnalysisResult, language: str) -> None:
    """Перепроверяет заявленные локальные пробелы, которым могут отвечать другие разделы."""
    candidates = {
        analysis.req_id: assessment for analysis in result.individual for assessment in analysis.assessments
        if assessment.criterion == Criterion.COMPLETENESS and assessment.status == Status.FAIL
    }
    if not candidates:
        return
    reviews = []
    if len(doc.text) <= prompts.MAX_CONTEXT_CHARS:
        index_text = "\n".join(prompts.requirement_block(req, full=False, limit=0) for req in result.requirements)
        claims = "\n".join(f"{req_id}: {a.explanation}" for req_id, a in candidates.items())
        try:
            reviews = client.generate(
                system=prompts.omission_review_system(language),
                prompt=f"=== FULL SOURCE ===\n{doc.text}\n\n=== REPORT IDS AND LINES ===\n{index_text}"
                       f"\n\n=== PROPOSED LOCAL OMISSIONS ===\n{claims}\nUse the target ID in req_id.",
                schema=LocalOmissionReviewResponse,
            ).items
        except LLMError as exc:
            log.warning("Не удалось перепроверить отдельные пропуски: %s", exc)
    counts = Counter(review.req_id for review in reviews)
    by_id = {review.req_id: review for review in reviews if counts[review.req_id] == 1}
    for req_id, item in candidates.items():
        review = by_id.get(req_id)
        if review and review.decision == OmissionDecision.SUPPORTED:
            continue
        if review and review.decision == OmissionDecision.COVERED:
            evidence, notes = validate_evidence(review.evidence, index, resolver)
            if evidence and all(ev.verified for ev in evidence):
                item.status = Status.PASS
                item.severity = Severity.NONE
                item.certainty = None
                item.explanation = review.explanation.strip()
                item.evidence = evidence
                item.notes = notes
                item.suggestion = ""
                continue
        item.status = Status.UNCERTAIN
        item.severity = Severity.LOW
        item.certainty = Certainty.POTENTIAL
        item.notes.append("Перепроверка по всему документу не подтвердила отсутствие информации."
                          if language == "ru" else "The second pass did not confirm this local omission.")
