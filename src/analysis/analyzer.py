"""Оркестратор: извлечение → индивидуальный анализ → набор → полнота → итог."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable

from src.analysis import prompts
from src.analysis.cross_analysis import run_completeness, run_cross_analysis
from src.analysis.findings import build_findings, findings_digest, stats_text
from src.analysis.individual import analyze_batch, link_cross_review, verify_local_omissions
from src.analysis.llm_client import LLMClient, LLMError
from src.ingestion.extractor import extract_requirements
from src.models.schemas import (
    SEVERITY_ORDER, AnalysisResult, Certainty, FinalAssessment, FinalResponse, ParsedDocument, RunStatus,
    Severity, Verdict,
)
from src.validation.validator import DocIndex, RefResolver

log = logging.getLogger(__name__)
Progress = Callable[[str], None]


def run_analysis(doc: ParsedDocument, client: LLMClient, agent_version: str, language: str = "ru",
                 progress: Progress | None = None, individual_batch_size: int = 1) -> AnalysisResult:
    say = progress or (lambda _msg: None)
    result = AnalysisResult(document_name=doc.name, document_sha256=doc.sha256,
                            started_at=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
                            model=client.model, agent_version=agent_version, language=language)

    say("1/5 Извлечение требований…")
    requirements, warnings = extract_requirements(doc, client)
    result.requirements, result.warnings = requirements, warnings
    if not requirements:
        result.status = RunStatus.FAILED
        result.errors.append("В документе не найдено ни одного элемента для анализа.")
        result.llm_calls = client.calls
        return result

    index = DocIndex(doc)
    resolver = RefResolver(requirements, doc)

    targets = [r for r in requirements if r.analyzable]
    batch_size = max(1, individual_batch_size)
    for start in range(0, len(targets), batch_size):
        batch = targets[start:start + batch_size]
        say(f"2/5 Индивидуальный анализ {start + 1}–{start + len(batch)}/{len(targets)}: "
            + ", ".join(req.report_id for req in batch))
        for analysis in analyze_batch(client, doc, index, batch, language, resolver):
            result.individual.append(analysis)
            if not analysis.ok:
                result.errors.append(f"{analysis.req_id}: индивидуальный анализ не выполнен — {analysis.error}")

    say("3/5 Проверка набора требований…")
    result.cross_issues, result.cross_complete, errs = run_cross_analysis(client, doc, index, resolver, requirements, language)
    result.errors += errs

    say("4/5 Проверка полноты спецификации…")
    result.completeness, result.completeness_complete, errs = run_completeness(
        client, doc, index, resolver, requirements, language, cross_issues=result.cross_issues,
    )
    result.errors += errs

    say("5/5 Итоговая оценка…")
    linked = link_cross_review(result, resolver, language)
    verify_local_omissions(client, doc, index, resolver, result, language)
    result.findings, result.duplicates_merged = build_findings(result)
    result.duplicates_merged += linked
    result.final = assess_final(client, result, language)

    ok = sum(1 for a in result.individual if a.ok)
    if ok == 0 and targets:
        result.status = RunStatus.FAILED
    elif result.errors or not result.final.complete:
        result.status = RunStatus.PARTIAL
    else:
        result.status = RunStatus.COMPLETE
    result.llm_calls = client.calls
    return result


def deterministic_verdict(result: AnalysisResult) -> Verdict:
    conf = [f for f in result.findings if f.certainty == Certainty.CONFIRMED]
    if any(f.severity == Severity.CRITICAL for f in conf):
        return Verdict.NOT_READY
    if any(f.severity == Severity.HIGH for f in conf) or result.errors:
        return Verdict.READY_WITH_RESERVATIONS
    return Verdict.READY_WITH_RESERVATIONS if any(f.certainty == Certainty.CONFIRMED for f in result.findings) else Verdict.READY


def assess_final(client: LLMClient, result: AnalysisResult, language: str) -> FinalAssessment:
    """LLM предлагает вердикт, но детерминированные правила не позволяют его смягчить."""
    floor = deterministic_verdict(result)
    known = {f.finding_id for f in result.findings}
    try:
        resp = client.generate(
            system=prompts.final_system(language),
            prompt=prompts.final_prompt(result.document_name, stats_text(result), findings_digest(result.findings)),
            schema=FinalResponse)
    except LLMError as exc:
        log.error("Итоговая оценка через LLM не получена: %s", exc)
        result.errors.append(f"Итоговая оценка через LLM не получена: {exc}")
        return FinalAssessment(verdict=floor, adjusted=False, complete=False, priority_finding_ids=_rank(result, []),
                               justification="Вердикт определён автоматически по правилам агента (ответ LLM недоступен): "
                                             "см. разделы «Сводка» и «Критические дефекты».")
    order = [Verdict.READY, Verdict.READY_WITH_RESERVATIONS, Verdict.NOT_READY]
    ceiling = (Verdict.NOT_READY if any(
        f.certainty == Certainty.CONFIRMED and f.severity in (Severity.CRITICAL, Severity.HIGH)
        for f in result.findings
    ) else Verdict.READY_WITH_RESERVATIONS)
    verdict = max(min(resp.verdict, ceiling, key=order.index), floor, key=order.index)
    priorities = _rank(result, [i for i in resp.priority_finding_ids if i in known])
    justification = _verdict_reason(result, language)
    return FinalAssessment(verdict=verdict, justification=justification, llm_verdict=resp.verdict,
                           adjusted=verdict != resp.verdict, priority_finding_ids=priorities)


def _verdict_reason(result: AnalysisResult, language: str) -> str:
    critical, high, other, incomplete = (
        ("Critical confirmed defects", "High-severity confirmed defects", "Other confirmed findings",
         "Some checks did not complete.")
        if language == "en" else
        ("Подтверждённые критические дефекты", "Подтверждённые дефекты высокой серьёзности",
         "Другие подтверждённые замечания", "Часть проверок не завершена.")
    )
    confirmed = [f for f in result.findings if f.certainty == Certainty.CONFIRMED]
    parts = []
    for severities, label in (((Severity.CRITICAL,), critical), ((Severity.HIGH,), high),
                              ((Severity.MEDIUM, Severity.LOW), other)):
        ids = [f.finding_id for f in confirmed if f.severity in severities]
        if ids:
            parts.append(f"{label} ({len(ids)}): {', '.join(ids[:8])}.")
    if result.errors:
        parts.append(incomplete)
    if not parts:
        return "No defects were confirmed." if language == "en" else "Подтверждённых дефектов нет."
    return " ".join(parts)


def _rank(result: AnalysisResult, preferred: list[str], limit: int = 10) -> list[str]:
    """Учитывает порядок LLM внутри одинаковой серьёзности и достоверности."""
    preference = {finding_id: i for i, finding_id in enumerate(preferred)}
    ranked = sorted(
        (f for f in result.findings if f.severity in (Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM)),
        key=lambda f: (f.certainty != Certainty.CONFIRMED, SEVERITY_ORDER[f.severity],
                       preference.get(f.finding_id, len(preference))),
    )
    return [f.finding_id for f in ranked[:limit]]
