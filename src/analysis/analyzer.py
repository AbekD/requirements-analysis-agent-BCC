"""Оркестратор: извлечение → индивидуальный анализ → набор → полнота → итог."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable

from src.analysis import prompts
from src.analysis.cross_analysis import run_completeness, run_cross_analysis
from src.analysis.findings import build_findings, findings_digest, stats_text
from src.analysis.individual import analyze_requirement
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
                 progress: Progress | None = None) -> AnalysisResult:
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
    for n, req in enumerate(targets, start=1):
        say(f"2/5 Индивидуальный анализ {n}/{len(targets)}: {req.report_id}")
        analysis = analyze_requirement(client, doc, index, req, language)
        result.individual.append(analysis)
        if not analysis.ok:
            result.errors.append(f"{req.report_id}: индивидуальный анализ не выполнен — {analysis.error}")

    say("3/5 Проверка набора требований…")
    result.cross_issues, result.cross_complete, errs = run_cross_analysis(client, doc, index, resolver, requirements, language)
    result.errors += errs

    say("4/5 Проверка полноты спецификации…")
    result.completeness, result.completeness_complete, errs = run_completeness(client, doc, index, resolver, requirements, language)
    result.errors += errs

    say("5/5 Итоговая оценка…")
    result.findings, result.duplicates_merged = build_findings(result)
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
    verdict = max(resp.verdict, floor, key=order.index)      # не мягче детерминированного минимума
    priorities = _rank(result, [i for i in resp.priority_finding_ids if i in known])
    return FinalAssessment(verdict=verdict, justification=resp.justification.strip(), llm_verdict=resp.verdict,
                           adjusted=verdict != resp.verdict, priority_finding_ids=priorities)


def _rank(result: AnalysisResult, preferred: list[str], limit: int = 10) -> list[str]:
    """Порядок LLM + добор по серьёзности (подтверждённые выше), без повторов."""
    ids = list(dict.fromkeys(preferred))
    rest = sorted(result.findings, key=lambda f: (SEVERITY_ORDER[f.severity], f.certainty != Certainty.CONFIRMED))
    for f in rest:
        if len(ids) >= limit:
            break
        if f.finding_id not in ids and f.severity != Severity.LOW:
            ids.append(f.finding_id)
    return ids[:limit]
