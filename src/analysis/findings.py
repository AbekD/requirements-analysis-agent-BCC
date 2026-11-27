"""Сведение результатов в единый список замечаний без двойного счёта + сводная статистика."""
from __future__ import annotations

from collections import Counter
import re

from src.models.schemas import (
    CRITERION_RU, SEVERITY_ORDER, AnalysisResult, Certainty, CompletenessStatus, EvidenceRef,
    Finding, Requirement, Severity, Status,
)
from src.validation.validator import normalize

_GET_DECLARATION = re.compile(r"^\|\s*(?:rest type|http method|метод http)\s*\|\s*get\s*\|", re.I)
_GET = re.compile(r"\bget\b", re.I)
_MUTATION = re.compile(r"сохран|измен|запис|модифиц|обнов|замен|созда|удал|mutat|modif|writ|sav|chang|updat|creat|delet", re.I)
_READ = re.compile(r"чтени|чита|получен|считы|\bread\w*|retriev", re.I)
_DESCRIPTION = re.compile(r"описан|назначени|описыв|смешива|description|purpose|describ", re.I)


def build_findings(result: AnalysisResult) -> tuple[list[Finding], int]:
    """Возвращает (замечания, число замечаний, не посчитанных повторно)."""
    drafts: list[Finding] = []
    get_methods = [req for req in result.requirements if _GET_DECLARATION.match(normalize(req.text))]

    for x in result.cross_issues:
        drafts.append(Finding(
            finding_id="", source="CROSS", severity=x.severity, certainty=x.certainty,
            title=f"{x.issue_id}: {x.category.value}", related_ids=x.related_ids, criteria=[x.category.value],
            cross_issue_ids=[x.issue_id],
            explanation=x.explanation, recommendation=x.recommendation, evidence=x.evidence))

    for c in result.completeness:
        if c.status in (CompletenessStatus.CONFIRMED_OMISSION, CompletenessStatus.POTENTIAL_OMISSION):
            drafts.append(Finding(
                finding_id="", source="COMPLETENESS", severity=c.severity, certainty=c.certainty or Certainty.POTENTIAL,
                title=f"Полнота: {c.area.value}", criteria=[c.area.value], explanation=c.explanation,
                recommendation=c.recommendation, evidence=c.evidence,
                related_ids=list(dict.fromkeys(e.ref for e in c.evidence if e.verified and e.ref.startswith("REQ-")))))

    for analysis in result.individual:
        if not analysis.ok:
            continue
        for a in analysis.assessments:
            if a.status not in (Status.FAIL, Status.UNCERTAIN):
                continue
            drafts.append(Finding(
                finding_id="", source="INDIVIDUAL", severity=a.severity,
                certainty=a.certainty or Certainty.POTENTIAL,
                title=CRITERION_RU[a.criterion], related_ids=[analysis.req_id], criteria=[a.criterion.value],
                explanation=a.explanation, recommendation=a.suggestion, evidence=a.evidence,
                missing_field=a.missing_field))

    source_rank = {"CROSS": 0, "COMPLETENESS": 1, "INDIVIDUAL": 2}
    drafts.sort(key=lambda f: (f.certainty != Certainty.CONFIRMED, SEVERITY_ORDER[f.severity], source_rank[f.source]))
    grouped: dict[tuple, Finding] = {}
    findings: list[Finding] = []
    for draft in drafts:
        key = _root_key(draft, get_methods)
        if key is None or key not in grouped:
            findings.append(draft)
            if key is not None:
                grouped[key] = draft
            continue
        main = grouped[key]
        main.related_ids = list(dict.fromkeys(main.related_ids + draft.related_ids))
        main.criteria = list(dict.fromkeys(main.criteria + draft.criteria))
        main.cross_issue_ids.extend(draft.cross_issue_ids)
        if main.certainty == draft.certainty:
            unique = {_evidence_key(e): e for e in main.evidence + draft.evidence if e.verified}
            if unique:
                main.evidence = list(unique.values())
    for k, f in enumerate(findings, start=1):
        f.finding_id = f"F-{k:03d}"
    return findings, len(drafts) - len(findings)


def _evidence_key(e: EvidenceRef) -> tuple:
    return e.line_start, e.line_end, normalize(e.quote)


def _root_key(finding: Finding, get_methods: list[Requirement]) -> tuple | None:
    evidence = [e for e in finding.evidence if e.verified]
    explanation = normalize(finding.explanation)
    if not evidence:
        return None
    if finding.missing_field:
        return "missing-field", tuple(finding.related_ids), normalize(finding.missing_field)
    if (_GET.search(explanation) and _MUTATION.search(explanation)
            and not (_READ.search(explanation) and _DESCRIPTION.search(explanation))):
        cited = {req.report_id for req in get_methods if any(
            e.line_start is not None and e.line_end is not None
            and req.line_start <= e.line_start <= e.line_end <= req.line_end for e in evidence
        )}
        if len(cited) == 1:
            return "get-mutation", next(iter(cited))
        if len(get_methods) == 1 and (get_methods[0].report_id in finding.related_ids
                                      or any(_MUTATION.search(e.quote) for e in evidence)):
            return "get-mutation", get_methods[0].report_id
    anchors = tuple(sorted({_evidence_key(e) for e in evidence}))
    return ("same-claim", anchors, explanation) if explanation else None


def severity_counts(findings: list[Finding]) -> Counter:
    return Counter(f.severity for f in findings)


def stats_text(result: AnalysisResult) -> str:
    reqs = result.requirements
    analyzable = [r for r in reqs if r.analyzable]
    ok = [a for a in result.individual if a.ok]
    conf = [f for f in result.findings if f.certainty == Certainty.CONFIRMED]
    pot = [f for f in result.findings if f.certainty == Certainty.POTENTIAL]
    sev = severity_counts(result.findings)
    sev_conf = severity_counts(conf)
    return (
        f"Extracted elements: {len(reqs)}; reviewed individually: {len(analyzable)}; analysed successfully: {len(ok)}.\n"
        f"Confirmed defects: {len(conf)} (critical {sev_conf[Severity.CRITICAL]}, high {sev_conf[Severity.HIGH]}, "
        f"medium {sev_conf[Severity.MEDIUM]}, low {sev_conf[Severity.LOW]}); potential issues: {len(pot)}.\n"
        f"All findings by severity: critical {sev[Severity.CRITICAL]}, high {sev[Severity.HIGH]}, "
        f"medium {sev[Severity.MEDIUM]}, low {sev[Severity.LOW]}.\n"
        f"Analysis incomplete parts: {len(result.errors)}."
    )


def findings_digest(findings: list[Finding], limit: int = 60) -> str:
    lines = []
    for f in findings[:limit]:
        ev = next((e for e in f.evidence if e.verified), None)
        quote = ev.quote.replace("\n", " ")[:160] if ev else "(нет проверенной цитаты)"
        lines.append(f"[{f.finding_id}] {f.severity.value}/{f.certainty.value} {f.source} {f.title} "
                     f"refs={','.join(f.related_ids) or '-'} | {f.explanation[:240]} | quote: {quote}")
    return "\n".join(lines) or "(no findings)"
