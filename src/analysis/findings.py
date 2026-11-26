"""Сведение результатов в единый список замечаний без двойного счёта + сводная статистика."""
from __future__ import annotations

from collections import Counter

from src.models.schemas import (
    CRITERION_RU, SEVERITY_ORDER, AnalysisResult, Certainty, CompletenessStatus, Criterion, EvidenceRef,
    Finding, Severity, Status,
)
from src.validation.validator import normalize


def build_findings(result: AnalysisResult) -> tuple[list[Finding], int]:
    """Возвращает (замечания, число замечаний, не посчитанных повторно)."""
    drafts: list[Finding] = []
    cross_related = {rid for x in result.cross_issues for rid in x.related_ids}
    merged_duplicates = 0

    for x in result.cross_issues:
        drafts.append(Finding(
            finding_id="", source="CROSS", severity=x.severity, certainty=x.certainty,
            title=f"{x.issue_id}: {x.category.value}", related_ids=x.related_ids, criteria=[x.category.value],
            explanation=x.explanation, recommendation=x.recommendation, evidence=x.evidence))

    for c in result.completeness:
        if c.status in (CompletenessStatus.CONFIRMED_OMISSION, CompletenessStatus.POTENTIAL_OMISSION):
            drafts.append(Finding(
                finding_id="", source="COMPLETENESS", severity=c.severity, certainty=c.certainty or Certainty.POTENTIAL,
                title=f"Полнота: {c.area.value}", criteria=[c.area.value], explanation=c.explanation,
                recommendation=c.recommendation, evidence=c.evidence))

    for analysis in result.individual:
        if not analysis.ok:
            continue
        groups: dict[tuple[str, str], list] = {}
        for a in analysis.assessments:
            if a.status not in (Status.FAIL, Status.UNCERTAIN):
                continue
            verified = [e for e in a.evidence if e.verified]
            key = (analysis.req_id, normalize(verified[0].quote) if verified else f"~{a.criterion.value}")
            groups.setdefault(key, []).append(a)
        for (req_id, _), members in groups.items():
            criteria = [m.criterion for m in members]
            if set(criteria) == {Criterion.CONSISTENCY} and req_id in cross_related:
                merged_duplicates += 1          # то же противоречие уже учтено в проверке набора
                continue
            confirmed = any(m.certainty == Certainty.CONFIRMED for m in members)
            evidence: list[EvidenceRef] = []
            for m in members:
                evidence += [e for e in m.evidence if normalize(e.quote) not in {normalize(x.quote) for x in evidence}]
            drafts.append(Finding(
                finding_id="", source="INDIVIDUAL", severity=min((m.severity for m in members), key=SEVERITY_ORDER.get),
                certainty=Certainty.CONFIRMED if confirmed else Certainty.POTENTIAL,
                title=", ".join(CRITERION_RU[c] for c in criteria), related_ids=[req_id],
                criteria=[c.value for c in criteria],
                explanation=" ".join(dict.fromkeys(m.explanation for m in members)),
                recommendation=" ".join(dict.fromkeys(m.suggestion for m in members if m.suggestion)),
                evidence=evidence))

    source_rank = {"CROSS": 0, "COMPLETENESS": 1, "INDIVIDUAL": 2}
    drafts.sort(key=lambda f: (SEVERITY_ORDER[f.severity], f.certainty != Certainty.CONFIRMED, source_rank[f.source]))
    for k, f in enumerate(drafts, start=1):
        f.finding_id = f"F-{k:03d}"
    return drafts, merged_duplicates


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
        quote = f.evidence[0].quote.replace("\n", " ")[:160] if f.evidence else "(нет цитаты)"
        lines.append(f"[{f.finding_id}] {f.severity.value}/{f.certainty.value} {f.source} {f.title} "
                     f"refs={','.join(f.related_ids) or '-'} | {f.explanation[:240]} | quote: {quote}")
    return "\n".join(lines) or "(no findings)"
