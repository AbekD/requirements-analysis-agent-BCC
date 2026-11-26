"""Генерация Markdown-отчёта на русском языке по результатам анализа."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.analysis.findings import severity_counts
from src.models.schemas import (
    COMPLETENESS_AREA_RU, CRITERION_RU, ISSUE_CATEGORY_RU, REQ_TYPE_RU, AnalysisResult, Assessment,
    Certainty, CompletenessStatus, CrossIssue, EvidenceRef, Finding, IndividualAnalysis, IssueCategory,
    Requirement, RunStatus, Severity, Status, Verdict,
)

STATUS_RU = {RunStatus.COMPLETE: "Анализ завершён полностью",
             RunStatus.PARTIAL: "Анализ завершён ЧАСТИЧНО (часть проверок не выполнена — см. ниже)",
             RunStatus.FAILED: "Анализ НЕ выполнен"}
VERDICT_RU = {Verdict.READY: "готов к разработке и тестированию",
              Verdict.READY_WITH_RESERVATIONS: "готов с оговорками",
              Verdict.NOT_READY: "не готов к разработке и тестированию"}
SEV_RU = {Severity.CRITICAL: "критичная", Severity.HIGH: "высокая", Severity.MEDIUM: "средняя",
          Severity.LOW: "низкая", Severity.NONE: "—"}
CERT_RU = {Certainty.CONFIRMED: "подтверждённый дефект", Certainty.POTENTIAL: "потенциальная проблема"}
STATUS_ICON = {Status.PASS: "✅ PASS", Status.FAIL: "❌ FAIL", Status.UNCERTAIN: "❓ UNCERTAIN",
               Status.NOT_APPLICABLE: "➖ NOT_APPLICABLE"}
COMPL_RU = {CompletenessStatus.COVERED: "✅ описано", CompletenessStatus.CONFIRMED_OMISSION: "❌ подтверждённый пропуск",
            CompletenessStatus.POTENTIAL_OMISSION: "❓ возможный пропуск (нужно уточнение)",
            CompletenessStatus.NOT_APPLICABLE: "➖ неприменимо"}

ISSUE_GROUPS: list[tuple[str, list[IssueCategory]]] = [
    ("Противоречия и логические несогласованности", [IssueCategory.CONTRADICTION, IssueCategory.LOGICAL_INCONSISTENCY,
                                                     IssueCategory.CONFLICTING_VALUES, IssueCategory.FUNC_NONFUNC_CONFLICT]),
    ("Дублирующиеся требования", [IssueCategory.DUPLICATE]),
    ("Непоследовательная терминология", [IssueCategory.TERMINOLOGY]),
    ("Отсутствующие зависимости", [IssueCategory.MISSING_DEPENDENCY]),
    ("Конфликты между основным и дочерним документами", [IssueCategory.CROSS_DOCUMENT]),
    ("Расхождения с критериями приёмки", [IssueCategory.ACCEPTANCE_MISMATCH]),
    ("Отсутствующие сценарии", [IssueCategory.MISSING_SCENARIO]),
    ("Прочее", [IssueCategory.OTHER]),
]


# --------------------------------------------------------------------------- #
def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\r", "").replace("\n", "<br>").strip() or "—"


def _quote_block(text: str) -> str:
    return "\n".join(f"> {line}" if line.strip() else ">" for line in text.splitlines())


def _ev_inline(ev: EvidenceRef) -> str:
    q = ev.quote.replace("\n", " ")
    ref = f"{ev.ref}: " if ev.ref else ""
    if ev.verified:
        return f"✔ {ref}«{q}» ({ev.location})"
    return f"⚠ {ref}«{q}» (цитата не найдена в документе)"


def _ev_cell(evidence: list[EvidenceRef]) -> str:
    return _cell("<br>".join(_ev_inline(e) for e in evidence)) if evidence else "—"


def _ev_list(evidence: list[EvidenceRef], indent: str = "") -> list[str]:
    return [f"{indent}- {_ev_inline(e)}" for e in evidence]


def _source(req: Requirement) -> str:
    lines = f"строка {req.line_start}" if req.line_start == req.line_end else f"строки {req.line_start}–{req.line_end}"
    return f"{lines}; «{req.section}»"


# --------------------------------------------------------------------------- #
def render_report(result: AnalysisResult) -> str:
    out: list[str] = ["# Отчёт о проверке требований", ""]
    out += _passport(result)
    out += _extraction(result)
    out += _individual(result)
    out += _set_checks(result)
    out += _summary(result)
    out += _critical(result)
    out += _verdict(result)
    out += _priorities(result)
    out += _appendix(result)
    return "\n".join(out).rstrip() + "\n"


def write_report(result: AnalysisResult, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_report(result), encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
def _passport(r: AnalysisResult) -> list[str]:
    rows = [
        ("Проверяемый документ", r.document_name),
        ("SHA-256 документа", f"`{r.document_sha256}`" if r.document_sha256 else "—"),
        ("Дата и время анализа", r.started_at),
        ("LLM-модель", r.model),
        ("Версия агента", r.agent_version),
        ("Статус выполнения анализа", STATUS_RU[r.status]),
        ("Число обращений к LLM", str(r.llm_calls)),
        ("Основание критериев", "ISO/IEC/IEEE 29148:2018 (5.2.5, 5.2.6), IEEE Std 830-1998 (4.3); "
                                "формальное соответствие стандартам не заявляется"),
    ]
    out = ["## 1. Паспорт прогона", "", "| Поле | Значение |", "| :--- | :--- |"]
    out += [f"| {k} | {_cell(v)} |" for k, v in rows]
    out.append("")
    if r.errors:
        out += ["**Незавершённые части анализа** (результаты по ним в отчёте отсутствуют, это не означает отсутствие дефектов):", ""]
        out += [f"- {e}" for e in r.errors]
        out.append("")
    if r.warnings:
        out += ["**Предупреждения прогона:**", ""] + [f"- {w}" for w in r.warnings] + [""]
    return out


def _extraction(r: AnalysisResult) -> list[str]:
    by_type = Counter(q.type for q in r.requirements)
    types = "; ".join(f"{REQ_TYPE_RU[t]} — {n}" for t, n in by_type.most_common())
    reviewed = {a.req_id for a in r.individual}
    out = ["## 2. Извлечение требований", "",
           "Документ нарезан на элементы детерминированно (пункты списков, строки таблиц, блоки кода, абзацы); "
           "тип элемента определён по разделу и уточнён LLM. Идентификаторы `REQ-NNN` присвоены агентом; "
           "идентификатор документа указан, если он был. Дословные цитаты приведены в разделе 3 "
           "(для контекстных элементов — в конце этого раздела).", "",
           f"Всего элементов: **{len(r.requirements)}**. По типам: {types}.", "",
           "| ID в отчёте | ID в документе | Тип | Краткая формулировка | Источник | Индивидуальная проверка |",
           "| :--- | :--- | :--- | :--- | :--- | :--- |"]
    for q in r.requirements:
        check = "да" if q.report_id in reviewed else ("нет (контекст)" if not q.analyzable else "не выполнена")
        out.append(f"| {q.report_id} | {q.doc_id or '—'} | {REQ_TYPE_RU[q.type]} | {_cell(q.short)} | "
                   f"{_cell(_source(q))} | {check} |")
    ctx = [q for q in r.requirements if not q.analyzable]
    if ctx:
        out += ["", "<details><summary>Дословные цитаты контекстных элементов (не проверяются индивидуально)</summary>", ""]
        for q in ctx:
            out += [f"**{q.report_id}** — {REQ_TYPE_RU[q.type]}, {_source(q)}", "", _quote_block(q.text), ""]
        out.append("</details>")
    out.append("")
    return out


def _individual(r: AnalysisResult) -> list[str]:
    by_req = {q.report_id: q for q in r.requirements}
    out = ["## 3. Индивидуальная проверка требований", "",
           "Статусы: PASS — соответствует; FAIL — подтверждённое нарушение (с цитатой); UNCERTAIN — недостаточно данных "
           "в документе; NOT_APPLICABLE — критерий неприменим. Значок ✔ у цитаты означает, что она дословно найдена в "
           "документе программной проверкой, ⚠ — не найдена (такое замечание не считается подтверждённым).", ""]
    for a in r.individual:
        q = by_req[a.req_id]
        out += _one_requirement(q, a)
    if not r.individual:
        out.append("Индивидуальный анализ не выполнялся.\n")
    return out


def _one_requirement(q: Requirement, a: IndividualAnalysis) -> list[str]:
    title = f"#### {q.report_id}" + (f" ({q.doc_id})" if q.doc_id else "") + f" — {REQ_TYPE_RU[q.type]}"
    out = [title, "", _quote_block(q.text), "", f"Источник: {_source(q)}", ""]
    if not a.ok:
        return out + [f"⚠ **Анализ не выполнен (неполный):** {a.error}", ""]
    out += ["| Критерий | Статус | Серьёзность | Пояснение | Доказательство (цитата и место) | Предложение по улучшению |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |"]
    for s in a.assessments:
        out.append(_assessment_row(s))
    notes = [f"{CRITERION_RU[s.criterion]}: {n}" for s in a.assessments for n in s.notes]
    if notes:
        out += [""] + [f"- ℹ {n}" for n in notes]
    out.append("")
    return out


def _assessment_row(s: Assessment) -> str:
    sev = SEV_RU[s.severity] if s.status in (Status.FAIL, Status.UNCERTAIN) else "—"
    certainty = ""
    if s.status == Status.FAIL and s.certainty == Certainty.POTENTIAL:
        certainty = " (не подтверждён)"
    return (f"| {CRITERION_RU[s.criterion]} | {STATUS_ICON[s.status]}{certainty} | {sev} | {_cell(s.explanation)} | "
            f"{_ev_cell(s.evidence)} | {_cell(s.suggestion)} |")


# --------------------------------------------------------------------------- #
def _set_checks(r: AnalysisResult) -> list[str]:
    out = ["## 4. Проверка набора требований", "",
           "Анализ выполнен по всему документу сразу (обе части — основная и дочерняя — сопоставляются между собой).", ""]
    out += ["### 4.1. Взаимосвязи между требованиями и документами", ""]
    if not r.cross_complete:
        out += ["⚠ **Анализ набора требований выполнен не полностью** — ниже только то, что удалось получить.", ""]
    for title, cats in ISSUE_GROUPS:
        items = [x for x in r.cross_issues if x.category in cats]
        out += [f"#### {title}", ""]
        if not items:
            out += ["Анализ не выполнен." if not r.cross_complete else "Замечаний не обнаружено.", ""]
            continue
        for x in items:
            out += _cross_issue(x)
    out += ["### 4.2. Полнота спецификации", ""]
    if not r.completeness_complete:
        out += ["⚠ Анализ полноты не выполнен.", ""]
    else:
        out += ["| Область | Статус | Серьёзность | Пояснение | Доказательства | Что уточнить |",
                "| :--- | :--- | :--- | :--- | :--- | :--- |"]
        for c in r.completeness:
            sev = SEV_RU[c.severity] if c.status in (CompletenessStatus.CONFIRMED_OMISSION,
                                                     CompletenessStatus.POTENTIAL_OMISSION) else "—"
            out.append(f"| {COMPLETENESS_AREA_RU[c.area]} | {COMPL_RU[c.status]} | {sev} | {_cell(c.explanation)} | "
                       f"{_ev_cell(c.evidence)} | {_cell(c.recommendation)} |")
        notes = [f"{COMPLETENESS_AREA_RU[c.area]}: {n}" for c in r.completeness for n in c.notes]
        if notes:
            out += [""] + [f"- ℹ {n}" for n in notes]
        out.append("")
    return out


def _cross_issue(x: CrossIssue) -> list[str]:
    refs = ", ".join(x.related_ids + [f"«{s}»" for s in x.related_sections]) or "—"
    out = [f"**{x.issue_id}** — {ISSUE_CATEGORY_RU[x.category]} · серьёзность: {SEV_RU[x.severity]} · "
           f"{CERT_RU[x.certainty]}", "",
           f"- Связанные элементы: {refs}", f"- Пояснение: {x.explanation}"]
    if x.evidence:
        out.append("- Доказательства:")
        out += _ev_list(x.evidence, "  ")
    out.append(f"- Рекомендация: {x.recommendation or '—'}")
    out += [f"- ℹ {n}" for n in x.notes]
    out.append("")
    return out


# --------------------------------------------------------------------------- #
def _summary(r: AnalysisResult) -> list[str]:
    reqs = r.requirements
    reviewed = [q for q in reqs if q.analyzable]
    ok = sum(1 for a in r.individual if a.ok)
    conf = [f for f in r.findings if f.certainty == Certainty.CONFIRMED]
    pot = [f for f in r.findings if f.certainty == Certainty.POTENTIAL]
    all_sev, conf_sev, pot_sev = severity_counts(r.findings), severity_counts(conf), severity_counts(pot)
    out = ["## 5. Сводка", "",
           "| Показатель | Значение |", "| :--- | :--- |",
           f"| Извлечено элементов (всего) | {len(reqs)} |",
           f"| Подлежит индивидуальной проверке | {len(reviewed)} |",
           f"| Успешно проанализировано | {ok} из {len(reviewed)} |",
           f"| Подтверждённых дефектов | {len(conf)} |",
           f"| Потенциальных проблем (требуют уточнения / цитата не подтверждена) | {len(pot)} |",
           f"| Всего уникальных замечаний | {len(r.findings)} |", "",
           "| Серьёзность | Подтверждённые дефекты | Потенциальные проблемы | Всего |", "| :--- | :--- | :--- | :--- |"]
    for s in (Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW):
        out.append(f"| {SEV_RU[s].capitalize()} | {conf_sev[s]} | {pot_sev[s]} | {all_sev[s]} |")
    out += ["", "Замечания по одному месту документа, найденные по разным критериям, объединены в одно; "
                "замечания по критерию «Непротиворечивость», уже учтённые в разделе 4, повторно не считаются"
                + (f" (не посчитано повторно: {r.duplicates_merged})." if r.duplicates_merged else "."), ""]
    if r.errors:
        out += [f"⚠ Незавершённых частей анализа: {len(r.errors)} (см. раздел 1). Итоговые числа могут быть занижены.", ""]
    return out


def _finding_block(f: Finding) -> list[str]:
    refs = ", ".join(f.related_ids) or "—"
    out = [f"**{f.finding_id}** · {_src(f)} · элементы: {refs} · {CERT_RU[f.certainty]}", "",
           f"- Суть: {f.explanation or f.title}"]
    if f.evidence:
        out.append("- Доказательства:")
        out += _ev_list(f.evidence, "  ")
    out += [f"- Что исправить: {f.recommendation or '—'}", ""]
    return out


def _src(f: Finding) -> str:
    return {"CROSS": "проверка набора", "COMPLETENESS": "полнота", "INDIVIDUAL": "индивидуальная проверка"}[f.source] \
        + f" ({f.title})"


def _critical(r: AnalysisResult) -> list[str]:
    crit = [f for f in r.findings if f.severity == Severity.CRITICAL]
    out = ["## 6. Критические дефекты", "",
           "Критичными считаются дефекты, из-за которых требование нельзя реализовать однозначно или объективно проверить.", ""]
    if not crit:
        return out + ["Критичных замечаний не выявлено" + ("." if not r.errors else " (но часть анализа не выполнена)."), ""]
    for label, cert in (("Подтверждённые", Certainty.CONFIRMED), ("Потенциальные (требуют уточнения)", Certainty.POTENTIAL)):
        items = [f for f in crit if f.certainty == cert]
        if items:
            out += [f"### {label}", ""]
            for f in items:
                out += _finding_block(f)
    return out


def _verdict(r: AnalysisResult) -> list[str]:
    out = ["## 7. Итоговая оценка", ""]
    if r.final is None:
        return out + ["Итоговая оценка не сформирована: анализ не завершён.", ""]
    out += [f"**Статус: `{r.final.verdict.value}`** — документ {VERDICT_RU[r.final.verdict]}.", "",
            r.final.justification, ""]
    if r.final.adjusted and r.final.llm_verdict:
        out += [f"ℹ LLM предложила `{r.final.llm_verdict.value}`; вердикт ужесточён детерминированными правилами агента "
                "(подтверждённые критичные/высокие дефекты или незавершённый анализ не допускают более мягкой оценки).", ""]
    if not r.final.complete:
        out += ["⚠ Обоснование сформировано без участия LLM.", ""]
    return out


def _priorities(r: AnalysisResult) -> list[str]:
    out = ["## 8. Приоритетные рекомендации", ""]
    by_id = {f.finding_id: f for f in r.findings}
    ids = r.final.priority_finding_ids if r.final else []
    if not ids:
        return out + ["Рекомендаций нет: значимых замечаний не выявлено.", ""]
    for n, fid in enumerate(ids, start=1):
        f = by_id.get(fid)
        if f is None:
            continue
        quote = ""
        ev = next((e for e in f.evidence if e.verified), None)
        if ev:
            quote = f" Основание: «{ev.quote.replace(chr(10), ' ')}» ({ev.location})."
        out.append(f"{n}. **{fid}** ({SEV_RU[f.severity]}, {CERT_RU[f.certainty]}; элементы: "
                   f"{', '.join(f.related_ids) or '—'}). {f.recommendation or f.explanation}{quote}")
    out.append("")
    return out


def _appendix(r: AnalysisResult) -> list[str]:
    out = ["## Приложение. Реестр замечаний", "",
           "| ID | Серьёзность | Характер | Источник | Элементы | Суть |", "| :--- | :--- | :--- | :--- | :--- | :--- |"]
    for f in sorted(r.findings, key=lambda f: f.finding_id):
        out.append(f"| {f.finding_id} | {SEV_RU[f.severity]} | {CERT_RU[f.certainty]} | {_cell(_src(f))} | "
                   f"{_cell(', '.join(f.related_ids))} | {_cell((f.explanation or f.title)[:300])} |")
    if not r.findings:
        out.append("| — | — | — | — | — | Замечаний нет |")
    out.append("")
    return out

