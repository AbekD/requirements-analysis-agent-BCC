"""Детерминированная проверка результатов LLM: цитаты, ссылки на ID, места в документе.

Принцип: LLM не может «подтвердить» дефект собственным словом. Дефект считается
подтверждённым только если его цитаты дословно найдены в тексте документа.
"""
from __future__ import annotations

from itertools import combinations
import re
import unicodedata

from src.analysis.arithmetic import arithmetic_claim, checked_sum
from src.models.schemas import (
    Assessment, AssessmentLLM, Certainty, CompletenessFinding, CompletenessItemLLM, CompletenessStatus,
    CrossIssue, CrossIssueLLM, Criterion, EvidenceLLM, EvidenceRef, IndividualAnalysis, IndividualResponse,
    IssueCategory, ParsedDocument, Requirement, SEVERITY_ORDER, Severity, Status, TWO_SOURCE_CATEGORIES,
)

MIN_QUOTE_CHARS = 3
_QUOTE_MAP = str.maketrans({"«": '"', "»": '"', "“": '"', "”": '"', "„": '"', "‘": "'", "’": "'",
                            "–": "-", "—": "-", "‑": "-", "−": "-"})
_STRIP_MARKUP = re.compile(r"[*`]")
_ELLIPSIS = re.compile(r"\s*(?:…|\.{3})\s*")
_REF_TOKEN = re.compile(r"[A-Za-zА-Яа-яЁё]{1,6}-\d+(?:\.\d+)*")
_MISSING_ID = re.compile(
    r"(?:нет|отсутств\w*|не\s+(?:указан\w*|имеет)|missing|no|lacks?)\s+"
    r"(?:\w+\s+){0,3}(?:идентификатор\w*|id\b)|"
    r"(?:идентификатор\w*|id\b)\s+(?:не\s+указан\w*|отсутств\w*|is\s+missing)",
    re.IGNORECASE,
)
_MISSING_LINK = re.compile(
    r"(?:нет|отсутств\w*|не\s+(?:указан\w*|определ[её]н\w*)|missing|no|lacks?)\s+"
    r"(?:\w+\s+){0,2}(?:связ\w*|ссыл\w*|link\b|reference\b)|"
    r"(?:нельзя|невозможно|cannot)\s+(?:однозначно\s+)?(?:сопоставить|найти|match|locate)",
    re.IGNORECASE,
)
_BUSINESS_ID = re.compile(
    r"(?:идентификатор\w*|id)\s+(?:карт\w*|операци\w*|заявк\w*|транзакци\w*|событи\w*|запис\w*)|"
    r"(?:card|operation|request|transaction|event|record)\s+id\b",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    """Нормализация для сравнения: NFKC, единые кавычки/тире, без markdown-разметки, пробелы схлопнуты, без регистра."""
    t = unicodedata.normalize("NFKC", text).translate(_QUOTE_MAP)
    t = _STRIP_MARKUP.sub("", t)
    return re.sub(r"\s+", " ", t).strip().casefold()


class DocIndex:
    """Нормализованный текст документа с отображением позиции символа → номер строки."""

    def __init__(self, doc: ParsedDocument) -> None:
        self.doc = doc
        chunks: list[str] = []
        self._line_of: list[int] = []
        for no, line in enumerate(doc.lines, start=1):
            norm = normalize(line)
            if not norm:
                continue
            chunks.append(norm + " ")
            self._line_of.extend([no] * (len(norm) + 1))
        self.full = "".join(chunks)

    def locate(self, quote: str, hint: tuple[int, int] | None = None) -> tuple[int, int] | None:
        """Ищет цитату (допускается «…» между фрагментами). Возвращает (строка_от, строка_до) или None."""
        parts = [normalize(p) for p in _ELLIPSIS.split(quote) if p.strip()]
        parts = [p for p in parts if p]
        if not parts or any(len(p) < MIN_QUOTE_CHARS for p in parts):
            return None
        matches: set[tuple[int, int]] = set()
        for start in self._occurrences(parts[0]):
            pos = start + len(parts[0])
            end = pos
            ok = True
            for part in parts[1:]:
                idx = next(self._occurrences(part, pos), None)
                if idx is None:
                    ok = False
                    break
                pos = end = idx + len(part)
            if not ok:
                continue
            span = (self._line_of[start], self._line_of[max(start, end - 1)])
            if hint is None or (span[0] >= hint[0] and span[1] <= hint[1]):
                matches.add(span)
        return next(iter(matches)) if len(matches) == 1 else None

    def _occurrences(self, needle: str, start: int = 0):
        idx = self.full.find(needle, start)
        while idx != -1:
            end = idx + len(needle)
            starts_inside_word = idx > 0 and needle[0].isalnum() and self.full[idx - 1].isalnum()
            ends_inside_word = end < len(self.full) and needle[-1].isalnum() and self.full[end].isalnum()
            if not starts_inside_word and not ends_inside_word:
                yield idx
            idx = self.full.find(needle, idx + 1)


class RefResolver:
    """Сопоставляет ссылки LLM (REQ-005, Ф-3, «REQ-005 (Ф-3)») с существующими требованиями."""

    def __init__(self, requirements: list[Requirement], doc: ParsedDocument) -> None:
        self._requirements = requirements
        self._by_key = {self._key(r.report_id): r for r in requirements}
        by_original: dict[str, list[Requirement]] = {}
        for r in requirements:
            if r.doc_id:
                by_original.setdefault(self._key(r.doc_id), []).append(r)
        for key, matches in by_original.items():
            if len(matches) == 1:
                self._by_key.setdefault(key, matches[0])
        self._sections: dict[str, list[tuple[int, int]]] = {}
        for i, heading in enumerate(doc.headings):
            end = next((h.line - 1 for h in doc.headings[i + 1:] if h.level <= heading.level), len(doc.lines))
            for title in {heading.title, doc.section_path(heading.line)}:
                self._sections.setdefault(normalize(title), []).append((heading.line, end))

    @staticmethod
    def _key(ref: str) -> str:
        return normalize(ref).replace(" ", "")

    def resolve(self, ref: str) -> Requirement | None:
        direct = self._by_key.get(self._key(ref))
        if direct:
            return direct
        for token in _REF_TOKEN.findall(ref):
            hit = self._by_key.get(self._key(token))
            if hit:
                return hit
        return None

    def mentioned_ids(self, text: str) -> set[str]:
        return {req.report_id for ref in _REF_TOKEN.findall(text) if (req := self.resolve(ref))}

    def section_exists(self, title: str) -> bool:
        return normalize(title) in self._sections

    def evidence_scope(self, ref: str) -> tuple[int, int] | None:
        req = self.resolve(ref)
        if req:
            return req.line_start, req.line_end
        sections = self._sections.get(normalize(ref), [])
        return sections[0] if len(sections) == 1 else None

    def requirement_at(self, line_start: int, line_end: int) -> Requirement | None:
        return next((r for r in self._requirements if r.line_start <= line_start <= line_end <= r.line_end), None)


# --------------------------------------------------------------------------- #
def build_evidence(quote: str, index: DocIndex, ref: str = "", hint: tuple[int, int] | None = None) -> EvidenceRef:
    span = index.locate(quote, hint)
    if span is None:
        return EvidenceRef(quote=quote.strip(), ref=ref, verified=False)
    source = "\n".join(index.doc.lines[span[0] - 1:span[1]])
    return EvidenceRef(quote=source, ref=ref, verified=True, line_start=span[0], line_end=span[1],
                       location=index.doc.location(*span) + f"; раздел «{index.doc.section_path(span[0])}»")


def _dedupe_quotes(quotes: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for q in quotes:
        k = normalize(q)
        if k and k not in seen:
            seen.add(k)
            out.append(q)
    return out


def _calibrate_severity(severity: Severity, certainty: Certainty, evidence: list[EvidenceRef],
                        ceiling: Severity = Severity.CRITICAL) -> Severity:
    if severity == Severity.NONE:
        severity = Severity.MEDIUM
    if certainty == Certainty.POTENTIAL:
        cap = Severity.MEDIUM if any(e.verified for e in evidence) else Severity.LOW
        ceiling = max(ceiling, cap, key=SEVERITY_ORDER.__getitem__)
    return max(severity, ceiling, key=SEVERITY_ORDER.__getitem__)


# --------------------------------------------------------------------------- #
# Индивидуальный анализ
# --------------------------------------------------------------------------- #
def validate_individual(resp: IndividualResponse, req: Requirement, index: DocIndex) -> IndividualAnalysis:
    hint = (req.line_start, req.line_end)
    amounts = checked_sum(req)
    assessments = []
    for a in resp.assessments:
        if a.status in (Status.FAIL, Status.UNCERTAIN):
            if (amounts and amounts[0] + amounts[1] == amounts[2]
                    and a.criterion in (Criterion.CORRECTNESS, Criterion.CONSISTENCY)
                    and arithmetic_claim(a.explanation)):
                assessments.append(Assessment(criterion=a.criterion, status=Status.PASS,
                                              explanation=f"Итог проверен: {amounts[0]} + {amounts[1]} = {amounts[2]}."))
                continue
            if (a.criterion == Criterion.TRACEABILITY and _MISSING_ID.search(a.explanation)
                    and not _MISSING_LINK.search(a.explanation) and not _BUSINESS_ID.search(a.explanation)):
                assessments.append(Assessment(criterion=a.criterion, status=Status.PASS,
                                              explanation=f"Элемент однозначно находится по адресу: {index.doc.location(*hint)}, "
                                                          f"раздел «{req.section}»."))
                continue
        assessments.append(_validate_assessment(a, index, hint))
    order = {c: i for i, c in enumerate(Criterion)}
    assessments.sort(key=lambda a: order[a.criterion])
    return IndividualAnalysis(req_id=req.report_id, ok=True, assessments=assessments)


def _validate_assessment(a: AssessmentLLM, index: DocIndex, hint: tuple[int, int]) -> Assessment:
    evidence = [build_evidence(q, index, hint=hint) for q in _dedupe_quotes(a.evidence)]
    notes: list[str] = []
    status, severity = a.status, a.severity
    certainty: Certainty | None = None
    if a.status in (Status.FAIL, Status.UNCERTAIN):
        verified = [e for e in evidence if e.verified]
        if a.status == Status.FAIL:
            if verified and len(verified) == len(evidence):
                certainty = Certainty.CONFIRMED
            else:
                status = Status.UNCERTAIN
                certainty = Certainty.POTENTIAL
                notes.append("Подтверждение требует проверки всех цитат в границах требования."
                             if evidence else "Модель не привела цитату.")
        else:
            certainty = Certainty.POTENTIAL
        if evidence and len(verified) < len(evidence):
            notes.append(f"Не найдено в документе цитат: {len(evidence) - len(verified)} из {len(evidence)}.")
        ceiling = Severity.MEDIUM if a.criterion in {
            Criterion.ATOMICITY, Criterion.DESIGN_INDEPENDENCE, Criterion.TRACEABILITY, Criterion.NECESSITY,
        } else Severity.CRITICAL
        severity = _calibrate_severity(severity, certainty, evidence, ceiling)
    else:
        severity = Severity.NONE
    field = a.missing_field.strip()
    if not field or index.locate(field, hint) is None:
        field = ""
    return Assessment(criterion=a.criterion, status=status, severity=severity, explanation=a.explanation.strip(),
                      evidence=evidence, suggestion=a.suggestion.strip(), missing_field=field,
                      certainty=certainty, notes=notes)


# --------------------------------------------------------------------------- #
# Анализ набора требований
# --------------------------------------------------------------------------- #
def validate_evidence(items: list[EvidenceLLM], index: DocIndex, resolver: RefResolver
                       ) -> tuple[list[EvidenceRef], list[str]]:
    out: list[EvidenceRef] = []
    notes: list[str] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        ref = item.ref.strip()
        key = (normalize(ref), normalize(item.quote))
        if key in seen:
            continue
        seen.add(key)
        req = resolver.resolve(ref)
        hint = resolver.evidence_scope(ref)
        if ref and hint is None:
            notes.append(f"Ссылка «{ref}» не определяет однозначное место в документе.")
            ev = EvidenceRef(quote=item.quote.strip(), ref=ref)
        else:
            ev = build_evidence(item.quote, index, ref=req.report_id if req else ref, hint=hint)
        out.append(ev)
    return out, notes


def false_arithmetic_issue(raw: CrossIssueLLM, issue: CrossIssue, resolver: RefResolver) -> bool:
    if raw.category not in {
        IssueCategory.ACCEPTANCE_MISMATCH, IssueCategory.CONFLICTING_VALUES,
        IssueCategory.LOGICAL_INCONSISTENCY, IssueCategory.ARITHMETIC, IssueCategory.OTHER,
    }:
        return False
    if raw.category != IssueCategory.ARITHMETIC and not arithmetic_claim(raw.explanation):
        return False
    sources = {r.report_id: r for ref in [*issue.related_ids, *(e.ref for e in issue.evidence if e.verified)]
               if (r := resolver.resolve(ref))}
    for e in issue.evidence:
        if e.verified and e.line_start is not None and e.line_end is not None:
            r = resolver.requirement_at(e.line_start, e.line_end)
            if r:
                sources[r.report_id] = r
    amounts = [a for req in sources.values() if (a := checked_sum(req)) is not None]
    if len(amounts) != 1:
        return False
    start, purchase, total = amounts[0]
    return start + purchase == total


def validate_cross_issue(raw: CrossIssueLLM, issue_id: str, index: DocIndex, resolver: RefResolver) -> CrossIssue:
    notes: list[str] = []
    related: list[str] = []
    for ref in raw.related_ids:
        req = resolver.resolve(ref)
        if req is None:
            notes.append(f"Ссылка на несуществующий идентификатор «{ref}» отброшена.")
        elif req.report_id not in related:
            related.append(req.report_id)
    sections: list[str] = []
    for title in raw.related_sections:
        if resolver.section_exists(title):
            sections.append(title)
        else:
            notes.append(f"Раздел «{title}» не найден в документе — ссылка отброшена.")
    evidence, ev_notes = validate_evidence(raw.evidence, index, resolver)
    notes += ev_notes
    verified = [e for e in evidence if e.verified]
    cited_ids = []
    for ev in verified:
        req = resolver.requirement_at(ev.line_start, ev.line_end)  # verified evidence always has a span
        if req and req.report_id not in cited_ids:
            cited_ids.append(req.report_id)
    for ref in related:
        if ref not in cited_ids:
            notes.append(f"Ссылка {ref} не подтверждена цитатой и исключена из списка связанных требований.")
    related = cited_ids
    if len(verified) < len(evidence):
        notes.append(f"Цитат не найдено в документе: {len(evidence) - len(verified)} из {len(evidence)}.")

    certainty = Certainty.CONFIRMED if raw.confirmed else Certainty.POTENTIAL
    needed = 2 if raw.category in TWO_SOURCE_CATEGORIES else 1
    enough = bool(verified) if needed == 1 else _has_two_places(verified)
    if certainty == Certainty.CONFIRMED and (not enough or len(verified) != len(evidence)):
        certainty = Certainty.POTENTIAL
        notes.append("Недостаточно проверенных цитат для подтверждения " + (
            "(нужны цитаты из обоих конфликтующих мест)." if needed == 2 else "(нужна хотя бы одна цитата)."))
    if not related and not sections and not verified:
        certainty = Certainty.POTENTIAL
        notes.append("Нет проверяемой привязки к документу.")
    ceiling = Severity.MEDIUM if raw.category in {IssueCategory.DUPLICATE, IssueCategory.TERMINOLOGY} else Severity.CRITICAL
    severity = _calibrate_severity(raw.severity, certainty, evidence, ceiling)
    return CrossIssue(issue_id=issue_id, category=raw.category, severity=severity, certainty=certainty,
                      related_ids=related, related_sections=sections, explanation=raw.explanation.strip(),
                      evidence=evidence, recommendation=raw.recommendation.strip(), notes=notes)


def _has_two_places(evidence: list[EvidenceRef]) -> bool:
    spans = sorted((e.line_start, e.line_end) for e in evidence if e.line_start is not None and e.line_end is not None)
    return any(left[1] < right[0] for left, right in combinations(spans, 2))


def dedupe_cross_issues(issues: list[CrossIssue]) -> list[CrossIssue]:
    """Убирает точные повторы после чанков, сохраняя разные проблемы в одном месте."""
    seen: set[tuple] = set()
    out: list[CrossIssue] = []
    for it in issues:
        places = frozenset((e.line_start, e.line_end, normalize(e.quote)) for e in it.evidence if e.verified)
        key = (it.category, it.certainty, tuple(sorted(it.related_ids)), places, normalize(it.explanation))
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    for k, it in enumerate(out, start=1):
        it.issue_id = f"X-{k:03d}"
    return out


def validate_completeness_item(raw: CompletenessItemLLM, index: DocIndex, resolver: RefResolver) -> CompletenessFinding:
    evidence, notes = validate_evidence(raw.evidence, index, resolver)
    verified = [e for e in evidence if e.verified]
    if len(verified) < len(evidence):
        notes.append(f"Цитат не найдено в документе: {len(evidence) - len(verified)} из {len(evidence)}.")
    certainty: Certainty | None = None
    status, severity = raw.status, raw.severity
    if status in (CompletenessStatus.CONFIRMED_OMISSION, CompletenessStatus.POTENTIAL_OMISSION):
        certainty = Certainty.CONFIRMED if status == CompletenessStatus.CONFIRMED_OMISSION else Certainty.POTENTIAL
        if certainty == Certainty.CONFIRMED and (not verified or len(verified) != len(evidence)):
            certainty = Certainty.POTENTIAL
            status = CompletenessStatus.POTENTIAL_OMISSION
            notes.append("Нет проверенной цитаты, показывающей пробел, — пропуск считается потенциальным.")
        severity = _calibrate_severity(severity, certainty, evidence)
    else:
        severity = Severity.NONE
    return CompletenessFinding(area=raw.area, status=status, severity=severity, certainty=certainty,
                               explanation=raw.explanation.strip(), evidence=evidence,
                               recommendation=raw.recommendation.strip(), notes=notes)

