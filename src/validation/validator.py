"""Детерминированная проверка результатов LLM: цитаты, ссылки на ID, места в документе.

Принцип: LLM не может «подтвердить» дефект собственным словом. Дефект считается
подтверждённым только если его цитаты дословно найдены в тексте документа.
"""
from __future__ import annotations

import re
import unicodedata

from src.models.schemas import (
    Assessment, AssessmentLLM, Certainty, CompletenessFinding, CompletenessItemLLM, CompletenessStatus,
    CrossIssue, CrossIssueLLM, Criterion, EvidenceLLM, EvidenceRef, IndividualAnalysis, IndividualResponse,
    ParsedDocument, Requirement, Severity, Status, TWO_SOURCE_CATEGORIES,
)

MIN_QUOTE_CHARS = 3
_QUOTE_MAP = str.maketrans({"«": '"', "»": '"', "“": '"', "”": '"', "„": '"', "‘": "'", "’": "'",
                            "–": "-", "—": "-", "‑": "-", "−": "-"})
_STRIP_MARKUP = re.compile(r"[*`\\]")
_ELLIPSIS = re.compile(r"\s*(?:…|\.{3})\s*")
_REF_TOKEN = re.compile(r"[A-Za-zА-Яа-яЁё]{1,6}-\d+(?:\.\d+)*")


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
        best: tuple[int, int] | None = None
        for start in self._occurrences(parts[0]):
            pos = start + len(parts[0])
            end = pos
            ok = True
            for part in parts[1:]:
                idx = self.full.find(part, pos)
                if idx == -1:
                    ok = False
                    break
                pos = end = idx + len(part)
            if not ok:
                continue
            span = (self._line_of[start], self._line_of[max(start, end - 1)])
            if hint is None or (span[0] >= hint[0] and span[1] <= hint[1]):
                return span
            best = best or span
        return best

    def _occurrences(self, needle: str):
        idx = self.full.find(needle)
        while idx != -1:
            yield idx
            idx = self.full.find(needle, idx + 1)


class RefResolver:
    """Сопоставляет ссылки LLM (REQ-005, Ф-3, «REQ-005 (Ф-3)») с существующими требованиями."""

    def __init__(self, requirements: list[Requirement], doc: ParsedDocument) -> None:
        self._by_key: dict[str, Requirement] = {}
        for r in requirements:
            self._by_key[self._key(r.report_id)] = r
        for r in requirements:
            if r.doc_id:
                self._by_key.setdefault(self._key(r.doc_id), r)
        self._headings = [h.title.casefold() for h in doc.headings]

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

    def known_ids(self) -> set[str]:
        return {r.report_id for r in self._by_key.values()}

    def section_exists(self, title: str) -> bool:
        t = title.casefold().strip()
        return bool(t) and any(t in h or h in t for h in self._headings)


# --------------------------------------------------------------------------- #
def build_evidence(quote: str, index: DocIndex, ref: str = "", hint: tuple[int, int] | None = None) -> EvidenceRef:
    span = index.locate(quote, hint)
    if span is None:
        return EvidenceRef(quote=quote.strip(), ref=ref, verified=False)
    return EvidenceRef(quote=quote.strip(), ref=ref, verified=True, line_start=span[0], line_end=span[1],
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


# --------------------------------------------------------------------------- #
# Индивидуальный анализ
# --------------------------------------------------------------------------- #
def validate_individual(resp: IndividualResponse, req: Requirement, index: DocIndex) -> IndividualAnalysis:
    hint = (req.line_start, req.line_end)
    assessments = [_validate_assessment(a, index, hint) for a in resp.assessments]
    order = {c: i for i, c in enumerate(Criterion)}
    assessments.sort(key=lambda a: order[a.criterion])
    return IndividualAnalysis(req_id=req.report_id, ok=True, assessments=assessments)


def _validate_assessment(a: AssessmentLLM, index: DocIndex, hint: tuple[int, int]) -> Assessment:
    evidence = [build_evidence(q, index, hint=hint) for q in _dedupe_quotes(a.evidence)]
    notes: list[str] = []
    severity = a.severity
    certainty: Certainty | None = None
    if a.status in (Status.FAIL, Status.UNCERTAIN):
        verified = [e for e in evidence if e.verified]
        if a.status == Status.FAIL:
            if verified:
                certainty = Certainty.CONFIRMED
            else:
                certainty = Certainty.POTENTIAL
                notes.append("Цитата-доказательство не найдена в документе — замечание не считается подтверждённым дефектом."
                             if evidence else "Модель не привела цитату — замечание не считается подтверждённым дефектом.")
        else:
            certainty = Certainty.POTENTIAL
        if evidence and len(verified) < len(evidence):
            notes.append(f"Не найдено в документе цитат: {len(evidence) - len(verified)} из {len(evidence)}.")
        if severity == Severity.NONE:
            severity = Severity.MEDIUM if a.status == Status.FAIL else Severity.LOW
    else:
        severity = Severity.NONE
    return Assessment(criterion=a.criterion, status=a.status, severity=severity, explanation=a.explanation.strip(),
                      evidence=evidence, suggestion=a.suggestion.strip(), certainty=certainty, notes=notes)


# --------------------------------------------------------------------------- #
# Анализ набора требований
# --------------------------------------------------------------------------- #
def _evidence_from_llm(items: list[EvidenceLLM], index: DocIndex, resolver: RefResolver
                       ) -> tuple[list[EvidenceRef], list[str]]:
    out: list[EvidenceRef] = []
    notes: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = normalize(item.quote)
        if not key or key in seen:
            continue
        seen.add(key)
        req = resolver.resolve(item.ref)
        hint = (req.line_start, req.line_end) if req else None
        ev = build_evidence(item.quote, index, ref=req.report_id if req else item.ref.strip(), hint=hint)
        if ev.verified and hint and not (hint[0] <= ev.line_start and ev.line_end <= hint[1]):  # type: ignore[operator]
            notes.append(f"Цитата, приписанная {req.report_id}, найдена в другом месте документа (см. местоположение).")  # type: ignore[union-attr]
        out.append(ev)
    return out, notes


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
    evidence, ev_notes = _evidence_from_llm(raw.evidence, index, resolver)
    notes += ev_notes
    for ev in evidence:                      # ID из ссылок цитат тоже считаются связанными
        if ev.ref in resolver.known_ids() and ev.ref not in related:
            related.append(ev.ref)
    verified = [e for e in evidence if e.verified]
    if len(verified) < len(evidence):
        notes.append(f"Цитат не найдено в документе: {len(evidence) - len(verified)} из {len(evidence)}.")

    certainty = Certainty.CONFIRMED if raw.confirmed else Certainty.POTENTIAL
    needed = 2 if raw.category in TWO_SOURCE_CATEGORIES else 1
    if certainty == Certainty.CONFIRMED and len(_distinct_places(verified)) < needed:
        certainty = Certainty.POTENTIAL
        notes.append("Недостаточно проверенных цитат для подтверждения " + (
            "(нужны цитаты из обоих конфликтующих мест)." if needed == 2 else "(нужна хотя бы одна цитата)."))
    if not related and not sections and not verified:
        certainty = Certainty.POTENTIAL
        notes.append("Нет проверяемой привязки к документу.")
    severity = raw.severity if raw.severity != Severity.NONE else Severity.MEDIUM
    return CrossIssue(issue_id=issue_id, category=raw.category, severity=severity, certainty=certainty,
                      related_ids=related, related_sections=sections, explanation=raw.explanation.strip(),
                      evidence=evidence, recommendation=raw.recommendation.strip(), notes=notes)


def _distinct_places(evidence: list[EvidenceRef]) -> set[tuple[int | None, int | None]]:
    return {(e.line_start, e.line_end) for e in evidence}


def dedupe_cross_issues(issues: list[CrossIssue]) -> list[CrossIssue]:
    """Убирает повторы (после чанков): одинаковые категория + множество ID + первая цитата."""
    seen: set[tuple] = set()
    out: list[CrossIssue] = []
    for it in issues:
        first = normalize(it.evidence[0].quote) if it.evidence else normalize(it.explanation)[:80]
        key = (it.category, tuple(sorted(it.related_ids)), first)
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    for k, it in enumerate(out, start=1):
        it.issue_id = f"X-{k:03d}"
    return out


def validate_completeness_item(raw: CompletenessItemLLM, index: DocIndex, resolver: RefResolver) -> CompletenessFinding:
    evidence, notes = _evidence_from_llm(raw.evidence, index, resolver)
    verified = [e for e in evidence if e.verified]
    if len(verified) < len(evidence):
        notes.append(f"Цитат не найдено в документе: {len(evidence) - len(verified)} из {len(evidence)}.")
    certainty: Certainty | None = None
    status, severity = raw.status, raw.severity
    if status in (CompletenessStatus.CONFIRMED_OMISSION, CompletenessStatus.POTENTIAL_OMISSION):
        certainty = Certainty.CONFIRMED if status == CompletenessStatus.CONFIRMED_OMISSION else Certainty.POTENTIAL
        if certainty == Certainty.CONFIRMED and not verified:
            certainty = Certainty.POTENTIAL
            notes.append("Нет проверенной цитаты, показывающей пробел, — пропуск считается потенциальным.")
        if severity == Severity.NONE:
            severity = Severity.MEDIUM
    else:
        severity = Severity.NONE
    return CompletenessFinding(area=raw.area, status=status, severity=severity, certainty=certainty,
                               explanation=raw.explanation.strip(), evidence=evidence,
                               recommendation=raw.recommendation.strip(), notes=notes)

