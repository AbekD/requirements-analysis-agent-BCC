import pytest
from pydantic import ValidationError

from src.analysis.llm_client import parse_model
from src.ingestion.extractor import extract_requirements
from src.ingestion.parser import parse_file
from src.models.schemas import (
    AssessmentLLM, Certainty, Criterion, CrossIssueLLM, EvidenceLLM, IndividualResponse, IssueCategory, Severity, Status,
)
from src.validation.validator import (
    DocIndex, RefResolver, build_evidence, normalize, validate_cross_issue, validate_individual,
)


@pytest.fixture
def ctx(sample_md):
    doc = parse_file(sample_md)
    reqs, _ = extract_requirements(doc)
    return doc, reqs, DocIndex(doc), RefResolver(reqs, doc)


def test_normalize_ignores_markup_quotes_and_case():
    assert normalize("**FR-1.**  «Быстро» — да") == normalize('fr-1. "быстро" - ДА')


def test_evidence_found_with_location(ctx):
    doc, _, index, _ = ctx
    ev = build_evidence("быстро обрабатывать заявки", index)
    assert ev.verified and ev.line_start == ev.line_end and "строка" in ev.location


def test_evidence_with_ellipsis_fragments(ctx):
    _, _, index, _ = ctx
    assert build_evidence("Клиент задаёт … до 5 000 ₸", index).verified


def test_invented_evidence_rejected(ctx):
    _, _, index, _ = ctx
    assert not build_evidence("Система обязана шифровать данные", index).verified
    assert not build_evidence("", index).verified


def test_ref_resolver_handles_doc_and_report_ids_and_garbage(ctx):
    _, reqs, _, resolver = ctx
    fr1 = next(r for r in reqs if r.doc_id == "FR-1")
    assert resolver.resolve("FR-1") is fr1
    assert resolver.resolve(fr1.report_id) is fr1
    assert resolver.resolve(f"{fr1.report_id} (FR-1)") is fr1
    assert resolver.resolve("REQ-999") is None


def _assess(status, evidence):
    out = [AssessmentLLM(criterion=c, status=Status.PASS, explanation="ок") for c in Criterion]
    out[1] = AssessmentLLM(criterion=Criterion.UNAMBIGUITY, status=status, explanation="x", evidence=evidence,
                           severity=Severity.HIGH)
    return IndividualResponse(assessments=out)


def test_fail_with_verified_quote_is_confirmed(ctx):
    _, reqs, index, _ = ctx
    req = next(r for r in reqs if r.doc_id == "FR-1")
    a = validate_individual(_assess(Status.FAIL, ["быстро"]), req, index).assessments[1]
    assert a.certainty == Certainty.CONFIRMED and a.evidence[0].verified


def test_fail_with_invented_quote_is_not_confirmed(ctx):
    _, reqs, index, _ = ctx
    req = next(r for r in reqs if r.doc_id == "FR-1")
    a = validate_individual(_assess(Status.FAIL, ["такого текста нет"]), req, index).assessments[1]
    assert a.certainty == Certainty.POTENTIAL and a.notes


def test_fail_without_quote_is_not_confirmed(ctx):
    _, reqs, index, _ = ctx
    a = validate_individual(_assess(Status.FAIL, []), reqs[0], index).assessments[1]
    assert a.certainty == Certainty.POTENTIAL


def test_cross_issue_needs_two_verified_quotes_and_drops_unknown_ids(ctx):
    _, _, index, resolver = ctx
    raw = CrossIssueLLM(category=IssueCategory.CONTRADICTION, severity=Severity.CRITICAL, confirmed=True,
                        related_ids=["FR-1", "REQ-999"], related_sections=["Несуществующий раздел"],
                        explanation="e", recommendation="r",
                        evidence=[EvidenceLLM(ref="FR-1", quote="быстро обрабатывать"),
                                  EvidenceLLM(ref="FR-2", quote="выдумано")])
    issue = validate_cross_issue(raw, "X-001", index, resolver)
    assert issue.certainty == Certainty.POTENTIAL               # вторая цитата не проверена
    assert all("REQ-999" not in i for i in issue.related_ids) and not issue.related_sections
    ok = raw.model_copy(update={"evidence": [EvidenceLLM(ref="FR-1", quote="быстро обрабатывать"),
                                              EvidenceLLM(ref="FR-2", quote="от 1 000 до 5 000")]})
    assert validate_cross_issue(ok, "X-002", index, resolver).certainty == Certainty.CONFIRMED


def test_invalid_llm_output_rejected():
    with pytest.raises(ValueError):
        parse_model("это не JSON", IndividualResponse)
    with pytest.raises(ValidationError):
        parse_model('{"assessments": []}', IndividualResponse)     # нет критериев
    with pytest.raises(ValueError):
        parse_model("", IndividualResponse)


def test_json_in_code_fence_is_accepted():
    raw = '```json\n{"assessments": ' + IndividualResponse(assessments=[
        AssessmentLLM(criterion=c, status=Status.PASS, explanation="ок") for c in Criterion]
    ).model_dump_json(include={"assessments"}).split('"assessments":', 1)[1] + "\n```"
    assert len(parse_model(raw, IndividualResponse).assessments) == 10
