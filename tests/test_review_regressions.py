from src.analysis.acceptance import check_explicit_rules
from src.analysis.analyzer import assess_final, run_analysis
from src.analysis.cross_analysis import run_completeness, run_cross_analysis
from src.analysis.findings import build_findings
from src.analysis.individual import analyze_batch, link_cross_review, verify_local_omissions
from src.ingestion.extractor import extract_requirements
from src.ingestion.parser import parse_file
from src.models.schemas import (
    AnalysisResult, AssessmentLLM, Certainty, CompletenessArea, CompletenessItemLLM, CompletenessStatus,
    CompletenessResponse, CrossIssueLLM, CrossResponse, Criterion, EvidenceLLM, IndividualBatchItem,
    IndividualBatchResponse, IndividualResponse, IssueCategory, LocalOmissionReviewItem,
    LocalOmissionReviewResponse, OmissionDecision, OmissionReviewItem, OmissionReviewResponse, ReqType, Severity,
    Status, Verdict,
)
from src.reporting.report_generator import render_report
from src.validation.validator import (
    DocIndex, RefResolver, validate_completeness_item, validate_cross_issue, validate_individual,
)
from tests.conftest import FakeClient


SPEC = """# Пример

## Функциональные требования
* **Ф-2.** Месяц определяется по дате совершения покупки в Алматы. В расходы входят проведённые покупки.

## Критерии приёмки
| Условия | Действие | Результат |
| --- | --- | --- |
| Расходы 48 500 ₸, порог 50 000 ₸ | Провести покупку на 1 500 ₸ | Итог 50 000 ₸ и один push |
| Покупка 31 января в 23:58 на 4 000 ₸ | Провести 1 февраля в 00:03 по Алматы | Февральский счётчик вырос на 4 000 ₸ |

## REST API
| Поле | Значение |
| --- | --- |
| REST type | GET |
| Description | Сохранение порога, чтение порога |
| amount | Целое число |

### Логика работы метода
Заменить настройку и вернуть результат сохранения.
"""


def setup_spec(tmp_path, source=SPEC):
    path = tmp_path / "spec.md"
    path.write_text(source, encoding="utf-8")
    doc = parse_file(path)
    reqs, _ = extract_requirements(doc)
    return doc, reqs, DocIndex(doc), RefResolver(reqs, doc)


def assessment(criterion, explanation, quote):
    return IndividualResponse(assessments=[
        AssessmentLLM(criterion=c, status=Status.FAIL, severity=Severity.CRITICAL,
                      explanation=explanation, evidence=[quote]) if c == criterion
        else AssessmentLLM(criterion=c, status=Status.PASS, explanation="Согласовано")
        for c in Criterion
    ])


def test_unverified_or_misattributed_evidence_cannot_confirm_omission(tmp_path):
    doc, reqs, index, resolver = setup_spec(tmp_path)
    rule = next(r for r in reqs if r.doc_id == "Ф-2")
    raw = CompletenessItemLLM(
        area=CompletenessArea.BUSINESS_RULES, status=CompletenessStatus.CONFIRMED_OMISSION,
        severity=Severity.CRITICAL, explanation="Сценарий не определён.",
        evidence=[EvidenceLLM(ref=rule.report_id, quote="Февральский счётчик"),
                  EvidenceLLM(ref="REST API", quote="выдуманная цитата")],
    )
    item = validate_completeness_item(raw, index, resolver)
    assert item.status == CompletenessStatus.POTENTIAL_OMISSION
    assert item.certainty == Certainty.POTENTIAL and item.severity == Severity.LOW
    result = AnalysisResult(document_name=doc.name, started_at="сегодня", model="fake", agent_version="test",
                            cross_complete=True, completeness_complete=True, completeness=[item])
    assert "выдуманная цитата" not in render_report(result)


def test_correct_arithmetic_and_named_parameter_do_not_create_findings(tmp_path):
    _, reqs, index, _ = setup_spec(tmp_path)
    row = next(r for r in reqs if "48 500" in r.text)
    param = next(r for r in reqs if "| amount |" in r.text)
    maths = assessment(Criterion.CORRECTNESS, "В примере арифметическая ошибка.", "Итог 50 000 ₸")
    id_only = assessment(Criterion.TRACEABILITY, "Нет уникального идентификатора.", "Целое число")
    broken_link = assessment(Criterion.TRACEABILITY, "Нет идентификатора и отсутствует связь с правилом.",
                             "Целое число")
    assert validate_individual(maths, row, index).assessments[0].status == Status.PASS
    assert validate_individual(id_only, param, index).assessments[6].status == Status.PASS
    assert validate_individual(broken_link, param, index).assessments[6].status == Status.FAIL


class CrossClient(FakeClient):
    def __init__(self, issues):
        super().__init__(verdict=Verdict.NOT_READY)
        self.issues = issues

    def generate(self, *, system, prompt, schema):
        if schema is CrossResponse:
            return CrossResponse(issues=self.issues)
        return super().generate(system=system, prompt=prompt, schema=schema)


def test_cross_checks_filter_false_arithmetic_and_find_month_boundary(tmp_path):
    doc, reqs, index, resolver = setup_spec(tmp_path)
    row = next(r for r in reqs if "48 500" in r.text)
    rule = next(r for r in reqs if r.doc_id == "Ф-2")
    raw = CrossIssueLLM(
        category=IssueCategory.ACCEPTANCE_MISMATCH, severity=Severity.HIGH, confirmed=True,
        explanation="В примере арифметическая ошибка. Итог должен быть 50 000 ₸.", recommendation="Исправить итог.",
        evidence=[EvidenceLLM(ref=row.report_id, quote="Итог 50 000 ₸"),
                  EvidenceLLM(ref=rule.report_id, quote="В расходы входят проведённые покупки.")],
    )
    issues, complete, errors = run_cross_analysis(CrossClient([raw]), doc, index, resolver, reqs)
    assert complete and not errors
    assert len(issues) == 1 and issues[0].category == IssueCategory.ACCEPTANCE_MISMATCH
    assert issues[0].certainty == Certainty.CONFIRMED
    assert all(e.verified for e in issues[0].evidence)
    assert "31 января" in issues[0].explanation and "1 февраля" in issues[0].explanation

    doc, reqs, index, resolver = setup_spec(tmp_path, SPEC.replace("Итог 50 000", "Итог 50 500"))
    issues, _, _ = run_cross_analysis(CrossClient([]), doc, index, resolver, reqs)
    assert any(x.category == IssueCategory.ARITHMETIC and x.certainty == Certainty.CONFIRMED for x in issues)


def test_root_cause_merges_get_side_effects_and_keeps_description_disagreement(tmp_path):
    doc, reqs, index, resolver = setup_spec(tmp_path)
    method = next(r for r in reqs if "| GET |" in r.text)
    description = next(r for r in reqs if "| Description |" in r.text)
    behaviour = next(r for r in reqs if "Заменить настройку" in r.text)
    issues = []
    for i, (explanation, refs) in enumerate([
        ("GET используется для сохранения настроек.", [method, description]),
        ("GET используется для изменения состояния.", [method, behaviour]),
        ("GET is used to modify data.", [method, behaviour]),
        ("Описание операции смешивает чтение и сохранение.", [description, behaviour]),
    ], 1):
        raw = CrossIssueLLM(
            category=IssueCategory.LOGICAL_INCONSISTENCY, severity=Severity.HIGH, confirmed=True,
            explanation=explanation, recommendation="Уточнить контракт.",
            evidence=[EvidenceLLM(ref=r.report_id, quote=r.text) for r in refs],
        )
        issues.append(validate_cross_issue(raw, f"X-{i:03}", index, resolver))
    result = AnalysisResult(document_name=doc.name, started_at="сегодня", model="fake", agent_version="test",
                            requirements=reqs, cross_issues=issues, cross_complete=True)
    result.findings, result.duplicates_merged = build_findings(result)
    assert len(result.findings) == 2 and result.duplicates_merged == 2
    merged = next(f for f in result.findings if len(f.cross_issue_ids) == 3)
    assert set(merged.related_ids) == {method.report_id, description.report_id, behaviour.report_id}
    cross_section = render_report(result).split("## 4. ")[1].split("## 5. ")[0]
    assert "X-001" in cross_section and "X-002" not in cross_section and "X-004" in cross_section


def test_potential_findings_cannot_set_not_ready(tmp_path):
    doc, reqs, index, resolver = setup_spec(tmp_path)
    row = next(r for r in reqs if r.type == ReqType.ACCEPTANCE)
    raw = CrossIssueLLM(
        category=IssueCategory.OTHER, severity=Severity.CRITICAL, confirmed=True,
        explanation="Непроверенное замечание.", recommendation="Проверить.",
        evidence=[EvidenceLLM(ref=row.report_id, quote="нет такой цитаты в исходнике")],
    )
    issue = validate_cross_issue(raw, "X-001", index, resolver)
    client = CrossClient([])
    result = run_analysis(doc, client, "test")
    result.cross_issues = [issue]
    result.findings, _ = build_findings(result)
    result.final = assess_final(client, result, "ru")
    assert issue.severity == Severity.LOW and result.final.verdict == Verdict.READY_WITH_RESERVATIONS
    assert result.final.adjusted


def test_recheck_recovers_conflict_and_excludes_uncited_ids(tmp_path):
    source = """# Бронирование

## Правила
FR-01. Бронь разрешено отменять не менее чем за 30 минут.
FR-02. Бронь разрешено отменять только за 60 минут или более.
FR-03. Система отображает номер брони.
"""
    doc, reqs, index, resolver = setup_spec(tmp_path, source)
    first, second, unrelated = reqs
    corrected = CrossIssueLLM(
        category=IssueCategory.CONFLICTING_VALUES, confirmed=True, severity=Severity.HIGH,
        related_ids=[first.report_id, second.report_id, unrelated.report_id],
        evidence=[EvidenceLLM(ref=r.report_id, quote=r.text) for r in (first, second)],
        explanation="Конфликтуют пороги отмены 30 и 60 минут.", recommendation="Согласовать порог.",
    )
    incorrect = corrected.model_copy(update={
        "category": IssueCategory.DUPLICATE,
        "evidence": [EvidenceLLM(ref=unrelated.report_id, quote=first.text), corrected.evidence[1]],
    })

    class RecheckClient(CrossClient):
        def generate(self, *, system, prompt, schema):
            assert schema is CrossResponse
            issue = corrected if "RECHECK UNVERIFIED CANDIDATES" in prompt else incorrect
            return CrossResponse(issues=[issue])

    issues, complete, errors = run_cross_analysis(RecheckClient([]), doc, index, resolver, reqs)
    assert complete and not errors and len(issues) == 1
    assert issues[0].category == IssueCategory.CONFLICTING_VALUES
    assert issues[0].certainty == Certainty.CONFIRMED
    assert issues[0].related_ids == [first.report_id, second.report_id]


def test_same_missing_field_is_counted_once_across_criteria(tmp_path):
    doc, reqs, index, _ = setup_spec(tmp_path, """# Бронирование

## Правила
FR-01. После истечения времени удержания бронь освобождается.
""")
    req = reqs[0]
    response = IndividualResponse(assessments=[
        AssessmentLLM(criterion=c, status=Status.FAIL, severity=Severity.HIGH,
                      explanation=f"{c.value}: не определено время удержания.",
                      evidence=["времени удержания"], missing_field="времени удержания")
        if c in (Criterion.UNAMBIGUITY, Criterion.VERIFIABILITY)
        else AssessmentLLM(criterion=c, status=Status.PASS, explanation="Согласовано")
        for c in Criterion
    ])
    result = AnalysisResult(document_name=doc.name, started_at="сегодня", model="fake", agent_version="test",
                            requirements=reqs, individual=[validate_individual(response, req, index)])
    findings, merged = build_findings(result)
    assert merged == 1 and len(findings) == 1
    assert set(findings[0].criteria) == {Criterion.UNAMBIGUITY.value, Criterion.VERIFIABILITY.value}


def test_final_reason_uses_confirmed_severity_instead_of_model_claim(tmp_path):
    doc, _, _, _ = setup_spec(tmp_path)
    result = run_analysis(doc, CrossClient([]), "test")
    assert result.final.verdict == Verdict.NOT_READY
    assert "Подтверждённые дефекты высокой серьёзности (1): F-001." in result.final.justification
    assert "критические" not in result.final.justification


def test_individual_cross_claim_links_to_verified_pair_without_hiding_local_gap(tmp_path):
    doc, reqs, index, resolver = setup_spec(tmp_path, """# Правила

FR-01. Бронь разрешено отменять за 30 минут.
FR-02. Бронь разрешено отменять только за 60 минут.
""")
    first, second = reqs
    raw = CrossIssueLLM(
        category=IssueCategory.CONFLICTING_VALUES, confirmed=True, severity=Severity.HIGH,
        evidence=[EvidenceLLM(ref=r.report_id, quote=r.text) for r in reqs],
        explanation="Разные пороги отмены.", recommendation="Согласовать срок.",
    )
    issue = validate_cross_issue(raw, "X-001", index, resolver)
    assessments = assessment(Criterion.UNAMBIGUITY, "Есть несоответствие с FR-02.", first.text)
    local_gap = AssessmentLLM(
        criterion=Criterion.COMPLETENESS, status=Status.FAIL, severity=Severity.MEDIUM,
        explanation="Не определено сообщение при отказе.", evidence=[first.text],
    )
    assessments.assessments = [local_gap if a.criterion == local_gap.criterion else a
                               for a in assessments.assessments]
    individual = validate_individual(assessments, first, index)
    result = AnalysisResult(
        document_name=doc.name, started_at="сегодня", model="fake", agent_version="test",
        requirements=reqs, individual=[individual], cross_issues=[issue],
    )
    assert link_cross_review(result, resolver, "ru") == 1
    findings, _ = build_findings(result)
    assert len(findings) == 2
    by_criterion = {a.criterion: a for a in individual.assessments}
    assert by_criterion[Criterion.UNAMBIGUITY].explanation.endswith("X-001.")
    assert by_criterion[Criterion.COMPLETENESS].status == Status.FAIL


def test_completeness_rechecks_scenario_against_actual_outcome(tmp_path):
    doc, reqs, index, resolver = setup_spec(tmp_path, """# Лист ожидания

FR-01. После истечения времени удержания система предлагает слот следующему сотруднику.
""")
    req = reqs[0]
    scenarios, limits = CompletenessArea.ALTERNATIVE_SCENARIOS, CompletenessArea.BOUNDARY_CONDITIONS
    quote = EvidenceLLM(ref=req.report_id, quote=req.text)

    class ReviewClient:
        def generate(self, *, system, prompt, schema):
            if schema is CompletenessResponse:
                return schema(items=[
                    CompletenessItemLLM(
                        area=area, status=CompletenessStatus.CONFIRMED_OMISSION, severity=Severity.HIGH,
                        explanation="Не описано, что происходит после истечения времени удержания."
                        if area == scenarios else "Не задано время удержания.", evidence=[quote],
                    ) if area in (scenarios, limits) else CompletenessItemLLM(
                        area=area, status=CompletenessStatus.NOT_APPLICABLE, explanation="Не применимо.",
                    ) for area in CompletenessArea
                ])
            assert schema is OmissionReviewResponse
            return schema(items=[
                OmissionReviewItem(area=scenarios, decision=OmissionDecision.COVERED,
                                   explanation="Слот предлагаетcя следующему сотруднику.", evidence=[quote]),
                OmissionReviewItem(area=limits, decision=OmissionDecision.SUPPORTED,
                                   explanation="Время удержания не задано."),
            ])

    items, completed, errors = run_completeness(ReviewClient(), doc, index, resolver, reqs)
    assert completed and not errors
    by_area = {item.area: item for item in items}
    assert by_area[scenarios].status == CompletenessStatus.COVERED
    assert by_area[scenarios].evidence[0].verified
    assert by_area[limits].status == CompletenessStatus.CONFIRMED_OMISSION


def test_batch_preserves_target_order_and_retries_only_missing_or_duplicate(tmp_path):
    doc, reqs, index, _ = setup_spec(tmp_path, """# Бронирование

FR-01. Открыть экран.
FR-02. Показать номер брони.
FR-03. Закрыть экран.
""")
    first, second, third = reqs
    passed = IndividualResponse(assessments=[
        AssessmentLLM(criterion=c, status=Status.PASS, explanation="Согласовано.") for c in Criterion
    ])

    class BatchClient:
        calls = 0

        def generate(self, *, system, prompt, schema):
            self.calls += 1
            if schema is IndividualBatchResponse:
                return schema(items=[IndividualBatchItem(req_id=rid, response=passed) for rid in (
                    second.report_id, "REQ-999", first.report_id, second.report_id,
                )])
            assert schema is IndividualResponse
            assert any(f"Report ID: {req.report_id}" in prompt for req in (second, third))
            return passed

    client = BatchClient()
    analyses = analyze_batch(client, doc, index, reqs, "ru")
    assert client.calls == 3
    assert [item.req_id for item in analyses] == [req.report_id for req in reqs]
    assert all(item.ok and len(item.assessments) == len(Criterion) for item in analyses)


def test_local_omission_uses_verified_evidence_from_another_section(tmp_path):
    doc, reqs, index, resolver = setup_spec(tmp_path, """# Спецификация

## Логика API
FR-01. Проверить допустимый диапазон суммы.

## Входные данные
FR-02. Допустимая сумма от 100 до 1000 тенге.
""")
    step, details = reqs
    response = assessment(Criterion.COMPLETENESS, "Диапазон суммы отсутствует.", step.text)

    class ReviewClient:
        def __init__(self, quote):
            self.quote = quote

        def generate(self, *, system, prompt, schema):
            assert schema is LocalOmissionReviewResponse
            return schema(items=[LocalOmissionReviewItem(
                req_id=step.report_id, decision=OmissionDecision.COVERED,
                explanation="Диапазон определён во входных данных.",
                evidence=[EvidenceLLM(ref=details.report_id, quote=self.quote)],
            )])

    for quote, expected in ((details.text, Status.PASS), ("До пяти тысяч", Status.UNCERTAIN)):
        result = AnalysisResult(
            document_name=doc.name, started_at="сегодня", model="fake", agent_version="test",
            requirements=reqs, individual=[validate_individual(response, step, index)],
        )
        verify_local_omissions(ReviewClient(quote), doc, index, resolver, result, "ru")
        reviewed = next(a for a in result.individual[0].assessments if a.criterion == Criterion.COMPLETENESS)
        assert reviewed.status == expected
        if reviewed.status == Status.PASS:
            assert reviewed.evidence[0].location.startswith("строка 7")


def test_batch_accepts_unique_original_ids_without_falling_back(tmp_path):
    doc, reqs, index, resolver = setup_spec(tmp_path, """# Бронирование

FR-01. Открыть экран.
FR-02. Показать номер брони.
""")
    passed = IndividualResponse(assessments=[
        AssessmentLLM(criterion=c, status=Status.PASS, explanation="Согласовано.") for c in Criterion
    ])

    class OriginalIdClient:
        def generate(self, *, system, prompt, schema):
            assert schema is IndividualBatchResponse
            return schema(items=[IndividualBatchItem(req_id=req.doc_id, response=passed) for req in reversed(reqs)])

    analyses = analyze_batch(OriginalIdClient(), doc, index, reqs, "ru", resolver)
    assert [a.req_id for a in analyses] == [r.report_id for r in reqs]


def test_explicit_rules_ground_push_conflict_and_api_write_in_actual_lines(tmp_path):
    source = """# Уведомления

## Бизнес-правила
FR-01. На карту приходится максимум один push за календарный месяц.
FR-02. При изменении порога сумма сохраняется. Если она ниже нового порога, следующее достижение вызывает push независимо от истории отправки.

## REST API
### Параметры
| Поле | Значение |
| --- | --- |
| REST type | GET |

### Логика
1. Проверить пользовательскую сессию.
2. Заменить настройку и вернуть срок действия.
"""
    _, reqs, index, resolver = setup_spec(tmp_path, source)
    findings = check_explicit_rules(reqs, "ru")
    by_category = {raw.category: validate_cross_issue(raw, "X-001", index, resolver) for raw in findings}
    push = by_category[IssueCategory.CONTRADICTION]
    assert push.certainty == Certainty.CONFIRMED and len(push.related_ids) == 2
    write = by_category[IssueCategory.CROSS_DOCUMENT]
    assert write.certainty == Certainty.CONFIRMED
    assert any("Заменить настройку" in ev.quote for ev in write.evidence)
    assert all("пользовательскую сессию" not in ev.quote for ev in write.evidence)

    _, reqs, _, _ = setup_spec(tmp_path, source.replace("| GET |", "| POST |")
                               .replace("независимо от истории отправки", "при соблюдении месячного лимита"))
    assert not check_explicit_rules(reqs, "ru")


def test_error_example_under_http_200_conflicts_with_invalid_parameters_rule(tmp_path):
    source = """# REST API

## Статусы
**400.** Ответ в случае, если параметры запроса заполнены неверно.

### 200. Successful
```json
{"result":"ERROR","message":"Проверьте сумму"}
```
"""
    _, reqs, index, resolver = setup_spec(tmp_path, source)
    raw, = check_explicit_rules(reqs, "ru")
    issue = validate_cross_issue(raw, "X-001", index, resolver)
    assert issue.category == IssueCategory.CONFLICTING_VALUES
    assert issue.certainty == Certainty.CONFIRMED
    assert len(issue.evidence) == 2 and all(ev.verified for ev in issue.evidence)

    _, reqs, _, _ = setup_spec(tmp_path, source.replace('"ERROR"', '"OK"'))
    assert not check_explicit_rules(reqs, "ru")
