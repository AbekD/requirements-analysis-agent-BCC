from src.analysis.analyzer import run_analysis
from src.config import ConfigError, get_settings
from src.ingestion.parser import parse_file
from src.main import EXIT_INPUT, main
from src.models.schemas import (
    Certainty, CompletenessResponse, IndividualResponse, RunStatus, Severity, Verdict,
)
from src.reporting.report_generator import render_report
from tests.conftest import FakeClient


def _run(sample_md, client=None):
    return run_analysis(parse_file(sample_md), client or FakeClient(), "test-version")


def test_end_to_end_report_has_all_sections(sample_md):
    result = _run(sample_md)
    assert result.status == RunStatus.COMPLETE
    report = render_report(result)
    for heading in ("## 1. Паспорт прогона", "## 2. Извлечение требований", "## 3. Индивидуальная проверка",
                    "## 4. Проверка набора требований", "## 5. Сводка", "## 6. Критические дефекты",
                    "## 7. Итоговая оценка", "## 8. Приоритетные рекомендации"):
        assert heading in report
    assert "fake-model" in report and "test-version" in report and "REQ-001" in report and "FR-1" in report


def test_invented_cross_issue_is_not_confirmed_and_verdict_is_not_softer_than_rules(sample_md):
    result = _run(sample_md)
    invented = next(x for x in result.cross_issues if "Выдуманное" in x.explanation)
    assert invented.certainty == Certainty.POTENTIAL
    real = next(x for x in result.cross_issues if "расходится" in x.explanation)
    assert real.certainty == Certainty.CONFIRMED and "REQ-999" not in real.related_ids
    # индивидуальный FAIL (CRITICAL, с проверенной цитатой) → LLM сказала READY, правила ужесточают до NOT_READY
    assert result.final.verdict == Verdict.NOT_READY and result.final.adjusted
    assert all(i in {f.finding_id for f in result.findings} for i in result.final.priority_finding_ids)


def test_no_double_counting_within_one_requirement(sample_md):
    result = _run(sample_md)
    crit = [f for f in result.findings if f.severity == Severity.CRITICAL and f.source == "INDIVIDUAL"]
    assert len(crit) == 1


def test_failed_individual_analysis_is_recorded_not_dropped(sample_md):
    result = _run(sample_md, FakeClient(fail_on=(IndividualResponse,)))
    assert result.status == RunStatus.FAILED
    assert all(not a.ok for a in result.individual) and result.errors


def test_partial_status_when_completeness_fails(sample_md):
    result = _run(sample_md, FakeClient(fail_on=(CompletenessResponse,)))
    assert result.status == RunStatus.PARTIAL and not result.completeness_complete
    report = render_report(result)
    assert "ЧАСТИЧНО" in report and "Анализ полноты спецификации не выполнен" in report
    assert result.final.verdict != Verdict.READY


def test_cli_exit_codes(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("src.config.load_dotenv", lambda *a, **k: None)
    assert main(["--input", str(tmp_path / "missing.md")]) == EXIT_INPUT
    empty = tmp_path / "e.md"
    empty.write_text("", encoding="utf-8")
    assert main(["--input", str(empty)]) == EXIT_INPUT
    bad = tmp_path / "x.docx"
    bad.write_text("x", encoding="utf-8")
    assert main(["--input", str(bad)]) == EXIT_INPUT
    ok = tmp_path / "ok.md"
    ok.write_text("# T\nСистема должна работать.\n", encoding="utf-8")
    assert main(["--input", str(ok)]) == EXIT_INPUT          # нет GEMINI_API_KEY


def test_missing_api_key_message(monkeypatch):
    import pytest
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("src.config.load_dotenv", lambda *a, **k: None)
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        get_settings()
