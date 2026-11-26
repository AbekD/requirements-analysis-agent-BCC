"""Общие заглушки для тестов: ни один тест не обращается к сети и не требует ключа."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.schemas import (  # noqa: E402
    AssessmentLLM, ClassificationResponse, CompletenessArea, CompletenessItemLLM, CompletenessResponse,
    CompletenessStatus, CrossIssueLLM, CrossResponse, Criterion, EvidenceLLM, FinalResponse, IndividualResponse,
    IssueCategory, ReqType, Severity, Status, UnitClassLLM, Verdict,
)

SAMPLE_MD = """# Тестовая спецификация

## Функциональные требования
* **FR-1.** Система должна быстро обрабатывать заявки.
* **FR-2.** Клиент задаёт лимит от 1 000 до 5 000 ₸.

## Нефункциональные требования
Сервис отвечает за 2 секунды.

## Критерии приемки

| Условие | Действие | Результат |
| :--- | :--- | :--- |
| Лимит 1 000 ₸ | Сохранить | Лимит сохранён |

## API
```json
{"result":"OK"}
```
"""


class FakeClient:
    """Скриптуемая замена LLM-клиента."""

    def __init__(self, model: str = "fake-model", fail_on: tuple[type, ...] = (), verdict: Verdict = Verdict.READY,
                 invented_cross_quote: bool = True) -> None:
        self.model = model
        self.calls = 0
        self.fail_on = fail_on
        self.verdict = verdict
        self.invented = invented_cross_quote

    def generate(self, *, system: str, prompt: str, schema):
        from src.analysis.llm_client import LLMError
        self.calls += 1
        if schema in self.fail_on:
            raise LLMError("искусственный сбой")
        if schema is ClassificationResponse:
            items = re.findall(r"\[(REQ-\d+)\] section: .*? \| current type: (\w+)", prompt)
            return ClassificationResponse(items=[UnitClassLLM(unit_id=i, type=ReqType(t)) for i, t in items])
        if schema is IndividualResponse:
            bad = "быстро" in prompt.split("=== REQUIREMENT UNDER REVIEW ===")[-1]
            out = []
            for c in Criterion:
                if bad and c == Criterion.UNAMBIGUITY:
                    out.append(AssessmentLLM(criterion=c, status=Status.FAIL, severity=Severity.CRITICAL,
                                             explanation="Слово «быстро» субъективно.", evidence=["быстро"],
                                             suggestion="Задать измеримый предел времени."))
                else:
                    out.append(AssessmentLLM(criterion=c, status=Status.PASS, explanation="Замечаний нет."))
            return IndividualResponse(assessments=out)
        if schema is CrossResponse:
            issues = [CrossIssueLLM(
                category=IssueCategory.CONFLICTING_VALUES, severity=Severity.HIGH, confirmed=True,
                related_ids=["FR-2", "REQ-999"], explanation="Диапазон лимита расходится.",
                evidence=[EvidenceLLM(ref="FR-2", quote="от 1 000 до 5 000"),
                          EvidenceLLM(ref="REQ-001", quote="быстро обрабатывать")],
                recommendation="Согласовать диапазон.")]
            if self.invented:
                issues.append(CrossIssueLLM(
                    category=IssueCategory.CONTRADICTION, severity=Severity.CRITICAL, confirmed=True,
                    related_ids=["FR-1"], explanation="Выдуманное противоречие.",
                    evidence=[EvidenceLLM(ref="FR-1", quote="этой цитаты нет в документе"),
                              EvidenceLLM(ref="FR-2", quote="и этой тоже")],
                    recommendation="—"))
            return CrossResponse(issues=issues)
        if schema is CompletenessResponse:
            return CompletenessResponse(items=[
                CompletenessItemLLM(area=a, status=CompletenessStatus.POTENTIAL_OMISSION if a == CompletenessArea.SECURITY
                                    else CompletenessStatus.COVERED,
                                    severity=Severity.MEDIUM if a == CompletenessArea.SECURITY else Severity.NONE,
                                    explanation="Безопасность не описана." if a == CompletenessArea.SECURITY else "Описано.",
                                    evidence=[EvidenceLLM(ref="Нефункциональные требования", quote="отвечает за 2 секунды")],
                                    recommendation="Уточнить требования безопасности.")
                for a in CompletenessArea])
        if schema is FinalResponse:
            return FinalResponse(verdict=self.verdict, justification="Тестовое обоснование.",
                                 priority_finding_ids=["F-001", "F-999"])
        raise AssertionError(f"неожиданная схема {schema}")


@pytest.fixture
def sample_md(tmp_path: Path) -> Path:
    p = tmp_path / "spec.md"
    p.write_text(SAMPLE_MD, encoding="utf-8")
    return p


@pytest.fixture
def fake_client() -> FakeClient:
    return FakeClient()
