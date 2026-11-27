"""Pydantic-модели: документ, требования, ответы LLM и результаты анализа."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


# --------------------------------------------------------------------------- #
# Перечисления
# --------------------------------------------------------------------------- #
class Criterion(str, Enum):
    CORRECTNESS = "Correctness"
    UNAMBIGUITY = "Unambiguity"
    COMPLETENESS = "Completeness"
    VERIFIABILITY = "Verifiability"
    FEASIBILITY = "Feasibility"
    NECESSITY = "Necessity"
    TRACEABILITY = "Traceability"
    ATOMICITY = "Atomicity"
    DESIGN_INDEPENDENCE = "DesignIndependence"
    CONSISTENCY = "Consistency"


CRITERION_RU: dict[Criterion, str] = {
    Criterion.CORRECTNESS: "Корректность",
    Criterion.UNAMBIGUITY: "Однозначность",
    Criterion.COMPLETENESS: "Полнота",
    Criterion.VERIFIABILITY: "Проверяемость",
    Criterion.FEASIBILITY: "Реализуемость",
    Criterion.NECESSITY: "Необходимость",
    Criterion.TRACEABILITY: "Прослеживаемость",
    Criterion.ATOMICITY: "Атомарность",
    Criterion.DESIGN_INDEPENDENCE: "Независимость от решения",
    Criterion.CONSISTENCY: "Непротиворечивость",
}


class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNCERTAIN = "UNCERTAIN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"


SEVERITY_ORDER = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3, Severity.NONE: 4}


class Certainty(str, Enum):
    CONFIRMED = "CONFIRMED"   # подтверждённый дефект (есть проверенная цитата)
    POTENTIAL = "POTENTIAL"   # потенциальный риск / недостаточно данных / цитата не подтверждена


class ReqType(str, Enum):
    FUNCTIONAL = "FUNCTIONAL"
    NON_FUNCTIONAL = "NON_FUNCTIONAL"
    API = "API"
    DATA = "DATA"
    INTEGRATION = "INTEGRATION"
    ACCEPTANCE = "ACCEPTANCE"
    PROCESS = "PROCESS"
    UI = "UI"
    TESTING = "TESTING"
    CONSTRAINT = "CONSTRAINT"
    UNCLASSIFIED = "UNCLASSIFIED"
    # Контекстные элементы: в таблице извлечения есть, индивидуально не анализируются
    GOAL = "GOAL"
    ARCHITECTURE = "ARCHITECTURE"
    METADATA = "METADATA"
    TEMPLATE = "TEMPLATE"
    CONTEXT = "CONTEXT"


CONTEXT_TYPES = {ReqType.GOAL, ReqType.ARCHITECTURE, ReqType.METADATA, ReqType.TEMPLATE, ReqType.CONTEXT}

REQ_TYPE_RU: dict[ReqType, str] = {
    ReqType.FUNCTIONAL: "Функциональное",
    ReqType.NON_FUNCTIONAL: "Нефункциональное",
    ReqType.API: "API / контракт",
    ReqType.DATA: "Данные",
    ReqType.INTEGRATION: "Интеграция",
    ReqType.ACCEPTANCE: "Критерий приёмки",
    ReqType.PROCESS: "Процесс / взаимодействие",
    ReqType.UI: "UI",
    ReqType.TESTING: "Тестирование",
    ReqType.CONSTRAINT: "Ограничение",
    ReqType.UNCLASSIFIED: "Не классифицировано",
    ReqType.GOAL: "Цель (контекст)",
    ReqType.ARCHITECTURE: "Архитектура (контекст)",
    ReqType.METADATA: "Метаданные (контекст)",
    ReqType.TEMPLATE: "Шаблон / чек-лист (контекст)",
    ReqType.CONTEXT: "Пояснение (контекст)",
}


class IssueCategory(str, Enum):
    CONTRADICTION = "CONTRADICTION"
    LOGICAL_INCONSISTENCY = "LOGICAL_INCONSISTENCY"
    DUPLICATE = "DUPLICATE"
    CONFLICTING_VALUES = "CONFLICTING_VALUES"
    TERMINOLOGY = "TERMINOLOGY"
    MISSING_DEPENDENCY = "MISSING_DEPENDENCY"
    MISSING_SCENARIO = "MISSING_SCENARIO"
    FUNC_NONFUNC_CONFLICT = "FUNC_NONFUNC_CONFLICT"
    CROSS_DOCUMENT = "CROSS_DOCUMENT"
    ACCEPTANCE_MISMATCH = "ACCEPTANCE_MISMATCH"
    ARITHMETIC = "ARITHMETIC"
    OTHER = "OTHER"


ISSUE_CATEGORY_RU: dict[IssueCategory, str] = {
    IssueCategory.CONTRADICTION: "Прямое противоречие",
    IssueCategory.LOGICAL_INCONSISTENCY: "Логическая несогласованность",
    IssueCategory.DUPLICATE: "Дублирование",
    IssueCategory.CONFLICTING_VALUES: "Конфликт числовых значений / типов",
    IssueCategory.TERMINOLOGY: "Непоследовательная терминология",
    IssueCategory.MISSING_DEPENDENCY: "Отсутствующая зависимость",
    IssueCategory.MISSING_SCENARIO: "Отсутствующий сценарий",
    IssueCategory.FUNC_NONFUNC_CONFLICT: "Конфликт функциональных и нефункциональных требований",
    IssueCategory.CROSS_DOCUMENT: "Несогласованность основной и дочерней спецификаций",
    IssueCategory.ACCEPTANCE_MISMATCH: "Несоответствие требований и критериев приёмки",
    IssueCategory.ARITHMETIC: "Арифметическая ошибка",
    IssueCategory.OTHER: "Прочее",
}

# Категории, для подтверждения которых нужны цитаты из двух мест документа
TWO_SOURCE_CATEGORIES = {
    IssueCategory.CONTRADICTION, IssueCategory.CONFLICTING_VALUES, IssueCategory.DUPLICATE,
    IssueCategory.CROSS_DOCUMENT, IssueCategory.ACCEPTANCE_MISMATCH, IssueCategory.FUNC_NONFUNC_CONFLICT,
    IssueCategory.LOGICAL_INCONSISTENCY,
}


class CompletenessArea(str, Enum):
    ACTORS = "ACTORS"
    PRECONDITIONS = "PRECONDITIONS"
    MAIN_SCENARIOS = "MAIN_SCENARIOS"
    ALTERNATIVE_SCENARIOS = "ALTERNATIVE_SCENARIOS"
    ERROR_SCENARIOS = "ERROR_SCENARIOS"
    INPUT_OUTPUT_DATA = "INPUT_OUTPUT_DATA"
    BUSINESS_RULES = "BUSINESS_RULES"
    BOUNDARY_CONDITIONS = "BOUNDARY_CONDITIONS"
    EXTERNAL_INTEGRATIONS = "EXTERNAL_INTEGRATIONS"
    SECURITY = "SECURITY"
    PERFORMANCE = "PERFORMANCE"
    DATA_RETENTION = "DATA_RETENTION"
    ACCEPTANCE_CRITERIA = "ACCEPTANCE_CRITERIA"
    COMPONENT_DEPENDENCIES = "COMPONENT_DEPENDENCIES"


COMPLETENESS_AREA_RU: dict[CompletenessArea, str] = {
    CompletenessArea.ACTORS: "Акторы и роли",
    CompletenessArea.PRECONDITIONS: "Предусловия",
    CompletenessArea.MAIN_SCENARIOS: "Основные сценарии",
    CompletenessArea.ALTERNATIVE_SCENARIOS: "Альтернативные сценарии",
    CompletenessArea.ERROR_SCENARIOS: "Сценарии ошибок",
    CompletenessArea.INPUT_OUTPUT_DATA: "Входные и выходные данные",
    CompletenessArea.BUSINESS_RULES: "Бизнес-правила",
    CompletenessArea.BOUNDARY_CONDITIONS: "Граничные условия",
    CompletenessArea.EXTERNAL_INTEGRATIONS: "Внешние интеграции",
    CompletenessArea.SECURITY: "Требования безопасности",
    CompletenessArea.PERFORMANCE: "Требования производительности",
    CompletenessArea.DATA_RETENTION: "Хранение данных и жизненный цикл",
    CompletenessArea.ACCEPTANCE_CRITERIA: "Критерии приёмки",
    CompletenessArea.COMPONENT_DEPENDENCIES: "Зависимости между компонентами",
}


class CompletenessStatus(str, Enum):
    COVERED = "COVERED"
    CONFIRMED_OMISSION = "CONFIRMED_OMISSION"
    POTENTIAL_OMISSION = "POTENTIAL_OMISSION"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class Verdict(str, Enum):
    READY = "READY"
    READY_WITH_RESERVATIONS = "READY_WITH_RESERVATIONS"
    NOT_READY = "NOT_READY"


class RunStatus(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


# --------------------------------------------------------------------------- #
# Документ и извлечённые требования
# --------------------------------------------------------------------------- #
class Heading(BaseModel):
    line: int
    level: int
    title: str


class ParsedDocument(BaseModel):
    path: str
    name: str
    fmt: str
    sha256: str = ""
    lines: list[str]
    headings: list[Heading] = Field(default_factory=list)
    page_of_line: dict[int, int] = Field(default_factory=dict)  # только для PDF

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    def section_path(self, line: int) -> str:
        """Путь заголовков для строки (1-based): «Часть A › Цели»."""
        stack: list[Heading] = []
        for h in self.headings:
            if h.line > line:
                break
            while stack and stack[-1].level >= h.level:
                stack.pop()
            stack.append(h)
        if not stack:
            return "(без раздела)"
        top = min(h.level for h in self.headings)
        sole_title = sum(1 for h in self.headings if h.level == top) == 1 and self.headings[0].level == top
        if sole_title and len(stack) > 1 and stack[0].level == top:
            stack = stack[1:]          # единственный заголовок-название документа не повторяем в каждом пути
        return " › ".join(h.title for h in stack)

    def location(self, line_start: int, line_end: int) -> str:
        span = f"строка {line_start}" if line_start == line_end else f"строки {line_start}–{line_end}"
        page = self.page_of_line.get(line_start)
        if page:
            span = f"стр. {page}, {span}"
        return span


class Requirement(BaseModel):
    report_id: str              # REQ-001 ...
    doc_id: str | None = None   # идентификатор из документа (Ф-1), если был
    type: ReqType
    text: str                   # точная цитата из документа
    short: str
    section: str
    line_start: int
    line_end: int
    context: str = ""           # напр. заголовок таблицы для строки таблицы
    classified_by: str = "heuristic"

    @property
    def analyzable(self) -> bool:
        return self.type not in CONTEXT_TYPES


# --------------------------------------------------------------------------- #
# Ответы LLM (структурированный вывод)
# --------------------------------------------------------------------------- #
class UnitClassLLM(BaseModel):
    unit_id: str
    type: ReqType


class ClassificationResponse(BaseModel):
    items: list[UnitClassLLM]


class AssessmentLLM(BaseModel):
    criterion: Criterion
    status: Status
    severity: Severity = Severity.NONE
    explanation: str
    evidence: list[str] = Field(default_factory=list)
    suggestion: str = ""
    missing_field: str = ""


class IndividualResponse(BaseModel):
    assessments: list[AssessmentLLM]

    @model_validator(mode="after")
    def _all_criteria_once(self) -> "IndividualResponse":
        seen = [a.criterion for a in self.assessments]
        missing = [c.value for c in Criterion if c not in seen]
        dupes = sorted({c.value for c in seen if seen.count(c) > 1})
        if missing or dupes:
            raise ValueError(f"Ожидалась ровно одна оценка на каждый из 10 критериев; нет: {missing}; дубли: {dupes}")
        return self


class IndividualBatchItem(BaseModel):
    req_id: str
    response: IndividualResponse


class IndividualBatchResponse(BaseModel):
    items: list[IndividualBatchItem]


class EvidenceLLM(BaseModel):
    ref: str      # ID требования (REQ-xxx или ID из документа) либо название раздела
    quote: str    # дословная цитата


class CrossIssueLLM(BaseModel):
    category: IssueCategory
    severity: Severity
    confirmed: bool            # True — подтверждённый дефект, False — потенциальный (зависит от допущений)
    related_ids: list[str] = Field(default_factory=list)
    related_sections: list[str] = Field(default_factory=list)
    explanation: str
    evidence: list[EvidenceLLM] = Field(default_factory=list)
    recommendation: str


class CrossResponse(BaseModel):
    issues: list[CrossIssueLLM]


class CompletenessItemLLM(BaseModel):
    area: CompletenessArea
    status: CompletenessStatus
    severity: Severity = Severity.NONE
    explanation: str
    evidence: list[EvidenceLLM] = Field(default_factory=list)
    recommendation: str = ""


class CompletenessResponse(BaseModel):
    items: list[CompletenessItemLLM]

    @model_validator(mode="after")
    def _all_areas_once(self) -> "CompletenessResponse":
        seen = [i.area for i in self.items]
        missing = [a.value for a in CompletenessArea if a not in seen]
        dupes = sorted({a.value for a in seen if seen.count(a) > 1})
        if missing or dupes:
            raise ValueError(f"Нужна ровно одна запись на каждую область полноты; нет: {missing}; дубли: {dupes}")
        return self


class OmissionDecision(str, Enum):
    SUPPORTED = "SUPPORTED"
    COVERED = "COVERED"
    UNPROVEN = "UNPROVEN"


class OmissionReviewItem(BaseModel):
    area: CompletenessArea
    decision: OmissionDecision
    explanation: str
    evidence: list[EvidenceLLM] = Field(default_factory=list)


class OmissionReviewResponse(BaseModel):
    items: list[OmissionReviewItem]


class LocalOmissionReviewItem(BaseModel):
    req_id: str
    decision: OmissionDecision
    explanation: str
    evidence: list[EvidenceLLM] = Field(default_factory=list)


class LocalOmissionReviewResponse(BaseModel):
    items: list[LocalOmissionReviewItem]


class FinalResponse(BaseModel):
    verdict: Verdict
    justification: str
    priority_finding_ids: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Проверенные результаты
# --------------------------------------------------------------------------- #
class EvidenceRef(BaseModel):
    quote: str
    ref: str = ""
    verified: bool = False
    line_start: int | None = None
    line_end: int | None = None
    location: str = ""


class Assessment(BaseModel):
    criterion: Criterion
    status: Status
    severity: Severity = Severity.NONE
    explanation: str
    evidence: list[EvidenceRef] = Field(default_factory=list)
    suggestion: str = ""
    missing_field: str = ""
    certainty: Certainty | None = None   # только для FAIL / UNCERTAIN
    notes: list[str] = Field(default_factory=list)


class IndividualAnalysis(BaseModel):
    req_id: str
    ok: bool = True
    error: str = ""
    assessments: list[Assessment] = Field(default_factory=list)


class CrossIssue(BaseModel):
    issue_id: str
    category: IssueCategory
    severity: Severity
    certainty: Certainty
    related_ids: list[str] = Field(default_factory=list)
    related_sections: list[str] = Field(default_factory=list)
    explanation: str
    evidence: list[EvidenceRef] = Field(default_factory=list)
    recommendation: str
    notes: list[str] = Field(default_factory=list)


class CompletenessFinding(BaseModel):
    area: CompletenessArea
    status: CompletenessStatus
    severity: Severity = Severity.NONE
    certainty: Certainty | None = None
    explanation: str
    evidence: list[EvidenceRef] = Field(default_factory=list)
    recommendation: str = ""
    notes: list[str] = Field(default_factory=list)


class Finding(BaseModel):
    """Единая запись о проблеме без двойного счёта (для сводки и приоритетов)."""
    finding_id: str
    source: str                 # INDIVIDUAL / CROSS / COMPLETENESS
    severity: Severity
    certainty: Certainty
    title: str
    related_ids: list[str] = Field(default_factory=list)
    criteria: list[str] = Field(default_factory=list)
    cross_issue_ids: list[str] = Field(default_factory=list)
    missing_field: str = ""
    explanation: str = ""
    recommendation: str = ""
    evidence: list[EvidenceRef] = Field(default_factory=list)


class FinalAssessment(BaseModel):
    verdict: Verdict
    justification: str
    priority_finding_ids: list[str] = Field(default_factory=list)
    llm_verdict: Verdict | None = None
    adjusted: bool = False
    complete: bool = True


class AnalysisResult(BaseModel):
    document_name: str
    document_sha256: str = ""
    started_at: str
    model: str
    agent_version: str
    language: str = "ru"
    status: RunStatus = RunStatus.COMPLETE
    llm_calls: int = 0
    requirements: list[Requirement] = Field(default_factory=list)
    individual: list[IndividualAnalysis] = Field(default_factory=list)
    cross_issues: list[CrossIssue] = Field(default_factory=list)
    cross_complete: bool = True
    completeness: list[CompletenessFinding] = Field(default_factory=list)
    completeness_complete: bool = True
    findings: list[Finding] = Field(default_factory=list)
    duplicates_merged: int = 0
    final: FinalAssessment | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
