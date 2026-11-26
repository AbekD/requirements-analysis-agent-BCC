"""Проверки явно сформулированных правил и примеров приёмки."""
from __future__ import annotations

import json
import re

from src.analysis.arithmetic import checked_sum
from src.models.schemas import CrossIssueLLM, EvidenceLLM, IssueCategory, ReqType, Requirement, Severity

_MONTHS = (
    ("января", "январский"), ("февраля", "февральский"), ("марта", "мартовский"),
    ("апреля", "апрельский"), ("мая", "майский"), ("июня", "июньский"),
    ("июля", "июльский"), ("августа", "августовский"), ("сентября", "сентябрьский"),
    ("октября", "октябрьский"), ("ноября", "ноябрьский"), ("декабря", "декабрьский"),
)
_DATE = r"\d{1,2}\s+(" + "|".join(name for name, _ in _MONTHS) + r")\b"
_PURCHASE = re.compile(r"^покупка\s+" + _DATE, re.I)
_PROCESS = re.compile(r"^провести\s+" + _DATE, re.I)
_COUNTER = re.compile(r"^(" + "|".join(adjective for _, adjective in _MONTHS) + r")\s+сч[её]тчик\b", re.I)
_PURCHASE_MONTH_RULE = re.compile(
    r"(?:^|[.!?]\**\s+)месяц определяется по дате совершения покупки(?P<zone>\s+в\s+[^.]+)?\.", re.I,
)
_MONTHLY_PUSH_CAP = re.compile(
    r"на\s+карту\s+приходится\s+(?:максимум|не более)\s+(?:один|1)\s+(?:такой\s+)?push"
    r"\s+за\s+календарный\s+месяц\b", re.I,
)
_REPEAT_PUSH = re.compile(
    r"при\s+изменении\s+порога[^.]*\.[^.]*?ниже\s+нового\s+порога\s*,"
    r"\s*следующее\s+достижение\s+вызывает\s+push\s+независимо\s+от\s+истории\b", re.I,
)
_HTTP_METHOD = re.compile(r"^\|\s*\**(?:rest type|http method|метод http)\**\s*\|\s*(\w+)\s*\|", re.I)
_WRITE_STEP = re.compile(
    r"^(?:заменить|сохранить|обновить)\s+(?:(?:действующую|текущую|новую)\s+)?настройк\w*\b|"
    r"^(?:replace|save|update)\s+(?:(?:the|current|existing|new)\s+){0,2}settings?\b", re.I,
)
_BAD_PARAMETERS = re.compile(
    r"^\**400\.\**\s+[^\n]*(?:параметры запроса заполнены неверно|invalid request parameters)", re.I,
)
_AMOUNT_ERROR = re.compile(r"проверьте сумму|неверн\w* сумм\w*|invalid amount|check (?:the )?amount", re.I)


def check_explicit_rules(requirements: list[Requirement], language: str) -> list[CrossIssueLLM]:
    findings = []
    push_limits = [req for req in requirements if _MONTHLY_PUSH_CAP.search(req.text)]
    push_resets = [req for req in requirements if _REPEAT_PUSH.search(req.text)]
    for limit in push_limits:
        for reset in push_resets:
            if limit.report_id == reset.report_id or limit.section != reset.section:
                continue
            findings.append(_issue(
                IssueCategory.CONTRADICTION, [limit, reset],
                ("A card has already received a notification this calendar month. If the threshold is raised above "
                 "current spending and crossed again, the reset rule sends a second push that the monthly limit forbids."
                 if language == "en" else "Карта уже получила push в текущем месяце. Если повысить порог выше "
                 "накопленных расходов и снова его достичь, правило изменения порога требует второй push, "
                 "а месячный лимит его запрещает."),
                ("Define whether the monthly limit survives a threshold change and add an acceptance case."
                 if language == "en" else "Уточнить, сохраняется ли месячный лимит после смены порога, "
                 "и описать этот сценарий приёмки."),
            ))

    methods: dict[str, list[tuple[Requirement, str]]] = {}
    for req in requirements:
        match = _HTTP_METHOD.match(req.text)
        if match:
            methods.setdefault(req.section.rpartition(" › ")[0], []).append((req, match[1].upper()))
    for scope, declarations in methods.items():
        if not scope or len(declarations) != 1 or declarations[0][1] != "GET":
            continue
        declaration = declarations[0][0]
        write = next((req for req in requirements if req.section.rpartition(" › ")[0] == scope
                      and _WRITE_STEP.match(req.text.strip())), None)
        if write:
            findings.append(_issue(
                IssueCategory.CROSS_DOCUMENT, [declaration, write],
                ("The contract declares GET while the operation explicitly writes settings."
                 if language == "en" else "Метод объявлен как GET, а его логика прямо предписывает "
                 "изменение сохранённой настройки."),
                ("Align the method and the state-changing behavior in the API contract."
                 if language == "en" else "Согласовать HTTP-метод и изменение состояния в контракте API."),
            ))
    for req in requirements:
        parent, _, status = req.section.rpartition(" › ")
        if not parent or not re.match(r"^200\b", status) or not req.text.strip().startswith("{"):
            continue
        try:
            payload = json.loads(req.text)
        except json.JSONDecodeError:
            continue
        if (not isinstance(payload, dict) or payload.get("result") != "ERROR"
                or not _AMOUNT_ERROR.search(str(payload.get("message", "")))):
            continue
        rule = next((candidate for candidate in requirements
                     if candidate.section == parent and _BAD_PARAMETERS.match(candidate.text)), None)
        if rule:
            findings.append(_issue(
                IssueCategory.CONFLICTING_VALUES, [rule, req],
                ("Invalid request parameters must return HTTP 400, but the example returns an amount validation "
                 "error under HTTP 200." if language == "en" else "По правилу неверные параметры запроса "
                 "возвращают HTTP 400, а пример ошибки проверки суммы помещён в ответы HTTP 200."),
                ("Specify one status for invalid amounts and update the example accordingly."
                 if language == "en" else "Указать единый статус для неверной суммы и согласовать с ним пример ответа."),
            ))
    return findings


def check_acceptance(requirements: list[Requirement], language: str) -> list[CrossIssueLLM]:
    findings = []
    month_rules = [(req, match) for req in requirements if req.type == ReqType.FUNCTIONAL
                   if (match := _PURCHASE_MONTH_RULE.search(req.text))]
    counter_months = {adjective: month for month, adjective in _MONTHS}
    for row in requirements:
        if row.type != ReqType.ACCEPTANCE or not row.text.startswith("|"):
            continue
        amounts = checked_sum(row)
        if amounts is not None:
            start, purchase, total = amounts
            if start + purchase != total:
                equation = f"{start} + {purchase} = {start + purchase}"
                findings.append(_issue(
                    IssueCategory.ARITHMETIC, [row],
                    (f"The stated total is {total}, but {equation}." if language == "en"
                     else f"В примере указан итог {total}, хотя {equation}."),
                    ("Correct the expected total." if language == "en" else "Исправить ожидаемую итоговую сумму."),
                ))
        cells = [cell.strip() for cell in row.text.strip("|").split("|")]
        if len(cells) != 3:
            continue
        purchase_date, processing_date, counter = (
            _PURCHASE.match(cells[0]), _PROCESS.match(cells[1]), _COUNTER.match(cells[2]),
        )
        if not (purchase_date and processing_date and counter):
            continue
        purchase_month, processing_month = purchase_date[1].lower(), processing_date[1].lower()
        if purchase_month == processing_month or counter_months[counter[1].lower()] != processing_month:
            continue
        for rule, match in month_rules:
            zone = (match["zone"] or "").casefold().strip().removeprefix("в ")
            if zone and not cells[1].casefold().rstrip(".").endswith(f"по {zone}"):
                continue
            findings.append(_issue(
                IssueCategory.ACCEPTANCE_MISMATCH, [rule, row],
                ("The rule uses the purchase date, but the example credits a different month based on the processing date."
                 if language == "en" else
                 f"Правило использует дату совершения покупки. По нему покупка от {purchase_date[0].split(maxsplit=1)[1]} "
                 f"относится к месяцу покупки, но пример увеличивает счётчик за другой месяц "
                 f"по дате проведения ({processing_date[0].split(maxsplit=1)[1]})."),
                ("Align the expected result with the agreed event date."
                 if language == "en" else "Согласовать, какая дата определяет месяц, и привести к ней пример приёмки."),
            ))
    return findings


def _issue(category: IssueCategory, requirements: list[Requirement], explanation: str,
           recommendation: str) -> CrossIssueLLM:
    return CrossIssueLLM(
        category=category, severity=Severity.HIGH, confirmed=True,
        related_ids=[req.report_id for req in requirements],
        evidence=[EvidenceLLM(ref=req.report_id, quote=req.text) for req in requirements],
        explanation=explanation, recommendation=recommendation,
    )
