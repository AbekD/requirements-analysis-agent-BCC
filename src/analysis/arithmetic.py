"""Проверяемые арифметические примеры в критериях приёмки."""
from __future__ import annotations

import re

from src.models.schemas import ReqType, Requirement

_INTEGER = r"(\d+(?:[ \u00a0\u202f]\d{3})*)\s*(?:₸|тенге\b|KZT\b)"
_START = re.compile(r"^(?:текущие\s+)?расходы\s*[:=]?\s*" + _INTEGER, re.IGNORECASE)
_PURCHASE = re.compile(r"^провести\s+покупку\s+на\s+" + _INTEGER + r"\s*\.?$", re.IGNORECASE)
_TOTAL = re.compile(r"^(?:итог|итого|общие расходы)\s*[:=]?\s*" + _INTEGER, re.IGNORECASE)
_ARITHMETIC = re.compile(
    r"арифмет|ошиб\w*\s+(?:в\s+)?(?:расч[её]т|сложен)|"
    r"(?:неверн|неправильн)\w*\s+(?:расч[её]т|сумм)|"
    r"сумм\w*\s+(?:рассчитан\w*\s+)?(?:неверно|неправильно)|"
    r"\barithmetic\b|\b(?:incorrect|wrong)\s+(?:calculation|sum)\b",
    re.IGNORECASE,
)
_OTHER_BEHAVIOUR = re.compile(
    r"(?:уведомлен|пуш|push|notification)\w*\s+(?:не\s+|долж|дублир|must|should)|"
    r"(?:не\s+долж\w*|should\s+not)\s+.{0,25}(?:уведомлен|пуш|push|notification)",
    re.IGNORECASE,
)


def checked_sum(req: Requirement) -> tuple[int, int, int] | None:
    """Читает узкий шаблон: расходы + проведённая покупка = итог в тенге."""
    if req.type != ReqType.ACCEPTANCE or not req.text.startswith("|"):
        return None
    cells = [cell.strip() for cell in req.text.strip("|").split("|")]
    if len(cells) != 3:
        return None
    matches = (_START.match(cells[0]), _PURCHASE.fullmatch(cells[1]), _TOTAL.match(cells[2]))
    if any(match is None for match in matches):
        return None
    amounts = [int(re.sub(r"\s+", "", match[1])) for match in matches if match is not None]
    return amounts[0], amounts[1], amounts[2]


def arithmetic_claim(explanation: str) -> bool:
    """Отделяет утверждение об ошибке вычисления от претензии к бизнес-результату."""
    return bool(_ARITHMETIC.search(explanation) and not _OTHER_BEHAVIOUR.search(explanation))


def arithmetic_context(requirements: list[Requirement]) -> str:
    lines = []
    for req in requirements:
        amounts = checked_sum(req)
        if amounts is not None:
            start, purchase, total = amounts
            lines.append(f"{req.report_id}: {start} + {purchase} = {start + purchase}; stated total = {total}.")
    return "\n".join(lines)
