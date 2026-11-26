"""Индивидуальная проверка одного требования (отдельный вызов LLM на каждое)."""
from __future__ import annotations

import logging

from src.analysis import prompts
from src.analysis.llm_client import LLMClient, LLMError
from src.models.schemas import IndividualAnalysis, IndividualResponse, ParsedDocument, Requirement
from src.validation.validator import DocIndex, validate_individual

log = logging.getLogger(__name__)


def build_context(doc: ParsedDocument, req: Requirement, max_chars: int = prompts.MAX_CONTEXT_CHARS) -> tuple[str, bool]:
    """Весь документ, если помещается; иначе оглавление + раздел требования. Возвращает (текст, это_весь_документ)."""
    full = doc.text
    if len(full) <= max_chars:
        return full, True
    outline = "\n".join(f"{'  ' * (h.level - 1)}- {h.title} (строка {h.line})" for h in doc.headings)
    starts = [h.line for h in doc.headings if h.line <= req.line_start]
    nexts = [h.line for h in doc.headings if h.line > req.line_start]
    first = starts[-1] if starts else 1
    last = (nexts[0] - 1) if nexts else len(doc.lines)
    section = "\n".join(doc.lines[first - 1:last])
    text = f"[ОГЛАВЛЕНИЕ]\n{outline}\n\n[РАЗДЕЛ ТРЕБОВАНИЯ, строки {first}-{last}]\n{section}"
    return text[:max_chars], False


def analyze_requirement(client: LLMClient, doc: ParsedDocument, index: DocIndex, req: Requirement,
                        language: str = "ru") -> IndividualAnalysis:
    context, is_full = build_context(doc, req)
    try:
        resp = client.generate(
            system=prompts.individual_system(language),
            prompt=prompts.individual_prompt(req, doc.name, context, is_full),
            schema=IndividualResponse,
        )
    except LLMError as exc:
        log.error("%s: анализ не выполнен: %s", req.report_id, exc)
        return IndividualAnalysis(req_id=req.report_id, ok=False, error=str(exc))
    return validate_individual(resp, req, index)
