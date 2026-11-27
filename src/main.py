"""CLI: python -m src.main --input requirements.md --output reports/requirements_report.md

Коды возврата: 0 — анализ завершён полностью; 1 — анализ не выполнен;
2 — ошибка входных данных/конфигурации; 3 — отчёт создан, но анализ завершён частично.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.analysis.analyzer import run_analysis
from src.analysis.llm_client import GeminiClient, LLMError
from src.config import ConfigError, get_agent_version, get_settings
from src.ingestion.parser import DocumentError, parse_file
from src.models.schemas import Certainty, RunStatus
from src.reporting.report_generator import write_report

EXIT_OK, EXIT_FAILED, EXIT_INPUT, EXIT_PARTIAL = 0, 1, 2, 3


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.main",
                                description="Агент статической проверки качества требований (LLM: Gemini).")
    p.add_argument("--input", "-i", required=True, help="путь к документу требований (.md, .txt, .pdf)")
    p.add_argument("--output", "-o", default="reports/requirements_report.md", help="куда записать отчёт (Markdown)")
    p.add_argument("--model", help="имя модели Gemini (иначе GEMINI_MODEL из окружения)")
    p.add_argument("--language", default="ru", choices=["ru", "en"],
                   help="язык пояснений LLM; каркас отчёта всегда русский (по умолчанию ru)")
    p.add_argument("--rpm", type=int, help="лимит запросов в минуту (иначе GEMINI_REQUESTS_PER_MINUTE)")
    p.add_argument("--batch-size", type=int, choices=range(1, 9), default=4,
                   help="требований в одном запросе для коротких документов (по умолчанию 4)")
    p.add_argument("--verbose", "-v", action="store_true", help="подробный лог")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    log = logging.getLogger("agent")

    try:
        doc = parse_file(args.input)
    except DocumentError as exc:
        log.error("%s", exc)
        return EXIT_INPUT
    log.info("Документ прочитан: %s (%s, строк: %d, заголовков: %d)", doc.name, doc.fmt, len(doc.lines), len(doc.headings))

    try:
        settings = get_settings(args.model, args.rpm)
    except ConfigError as exc:
        log.error("%s", exc)
        return EXIT_INPUT

    client = GeminiClient(settings.api_key, settings.model, settings.requests_per_minute)
    try:
        result = run_analysis(doc, client, get_agent_version(), args.language,
                              progress=lambda m: log.info("%s", m), individual_batch_size=args.batch_size)
    except LLMError as exc:
        log.error("Анализ прерван: %s", exc)
        return EXIT_FAILED

    if result.status == RunStatus.FAILED:
        log.error("Анализ не выполнен: %s", "; ".join(result.errors[:3]))
        return EXIT_FAILED
    path = write_report(result, args.output)
    _print_summary(result, path)
    return EXIT_OK if result.status == RunStatus.COMPLETE else EXIT_PARTIAL


def _print_summary(result, path: Path) -> None:
    conf = sum(1 for f in result.findings if f.certainty == Certainty.CONFIRMED)
    pot = len(result.findings) - conf
    ok = sum(1 for a in result.individual if a.ok)
    print("\n=== Краткая сводка ===")
    print(f"Документ:            {result.document_name}")
    print(f"Модель:              {result.model} (обращений к LLM: {result.llm_calls})")
    print(f"Элементов извлечено: {len(result.requirements)}; проанализировано индивидуально: {ok}")
    print(f"Замечаний:           подтверждённых {conf}, потенциальных {pot}")
    print(f"Статус анализа:      {result.status.value}")
    if result.final:
        print(f"Вердикт:             {result.final.verdict.value}")
    print(f"Отчёт:               {path}")


if __name__ == "__main__":
    sys.exit(main())
