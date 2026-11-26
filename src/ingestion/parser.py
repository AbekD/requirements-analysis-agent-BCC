"""Чтение входного документа: Markdown, TXT, PDF. Сохраняет заголовки и номера строк."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from statistics import median

from src.models.schemas import Heading, ParsedDocument

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf"}

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
# TXT: «1.2 Название» (многоуровневая нумерация) или короткая строка ЗАГЛАВНЫМИ
_TXT_NUMBERED = re.compile(r"^(\d+(?:\.\d+)+)[.)]?\s+\S.{0,80}$")


class DocumentError(RuntimeError):
    """Документ нельзя обработать (нет файла, формат, пустой, ошибка чтения)."""


class UnsupportedFormatError(DocumentError):
    pass


class EmptyDocumentError(DocumentError):
    pass


def parse_file(path: str | Path) -> ParsedDocument:
    p = Path(path)
    if not p.is_file():
        raise DocumentError(f"Файл не найден: {p}")
    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Неподдерживаемый формат «{ext or 'без расширения'}». Поддерживаются: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    raw = p.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if ext == ".pdf":
        doc = _parse_pdf(p)
    else:
        doc = _parse_text(raw, "md" if ext in {".md", ".markdown"} else "txt", p)
    doc.sha256 = sha
    if not any(line.strip() for line in doc.lines):
        raise EmptyDocumentError(f"Документ пуст: {p}")
    return doc


def _decode(raw: bytes, path: Path) -> str:
    for enc in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise DocumentError(f"Не удалось определить кодировку файла: {path}")


def _parse_text(raw: bytes, fmt: str, path: Path) -> ParsedDocument:
    text = _decode(raw, path)
    lines = text.splitlines()
    headings = _md_headings(lines) if fmt == "md" else _txt_headings(lines)
    return ParsedDocument(path=str(path), name=path.name, fmt=fmt, lines=lines, headings=headings)


def _md_headings(lines: list[str]) -> list[Heading]:
    headings: list[Heading] = []
    in_fence = False
    for no, line in enumerate(lines, start=1):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _MD_HEADING.match(line)
        if m:
            headings.append(Heading(line=no, level=len(m.group(1)), title=m.group(2).strip()))
    return headings


def _txt_headings(lines: list[str]) -> list[Heading]:
    headings: list[Heading] = []
    for no, line in enumerate(lines, start=1):
        s = line.strip()
        if not s:
            continue
        m = _TXT_NUMBERED.match(s)
        if m and not s.endswith((".", ";", ",")):
            headings.append(Heading(line=no, level=m.group(1).count(".") + 1, title=s))
        elif len(s) <= 80 and s.upper() == s and sum(c.isalpha() for c in s) >= 4:
            headings.append(Heading(line=no, level=1, title=s))
    return headings


def _parse_pdf(path: Path) -> ParsedDocument:
    try:
        import pymupdf  # PyMuPDF >= 1.24.3
    except ImportError:
        try:
            import fitz as pymupdf  # старые версии PyMuPDF
        except ImportError as exc:  # pragma: no cover
            raise DocumentError("Для PDF требуется пакет PyMuPDF (pip install PyMuPDF)") from exc
    try:
        pdf = pymupdf.open(path)
    except Exception as exc:  # noqa: BLE001 — PyMuPDF бросает разные типы
        raise DocumentError(f"Не удалось открыть PDF: {exc}") from exc

    lines: list[str] = []
    page_of_line: dict[int, int] = {}
    candidates: list[tuple[int, float, str]] = []  # (line_no, font_size, text)
    sizes: list[float] = []
    with pdf:
        for page_no, page in enumerate(pdf, start=1):
            for block in page.get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
                    if not spans:
                        continue
                    text = "".join(s["text"] for s in spans).rstrip()
                    size = max(s["size"] for s in spans)
                    lines.append(text)
                    no = len(lines)
                    page_of_line[no] = page_no
                    candidates.append((no, size, text.strip()))
                    sizes.append(size)

    headings: list[Heading] = []
    if sizes:
        body = median(sizes)
        big = sorted({round(s, 1) for _, s, _ in candidates if s > body * 1.15}, reverse=True)
        for no, size, text in candidates:
            if size > body * 1.15 and len(text) <= 120:
                headings.append(Heading(line=no, level=1 + big.index(round(size, 1)), title=text))
    return ParsedDocument(path=str(path), name=path.name, fmt="pdf", lines=lines,
                          headings=headings, page_of_line=page_of_line)
