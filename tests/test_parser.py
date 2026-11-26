import pymupdf
import pytest

from src.ingestion.extractor import extract_requirements
from src.ingestion.parser import DocumentError, EmptyDocumentError, UnsupportedFormatError, parse_file


def test_markdown_headings_and_lines(sample_md):
    doc = parse_file(sample_md)
    assert doc.fmt == "md" and doc.sha256
    titles = [h.title for h in doc.headings]
    assert titles[:3] == ["Тестовая спецификация", "Функциональные требования", "Нефункциональные требования"]
    line_fr1 = next(i for i, l in enumerate(doc.lines, 1) if "FR-1" in l)
    assert doc.section_path(line_fr1) == "Функциональные требования"   # единственный H1 (название) опущен


def test_markdown_ignores_hash_inside_code_fence(tmp_path):
    p = tmp_path / "a.md"
    p.write_text("# Заголовок\n```\n# не заголовок\n```\nТекст\n", encoding="utf-8")
    assert [h.title for h in parse_file(p).headings] == ["Заголовок"]


def test_txt_parsing_numbered_headings(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("1.1 Общие положения\nСистема должна работать.\n\n1.2 Ограничения\nНе более 5 с.\n", encoding="utf-8")
    doc = parse_file(p)
    assert doc.fmt == "txt"
    assert [h.title for h in doc.headings] == ["1.1 Общие положения", "1.2 Ограничения"]
    assert doc.lines[1] == "Система должна работать."


def test_cp1251_text_is_decoded(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes("Требование должно быть проверяемым.".encode("cp1251"))
    assert "Требование" in parse_file(p).text


def test_pdf_parsing_keeps_pages_and_headings(tmp_path):
    p = tmp_path / "a.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 80), "Functional requirements", fontsize=20)
    page.insert_text((72, 120), "The system shall store the limit.", fontsize=10)
    page.insert_text((72, 140), "The system shall notify the user.", fontsize=10)
    pdf.save(p)
    doc = parse_file(p)
    assert doc.fmt == "pdf" and doc.page_of_line[1] == 1
    assert any("Functional requirements" == h.title for h in doc.headings)
    assert "стр. 1" in doc.location(2, 2)


def test_adjacent_numbered_pdf_requirements_stay_separate(tmp_path):
    path = tmp_path / "spec.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 80), "Functional requirements", fontsize=20)
        page.insert_text((72, 108), "FR-01. The service records each booking.", fontsize=10)
        page.insert_text((72, 122), "It returns the booking number.", fontsize=10)
        page.insert_text((72, 136), "FR-02. The service shows the room.", fontsize=10)
        pdf.save(path)
    items, _ = extract_requirements(parse_file(path))
    assert [r.doc_id for r in items] == ["FR-01", "FR-02"]
    assert "It returns" in items[0].text and "FR-02" not in items[0].text


def test_empty_file_rejected(tmp_path):
    p = tmp_path / "empty.md"
    p.write_text("  \n\n", encoding="utf-8")
    with pytest.raises(EmptyDocumentError):
        parse_file(p)


def test_unsupported_format_rejected(tmp_path):
    p = tmp_path / "a.xyz"
    p.write_text("text", encoding="utf-8")
    with pytest.raises(UnsupportedFormatError):
        parse_file(p)


def test_missing_file_rejected(tmp_path):
    with pytest.raises(DocumentError):
        parse_file(tmp_path / "nope.md")
