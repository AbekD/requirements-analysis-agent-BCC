from src.ingestion.extractor import (
    assign_report_ids, classify_with_llm, extract_requirements, heuristic_type, merge_classification, split_into_units,
)
from src.ingestion.parser import parse_file
from src.models.schemas import ClassificationResponse, ReqType, UnitClassLLM
from tests.conftest import FakeClient


def _units(sample_md):
    return assign_report_ids(split_into_units(parse_file(sample_md)))


def test_original_ids_are_preserved(sample_md):
    units = _units(sample_md)
    ids = {u.doc_id for u in units if u.doc_id}
    assert ids == {"FR-1", "FR-2"}


def test_report_ids_generated_for_all_units(sample_md):
    units = _units(sample_md)
    assert [u.report_id for u in units][:3] == ["REQ-001", "REQ-002", "REQ-003"]
    assert len({u.report_id for u in units}) == len(units)


def test_ids_generated_when_document_has_no_ids(tmp_path):
    p = tmp_path / "noid.md"
    p.write_text("# Док\n## Требования\nСистема должна сохранять данные.\n\nСистема должна логировать вход.\n",
                 encoding="utf-8")
    units = assign_report_ids(split_into_units(parse_file(p)))
    assert [(u.report_id, u.doc_id) for u in units] == [("REQ-001", None), ("REQ-002", None)]


def test_quotes_are_exact_substrings_with_line_ranges(sample_md):
    doc = parse_file(sample_md)
    for u in assign_report_ids(split_into_units(doc)):
        for fragment in u.text.split("\n"):
            assert fragment in "\n".join(doc.lines[u.line_start - 1:u.line_end]), u.report_id


def test_table_rows_and_code_blocks_become_units(sample_md):
    units = _units(sample_md)
    row = next(u for u in units if "Лимит 1 000 ₸" in u.text)
    assert row.type == ReqType.ACCEPTANCE and "Условие" in row.context      # заголовок таблицы не теряется
    code = next(u for u in units if u.text.startswith('{"result"'))
    assert code.type == ReqType.API
    assert not any("---" in u.text for u in units)                           # разделитель таблицы пропущен


def test_heuristic_types_by_section(sample_md):
    doc = parse_file(sample_md)
    fr = next(i for i, l in enumerate(doc.lines, 1) if "FR-1" in l)
    nfr = next(i for i, l in enumerate(doc.lines, 1) if "2 секунды" in l)
    assert heuristic_type(doc, fr) == ReqType.FUNCTIONAL
    assert heuristic_type(doc, nfr) == ReqType.NON_FUNCTIONAL   # «нефункциональные» не принимается за «функциональные»


def test_llm_cannot_demote_explicit_requirements(sample_md):
    unit = next(u for u in _units(sample_md) if u.doc_id == "FR-1")
    assert merge_classification(unit, ReqType.CONTEXT) == ReqType.FUNCTIONAL
    assert merge_classification(unit, ReqType.CONSTRAINT) == ReqType.CONSTRAINT


def test_extract_with_llm_and_with_llm_failure(sample_md):
    doc = parse_file(sample_md)
    units, warnings = extract_requirements(doc, FakeClient())
    assert units and not warnings and all(u.classified_by == "heuristic+llm" for u in units)
    units2, warnings2 = extract_requirements(doc, FakeClient(fail_on=(ClassificationResponse,)))
    assert len(units2) == len(units) and warnings2 and all(u.classified_by == "heuristic" for u in units2)


def test_classification_rejects_duplicate_and_out_of_batch_ids(sample_md):
    first, second = _units(sample_md)[:2]

    class DuplicateClient:
        def generate(self, *, system, prompt, schema):
            assert schema is ClassificationResponse
            return schema(items=[
                UnitClassLLM(unit_id=first.report_id, type=ReqType.CONSTRAINT),
                UnitClassLLM(unit_id=first.report_id, type=ReqType.PROCESS),
                UnitClassLLM(unit_id="REQ-999", type=ReqType.PROCESS),
            ])

    original = first.type, second.type
    warnings = classify_with_llm([first, second], DuplicateClient(), batch_size=1)
    assert (first.type, second.type) == original
    assert first.classified_by == second.classified_by == "heuristic"
    assert any("REQ-999" in warning for warning in warnings)
