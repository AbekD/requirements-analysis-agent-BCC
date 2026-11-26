"""Промпты для LLM. Инструкции на английском (точность), результат — на русском."""
from __future__ import annotations

from src.models.schemas import COMPLETENESS_AREA_RU, CompletenessArea, Requirement, ReqType

MAX_CONTEXT_CHARS = 24_000

_COMMON_RULES = """\
General rules (apply to every answer):
- This is STATIC REQUIREMENTS REVIEW (requirements engineering), not software testing.
- Base every statement ONLY on the provided document. Never invent context, facts, numbers or quotations.
- All explanations and recommendations must be written in {language_name}.
- Evidence quotations must be copied VERBATIM from the document, in the original language, exactly as written
  (you may omit markdown markers; you may join non-adjacent fragments with "..."). Never paraphrase inside a quote.
- Do not confuse MISSING information with a DEFECT. If the document does not give enough information to decide,
  use UNCERTAIN / potential and say what exactly is missing.
- Do not propose arbitrary implementation decisions. Recommendations must preserve the original business intent
  and say WHAT must be specified/clarified, not how to build it.
- Criteria are based on ISO/IEC/IEEE 29148:2018 (5.2.5, 5.2.6) and IEEE Std 830-1998 (4.3); no claim of formal compliance.
- Return ONLY JSON matching the provided schema.
"""

CRITERIA_DEFINITIONS = """\
Criteria for ONE requirement (assess ALL ten, exactly once each):
1. Correctness – the statement is factually/logically sound and does not contradict itself or evidently known domain facts.
2. Unambiguity – has a single interpretation; no subjective/vague terms ("quickly", "user-friendly", "etc."), undefined terms or pronouns.
3. Completeness – contains what is needed to implement it: actor, trigger/conditions, inputs, outputs/expected result, limits,
   error behaviour (where relevant to THIS requirement).
4. Verifiability – an objective test/inspection can decide if it is met (measurable parameters, concrete expected results).
5. Feasibility – can be realised within reasonable technical/business limits (flag only evident infeasibility; otherwise UNCERTAIN/PASS).
6. Necessity – it carries a real stakeholder need; not redundant or gold-plating (use UNCERTAIN if the need cannot be judged).
7. Traceability – has an identifier and/or a clear link to a business goal, source or related requirements.
8. Atomicity – states exactly one obligation/condition; flag requirements that bundle several independent obligations.
9. DesignIndependence – states WHAT, not HOW; flag unnecessary implementation details (technologies, protocols, storage choices)
   unless the document explicitly presents them as a constraint (e.g. an API contract is allowed to be specific).
10. Consistency – no contradiction inside the requirement and none with other parts of the document.

Statuses: PASS | FAIL | UNCERTAIN | NOT_APPLICABLE.
- FAIL only for a defect you can point to with a verbatim quote (put the quote in "evidence").
- UNCERTAIN when the document is insufficient to decide or the defect depends on assumptions.
- NOT_APPLICABLE when the criterion makes no sense for this kind of element (say why).
- PASS needs no evidence, but give a short reason.
Severity (for FAIL/UNCERTAIN; use NONE for PASS/NOT_APPLICABLE):
- CRITICAL: prevents unambiguous implementation or objective verification.
- HIGH: very likely to cause wrong implementation/testing or rework.
- MEDIUM: noticeable quality problem, workable with clarification.
- LOW: stylistic/minor risk.
"""


def language_name(code: str) -> str:
    return {"ru": "Russian", "en": "English"}.get(code, "Russian")


def individual_system(language: str) -> str:
    return (
        "You are a senior business/system analyst performing a static quality review of ONE requirement "
        "taken from a requirements document.\n\n"
        + _COMMON_RULES.format(language_name=language_name(language))
        + "\n" + CRITERIA_DEFINITIONS
        + "\nAssess ONLY the given requirement. The full document is provided as context so that you do not report "
          "information as missing when it is stated elsewhere. Cross-requirement conflicts are analysed separately; "
          "mention them under Consistency only when they directly involve the given requirement.\n"
    )


def individual_prompt(req: Requirement, doc_title: str, context_text: str, context_is_full: bool) -> str:
    ctx_label = "FULL DOCUMENT" if context_is_full else "DOCUMENT OUTLINE AND THE REQUIREMENT'S SECTION (document is large)"
    extra = f"\nTable header / context: {req.context}" if req.context else ""
    return (
        f"Document: {doc_title}\n\n"
        f"=== {ctx_label} (context only) ===\n{context_text}\n=== END OF CONTEXT ===\n\n"
        f"=== REQUIREMENT UNDER REVIEW ===\n"
        f"Report ID: {req.report_id}\nID in document: {req.doc_id or '(none)'}\n"
        f"Element type: {req.type.value}\nSection: {req.section}\n"
        f"Lines: {req.line_start}-{req.line_end}{extra}\n"
        f"Exact text:\n<<<\n{req.text}\n>>>\n\n"
        "Return the assessments of all ten criteria."
    )


# --------------------------------------------------------------------------- #
CLASSIFY_SYSTEM = (
    "You classify elements of a requirements document. For each element choose ONE type:\n"
    + "\n".join(f"- {t.value}" for t in ReqType)
    + "\n\nGuidance: FUNCTIONAL/NON_FUNCTIONAL/DATA/INTEGRATION/API/UI/PROCESS/TESTING/CONSTRAINT/ACCEPTANCE are reviewable "
      "requirement-like statements. GOAL, ARCHITECTURE, METADATA (passport fields, authors, repos), TEMPLATE (empty review "
      "checklists, signature blocks) and CONTEXT (labels, explanatory notes) are context elements that are not reviewed "
      "individually. Do not assume every sentence is a requirement, but when in doubt prefer a reviewable type over a "
      "context type so that nothing is lost. Return one item per element id, using the ids exactly as given."
)


def classify_prompt(units: list[Requirement]) -> str:
    rows = []
    for u in units:
        snippet = u.text.replace("\n", " ⏎ ")
        rows.append(f"[{u.report_id}] section: {u.section} | current type: {u.type.value} | text: {snippet[:400]}")
    return "Classify these elements:\n\n" + "\n".join(rows)


# --------------------------------------------------------------------------- #
CROSS_SYSTEM_TMPL = """\
You are a senior requirements reviewer analysing the WHOLE specification (all parts and any child/API specification) for
relationships BETWEEN requirements. This is static review, not testing.

{rules}
Look for these issue categories (use the enum names):
- CONTRADICTION: two statements that cannot both be true.
- LOGICAL_INCONSISTENCY: rules that cannot be applied together or produce impossible/undefined outcomes in a scenario.
- DUPLICATE: the same obligation stated twice (possibly with diverging wording).
- CONFLICTING_VALUES: different numbers, ranges, units, types, formats for the same thing.
- TERMINOLOGY: the same concept named differently, or one term used for different concepts.
- MISSING_DEPENDENCY: a requirement relies on something (data, service, rule, state) that is not specified.
- MISSING_SCENARIO: a realistic scenario implied by the requirements has no specified behaviour.
- FUNC_NONFUNC_CONFLICT: functional and non-functional requirements that conflict.
- CROSS_DOCUMENT: inconsistency between the main specification and the child API specification (HTTP method vs side effects,
  parameters/types/limits vs functional rules, status codes and error semantics vs described behaviour, response fields vs data
  requirements, description vs actual behaviour, etc.).
- ACCEPTANCE_MISMATCH: acceptance criteria that do not match, or do not cover, the requirements.
- OTHER.

Rules for issues:
- "confirmed": true ONLY if the problem follows directly from the quoted text without extra assumptions.
  If it depends on an assumption, set "confirmed": false and explain which information is missing.
- Two requirements that describe different aspects are NOT contradictory. Missing info is not a contradiction.
- For CONTRADICTION, CONFLICTING_VALUES, DUPLICATE, CROSS_DOCUMENT, ACCEPTANCE_MISMATCH, FUNC_NONFUNC_CONFLICT and
  LOGICAL_INCONSISTENCY give verbatim evidence from BOTH conflicting places (one evidence item per place).
- "ref" of an evidence item is the Report ID (REQ-xxx) or the ID in the document of the element the quote comes from.
- "related_ids": Report IDs (REQ-xxx) of the involved elements. "related_sections": section titles when no ID fits.
- One issue per problem; do not split the same root cause into several issues. Do not report style nitpicks that belong to
  single-requirement review.
- Severity: CRITICAL = prevents unambiguous implementation or objective verification; HIGH; MEDIUM; LOW.
- If you find no issue in a category, simply return no issue for it. Return an empty list if the document is clean.
"""


def cross_system(language: str) -> str:
    return CROSS_SYSTEM_TMPL.format(rules=_COMMON_RULES.format(language_name=language_name(language)))


def requirement_block(r: Requirement, full: bool = True, limit: int = 220) -> str:
    did = f" (doc id: {r.doc_id})" if r.doc_id else ""
    body = r.text if full else (r.text.replace("\n", " ")[:limit])
    return f"[{r.report_id}]{did} type={r.type.value} | section: {r.section} | lines {r.line_start}-{r.line_end}\n{body}"


def cross_prompt(doc_title: str, focus: list[Requirement], others: list[Requirement], chunk_info: str = "") -> str:
    parts = [f"Document: {doc_title}\n"]
    if chunk_info:
        parts.append(chunk_info + "\n")
    parts.append("=== ELEMENTS TO ANALYSE IN DETAIL (full text) ===")
    parts.extend(requirement_block(r) for r in focus)
    if others:
        parts.append("\n=== OTHER ELEMENTS OF THE SAME DOCUMENT (abbreviated, for cross-checking) ===")
        parts.extend(requirement_block(r, full=False) for r in others)
    parts.append("\nFind relationship issues involving the detailed elements (the other elements may be the second party "
                 "of a conflict). Quote verbatim from the full texts you were given; if you need an abbreviated element's "
                 "exact wording and it is cut, do not quote it.")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
COMPLETENESS_SYSTEM_TMPL = """\
You are a senior requirements reviewer checking COMPLETENESS of the whole specification.

{rules}
For EACH of these areas return exactly one item:
{areas}

Statuses: COVERED | CONFIRMED_OMISSION | POTENTIAL_OMISSION | NOT_APPLICABLE.
- COVERED: the document addresses the area sufficiently (quote where).
- CONFIRMED_OMISSION: the area is clearly relevant for this document and it is absent or clearly insufficient. Quote the
  nearest related text that shows the gap (e.g. the only statement touching the area, or the requirement that needs it).
- POTENTIAL_OMISSION: it may be a gap but depends on context not in the document; say what to clarify.
- NOT_APPLICABLE: the area is irrelevant for this kind of document (explain).
Do NOT invent requirements that "must be added"; describe the gap and why it affects implementation/testing.
Severity applies to omissions (CRITICAL/HIGH/MEDIUM/LOW), NONE for COVERED/NOT_APPLICABLE.
"""


def completeness_system(language: str) -> str:
    areas = "\n".join(f"- {a.value} ({COMPLETENESS_AREA_RU[a]})" for a in CompletenessArea)
    return COMPLETENESS_SYSTEM_TMPL.format(rules=_COMMON_RULES.format(language_name=language_name(language)), areas=areas)


def completeness_prompt(doc_title: str, doc_text: str, is_full: bool, digest: str = "") -> str:
    label = "FULL DOCUMENT" if is_full else "DOCUMENT (TRUNCATED) — judge only what you can see; use POTENTIAL_OMISSION when unsure"
    extra = f"\n\n=== ELEMENT DIGEST ===\n{digest}" if digest else ""
    return f"Document: {doc_title}\n\n=== {label} ===\n{doc_text}\n=== END ==={extra}\n\nReturn one item per area."


# --------------------------------------------------------------------------- #
FINAL_SYSTEM_TMPL = """\
You are the lead reviewer giving the final readiness verdict for a requirements document, based ONLY on the statistics and the
validated findings provided.

{rules}
Verdict:
- READY: development and testing can start; only minor remarks.
- READY_WITH_RESERVATIONS: can start only after the listed clarifications, no blocking defects.
- NOT_READY: confirmed critical/high defects prevent unambiguous implementation or objective verification.
In "justification" explain the decision citing finding ids (F-xxx) and the concrete document places they point to.
In "priority_finding_ids" list up to 10 finding ids in descending order of importance (only ids that were provided).
"""


def final_system(language: str) -> str:
    return FINAL_SYSTEM_TMPL.format(rules=_COMMON_RULES.format(language_name=language_name(language)))


def final_prompt(doc_title: str, stats: str, findings_digest: str) -> str:
    return (f"Document: {doc_title}\n\n=== STATISTICS ===\n{stats}\n\n=== VALIDATED FINDINGS ===\n{findings_digest}\n\n"
            "Give the verdict, justification and priority order.")

