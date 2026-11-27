"""Промпты для LLM. Инструкции на английском (точность), результат — на русском."""
from __future__ import annotations

from src.analysis.arithmetic import arithmetic_context
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
- Recalculate numeric examples and distinguish arithmetic from business logic. When an expected value already
  equals the computed value, do not report an arithmetic error or ask to replace it with the same value.
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
7. Traceability – can be located and linked where needed. A named API operation or parameter under its section
   can be uniquely traceable without a dedicated ID. Missing ID alone is not a defect. Name the specific missing
   or ambiguous relationship if it prevents tracing the requirement.
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
- CRITICAL: confirmed defect that blocks unambiguous implementation or objective testing of material behavior.
  State the concrete blocked scenario. Missing IDs, style, redundancy, and unsupported suspicions cannot be critical.
- HIGH: very likely to cause wrong implementation/testing or rework.
- MEDIUM: noticeable quality problem, workable with clarification.
- LOW: stylistic/minor risk.
"""


def language_name(code: str) -> str:
    return {"ru": "Russian", "en": "English"}.get(code, "Russian")


def individual_system(language: str, batch: bool = False) -> str:
    target = "EACH target requirement independently" if batch else "ONE requirement"
    return (
        f"You are a senior business/system analyst performing a static quality review of {target}.\n\n"
        + _COMMON_RULES.format(language_name=language_name(language))
        + "\n" + CRITERIA_DEFINITIONS
        + "\nAssess each target on its own. The full document is provided as context so that you do not report "
          "information as missing when it is stated elsewhere. Report only defects internal to this requirement. "
          "Leave conflicts or duplicates between separate requirements to the cross-review. Under Consistency "
          "assess internal consistency only. For a local omission, name the single missing condition or parameter "
          "in missing_field using the smallest relevant phrase copied from this requirement. Reuse that exact "
          "phrase when the same gap affects several criteria. Do not combine separate gaps in one assessment.\n"
    )


def individual_batch_prompt(doc_text: str, requirements: list[Requirement]) -> str:
    targets = "\n\n".join(
        f"Output req_id: {req.report_id}\n" + requirement_block(req)
        + (f"\nTable header: {req.context}" if req.context else "") for req in requirements
    )
    calculations = arithmetic_context(requirements)
    return (
        f"=== FULL DOCUMENT FOR CONTEXT ===\n{doc_text}\n=== END OF CONTEXT ===\n\n"
        f"=== TARGETS TO REVIEW INDEPENDENTLY ===\n{targets}\n"
        f"\n=== CHECKED ARITHMETIC (not source evidence) ===\n{calculations or 'None'}\n"
        "Return exactly one item for each target, in this order: "
        + ", ".join(req.report_id for req in requirements) + ". Copy req_id from 'Output req_id'. "
        "Each response must include all ten criteria. Keep explanations concise and cite only the target's text "
        "as local evidence."
    )


def individual_prompt(req: Requirement, doc_title: str, context_text: str, context_is_full: bool) -> str:
    ctx_label = "FULL DOCUMENT" if context_is_full else "DOCUMENT OUTLINE AND THE REQUIREMENT'S SECTION (document is large)"
    extra = f"\nTable header / context: {req.context}" if req.context else ""
    arithmetic = arithmetic_context([req])
    if arithmetic:
        extra += f"\nProgrammatically checked arithmetic (not source evidence): {arithmetic}"
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
- DUPLICATE: the same obligation and outcome stated twice (possibly with different wording). If one rule allows
  a concrete input, time, or transition that another forbids under the same conditions, report a conflict instead.
- CONFLICTING_VALUES: different numbers, ranges, units, types, formats for the same thing.
- TERMINOLOGY: the same concept named differently, or one term used for different concepts.
- MISSING_DEPENDENCY: a requirement relies on something (data, service, rule, state) that is not specified.
- MISSING_SCENARIO: a realistic scenario implied by the requirements has no specified behaviour.
- FUNC_NONFUNC_CONFLICT: functional and non-functional requirements that conflict.
- CROSS_DOCUMENT: inconsistency between the main specification and the child API specification (HTTP method vs side effects,
  parameters/types/limits vs functional rules, status codes and error semantics vs described behaviour, response fields vs data
  requirements, description vs actual behaviour, etc.).
- ACCEPTANCE_MISMATCH: acceptance criteria that do not match, or do not cover, the requirements.
- ARITHMETIC: a stated numeric total demonstrably differs from its calculation. Cite the complete acceptance row.
- OTHER.

Rules for issues:
- "confirmed": true ONLY if the problem follows directly from the quoted text without extra assumptions.
  If it depends on an assumption, set "confirmed": false and explain which information is missing.
- Two requirements that describe different aspects are NOT contradictory. Missing info is not a contradiction.
- A general rule and a specific exception can coexist. A similar status label or shared domain term does not
  prove conflict. Identify the same scenario that yields two incompatible outcomes before confirming.
- For eligibility and state changes, check action sequences (for example, repeated events, changes of settings,
  or retries). Compare explicit per-period limits with any rule that resets or bypasses earlier state.
- Compare each acceptance scenario with the applicable business rules even when they are in other sections.
  Follow the scenario from inputs and event dates through transitions to observable results. At month/day boundaries
  distinguish the time of the event from the time it is processed, and apply the stated time zone before comparing
  calendar periods. For amounts compare units, thresholds, and inclusive/exclusive bounds; for states compare
  preconditions and transitions. For a mismatch, say what the rule would produce and what the example expects.
- For CONTRADICTION, CONFLICTING_VALUES, DUPLICATE, CROSS_DOCUMENT, ACCEPTANCE_MISMATCH, FUNC_NONFUNC_CONFLICT and
  LOGICAL_INCONSISTENCY give verbatim evidence from BOTH conflicting places (one evidence item per place).
- "ref" of an evidence item is the Report ID (REQ-xxx) or the ID in the document of the element the quote comes from.
- "related_ids": Report IDs (REQ-xxx) of the involved elements. "related_sections": section titles when no ID fits.
- Copy the IDs from the headers directly above the quoted requirement texts. Provide one evidence item per involved
  requirement. Do not include indirect or background requirements in related_ids.
- One issue per problem; do not split the same root cause into several issues. Do not report style nitpicks that belong to
  single-requirement review.
- Separate an HTTP operation's side-effect semantics from a contradiction between its documented purpose and behavior.
- Severity: CRITICAL requires a concrete blocked implementation or testing scenario with verified evidence.
  Use HIGH for likely wrong behavior or rework, MEDIUM for non-blocking problems, and LOW for minor remarks.
  Duplication alone and potential issues cannot be critical.
- If you find no issue in a category, simply return no issue for it. Return an empty list if the document is clean.
"""


def cross_system(language: str) -> str:
    return CROSS_SYSTEM_TMPL.format(rules=_COMMON_RULES.format(language_name=language_name(language)))


def requirement_block(r: Requirement, full: bool = True, limit: int = 220) -> str:
    did = f" (doc id: {r.doc_id})" if r.doc_id else ""
    body = r.text if full else (r.text.replace("\n", " ")[:limit])
    return f"[{r.report_id}]{did} type={r.type.value} | section: {r.section} | lines {r.line_start}-{r.line_end}\n{body}"


def cross_prompt(doc_title: str, focus: list[Requirement], others: list[Requirement], chunk_info: str = "",
                 checked_findings: str = "") -> str:
    parts = [f"Document: {doc_title}\n"]
    if chunk_info:
        parts.append(chunk_info + "\n")
    parts.append("=== ELEMENTS TO ANALYSE IN DETAIL (full text) ===")
    parts.extend(requirement_block(r) for r in focus)
    if others:
        parts.append("\n=== OTHER ELEMENTS OF THE SAME DOCUMENT (abbreviated, for cross-checking) ===")
        parts.extend(requirement_block(r, full=False) for r in others)
    arithmetic = arithmetic_context(focus + others)
    if arithmetic:
        parts.append("\n=== CHECKED ARITHMETIC (not source evidence) ===\n" + arithmetic)
    if checked_findings:
        parts.append("\n=== ISSUES ALREADY RECORDED BY THE PROGRAM ===\n" + checked_findings
                     + "\nDo not report these same issues again. Review other problems independently.")
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
- For evidence references use an ID that appears in the source or the actual section title/path. Use an empty reference
  if the quote is unique and has no named section. Never supply a guessed section title.
- POTENTIAL_OMISSION: it may be a gap but depends on context not in the document; say what to clarify.
- NOT_APPLICABLE: the area is irrelevant for this kind of document (explain).
Do NOT invent requirements that "must be added"; describe the gap and why it affects implementation/testing.
Only report missing behavior or information. Existing contradictory or duplicate rules belong to cross-review;
if an area is described but inconsistent, use COVERED for coverage and leave the inconsistency to cross-review.
Identify one specific gap per area. Distinguish a missing trigger or outcome from a provided outcome with an
unspecified parameter. A rule that limits an action to the actor's own records already defines ownership.
Do not require speculative integrations, architecture details, or a dedicated acceptance heading when equivalent
coverage exists elsewhere or the need does not follow from the specified scope.
Severity applies to omissions (CRITICAL/HIGH/MEDIUM/LOW), NONE for COVERED/NOT_APPLICABLE.
"""


def completeness_system(language: str) -> str:
    areas = "\n".join(f"- {a.value} ({COMPLETENESS_AREA_RU[a]})" for a in CompletenessArea)
    return COMPLETENESS_SYSTEM_TMPL.format(rules=_COMMON_RULES.format(language_name=language_name(language)), areas=areas)


def completeness_prompt(doc_title: str, doc_text: str, is_full: bool, digest: str = "") -> str:
    label = "FULL DOCUMENT" if is_full else "DOCUMENT (TRUNCATED) — judge only what you can see; use POTENTIAL_OMISSION when unsure"
    extra = f"\n\n=== ELEMENT DIGEST ===\n{digest}" if digest else ""
    return f"Document: {doc_title}\n\n=== {label} ===\n{doc_text}\n=== END ==={extra}\n\nReturn one item per area."


def omission_review_system(language: str) -> str:
    return (
        "Independently verify proposed omissions against the complete specification. Search for counterexamples, "
        "including paraphrases and separately specified triggers and outcomes. Return one answer for each candidate. "
        "Use COVERED only if verbatim evidence answers every material part of the allegation. "
        "Use SUPPORTED only if the alleged gap is concrete, needed by behavior actually specified in the document, "
        "and no section supplies it. Use UNPROVEN for a mixed claim, a speculative expectation, a design choice, "
        "or insufficient evidence. Do not infer that an authorization rule is missing when 'own' already limits "
        "the action. A defined action after expiry is an existing scenario even if the duration is unspecified. "
        "A stated conditional branch is an existing scenario even if its condition is imprecise. Copy the relevant "
        "phrases exactly and use their report IDs. A referenced table, example, or API response can cover information "
        "without repeating it in each workflow step. Assess the alleged gap at the level of the whole specification. "
        "Do not invent or reclassify other issues.\n\n"
        + _COMMON_RULES.format(language_name=language_name(language))
    )


# --------------------------------------------------------------------------- #
FINAL_SYSTEM_TMPL = """\
You are the lead reviewer giving the final readiness verdict for a requirements document, based ONLY on the statistics and the
validated findings provided.

{rules}
Verdict:
- READY: development and testing can start; only minor remarks.
- READY_WITH_RESERVATIONS: can start only after the listed clarifications, no blocking defects.
- NOT_READY: confirmed critical/high defects prevent unambiguous implementation or objective verification.
Potential issues may justify clarification, but cannot be the basis for NOT_READY.
In "justification" explain the decision citing finding ids (F-xxx) and the concrete document places they point to.
In "priority_finding_ids" list up to 10 finding ids in descending order of importance (only ids that were provided).
"""


def final_system(language: str) -> str:
    return FINAL_SYSTEM_TMPL.format(rules=_COMMON_RULES.format(language_name=language_name(language)))


def final_prompt(doc_title: str, stats: str, findings_digest: str) -> str:
    return (f"Document: {doc_title}\n\n=== STATISTICS ===\n{stats}\n\n=== VALIDATED FINDINGS ===\n{findings_digest}\n\n"
            "Give the verdict, justification and priority order.")

