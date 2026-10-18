"""Prompt templates for the November 2026 revision experiments (E1-0/a/b, E2, E2-0, E3).

The frozen 2026-03 templates are reproduced verbatim from ``prompts/2026_prompts.md``; the new
templates (E1 joint prompt, E2 four-option prompt) are derived from them with the minimal edits
described in ``prompts/2026_11_revision_prompts.md``.  ``tests/test_revision_2026_11.py`` checks
that this module and that markdown file agree.

Placeholders: ``{assertion}`` is the assertion text; ``{context}`` is the per-assertion
``source_question`` field (s0 in the BBL formalism), exactly as in the 2026 run.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------------------------
# Frozen 2026-03 system prompts
# ---------------------------------------------------------------------------------------------

BILATERAL_SYSTEM_PROMPT = """You are an expert evaluating factual statements for accuracy across diverse domains. Your task is to determine whether claims can be verified or refuted based on established knowledge.

Focus on:
- Factual accuracy based on authoritative sources
- Current and historical knowledge across all domains
- Precise details including names, dates, numbers, and places
- Distinguishing between correct facts and plausible misinformation

Be especially careful with:
- Statements that sound plausible but contain factual errors
- Claims mixing accurate and inaccurate information
- Details that may be close to correct but are precisely wrong
- Domain-specific expertise requirements"""

TERNARY_SYSTEM_PROMPT = (
    "You are an expert fact-checker evaluating claims based on evidence. "
    "Respond with only TRUE, FALSE, or UNCERTAIN."
)

# ---------------------------------------------------------------------------------------------
# Frozen 2026-03 user prompts (E1-0 and E3 use these unchanged)
# ---------------------------------------------------------------------------------------------

VERIFICATION_PROMPT = """You are tasked with determining whether the following assertion can be verified as true based on available evidence and knowledge.

Assertion: {assertion}

Context: {context}

Your task is to determine if this assertion can be verified. Consider all available evidence, facts, and reliable sources of information.

You must respond with exactly one of these two token sequences:
- VERIFIED (if the assertion can be confirmed as true based on evidence)
- CANNOT VERIFY (if the assertion cannot be confirmed as true, either due to lack of evidence, uncertainty, or because it is false)

Do not provide any explanation or additional text. Respond with only the required token sequence.

Response:"""

REFUTATION_PROMPT = """You are tasked with determining whether the following assertion can be refuted (shown to be false) based on available evidence and knowledge.

Assertion: {assertion}

Context: {context}

Your task is to determine if this assertion can be refuted. Consider all available evidence, facts, and reliable sources of information that might contradict the assertion.

You must respond with exactly one of these two token sequences:
- REFUTED (if the assertion can be shown to be false based on evidence)
- CANNOT REFUTE (if the assertion cannot be shown to be false, either due to lack of contradictory evidence, uncertainty, or because it is true)

Do not provide any explanation or additional text. Respond with only the required token sequence.

Response:"""

TERNARY_PROMPT = """Based on available evidence and knowledge, determine whether the following statement is supported by evidence, contradicted by evidence, or undetermined.

Context: {context}

Statement: {assertion}

Consider all available facts and reliable sources. You must respond with exactly one of:
- TRUE (if evidence supports the statement as correct)
- FALSE (if evidence contradicts the statement as incorrect)
- UNCERTAIN (if there is insufficient evidence to either support or refute it)

Do not provide any explanation. Respond with only TRUE, FALSE, or UNCERTAIN.

Response:"""

# ---------------------------------------------------------------------------------------------
# E1-a: joint neutral prompt (approved 2026-10-18).  Token definitions verbatim from the frozen
# verification and refutation prompts.  E1-b is the same text with PERMISSION_CLAUSE removed.
# ---------------------------------------------------------------------------------------------

PERMISSION_CLAUSE = (
    " The two questions are independent: it is acceptable to answer that the assertion can be "
    "neither verified nor refuted, and it is acceptable to answer that it can be both verified and "
    "refuted if the evidence genuinely points both ways."
)

_JOINT_TEMPLATE = """You are tasked with assessing the following assertion on two independent questions: whether it can be verified as true based on available evidence and knowledge, and whether it can be refuted (shown to be false) based on available evidence and knowledge.

Assertion: {assertion}

Context: {context}

Consider all available evidence, facts, and reliable sources of information.{permission}

For the first question, respond with exactly one of:
- VERIFIED (if the assertion can be confirmed as true based on evidence)
- CANNOT VERIFY (if the assertion cannot be confirmed as true, either due to lack of evidence, uncertainty, or because it is false)

For the second question, respond with exactly one of:
- REFUTED (if the assertion can be shown to be false based on evidence)
- CANNOT REFUTE (if the assertion cannot be shown to be false, either due to lack of contradictory evidence, uncertainty, or because it is true)

Respond on one line in exactly this format, with no explanation or additional text:
VERIFICATION: <token>; REFUTATION: <token>

Response:"""

JOINT_PROMPT_E1A = _JOINT_TEMPLATE.replace("{permission}", PERMISSION_CLAUSE)
JOINT_PROMPT_E1B = _JOINT_TEMPLATE.replace("{permission}", "")

# ---------------------------------------------------------------------------------------------
# E2: four-option prompt.  Frozen ternary prompt with one clause, one option and the token list
# extended; nothing else changed.
# ---------------------------------------------------------------------------------------------

FOUR_OPTION_SYSTEM_PROMPT = (
    "You are an expert fact-checker evaluating claims based on evidence. "
    "Respond with only TRUE, FALSE, UNCERTAIN, or CONFLICTING."
)

FOUR_OPTION_PROMPT = """Based on available evidence and knowledge, determine whether the following statement is supported by evidence, contradicted by evidence, undetermined, or supported and contradicted by conflicting evidence.

Context: {context}

Statement: {assertion}

Consider all available facts and reliable sources. You must respond with exactly one of:
- TRUE (if evidence supports the statement as correct)
- FALSE (if evidence contradicts the statement as incorrect)
- UNCERTAIN (if there is insufficient evidence to either support or refute it)
- CONFLICTING (if there is evidence that supports the statement and evidence that contradicts it)

Do not provide any explanation. Respond with only TRUE, FALSE, UNCERTAIN, or CONFLICTING.

Response:"""

# ---------------------------------------------------------------------------------------------
# Parsing.  One normaliser for all modes: strip whitespace, case-fold, strip ONE trailing period,
# collapse internal whitespace; then exact match on the legal set.  Anything else is PARSE_FAIL.
# ---------------------------------------------------------------------------------------------

PARSE_FAIL = "PARSE_FAIL"
API_ERROR = "API_ERROR"


def normalise(text: str) -> str:
    t = text.strip()
    if t.endswith("."):
        t = t[:-1]
    t = re.sub(r"\s+", " ", t)
    return t.upper()


def parse_single(text: str, legal: tuple[str, ...]) -> str:
    """Exact match of the normalised response against ``legal``; otherwise PARSE_FAIL."""
    t = normalise(text)
    return t if t in legal else PARSE_FAIL


VERIFICATION_TOKENS = ("VERIFIED", "CANNOT VERIFY")
REFUTATION_TOKENS = ("REFUTED", "CANNOT REFUTE")
TERNARY_TOKENS = ("TRUE", "FALSE", "UNCERTAIN")
FOUR_OPTION_TOKENS = ("TRUE", "FALSE", "UNCERTAIN", "CONFLICTING")

_JOINT_RE = re.compile(
    r"^VERIFICATION:\s*(VERIFIED|CANNOT VERIFY)\s*[;|,]?\s*REFUTATION:\s*(REFUTED|CANNOT REFUTE)\s*$"
)


def parse_joint(text: str) -> tuple[str, str] | str:
    """Parse an E1 joint response into (verification_token, refutation_token) or PARSE_FAIL.

    Rule (prompts/2026_11_revision_prompts.md, "Parsing"): the response must BEGIN with the two
    labelled fields, in order, each holding a token from its legal set. Text after the first line
    break following a complete answer (an explanation the model appended) is ignored; the record is
    flagged ``trailing_text`` by the caller. Leading text, a missing or swapped field, or an illegal
    token is PARSE_FAIL for the whole elicitation.

    Widened on 2026-10-18 after the smoke test showed Llama 4 Scout appending explanations to
    otherwise well-formed answers; applied uniformly to all models by re-parsing stored raw text.
    """
    # Whole response is the answer (fields on one line or two): the strict rule.
    m = _JOINT_RE.match(normalise(text))
    if m:
        return m.group(1), m.group(2)
    # First-line rule: the answer on the first line (or first two, if the fields are split),
    # followed by an explanation the model appended.
    lines = [ln.strip() for ln in text.strip().upper().split("\n") if ln.strip()]
    if not lines:
        return PARSE_FAIL
    head = lines[0]
    if "REFUTATION:" not in head and len(lines) > 1:
        head = head + " " + lines[1]
    head = re.sub(r"\s+", " ", head)
    if head.endswith("."):
        head = head[:-1]
    m = _JOINT_RE.match(head)
    if m:
        return m.group(1), m.group(2)
    return PARSE_FAIL


def has_trailing_text(text: str) -> bool:
    """True when parse_joint succeeded only by ignoring text after the answer (first-line rule)."""
    if parse_joint(text) == PARSE_FAIL:
        return False
    return _JOINT_RE.match(normalise(text)) is None


def component(token: str) -> str:
    """Map a parsed token to a bilateral component value: 't', 'f', or 'e' (unparsed)."""
    return {"VERIFIED": "t", "CANNOT VERIFY": "f", "REFUTED": "t", "CANNOT REFUTE": "f"}.get(token, "e")


def majority(values: list[str]) -> str:
    """Majority vote over component values; 'e' when no value has a strict majority."""
    counts = {v: values.count(v) for v in set(values)}
    best = max(counts, key=counts.get)
    return best if counts[best] * 2 > len(values) else "e"
