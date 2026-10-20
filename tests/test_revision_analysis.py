"""
Tests for the analysis-stage E1 joint-response parser in ``analysis/revision_2026_11.py``.

The evaluator's parser (``evaluators/revision_2026_11_prompts.py``, frozen with the prompts and
covered by ``test_revision_2026_11.py``) requires the whole response to be the two labelled fields.
The analysis stage re-parses stored raw text under a first-line rule that also accepts a well-formed
answer followed by an appended explanation. These tests pin that rule and its relation to the strict one.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "analysis"))
sys.path.insert(0, str(REPO_ROOT / "evaluators"))

import revision_2026_11 as A  # noqa: E402
import revision_2026_11_prompts as P  # noqa: E402


def test_strict_answers_parse_identically() -> None:
    """Anything the frozen rule accepts, the first-line rule accepts with the same tokens and no flag."""
    for text in ("VERIFICATION: VERIFIED; REFUTATION: CANNOT REFUTE",
                 "verification: cannot verify\nrefutation: refuted",
                 "VERIFICATION: VERIFIED, REFUTATION: REFUTED."):
        assert P.parse_joint(text) != P.PARSE_FAIL
        assert A.parse_joint_first_line(text) == P.parse_joint(text)
        assert A.has_trailing_text(text) is False


def test_first_line_rule_recovers_appended_explanation() -> None:
    scout = ("VERIFICATION: CANNOT VERIFY; REFUTATION: REFUTED\n\n"
             "Roberto Clemente was a Puerto Rican professional baseball player who")
    assert P.parse_joint(scout) == P.PARSE_FAIL  # the frozen rule rejects it
    assert A.parse_joint_first_line(scout) == ("CANNOT VERIFY", "REFUTED")
    assert A.has_trailing_text(scout) is True
    # Answer split across two lines, then explanation.
    split = "VERIFICATION: VERIFIED\nREFUTATION: CANNOT REFUTE\n\nBecause the sources agree."
    assert A.parse_joint_first_line(split) == ("VERIFIED", "CANNOT REFUTE")
    assert A.has_trailing_text(split) is True
    # Trailing period on the answer line, then explanation.
    assert A.parse_joint_first_line("VERIFICATION: VERIFIED; REFUTATION: CANNOT REFUTE.\nThe claim is well documented.") \
        == ("VERIFIED", "CANNOT REFUTE")


def test_first_line_rule_still_fails_where_it_should() -> None:
    # Leading text.
    assert A.parse_joint_first_line("Sure. VERIFICATION: VERIFIED; REFUTATION: REFUTED") == P.PARSE_FAIL
    # Swapped tokens are illegal for their fields (seen from Llama 4 Scout and DeepSeek).
    assert A.parse_joint_first_line("VERIFICATION: REFUTED; REFUTATION: VERIFIED\n\nExplanation: ...") == P.PARSE_FAIL
    # Reasoning instead of answering.
    assert A.parse_joint_first_line("To assess the assertion, let's calculate Jane's mean score.\n\nThe scores are") == P.PARSE_FAIL
    # Two alternative answers offered: the first line is taken only if complete and legal.
    two = ("VERIFICATION: REFUTED; REFUTATION: VERIFIED \n\nor alternatively in the required format:\n"
           "VERIFICATION: CANNOT VERIFY; REFUTATION: REFUTED")
    assert A.parse_joint_first_line(two) == P.PARSE_FAIL
    # Empty.
    assert A.parse_joint_first_line("") == P.PARSE_FAIL
    assert A.has_trailing_text("") is False
