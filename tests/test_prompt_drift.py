"""
Detect drift between the installed `bilateral-truth` package's prompts and the
verbatim prompts frozen in `prompts/2026_prompts.md`.

If anyone bumps the bilateral-truth dependency and the prompts have changed,
this test fails loudly. It is the canary that protects the experiments repo
from silent methodology shifts.

Requires `bilateral-truth` to be installed (e.g. via `pip install -r requirements.txt`).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPT_FILE = REPO_ROOT / "prompts" / "2026_prompts.md"

# Marker substrings that must appear in each prompt. We don't check the full
# verbatim text because the markdown file wraps with surrounding context — we
# just verify the load-bearing wording is still present in the package.
EXPECTED_MARKERS = {
    "verification": [
        "determining whether the following assertion can be verified",
        "VERIFIED (if the assertion can be confirmed as true based on evidence)",
        "CANNOT VERIFY (if the assertion cannot be confirmed as true",
    ],
    "refutation": [
        "determining whether the following assertion can be refuted",
        "REFUTED (if the assertion can be shown to be false based on evidence)",
        "CANNOT REFUTE (if the assertion cannot be shown to be false",
    ],
    "forced_unilateral": [
        "Determine whether the following statement is correct or incorrect",
        "TRUE (if the statement is correct)",
        "FALSE (if the statement is incorrect)",
    ],
    "ternary": [
        "Based on available evidence and knowledge",
        "supported by evidence, contradicted by evidence, or undetermined",
        "TRUE (if evidence supports the statement as correct)",
        "FALSE (if evidence contradicts the statement as incorrect)",
        "UNCERTAIN (if there is insufficient evidence to either support or refute it)",
    ],
    "confidence": [
        "Rate your confidence that the following statement is correct",
        "0.0 = Definitely incorrect",
        "1.0 = Definitely correct",
        "CONFIDENCE: [number]",
    ],
}


@pytest.fixture(scope="module")
def evaluator_class():
    """Return the bilateral_truth.llm_evaluators.LLMEvaluator base class."""
    pytest.importorskip("bilateral_truth")
    from bilateral_truth.llm_evaluators import LLMEvaluator
    return LLMEvaluator


@pytest.fixture(scope="module")
def assertion_class():
    pytest.importorskip("bilateral_truth")
    from bilateral_truth.assertions import Assertion
    return Assertion


class _TestEvaluator:
    """Wraps the LLMEvaluator base class to call its prompt-creation methods.
    Subclasses skip the abstract `evaluate_bilateral` to avoid instantiation issues."""


def _instantiate(evaluator_class):
    # The base class is abstract; build a minimal subclass for testing.
    class _Concrete(evaluator_class):
        def evaluate_bilateral(self, *args, **kwargs):
            raise NotImplementedError
        def _evaluate_verification(self, *args, **kwargs):
            raise NotImplementedError
        def _evaluate_refutation(self, *args, **kwargs):
            raise NotImplementedError
        def _raw_complete(self, *args, **kwargs):
            raise NotImplementedError
    return _Concrete()


def test_verification_prompt_markers(evaluator_class, assertion_class) -> None:
    ev = _instantiate(evaluator_class)
    prompt = ev._create_verification_prompt(assertion_class("the sky is blue"), context="What color is the sky?")
    for marker in EXPECTED_MARKERS["verification"]:
        assert marker in prompt, f"missing marker in verification prompt: {marker!r}"


def test_refutation_prompt_markers(evaluator_class, assertion_class) -> None:
    ev = _instantiate(evaluator_class)
    prompt = ev._create_refutation_prompt(assertion_class("the sky is blue"), context="What color is the sky?")
    for marker in EXPECTED_MARKERS["refutation"]:
        assert marker in prompt, f"missing marker in refutation prompt: {marker!r}"


def test_forced_unilateral_prompt_markers(evaluator_class, assertion_class) -> None:
    ev = _instantiate(evaluator_class)
    prompt = ev._create_forced_unilateral_prompt(assertion_class("the sky is blue"), context="What color is the sky?")
    for marker in EXPECTED_MARKERS["forced_unilateral"]:
        assert marker in prompt, f"missing marker in forced_unilateral prompt: {marker!r}"


def test_ternary_prompt_is_evidence_based(evaluator_class, assertion_class) -> None:
    ev = _instantiate(evaluator_class)
    prompt = ev._create_ternary_prompt(assertion_class("the sky is blue"), context="What color is the sky?")
    for marker in EXPECTED_MARKERS["ternary"]:
        assert marker in prompt, f"missing marker in ternary prompt: {marker!r}"
    # The 2025 confidence-based phrasing must NOT have crept back in:
    assert "if you are confident" not in prompt.lower(), (
        "ternary prompt looks confidence-based — bilateral-truth may have regressed. "
        "Check pinned commit in pyproject.toml."
    )


def test_confidence_prompt_markers(evaluator_class, assertion_class) -> None:
    ev = _instantiate(evaluator_class)
    prompt = ev._create_confidence_prompt_unilateral(assertion_class("the sky is blue"), context="What color is the sky?")
    for marker in EXPECTED_MARKERS["confidence"]:
        assert marker in prompt, f"missing marker in confidence prompt: {marker!r}"


def test_prompts_md_file_exists() -> None:
    assert PROMPT_FILE.exists(), f"missing audit file: {PROMPT_FILE}"
    content = PROMPT_FILE.read_text()
    # Make sure the audit file itself contains the load-bearing markers,
    # so it can't silently drift from the package while passing tests.
    for category, markers in EXPECTED_MARKERS.items():
        for marker in markers:
            assert marker in content, (
                f"prompts/2026_prompts.md is out of sync with this test "
                f"— missing {category} marker: {marker!r}"
            )
