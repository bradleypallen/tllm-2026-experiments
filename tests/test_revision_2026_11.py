"""Tests for the November 2026 revision experiments: prompt audit trail, parsing, voting, item lists.

Run:  python -m pytest tests/test_revision_2026_11.py -q
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluators"))

import revision_2026_11_prompts as P  # noqa: E402

MD = (ROOT / "prompts" / "2026_11_revision_prompts.md").read_text()
FROZEN_MD = (ROOT / "prompts" / "2026_prompts.md").read_text()


def _code_blocks(text: str) -> list[str]:
    return [b.strip("\n") for b in re.findall(r"```\n(.*?)```", text, flags=re.S)]


# ---------------------------------------------------------------------------- audit trail

def test_frozen_prompts_match_2026_prompts_md():
    blocks = _code_blocks(FROZEN_MD)
    assert P.VERIFICATION_PROMPT in blocks, "verification prompt drifted from prompts/2026_prompts.md"
    assert P.REFUTATION_PROMPT in blocks, "refutation prompt drifted from prompts/2026_prompts.md"
    assert P.TERNARY_PROMPT in blocks, "ternary prompt drifted from prompts/2026_prompts.md"
    assert P.BILATERAL_SYSTEM_PROMPT in blocks, "bilateral system prompt drifted"
    assert P.TERNARY_SYSTEM_PROMPT in FROZEN_MD


def test_revision_prompts_match_revision_md():
    blocks = _code_blocks(MD)
    assert P.JOINT_PROMPT_E1A in blocks, "E1-a prompt differs between module and markdown"
    assert P.FOUR_OPTION_PROMPT in blocks, "E2 prompt differs between module and markdown"
    assert P.FOUR_OPTION_SYSTEM_PROMPT in MD
    assert P.PERMISSION_CLAUSE.strip() in MD


def test_e1b_is_e1a_minus_permission_clause_only():
    assert P.JOINT_PROMPT_E1B == P.JOINT_PROMPT_E1A.replace(P.PERMISSION_CLAUSE, "")
    assert P.PERMISSION_CLAUSE not in P.JOINT_PROMPT_E1B


def test_joint_prompt_reuses_frozen_token_definitions_verbatim():
    for line in ("- VERIFIED (if the assertion can be confirmed as true based on evidence)",
                 "- CANNOT VERIFY (if the assertion cannot be confirmed as true, either due to lack of evidence, uncertainty, or because it is false)",
                 "- REFUTED (if the assertion can be shown to be false based on evidence)",
                 "- CANNOT REFUTE (if the assertion cannot be shown to be false, either due to lack of contradictory evidence, uncertainty, or because it is true)"):
        assert line in P.JOINT_PROMPT_E1A
        assert line in P.VERIFICATION_PROMPT + P.REFUTATION_PROMPT


def test_four_option_prompt_is_minimal_edit_of_ternary():
    """Removing the three documented edits recovers the frozen ternary prompt exactly."""
    back = (P.FOUR_OPTION_PROMPT
            .replace(", undetermined, or supported and contradicted by conflicting evidence.", ", or undetermined.")
            .replace("- CONFLICTING (if there is evidence that supports the statement and evidence that contradicts it)\n", "")
            .replace("TRUE, FALSE, UNCERTAIN, or CONFLICTING.", "TRUE, FALSE, or UNCERTAIN."))
    assert back == P.TERNARY_PROMPT


# ---------------------------------------------------------------------------- parsing

def test_single_token_parsing():
    assert P.parse_single(" verified ", P.VERIFICATION_TOKENS) == "VERIFIED"
    assert P.parse_single("Cannot   verify.", P.VERIFICATION_TOKENS) == "CANNOT VERIFY"
    assert P.parse_single("CONFLICTING.", P.FOUR_OPTION_TOKENS) == "CONFLICTING"
    assert P.parse_single("CONFLICTING", P.TERNARY_TOKENS) == P.PARSE_FAIL      # not legal for three-option
    assert P.parse_single("TRUE. The statement is", P.TERNARY_TOKENS) == P.PARSE_FAIL
    assert P.parse_single("VERIFIED..", P.VERIFICATION_TOKENS) == P.PARSE_FAIL  # only ONE trailing period stripped


def test_joint_parsing():
    assert P.parse_joint("VERIFICATION: VERIFIED; REFUTATION: CANNOT REFUTE") == ("VERIFIED", "CANNOT REFUTE")
    assert P.parse_joint("verification: cannot verify\nrefutation: refuted") == ("CANNOT VERIFY", "REFUTED")
    assert P.parse_joint("VERIFICATION: VERIFIED | REFUTATION: REFUTED.") == ("VERIFIED", "REFUTED")
    assert P.parse_joint("VERIFICATION: VERIFIED") == P.PARSE_FAIL                        # missing field
    assert P.parse_joint("VERIFICATION: TRUE; REFUTATION: REFUTED") == P.PARSE_FAIL        # illegal token
    assert P.parse_joint("REFUTATION: REFUTED; VERIFICATION: VERIFIED") == P.PARSE_FAIL    # wrong order
    assert P.parse_joint("Sure. VERIFICATION: VERIFIED; REFUTATION: REFUTED") == P.PARSE_FAIL  # extra text


def test_component_and_majority():
    assert P.component("VERIFIED") == "t" and P.component("CANNOT VERIFY") == "f"
    assert P.component("REFUTED") == "t" and P.component("CANNOT REFUTE") == "f"
    assert P.component(P.PARSE_FAIL) == "e" and P.component(P.API_ERROR) == "e"
    assert P.majority(["t", "t", "f"]) == "t"
    assert P.majority(["t", "t", "e"]) == "t"
    assert P.majority(["t", "e", "e"]) == "e"
    assert P.majority(["t", "f", "e"]) == "e"


# ---------------------------------------------------------------------------- item lists

def test_item_lists_match_manifest_and_are_deterministic(tmp_path):
    d = ROOT / "datasets" / "revision_2026_11"
    manifest = json.loads((d / "MANIFEST.json").read_text())
    for name, info in manifest.items():
        assert hashlib.sha256((d / name).read_bytes()).hexdigest() == info["sha256"], f"{name} differs from MANIFEST"
    # Rebuild into a temp copy and compare hashes: selection must be a pure function of the frozen results.
    subprocess.check_call([sys.executable, str(d / "build_item_lists.py"), "--out", str(tmp_path)],
                          cwd=ROOT, stdout=subprocess.DEVNULL)
    for name, info in manifest.items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == info["sha256"], f"{name} not deterministic"


def test_e1_items_shape():
    recs = [json.loads(l) for l in (ROOT / "datasets" / "revision_2026_11" / "e1_items.jsonl").read_text().splitlines()]
    keys = {(r["model"], r["assertion_id"]) for r in recs}
    assert len(keys) == len(recs), "duplicate (model, assertion) in E1 list"
    assert all(r["orig_bilateral_tv"] == "<t,t>" for r in recs if r["condition"] == "treatment")
    assert all(r["orig_bilateral_tv"] == "<t,f>" for r in recs if r["condition"] == "control_tf")
    assert all(r["orig_bilateral_tv"] == "<f,f>" for r in recs if r["condition"] == "control_ff")
    assert all(r["baseline_subset"] for r in recs if r["condition"] == "treatment")
    assert not any(r["orig_bilateral_tv"] == "<e,e>" for r in recs)
    assert "claude-opus-4-1-20250805" not in {r["model"] for r in recs}


def test_e3_items_shape():
    recs = [json.loads(l) for l in (ROOT / "datasets" / "revision_2026_11" / "e3_items.jsonl").read_text().splitlines()]
    for r in recs:
        if r["polarity"] == "robust_true":
            assert r["orig_bilateral_tv"] == "<t,f>" and r["ground_truth"] and r["orig_ternary_prediction"] == "TRUE"
            assert r["probe"] == "refutation"
        else:
            assert r["orig_bilateral_tv"] == "<f,t>" and not r["ground_truth"] and r["orig_ternary_prediction"] == "FALSE"
            assert r["probe"] == "verification"
    per_model = {}
    for r in recs:
        per_model.setdefault(r["model"], []).append(r["polarity"])
    assert all(v.count("robust_true") == 50 and v.count("robust_false") == 50 for v in per_model.values())
