"""
Pin the headline numbers of the November 2026 revision (§7.2 of the paper) by re-deriving them
from the raw per-elicitation records in results/2026_11/, independently of analysis/.

The analysis-stage rules are re-implemented here (strict joint parse with the first-line
fallback; exclusion of parse-failure <e,e> records from persistence denominators) so that
agreement with results/derived/2026_11/*.json is evidence, not tautology. Bootstrap CIs are
not re-derived (seeded in the analysis); point estimates and counts are.

Expected values are the ones the derived tables carry as of commit f079283. If an upstream
file is regenerated with different numbers, update the constants here deliberately.

Run:  pytest tests/test_revision_numbers.py
"""
from __future__ import annotations

import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NEW = REPO_ROOT / "results" / "2026_11"
DERIVED = REPO_ROOT / "results" / "derived" / "2026_11"
sys.path.insert(0, str(REPO_ROOT / "evaluators"))
import revision_2026_11_prompts as P  # noqa: E402  frozen tokens, component(), majority()

log = logging.getLogger(__name__)

MODELS5 = {"meta-llama/llama-4-maverick": "Llama 4 Maverick", "meta-llama/llama-4-scout": "Llama 4 Scout",
           "google/gemini-2.5-flash": "Gemini 2.5 Flash", "deepseek/deepseek-chat": "DeepSeek-V3",
           "qwen/qwen-2.5-72b-instruct": "Qwen 2.5 72B"}

# ── expected (results/derived/2026_11 at f079283) ─────────────────────────────
PERSISTENCE = {  # (arm, condition) -> (n_valid, persistence)
    ("e1-0", "treatment"): (338, 0.589), ("e1-a", "treatment"): (298, 0.044), ("e1-b", "treatment"): (295, 0.024),
    ("e1-0", "control_tf"): (394, 0.906), ("e1-a", "control_tf"): (540, 0.828),
    ("e1-0", "control_ff"): (374, 0.845), ("e1-a", "control_ff"): (323, 0.901),
}
E1A_TREATMENT_MATRIX = {"<t,f>": 165, "<f,t>": 88, "<t,t>": 13, "<f,f>": 32, "<e,e>": 48}
P1_POOLED = (298, 0.587)            # E1-0 minus E1-a, paired, treatment
F3PRIME = {                          # value -> (n_valid-independent shares) uncertain, conflicting, e2-0 same-day uncertain
    "<f,f>": (0.600, 0.023, 0.520), "<t,t>": (0.006, 0.015, 0.018), "<t,f>": (0.030, 0.015, 0.031),
}
P2_POOLED = (340, 0.015, 0.019)     # n_tt, P(C|tt), P(C|ctrl)
E3 = {"Llama 4 Maverick": (100, 0.000), "Llama 4 Scout": (98, 0.031), "Gemini 2.5 Flash": (100, 0.150),
      "DeepSeek-V3": (100, 0.020), "Qwen 2.5 72B": (100, 0.020)}
FIRST_LINE_RECOVERED, STILL_FAIL, JOINT_ELICITATIONS = 7, 725, 5091

# ── independent re-implementation of the parsing rules ────────────────────────
_STRICT = re.compile(r"^VERIFICATION:\s*(VERIFIED|CANNOT VERIFY)\s*[;\n|,]?\s*REFUTATION:\s*(REFUTED|CANNOT REFUTE)$")
_HEAD = re.compile(r"^VERIFICATION:\s*(VERIFIED|CANNOT VERIFY)\s*[;|,]?\s*REFUTATION:\s*(REFUTED|CANNOT REFUTE)\s*$")


def _norm(t: str) -> str:
    t = t.strip()
    t = t[:-1] if t.endswith(".") else t
    return re.sub(r"\s+", " ", t).upper()


def joint(text: str):
    """(ver, ref) or None: strict whole-response rule, then the first-line fallback."""
    m = _STRICT.match(_norm(text))
    if m:
        return m.group(1), m.group(2)
    lines = [l.strip() for l in text.strip().upper().split("\n") if l.strip()]
    if not lines:
        return None
    head = lines[0] if "REFUTATION:" in lines[0] or len(lines) == 1 else lines[0] + " " + lines[1]
    head = re.sub(r"\s+", " ", head)
    head = head[:-1] if head.endswith(".") else head
    m = _HEAD.match(head)
    return (m.group(1), m.group(2)) if m else None


def single(text: str, legal) -> str | None:
    t = _norm(text)
    return t if t in legal else None


def _load(mode: str) -> list[dict]:
    recs = []
    for p in sorted((NEW / mode).glob(f"{mode}_*.jsonl")):
        recs += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return recs


def tv_and_validity(rec: dict) -> tuple[str, bool]:
    """Re-elicited <u,v> and whether the record counts toward persistence denominators."""
    els = rec["elicitations"]
    failed = False
    if isinstance(els, dict):
        u = []; w = []
        for e in els["verification"]:
            tok = None if e.get("error") else single(e["raw"] or "", P.VERIFICATION_TOKENS)
            failed |= tok is None; u.append("e" if tok is None else P.component(tok))
        for e in els["refutation"]:
            tok = None if e.get("error") else single(e["raw"] or "", P.REFUTATION_TOKENS)
            failed |= tok is None; w.append("e" if tok is None else P.component(tok))
    else:
        u = []; w = []
        for e in els:
            pr = None if e.get("error") else joint(e["raw"] or "")
            failed |= pr is None
            u.append("e" if pr is None else P.component(pr[0])); w.append("e" if pr is None else P.component(pr[1]))
    tv = f"<{P.majority(u)},{P.majority(w)}>"
    return tv, not (tv == "<e,e>" and failed)


ORIG = {"treatment": "<t,t>", "control_tf": "<t,f>", "control_ff": "<f,f>"}


@pytest.fixture(scope="module")
def e1():
    return {arm: {(r["model"], r["assertion_id"]): (r, *tv_and_validity(r)) for r in _load(arm)} for arm in ("e1-0", "e1-a", "e1-b")}


@pytest.mark.parametrize("arm,cond", sorted(PERSISTENCE), ids=lambda x: x)
def test_e1_persistence(e1, arm, cond) -> None:
    rows = [(tv, ok) for (r, tv, ok) in e1[arm].values() if r["condition"] == cond]
    valid = [tv for tv, ok in rows if ok]
    pers = sum(tv == ORIG[cond] for tv in valid) / len(valid)
    log.info("%s %s: n_valid=%d persistence=%.4f", arm, cond, len(valid), pers)
    n, p = PERSISTENCE[(arm, cond)]
    assert len(valid) == n
    assert round(pers, 3) == p


def test_e1a_treatment_matrix(e1) -> None:
    c = Counter(tv for (r, tv, ok) in e1["e1-a"].values() if r["condition"] == "treatment")
    assert sum(c.values()) == 346
    for v, n in E1A_TREATMENT_MATRIX.items():
        assert c[v] == n, (v, c[v])


def test_p1_paired_pooled(e1) -> None:
    keep0 = keepa = n = 0
    for key, (r0, tv0, ok0) in e1["e1-0"].items():
        if r0["condition"] != "treatment" or key not in e1["e1-a"]:
            continue
        ra, tva, oka = e1["e1-a"][key]
        if not (ok0 and oka):
            continue
        n += 1; keep0 += tv0 == "<t,t>"; keepa += tva == "<t,t>"
    diff = keep0 / n - keepa / n
    log.info("P1 pooled: n=%d diff=%.4f", n, diff)
    assert (n, round(diff, 3)) == P1_POOLED


def test_first_line_rule_recovery_count() -> None:
    rec = fail = tot = 0
    for arm in ("e1-a", "e1-b"):
        for r in _load(arm):
            for e in r["elicitations"]:
                if e.get("error"):
                    continue
                raw = e["raw"] or ""
                tot += 1
                strict = _STRICT.match(_norm(raw)) is not None
                wide = joint(raw) is not None
                rec += (not strict) and wide
                fail += not wide
    log.info("joint elicitations %d, recovered %d, still fail %d", tot, rec, fail)
    assert (tot, rec, fail) == (JOINT_ELICITATIONS, FIRST_LINE_RECOVERED, STILL_FAIL)


@pytest.fixture(scope="module")
def e2():
    return _load("e2"), {(r["model"], r["assertion_id"]): r for r in _load("e2-0")}


def _pred(r, legal):
    e = r["elicitations"][0]
    return None if e.get("error") else single(e["raw"] or "", legal)


@pytest.mark.parametrize("value", sorted(F3PRIME))
def test_f3prime_shares(e2, value) -> None:
    recs, e20 = e2
    rows = [r for r in recs if r["orig_bilateral_tv"] == value]
    preds = [q for q in (_pred(r, P.FOUR_OPTION_TOKENS) for r in rows) if q]
    same = [q for q in (_pred(e20[(r["model"], r["assertion_id"])], P.TERNARY_TOKENS)
                        for r in rows if (r["model"], r["assertion_id"]) in e20) if q]
    got = (round(preds.count("UNCERTAIN") / len(preds), 3), round(preds.count("CONFLICTING") / len(preds), 3),
           round(same.count("UNCERTAIN") / len(same), 3))
    log.info("F-3' %s: valid=%d same-day=%d shares=%s", value, len(preds), len(same), got)
    assert got == F3PRIME[value]


def test_p2_pooled(e2) -> None:
    recs, _ = e2
    n_tt = c_tt = n_c = c_c = 0
    for r in recs:
        q = _pred(r, P.FOUR_OPTION_TOKENS)
        if q is None:
            continue
        if r["orig_bilateral_tv"] == "<t,t>":
            n_tt += 1; c_tt += q == "CONFLICTING"
        elif r["orig_bilateral_tv"] in ("<t,f>", "<f,f>"):
            n_c += 1; c_c += q == "CONFLICTING"
    assert (n_tt, round(c_tt / n_tt, 3), round(c_c / n_c, 3)) == P2_POOLED


def test_e3_compliance() -> None:
    by = defaultdict(lambda: [0, 0])
    for r in _load("e3"):
        if r.get("complied") is None:
            continue
        by[MODELS5[r["model"]]][0] += 1; by[MODELS5[r["model"]]][1] += bool(r["complied"])
    for lab, (n, c) in by.items():
        log.info("E3 %s: %d/%d", lab, c, n)
        assert (n, round(c / n, 3)) == E3[lab], lab


def test_derived_json_agrees() -> None:
    """The committed derived tables carry the same numbers this file pins."""
    e1 = json.loads((DERIVED / "T-E1_neutral_reelicitation.json").read_text())["T-E1b_persistence"]
    for (arm, cond), (n, p) in PERSISTENCE.items():
        r = e1[f"{arm}|{cond}|pooled"]
        assert (r["n_valid"], round(r["persistence"], 3)) == (n, p)
    e3 = json.loads((DERIVED / "T-E3_compliance_base_rate.json").read_text())["rows"]
    for lab, (n, c) in E3.items():
        assert (e3[lab]["n_valid"], round(e3[lab]["c_M"], 3)) == (n, c)
