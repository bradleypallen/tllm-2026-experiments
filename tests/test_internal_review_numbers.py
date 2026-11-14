"""
Pins the numbers added to the manuscript on 2026-11-13 after an internal review of the revision.

Everything is re-derived from raw records (results/2026_03/proper, results/2026_11_e4),
independently of analysis/. No API calls. One section per point of the review:

  Table 4   which scheme has the higher coverage and the higher macro-F1 in each cell, and how
            accurate each scheme is where both commit and where only one does
  E4        the 70 changes to a value with an e component: which models, and whether any is a
            failed call; how many changes alter the verdict of Definition 10
  E4        whether a paraphrase that changes one model's value changes the other models' values
            (an unfaithful paraphrase would be expected to move every model)
  fidelity  the share of assertions with at least one unfaithful paraphrase implied by the audit

The E1 figure without Gemini 2.5 Flash is pinned in tests/test_revision_numbers.py, which holds
the E1 parsing rules.

Run:  pytest tests/test_internal_review_numbers.py
"""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import pytest

log = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parent.parent
PROPER, E4 = REPO / "results/2026_03/proper", REPO / "results/2026_11_e4"

MODELS6 = ["claude-opus-4-1-20250805", "deepseek_deepseek-chat", "google_gemini-2.5-flash",
           "meta-llama_llama-4-maverick", "meta-llama_llama-4-scout", "qwen_qwen-2.5-72b-instruct"]
BENCH = ["truthfulqa", "simpleqa", "mmlu-pro", "factscore"]


def dec(tv: str) -> str:
    """Definition 10: TRUE on F1 \\ F1', FALSE on F1' \\ F1, ABSTAIN otherwise."""
    u, v = (x.strip() for x in tv.strip("<>").split(","))
    d, a = u == "t", v == "t"
    return "TRUE" if d and not a else "FALSE" if a and not d else "ABSTAIN"


def macro_f1(pairs: list[tuple[bool, bool]]) -> float:
    """Macro-F1 over the TRUE and FALSE classes, on the (gold, predicted) pairs given."""
    f1s = []
    for cls in (True, False):
        tp = sum(1 for g, p in pairs if g == cls and p == cls)
        fp = sum(1 for g, p in pairs if g != cls and p == cls)
        fn = sum(1 for g, p in pairs if g == cls and p != cls)
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(f1s) / 2


@pytest.fixture(scope="module")
def proper() -> dict:
    return {(b, m): json.loads((PROPER / f"{b}_{m}_n3_proper_results.json").read_text())["detailed_results"]
            for b in BENCH for m in MODELS6}


# ── Table 4: coverage and macro-F1 by cell ──────────────────────────────────────
def test_table4_which_scheme_has_higher_coverage_and_f1(proper) -> None:
    tern_both = bil_both = bil_cov_higher = 0
    for cell, rs in proper.items():
        assert len(rs) == 250
        bil = [(r["ground_truth"], dec(r["bilateral_tv"]) == "TRUE") for r in rs if dec(r["bilateral_tv"]) != "ABSTAIN"]
        tern = [(r["ground_truth"], r["ternary_prediction"] == "TRUE") for r in rs if r["ternary_prediction"] in ("TRUE", "FALSE")]
        fb, ft = macro_f1(bil), macro_f1(tern)
        tern_both += len(tern) > len(bil) and ft >= fb
        bil_both += len(bil) > len(tern) and fb >= ft
        bil_cov_higher += len(bil) > len(tern)
    log.info("ternary higher coverage and F1 >=: %d; bilateral: %d; bilateral coverage higher: %d",
             tern_both, bil_both, bil_cov_higher)
    assert (tern_both, bil_both, bil_cov_higher) == (14, 0, 3)


COMMIT_SPLIT = {  # scope -> (both n, bilateral correct, ternary correct, bilateral-only n, correct, ternary-only n, correct)
    "all": (3311, 2622, 2684, 400, 251, 1033, 703),
    "simpleqa": (828, 721, 743, 108, 68, 254, 206),
}


@pytest.mark.parametrize("scope", sorted(COMMIT_SPLIT))
def test_accuracy_where_both_commit_and_where_one_does(proper, scope) -> None:
    both = [0, 0, 0]; bo = [0, 0]; to = [0, 0]
    for (b, _), rs in proper.items():
        if scope != "all" and b != scope:
            continue
        for r in rs:
            bv, tp, g = dec(r["bilateral_tv"]), r["ternary_prediction"], r["ground_truth"]
            bc, tc = bv != "ABSTAIN", tp in ("TRUE", "FALSE")
            if bc and tc:
                both[0] += 1; both[1] += (bv == "TRUE") == g; both[2] += (tp == "TRUE") == g
            elif bc:
                bo[0] += 1; bo[1] += (bv == "TRUE") == g
            elif tc:
                to[0] += 1; to[1] += (tp == "TRUE") == g
    got = (*both, *bo, *to)
    log.info("%s: %s", scope, got)
    assert got == COMMIT_SPLIT[scope]
    n, bcor, tcor, bon, bocor, ton, tocor = got
    pct = {"all": (79.2, 81.1, 62.8, 68.1), "simpleqa": (87.1, 89.7, 63.0, 81.1)}[scope]
    assert (round(100 * bcor / n, 1), round(100 * tcor / n, 1),
            round(100 * bocor / bon, 1), round(100 * tocor / ton, 1)) == pct


# ── E4: the changes to a value with an e component, and the verdict ─────────────
@pytest.fixture(scope="module")
def e4() -> list[dict]:
    rows = []
    for p in sorted(E4.glob("e4_*.jsonl")):
        rows += [r for r in map(json.loads, p.read_text().splitlines()) if r["base_tv"] == "<t,f>"]
    return rows


VERDICT_TOKENS = {"VERIFIED", "CANNOT VERIFY", "REFUTED", "CANNOT REFUTE"}


def test_e4_e_bearing_changes_are_unparsed_prose_from_two_models(e4) -> None:
    assert len(e4) == 1165
    changed = [r for r in e4 if r["box_par"] != "<t,f>"]
    eb = [r for r in changed if "e" in r["box_par"]]
    assert (len(changed), len(eb)) == (186, 70)
    by_model = Counter(r["model"] for r in eb)
    assert by_model == {"meta-llama/llama-4-maverick": 52, "meta-llama/llama-4-scout": 18}
    # None is a failed call: no API error on any call of these pairs, and every component without
    # a verdict token is non-empty text that stops short of the 32-token response limit's end.
    assert sum(r["n_api_error"] for r in eb) == 0
    words = []
    for r in eb:
        for s in r["situations"]:
            if s["arm"] != "par":
                continue
            for side in ("verification", "refutation"):
                x = s[side]
                if x.get("parsed") not in VERDICT_TOKENS:
                    assert not x.get("error") and (x.get("raw") or "").strip()
                    words.append(len(x["raw"].split()))
    log.info("unparsed components: %d, words min %d max %d", len(words), min(words), max(words))
    assert len(words) == 122 and min(words) >= 17 and max(words) <= 32
    for p in sorted(E4.glob("e4_*.meta.json")):
        rp = json.loads(p.read_text())["request_params"]
        rp = json.loads(rp) if isinstance(rp, str) else rp
        assert rp["max_tokens"] == 32


def test_e4_verdict_changes(e4) -> None:
    par = sum(dec(r["box_par"]) != "TRUE" for r in e4)
    rep = sum(dec(r["box_rep"]) != "TRUE" for r in e4)
    log.info("verdict changes: paraphrase %d, repetition %d of %d", par, rep, len(e4))
    assert (par, rep) == (171, 10)
    assert (round(100 * par / len(e4), 1), round(100 * rep / len(e4), 1)) == (14.7, 0.9)
    # the 15 that keep the verdict are <t,e>
    assert Counter(r["box_par"] for r in e4 if r["box_par"] != "<t,f>" and dec(r["box_par"]) == "TRUE") == {"<t,e>": 15}


# ── E4: is a change at a paraphrase shared across the models that hold the assertion? ──
def _value(s: dict) -> str:
    u = {"VERIFIED": "t", "CANNOT VERIFY": "f"}.get(s["verification"].get("parsed"), "e")
    v = {"REFUTED": "t", "CANNOT REFUTE": "f"}.get(s["refutation"].get("parsed"), "e")
    return f"<{u},{v}>"


def test_e4_changes_at_a_paraphrase_are_mostly_specific_to_one_model(e4) -> None:
    holders = defaultdict(set)     # assertion -> models for which it is <t,f> at s0 on the day
    movers = defaultdict(set)      # (assertion, paraphrase index) -> models whose value there is not <t,f>
    for r in e4:
        a = (r["benchmark"], r["assertion_id"])
        holders[a].add(r["model"])
        for s in r["situations"]:
            if s["arm"] == "par" and _value(s) != "<t,f>":
                movers[(a, s["index"])].add(r["model"])
    shared = {k: ms for k, ms in movers.items() if len(holders[k[0]]) >= 2}
    one = sum(len(ms) == 1 for ms in shared.values())
    everyone = sum(len(ms) == len(holders[k[0]]) for k, ms in shared.items())
    log.info("paraphrases that change a value: %d; held by >=2 models: %d; one model: %d; all: %d",
             len(movers), len(shared), one, everyone)
    assert (len(holders), len(movers), len(shared), one, everyone) == (434, 228, 170, 148, 5)


# ── Paraphrase fidelity: what the audit implies per assertion ───────────────────
def test_share_of_assertions_with_an_unfaithful_paraphrase() -> None:
    audit = json.loads((REPO / "audit/audit_results.json").read_text())
    assert (audit["n"], audit["pass"]) == (100, 95)   # 95 of 100 audited variants pass
    assert round(100 * (1 - 0.95 ** 3)) == 14      # three paraphrases per assertion
    assert round(100 * (1 - 0.95 ** 2)) == 10      # two (FACTScore)
