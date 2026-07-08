"""
Canonical BBL commit/abstain decision rule -- the single source of truth.

This module IS the decision rule defined in the paper (the "Venn" rule, §4, after
the consequence relation). Every decision-rule-dependent number in the paper
(Table 1 bilateral macro-F1, figure coverage annotations) is derived from
`commit_abstain` here, so the code's classifier and the paper's table agree by
construction rather than by coincidence.

The rule separates two jobs that a single designated set cannot do at once:
  - consequence designation  : D = F1 = {<u,v> : u = t}            (metalogic, |=)
  - commit/abstain decision  : the Venn of NINE's two prime bifilters (this file)

with F1' = ~F1 = {<u,v> : v = t} the second prime bifilter and
I = F1 ∩ F1' = {<t,t>} the inconsistency set. A value commits to its designated
side only when it is *consistent* (designated on exactly one side):

  TRUE    = F1 \\ F1' = { <t,f>, <t,e> }      designated, consistent
  FALSE   = F1' \\ F1 = { <f,t>, <e,t> }      negation-designated, consistent
  ABSTAIN = I = {<t,t>}  (inconsistency: both fire)
            ∪ complement of F1 ∪ F1' = { <f,f>, <e,f>, <f,e>, <e,e> }  (ignorance/partial)

Note FALSE-on-phi = TRUE-on-not-phi, since ~<u,v> = <v,u>.

Run `python3 decision_rule.py` to self-test the 9-value enumeration (it must match
the table printed in the paper exactly).
"""
from __future__ import annotations
import logging

log = logging.getLogger(__name__)

# The four cells on which the rule commits (designated on exactly one side).
TRUE_CELLS = frozenset({"<t,f>", "<t,e>"})       # F1 \ F1'
FALSE_CELLS = frozenset({"<f,t>", "<e,t>"})       # F1' \ F1
COMMIT_CELLS = TRUE_CELLS | FALSE_CELLS

# Verdict labels match the strings already persisted by the evaluator
# (`bilateral_<policy>_projected`) so downstream metric code is reusable verbatim.
TRUE, FALSE, ABSTAIN = "TRUE", "FALSE", "ABSTAIN"


def commit_abstain(tv_str: str) -> str:
    """Map a bilateral truth value '<u,v>' to TRUE / FALSE / ABSTAIN under the Venn rule.

    >>> commit_abstain("<t,f>")
    'TRUE'
    >>> commit_abstain("<t,t>")     # inconsistency -> abstain
    'ABSTAIN'
    >>> commit_abstain("<f,f>")     # ignorance -> abstain
    'ABSTAIN'
    """
    if tv_str in TRUE_CELLS:
        return TRUE
    if tv_str in FALSE_CELLS:
        return FALSE
    return ABSTAIN


def commit_bool(tv_str: str):
    """Decision rule as a {True, False, None} prediction (None = abstain), matching
    the `_component_label` convention in proper_benchmark_evaluator.py so that the
    existing `compute_metrics` can be reused unchanged."""
    v = commit_abstain(tv_str)
    return True if v == TRUE else False if v == FALSE else None


# The authoritative 9-value enumeration. Mirrors the paper's decision-rule table;
# the self-test below fails loudly if `commit_abstain` ever drifts from it.
ENUMERATION = {
    "<t,f>": (TRUE,    "designated, consistent"),
    "<t,e>": (TRUE,    "designated, consistent"),
    "<f,t>": (FALSE,   "negation-designated, consistent"),
    "<e,t>": (FALSE,   "negation-designated, consistent"),
    "<t,t>": (ABSTAIN, "inconsistency -- both fire (= I)"),
    "<f,f>": (ABSTAIN, "ignorance -- neither fires"),
    "<e,f>": (ABSTAIN, "partial -- neither fires"),
    "<f,e>": (ABSTAIN, "partial -- neither fires"),
    "<e,e>": (ABSTAIN, "partial -- neither fires"),
}

ALL_NINE = list(ENUMERATION.keys())


def _selftest() -> bool:
    ok = True
    for tv, (expected, reason) in ENUMERATION.items():
        got = commit_abstain(tv)
        flag = "OK" if got == expected else "MISMATCH"
        if got != expected:
            ok = False
        log.info("  %-7s -> %-8s (%-32s) [%s]", tv, got, reason, flag)
    # structural invariants asserted in the paper
    assert TRUE_CELLS == {"<t,f>", "<t,e>"}
    assert FALSE_CELLS == {"<f,t>", "<e,t>"}
    # inconsistency cell I = F1 ∩ F1' = {<t,t>} abstains; it is NOT a commit cell
    assert commit_abstain("<t,t>") == ABSTAIN
    assert "<t,t>" not in COMMIT_CELLS
    return ok


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log.info("BBL decision rule -- 9-value enumeration:")
    ok = _selftest()
    log.info("")
    log.info("SELF-TEST %s", "PASS" if ok else "FAIL")
