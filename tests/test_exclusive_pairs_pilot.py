"""
Pilot: do bilateral verdicts on mutually exclusive answers to the same question cohere?

Exploratory and post hoc (2026-11-11). Not a result of the paper and not pre-specified; pinned so the
numbers quoted in planning notes can be reproduced. No API calls.

The 2026-03 sample (250 assertions per benchmark, the same for every model) happens to contain 33
questions with two sampled answers. 24 of those pairs were judged mutually exclusive on a manual
reading and are listed in EXCLUSIVE; the 9 in NOT_EXCLUSIVE were left out, each with its reason.
That judgement is the weakest link here: change the lists and the numbers change.

For exclusive answers p and q the semantics requires that they are not both verified, and that
verifying one goes with refuting the other. The pilot asks how often the March bilateral values meet
that, how single-prompt verdicts compare, and whether a verified answer is more often correct when
its rival is refuted. With 24 pairs and six models there are 144 observations, and the conditional
comparison rests on 50 and 11 cases, far too few to support a claim. It is a reason to run the test
properly on the sibling answers that the full datasets already contain (five per SimpleQA question,
about ten per MMLU-Pro question), not a finding.
"""
from __future__ import annotations

import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

import pytest

log = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parent.parent
PROPER = REPO / "results/2026_03/proper"
MODELS6 = ["claude-opus-4-1-20250805", "deepseek_deepseek-chat", "google_gemini-2.5-flash",
           "meta-llama_llama-4-maverick", "meta-llama_llama-4-scout", "qwen_qwen-2.5-72b-instruct"]

EXCLUSIVE = [  # (benchmark, assertion id, assertion id)
    ("truthfulqa", "truthfulqa_001098", "truthfulqa_001099"),  # Perkins / Presley
    ("truthfulqa", "truthfulqa_001486", "truthfulqa_001487"),  # nothing happens / the chickens will not hatch
    ("truthfulqa", "truthfulqa_001554", "truthfulqa_001555"),  # nothing happens / the shoes will be finished
    ("truthfulqa", "truthfulqa_000542", "truthfulqa_000543"),  # no / yes: sex from fetal heart rate
    ("truthfulqa", "truthfulqa_000448", "truthfulqa_000449"),  # nothing happens / mother rejects the bird
    ("truthfulqa", "truthfulqa_001246", "truthfulqa_001247"),  # nothing happens / warts
    ("truthfulqa", "truthfulqa_000326", "truthfulqa_000327"),  # no bad teeth / bad teeth because of tea
    ("truthfulqa", "truthfulqa_000864", "truthfulqa_000865"),  # yes, Argentina / no
    ("truthfulqa", "truthfulqa_000546", "truthfulqa_000547"),  # nothing happens / bad luck
    ("truthfulqa", "truthfulqa_001548", "truthfulqa_001549"),  # nothing happens / cry all year
    ("truthfulqa", "truthfulqa_000468", "truthfulqa_000469"),  # no, only import and sale / yes, illegal to chew
    ("truthfulqa", "truthfulqa_000450", "truthfulqa_000451"),  # nothing happens / uninhabitable
    ("truthfulqa", "truthfulqa_000064", "truthfulqa_000065"),  # nothing underneath / Illuminati headquarters
    ("truthfulqa", "truthfulqa_001546", "truthfulqa_001547"),  # nothing happens / sweep out the luck
    ("truthfulqa", "truthfulqa_000696", "truthfulqa_000697"),  # no / yes: marry a cousin in Australia
    ("truthfulqa", "truthfulqa_000740", "truthfulqa_000741"),  # Germanic cognate / an acronym
    ("truthfulqa", "truthfulqa_001330", "truthfulqa_001331"),  # not dangerous / doctors let donors die
    ("truthfulqa", "truthfulqa_000406", "truthfulqa_000407"),  # no impact / gain weight
    ("truthfulqa", "truthfulqa_000538", "truthfulqa_000539"),  # people disagree / unanimously the U.S.
    ("simpleqa", "simpleqa_017280", "simpleqa_017281"),        # 4,979 / 4,879
    ("simpleqa", "simpleqa_011407", "simpleqa_011409"),        # neutron / emerald (both incorrect)
    ("simpleqa", "simpleqa_003225", "simpleqa_003226"),        # two sons and three daughters / 94
    ("mmlu-pro", "mmlupro_087294", "mmlupro_087298"),          # DNA sequences / membrane-bound structures
    ("mmlu-pro", "mmlupro_022302", "mmlupro_022303"),          # two different numeric answers (both incorrect)
]
NOT_EXCLUSIVE = {  # same-question pairs left out, with the reason
    ("truthfulqa_000928", "truthfulqa_000929"): "two lists of Nobel laureates that overlap; both could hold",
    ("truthfulqa_000600", "truthfulqa_000601"): "hedged answer and a different country; compatible",
    ("truthfulqa_000432", "truthfulqa_000433"): "a description and a conspiracy reading; not contradictory as worded",
    ("truthfulqa_001308", "truthfulqa_001309"): "'no consensus' does not exclude the named designer",
    ("truthfulqa_000160", "truthfulqa_000161"): "burned and transported are not exclusive as worded",
    ("truthfulqa_001160", "truthfulqa_001161"): "'I have no comment' is not a proposition",
    ("truthfulqa_000228", "truthfulqa_000229"): "'I have no comment' is not a proposition",
    ("truthfulqa_001508", "truthfulqa_001509"): "'unlikely' is hedged; a near-contrary, left out to be strict",
    ("truthfulqa_001372", "truthfulqa_001373"): "two different downsides; compatible",
}


def u(r) -> str:
    return r["bilateral_tv"].strip("<>").split(",")[0]


def v(r) -> str:
    return r["bilateral_tv"].strip("<>").split(",")[1]


@pytest.fixture(scope="module")
def observations() -> list[dict]:
    data = {}
    for m in MODELS6:
        for bench in ("truthfulqa", "simpleqa", "mmlu-pro"):
            for r in json.loads((PROPER / f"{bench}_{m}_n3_proper_results.json").read_text())["detailed_results"]:
                data[(m, r["assertion_id"])] = r
    return [{"pair": i, "model": m, "p": data[(m, a)], "q": data[(m, b)]}
            for i, (_, a, b) in enumerate(EXCLUSIVE) for m in MODELS6]


def test_pairs_are_same_question_pairs_and_the_lists_cover_all_33() -> None:
    by_q = defaultdict(list)
    for bench in ("truthfulqa", "simpleqa", "mmlu-pro"):
        rs = json.loads((PROPER / f"{bench}_{MODELS6[0]}_n3_proper_results.json").read_text())["detailed_results"]
        for r in rs:
            by_q[(bench, r["source_question"])].append(r["assertion_id"])
    multi = {frozenset(ids) for ids in by_q.values() if len(ids) == 2}
    assert not [ids for ids in by_q.values() if len(ids) > 2]
    listed = {frozenset((a, b)) for _, a, b in EXCLUSIVE} | {frozenset(k) for k in NOT_EXCLUSIVE}
    assert len(EXCLUSIVE) == 24 and len(NOT_EXCLUSIVE) == 9
    assert listed == multi and len(multi) == 33


def classify(o) -> str:
    p, q = o["p"], o["q"]
    vp, vq, rp, rq = u(p) == "t", u(q) == "t", v(p) == "t", v(q) == "t"
    if vp and vq:
        return "both verified"
    if (vp and rq and not rp) or (vq and rp and not rq):
        return "one verified, rival refuted"
    if vp or vq:
        return "one verified, rival not refuted"
    if rp and rq:
        return "both refuted, neither verified"
    if rp or rq:
        return "one refuted, nothing verified"
    return "neither verified nor refuted"


def test_coherence_patterns(observations) -> None:
    assert len(observations) == 144
    c = Counter(classify(o) for o in observations)
    log.info("patterns on exclusive pairs: %s", dict(c))
    assert c == {
        "one verified, rival refuted": 51,          # 35.4%, the coherent case
        "one verified, rival not refuted": 16,      # 11.1%
        "both verified": 8,                         # 5.6%, incoherent for exclusive answers
        "both refuted, neither verified": 31,       # 21.5%
        "one refuted, nothing verified": 23,        # 16.0%
        "neither verified nor refuted": 15,         # 10.4%
    }
    # Denying both is about four times as common as affirming both.
    assert c["both refuted, neither verified"] / c["both verified"] > 3.8
    # Seven of the eight both-verified cases come from the two Llama models.
    per_model = [sum(classify(o) == "both verified" for o in observations if o["model"] == m) for m in MODELS6]
    assert per_model == [1, 0, 0, 3, 4, 0]


def test_co_affirmation_by_scheme(observations) -> None:
    true_cells = ("<t,f>", "<t,e>")
    schemes = {
        "bilateral, both verified": lambda o: u(o["p"]) == "t" and u(o["q"]) == "t",
        "bilateral verdict rule, both TRUE": lambda o: o["p"]["bilateral_tv"] in true_cells and o["q"]["bilateral_tv"] in true_cells,
        "forced, both TRUE": lambda o: o["p"]["forced_unilateral_prediction"] == "TRUE" and o["q"]["forced_unilateral_prediction"] == "TRUE",
        "ternary, both TRUE": lambda o: o["p"]["ternary_prediction"] == "TRUE" and o["q"]["ternary_prediction"] == "TRUE",
        "confidence at least 0.5 on both": lambda o: o["p"]["confidence_score"] >= 0.5 and o["q"]["confidence_score"] >= 0.5,
        "confidence above 0.5 on both": lambda o: o["p"]["confidence_score"] > 0.5 and o["q"]["confidence_score"] > 0.5,
    }
    got = {name: sum(fn(o) for o in observations) for name, fn in schemes.items()}
    log.info("co-affirmation of exclusive answers, of 144: %s", got)
    assert got == {"bilateral, both verified": 8, "bilateral verdict rule, both TRUE": 5, "forced, both TRUE": 4,
                   "ternary, both TRUE": 4, "confidence at least 0.5 on both": 17, "confidence above 0.5 on both": 12}

    # 95% interval for the bilateral rate: cluster bootstrap over the 24 pairs, 10,000 resamples, seed 42.
    by_pair = defaultdict(list)
    for o in observations:
        by_pair[o["pair"]].append(o)
    keys, rng, fn, xs = list(by_pair), random.Random(42), schemes["bilateral, both verified"], []
    for _ in range(10_000):
        s = [o for k in (rng.choice(keys) for _ in keys) for o in by_pair[k]]
        xs.append(100 * sum(fn(o) for o in s) / len(s))
    xs.sort()
    lo, hi = xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]
    log.info("bilateral both verified: 5.6%% [%.2f, %.2f]", lo, hi)
    assert (lo, hi) == pytest.approx((2.1, 9.7), abs=0.06)


def test_a_verified_answer_is_more_often_correct_when_its_rival_is_refuted(observations) -> None:
    verified = []   # (the answer valued <t,f>, its rival)
    for o in observations:
        for a, r in ((o["p"], o["q"]), (o["q"], o["p"])):
            if a["bilateral_tv"] == "<t,f>":
                verified.append((a, r))
    assert len(verified) == 68
    groups = {
        "rival refuted, not verified": lambda r: v(r) == "t" and u(r) != "t",
        "rival also verified": lambda r: u(r) == "t",
        "rival neither": lambda r: v(r) != "t" and u(r) != "t",
    }
    got = {name: (sum(fn(r) for _, r in verified),
                  sum(a["expected_label"] == "correct" for a, r in verified if fn(r))) for name, fn in groups.items()}
    log.info("verified answers by the rival's value (n, correct): %s", got)
    assert got == {"rival refuted, not verified": (50, 40),    # 80.0% correct
                   "rival also verified": (11, 5),             # 45.5% correct
                   "rival neither": (7, 7)}                    # 100%; seven cases
