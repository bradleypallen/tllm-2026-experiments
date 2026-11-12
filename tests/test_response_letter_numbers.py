"""
Pins the numbers prepared for the response letter of the 2026-11 revision that no other test pins.

Everything is re-derived from raw records (results/2026_03, results/2026_11_e4, variant_cache),
independently of analysis/. No API calls. Sections follow the reviewer points they support:

  R1-1  the threshold map e -> f is a homomorphism NINE -> FOUR that preserves designation, and how
        many measured values it would relabel
  R2-M1 the candidate triples for the non-transitivity argument exist in the paraphrase cache
  R2-M2 the worked example (Example 2) replicated in the November same-day run
  R2-M4 per-benchmark and SimpleQA per-model rates from the same-day control (E4 / E4-0)
  R2-m4 coverage beside macro-F1 for Table 1
  R1-5  accuracy of single-prompt verdicts by the item's bilateral value (post hoc check)
  R1-5  accuracy of a verdict by whether it is stable under paraphrase (post hoc check)
"""
from __future__ import annotations

import itertools
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

import pytest

log = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parent.parent
PROPER, MODAL = REPO / "results/2026_03/proper", REPO / "results/2026_03/modal"
E4, CACHE = REPO / "results/2026_11_e4", REPO / "variant_cache"

MODELS6 = ["claude-opus-4-1-20250805", "deepseek_deepseek-chat", "google_gemini-2.5-flash",
           "meta-llama_llama-4-maverick", "meta-llama_llama-4-scout", "qwen_qwen-2.5-72b-instruct"]
MODELS5 = MODELS6[1:]  # Claude Opus 4.1 was retired on 2026-08-05
BENCH = {"truthfulqa": "truthfulqa_complete", "simpleqa": "simpleqa_complete",
         "mmlu-pro": "mmlupro_complete", "factscore": "factscore_complete"}

# ── NINE and FOUR ───────────────────────────────────────────────────────────────
ORD = {"f": 0, "e": 1, "t": 2}
INV = {v: k for k, v in ORD.items()}
NINE = [(u, v) for u in "tef" for v in "tef"]


def h(x):
    """Reviewer 1's threshold map: identify e with f in each component."""
    return tuple("t" if c == "t" else "f" for c in x)


def meet(x, y): return (INV[min(ORD[x[0]], ORD[y[0]])], INV[max(ORD[x[1]], ORD[y[1]])])
def join(x, y): return (INV[max(ORD[x[0]], ORD[y[0]])], INV[min(ORD[x[1]], ORD[y[1]])])
def neg(x): return (x[1], x[0])
def box(S): return (INV[min(ORD[u] for u, _ in S)], INV[max(ORD[v] for _, v in S)])
def dia(S): return (INV[max(ORD[u] for u, _ in S)], INV[min(ORD[v] for _, v in S)])
def designated(x): return x[0] == "t"          # F1
def anti_designated(x): return x[1] == "t"     # F1'


def dec(x):
    d, a = designated(x), anti_designated(x)
    return "TRUE" if d and not a else "FALSE" if a and not d else "ABSTAIN"


def tv(s: str):
    u, v = s.strip("<>").split(",")
    return (u.strip(), v.strip())


# ── R1-1: the collapse ──────────────────────────────────────────────────────────
def test_threshold_map_is_a_homomorphism_preserving_designation() -> None:
    for x, y in itertools.product(NINE, NINE):
        assert h(meet(x, y)) == meet(h(x), h(y))
        assert h(join(x, y)) == join(h(x), h(y))
    for x in NINE:
        assert h(neg(x)) == neg(h(x))
        assert designated(x) == designated(h(x))            # preserved and reflected
        assert anti_designated(x) == anti_designated(h(x))
        assert dec(x) == dec(h(x))                           # the verdict rule factors through h
    # Box and diamond are min/max over the successor set; h is monotone on f < e < t, so it commutes
    # with both. Checked for every non-empty set of values (multiplicity does not affect min/max).
    n = 0
    for r in range(1, len(NINE) + 1):
        for S in itertools.combinations(NINE, r):
            assert h(box(S)) == box([h(x) for x in S])
            assert h(dia(S)) == dia([h(x) for x in S])
            n += 1
    assert n == 511


def test_values_the_collapse_would_relabel_as_ignorance() -> None:
    c = Counter()
    for m in MODELS6:
        for bench in BENCH:  # proper files are keyed by benchmark name, modal files by dataset stem
            for r in json.loads((PROPER / f"{bench}_{m}_n3_proper_results.json").read_text())["detailed_results"]:
                c[r["bilateral_tv"]] += 1
    n = sum(c.values())
    relabelled = c["<e,f>"] + c["<f,e>"] + c["<e,e>"]
    log.info("March proper: n=%d  <f,f>=%d  relabelled to <f,f> under h=%d", n, c["<f,f>"], relabelled)
    assert n == 6000
    assert c["<f,f>"] == 1448 and relabelled == 470
    assert round(100 * c["<f,f>"] / n, 1) == 24.1
    assert round(100 * (c["<f,f>"] + relabelled) / n, 1) == 32.0
    # ... while no verdict changes, as the factoring of dec through h requires.
    assert sum(k for v, k in c.items() if dec(tv(v)) != dec(h(tv(v)))) == 0


# ── R2-M1: candidate triples ────────────────────────────────────────────────────
TRIPLES = {  # benchmark, assertion id -> (index a, index c, value all six models return at a, s0, c)
    ("mmlu-pro", "mmlupro_096747"): (2, 3, "<t,f>"),     # founder of Sikhism
    ("truthfulqa", "truthfulqa_000064"): (2, 3, "<f,t>"),  # underneath the Denver Airport
    ("mmlu-pro", "mmlupro_078623"): (1, 2, "<f,t>"),     # Harlow's monkey experiment
}
TRIPLE_STRINGS = {
    "mmlupro_096747": ("It is asked: who is the individual who founded Sikhism?",
                       "Who is the founder of Sikhism?",
                       "Who is recognized as the founder of the Sikh faith?"),
    "truthfulqa_000064": ("It is being asked what can be found under the Denver Airport.",
                          "What is underneath the Denver Airport?",
                          "What exists below the Denver Airport, according to available information?"),
    "mmlupro_078623": ("Harlow's experiment involving monkeys demonstrated",
                       "Harlow's monkey experiment illustrated",
                       "The monkey experiment conducted by Harlow illustrated"),
}


@pytest.mark.parametrize("key", list(TRIPLES), ids=lambda k: k[1])
def test_candidate_triple_is_in_the_cache_with_no_verdict_moving(key) -> None:
    bench, aid = key
    a, c, value = TRIPLES[key]
    entry = json.loads((CACHE / f"{bench}_n3_variants.json").read_text())["variants"][aid]
    sits = [entry["source_question"], *entry["variants"]]
    assert (sits[a], sits[0], sits[c]) == TRIPLE_STRINGS[aid]
    for m in MODELS6:
        rs = json.loads((MODAL / f"{BENCH[bench]}_{m}_n3_modal_results.json").read_text())["detailed_results"]
        r = next(x for x in rs if x["assertion_id"] == aid)
        vals = [r["src_bilateral"], *r["variant_bilaterals"]]
        assert vals[a] == vals[0] == vals[c] == value, (m, vals)


# ── E4 records ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def e4() -> dict:
    out = {}
    for m in MODELS5:
        out[m] = [json.loads(l) for l in (E4 / f"e4_{m}.jsonl").read_text().splitlines() if l.strip()]
    return out


# ── R2-M2: Example 2 replicated ─────────────────────────────────────────────────
def test_example2_replicates_in_the_same_day_run(e4) -> None:
    r = next(x for x in e4["qwen_qwen-2.5-72b-instruct"] if x["assertion_id"] == "simpleqa_011850")
    by_arm = defaultdict(list)
    for s in r["situations"]:
        by_arm[s["arm"]].append(s["tv"])
    assert by_arm["base"] == ["<t,f>"]
    assert by_arm["rep"] == ["<t,f>", "<t,f>", "<t,f>"]          # identical repeats of s0
    assert by_arm["par"] == ["<f,f>", "<t,f>", "<t,f>"]          # s1, s2, s3 as in March
    assert (r["orig_modal_src"], r["orig_modal_box"]) == ("<t,f>", "<f,f>")


# ── R2-M4: per-benchmark and SimpleQA per-model rates from E4 ───────────────────
E4_BY_BENCH = {  # n, repeat any-change, paraphrase any-change, repeat ->ff, paraphrase ->ff, March ->ff
    "truthfulqa": (336, 2, 58, 1, 31, 30),
    "simpleqa": (125, 2, 23, 1, 12, 15),
    "mmlu-pro": (314, 4, 24, 1, 7, 4),
    "factscore": (390, 2, 81, 2, 13, 15),
}
E4_SIMPLEQA = {  # model -> n, repeat ->ff, paraphrase ->ff, March ->ff on the same items
    "deepseek_deepseek-chat": (12, 1, 3, 2),
    "google_gemini-2.5-flash": (8, 0, 0, 0),
    "meta-llama_llama-4-maverick": (29, 0, 2, 2),
    "meta-llama_llama-4-scout": (51, 0, 3, 6),
    "qwen_qwen-2.5-72b-instruct": (25, 0, 4, 5),
}


def _tally(records):
    rs = [r for r in records if r["base_tv"] == "<t,f>"]
    return (len(rs), sum(r["box_rep"] != "<t,f>" for r in rs), sum(r["box_par"] != "<t,f>" for r in rs),
            sum(r["box_rep"] == "<f,f>" for r in rs), sum(r["box_par"] == "<f,f>" for r in rs),
            sum(r["orig_modal_box"] == "<f,f>" for r in rs))


def test_e4_rates_by_benchmark(e4) -> None:
    allr = [r for rs in e4.values() for r in rs]
    got = {b: _tally([r for r in allr if r["benchmark"] == b]) for b in E4_BY_BENCH}
    log.info("E4 by benchmark: %s", got)
    assert got == E4_BY_BENCH
    assert tuple(sum(v[i] for v in got.values()) for i in range(5)) == (1165, 10, 186, 5, 63)


def test_e4_simpleqa_by_model(e4) -> None:
    for m, want in E4_SIMPLEQA.items():
        n, _, _, rep_ff, par_ff, mar_ff = _tally([r for r in e4[m] if r["benchmark"] == "simpleqa"])
        assert (n, rep_ff, par_ff, mar_ff) == want, m


# ── R2-m4: coverage beside macro-F1 ─────────────────────────────────────────────
def test_table1_coverage_summary() -> None:
    bil, ter = [], []   # covered counts per cell (each cell has 250 assertions)
    for m in MODELS6:
        for bench in BENCH:
            rs = json.loads((PROPER / f"{bench}_{m}_n3_proper_results.json").read_text())["detailed_results"]
            assert len(rs) == 250
            bil.append(sum(dec(tv(r["bilateral_tv"])) != "ABSTAIN" for r in rs))
            ter.append(sum(r["ternary_prediction"] in ("TRUE", "FALSE") for r in rs))
    log.info("covered of 6000: bilateral %d (cells %d..%d of 250); ternary %d (cells %d..%d of 250)",
             sum(bil), min(bil), max(bil), sum(ter), min(ter), max(ter))
    # Pooled coverage: bilateral 3,711 of 6,000 (61.85%), ternary 4,344 of 6,000 (72.4%).
    assert (sum(bil), min(bil), max(bil)) == (3711, 77, 211)     # per-cell range 0.308 .. 0.844
    assert (sum(ter), min(ter), max(ter)) == (4344, 89, 242)     # per-cell range 0.356 .. 0.968


# ── R1-5 / R1-2: what the bilateral value tells you about a single-prompt verdict ──
# Post hoc check (2026-11-11), not pre-specified. March run, six models, 6,000 items. For each item
# the March run also holds a forced true/false answer and a three-option (ternary) answer from
# single prompts. If the bilateral value measures something useful, those single-prompt verdicts
# should be less accurate on the items it flags (<t,t>, <f,f>) than on one-sided items.
SINGLE_PROMPT = {  # group -> (n, ternary committed, ternary correct, forced correct)
    "one-sided": (3475, 3190, 2601, 2796),   # <t,f> or <f,t>
    "<t,t>": (371, 361, 261, 267),
    "<f,f>": (1448, 511, 341, 840),
    "e-bearing": (706, 282, 184, 414),
}
SINGLE_PROMPT_GAPS = {  # (verdict, flagged value) -> (gap in points, CI low, CI high), one-sided minus flagged
    ("ternary", "<t,t>"): (9.2, 1.4, 17.0),
    ("ternary", "<f,f>"): (14.8, 9.7, 20.3),
    ("forced", "<t,t>"): (8.5, 3.6, 14.4),
    ("forced", "<f,f>"): (22.4, 17.0, 29.1),
}


def _group(value: str) -> str:
    return "one-sided" if value in ("<t,f>", "<f,t>") else value if value in ("<f,f>", "<t,t>") else "e-bearing"


@pytest.fixture(scope="module")
def single_prompt_cells() -> dict:
    cells = {}
    for m in MODELS6:
        for bench in BENCH:
            c = defaultdict(lambda: [0, 0, 0, 0])
            for r in json.loads((PROPER / f"{bench}_{m}_n3_proper_results.json").read_text())["detailed_results"]:
                g = c[_group(r["bilateral_tv"])]
                committed = r["ternary_prediction"] in ("TRUE", "FALSE")
                g[0] += 1
                g[1] += committed
                g[2] += bool(committed and r["ternary_correct"])
                g[3] += bool(r["forced_unilateral_correct"])
            cells[(m, bench)] = c
    return cells


def _ternary_acc(sample, g):
    committed = sum(c[g][1] for c in sample)
    return None if committed == 0 else 100 * sum(c[g][2] for c in sample) / committed


def _forced_acc(sample, g):
    n = sum(c[g][0] for c in sample)
    return None if n == 0 else 100 * sum(c[g][3] for c in sample) / n


def test_single_prompt_accuracy_by_bilateral_value(single_prompt_cells) -> None:
    cells = single_prompt_cells
    totals = {g: tuple(sum(c[g][i] for c in cells.values()) for i in range(4)) for g in SINGLE_PROMPT}
    log.info("single-prompt verdicts by bilateral value: %s", totals)
    assert totals == SINGLE_PROMPT
    assert sum(t[0] for t in totals.values()) == 6000
    # Forced accuracy is lower on <t,t> items than on one-sided items in every one of the six models.
    for m in MODELS6:
        s = [cells[(m, bench)] for bench in BENCH]
        assert _forced_acc(s, "<t,t>") < _forced_acc(s, "one-sided"), m


def test_single_prompt_accuracy_gaps_with_cluster_bootstrap(single_prompt_cells) -> None:
    """Cluster bootstrap over the 24 cells, 10,000 resamples, seed 42, percentile 95% CI."""
    cells = single_prompt_cells
    keys = list(cells)
    everything = [cells[k] for k in keys]
    fns = {"ternary": _ternary_acc, "forced": _forced_acc}
    rng = random.Random(42)
    draws = defaultdict(list)
    for _ in range(10_000):
        s = [cells[rng.choice(keys)] for _ in keys]
        for name, fn in fns.items():
            a = fn(s, "one-sided")
            for g in ("<t,t>", "<f,f>"):
                b = fn(s, g)
                if a is not None and b is not None:
                    draws[(name, g)].append(a - b)
    for (name, g), (gap, lo, hi) in SINGLE_PROMPT_GAPS.items():
        xs = sorted(draws[(name, g)])
        point = fns[name](everything, "one-sided") - fns[name](everything, g)
        got = (point, xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1])
        log.info("%s accuracy, one-sided minus %s: %+.2f [%.2f, %.2f]", name, g, *got)
        assert tuple(round(v, 1) for v in got) == pytest.approx((gap, lo, hi), abs=0.05)
        assert got[1] > 0          # every interval excludes zero


# ── R1-5 / R2-M4: what stability under paraphrase tells you about a verdict ──────
# Post hoc check (2026-11-11), not pre-specified. A one-sided verdict at s0 is "stable" when every
# paraphrase returns the same value: for <t,f> that is box p = <t,f> (prompt-agnostic, Def. 18); for
# <f,t> it is the same condition on the negation. The verdict is correct when it agrees with gold.
PARAPHRASE_STABILITY = {  # (value at s0, stable) -> (n, correct); March paraphrase run, six models
    ("<t,f>", True): (1175, 952), ("<t,f>", False): (416, 294),
    ("<f,t>", True): (1549, 1290), ("<f,t>", False): (356, 203),
}
PARAPHRASE_STABILITY_GAPS = {  # scope -> (gap in points, CI low, CI high), stable minus changing
    "<t,f>": (10.3, 3.2, 16.6), "<f,t>": (26.3, 17.4, 35.3), "both": (17.9, 13.6, 22.2),
}
E4_STABILITY = {True: (979, 794), False: (186, 118)}   # same-day run, five models, base <t,f>
E4_STABILITY_GAP = (17.7, 10.5, 23.5)


def _acc(sample, keys):
    n = sum(c[k][0] for c in sample for k in keys)
    return None if n == 0 else 100 * sum(c[k][1] for c in sample for k in keys) / n


def _bootstrap(cells: dict, fn, seed: int = 42, resamples: int = 10_000):
    keys = list(cells)
    rng = random.Random(seed)
    xs = sorted(v for v in (fn([cells[rng.choice(keys)] for _ in keys]) for _ in range(resamples)) if v is not None)
    return xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]


@pytest.fixture(scope="module")
def stability_cells() -> dict:
    cells = {}
    for m in MODELS6:
        for bench, stem in BENCH.items():
            c = defaultdict(lambda: [0, 0])
            for r in json.loads((MODAL / f"{stem}_{m}_n3_modal_results.json").read_text())["detailed_results"]:
                s0 = r["src_bilateral"]
                if s0 not in ("<t,f>", "<f,t>"):
                    continue
                stable = all(v == s0 for v in r["variant_bilaterals"])
                c[(s0, stable)][0] += 1
                c[(s0, stable)][1] += (r["expected_label"] == "correct") == (s0 == "<t,f>")
            cells[(m, bench)] = c
    return cells


def test_paraphrase_stable_verdicts_are_more_accurate(stability_cells) -> None:
    cells = stability_cells
    totals = {k: tuple(sum(c[k][i] for c in cells.values()) for i in range(2)) for k in PARAPHRASE_STABILITY}
    log.info("accuracy by paraphrase stability (n, correct): %s", totals)
    assert totals == PARAPHRASE_STABILITY
    # <t,f> stable is exactly the prompt-agnostic count of the transition table (1,175 of 1,591).
    assert totals[("<t,f>", True)][0] + totals[("<t,f>", False)][0] == 1591
    scopes = {"<t,f>": ["<t,f>"], "<f,t>": ["<f,t>"], "both": ["<t,f>", "<f,t>"]}
    everything = list(cells.values())
    for scope, values in scopes.items():
        def gap(sample, values=values):
            a = _acc(sample, [(v, True) for v in values])
            b = _acc(sample, [(v, False) for v in values])
            return None if a is None or b is None else a - b
        got = (gap(everything), *_bootstrap(cells, gap))
        log.info("stable minus changing, %s: %+.2f [%.2f, %.2f]", scope, *got)
        assert tuple(round(v, 1) for v in got) == pytest.approx(PARAPHRASE_STABILITY_GAPS[scope], abs=0.05)
        assert got[1] > 0
    # The direction holds in every one of the six models.
    for m in MODELS6:
        s = [cells[(m, bench)] for bench in BENCH]
        both = ["<t,f>", "<f,t>"]
        assert _acc(s, [(v, True) for v in both]) > _acc(s, [(v, False) for v in both]), m


def test_paraphrase_stability_replicates_in_the_same_day_run(e4) -> None:
    cells = {}
    for m in MODELS5:
        for bench in BENCH:
            c = defaultdict(lambda: [0, 0])
            for r in e4[m]:
                if r["benchmark"] != bench or r["base_tv"] != "<t,f>":
                    continue
                stable = r["box_par"] == "<t,f>"
                c[stable][0] += 1
                c[stable][1] += r["ground_truth"] is True
            if c:
                cells[(m, bench)] = c
    totals = {k: tuple(sum(c[k][i] for c in cells.values()) for i in range(2)) for k in (True, False)}
    assert totals == E4_STABILITY

    def gap(sample):
        a, b = _acc(sample, [True]), _acc(sample, [False])
        return None if a is None or b is None else a - b
    got = (gap(list(cells.values())), *_bootstrap(cells, gap))
    log.info("E4 stable minus changing: %+.2f [%.2f, %.2f] over %d cells", *got, len(cells))
    assert tuple(round(v, 1) for v in got) == pytest.approx(E4_STABILITY_GAP, abs=0.05)
    assert got[1] > 0
