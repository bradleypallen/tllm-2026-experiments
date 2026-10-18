"""
Re-derive every quantitative claim in the BBL paper from the raw per-assertion records.

Paper version: bbl_paper_lncs_v2 (LNCS, 25 pp.). Expected values below are transcribed
from that PDF; if a later revision changes a number, update the constant here.

This is an *independent* implementation: it does not import `analysis/` or read
`results/derived/`, so agreement with the paper is evidence rather than tautology.
It reads only `results/2026_03/{proper,modal}/*.json` for the canonical six models.

Covered:
- §7 setup             : N=250 balanced, same assertions across models, k paraphrases, paraphraser
- Def. 7               : stored [[□p]] == <min u, max v> over s₀ + variants
- Table 1              : per-cell bilateral (dec, Def. 10) and ternary macro-F1, Δ, median/mean/max |Δ|
- Fig. 2               : per-cell <t,f> → <f,f> migration rate and base-<t,f> count
- Fig. 3 + §7.2        : ternary abstention by bilateral value; cluster-bootstrap CIs
- §7.2 / §8            : <t,t> share range and pooled; e-bearing share; confidence cross-check
- §6.3                 : both worked examples (Qwen 2.5 72B, SimpleQA), incl. Example 2 modal field

Provenance note (see docs/methodology.md): the modal run elicited each situation once
(samples=1); the proper run used a 3-vote majority. Fig. 2 and §6.3 therefore read s₀ from
the modal files, Table 1 and Fig. 3 from the proper files. `test_modal_s0_is_separate_elicitation`
pins that fact so it cannot be silently lost.

Run:  pytest tests/test_paper_numbers.py
"""
from __future__ import annotations

import json
import logging
import random
import statistics
from collections import defaultdict
from pathlib import Path

import pytest

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROPER = REPO_ROOT / "results" / "2026_03" / "proper"
MODAL = REPO_ROOT / "results" / "2026_03" / "modal"

MODELS = [  # (filename key, paper label), Table 1 order
    ("claude-opus-4-1-20250805", "Claude Opus 4.1"),
    ("meta-llama_llama-4-maverick", "Llama 4 Maverick"),
    ("meta-llama_llama-4-scout", "Llama 4 Scout"),
    ("google_gemini-2.5-flash", "Gemini 2.5 Flash"),
    ("deepseek_deepseek-chat", "DeepSeek-V3"),
    ("qwen_qwen-2.5-72b-instruct", "Qwen 2.5 72B"),
]
BENCHMARKS = [  # (proper filename key, modal filename key, paper label)
    ("truthfulqa", "truthfulqa_complete", "TruthfulQA"),
    ("simpleqa", "simpleqa_complete", "SimpleQA"),
    ("mmlu-pro", "mmlupro_complete", "MMLU-Pro"),
    ("factscore", "factscore_complete", "FACTScore"),
]
CELLS = [(b, m) for _, _, b in BENCHMARKS for _, m in MODELS]

# ── Paper values ────────────────────────────────────────────────────────────────
TABLE1 = {  # (benchmark, model) -> (bilateral F1, ternary F1, Δ)
    ("TruthfulQA", "Claude Opus 4.1"): (0.905, 0.938, -0.033),
    ("TruthfulQA", "Llama 4 Maverick"): (0.769, 0.781, -0.012),
    ("TruthfulQA", "Llama 4 Scout"): (0.719, 0.772, -0.052),
    ("TruthfulQA", "Gemini 2.5 Flash"): (0.828, 0.856, -0.028),
    ("TruthfulQA", "DeepSeek-V3"): (0.827, 0.831, -0.005),
    ("TruthfulQA", "Qwen 2.5 72B"): (0.834, 0.839, -0.004),
    ("SimpleQA", "Claude Opus 4.1"): (0.892, 0.943, -0.051),
    ("SimpleQA", "Llama 4 Maverick"): (0.756, 0.865, -0.109),
    ("SimpleQA", "Llama 4 Scout"): (0.904, 0.858, +0.046),
    ("SimpleQA", "Gemini 2.5 Flash"): (0.624, 0.809, -0.184),
    ("SimpleQA", "DeepSeek-V3"): (0.719, 0.843, -0.124),
    ("SimpleQA", "Qwen 2.5 72B"): (0.775, 0.925, -0.150),
    ("MMLU-Pro", "Claude Opus 4.1"): (0.897, 0.898, 0.000),
    ("MMLU-Pro", "Llama 4 Maverick"): (0.696, 0.749, -0.054),
    ("MMLU-Pro", "Llama 4 Scout"): (0.684, 0.712, -0.029),
    ("MMLU-Pro", "Gemini 2.5 Flash"): (0.784, 0.768, +0.017),
    ("MMLU-Pro", "DeepSeek-V3"): (0.762, 0.734, +0.028),
    ("MMLU-Pro", "Qwen 2.5 72B"): (0.761, 0.705, +0.055),
    ("FACTScore", "Claude Opus 4.1"): (0.511, 0.534, -0.022),
    ("FACTScore", "Llama 4 Maverick"): (0.439, 0.442, -0.003),
    ("FACTScore", "Llama 4 Scout"): (0.435, 0.463, -0.028),
    ("FACTScore", "Gemini 2.5 Flash"): (0.629, 0.612, +0.017),
    ("FACTScore", "DeepSeek-V3"): (0.527, 0.516, +0.011),
    ("FACTScore", "Qwen 2.5 72B"): (0.566, 0.472, +0.094),
}
FIG2 = {  # (benchmark, model) -> (rounded % migrating <t,f> → <f,f>, base-<t,f> count n)
    ("TruthfulQA", "Claude Opus 4.1"): (7, 86), ("TruthfulQA", "Llama 4 Maverick"): (6, 70),
    ("TruthfulQA", "Llama 4 Scout"): (11, 72), ("TruthfulQA", "Gemini 2.5 Flash"): (5, 74),
    ("TruthfulQA", "DeepSeek-V3"): (18, 72), ("TruthfulQA", "Qwen 2.5 72B"): (13, 76),
    ("SimpleQA", "Claude Opus 4.1"): (4, 25), ("SimpleQA", "Llama 4 Maverick"): (12, 33),
    ("SimpleQA", "Llama 4 Scout"): (20, 71), ("SimpleQA", "Gemini 2.5 Flash"): (0, 15),
    ("SimpleQA", "DeepSeek-V3"): (25, 16), ("SimpleQA", "Qwen 2.5 72B"): (19, 26),
    ("MMLU-Pro", "Claude Opus 4.1"): (4, 57), ("MMLU-Pro", "Llama 4 Maverick"): (0, 62),
    ("MMLU-Pro", "Llama 4 Scout"): (5, 85), ("MMLU-Pro", "Gemini 2.5 Flash"): (0, 64),
    ("MMLU-Pro", "DeepSeek-V3"): (1, 68), ("MMLU-Pro", "Qwen 2.5 72B"): (3, 70),
    ("FACTScore", "Claude Opus 4.1"): (2, 101), ("FACTScore", "Llama 4 Maverick"): (0, 102),
    ("FACTScore", "Llama 4 Scout"): (7, 100), ("FACTScore", "Gemini 2.5 Flash"): (10, 117),
    ("FACTScore", "DeepSeek-V3"): (12, 67), ("FACTScore", "Qwen 2.5 72B"): (8, 62),
}
FIG3 = {"<t,f>": 4, "<f,t>": 12, "<f,f>": 65, "<t,t>": 3, "partial": 60}  # % ternary abstains
BOOTSTRAP_CI = {"<f,f>": (54.0, 72.8), "<t,t>": (1.2, 4.7), "diff": (51.2, 70.3)}  # §7.2, pp

# ── Semantics (Defs. 7, 10) ─────────────────────────────────────────────────────
_ORD = {"f": 0, "e": 1, "t": 2}
_INV = {v: k for k, v in _ORD.items()}
TRUE_CELLS = {"<t,f>", "<t,e>"}    # F1 \ F1'
FALSE_CELLS = {"<f,t>", "<e,t>"}   # F1' \ F1


def _parse(tv: str) -> tuple[str, str]:
    u, v = tv.strip("<>").split(",")
    return u.strip(), v.strip()


def dec(tv: str) -> str:
    """Definition 10 verdict rule."""
    return "TRUE" if tv in TRUE_CELLS else "FALSE" if tv in FALSE_CELLS else "ABSTAIN"


def box(tvs: list[str]) -> str:
    """Definition 7: [[□p]] = <min u, max v> under f < e < t."""
    us, vs = zip(*(_parse(t) for t in tvs))
    return f"<{_INV[min(_ORD[u] for u in us)]},{_INV[max(_ORD[v] for v in vs)]}>"


def macro_f1(golds: list[bool], preds: list[str]) -> float:
    """Macro-F1 over the TRUE/FALSE classes, on covered (non-abstained) items only."""
    pairs = [(g, p == "TRUE") for g, p in zip(golds, preds) if p in ("TRUE", "FALSE")]
    f1s = []
    for cls in (True, False):
        tp = sum(1 for g, p in pairs if g == cls and p == cls)
        fp = sum(1 for g, p in pairs if g != cls and p == cls)
        fn = sum(1 for g, p in pairs if g == cls and p != cls)
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(f1s) / 2


def _bucket(tv: str) -> str:
    return tv if tv in ("<t,f>", "<f,t>", "<f,f>", "<t,t>") else "partial"


def _ternary_abstains(r: dict) -> bool:
    return r["ternary_prediction"] not in ("TRUE", "FALSE")


# ── Fixtures ────────────────────────────────────────────────────────────────────
def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def proper() -> dict:
    d = {}
    for mk, ml in MODELS:
        for pk, _, bl in BENCHMARKS:
            d[(bl, ml)] = _load(PROPER / f"{pk}_{mk}_n3_proper_results.json")["detailed_results"]
    log.info("Loaded %d proper cells", len(d))
    return d


@pytest.fixture(scope="module")
def modal() -> dict:
    d = {}
    for mk, ml in MODELS:
        for _, mok, bl in BENCHMARKS:
            d[(bl, ml)] = _load(MODAL / f"{mok}_{mk}_n3_modal_results.json")
    log.info("Loaded %d modal cells", len(d))
    return d


@pytest.fixture(scope="module")
def all_proper(proper) -> list[dict]:
    return [r for rs in proper.values() for r in rs]


@pytest.fixture(scope="module")
def abstain_counts(proper) -> dict:
    """(cell) -> bucket -> [ternary abstains, total]."""
    out = {}
    for key, rs in proper.items():
        c = defaultdict(lambda: [0, 0])
        for r in rs:
            b = _bucket(r["bilateral_tv"])
            c[b][0] += _ternary_abstains(r)
            c[b][1] += 1
        out[key] = c
    return out


# ── §7 setup ────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("cell", CELLS, ids=lambda c: f"{c[0]}-{c[1]}")
def test_setup_n250_balanced(proper, cell) -> None:
    rs = proper[cell]
    assert len(rs) == 250
    assert sum(r["ground_truth"] for r in rs) == 125


@pytest.mark.parametrize("bench", [b for _, _, b in BENCHMARKS])
def test_setup_same_assertions_across_models(proper, bench) -> None:
    id_sets = {frozenset(r["assertion_id"] for r in proper[(bench, m)]) for _, m in MODELS}
    assert len(id_sets) == 1


@pytest.mark.parametrize("cell", CELLS, ids=lambda c: f"{c[0]}-{c[1]}")
def test_setup_paraphrases(modal, proper, cell) -> None:
    d = modal[cell]
    k = 2 if cell[0] == "FACTScore" else 3
    assert d["variation_model"] == "claude-opus-4-6"
    assert {len(r["variants"]) for r in d["detailed_results"]} == {k}
    assert {r["assertion_id"] for r in d["detailed_results"]} == {r["assertion_id"] for r in proper[cell]}


# ── Def. 7 ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("cell", CELLS, ids=lambda c: f"{c[0]}-{c[1]}")
def test_def7_modal_aggregation(modal, cell) -> None:
    bad = [r["assertion_id"] for r in modal[cell]["detailed_results"]
           if box([r["src_bilateral"]] + r["variant_bilaterals"]) != r["modal_bilateral"]]
    assert not bad, f"stored [[□p]] disagrees with <min u, max v> for {bad[:5]}"


def test_modal_s0_is_separate_elicitation(proper, modal) -> None:
    """The modal run re-elicited s₀ (samples=1), so its s₀ values differ from the
    proper run's 3-vote majority on 579 of 6,000 assertions (Qwen: 0). Frozen data,
    so the count is exact; a change means a result file was overwritten."""
    per_model = defaultdict(int)
    for (bench, model) in CELLS:
        p = {r["assertion_id"]: r["bilateral_tv"] for r in proper[(bench, model)]}
        per_model[model] += sum(1 for r in modal[(bench, model)]["detailed_results"]
                                if r["src_bilateral"] != p[r["assertion_id"]])
    log.info("proper-vs-modal s₀ disagreements by model: %s", dict(per_model))
    assert sum(per_model.values()) == 579
    assert per_model["Qwen 2.5 72B"] == 0


# ── Table 1 ─────────────────────────────────────────────────────────────────────
def _table1_cell(rs: list[dict]) -> tuple[float, float]:
    golds = [r["ground_truth"] for r in rs]
    bil = macro_f1(golds, [dec(r["bilateral_tv"]) for r in rs])
    tern = macro_f1(golds, [r["ternary_prediction"] for r in rs])
    return bil, tern


@pytest.mark.parametrize("cell", CELLS, ids=lambda c: f"{c[0]}-{c[1]}")
def test_table1_cell(proper, cell) -> None:
    bil, tern = _table1_cell(proper[cell])
    pb, pt, pd = TABLE1[cell]
    log.info("Table 1 %s: bil %.4f tern %.4f Δ %+.4f", cell, bil, tern, bil - tern)
    assert round(bil, 3) == pytest.approx(pb, abs=1e-3)
    assert round(tern, 3) == pytest.approx(pt, abs=1e-3)
    assert bil - tern == pytest.approx(pd, abs=1.5e-3)


def test_table1_summary(proper) -> None:
    ad = [abs(b - t) for b, t in (_table1_cell(proper[c]) for c in CELLS)]
    log.info("|Δ| median %.4f mean %.4f max %.4f", statistics.median(ad), statistics.mean(ad), max(ad))
    assert round(statistics.median(ad), 3) == 0.028
    assert round(statistics.mean(ad), 3) == 0.048
    assert round(max(ad), 3) == 0.184


# ── Fig. 2 ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("cell", CELLS, ids=lambda c: f"{c[0]}-{c[1]}")
def test_fig2_migration(modal, cell) -> None:
    base = [r for r in modal[cell]["detailed_results"] if r["src_bilateral"] == "<t,f>"]
    migrated = sum(1 for r in base if r["modal_bilateral"] == "<f,f>")
    pct, n = FIG2[cell]
    log.info("Fig. 2 %s: %d/%d", cell, migrated, len(base))
    assert len(base) == n
    assert round(100 * migrated / n) == pct


# ── Fig. 3 and the §7.2 bootstrap ───────────────────────────────────────────────
@pytest.mark.parametrize("bucket", list(FIG3))
def test_fig3_ternary_abstention(abstain_counts, bucket) -> None:
    a = sum(c[bucket][0] for c in abstain_counts.values())
    n = sum(c[bucket][1] for c in abstain_counts.values())
    log.info("Fig. 3 %s: %d/%d = %.2f%%", bucket, a, n, 100 * a / n)
    assert round(100 * a / n) == FIG3[bucket]


def test_s72_bootstrap_ci(abstain_counts) -> None:
    """Cluster bootstrap over the 24 cells, 10,000 resamples, seed 42, percentile 95% CI."""
    rng = random.Random(42)
    keys = list(abstain_counts)
    draws = {"<f,f>": [], "<t,t>": [], "diff": []}
    for _ in range(10_000):
        s = [abstain_counts[rng.choice(keys)] for _ in keys]
        ff = 100 * sum(c["<f,f>"][0] for c in s) / sum(c["<f,f>"][1] for c in s)
        n_tt = sum(c["<t,t>"][1] for c in s)
        tt = 100 * sum(c["<t,t>"][0] for c in s) / n_tt if n_tt else 0.0
        draws["<f,f>"].append(ff)
        draws["<t,t>"].append(tt)
        draws["diff"].append(ff - tt)
    for name, xs in draws.items():
        xs.sort()
        lo, hi = xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]
        log.info("bootstrap %s: [%.2f, %.2f]", name, lo, hi)
        assert (round(lo, 1), round(hi, 1)) == pytest.approx(BOOTSTRAP_CI[name], abs=0.15)


# ── §7.2 / §8 shares and the confidence cross-check ─────────────────────────────
def test_s72_tt_share(proper, all_proper) -> None:
    per_cell = [100 * sum(r["bilateral_tv"] == "<t,t>" for r in proper[c]) / 250 for c in CELLS]
    pooled = 100 * sum(r["bilateral_tv"] == "<t,t>" for r in all_proper) / len(all_proper)
    log.info("<t,t> share: min %.1f%% max %.1f%% pooled %.2f%%", min(per_cell), max(per_cell), pooled)
    assert round(min(per_cell), 1) == 0.4
    assert round(max(per_cell)) == 28
    assert round(pooled, 1) == 6.2


def test_s8_e_bearing_share(all_proper) -> None:
    share = 100 * sum("e" in _parse(r["bilateral_tv"]) for r in all_proper) / len(all_proper)
    log.info("e-bearing share: %.2f%%", share)
    assert round(share, 1) == 11.8


@pytest.mark.parametrize("tv,expected", [("<t,t>", 0.78), ("<t,f>", 0.86), ("<f,f>", 0.49)])
def test_s72_confidence_cross_check(all_proper, tv, expected) -> None:
    xs = [r["confidence_score"] for r in all_proper
          if r["bilateral_tv"] == tv and r.get("confidence_score") is not None]
    log.info("mean confidence %s: %.4f (n=%d)", tv, statistics.mean(xs), len(xs))
    assert round(statistics.mean(xs), 2) == expected


# ── §6.3 worked examples ────────────────────────────────────────────────────────
def _qwen_simpleqa(modal, needle: str) -> dict:
    hits = [r for r in modal[("SimpleQA", "Qwen 2.5 72B")]["detailed_results"]
            if needle in r["source_question"]]
    assert len(hits) == 1, f"expected one assertion matching {needle!r}, got {len(hits)}"
    return hits[0]


def test_s63_example1_prompt_agnostic(modal) -> None:
    r = _qwen_simpleqa(modal, "matrix algebra")
    assert r["src_bilateral"] == "<t,f>"
    assert r["variant_bilaterals"] == ["<t,f>", "<t,f>", "<t,f>"]
    assert r["modal_bilateral"] == "<t,f>"


def test_s63_example2_prompt_sensitive(modal) -> None:
    r = _qwen_simpleqa(modal, "Mark Scout")
    assert r["src_bilateral"] == "<t,f>"
    assert r["variant_bilaterals"] == ["<f,f>", "<t,f>", "<t,f>"]
    assert r["modal_bilateral"] == "<f,f>"
    # Modal field at the variant nodes: succ(s_i) = {s_i, s_0} under the symmetric back-edge.
    field = [box([v, r["src_bilateral"]]) for v in r["variant_bilaterals"]]
    assert field == ["<f,f>", "<t,f>", "<t,f>"]
