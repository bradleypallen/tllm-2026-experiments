"""
Tests for the E4 / E4-0 harness (docs/revision_2026_11_E4_spec_addition.md).

Guards: the item list is the deterministic function of the frozen March results that the spec
describes; the paraphrases are the cached ones the March run used; the prompts and system prompt
are the frozen ones; the issue order interleaves the two arms; aggregation is Definition 7; and
the evaluator runs end to end offline.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
E4_DATA = REPO_ROOT / "datasets" / "revision_2026_11_e4"
sys.path.insert(0, str(REPO_ROOT / "evaluators"))

import revision_2026_11_prompts as P  # noqa: E402
import revision_e4_evaluator as E4  # noqa: E402

EXPECTED_BY_MODEL = {"meta-llama/llama-4-maverick": 267, "meta-llama/llama-4-scout": 328,
                     "google/gemini-2.5-flash": 270, "deepseek/deepseek-chat": 223,
                     "qwen/qwen-2.5-72b-instruct": 234}
EXPECTED_BY_BENCH = {"truthfulqa": 364, "simpleqa": 161, "mmlu-pro": 349, "factscore": 448}
PLANNED_CALLS = 16_716


def _items() -> list[dict]:
    return [json.loads(l) for l in (E4_DATA / "e4_items.jsonl").read_text().splitlines() if l.strip()]


def test_item_list_matches_manifest_and_is_deterministic(tmp_path) -> None:
    subprocess.run([sys.executable, str(E4_DATA / "build_e4_items.py"), "--out-dir", str(tmp_path)],
                   check=True, capture_output=True)
    committed = (E4_DATA / "e4_items.jsonl").read_bytes()
    assert (tmp_path / "e4_items.jsonl").read_bytes() == committed
    manifest = json.loads((E4_DATA / "MANIFEST.json").read_text())["e4_items.jsonl"]
    assert manifest == {"sha256": hashlib.sha256(committed).hexdigest(), "size_bytes": len(committed)}


def test_item_list_shape_and_counts() -> None:
    items = _items()
    assert len(items) == 1322
    assert Counter(r["model"] for r in items) == EXPECTED_BY_MODEL
    assert Counter(r["benchmark"] for r in items) == EXPECTED_BY_BENCH
    assert {r["orig_modal_src"] for r in items} == {"<t,f>"}
    assert len({(r["model"], r["assertion_id"]) for r in items}) == len(items)
    for r in items:
        assert len(r["variants"]) == (2 if r["benchmark"] == "factscore" else 3)
        assert r["source_question"] and all(r["variants"]) and r["source_question"] not in r["variants"]
    assert sum(2 * len(E4.schedule(r)) for r in items) == PLANNED_CALLS


def test_paraphrases_are_the_cached_ones() -> None:
    cache = {b: json.loads((REPO_ROOT / "variant_cache" / f"{b}_n3_variants.json").read_text())["variants"]
             for b in EXPECTED_BY_BENCH}
    for r in _items():
        c = cache[r["benchmark"]][r["assertion_id"]]
        assert c["variants"] == r["variants"] and c["source_question"] == r["source_question"]


def test_prompts_are_the_frozen_ones() -> None:
    """The prompts module is byte-identical to the one the October run recorded, and its system
    prompt is the one the March paraphrase run used (evaluators/modal_evaluator.py)."""
    meta = json.loads(next((REPO_ROOT / "results" / "2026_11" / "e2").glob("*.meta.json")).read_text())
    src = (REPO_ROOT / "evaluators" / "revision_2026_11_prompts.py").read_bytes()
    assert hashlib.sha256(src).hexdigest() == meta["prompt_hashes"]["revision_2026_11_prompts.py"]
    assert P.BILATERAL_SYSTEM_PROMPT in (REPO_ROOT / "evaluators" / "modal_evaluator.py").read_text()


def test_schedule_interleaves_arms() -> None:
    item = {"source_question": "Q?", "variants": ["A?", "B?", "C?"]}
    assert E4.schedule(item) == [("base", 0, "Q?"), ("rep", 1, "Q?"), ("par", 1, "A?"),
                                 ("rep", 2, "Q?"), ("par", 2, "B?"), ("rep", 3, "Q?"), ("par", 3, "C?")]


def test_box_is_definition_7() -> None:
    assert E4.box(["<t,f>", "<t,f>"]) == "<t,f>"
    assert E4.box(["<t,f>", "<f,f>", "<t,f>"]) == "<f,f>"
    assert E4.box(["<t,f>", "<t,t>"]) == "<t,t>"
    assert E4.box(["<t,f>", "<e,f>"]) == "<e,f>"      # f < e < t: min(t, e) = e
    assert E4.box(["<t,f>", "<t,e>"]) == "<t,e>"      # max(f, e) = e
    assert E4.box(["<t,f>", "<e,e>", "<f,t>"]) == "<f,t>"


def test_offline_dry_run(tmp_path) -> None:
    subprocess.run([sys.executable, str(REPO_ROOT / "evaluators" / "revision_e4_evaluator.py"),
                    "--model", "qwen/qwen-2.5-72b-instruct", "--dry-run", "--limit", "4", "--out-dir", str(tmp_path)],
                   check=True, capture_output=True)
    out = tmp_path / "dry_run" / "e4_qwen_qwen-2.5-72b-instruct.jsonl"
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    meta = json.loads(out.with_name("e4_qwen_qwen-2.5-72b-instruct.meta.json").read_text())
    assert len(recs) == 4 == meta["n_records"] and meta["finished_at"] and meta["dry_run"] is True
    for r in recs:
        arms = [s["arm"] for s in r["situations"]]
        assert arms == ["base"] + ["rep", "par"] * r["k"]
        base = r["situations"][0]["tv"]
        # The mock is a deterministic function of the prompt, so identical repeats reproduce the base.
        assert all(s["tv"] == base for s in r["situations"] if s["arm"] == "rep")
        assert r["box_rep"] == base == r["base_tv"]
        assert r["box_par"] == E4.box([base] + [s["tv"] for s in r["situations"] if s["arm"] == "par"])
        for s in r["situations"]:
            assert set(s["verification"]) >= {"raw", "parsed", "provider", "model_returned", "error"}


def test_provider_pins_are_octobers_except_maverick() -> None:
    """Spec §10 (amendment of 2026-11-11): DigitalOcean withdrew Llama 4 Maverick; E4 pins Parasail."""
    import revision_2026_11_evaluator as R
    assert E4.E4_PROVIDER["meta-llama/llama-4-maverick"] == "parasail"
    assert {m: v for m, v in E4.E4_PROVIDER.items() if m != "meta-llama/llama-4-maverick"} == \
        {m: v for m, v in R.DEFAULT_PROVIDER.items() if m != "meta-llama/llama-4-maverick"}
    assert R.DEFAULT_PROVIDER["meta-llama/llama-4-maverick"] == "digitalocean"   # October evaluator unmodified


# ── E4 results: headline counts re-derived from raw text, independently of analysis/revision_e4.py ──
E4_POOLED = {"n": 1165, "rep_changed": 10, "par_changed": 186, "rep_ff": 5, "par_ff": 63}


def test_e4_results_complete_and_headline_counts() -> None:
    res = REPO_ROOT / "results" / "2026_11_e4"
    order = {"f": 0, "e": 1, "t": 2}
    tot: Counter = Counter()
    n_elic = 0
    for model, n_items in EXPECTED_BY_MODEL.items():
        stem = "e4_" + model.replace("/", "_")
        meta = json.loads((res / f"{stem}.meta.json").read_text())
        recs = [json.loads(l) for l in (res / f"{stem}.jsonl").read_text().splitlines() if l.strip()]
        assert len(recs) == n_items == meta["n_items"] and meta["finished_at"]
        assert meta["api_error_rate"] == 0.0 and meta["allow_fallbacks"] is False
        assert len(meta["providers_seen"]) == 1 and meta["models_returned"] == [model]
        for r in recs:
            tv = {"base": [], "rep": [], "par": []}
            for s in r["situations"]:
                n_elic += 2
                u = P.component(P.parse_single(s["verification"]["raw"] or "", P.VERIFICATION_TOKENS))
                v = P.component(P.parse_single(s["refutation"]["raw"] or "", P.REFUTATION_TOKENS))
                tv[s["arm"]].append((u, v))
            if tv["base"][0] != ("t", "f"):
                continue
            agg = lambda xs: ("fet"[min(order[x[0]] for x in xs)], "fet"[max(order[x[1]] for x in xs)])
            rep, par = agg(tv["base"] + tv["rep"]), agg(tv["base"] + tv["par"])
            tot["n"] += 1
            tot["rep_changed"] += rep != ("t", "f"); tot["par_changed"] += par != ("t", "f")
            tot["rep_ff"] += rep == ("f", "f"); tot["par_ff"] += par == ("f", "f")
    assert n_elic == PLANNED_CALLS
    assert dict(tot) == E4_POOLED
    derived = json.loads((REPO_ROOT / "results" / "derived" / "2026_11_e4" / "T-E4_paraphrase_vs_repeat.json").read_text())
    pooled = derived["T-E4b_rates"]["pooled"]
    assert pooled["n"] == E4_POOLED["n"]
    assert pooled["any_change"]["repeat"] == round(10 / 1165, 4) and pooled["any_change"]["paraphrase"] == round(186 / 1165, 4)
    assert derived["P3"]["holds"] is True
