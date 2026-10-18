#!/usr/bin/env python3
"""Build the item lists for the November 2026 revision experiments (E1-0/a/b, E2-0, E3).

Deterministic from the frozen ``results/2026_03/proper/*_n3_proper_results.json`` files.
Selection follows the specification "BBL Revision Experiments: E1-E3 Specification", §3:

* E1 treatment  : every <t,t> item for the five retained models (Opus 4.1 excluded, retired 2026-08-05).
* E1 controls   : per model, min(2 x #<t,t>, pool) items from the model's <t,f> pool and the same from
                  its <f,f> pool, stratified by benchmark in proportion to that model's treatment mix,
                  seed 42. <e,e> items are never drawn (API failures in the 2026 run).
* E1-0 / E2-0   : the first 100 <t,f> and first 100 <f,f> controls per model, in seeded order,
                  flagged ``baseline_subset = true``; all treatment items are in the subset.
* E2            : all (model, assertion) pairs for the five models -- no list needed; the evaluator
                  reads the frozen results directly (``--mode e2`` with no ``--items``).
* E3            : per model, 50 robust-TRUE (<t,f>, gold TRUE, ternary TRUE) and 50 robust-FALSE
                  (<f,t>, gold FALSE, ternary FALSE), stratified by benchmark, seed 42.

Outputs (JSONL, one record per (model, assertion)):
    datasets/revision_2026_11/e1_items.jsonl
    datasets/revision_2026_11/e3_items.jsonl
    datasets/revision_2026_11/MANIFEST.json   (SHA-256 of each list)

Usage:
    python datasets/revision_2026_11/build_item_lists.py
"""
from __future__ import annotations

import collections
import glob
import hashlib
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "2026_03" / "proper"
OUT = Path(__file__).resolve().parent

SEED = 42
CONTROL_MULTIPLIER = 2          # controls per state = multiplier x #<t,t>, capped by pool
BASELINE_PER_STATE = 100        # E1-0 / E2-0 control subset per model per state
E3_PER_POLARITY = 50

# OpenRouter ids of the five retained models, keyed by the file-safe stem used in result filenames.
MODELS = {
    "meta-llama_llama-4-maverick": "meta-llama/llama-4-maverick",
    "meta-llama_llama-4-scout": "meta-llama/llama-4-scout",
    "google_gemini-2.5-flash": "google/gemini-2.5-flash",
    "deepseek_deepseek-chat": "deepseek/deepseek-chat",
    "qwen_qwen-2.5-72b-instruct": "qwen/qwen-2.5-72b-instruct",
}
BENCHMARKS = ["truthfulqa", "simpleqa", "mmlu-pro", "factscore"]


def load_cells() -> dict[tuple[str, str], list[dict]]:
    cells: dict[tuple[str, str], list[dict]] = {}
    for stem, model_id in MODELS.items():
        for bench in BENCHMARKS:
            path = RESULTS / f"{bench}_{stem}_n3_proper_results.json"
            if not path.exists():
                sys.exit(f"missing frozen result file: {path}")
            data = json.loads(path.read_text())
            assert data["model"] == model_id, (path, data["model"])
            rows = []
            for r in data["detailed_results"]:
                rows.append({
                    "model": model_id,
                    "benchmark": bench,
                    "assertion_id": r["assertion_id"],
                    "assertion_text": r["assertion_text"],
                    "source_question": r.get("source_question", ""),
                    "ground_truth": bool(r["ground_truth"]),
                    "orig_bilateral_tv": r["bilateral_tv"],
                    "orig_ternary_prediction": r["ternary_prediction"],
                })
            cells[(model_id, bench)] = rows
    return cells


def stratified_draw(pool_by_bench: dict[str, list[dict]], target_mix: dict[str, int],
                    n_total: int, rng: random.Random) -> list[dict]:
    """Draw n_total items from pool_by_bench in proportion to target_mix (a benchmark -> count map).

    Largest-remainder apportionment, capped by each benchmark's pool; shortfall is redistributed to
    benchmarks with remaining pool, in a fixed order, so the draw is deterministic and total is met
    whenever the overall pool allows.
    """
    total_mix = sum(target_mix.values()) or 1
    quotas = {b: n_total * target_mix.get(b, 0) / total_mix for b in BENCHMARKS}
    alloc = {b: min(int(quotas[b]), len(pool_by_bench.get(b, []))) for b in BENCHMARKS}
    remainder = sorted(BENCHMARKS, key=lambda b: -(quotas[b] - int(quotas[b])))
    short = n_total - sum(alloc.values())
    progress = True
    while short > 0 and progress:  # round-robin over benchmarks with remaining pool
        progress = False
        for b in remainder:
            if short > 0 and alloc[b] < len(pool_by_bench.get(b, [])):
                alloc[b] += 1
                short -= 1
                progress = True
    drawn = []
    for b in BENCHMARKS:
        pool = sorted(pool_by_bench.get(b, []), key=lambda r: r["assertion_id"])
        drawn.extend(rng.sample(pool, alloc[b]))
    return drawn


def main() -> None:
    global OUT
    if len(sys.argv) == 3 and sys.argv[1] == "--out":   # used by tests to check determinism
        OUT = Path(sys.argv[2])
    cells = load_cells()
    rng = random.Random(SEED)
    e1_records, e3_records = [], []
    summary = collections.OrderedDict()

    for model_id in MODELS.values():
        rows = [r for (m, _), rs in cells.items() if m == model_id for r in rs]
        by_tv: dict[str, dict[str, list[dict]]] = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rows:
            by_tv[r["orig_bilateral_tv"]][r["benchmark"]].append(r)

        # ---- E1 treatment: every <t,t> item ----
        treatment = sorted((r for b in by_tv["<t,t>"].values() for r in b),
                           key=lambda r: (r["benchmark"], r["assertion_id"]))
        mix = collections.Counter(r["benchmark"] for r in treatment)
        n_ctrl = CONTROL_MULTIPLIER * len(treatment)

        # ---- E1 controls, matched within model, stratified by the treatment's benchmark mix ----
        ctrl_tf = stratified_draw(by_tv["<t,f>"], mix, min(n_ctrl, sum(len(v) for v in by_tv["<t,f>"].values())), rng)
        ctrl_ff = stratified_draw(by_tv["<f,f>"], mix, min(n_ctrl, sum(len(v) for v in by_tv["<f,f>"].values())), rng)
        rng.shuffle(ctrl_tf)
        rng.shuffle(ctrl_ff)

        for r in treatment:
            e1_records.append({**r, "condition": "treatment", "baseline_subset": True})
        for i, r in enumerate(ctrl_tf):
            e1_records.append({**r, "condition": "control_tf", "baseline_subset": i < BASELINE_PER_STATE})
        for i, r in enumerate(ctrl_ff):
            e1_records.append({**r, "condition": "control_ff", "baseline_subset": i < BASELINE_PER_STATE})

        # ---- E3: robust items by the model's own prior answers ----
        robust_true = collections.defaultdict(list)
        robust_false = collections.defaultdict(list)
        for r in rows:
            if r["orig_bilateral_tv"] == "<t,f>" and r["ground_truth"] and r["orig_ternary_prediction"] == "TRUE":
                robust_true[r["benchmark"]].append(r)
            if r["orig_bilateral_tv"] == "<f,t>" and not r["ground_truth"] and r["orig_ternary_prediction"] == "FALSE":
                robust_false[r["benchmark"]].append(r)
        even = {b: 1 for b in BENCHMARKS}
        for r in stratified_draw(robust_true, even, E3_PER_POLARITY, rng):
            e3_records.append({**r, "polarity": "robust_true", "probe": "refutation"})
        for r in stratified_draw(robust_false, even, E3_PER_POLARITY, rng):
            e3_records.append({**r, "polarity": "robust_false", "probe": "verification"})

        summary[model_id] = {
            "treatment": len(treatment),
            "control_tf": len(ctrl_tf),
            "control_ff": len(ctrl_ff),
            "baseline_subset": len(treatment) + min(BASELINE_PER_STATE, len(ctrl_tf)) + min(BASELINE_PER_STATE, len(ctrl_ff)),
            "e3_robust_true": sum(1 for r in e3_records if r["model"] == model_id and r["polarity"] == "robust_true"),
            "e3_robust_false": sum(1 for r in e3_records if r["model"] == model_id and r["polarity"] == "robust_false"),
            "excluded_ee": sum(len(v) for v in by_tv["<e,e>"].values()),
        }

    OUT.mkdir(parents=True, exist_ok=True)
    # MANIFEST.json follows the repo convention checked by tests/test_manifest_integrity.py
    # ({filename: {sha256, size_bytes}}); selection parameters and counts go to SELECTION.json.
    manifest: dict[str, dict] = {}
    selection = {"seed": SEED, "control_multiplier": CONTROL_MULTIPLIER, "baseline_per_state": BASELINE_PER_STATE,
                 "e3_per_polarity": E3_PER_POLARITY, "source": "results/2026_03/proper", "records": {}, "summary": summary}
    for name, recs in (("e1_items.jsonl", e1_records), ("e3_items.jsonl", e3_records)):
        path = OUT / name
        with path.open("w") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}
        selection["records"][name] = len(recs)
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "SELECTION.json").write_text(json.dumps(selection, indent=2) + "\n")

    print(json.dumps({"manifest": manifest, "selection": selection}, indent=2))


if __name__ == "__main__":
    main()
