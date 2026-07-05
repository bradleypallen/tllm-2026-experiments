"""
Definition-16 paraphrase-fidelity audit -- Phases 2-3: draw the seeded edge sample
and emit the human labeling file.

Protocol (must match the paper sentence's claim):
  - Population: all <base, variant> edges in variant_cache/ (each assertion
    contributes one edge per cached variant).
  - Stratification: 25 edges per benchmark, 100 total.
  - Sampling: uniform without replacement within each benchmark, single
    random.Random(42) over the canonically sorted edge list -- sorted by
    (benchmark, assertion_id, variant_index) BEFORE sampling, so the draw does
    not depend on file order. Benchmarks are processed in sorted order.
  - Constraint: at most one edge per assertion; a draw hitting an
    already-sampled assertion is rejected (counted) and redrawn.
  - Row order in the labeling CSV is shuffled with an independent
    random.Random(42) so benchmark blocks do not bias the human pass.

Machine involvement is limited to `suggested_transformation_types` -- a
deterministic HEURISTIC suggestion of Definition-16 transformation types
(documented in CACHE_NOTES.md). No machine fidelity opinion is emitted;
`human_pass`, `human_transformation_types`, `human_failure_mode`, `notes`
are blank for the author.

Outputs (all under audit/):
  - sample_manifest.json   seed, sort key, per-benchmark population sizes,
                           the 100 sampled edge ids, rejection count
  - edge_audit_sample.csv  one row per sampled edge, UTF-8, byte-stable

Deterministic: re-running reproduces both outputs byte-identically.

Run from the repo root:  python3 audit/make_audit_sample.py
"""
from __future__ import annotations
import csv
import json
import logging
import random
import re
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [def16-audit] %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "variant_cache"
OUT_DIR = Path(__file__).resolve().parent
OUT_MANIFEST = OUT_DIR / "sample_manifest.json"
OUT_CSV = OUT_DIR / "edge_audit_sample.csv"

SEED = 42
PER_BENCH = 25
BENCHES = ["factscore", "mmlu-pro", "simpleqa", "truthfulqa"]  # canonical sorted order
SORT_KEY = "(benchmark, assertion_id, variant_index)"


# ── deterministic transformation-type heuristic (SUGGESTION ONLY) ─────────────
_word_re = re.compile(r"[^\w\s]")


def _tokens(s: str) -> list[str]:
    return _word_re.sub(" ", s.lower()).split()


def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    it = iter(haystack)
    return all(tok in it for tok in needle)


def suggest_types(base: str, variant: str) -> str:
    """Heuristic Definition-16 typing. Canonical order: lexical, reorder,
    elaboration, formatting. Documented in CACHE_NOTES.md; suggestion only."""
    bt, vt = _tokens(base), _tokens(variant)
    if bt == vt:
        return "formatting" if base != variant else ""
    types: list[str] = []
    b_multi, v_multi = sorted(bt), sorted(vt)
    if b_multi == v_multi:
        return "reorder"
    b_set, v_set = set(bt), set(vt)
    removed, added = b_set - v_set, v_set - b_set
    common_in_base = [t for t in bt if t in v_set]
    if removed:
        types.append("lexical")
        if not _is_subsequence(common_in_base, vt):
            types.append("reorder")
        if len(vt) - len(bt) >= 4:
            types.append("elaboration")
    else:  # pure additions
        if not _is_subsequence(bt, vt):
            types.append("reorder")
        types.append("elaboration")
    order = {"lexical": 0, "reorder": 1, "elaboration": 2, "formatting": 3}
    return ",".join(sorted(set(types), key=order.__getitem__))


# ── phases 2-3 ────────────────────────────────────────────────────────────────
def load_edges() -> dict[str, list[dict]]:
    """Canonically sorted edge list per benchmark."""
    per_bench: dict[str, list[dict]] = {}
    for bench in BENCHES:
        blob = json.loads((CACHE / f"{bench}_n3_variants.json").read_text())
        edges = []
        for aid in sorted(blob["variants"].keys()):
            entry = blob["variants"][aid]
            for vi, vtext in enumerate(entry["variants"]):
                edges.append({
                    "benchmark": bench, "assertion_id": aid, "variant_index": vi,
                    "base_text": entry["source_question"], "variant_text": vtext,
                })
        per_bench[bench] = edges  # already sorted: aid asc, vi asc
        log.info("%-11s population: %d edges (%d assertions)",
                 bench, len(edges), len(blob["variants"]))
    return per_bench


def draw_sample(per_bench: dict[str, list[dict]]):
    rng = random.Random(SEED)
    accepted: list[dict] = []
    rejections = {b: 0 for b in BENCHES}
    for bench in BENCHES:  # canonical benchmark order
        pool = list(per_bench[bench])
        used_assertions: set[str] = set()
        taken = 0
        while taken < PER_BENCH:
            edge = pool.pop(rng.randrange(len(pool)))  # without replacement
            if edge["assertion_id"] in used_assertions:
                rejections[bench] += 1
                continue
            used_assertions.add(edge["assertion_id"])
            accepted.append(edge)
            taken += 1
        log.info("%-11s sampled %d edges (%d same-assertion rejections)",
                 bench, taken, rejections[bench])
    return accepted, rejections


def main():
    per_bench = load_edges()
    accepted, rejections = draw_sample(per_bench)

    # Independent shuffle so benchmark blocks don't bias the human pass.
    shuffled = list(accepted)
    random.Random(SEED).shuffle(shuffled)

    manifest = {
        "seed": SEED,
        "sort_key": SORT_KEY,
        "per_benchmark_population_edges": {b: len(per_bench[b]) for b in BENCHES},
        "per_benchmark_sample_size": PER_BENCH,
        "constraint": "at most one edge per assertion (reject and redraw)",
        "rejection_count": {**rejections, "total": sum(rejections.values())},
        "row_order": "independent random.Random(42) shuffle of the accepted edges",
        "sampled_edges": [
            {"row_id": i + 1, "benchmark": e["benchmark"],
             "assertion_id": e["assertion_id"], "variant_index": e["variant_index"]}
            for i, e in enumerate(shuffled)
        ],
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    log.info("wrote %s", OUT_MANIFEST)

    fieldnames = ["row_id", "benchmark", "assertion_id", "variant_index",
                  "base_text", "variant_text", "suggested_transformation_types",
                  "human_pass", "human_transformation_types",
                  "human_failure_mode", "notes"]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL,
                           lineterminator="\n")
        w.writeheader()
        for i, e in enumerate(shuffled):
            w.writerow({
                "row_id": i + 1,
                "benchmark": e["benchmark"],
                "assertion_id": e["assertion_id"],
                "variant_index": e["variant_index"],
                "base_text": e["base_text"],
                "variant_text": e["variant_text"],
                "suggested_transformation_types":
                    suggest_types(e["base_text"], e["variant_text"]),
                "human_pass": "",
                "human_transformation_types": "",
                "human_failure_mode": "",
                "notes": "",
            })
    log.info("wrote %s (%d rows; judgment columns BLANK by design)",
             OUT_CSV, len(shuffled))


if __name__ == "__main__":
    main()
