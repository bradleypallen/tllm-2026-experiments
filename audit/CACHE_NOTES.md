# Variant-cache characterization (Definition-16 fidelity audit, Phase 1)

## Location and format

The paraphrase variant cache is `variant_cache/{benchmark}_n3_variants.json` for the four
benchmarks `truthfulqa`, `simpleqa`, `mmlu-pro`, `factscore` (integrity hashes in
`variant_cache/MANIFEST.json`; matches the artifact the paper describes: generated once
per dataset by Claude Opus 4.6, seed 42, cached across evaluation models).

Each file is a JSON object with two keys:

- `metadata` — benchmark, `n_variants`, `variation_model` (`claude-opus-4-6`),
  `dataset_path`, `total_assertions`, `seed`, generation timestamps,
  `total_generated`, `total_skipped`.
- `variants` — a dict keyed by assertion id. Each entry has:
  - `source_question` (string) — the **base text** (s₀),
  - `variants` (list of strings) — the **variant texts**, index = variant_index.

There is exactly one candidate cache per benchmark; no structural ambiguity.

## Record counts and edge populations

| benchmark  | assertions in cache | variants per assertion | edges (population) |
|------------|--------------------:|------------------------|-------------------:|
| factscore  | 250                 | **2** (all entries)    | 500                |
| mmlu-pro   | 250                 | 3 (all entries)        | 750                |
| simpleqa   | 250                 | 3 (all entries)        | 750                |
| truthfulqa | **268**             | 3 (all entries)        | 804                |

No empty variant strings, no duplicate variants within an assertion, no variant identical
to its base text, in any benchmark.

## Anomalies (reported per spec; audit proceeds — structure is unambiguous)

**A1 — truthfulqa cache is a superset of the run sample.** The cache holds 268 assertions
while its own `metadata.total_assertions` = 250 (`total_generated` = 268,
`total_skipped` = 32). The evaluators draw their N = 250 (seed 42) sample independently,
so a cache superset is benign for the experiments. Per the audit spec, the sampling
population is **the cache**, i.e. all 804 truthfulqa edges.

**A2 — factscore has 2 variants per assertion, not 3 (all 250 entries), despite
`metadata.n_variants` = 3.** Cross-checked against the frozen modal results: every one of
the 250 factscore records in
`results/2026_03/modal/factscore_complete_claude-opus-4-1-20250805_n3_modal_results.json`
has `len(variants)` = `len(variant_bilaterals)` = 2, while the other three benchmarks have
3 everywhere. So the modal aggregation for factscore ran over s₀ + 2 variants.
**Paper-facing:** §7's methods sentence ("generate n = 3 paraphrase variants per
assertion") is inaccurate for FACTScore; flagged for the author's decision (editing the
paper is out of scope for this audit).

## Sampling summary (Phase 2; details in `sample_manifest.json`)

Population = all 2,804 ⟨base, variant⟩ edges; canonical sort by
(benchmark, assertion_id, variant_index); single `random.Random(42)`; 25 edges per
benchmark, uniform without replacement, at most one edge per assertion (rejected draws
redrawn). Rejections: 1 per benchmark, 4 total. Row order in the labeling CSV is an
independent `random.Random(42)` shuffle. Re-running `audit/make_audit_sample.py`
reproduces `edge_audit_sample.csv` and `sample_manifest.json` byte-identically.

## Machine suggestion column (typing only — no fidelity opinion)

`suggested_transformation_types` is produced by a **deterministic heuristic** in
`audit/make_audit_sample.py` (`suggest_types`); no LLM was used anywhere in this audit.
On punctuation-stripped, lowercased token sequences:

- token sequences equal but raw strings differ → `formatting`;
- token multisets equal, order differs → `reorder`;
- tokens only added (none removed) → `elaboration` (+ `reorder` if the base tokens are
  not a subsequence of the variant);
- tokens removed (substitutions) → `lexical` (+ `reorder` if the surviving common tokens
  change relative order; + `elaboration` if the variant is ≥ 4 tokens longer).

Output is comma-separated in the canonical order lexical, reorder, elaboration,
formatting. It is a **suggestion only**: the author's `human_transformation_types` column
is authoritative, and the fidelity judgment (`human_pass`) is exclusively human.
