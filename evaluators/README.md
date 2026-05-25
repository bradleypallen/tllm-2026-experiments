# Evaluators

The scripts that produce the result files in `results/2026_03/`.

| Script | Purpose |
|--------|---------|
| `proper_benchmark_evaluator.py` | Five-approach evaluator: bilateral (×3 epistemic policies) + forced unilateral + ternary + confidence@0.5, single pass per assertion. |
| `modal_evaluator.py` | [[□p]] = ⟨min uᵢ, max vᵢ⟩ aggregated over s₀ + 2 paraphrases. Reads paraphrases from `../variant_cache/`. |
| `pregenerate_variants.py` | One-time paraphrase generation (Claude Opus 4.6). Output goes to `../variant_cache/`. |
| `run_modal_suite.py` | Orchestrates `pregenerate_variants.py` + `modal_evaluator.py` across all model × benchmark combinations. |

All scripts default to N=250 with seed=42 balanced sampling, matching the 2026 run.

For the exact bilateral, ternary, forced-unilateral, and confidence prompts these scripts dispatch through `bilateral_truth`, see `../prompts/2026_prompts.md`.

The 2025 run used a different set of evaluator scripts (located in the bilateral-truth repo's `evaluations/` directory at commit `8545986`). They are not shipped here — only the resulting JSON files in `results/2025_09/`.
