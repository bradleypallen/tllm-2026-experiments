# 2026-03 snapshot

Empirical results generated on 2026-03-18 and 2026-03-19.

## Configuration

| Setting | Value |
|---------|-------|
| `bilateral-truth` commit | `b71915fda6faf252cdb506294e0b9c5e04511d5e` |
| Samples per cell (N) | 250 |
| Bilateral samples (majority vote) | **3** (majority vote across 3 calls per assertion) |
| Random seed | 42 (balanced positive/negative) |
| Context (s₀) | `source_question` field from each assertion |
| Models | Claude Opus 4.1, DeepSeek-V3, Gemini 2.5 Flash, Llama 4 Maverick, Llama 4 Scout, Qwen 2.5 72B (six models; OpenAI dropped due to quota issues — see `../../CHANGELOG.md`) |
| Benchmarks | TruthfulQA, SimpleQA, MMLU-Pro, FACTScore |

## Approach definitions (this run)

A single pass per assertion computes all approaches:

| Subdirectory | Approach | Prompt frame |
|--------------|----------|--------------|
| `proper/` | Bilateral ⟨u,v⟩ + projection under classical / paracomplete / paraconsistent; forced unilateral (TRUE/FALSE); **evidence-based ternary**; numerical confidence | All five are recorded per-assertion. The bilateral verification/refutation prompts come from `bilateral_truth.llm_evaluators`. |
| `modal/` | [[□p]] = ⟨min uᵢ, max vᵢ⟩ aggregated over s₀ + 2 paraphrases (n=3 situations) | Paraphrases pulled from `../../variant_cache/`. |

The 2026 ternary asks: *"Based on available evidence and knowledge, determine whether the following statement is supported by evidence, contradicted by evidence, or undetermined."* This is the same epistemic frame as the bilateral verification/refutation prompts — fundamentally different from the 2025 confidence-self-report ternary.

## File naming convention

- `proper/{benchmark}_{model_safe}_n3_proper_results.json`
- `modal/{benchmark_complete}_{model_safe}_n3_modal_results.json` (dataset stem used here, e.g. `truthfulqa_complete`)

`model_safe = model_id` with `/` and `:` replaced by `_`.

`MANIFEST.json` lists SHA-256 hashes of every result file.

## Headline F1-macro averages (24 cells = 6 models × 4 benchmarks)

| Approach | Mean F1-macro |
|----------|---------------|
| Bilateral classical projection | 0.730 |
| Bilateral paracomplete projection | 0.678 |
| Bilateral paraconsistent projection | 0.721 |
| Forced unilateral | 0.710 |
| **Evidence-based ternary** | **0.744** |
| Confidence @ 0.5 | 0.693 |
