# 2025-09 snapshot

Empirical results generated on 2025-09-12 and 2025-09-13.

## Configuration

| Setting | Value |
|---------|-------|
| `bilateral-truth` commit | `8545986a51ce1af5d3df9dfc373a8fad58914636` |
| Samples per cell (N) | 1,000 |
| Bilateral samples (majority vote) | **1** (no majority voting) |
| Random seed | 42 (balanced positive/negative) |
| Context (s₀) | Generic preamble: `"This statement is being evaluated for factual accuracy as part of a bilateral truth evaluation benchmark."` + category appended |
| Models | Claude Opus 4.1, Claude 3.5 Haiku, GPT-4.1, GPT-4.1-mini, Gemini 2.5 Flash, Llama 4 Maverick, Llama 4 Scout |
| Benchmarks | TruthfulQA, SimpleQA, MMLU-Pro, FACTScore |

## Approach definitions (this run)

| Subdirectory | Approach | Prompt frame |
|--------------|----------|--------------|
| `classical/` | Bilateral with classical projection (D = {⟨t,f⟩}) | Verifiability + refutability calls, separate (bilateral-truth `evaluate_bilateral`, samples=1) |
| `unilateral_direct/` | Forced unilateral | "Determine whether the following statement is correct. Conclude with CORRECT / INCORRECT." |
| `unilateral_uncertain/` | **Confidence-based ternary** | "CORRECT if you are confident the statement is true / INCORRECT if confident false / UNCERTAIN if you lack sufficient confidence." |
| `unilateral_confidence/` | Numerical confidence rating + thresholds 0.5, 0.7, 0.9 | "Rate your confidence … 0.0–1.0." Threshold analysis lives inside each file's `threshold_analysis` field. |

The `unilateral_uncertain` ternary asks the model to self-report meta-confidence. This is **not** the same construct as the 2026 evidence-based ternary (which asks for an evidence judgement). When comparing across snapshots, use this distinction.

## File naming convention

`{benchmark}_{model_safe}_{approach}_results.json` where `model_safe = model_id` with `/` and `:` replaced by `_`.

`MANIFEST.json` lists SHA-256 hashes of every result file.

## Headline F1-macro averages (28 cells = 7 models × 4 benchmarks)

| Approach | Mean F1-macro |
|----------|---------------|
| Bilateral (classical projection) | 0.739 |
| Forced unilateral | 0.600 |
| Confidence-based ternary | 0.585 |
| Confidence @ 0.5 | 0.544 |
| Confidence @ 0.7 | 0.645 |
| Confidence @ 0.9 | 0.638 |
