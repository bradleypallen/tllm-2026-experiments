# CHANGELOG

This file documents methodology changes between experimental runs. It is load-bearing for any cross-run comparison.

## 2026-03 run

- **`bilateral-truth` commit**: `b71915fda6faf252cdb506294e0b9c5e04511d5e`
- **Sample size**: N = 250 per benchmark, seed = 42, balanced positive/negative
- **Bilateral**: `samples=3`, majority vote across three (verification, refutation) call pairs
- **Context (s₀)**: per-assertion `source_question` field
- **Ternary**: **evidence-based** prompt — *"Based on available evidence and knowledge, determine whether the following statement is supported by evidence, contradicted by evidence, or undetermined."* See [`prompts/2026_prompts.md`](prompts/2026_prompts.md).
- **Confidence**: thresholded at 0.5 in the canonical metrics; raw scores are preserved in `detailed_results` for post-hoc thresholding.
- **Modal evaluator added**: [[□p]] = ⟨min uᵢ, max vᵢ⟩ over s₀ + 2 paraphrases. Paraphrases pre-generated once with Claude Opus 4.6 and cached in [`variant_cache/`](variant_cache/).
- **Model lineup (6 models)**:
  - Claude Opus 4.1 (`claude-opus-4-1-20250805`) — Anthropic, closed
  - DeepSeek-V3 (`deepseek/deepseek-chat`) — open, via OpenRouter
  - Gemini 2.5 Flash (`google/gemini-2.5-flash`) — Google, closed, via OpenRouter
  - Llama 4 Maverick (`meta-llama/llama-4-maverick`) — open, via OpenRouter
  - Llama 4 Scout (`meta-llama/llama-4-scout`) — open, via OpenRouter
  - Qwen 2.5 72B Instruct (`qwen/qwen-2.5-72b-instruct`) — open, via OpenRouter
- **Note on extra files**: `results/2026_03/proper/` and `results/2026_03/modal/` contain a partial set of GPT-4.1 and GPT-4.1-mini result files. These were collected before OpenAI quota issues caused the model lineup to be revised. They are **not part of the canonical 6-model lineup** and should not be referenced in new analysis.

## 2025-09 run

- **`bilateral-truth` commit**: `8545986a51ce1af5d3df9dfc373a8fad58914636`
- **Sample size**: N = 1,000 per benchmark, seed = 42, balanced positive/negative
- **Bilateral**: `samples=1` (no majority voting)
- **Context (s₀)**: generic preamble `"This statement is being evaluated for factual accuracy as part of a bilateral truth evaluation benchmark."` with `\n\nCategory: {category}` appended
- **Ternary**: **confidence-based** prompt — *"CORRECT if you are confident the statement is true / INCORRECT if confident false / UNCERTAIN if you lack sufficient confidence."* See [`prompts/2025_prompts.md`](prompts/2025_prompts.md).
- **Confidence**: numerical 0.0–1.0 scoring, then thresholded at 0.5, 0.7, and 0.9 to produce three derived approaches.
- **Model lineup (7 models)**:
  - Claude Opus 4.1 (`claude-opus-4-1-20250805`)
  - Claude 3.5 Haiku (`claude-3-5-haiku-20241022`)
  - GPT-4.1 (`gpt-4.1-2025-04-14`)
  - GPT-4.1-mini (`gpt-4.1-mini-2025-04-14`)
  - Gemini 2.5 Flash (`google/gemini-2.5-flash`)
  - Llama 4 Maverick (`meta-llama/llama-4-maverick`)
  - Llama 4 Scout (`meta-llama/llama-4-scout`)

## What changed between runs

### Critical: ternary prompt was redefined

The 2025 "ternary" prompt asked the model to self-report meta-confidence. The 2026 "ternary" prompt asks for an evidence judgement — the same epistemic frame as the bilateral verification/refutation prompts.

These are **not the same construct**. They use the same English word "uncertain" / "UNCERTAIN" to mean different things:

| Run | "UNCERTAIN" means |
|-----|---------------------|
| 2025 | "I lack sufficient confidence to commit." (meta-cognitive self-report) |
| 2026 | "Available evidence is insufficient to either support or refute." (evidential judgement) |

This single change accounts for almost the entire shift in reported "ternary vs bilateral" comparisons:

| Approach | 2025 mean F1-macro | 2026 mean F1-macro | Δ |
|----------|--------------------|--------------------|------|
| Bilateral (classical projection) | 0.739 | 0.730 | −0.009 |
| Forced unilateral | 0.600 | 0.710 | +0.110 |
| **Ternary** | **0.585** | **0.744** | **+0.159** |
| Confidence @ 0.5 | 0.544 | 0.693 | +0.149 |

Bilateral barely moved. Ternary jumped 16 F1 points, putting it slightly *above* bilateral-classical on average. The 2026 paper draft's "diagnostic resolution, not accuracy parity" framing is the correct reading once the ternary baseline is the evidence-based one.

### Other changes (smaller effects but worth recording)

- **Bilateral sampling**: n=1 → n=3 majority vote. Per-cell shifts of ±0.05 F1 max; aggregate mean F1 changed only −0.01.
- **Context (s₀)**: generic factual-accuracy preamble + category → per-assertion `source_question`. Hard to isolate the per-cell effect from the other simultaneous changes.
- **N per cell**: 1000 → 250. Standard error widens by ~2× but mean estimates are stable.
- **Model lineup**: dropped Claude 3.5 Haiku (older/weaker model); dropped GPT-4.1 and GPT-4.1-mini (OpenAI quota issues); added DeepSeek-V3 and Qwen 2.5 72B (open-source representation). Llama 4 Scout was already in the 2025 lineup.
- **Modal evaluator added** (new in 2026). No prior comparison.

## Reproducing the 2025 numbers from scratch

The evaluator scripts in [`evaluators/`](evaluators/) target the 2026 run. To reproduce the 2025 numbers you would need:

1. The 2025-era evaluator scripts, which live in the `bilateral-truth` repo under `evaluations/` at commit `8545986`. They are intentionally **not** carried into this repo — only the 2025 result files in [`results/2025_09/`](results/2025_09/) are.
2. The `bilateral-truth` package pinned to commit `8545986` (e.g. via `pip install "bilateral-truth @ git+https://github.com/bradleypallen/bilateral-truth.git@8545986a51ce1af5d3df9dfc373a8fad58914636"`).
3. The datasets in [`datasets/`](datasets/) (unchanged across runs — these are the same input data).

The 2025 results are preserved primarily as a comparison point for the published paper's revised methodology, not as a target for ongoing reproduction.
