# Revision 2026-11: derived numbers

Seed 42, 10000 bootstrap resamples. Files: `results/derived/2026_11/`.

## T-TM  <t,f> -> NINE under paraphrase (pooled, 6 models)

| value | count | share |
| --- | --- | --- |
| <t,f> | 830 | 0.700 |
| <t,e> | 24 | 0.020 |
| <f,t> | 31 | 0.026 |
| <e,t> | 11 | 0.009 |
| <t,t> | 71 | 0.060 |
| <f,f> | 107 | 0.090 |
| <e,f> | 40 | 0.034 |
| <f,e> | 5 | 0.004 |
| <e,e> | 66 | 0.056 |

## T-RK  <t,f> -> <f,f> migration by model (95% CI over benchmark cells)

| model | n <t,f> | migration % | CI |
| --- | --- | --- | --- |
| DeepSeek-V3 | 155 | 16.1 | [11.9, 25.0] |
| Qwen 2.5 72B | 164 | 12.2 | [8.1, 19.2] |
| Llama 4 Scout | 243 | 11.9 | [7.0, 19.7] |
| Gemini 2.5 Flash | 206 | 7.8 | [0.0, 10.3] |
| Claude Opus 4.1 | 212 | 4.2 | [2.0, 7.0] |
| Llama 4 Maverick | 205 | 3.9 | [0.0, 12.1] |

Pairwise orderings whose CI excludes zero: Claude Opus 4.1 - Llama 4 Scout, Claude Opus 4.1 - DeepSeek-V3, Claude Opus 4.1 - Qwen 2.5 72B, Llama 4 Maverick - Llama 4 Scout, Llama 4 Maverick - DeepSeek-V3, Llama 4 Maverick - Qwen 2.5 72B, Llama 4 Scout - DeepSeek-V3, Gemini 2.5 Flash - DeepSeek-V3, DeepSeek-V3 - Qwen 2.5 72B

## T-COV  Table 1 with coverage

| benchmark | model | bilateral F1 (cov) | ternary F1 (cov) | dF1 |
| --- | --- | --- | --- | --- |
| truthfulqa | Claude Opus 4.1 | 0.905 (0.84) | 0.938 (0.90) | -0.033 |
| truthfulqa | Llama 4 Maverick | 0.769 (0.74) | 0.781 (0.92) | -0.012 |
| truthfulqa | Llama 4 Scout | 0.719 (0.66) | 0.772 (0.88) | -0.052 |
| truthfulqa | Gemini 2.5 Flash | 0.828 (0.80) | 0.856 (0.89) | -0.028 |
| truthfulqa | DeepSeek-V3 | 0.827 (0.62) | 0.831 (0.71) | -0.005 |
| truthfulqa | Qwen 2.5 72B | 0.834 (0.73) | 0.839 (0.75) | -0.004 |
| simpleqa | Claude Opus 4.1 | 0.892 (0.60) | 0.943 (0.65) | -0.051 |
| simpleqa | Llama 4 Maverick | 0.756 (0.70) | 0.865 (0.89) | -0.109 |
| simpleqa | Llama 4 Scout | 0.904 (0.60) | 0.858 (0.78) | +0.046 |
| simpleqa | Gemini 2.5 Flash | 0.624 (0.71) | 0.809 (0.97) | -0.184 |
| simpleqa | DeepSeek-V3 | 0.719 (0.51) | 0.843 (0.48) | -0.124 |
| simpleqa | Qwen 2.5 72B | 0.775 (0.63) | 0.925 (0.56) | -0.150 |
| mmlu-pro | Claude Opus 4.1 | 0.897 (0.54) | 0.898 (0.66) | -0.000 |
| mmlu-pro | Llama 4 Maverick | 0.696 (0.80) | 0.749 (0.57) | -0.054 |
| mmlu-pro | Llama 4 Scout | 0.684 (0.58) | 0.712 (0.67) | -0.029 |
| mmlu-pro | Gemini 2.5 Flash | 0.784 (0.74) | 0.768 (0.96) | +0.017 |
| mmlu-pro | DeepSeek-V3 | 0.762 (0.59) | 0.734 (0.66) | +0.028 |
| mmlu-pro | Qwen 2.5 72B | 0.761 (0.64) | 0.705 (0.76) | +0.055 |
| factscore | Claude Opus 4.1 | 0.511 (0.54) | 0.534 (0.62) | -0.022 |
| factscore | Llama 4 Maverick | 0.439 (0.49) | 0.442 (0.81) | -0.003 |
| factscore | Llama 4 Scout | 0.435 (0.41) | 0.463 (0.57) | -0.028 |
| factscore | Gemini 2.5 Flash | 0.629 (0.70) | 0.612 (0.89) | +0.017 |
| factscore | DeepSeek-V3 | 0.527 (0.31) | 0.516 (0.36) | +0.011 |
| factscore | Qwen 2.5 72B | 0.566 (0.36) | 0.472 (0.46) | +0.094 |

macro-F1 is computed on covered items only (abstentions excluded from both numerator and denominator); coverage = covered / N is reported beside it. Bilateral abstains on dec = abstain (<t,t> and the neither-cells); ternary on UNCERTAIN. Replicates compute_metrics in proper_benchmark_evaluator.py.

