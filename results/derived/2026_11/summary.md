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

## T-E1b  persistence (re-parsed)

| arm | condition | scope | n valid | persistence | CI |
| --- | --- | --- | --- | --- | --- |
| e1-0 | treatment | pooled | 338 | 0.589 | [0.447, 0.673] |
| e1-0 | treatment | DeepSeek-V3 | 18 | 0.389 | [0.167, 0.484] |
| e1-0 | treatment | Gemini 2.5 Flash | 182 | 0.643 | [0.453, 0.721] |
| e1-0 | treatment | Llama 4 Maverick | 56 | 0.321 | [0.130, 0.500] |
| e1-0 | treatment | Llama 4 Scout | 29 | 0.483 | [0.333, 0.583] |
| e1-0 | treatment | Qwen 2.5 72B | 53 | 0.811 | [0.727, 0.909] |
| e1-0 | control_tf | pooled | 394 | 0.906 | [0.872, 0.936] |
| e1-0 | control_tf | DeepSeek-V3 | 36 | 0.944 | [0.900, 1.000] |
| e1-0 | control_tf | Gemini 2.5 Flash | 100 | 0.860 | [0.773, 0.900] |
| e1-0 | control_tf | Llama 4 Maverick | 100 | 0.940 | [0.910, 0.975] |
| e1-0 | control_tf | Llama 4 Scout | 58 | 0.828 | [0.773, 1.000] |
| e1-0 | control_tf | Qwen 2.5 72B | 100 | 0.950 | [0.946, 0.958] |
| e1-0 | control_ff | pooled | 374 | 0.845 | [0.771, 0.913] |
| e1-0 | control_ff | DeepSeek-V3 | 36 | 0.889 | [0.818, 1.000] |
| e1-0 | control_ff | Gemini 2.5 Flash | 80 | 0.725 | [0.596, 0.853] |
| e1-0 | control_ff | Llama 4 Maverick | 100 | 0.920 | [0.859, 1.000] |
| e1-0 | control_ff | Llama 4 Scout | 58 | 0.707 | [0.660, 1.000] |
| e1-0 | control_ff | Qwen 2.5 72B | 100 | 0.930 | [0.873, 0.986] |
| e1-a | treatment | pooled | 298 | 0.044 | [0.018, 0.092] |
| e1-a | treatment | DeepSeek-V3 | 18 | 0.056 | [0.000, 0.091] |
| e1-a | treatment | Gemini 2.5 Flash | 181 | 0.011 | [0.000, 0.032] |
| e1-a | treatment | Llama 4 Maverick | 25 | 0.000 | [0.000, 0.000] |
| e1-a | treatment | Llama 4 Scout | 22 | 0.227 | [0.143, 1.000] |
| e1-a | treatment | Qwen 2.5 72B | 52 | 0.096 | [0.000, 0.182] |
| e1-a | control_tf | pooled | 540 | 0.828 | [0.732, 0.886] |
| e1-a | control_tf | DeepSeek-V3 | 36 | 0.806 | [0.567, 0.917] |
| e1-a | control_tf | Gemini 2.5 Flash | 269 | 0.918 | [0.899, 0.939] |
| e1-a | control_tf | Llama 4 Maverick | 81 | 0.889 | [0.848, 1.000] |
| e1-a | control_tf | Llama 4 Scout | 49 | 0.510 | [0.400, 0.579] |
| e1-a | control_tf | Qwen 2.5 72B | 105 | 0.705 | [0.545, 0.809] |
| e1-a | control_ff | pooled | 323 | 0.901 | [0.826, 0.963] |
| e1-a | control_ff | DeepSeek-V3 | 36 | 0.917 | [0.864, 1.000] |
| e1-a | control_ff | Gemini 2.5 Flash | 80 | 0.762 | [0.635, 0.889] |
| e1-a | control_ff | Llama 4 Maverick | 50 | 0.960 | [0.833, 1.000] |
| e1-a | control_ff | Llama 4 Scout | 51 | 0.843 | [0.765, 0.909] |
| e1-a | control_ff | Qwen 2.5 72B | 106 | 1.000 | [1.000, 1.000] |
| e1-b | treatment | pooled | 295 | 0.024 | [0.008, 0.054] |
| e1-b | treatment | DeepSeek-V3 | 18 | 0.000 | [0.000, 0.000] |
| e1-b | treatment | Gemini 2.5 Flash | 182 | 0.011 | [0.000, 0.030] |
| e1-b | treatment | Llama 4 Maverick | 23 | 0.043 | [0.000, 0.052] |
| e1-b | treatment | Llama 4 Scout | 21 | 0.191 | [0.154, 0.333] |
| e1-b | treatment | Qwen 2.5 72B | 51 | 0.000 | [0.000, 0.000] |

### P1  E1-0 minus E1-a, treatment (paired)

| scope | n | diff | CI |
| --- | --- | --- | --- |
| pooled | 298 | +0.587 | [0.433, 0.664] |
| DeepSeek-V3 | 18 | +0.333 | [0.167, 0.387] |
| Gemini 2.5 Flash | 181 | +0.630 | [0.434, 0.711] |
| Llama 4 Maverick | 25 | +0.480 | [0.000, 0.632] |
| Llama 4 Scout | 22 | +0.227 | [-1.000, 0.357] |
| Qwen 2.5 72B | 52 | +0.731 | [0.700, 0.769] |

### T-E1c  collapse direction x gold (pooled, E1-a treatment)

- gold_TRUE|verification: 53
- gold_TRUE|refutation: 119
- gold_FALSE|verification: 35
- gold_FALSE|refutation: 46
- gold_FALSE|both: 14
- gold_TRUE|both: 18

## F-3'  by original bilateral value

| value | n | UNCERTAIN | CONFLICTING | either | E2-0 UNCERTAIN (same day) |
| --- | --- | --- | --- | --- | --- |
| <t,f> | 1322 | 0.030 | 0.015 | 0.044 | 0.031 |
| <t,e> | 46 | 0.000 | 0.000 | 0.000 | - |
| <f,t> | 1570 | 0.121 | 0.024 | 0.145 | - |
| <e,t> | 143 | 0.043 | 0.000 | 0.043 | - |
| <t,t> | 346 | 0.006 | 0.015 | 0.021 | 0.018 |
| <f,f> | 1375 | 0.600 | 0.023 | 0.622 | 0.520 |
| <e,f> | 65 | 0.214 | 0.018 | 0.232 | - |
| <f,e> | 13 | 0.273 | 0.091 | 0.364 | - |
| <e,e> | 120 | 0.143 | 0.000 | 0.143 | - |

### P2  P(CONFLICTING|<t,t>) - P(CONFLICTING|controls)

| scope | n <t,t> | P(C|tt) | P(C|ctrl) | gap | CI |
| --- | --- | --- | --- | --- | --- |
| pooled | 340 | 0.015 | 0.019 | -0.004 | [-0.021, 0.013] |
| DeepSeek-V3 | 18 | 0.000 | 0.013 | -0.013 | [-0.042, 0.000] |
| Gemini 2.5 Flash | 182 | 0.011 | 0.020 | -0.009 | [-0.028, 0.011] |
| Llama 4 Maverick | 60 | 0.050 | 0.033 | +0.017 | [-0.035, 0.062] |
| Llama 4 Scout | 27 | 0.000 | 0.029 | -0.029 | [-0.066, -0.005] |
| Qwen 2.5 72B | 53 | 0.000 | 0.003 | -0.003 | [-0.011, 0.000] |

## T-E3  compliance base rate

| model | n | c(M) | Wilson 95% | <t,t> share (Mar) |
| --- | --- | --- | --- | --- |
| Llama 4 Maverick | 100 | 0.000 | [0.000, 0.037] | 0.064 |
| Llama 4 Scout | 98 | 0.031 | [0.011, 0.086] | 0.029 |
| Gemini 2.5 Flash | 100 | 0.150 | [0.093, 0.233] | 0.182 |
| DeepSeek-V3 | 100 | 0.020 | [0.005, 0.070] | 0.018 |
| Qwen 2.5 72B | 100 | 0.020 | [0.005, 0.070] | 0.053 |
