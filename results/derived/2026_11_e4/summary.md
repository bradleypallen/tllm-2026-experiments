# E4 / E4-0: paraphrase versus identical repetition

Models: Llama 4 Maverick, Llama 4 Scout, Gemini 2.5 Flash, DeepSeek-V3, Qwen 2.5 72B. Seed 42, 10000 bootstrap resamples over (model, benchmark) cells.
Items with no API error: 1322 (excluded: 0); same-day base ⟨t,f⟩: 1165 (11.9% of items no longer ⟨t,f⟩ at s₀).

## T-E4b  change from a same-day ⟨t,f⟩ base

| scope | n | repeat: any change | paraphrase: any change | difference [95% CI] | repeat: →⟨f,f⟩ | paraphrase: →⟨f,f⟩ | difference [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | 1165 | 0.009 | 0.160 | +0.151 [0.092, 0.224] | 0.004 | 0.054 | +0.050 [0.029, 0.075] |
| Llama 4 Maverick | 254 | 0.004 | 0.248 | +0.244 [0.048, 0.434] | 0.000 | 0.024 | +0.024 [0.000, 0.062] |
| Llama 4 Scout | 259 | 0.000 | 0.193 | +0.193 [0.157, 0.229] | 0.000 | 0.073 | +0.073 [0.021, 0.156] |
| Gemini 2.5 Flash | 227 | 0.009 | 0.132 | +0.123 [0.093, 0.184] | 0.000 | 0.044 | +0.044 [0.013, 0.067] |
| DeepSeek-V3 | 201 | 0.025 | 0.100 | +0.075 [0.025, 0.150] | 0.015 | 0.060 | +0.045 [0.008, 0.100] |
| Qwen 2.5 72B | 224 | 0.009 | 0.103 | +0.094 [0.024, 0.186] | 0.009 | 0.071 | +0.062 [0.024, 0.113] |

**P3** (pooled any-change rate under paraphrase exceeds that under identical repetition, and the CI of the paired difference excludes zero): holds (difference 0.1511, CI [0.0924, 0.2236]).

## T-E4a  distribution of the aggregate over NINE (pooled)

| value | repeat | paraphrase |
| --- | --- | --- |
| <t,f> | 1155 (0.991) | 979 (0.840) |
| <t,e> | 0 (0.000) | 15 (0.013) |
| <f,t> | 0 (0.000) | 7 (0.006) |
| <e,t> | 0 (0.000) | 0 (0.000) |
| <t,t> | 5 (0.004) | 46 (0.040) |
| <f,f> | 5 (0.004) | 63 (0.054) |
| <e,f> | 0 (0.000) | 23 (0.020) |
| <f,e> | 0 (0.000) | 3 (0.003) |
| <e,e> | 0 (0.000) | 29 (0.025) |

## T-E4c  per-model excess (paraphrase − repeat, any change)

| model | n | excess | 95% CI |
| --- | --- | --- | --- |
| Llama 4 Maverick | 254 | +0.244 | [0.048, 0.434] |
| Llama 4 Scout | 259 | +0.193 | [0.157, 0.229] |
| Gemini 2.5 Flash | 227 | +0.123 | [0.093, 0.184] |
| Qwen 2.5 72B | 224 | +0.094 | [0.024, 0.186] |
| DeepSeek-V3 | 201 | +0.075 | [0.025, 0.150] |

Pairwise orderings whose CI excludes zero: Llama 4 Scout - Gemini 2.5 Flash, Llama 4 Scout - DeepSeek-V3, Llama 4 Scout - Qwen 2.5 72B, Gemini 2.5 Flash - DeepSeek-V3

## Runs

| model | records | elicitations | parse fail | API error | provider | finished |
| --- | --- | --- | --- | --- | --- | --- |
| Llama 4 Maverick | 267 | 3330 | 0.0399 | 0.0 | ['Parasail'] | 2026-11-11T20:06:45.776298+00:00 |
| Llama 4 Scout | 328 | 4192 | 0.0191 | 0.0 | ['DeepInfra'] | 2026-11-11T20:20:09.021551+00:00 |
| Gemini 2.5 Flash | 270 | 3312 | 0.0 | 0.0 | ['Google AI Studio'] | 2026-11-11T20:03:24.661335+00:00 |
| DeepSeek-V3 | 223 | 2854 | 0.0 | 0.0 | ['DeepInfra'] | 2026-11-11T20:17:14.555302+00:00 |
| Qwen 2.5 72B | 234 | 3028 | 0.0 | 0.0 | ['DeepInfra'] | 2026-11-11T20:09:07.108460+00:00 |
