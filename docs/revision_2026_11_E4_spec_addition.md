# Addition to "BBL Revision Experiments: E1–E3 Specification": E4 / E4-0

Status: **approved by the author on 2026-11-11 (full design, 16,716 calls), before any E4 call was made.**
Written before any E4 data existed. Conventions (pins, parsing, bootstrap, gates) follow the
2026-10-18 specification unless stated.

## 1. Purpose

Reviewer 2, major comment 4: the robustness rankings and the "upper bound" reading of the
⟨t,f⟩→⟨f,f⟩ migration rates in §7.1 need stronger support. Reviewer 1 asks for the full
transition distribution under paraphrase rather than migration to ignorance alone.

The March paraphrase run used one elicitation per situation, on unpinned providers, with no
measured noise floor. E4 measures, on the same items, the same day and the same pinned
providers, how much a ⟨t,f⟩ valuation changes under **paraphrase** (E4) against how much it
changes under **identical repetition** (E4-0). The difference is the paraphrase effect net of
elicitation noise.

## 2. Design (paired, within item)

For each item (model *M*, assertion *p*) with source question *s₀* and cached paraphrases
*s₁ … s_k* (*k* = 3; *k* = 2 for FACTScore, as in `variant_cache/`):

| step | situation | elicitations |
|---|---|---|
| base | *s₀* | 1 verification + 1 refutation → *b* = ⟨u₀, v₀⟩ |
| E4-0 (placebo) | *s₀* again, *k* times | 1 + 1 each |
| E4 (paraphrase) | *s₁ … s_k* | 1 + 1 each |

- Aggregates share the base: □_rep = ⟨min u, max v⟩ over base + the *k* repeats;
  □_par = ⟨min u, max v⟩ over base + the *k* paraphrases (Definition 7, order f < e < t).
- Order within an item: base, then repeat 1, paraphrase 1, repeat 2, paraphrase 2, …, so
  neither arm is systematically later.
- One elicitation per situation, as in the March paraphrase run (no majority vote).

## 3. Items

All (model, assertion) pairs whose **March paraphrase-run** value at *s₀* is ⟨t,f⟩ (the
denominators of Figure 2), for the five retained models. No sampling, so no seed.

| model | items |
|---|---|
| Llama 4 Maverick | 267 |
| Llama 4 Scout | 328 |
| Gemini 2.5 Flash | 270 |
| DeepSeek-V3 | 223 |
| Qwen 2.5 72B | 234 |
| **total** | **1,322** (TruthfulQA 364, SimpleQA 161, MMLU-Pro 349, FACTScore 448) |

The March value only fixes the list. The primary analysis conditions on the **same-day** base
*b* = ⟨t,f⟩ (§6), so selection on a seven-month-old value cannot manufacture change.
Claude Opus 4.1 is excluded (retired 2026-08-05), as in E1–E3.

## 4. Requests and parsing

- Prompts: the frozen `VERIFICATION_PROMPT`, `REFUTATION_PROMPT` and `BILATERAL_SYSTEM_PROMPT`
  of `evaluators/revision_2026_11_prompts.py`, **unchanged**; the `{context}` slot carries the
  situation text (*s₀* or a paraphrase), as in the March run.
- temperature 0, `max_tokens` 32, three retries with backoff, as in E1–E3.
- Providers pinned as on 2026-10-18 (DigitalOcean for Maverick; DeepInfra for Scout, DeepSeek
  and Qwen; Google AI Studio for Gemini), `allow_fallbacks = false`; provider and model
  strings recorded per call.
- Parsing: the frozen single-token rule (`parse_single`). `PARSE_FAIL` is recorded as an
  outcome and enters the aggregate as *e*, as in the March run. An item with any `API_ERROR`
  in either arm is excluded from both arms and counted.

## 5. Size

2 × (1 + 2*k*) calls per item: **16,716 calls** (the October run was 18,389). Five models in
parallel; the largest cell is Scout at about 4,200 calls. Expected 1–1.5 hours and a few
dollars. Smoke test: 5 items per model, at most 350 calls.

Reduced variant if the placebo arm is dropped: 9,680 calls. Not recommended: the existing
E1-0 floor is on other items and another day, which is the comparison E4 exists to avoid.

## 6. Analysis and prediction (fixed before the run)

Primary analyses use items with same-day base *b* = ⟨t,f⟩ and no `API_ERROR`.

- **T-E4a.** Distribution of □_rep and of □_par over NINE, pooled and per model.
- **T-E4b.** Per arm: any-change rate (aggregate ≠ ⟨t,f⟩) and →⟨f,f⟩ rate. Paired difference
  (paraphrase − repeat) with 95% cluster-bootstrap CI over (model, benchmark) cells, 10,000
  resamples, seed 42.
- **T-E4c.** Per-model paraphrase-minus-repeat excess with CIs, and the pairwise model
  orderings on that excess whose CI excludes zero.
- **Secondary.** Share of the 1,322 items whose same-day base is not ⟨t,f⟩ (drift since March),
  reported and not analysed further.

**P3.** Pooled, the any-change rate under paraphrase exceeds that under identical repetition,
and the CI of the paired difference excludes zero.

Reference values already in hand (not part of E4): same-day identical-prompt floor from E1-0,
1.4% any change over two repeats (n = 439); March paraphrase run, six models, 26.2% any change
and 7.3% →⟨f,f⟩ (n = 1,591).

## 7. Reporting commitment

E4 is reported whichever way P3 comes out, as a same-day control beside Figure 2 in §7.1.
Figure 2 is not replaced. If the E4 paraphrase rates differ from the March five-model figures,
both are stated with the conditions that differ (date, pinned providers).

## 8. Gates and cut-off

- Smoke first, pinned (the purpose is to confirm the October pins still serve). **Stop and ask**
  if: `providers_seen` ≠ the pinned provider; `models_returned` ≠ the requested id; any
  `api_error_rate` > 5%; or a model's `parse_fail_rate` exceeds twice its E1-0 rate
  (Maverick 5.6%, Scout 2.8%, others ≈ 0), which would signal a changed endpoint.
- Stop and ask if any cell's call count exceeds twice the plan.
- **Cut-off:** if E4 is not run and analysed by Thursday 12 November, 23:59, the paper goes
  with the existing floor; any data collected are still committed.

## 9. Harness (nothing from the October run is edited)

New files only:

- `datasets/revision_2026_11_e4/` — `build_e4_items.py`, `e4_items.jsonl`, `MANIFEST.json`
- `evaluators/revision_e4_evaluator.py` — imports the frozen prompts module and the October
  evaluator's client; one invocation per model; appends and resumes; `.meta.json` as before
- `results/2026_11_e4/` — results, `MANIFEST.json`, `requirements.lock`
- `analysis/revision_e4.py` → `results/derived/2026_11_e4/`
- `tests/test_revision_e4.py` — list determinism and counts, paraphrases match the cache,
  prompts imported unchanged, aggregation, offline dry run

Untouched: `prompts/`, `evaluators/revision_2026_11_prompts.py`,
`evaluators/revision_2026_11_evaluator.py`, `datasets/revision_2026_11/`, `results/2026_11/`.

## 10. Amendments

**2026-11-11, after the first smoke test and before any full-run call (approved by the author).**
The pinned smoke test (5 items per model) passed every gate for Gemini 2.5 Flash, Llama 4 Scout
and DeepSeek-V3 (70 calls each, no API or parse errors, served by the pinned provider). Two
stop conditions fired:

1. **Llama 4 Maverick.** All 42 calls failed with HTTP 404: DigitalOcean, the October pin, no
   longer serves `meta-llama/llama-4-maverick`. E4 pins **Parasail** (fp8) for Maverick instead
   (`E4_PROVIDER` in `evaluators/revision_e4_evaluator.py`). §4's "pinned as on 2026-10-18" holds
   for the other four models. Both E4 arms for an item run on the same provider on the same day,
   so the paired comparison is unaffected; Maverick's E4 results are not comparable with its
   October (DigitalOcean) results at the level of the serving endpoint, and the paper says so.
2. **Qwen 2.5 72B.** 13 of 14 calls failed with HTTP 429 ("temporarily rate-limited upstream")
   at DeepInfra, the model's only endpoint. Qwen is run separately when the endpoint clears;
   the §8 cut-off applies to it on its own. If it does not clear, E4 is reported on four models.

The failed smoke records are kept in `results/2026_11_e4/smoke/`; Maverick's re-smoke on
Parasail is in `results/2026_11_e4/smoke_parasail/`.

**2026-11-11, later the same day.** Maverick's re-smoke on Parasail passed every gate (70 calls,
no errors, served only by Parasail). The four-model run (Maverick, Scout, DeepSeek-V3, Gemini)
started at 19:41Z. Qwen's endpoint began answering about ten minutes after its failed smoke; its
repeat smoke test passed every gate (`results/2026_11_e4/smoke_qwen_retry/`, 70 calls, no
errors) and its run started at about 19:44Z, so E4 covers all five models on the same day.
