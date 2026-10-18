# Methodology

This document defines what each approach name means in each run. It is the canonical reference when cross-referencing results.

## Bilateral evaluation

Given an assertion `p` and a situation context `s₀`, the bilateral evaluator makes two independent calls to the LLM:

- **Verification call** — asks whether `p` can be verified as true based on available evidence. Output: `VERIFIED` (→ `t`) or `CANNOT VERIFY` (→ `f`), parse failure → `e`.
- **Refutation call** — asks whether `p` can be refuted (shown false) based on available evidence. Output: `REFUTED` (→ `t`) or `CANNOT REFUTE` (→ `f`), parse failure → `e`.

The pair ⟨u,v⟩ ∈ {t,e,f}² is the bilateral truth value, one of nine values from the bilattice NINE = ⟨V₃ × V₃, ≤_t, ≤_k⟩.

Projection to a binary prediction depends on the choice of designated values D:

| Policy | D | Behavior |
|--------|---|----------|
| `classical` | {⟨t,f⟩} | Predict TRUE only on clean verifications; abstain on everything else. |
| `paracomplete` | {⟨t,f⟩, ⟨t,e⟩} | Also commit on partial verifications. |
| `paraconsistent` | {⟨t,f⟩, ⟨t,t⟩} | Also commit on contradictions. |

The empirical "F1 across the three policies" is a study of how the choice of D affects downstream performance.

**Sampling**: the 2026 proper run (`results/2026_03/proper/`) uses `samples=3` with majority voting on each component (u, v) independently; the 2026 modal run (`results/2026_03/modal/`) elicits each situation once (`samples=1`, see below); 2025 ran with `samples=1`.

**Tiebreak**: with `samples=3`, a tie can only arise on an all-distinct component vote {t, e, f}. The run broke such ties with `bilateral-truth`'s default `tiebreak_strategy="random"` (an unseeded `random.choice` among the tied values). This is *not* the deterministic f ≺ t ≺ e tiebreak (all-distinct → e) of the paper's Definition 14 / Remark 4; the library has no strategy implementing that order. Only the majority-voted ⟨u,v⟩ is persisted, so how many assertions were affected cannot be recovered.

## Forced unilateral (binary)

A single call that forces TRUE/FALSE with no abstention option. 100% coverage by construction. F1-macro is computed against the binary ground-truth label.

## Ternary

A single call returning TRUE / FALSE / UNCERTAIN, where UNCERTAIN means abstention. Coverage < 100% by construction.

**Critical**: the ternary prompt has been redefined between runs.

- **2025-09 ternary** asks the model to self-report meta-confidence: *"if you are confident the statement is true / if you are confident it is false / if you lack sufficient confidence."* This is a confidence-elicitation prompt.
- **2026-03 ternary** asks the model to make an evidence judgement: *"if evidence supports / if evidence contradicts / if there is insufficient evidence to either support or refute."* This is an evidence-based prompt — the same epistemic frame as the bilateral verification and refutation calls.

**The two prompts produce different constructs.** When comparing across runs, do not assume "ternary" means the same thing in both columns. See [`../CHANGELOG.md`](../CHANGELOG.md).

## Confidence

A single call returning a number in [0.0, 1.0]. Thresholded:

- 2025-09: at 0.5, 0.7, 0.9 (three derived approaches per assertion)
- 2026-03: at 0.5 in canonical metrics; raw score preserved per assertion so any threshold can be applied post-hoc

Below threshold = abstention. Above = predict TRUE if confidence ≥ threshold else FALSE.

## Modal — [[□p]] (2026 only)

For each assertion `p` with source situation `s₀`, take `k` paraphrases `s₁, …, s_k` of the source question (generated once with Claude Opus 4.6, cached in `variant_cache/`). Evaluate bilateral VM(sᵢ, p) for each situation, then aggregate:

[[□p]] = ⟨ min(uᵢ), max(vᵢ) ⟩

This is the modal necessity estimate: a verification only "counts" if the model can verify p under every situation we tested; a refutation counts if the model can refute under any. In the canonical run **k = 3** (four situations: s₀ plus three paraphrases), except **FACTScore, where k = 2**. This matches the paper (§7). The `n3` in the cache and result filenames is the generator's `n_variants=3` setting, not the number of situations. The null situation (no context) is **not** included in the aggregation.

**Single elicitation per situation.** `modal_evaluator.py` calls `zeta(...)` without a `samples` argument, so every situation, s₀ included, was evaluated with `samples=1`, not by the 3-vote majority used in the proper run. Consequences:

- The modal files' `src_bilateral` (VM(s₀, p)) is a separate elicitation from the proper files' `bilateral_tv`. They disagree on 579 of 6,000 canonical assertions (Qwen 2.5 72B: 0; others up to 65/250 per cell). `tests/test_paper_numbers.py::test_modal_s0_is_separate_elicitation` pins this.
- Paper quantities computed from the modal files (Figure 2 migration rates, the §6.3 worked examples) rest on single elicitations. Table 1 and Figure 3 come from the proper files and rest on the 3-vote majority.

**On-the-fly fallback.** If an assertion is missing from the cache, `modal_evaluator.py` generates `n_variants − 1` paraphrases (2 by default), not the 3 the cache holds. Use `--skip-pregen` with the shipped cache to reproduce the canonical neighbourhoods.

The 2026 modal result files include both VM(s₀, p) (the bilateral source distribution) and [[□p]] (the modal aggregated distribution).

## Sampling

- **Selection**: `seed=42`, balanced positive/negative split, `max_samples` controls N
- **N per cell**: 1000 (2025), 250 (2026)
- **Per assertion**: bilateral calls majority-vote across `samples` evaluator invocations (n=1 in 2025; n=3 in the 2026 proper run; n=1 per situation in the 2026 modal run); all unilateral approaches are single-call

## What is *not* in this evaluation pipeline

- Chain-of-thought prompting
- Few-shot exemplars
- Fine-tuning
- External retrieval (no RAG)
- Self-consistency beyond the n=3 majority vote for bilateral

All approaches are zero-shot, single-turn API calls (single per-method for unilateral; six per assertion for n=3 bilateral in the proper run; two per situation in the modal run). See [`../prompts/`](../prompts/) for the verbatim prompt strings.
