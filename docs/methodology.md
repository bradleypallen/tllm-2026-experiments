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

**Sampling**: 2026 runs use `samples=3` with majority voting on each component (u, v) independently; 2025 ran with `samples=1`.

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

For each assertion `p` with source situation `s₀`, generate `n−1` paraphrases `s₁, …, s_{n−1}` (using Claude Opus 4.6, cached in `variant_cache/`). Evaluate bilateral VM(sᵢ, p) for each situation, then aggregate:

[[□p]] = ⟨ min(uᵢ), max(vᵢ) ⟩

This is the modal necessity estimate: a verification only "counts" if the model can verify p under every situation we tested; a refutation counts if the model can refute under any. n=3 in the canonical run (s₀ plus two paraphrases). The null situation (no context) is **not** included in the aggregation.

The 2026 result files include both VM(s₀, p) (the bilateral source distribution) and [[□p]] (the modal aggregated distribution).

## Sampling

- **Selection**: `seed=42`, balanced positive/negative split, `max_samples` controls N
- **N per cell**: 1000 (2025), 250 (2026)
- **Per assertion**: bilateral calls majority-vote across `samples` evaluator invocations (n=1 in 2025, n=3 in 2026); all unilateral approaches are single-call

## What is *not* in this evaluation pipeline

- Chain-of-thought prompting
- Few-shot exemplars
- Fine-tuning
- External retrieval (no RAG)
- Self-consistency beyond the n=3 majority vote for bilateral

All approaches are zero-shot, single-turn API calls (single per-method for unilateral; six per assertion for n=3 bilateral). See [`../prompts/`](../prompts/) for the verbatim prompt strings.
