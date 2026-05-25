# 2025-09 prompts (verbatim)

Frozen from `bilateral-truth` commit `8545986a51ce1af5d3df9dfc373a8fad58914636` and the 2025-era evaluators that lived in that repo's `evaluations/` directory (`generic_evaluator.py`, `unilateral_evaluator.py`, `unilateral_confidence_evaluator.py`). Do not edit — these are an audit trail.

`{assertion}` is the assertion text; `{context}` is the per-call context block (see [Contexts](#contexts)); `{category}` is the assertion's category field.

---

## Bilateral — verification call

System prompt:
> You are an expert in factual verification. You must respond with only the exact required token sequences.

User prompt:
```
You are tasked with determining whether the following assertion can be verified as true based on available evidence and knowledge.

Assertion: {assertion}

Context: {context}

Your task is to determine if this assertion can be verified. Consider all available evidence, facts, and reliable sources of information.

You must respond with exactly one of these two token sequences:
- VERIFIED (if the assertion can be confirmed as true based on evidence)
- CANNOT VERIFY (if the assertion cannot be confirmed as true, either due to lack of evidence, uncertainty, or because it is false)

Do not provide any explanation or additional text. Respond with only the required token sequence.

Response:
```

Bilateral was run with `samples=1` (no majority voting).

## Bilateral — refutation call

System prompt:
> You are an expert in factual verification. You must respond with only the exact required token sequences.

User prompt:
```
You are tasked with determining whether the following assertion can be refuted (shown to be false) based on available evidence and knowledge.

Assertion: {assertion}

Context: {context}

Your task is to determine if this assertion can be refuted. Consider all available evidence, facts, and reliable sources of information that might contradict the assertion.

You must respond with exactly one of these two token sequences:
- REFUTED (if the assertion can be shown to be false based on evidence)
- CANNOT REFUTE (if the assertion cannot be shown to be false, either due to lack of contradictory evidence, uncertainty, or because it is true)

Do not provide any explanation or additional text. Respond with only the required token sequence.

Response:
```

## Forced unilateral (direct binary)

System prompt:
> You are an expert evaluator determining the correctness of statements.

User prompt:
```
Determine whether the following statement is correct.
Conclude with a single line containing ONLY one of these two words:
CORRECT
INCORRECT

Statement: {assertion}
```

No context block was injected for the unilateral approaches.

## Ternary — confidence-based ("uncertain" prompt style)

System prompt:
> You are an expert evaluator determining the correctness of statements.

User prompt:
```
Determine whether the following statement is correct.

You have three options:
1. CORRECT - if you are confident the statement is true
2. INCORRECT - if you are confident the statement is false
3. UNCERTAIN - if you lack sufficient confidence to determine correctness

Important: Only respond CORRECT or INCORRECT if you are reasonably confident.
If you have doubts or the statement is ambiguous, respond UNCERTAIN.

Conclude with a single line containing ONLY one of these three words:
CORRECT
INCORRECT
UNCERTAIN

Statement: {assertion}
```

**Note:** This is a *self-confidence* prompt — the model is asked to introspect on its own certainty, not to make an evidence judgement. This is **not** equivalent to the 2026 evidence-based ternary; see [`../CHANGELOG.md`](../CHANGELOG.md).

## Numerical confidence

System prompt:
> You are an expert evaluator providing confidence ratings for statement correctness. Always conclude with 'CONFIDENCE: X.X' where X.X is a number between 0.0 and 1.0.

User prompt:
```
Rate your confidence that the following statement is correct.

Provide a single number between 0.0 and 1.0:
- 0.0 = Definitely incorrect
- 0.5 = Completely uncertain
- 1.0 = Definitely correct

Statement: {assertion}

Respond with ONLY this format:
CONFIDENCE: [number]
```

Numerical confidence rating, then thresholded at 0.5, 0.7, and 0.9 to produce three derived approaches. Below threshold = abstention.

## Contexts

The 2025 bilateral evaluator (`generic_evaluator.py`) injects the following context block:

```
This statement is being evaluated for factual accuracy as part of a bilateral truth evaluation benchmark.

Category: {category}
```

The unilateral / ternary / confidence evaluators do **not** inject a context block.

## System prompt (bilateral default)

`generic_evaluator.py` uses this default system prompt for bilateral calls (passed through `zeta()` via the `system_prompt` parameter):

```
You are an expert evaluating factual statements for accuracy across diverse domains. Your task is to determine whether claims can be verified or refuted based on established knowledge.

Focus on:
- Factual accuracy based on authoritative sources
- Current and historical knowledge across all domains
- Precise details including names, dates, numbers, and places
- Distinguishing between correct facts and plausible misinformation

Be especially careful with:
- Statements that sound plausible but contain factual errors
- Claims mixing accurate and inaccurate information
- Details that may be close to correct but are precisely wrong
- Domain-specific expertise requirements
```
