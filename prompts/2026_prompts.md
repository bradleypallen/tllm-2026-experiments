# 2026-03 prompts (verbatim)

Frozen from `bilateral-truth` commit `b71915fda6faf252cdb506294e0b9c5e04511d5e`, with the `proper_benchmark_evaluator.py` driver passing `context=source_question` for every call. Do not edit — these are an audit trail.

`{assertion}` is the assertion text; `{context}` is the per-assertion `source_question` field from the dataset.

---

## Bilateral — verification call (unchanged from 2025)

System prompt:
> You are an expert in factual verification. You must respond with only the exact required token sequences.

(or the caller's `system_prompt` argument if provided — `proper_benchmark_evaluator.py` overrides with the default bilateral system prompt below.)

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

Bilateral was run with `samples=3` and majority voting; each sample is one verification + one refutation call.

## Bilateral — refutation call (unchanged from 2025)

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

## Forced unilateral

System prompt:
> You are an expert fact-checker. You must respond with only TRUE or FALSE.

User prompt:
```
Determine whether the following statement is correct or incorrect.

Context: {context}

Statement: {assertion}

You must respond with exactly one of:
- TRUE (if the statement is correct)
- FALSE (if the statement is incorrect)

Do not provide any explanation. Respond with only TRUE or FALSE.

Response:
```

## Ternary — evidence-based

System prompt:
> You are an expert fact-checker evaluating claims based on evidence. Respond with only TRUE, FALSE, or UNCERTAIN.

User prompt:
```
Based on available evidence and knowledge, determine whether the following statement is supported by evidence, contradicted by evidence, or undetermined.

Context: {context}

Statement: {assertion}

Consider all available facts and reliable sources. You must respond with exactly one of:
- TRUE (if evidence supports the statement as correct)
- FALSE (if evidence contradicts the statement as incorrect)
- UNCERTAIN (if there is insufficient evidence to either support or refute it)

Do not provide any explanation. Respond with only TRUE, FALSE, or UNCERTAIN.

Response:
```

**Note:** This is an *evidence judgement* prompt — same epistemic frame as the bilateral verification/refutation prompts. It is **not** equivalent to the 2025 confidence-self-report ternary; see [`../CHANGELOG.md`](../CHANGELOG.md).

## Numerical confidence

System prompt:
> You are an expert evaluator. Always respond with 'CONFIDENCE: X.X' where X.X is between 0.0 and 1.0.

User prompt:
```
Rate your confidence that the following statement is correct.

Context: {context}

Statement: {assertion}

Provide a single number between 0.0 and 1.0:
- 0.0 = Definitely incorrect
- 0.5 = Completely uncertain
- 1.0 = Definitely correct

Respond with ONLY: CONFIDENCE: [number]

Response:
```

Thresholded at 0.5 only (the 2025 run additionally reported 0.7 and 0.9; the 2026 driver records the raw confidence score so post-hoc thresholding at any level is possible from the result JSON's `detailed_results[*].confidence_score` field).

## Context

For all five approaches, `{context}` is the assertion's `source_question` field from the dataset (s₀ in the BBL formalism). The category field is **not** appended; the generic factual-accuracy preamble from 2025 is **not** used.

## Bilateral system prompt (default in `proper_benchmark_evaluator.py`)

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

The forced-unilateral, ternary, and confidence approaches use their own per-method system prompts (shown above); the bilateral-default system prompt above applies only to the verification + refutation calls.
