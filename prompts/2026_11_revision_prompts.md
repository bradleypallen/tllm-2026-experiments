# 2026-11 revision prompts (verbatim)

Prompts for the revision experiments E1-0, E1-a, E1-b, E2, E2-0 and E3, specified in
"BBL Revision Experiments: E1–E3 Specification" (2026-10-18). Source of truth is
`evaluators/revision_2026_11_prompts.py`; `tests/test_revision_2026_11.py` checks that this file
and that module agree. Do not edit either without the other — these are an audit trail.

`{assertion}` is the assertion text; `{context}` is the per-assertion `source_question` field
(s₀), exactly as in `2026_prompts.md`.

All calls: temperature 0, `max_tokens` 32, one provider pinned per model (see `request_params`
in each result file).

---

## Which prompt each arm uses

| Arm | System prompt | User prompt | n |
| --- | --- | --- | --- |
| E1-0 | Bilateral default (2026-03) | Verification + Refutation (2026-03, verbatim) | 3 each |
| E1-a | Bilateral default (2026-03) | Joint, with permission clause | 3 |
| E1-b | Bilateral default (2026-03) | Joint, permission clause removed | 3 |
| E2 | Four-option system prompt | Four-option | 1 |
| E2-0 | Ternary system prompt (2026-03) | Ternary (2026-03, verbatim) | 1 |
| E3 | Bilateral default (2026-03) | Refutation on robust-TRUE items; Verification on robust-FALSE items (2026-03, verbatim) | 1 |

The 2026-03 bilateral default system prompt, verification, refutation and ternary prompts are
reproduced in `2026_prompts.md` and are not repeated here.

## E1 joint prompt (E1-a)

The four token definitions are copied word for word from the 2026-03 verification and refutation
prompts. The sentence marked ‹permission clause› is present in E1-a and deleted in E1-b; nothing
else differs between the two arms.

```
You are tasked with assessing the following assertion on two independent questions: whether it can be verified as true based on available evidence and knowledge, and whether it can be refuted (shown to be false) based on available evidence and knowledge.

Assertion: {assertion}

Context: {context}

Consider all available evidence, facts, and reliable sources of information. The two questions are independent: it is acceptable to answer that the assertion can be neither verified nor refuted, and it is acceptable to answer that it can be both verified and refuted if the evidence genuinely points both ways.

For the first question, respond with exactly one of:
- VERIFIED (if the assertion can be confirmed as true based on evidence)
- CANNOT VERIFY (if the assertion cannot be confirmed as true, either due to lack of evidence, uncertainty, or because it is false)

For the second question, respond with exactly one of:
- REFUTED (if the assertion can be shown to be false based on evidence)
- CANNOT REFUTE (if the assertion cannot be shown to be false, either due to lack of contradictory evidence, uncertainty, or because it is true)

Respond on one line in exactly this format, with no explanation or additional text:
VERIFICATION: <token>; REFUTATION: <token>

Response:
```

‹permission clause› = "The two questions are independent: it is acceptable to answer that the
assertion can be neither verified nor refuted, and it is acceptable to answer that it can be both
verified and refuted if the evidence genuinely points both ways."

## E1 joint prompt (E1-b)

Identical to E1-a with the permission clause deleted, so the fifth paragraph reads only:

```
Consider all available evidence, facts, and reliable sources of information.
```

## E2 four-option prompt

The 2026-03 ternary prompt with three edits and nothing else: (1) the first sentence gains the
clause ", or supported and contradicted by conflicting evidence"; (2) one option is appended in the
same style as the existing three; (3) the token lists in the system and user prompts are extended.

System prompt:
> You are an expert fact-checker evaluating claims based on evidence. Respond with only TRUE, FALSE, UNCERTAIN, or CONFLICTING.

User prompt:
```
Based on available evidence and knowledge, determine whether the following statement is supported by evidence, contradicted by evidence, undetermined, or supported and contradicted by conflicting evidence.

Context: {context}

Statement: {assertion}

Consider all available facts and reliable sources. You must respond with exactly one of:
- TRUE (if evidence supports the statement as correct)
- FALSE (if evidence contradicts the statement as incorrect)
- UNCERTAIN (if there is insufficient evidence to either support or refute it)
- CONFLICTING (if there is evidence that supports the statement and evidence that contradicts it)

Do not provide any explanation. Respond with only TRUE, FALSE, UNCERTAIN, or CONFLICTING.

Response:
```

## Parsing (all arms)

One normaliser: strip leading and trailing whitespace; strip one trailing period; collapse internal
whitespace to single spaces; upper-case. Then:

- Single-token arms (E1-0, E2, E2-0, E3): exact match against the arm's legal set, else `PARSE_FAIL`.
- Joint arms (E1-a, E1-b): the response must *begin* with
  `VERIFICATION: <V>[;|,|newline] REFUTATION: <R>` with `<V>` in {VERIFIED, CANNOT VERIFY} and
  `<R>` in {REFUTED, CANNOT REFUTE}. Text after that answer (an explanation the model appended
  after a line break) is ignored and the elicitation is flagged `trailing_text`. Leading text, a
  missing or swapped field, an illegal token, or two alternative answers is `PARSE_FAIL` for the
  whole elicitation.
  *Widened 2026-10-18*, after the live smoke test showed Llama 4 Scout appending explanations to
  otherwise well-formed answers (the initial rule required the whole response to be the answer).
  The widening is applied uniformly to all models by re-parsing stored raw text in
  `analysis/revision_2026_11.py`; `parsed` fields written by the evaluator before the change
  reflect the strict rule and are superseded by the re-parse. Scout also sometimes reasons in
  prose instead of answering; those remain `PARSE_FAIL` and their rate is reported per model.
- Component mapping: VERIFIED → u = t, CANNOT VERIFY → u = f, REFUTED → v = t, CANNOT REFUTE → v = f;
  `PARSE_FAIL` or `API_ERROR` → e for that elicitation.
- Majority vote per component over n = 3; a component with no strict majority is e.
- `API_ERROR` is recorded after 3 failed attempts with exponential backoff and is never folded
  into an indeterminate value in the analysis.

Raw response text is stored for every elicitation, so re-parsing under a widened normaliser never
requires new calls.
