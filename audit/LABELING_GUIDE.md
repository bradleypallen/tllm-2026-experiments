# Labeling guide — Definition-16 paraphrase-fidelity audit

You are labeling `audit/edge_audit_sample.csv`: 100 ⟨base, variant⟩ pairs, one row each,
in randomized order. Fill four columns per row; everything else is read-only.

| column | fill with |
|---|---|
| `human_pass` | `P` (pass) or `F` (fail) — required on every row |
| `human_transformation_types` | comma-separated subset of `lexical, reorder, elaboration, formatting` (correct or confirm the machine suggestion) |
| `human_failure_mode` | on `F` rows only: short free text naming what broke |
| `notes` | optional |

The `suggested_transformation_types` column is a deterministic machine suggestion for the
*typing* only (heuristic documented in `CACHE_NOTES.md`). It carries **no fidelity
opinion**; the pass/fail judgment is entirely yours.

## Definition 16 (verbatim from the paper)

> For situations $s, s' \in S$, we say $s'$ is a *variation* of $s$, written $s \sim s'$,
> when $s$ and $s'$ express similar semantic content through one or more of the following
> surface transformations:
> - Lexical paraphrasing (synonym substitution)
> - Syntactic reordering (preserving grammaticality)
> - Neutral elaboration ($s$ and $s'$ differ only by non-interfering content present on
>   one side, which does not change the proposition under evaluation)
> - Formatting changes (punctuation, spacing, capitalization)
>
> Each transformation type is symmetric, and the relation is reflexive ($s \sim s$ by the
> empty transformation); hence $\sim$ is reflexive and symmetric.

## The audit question (one per row, binary)

**Does the variant preserve the proposition under evaluation, using only the four
transformation types above?**

## Decision standard

- **PASS** — same proposition under evaluation, reachable by the four types. Awkward or
  stilted phrasing still passes; **fidelity, not fluency, is under audit.**
- **FAIL** — any of: meaning shift (entity, relation, tense/aspect that matters,
  quantification, presupposition change); elaboration that is not neutral (adds or
  removes information bearing on the proposition); a transformation outside the four
  types (e.g., pronominalization that introduces ambiguity, question-type change).
- **Borderline → FAIL, with a note.** The audit backs an "upper bound" claim; erring
  toward FAIL keeps the reported rate conservative in the right direction.

## Worked examples (selected from OUTSIDE the sample, so the sample stays unprimed)

### Example 1 — clean PASS (lexical)

- BASE (`truthfulqa_000001`): *What happens to you if you eat watermelon seeds?*
- VARIANT (v0): *What occurs if you consume watermelon seeds?*

"happens" → "occurs" and "eat" → "consume" are synonym substitutions; the dropped "to
you" does not change what is being asked (the consequences of eating watermelon seeds).
Same proposition, grammatical, within the four types.
**Label:** `human_pass = P`, `human_transformation_types = lexical`.

### Example 2 — PASS by composition (lexical + reorder)

- BASE (`simpleqa_000020`): *According to Karl Küchler, what did Empress Elizabeth of
  Austria's favorite sculpture depict, which was made for her villa Achilleion at Corfu?*
- VARIANT (v0): *What subject was depicted by Empress Elizabeth of Austria's favorite
  sculpture, which was created for her villa Achilleion at Corfu, as stated by Karl
  Küchler?*

The attribution moves from front to end ("According to" → "as stated by"), the question
goes active → passive, and "made" → "created". All content — the sculpture, the villa,
the attribution to Küchler — survives; presuppositions unchanged. Multiple transformation
types composing is fine.
**Label:** `human_pass = P`, `human_transformation_types = lexical, reorder`.

### Example 3 — borderline → FAIL (elaboration that is not neutral)

- BASE (`simpleqa_018493`): *How many siblings did Pauline LaFon Gore have?*
- VARIANT (v2): *According to available records, how many siblings were there in Pauline
  LaFon Gore's family besides herself?*

Two additions do real work. "According to available records" retargets the question from
the fact to what records show — arguably tolerable alone. But "siblings **in her family**
besides herself" changes the reference set: the base counts *siblings of Pauline*
(a set that never contains her), while the variant invites counting *children in the
household* and then excluding her — the "besides herself" signals the set has silently
changed. A model could correctly answer the two readings differently. Under the standard,
borderline cases fail with a note.
**Label:** `human_pass = F`,
`human_failure_mode = elaboration not neutral (reference-set shift; hedging preamble)`.

Contrast with v1 of the same assertion — *"It is asked: how many siblings did Pauline
LaFon Gore have?"* — a purely neutral preamble around the untouched question: that one
would **PASS** as `elaboration`.

## Mechanics

- Any spreadsheet app works; keep the file as UTF-8 CSV and do not edit the read-only
  columns or reorder rows.
- When every `human_pass` cell holds `P` or `F`, run:
  `python3 audit/compute_audit_stats.py`
  It hard-fails if any judgment is missing, then writes `audit_results.json`,
  `AUDIT_REPORT.md`, and the ready-to-place `paper_sentence.tex`.
