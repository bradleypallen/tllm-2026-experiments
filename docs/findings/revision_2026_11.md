# Revision 2026-11: what the E1–E3 results say and what §7.2 needs

Working notes for incorporating the November 2026 revision experiments into the paper
(`bbl_paper/paper.tex`, §7.2 = `sec:diagnostic`). Everything here is derivable offline from
`results/derived/2026_11/` and the raw records in `results/2026_11/`; no network needed.
Numbers are pinned by `tests/test_revision_numbers.py` (re-derived from raw) and
`tests/test_paper_numbers.py` (the existing paper numbers). Run `pytest tests/` after any change.

Status of the data: full run 2026-10-18 (harness `28ce38d`, frozen prompts at `e9f3707`), five
models (Claude Opus 4.1 retired 2026-08-05), 30 cells, 18,389 elicitations, provider pins held
everywhere. Analysis commit `f079283`; figure/table artifacts from the commit that adds this file.

## Where things are

| artifact | path | feeds |
|---|---|---|
| F-3′ figure (replaces Fig. 3, `fig:ternary-abstention`) | `results/derived/2026_11/f3prime_four_option.{pdf,png}` + `_audit.json` | §7.2 |
| E1 persistence table (`tab:e1-persistence`) | `results/derived/2026_11/tables/e1_persistence.tex` | §7.2 |
| E3 compliance table (`tab:e3-compliance`) | `results/derived/2026_11/tables/e3_compliance.tex` | §7.2 / appendix |
| Table 1 with coverage (`tab:f1-percell`, replaces the existing one) | `results/derived/2026_11/tables/table1_coverage.tex` | Table 1 |
| P2 gap table (`tab:f3prime-p2`) | `results/derived/2026_11/tables/f3prime_p2.tex` | §7.2 |
| all derived numbers | `results/derived/2026_11/summary.md` and the `T-*.json` files | — |
| regenerate | `python analysis/revision_2026_11.py && python analysis/make_f3prime_figure.py && python analysis/emit_revision_tables.py` | — |

Paper repo convention: figures are read from `bbl_paper/bbl_revisions/` (`migration_tf_to_ff.pdf`,
`tv_diagnostic.pdf` live there). Copies of the F-3′ PDF and the four `.tex` tables are staged in
`bbl_paper/bbl_revisions/` (uncommitted there). Captions in the `.tex` files are marked `TODO`.

## 1. E1 answers the ⟨t,t⟩ question §7.2 left open

§7.2 offers two readings of ⟨t,t⟩ (evidential conflict vs. conditional compliance) and §9 defers
the "valence-swap" experiment. E1 is that experiment.

| arm | prompt | ⟨t,t⟩ persistence, pooled (n valid; excluded parse failures) | 95% CI |
|---|---|---|---|
| E1-0 | frozen verification/refutation prompts re-run, n=3 | **0.589** (338; 8) | [0.447, 0.673] |
| E1-a | joint neutral prompt, with permission clause | **0.044** (298; 48) | [0.018, 0.092] |
| E1-b | joint neutral prompt, without permission clause | **0.024** (295; 51) | [0.008, 0.054] |

- P1, paired E1-0 − E1-a on treatment items: **+0.587 [0.433, 0.664]** (n = 298).
- E1-a − E1-b: +0.017 [−0.004, 0.051] (n = 288). The permission clause does nothing.
- Controls under the joint prompt hold: ⟨t,f⟩ 0.828 [0.732, 0.886] (n = 540), ⟨f,f⟩ 0.901 [0.826, 0.963] (n = 323).
- But the joint prompt itself lowers ⟨t,f⟩ persistence relative to E1-0 (E1-0 − E1-a on ⟨t,f⟩
  controls: +0.133 [0.036, 0.222]); on ⟨f,f⟩ the sign flips (−0.069 [−0.107, −0.033]). Part of
  the treatment effect is format, not valence — say so.

**Where the 346 ⟨t,t⟩ items go under E1-a** (pooled): ⟨t,f⟩ 165, ⟨f,t⟩ 88, ⟨f,f⟩ 32, ⟨t,t⟩ 13,
⟨e,e⟩ 48 (parse failures). Under E1-0: ⟨t,t⟩ 199, ⟨f,t⟩ 64, ⟨t,f⟩ 57, ⟨t,e⟩ 10, ⟨e,e⟩ 8, ⟨f,f⟩ 6.

**Decision the paper has to make.** §9 predicted that under compliance the double commitment
"collapses to ignorance under neutral framing". It collapses to a **one-sided commitment**, not to
⟨f,f⟩ (32 of 346). And the side is not gold-aligned: for gold-FALSE items the refutation side
drops (→⟨t,f⟩, wrong) 46 times vs the verification side 35; for gold-TRUE, refutation drops 119
vs verification 53 (T-E1c). Reword §9's prediction, the §1 mitigation prescription
("reconciliation rather than retrieval"), and the two-readings paragraph of §7.2 accordingly.

Per-model caveat: pooled E1 is **61% Gemini** (181 of 298 valid treatment items). Maverick loses
39 of 64 treatment items to parse failure in E1-a (41 of 64 in E1-b), Scout 7 of 29; their
per-model CIs are not usable alone (Scout P1 CI [−1.000, 0.357]). Report the per-model table
(`tab:e1-persistence`) with the exclusions, not just the pooled number.

## 2. F-3′ replaces Figure 3 — and licenses a stronger sentence

Four-option prompt (TRUE / FALSE / UNCERTAIN / CONFLICTING), all 1,000 frozen items per model,
with the three-option ternary (E2-0) re-run the same day on the baseline subset.

| original value | n valid | UNCERTAIN | CONFLICTING | either | E2-0 UNCERTAIN (same day) |
|---|---|---|---|---|---|
| ⟨t,f⟩ | 1309 | 0.030 | 0.015 | 0.044 | 0.031 (n = 392) |
| ⟨f,t⟩ | 1564 | 0.121 | 0.024 | 0.145 | — |
| ⟨f,f⟩ | 1356 | **0.600** | 0.023 | 0.622 | **0.520** (n = 371) |
| ⟨t,t⟩ | 340 | 0.006 | **0.015** | 0.021 | 0.018 (n = 341) |
| e-bearing (pooled) | 184 of 387 | 0.125 | 0.011 | — | — |

- P2, P(CONFLICTING | ⟨t,t⟩) − P(CONFLICTING | ⟨t,f⟩ ∪ ⟨f,f⟩): **−0.004 [−0.021, 0.013]** (n_tt = 340).
  Offering the CONFLICTING token does not flag the ⟨t,t⟩ items.
- The March 65/3 asymmetry (six models) replicates in the five-model lineup, same day: 62/2
  (E2 "either") and 52/2 (E2-0).
- The §7.2 sentence "a scheme that emits one abstention token for both is conflating opposites"
  can become: even given a token for the conflict, the unilateral scheme does not use it.
- Caveat: the e-bearing cells lose 203 of 387 responses to parse failure in E2 (⟨e,t⟩ 96/143,
  ⟨e,e⟩ 78/120); the figure's "middle" bar is on 184 valid responses. State it or drop the bar.

## 3. E3: compliance base rate

| model | n | c(M) | Wilson 95% | ⟨t,t⟩ share (March) |
|---|---|---|---|---|
| Gemini 2.5 Flash | 100 | 0.150 | [0.093, 0.233] | 0.182 |
| Llama 4 Scout | 98 | 0.031 | [0.011, 0.086] | 0.029 |
| DeepSeek-V3 | 100 | 0.020 | [0.005, 0.070] | 0.018 |
| Qwen 2.5 72B | 100 | 0.020 | [0.005, 0.070] | 0.053 |
| Llama 4 Maverick | 100 | 0.000 | [0.000, 0.037] | 0.064 |

Rank agreement is partial: Gemini tops both; **Maverick is 2nd on ⟨t,t⟩ share but last on
compliance (0/100)**. Five points — table and rank statement, no coefficient. The DeepSeek/Qwen
tie at 0.020 is broken alphabetically in the rank list (stated in the table note).

## 4. Methods and limitations sentences now owed

- **Lineup.** Five models; Opus 4.1 retired on the API 2026-08-05; 346 of the 371 ⟨t,t⟩ items.
  Fig. 3 (six models, March) vs F-3′ (five, October).
- **Parse failures as outcomes.** Maverick fails 52% / 65% of E1-a / E1-b joint elicitations,
  Scout 17% / 31%; modes are chain-of-thought prose exhausting `max_tokens = 32` and
  verification/refutation vocabulary swapped across slots. Raw text is kept for every
  elicitation. Exclusion counts are in `tab:e1-persistence`.
- **Analysis-stage re-parse.** The first-line rule (`analysis/revision_2026_11.py`, dated
  2026-10-18) recovers 7 of 5,091 joint elicitations (all Scout); 725 remain PARSE_FAIL. The
  evaluator's parser and prompts are unchanged; their hashes are in every `.meta.json`.
- **Providers / quantization.** Pinned: DeepInfra (fp4) for DeepSeek, Scout and Qwen;
  DigitalOcean for Maverick; Google AI Studio for Gemini. The March run went through
  `bilateral-truth` unpinned. This is why the same-day E1-0 and E2-0 controls exist.
- **§8 test–retest.** No longer unestimated: E1-0 gives ⟨t,t⟩ 0.589, ⟨t,f⟩ 0.906, ⟨f,f⟩ 0.845
  (same prompts, same day, n = 3). Update §8 and strike the matching §9 item; the
  "re-eliciting under an outcome-neutral prompt" item of §9 is also done (E1-a/E1-b).
- **Still open from the 2026-10-18 verification of the paper against the repo**
  (`docs/methodology.md`): §7 says three elicitations per prompt, but the modal run used one per
  situation; Definition 14's deterministic tiebreak was not the code's (random); the DeepSeek
  25% SimpleQA example drops to 11% when the base ⟨t,f⟩ is taken from the n = 3 run.

## 5. §7.1 inputs that already exist (from the frozen March data)

- T-RK: per-model ⟨t,f⟩→⟨f,f⟩ migration with cluster-bootstrap CIs over benchmark cells;
  orderings whose pairwise CI excludes zero are listed in `summary.md` (nine of fifteen pairs).
- T-TM: full ⟨t,f⟩→NINE transition (pooled: ⟨t,f⟩ 0.700, ⟨f,f⟩ 0.090, ⟨t,t⟩ 0.060, ⟨e,e⟩ 0.056, …).
- T-COV: Table 1 with coverage — `tables/table1_coverage.tex` is a drop-in replacement for
  `tab:f1-percell` (caption marked TODO).
