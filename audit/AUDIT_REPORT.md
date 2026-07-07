# Definition-16 fidelity audit — results

**Overall pass rate: 95/100 = 95.0%**  (95% Wilson interval [88.8%, 97.8%])

## Per-benchmark

- factscore: 25/25
- mmlu-pro: 23/25
- simpleqa: 24/25
- truthfulqa: 23/25

## Failure modes (clustered)

- (1) ambiguous elaboration — rows [74]
- (1) capacity is not the same as capacitance — rows [43]
- (1) changes the question — rows [84]
- (1) making a decision is not the same as issuing a decision — rows [48]
- (1) missing “life” in base_text — rows [26]

## Transformation types over PASS rows (human column)

- (none typed)

## Machine-suggestion agreement (diagnostic only)

- suggested ⊆ human on 0/0 typed rows

Sentence branch selected: **primary (rate >= 90%)** → `paper_sentence.tex`.
