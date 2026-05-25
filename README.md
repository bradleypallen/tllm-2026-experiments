# tllm-2026-experiments

Dataset of record for empirical experiments on bilateral modal logic for LLM factuality evaluation (BBL paper, 2026). This repository contains the canonical input datasets, evaluator scripts, frozen result snapshots, and analysis code for two experimental runs:

- **2026-03** — current canonical run: 6 models × 4 benchmarks, N=250, evidence-based ternary, n=3 bilateral majority voting, modal evaluation included.
- **2025-09** — earlier run preserved for comparison: 7 models × 4 benchmarks, N=1000, **confidence-based** ternary, n=1 bilateral, no modal.

The two runs are not directly comparable on the "ternary" approach — see [`CHANGELOG.md`](CHANGELOG.md) and [`docs/methodology.md`](docs/methodology.md). The 2025 results are kept because they were the basis of an earlier statement in the paper draft that subsequent work led us to revise.

## What's inside

| Path | What |
|------|------|
| [`datasets/`](datasets/) | Four input assertion sets (TruthfulQA, SimpleQA, MMLU-Pro, FACTScore) in a unified format. |
| [`variant_cache/`](variant_cache/) | Pre-generated paraphrases used by the modal evaluator. |
| [`evaluators/`](evaluators/) | The 2026-run evaluator scripts (`proper_benchmark_evaluator.py`, `modal_evaluator.py`, etc.). |
| [`prompts/`](prompts/) | Verbatim prompt strings used by each run (audit trail). |
| [`results/2025_09/`](results/2025_09/) | Frozen 2025 results (classical / unilateral_direct / unilateral_uncertain / unilateral_confidence). |
| [`results/2026_03/`](results/2026_03/) | Frozen 2026 results (proper / modal). |
| [`results/derived/`](results/derived/) | Figures and tables produced from the result snapshots. |
| [`analysis/`](analysis/) | Visualization scripts (`visualize_*.py`). |
| [`docs/`](docs/) | Methodology, benchmark provenance, curated findings. |
| [`talk/`](talk/) | Presentation materials. |
| [`tests/`](tests/) | Smoke tests for evaluators and result schemas. |

## Quick start

```bash
git clone https://github.com/bradleyallen/tllm-2026-experiments.git
cd tllm-2026-experiments
./setup_venv.sh
source venv/bin/activate
cp .env.example .env  # then add API keys
```

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for end-to-end reproduction commands and API cost estimates.

## Headline result (2026)

Across the six models and four benchmarks, bilateral-classical (D = {⟨t,f⟩}) and the evidence-based ternary baseline agree closely on F1-macro (median |Δ| = 0.022, mean 0.041), with the largest per-cell gaps on SimpleQA where the classical coverage policy abstains on ⟨f,f⟩-valued assertions that the ternary scheme still commits on.

**The case for bilateral valuation is not one of accuracy parity but of diagnostic resolution**: the ⟨u,v⟩ pair distinguishes ignorance (⟨f,f⟩) from contradiction (⟨t,t⟩) from asymmetric partial knowledge (⟨t,e⟩, ⟨e,f⟩) — epistemic states with no representation in unilateral or confidence-based frameworks.

## Dependencies

This repo pins [`bilateral-truth`](https://github.com/bradleyallen/bilateral-truth) to a specific commit per run, since the package's prompt strings and sampling defaults changed between runs (see [`CHANGELOG.md`](CHANGELOG.md)). The 2026 run pins commit `b71915f`; reproducing the 2025 run requires switching to commit `8545986`.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT. See [`LICENSE`](LICENSE).
