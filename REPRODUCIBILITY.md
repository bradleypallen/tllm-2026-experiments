# Reproducibility

How to reproduce the empirical results in [`results/2026_03/`](results/2026_03/) from this repository.

## Environment setup

This repo ships `datasets/mmlupro_complete.json` (114 MB) via Git LFS. Install LFS once on your machine before cloning:

```bash
brew install git-lfs       # macOS — or use your distro's package manager
git lfs install
```

Then:

```bash
git clone https://github.com/bradleypallen/tllm-2026-experiments.git
cd tllm-2026-experiments
./setup_venv.sh
source venv/bin/activate
```

If you have a clone made before LFS was set up, run `git lfs pull` from inside the clone to fetch the large files.

`setup_venv.sh` creates a Python virtualenv and installs `bilateral-truth` pinned to commit `b71915f` (the 2026-run version) along with provider SDKs.

Copy `.env.example` to `.env` and fill in API keys:

```bash
cp .env.example .env
# edit .env: OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY
```

The 2026 run uses Anthropic for Claude Opus and OpenRouter for everything else (DeepSeek, Gemini, Llama, Qwen).

## Verify input integrity

Each `MANIFEST.json` lists SHA-256 hashes. Confirm your local copy matches:

```bash
python tests/test_manifest_integrity.py
```

## Verify the paper's numbers

`tests/test_paper_numbers.py` re-derives every quantitative claim in the paper (Table 1, Figures 2–3, the §7.2 bootstrap CIs and shares, the §6.3 worked examples) from the raw per-assertion records, independently of `analysis/`. No API calls:

```bash
pytest tests/test_paper_numbers.py
```

## Reproduce the proper-benchmark results (24 cells)

```bash
# Single cell:
python evaluators/proper_benchmark_evaluator.py \
    --model claude-opus-4-1-20250805 \
    --dataset datasets/truthfulqa_complete.json \
    --output-dir results/2026_03/proper/ \
    --samples 250

# All six models × four benchmarks: invoke once per cell from a shell loop, or wrap in your own driver.
```

Outputs are written to `results/2026_03/proper/{benchmark}_{model_safe}_n3_proper_results.json`.

## Reproduce the modal results (24 cells)

Variants are shipped in [`variant_cache/`](variant_cache/); skip pre-generation:

```bash
python evaluators/run_modal_suite.py --samples 250 --skip-pregen
```

To regenerate variants from scratch (requires Anthropic Opus 4.6 access):

```bash
python evaluators/run_modal_suite.py --samples 250
```

The pre-generation step costs significant Opus credits and is the dominant cost of running modal from scratch — keep `--skip-pregen` when possible.

## Cost estimate (rough)

Per-cell cost is dominated by the bilateral evaluation (n=3 majority vote × 2 calls = 6 calls per assertion) plus three single-call approaches (forced unilateral, ternary, confidence). That's 9 API calls per assertion per cell.

- 24 cells × 250 assertions × 9 calls = ~54,000 API calls for `proper_benchmark_evaluator`
- 250 assertions × 2 calls per situation × (18 cells × 4 situations + 6 FACTScore cells × 3 situations) = ~45,000 API calls for `modal_evaluator` (variants pre-generated; one elicitation per situation)
- Variant pre-generation (4 benchmarks × 250 assertions × 3 paraphrases each, 2 for FACTScore; single Opus call per assertion) = ~1,000 Opus calls

Total dollar cost is dominated by the bilateral evaluation across 6 models. Claude Opus dominates the per-token spend; OpenRouter cells are cheaper. As a rough order of magnitude, a full re-run of `proper_benchmark_evaluator` across all 24 cells is on the order of low hundreds of US dollars at March 2026 pricing.

## Partial reproduction

To reproduce a single cell:

```bash
python evaluators/proper_benchmark_evaluator.py \
    --model meta-llama/llama-4-maverick \
    --dataset datasets/truthfulqa_complete.json \
    --output-dir /tmp/repro/ \
    --samples 250

python -c "
import json
ref  = json.load(open('results/2026_03/proper/truthfulqa_meta-llama_llama-4-maverick_n3_proper_results.json'))
test = json.load(open('/tmp/repro/truthfulqa_meta-llama_llama-4-maverick_n3_proper_results.json'))
for app in ref['approaches']:
    print(f\"{app:30s} ref={ref['approaches'][app]['f1_macro']:.4f}  new={test['approaches'][app]['f1_macro']:.4f}\")
"
```

LLM stochasticity means F1 values will move by a few hundredths per cell; the structure of the bilateral distribution and the qualitative findings should be stable.

## Reproducing the 2025 results

The 2025 evaluator scripts live in the `bilateral-truth` repo's `evaluations/` directory at commit `8545986`. They are intentionally not carried into this repo (see [`CHANGELOG.md`](CHANGELOG.md)). To reproduce them you would have to pin `bilateral-truth` to that older SHA and re-run the original scripts:

```bash
git clone https://github.com/bradleypallen/bilateral-truth.git
cd bilateral-truth
git checkout 8545986a51ce1af5d3df9dfc373a8fad58914636
# Then run evaluations/generic_evaluator.py, evaluations/unilateral_evaluator.py, etc.
```

This is provided for completeness; ongoing work targets the 2026 methodology.

## The November 2026 revision experiments (E1-0, E1-a, E1-b, E2, E2-0, E3)

Specified in "BBL Revision Experiments: E1–E3 Specification" (2026-10-18) and run in response to
the LNCS reviews. Prompts: [`prompts/2026_11_revision_prompts.md`](prompts/2026_11_revision_prompts.md).
Item lists: [`datasets/revision_2026_11/`](datasets/revision_2026_11/), built deterministically
(seed 42) from the frozen `results/2026_03/proper/` files by `build_item_lists.py`; their SHA-256s are
in that directory's `MANIFEST.json` and selection counts in `SELECTION.json`.

Five of the paper's six models are run. **Claude Opus 4.1 (`claude-opus-4-1-20250805`) was retired
on the Claude API on 2026-08-05** and is excluded; it contributed 25 of the 371 ⟨t,t⟩ items.

Calls go directly to OpenRouter's OpenAI-compatible endpoint so that the serving provider can be
pinned and recorded; prompts, temperature (0) and context handling are those of the 2026-03 run.
Every result file is accompanied by a `.meta.json` with the harness commit, prompt hashes, item-list
hash, full request parameters, pinned provider, and the `provider` and `model` strings the endpoint
returned. Raw response text is kept for every elicitation.

```bash
# 0. item lists (already committed; re-running must reproduce the MANIFEST hashes)
python datasets/revision_2026_11/build_item_lists.py
python -m pytest tests/test_revision_2026_11.py -q

# 1. offline dry run of the whole plan (mock client, no calls)
python evaluators/run_revision_suite.py --dry-run

# 2. live smoke test: 5 items per cell, provider recorded but not pinned (375 calls:
#    5 items x 15 calls/item [E1-0 6, E1-a 3, E1-b 3, E2 1, E2-0 1, E3 1] x 5 models)
#    -> inspect results/2026_11/smoke/*/*.meta.json: providers_seen, models_returned, parse_fail_rate
python evaluators/run_revision_suite.py --smoke

# 3. full run (Qwen first; E2, E2-0, E1-0, E1-a, E1-b, E3), resumable
python evaluators/run_revision_suite.py            # add --parallel to run the four non-Qwen models concurrently

# single cell, e.g. with an explicit provider pin
python evaluators/revision_2026_11_evaluator.py --mode e1-a --model deepseek/deepseek-chat --provider <slug>
```

Outputs: `results/2026_11/<mode>/<mode>_<model_safe>.jsonl` (+ `.meta.json`). Call counts at the
committed item lists: E1-0 ≈ 6,700; E1-a ≈ 4,050; E1-b ≈ 1,040; E2 5,000; E2-0 ≈ 1,110; E3 500;
total ≈ 18,400.
