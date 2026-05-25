# Benchmark Data Transformations for Bilateral Truth Evaluation

This document describes the transformation of four standard NLP benchmarks into the unified format used by every experimental run in this repository.

**Note**: The data generation scripts (`data_generators/*.py`) themselves are not shipped in this repo — they live in the `bilateral-truth` repo at commit `8545986`, under `evaluations/data_generators/`. The canonical dataset files this repo ships are the *outputs* of that pipeline (in [`../datasets/`](../datasets/)) — these are what every result file in [`../results/`](../results/) was computed against. The generators are referenced here for provenance and to allow someone to recreate them from scratch if needed.

## Standard Format Specification

All benchmarks are transformed into a common JSON structure with the following schema:

```json
{
  "metadata": {
    "benchmark": "benchmark_name",
    "version": "dataset_version",
    "total_assertions": integer,
    "generation_timestamp": "ISO-8601 timestamp",
    "source_info": {...}
  },
  "assertions": [
    {
      "assertion_id": "unique_identifier",
      "assertion_text": "statement to evaluate",
      "expected_label": "correct|incorrect",
      "context": {...},
      "metadata": {...}
    }
  ]
}
```

Each assertion represents an atomic claim that can be evaluated for verifiability (can it be confirmed as true?) and refutability (can it be rejected as false?), yielding a bilateral truth value <u,v>.

## Benchmark Transformations

### 1. TruthfulQA (870KB, ~800 assertions)

**Source**: Lin et al. (2022) - Questions testing factual accuracy and truthfulness across multiple categories.

**Transformation Process**:
- **Input**: Original TruthfulQA dataset containing questions with multiple-choice answers and best/correct answer annotations
- **Assertion Generation**: 
  - Each answer option becomes an independent assertion in the form "Q: [question] A: [answer]"
  - Binary labeling: answers marked as "best" or in "correct_answers" → "correct"; all others → "incorrect"
- **Context Preservation**: Original question category (e.g., Misconceptions, Science, History) and source information retained
- **Quality Control**: Filtered to MC1 (single-choice) and MC2 (multiple-choice) tasks only

**Data Generator**: `evaluations/data_generators/truthfulqa_generator.py`

### 2. SimpleQA (14MB, ~4,000 assertions)

**Source**: OpenAI (2024) - Simple factual questions with known correct answers.

**Transformation Process**:
- **Input**: JSONL format with questions, answers, and metadata including subject areas and grades
- **Assertion Generation**:
  - Format 1: Direct question-answer pairs as "Q: [question] A: [answer]"
  - Format 2: Statement form by converting Q&A into declarative sentences
  - Both correct answers and generated incorrect answers (via negation or alternatives) included
- **Context Preservation**: Subject area, grade level, and original problem retained
- **Balanced Dataset**: Ensures roughly equal distribution of correct/incorrect labels

**Data Generator**: `evaluations/data_generators/simpleqa_generator.py`

### 3. MMLU-Pro (115MB, ~12,000 assertions)

**Source**: Hendrycks et al. (2021) - Massive Multitask Language Understanding across 57 subjects.

**Transformation Process**:
- **Input**: CSV files per subject containing questions, multiple choice options (A-J), and correct answer keys
- **Assertion Generation**:
  - Each option transformed into assertion: "Q: [question] A: [option_text]"
  - Options matched against answer key for binary labeling
  - Invalid or missing options filtered out
- **Context Preservation**: Subject category, subcategory, and difficulty level maintained
- **Comprehensive Coverage**: All 57 subjects processed maintaining disciplinary diversity

**Data Generator**: `evaluations/data_generators/mmlupro_generator.py`

### 4. FACTScore (28MB after processing, ~33,820 assertions)

**Source**: Min et al. (2023) - Atomic facts extracted from LLM-generated biographies.

**Transformation Process**:
- **Input**: JSONL files containing LLM-generated biographies with human and model annotations
- **Assertion Generation**:
  - Biographies decomposed into atomic facts using provided annotations
  - Human-annotated facts: Labels derived from S (Supported) → "correct", NS (Not Supported) → "incorrect"
  - Model-generated facts: Included with default "correct" label when human annotations unavailable
- **Multi-Model Coverage**: Processes outputs from ChatGPT, InstructGPT, PerplexityAI, and other models
- **Dual Annotation**: Preserves both LLAMA+NP and ChatGPT labels where available

**Data Generator**: `evaluations/data_generators/factscore_generator.py`

## Reproducibility

### Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Data Generation Pipeline
```bash
# Generate all benchmark datasets
cd evaluations

# TruthfulQA - downloads automatically from HuggingFace
python data_generators/truthfulqa_generator.py

# SimpleQA - downloads automatically from OpenAI
python data_generators/simpleqa_generator.py  

# MMLU-Pro - downloads automatically from HuggingFace
python data_generators/mmlupro_generator.py

# FACTScore - requires manual download from Google Drive
# Download from: https://drive.google.com/drive/folders/1kFey69z8hGXScln01mVxrOhrqgM62X7I
# Extract to: evaluations/factscore_data/raw/full/
python data_generators/factscore_generator.py
```

### Output Files (canonical inputs shipped in this repo)

| File (in `datasets/`) | Approx. count |
|-----------------------|---------------|
| `truthfulqa_complete.json` | ~1,580 |
| `simpleqa_complete.json` | ~21,630 |
| `mmlupro_complete.json` | ~110,225 |
| `factscore_complete.json` | ~33,820 |

(The counts above reflect the actual files in this repo, not the smaller estimates given by the original data-generator documentation.)

### Evaluation Framework
```bash
# Run evaluation on any benchmark (2026 methodology)
python evaluators/proper_benchmark_evaluator.py \
  --dataset datasets/[benchmark]_complete.json \
  --model [model_id] \
  --output-dir results/2026_03/proper/ \
  --samples 250
```

## Key Design Decisions

1. **Atomic Assertions**: All benchmarks decomposed into atomic, self-contained claims suitable for bilateral evaluation
2. **Context Preservation**: Original question/answer context maintained to enable informed evaluation
3. **Binary Labeling**: Multi-class problems reduced to binary (correct/incorrect) for compatibility with bilateral truth values
4. **Metadata Richness**: Extensive metadata preserved for post-hoc analysis and filtering
5. **Format Consistency**: Uniform structure enables single evaluation pipeline across diverse benchmarks

## Validation

Each transformation includes:
- Assertion count verification against expected totals
- Label distribution analysis (correct vs incorrect balance)
- Sample inspection for format compliance
- Metadata completeness checks

The transformation process maintains the semantic content and evaluation objectives of each original benchmark while adapting them for bilateral truth evaluation.