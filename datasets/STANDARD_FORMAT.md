# Standard Dataset Format for Bilateral Truth Evaluation

## Overview
All benchmark datasets must be converted to this standard format for evaluation. The generic evaluator processes only this format.

## Dataset File Structure
```json
{
  "metadata": {
    "benchmark": "truthfulqa|simpleqa|mmlu-pro",
    "version": "string",
    "total_assertions": 1234,
    "generation_timestamp": "2025-08-21T12:34:56Z",
    "source_info": {
      "original_questions": 567,
      "original_dataset": "path/to/source",
      "generation_params": {}
    }
  },
  "assertions": [
    {
      "assertion_id": "unique_string_id",
      "assertion_text": "The statement to be evaluated for truth",
      "expected_label": "correct|incorrect", 
      "context": {
        "category": "string (e.g., Misconceptions, Science)",
        "topic": "string (e.g., Physics, History)",
        "difficulty": "string (optional)",
        "source_question": "original question text",
        "source_answer": "original answer text"
      },
      "metadata": {
        "assertion_type": "original|distractor",
        "generation_method": "string (if distractor)"
      }
    }
  ]
}
```

## Field Specifications

### Required Fields
- `assertion_id`: Unique identifier for the assertion
- `assertion_text`: The statement to evaluate (clean, grammatical sentence)
- `expected_label`: Either "correct" (should be verifiable) or "incorrect" (should be refutable)

### Context Fields
- `category`: High-level category for grouping results
- `topic`: More specific topic for detailed analysis
- `source_question`: The original question this assertion relates to
- `source_answer`: The original answer this assertion is based on

### Metadata Fields
- `assertion_type`: "original" for correct statements from source, "distractor" for generated incorrect statements
- `generation_method`: Method used to generate distractors (if applicable)

## Evaluation Processing
The generic evaluator will:
1. Load this JSON format
2. Process each assertion through bilateral truth evaluation
3. Apply epistemic policy projection
4. Calculate metrics grouped by category/topic
5. Generate standardized output format

## Data Generator Requirements
Each benchmark data generator must:
1. Read the original benchmark dataset
2. Generate assertions (correct + incorrect as needed)
3. Output in this exact JSON format
4. Preserve all necessary context for results analysis

## Benefits
- Single evaluation codebase
- Consistent checkpointing and recovery
- Standardized output format
- Easy to add new benchmarks
- No benchmark-specific evaluation logic