# Variant cache

Pre-generated paraphrases (s₁, s₂, s₃) of each assertion's source question, used by the modal evaluator to compute [[□p]] = ⟨min uᵢ, max vᵢ⟩ over four situations (s₀ = original, s₁–s₃ = paraphrases). FACTScore entries carry two paraphrases (three situations). The `n3` in the filenames is the generator's `n_variants=3` setting.

These were generated once with Claude Opus 4.6 (`bilateral_truth.variation_generator.SituationVariationGenerator`). Shipping them lets anyone reproduce modal results without spending Opus credits to regenerate paraphrases — the variation step is the costliest part of pre-processing.

Files match the benchmark identifier:
- `truthfulqa_n3_variants.json`
- `simpleqa_n3_variants.json`
- `mmlupro_n3_variants.json`
- `factscore_n3_variants.json`

To regenerate (only if you change the dataset or seed):
```bash
python evaluators/pregenerate_variants.py \
    --dataset datasets/truthfulqa_complete.json \
    --n-variants 3 \
    --samples 250
```
