# Datasets

Input assertion sets used by every experimental run in this repository.

| File | Benchmark identifier | Approx. assertion count |
|------|----------------------|-------------------------|
| `truthfulqa_complete.json` | `truthfulqa` | ~1,580 |
| `simpleqa_complete.json` | `simpleqa` | ~21,630 |
| `mmlupro_complete.json` | `mmlu-pro` | ~110,225 |
| `factscore_complete.json` | `factscore` | ~33,820 |

Each evaluator samples from these with `seed=42` and balanced positive/negative splits. See `STANDARD_FORMAT.md` for the per-assertion schema.

`MANIFEST.json` records SHA-256 hashes so callers can verify integrity.
