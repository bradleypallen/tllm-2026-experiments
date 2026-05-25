#!/usr/bin/env python3
"""
Situation Variant Pre-generator

Generates meaning-preserving situation variants for all assertions in a
standard-format dataset using Claude Opus 4.6, saving a shared cache file
that all modal_evaluator.py runs can read.

Run once per benchmark before starting model evaluations (from repo root):

    python evaluators/pregenerate_variants.py \\
        --dataset datasets/truthfulqa_complete.json \\
        --samples 250 \\
        --n-variants 3 \\
        --output-dir variant_cache/

This produces:
    variant_cache/truthfulqa_n3_variants.json

All subsequent modal_evaluator.py runs pass --variant-cache to this file,
so Opus 4.6 is called once per question rather than once per model run.
"""

import json
import time
import argparse
import random
import sys
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bilateral_truth.variation_generator import SituationVariationGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pregenerate_variants")

try:
    from dotenv import load_dotenv
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass


def load_dataset(dataset_path: str) -> dict:
    p = Path(dataset_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    with open(p) as f:
        dataset = json.load(f)
    if "metadata" not in dataset or "assertions" not in dataset:
        raise ValueError("Invalid dataset format")
    logger.info(
        "Loaded %s: %d assertions",
        dataset["metadata"]["benchmark"],
        dataset["metadata"]["total_assertions"],
    )
    return dataset


def sample_assertions(dataset: dict, max_samples: int, seed: int = 42) -> list:
    """Balanced sampling matching modal_evaluator.py and generic_evaluator.py."""
    all_assertions = dataset["assertions"]
    if max_samples is None or max_samples >= len(all_assertions):
        return all_assertions

    random.seed(seed)
    positive = [a for a in all_assertions if a["expected_label"] == "correct"]
    negative = [a for a in all_assertions if a["expected_label"] == "incorrect"]
    half = max_samples // 2
    pos_n = min(half + max_samples % 2, len(positive))
    neg_n = min(max_samples - pos_n, len(negative))
    sampled = random.sample(positive, pos_n) + random.sample(negative, neg_n)
    random.shuffle(sampled)
    logger.info("Sampled %d assertions (%d positive, %d negative)", len(sampled), pos_n, neg_n)
    return sampled


def load_existing_cache(output_path: Path) -> dict:
    """Load partially-completed cache for resumption."""
    if not output_path.exists():
        return {}
    try:
        with open(output_path) as f:
            data = json.load(f)
        existing = data.get("variants", {})
        logger.info("Resuming: %d variants already generated", len(existing))
        return existing
    except Exception as e:
        logger.warning("Could not load existing cache (%s) — starting fresh", e)
        return {}


def save_cache(output_path: Path, variants: dict, metadata: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "metadata": {**metadata, "total_generated": len(variants)},
        "variants": variants,
    }
    # Atomic write via temp file
    tmp = output_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-generate situation variants for modal evaluation"
    )
    parser.add_argument("--dataset", required=True,
                        help="Path to standard-format dataset JSON")
    parser.add_argument("--n-variants", type=int, default=3,
                        help="Variants per situation (default: 3)")
    parser.add_argument("--samples", type=int,
                        help="Max assertions to process (default: all)")
    parser.add_argument("--output-dir", default="variant_cache",
                        help="Directory for output cache files (default: variant_cache)")
    parser.add_argument("--variation-model", default="claude-opus-4-6",
                        help="Model for variant generation (default: claude-opus-4-6)")
    parser.add_argument("--save-interval", type=int, default=25,
                        help="Save progress every N assertions (default: 25)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    benchmark = dataset["metadata"]["benchmark"]
    assertions = sample_assertions(dataset, args.samples, seed=args.seed)
    total = len(assertions)

    output_path = Path(args.output_dir) / f"{benchmark}_n{args.n_variants}_variants.json"

    metadata = {
        "benchmark": benchmark,
        "n_variants": args.n_variants,
        "variation_model": args.variation_model,
        "dataset_path": args.dataset,
        "total_assertions": total,
        "seed": args.seed,
        "generation_started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Resume from partial cache if it exists
    variants = load_existing_cache(output_path)
    already_done = set(variants.keys())

    gen = SituationVariationGenerator(model=args.variation_model)

    logger.info(
        "Pre-generating variants: %d assertions, n=%d, model=%s",
        total, args.n_variants, args.variation_model
    )
    logger.info("Output: %s", output_path)

    start_time = time.time()
    n_generated = 0
    n_skipped = 0

    for i, assertion_data in enumerate(assertions):
        assertion_id = assertion_data["assertion_id"]
        source_question = assertion_data["context"].get("source_question", "")

        if assertion_id in already_done:
            n_skipped += 1
            continue

        if not source_question:
            logger.warning(
                "No source_question for %s — storing empty variants", assertion_id
            )
            variants[assertion_id] = {
                "source_question": "",
                "variants": [],
            }
            n_generated += 1
            continue

        elapsed = time.time() - start_time
        done_so_far = n_generated + n_skipped
        eta = (elapsed / done_so_far * (total - done_so_far)) if done_so_far > 0 else 0

        logger.info(
            "[%d/%d] %s | ETA %.0fs | %s",
            i + 1, total,
            assertion_id,
            eta,
            source_question[:60],
        )

        # n_variants includes s₀; generate n_variants-1 paraphrases (s₁..s_{n-1})
        generated = gen.generate(source_question, n=args.n_variants - 1)
        variants[assertion_id] = {
            "source_question": source_question,
            "variants": generated,
        }
        n_generated += 1

        for j, v in enumerate(generated, 1):
            logger.info("  %d. %s", j, v[:80])

        if n_generated % args.save_interval == 0:
            save_cache(output_path, variants, metadata)
            logger.info("Progress saved (%d/%d)", len(variants), total)

    # Final save
    metadata["generation_completed"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    metadata["total_generated"] = n_generated
    metadata["total_skipped"] = n_skipped
    save_cache(output_path, variants, metadata)

    total_time = time.time() - start_time
    logger.info(
        "Done: %d generated, %d skipped in %.1fs → %s",
        n_generated, n_skipped, total_time, output_path
    )


if __name__ == "__main__":
    main()
