#!/usr/bin/env python3
"""
Modal evaluation suite — 6 models × 4 benchmarks × N=250, [[□p]] with n=3 variants.

Phase 1: Pre-generate situation variants for all 4 datasets (one-time Opus 4.6 cost,
         shared across all evaluation models).
Phase 2: Run modal_evaluator.py for all 6 models × 4 datasets in parallel, reading
         from the pre-generated variant cache.

Usage:
    python run_modal_suite.py                        # full run
    python run_modal_suite.py --skip-pregen          # skip if caches already exist
    python run_modal_suite.py --samples 50           # quick smoke-test
    python run_modal_suite.py --models opus gpt41    # subset of models
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_modal_suite")

SCRIPT_DIR = Path(__file__).parent
BASE = SCRIPT_DIR.parent  # repo root

DATASETS = {
    "truthfulqa": str(BASE / "datasets" / "truthfulqa_complete.json"),
    "simpleqa":   str(BASE / "datasets" / "simpleqa_complete.json"),
    "mmlu-pro":   str(BASE / "datasets" / "mmlupro_complete.json"),
    "factscore":  str(BASE / "datasets" / "factscore_complete.json"),
}

# Maps short label → model ID for the evaluator
MODELS = {
    "opus":     "claude-opus-4-1-20250805",
    "llama":    "meta-llama/llama-4-maverick",
    "scout":    "meta-llama/llama-4-scout",
    "gemini":   "google/gemini-2.5-flash",
    "deepseek": "deepseek/deepseek-chat",
    "qwen":     "qwen/qwen-2.5-72b-instruct",
}

VARIANT_CACHE_DIR = BASE / "variant_cache"
LOG_DIR           = BASE / "logs"
RESULTS_DIR       = BASE / "results" / "2026_03" / "modal"
CHECKPOINT_DIR    = BASE / "checkpoints"


def variant_cache_path(benchmark: str, n_variants: int) -> Path:
    return VARIANT_CACHE_DIR / f"{benchmark}_n{n_variants}_variants.json"


def variant_cache_complete(benchmark: str, n_variants: int, samples: int) -> bool:
    """True if the cache exists and covers at least `samples` assertions."""
    import json
    p = variant_cache_path(benchmark, n_variants)
    if not p.exists():
        return False
    try:
        with open(p) as f:
            d = json.load(f)
        count = len(d.get("variants", {}))
        logger.info("Variant cache %s: %d entries (need %d)", p.name, count, samples)
        return count >= samples
    except Exception:
        return False


def run_pregeneration(benchmark: str, dataset: str, n_variants: int, samples: int,
                      variation_model: str = "claude-opus-4-6") -> int:
    """Run pregenerate_variants.py and return its exit code."""
    out = variant_cache_path(benchmark, n_variants)
    VARIANT_CACHE_DIR.mkdir(exist_ok=True)
    cmd = [
        sys.executable, str(SCRIPT_DIR / "pregenerate_variants.py"),
        "--dataset", dataset,
        "--n-variants", str(n_variants),
        "--samples", str(samples),
        "--output-dir", str(VARIANT_CACHE_DIR),
        "--variation-model", variation_model,
    ]
    log_path = LOG_DIR / f"pregen_{benchmark}_n{n_variants}.log"
    LOG_DIR.mkdir(exist_ok=True)
    logger.info("Pre-generating variants: %s → %s", benchmark, out)
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=BASE)
    if proc.returncode != 0:
        logger.error("Pre-generation FAILED for %s (rc=%d) — see %s", benchmark, proc.returncode, log_path)
    else:
        logger.info("Pre-generation done: %s", benchmark)
    return proc.returncode


def modal_output_path(dataset_stem: str, model_id: str, n_variants: int) -> Path:
    model_safe = model_id.replace("/", "_").replace(":", "_")
    return RESULTS_DIR / f"{dataset_stem}_{model_safe}_n{n_variants}_modal_results.json"


def run_modal_evaluation(
    model_id: str, benchmark: str, dataset: str,
    n_variants: int, samples: int,
) -> subprocess.Popen:
    """Launch modal_evaluator.py as a background subprocess, returning the Popen handle."""
    dataset_stem = Path(dataset).stem
    cache = variant_cache_path(benchmark, n_variants)
    out   = modal_output_path(dataset_stem, model_id, n_variants)

    if out.exists():
        import json
        try:
            with open(out) as f:
                d = json.load(f)
            if d.get("total_samples", 0) >= samples:
                logger.info("SKIP (already complete): %s × %s", model_id, benchmark)
                return None
        except Exception:
            pass

    cmd = [
        sys.executable, str(SCRIPT_DIR / "modal_evaluator.py"),
        "--model", model_id,
        "--dataset", dataset,
        "--n-variants", str(n_variants),
        "--samples", str(samples),
        "--output-dir", str(RESULTS_DIR),
        "--checkpoint-dir", str(CHECKPOINT_DIR),
        "--variant-cache", str(cache),
    ]
    model_safe = model_id.replace("/", "_").replace(":", "_")
    log_path = LOG_DIR / f"modal_{benchmark}_{model_safe}_n{n_variants}.log"
    LOG_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    logger.info("Launching: %s × %s", model_id, benchmark)
    lf = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=BASE)
    proc._log_path = log_path  # stash for status reporting
    return proc


def wait_all(procs: dict, poll_interval: int = 30) -> dict:
    """Poll procs dict {label: Popen} until all finish. Returns {label: returncode}."""
    results = {}
    running = dict(procs)
    while running:
        done = []
        for label, proc in running.items():
            rc = proc.poll()
            if rc is not None:
                done.append(label)
                results[label] = rc
                status = "OK" if rc == 0 else f"FAILED(rc={rc})"
                logger.info("Finished %s → %s  [%s]", label, status, proc._log_path)
        for label in done:
            del running[label]
        if running:
            logger.info(
                "Still running (%d/%d): %s",
                len(running), len(procs), ", ".join(running)
            )
            time.sleep(poll_interval)
    return results


def main():
    parser = argparse.ArgumentParser(description="Run full modal evaluation suite")
    parser.add_argument("--samples", type=int, default=250,
                        help="Assertions per benchmark (default: 250)")
    parser.add_argument("--n-variants", type=int, default=3,
                        help="Situation variants per assertion (default: 3)")
    parser.add_argument("--skip-pregen", action="store_true",
                        help="Skip pre-generation if cache already exists")
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()),
                        default=list(MODELS.keys()),
                        help="Subset of models to run (default: all)")
    parser.add_argument("--benchmarks", nargs="+", choices=list(DATASETS.keys()),
                        default=list(DATASETS.keys()),
                        help="Subset of benchmarks to run (default: all)")
    args = parser.parse_args()

    selected_models    = {k: MODELS[k] for k in args.models}
    selected_benchmarks = {k: DATASETS[k] for k in args.benchmarks}

    # ── Phase 1: Pre-generate variants ──────────────────────────────────────
    logger.info("=== Phase 1: Variant pre-generation ===")
    for bench, dataset in selected_benchmarks.items():
        if args.skip_pregen and variant_cache_complete(bench, args.n_variants, args.samples):
            logger.info("Cache already complete for %s — skipping", bench)
            continue
        rc = run_pregeneration(bench, dataset, args.n_variants, args.samples)
        if rc != 0:
            logger.error("Aborting: pre-generation failed for %s", bench)
            sys.exit(1)

    logger.info("=== Phase 1 complete ===")

    # ── Phase 2: Modal evaluations (parallel) ───────────────────────────────
    logger.info("=== Phase 2: Modal evaluations (%d models × %d benchmarks) ===",
                len(selected_models), len(selected_benchmarks))

    procs = {}
    for bench, dataset in selected_benchmarks.items():
        for short, model_id in selected_models.items():
            label = f"{short}×{bench}"
            proc = run_modal_evaluation(model_id, bench, dataset, args.n_variants, args.samples)
            if proc is not None:
                procs[label] = proc

    if not procs:
        logger.info("All results already exist — nothing to run")
        return

    logger.info("Waiting for %d evaluation jobs...", len(procs))
    results = wait_all(procs, poll_interval=30)

    failed = [lbl for lbl, rc in results.items() if rc != 0]
    if failed:
        logger.error("FAILED jobs: %s", ", ".join(failed))
        sys.exit(1)

    logger.info("=== All modal evaluations complete ===")


if __name__ == "__main__":
    main()
