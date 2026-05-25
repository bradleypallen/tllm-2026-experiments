#!/usr/bin/env python3
"""
Proper Benchmark Evaluator — BBL vs. Alternative Epistemic Approaches

For each assertion evaluates five approaches in a single pass (5 API calls),
all using s₀ = source_question as the situation context:

  1. bilateral  — VM(s₀, p)  →  <u,v>  →  projected under 3 epistemic policies
                  (classical, paracomplete, paraconsistent)
  2. forced_unilateral  — forced binary TRUE / FALSE (no abstention)
  3. ternary     — TRUE / FALSE / UNCERTAIN (abstain on UNCERTAIN)
  4. confidence  — numerical 0.0–1.0 → thresholded at 0.5

All metrics: coverage (fraction not abstaining), F1-macro, accuracy-on-covered.

Run (from repo root):
    python evaluators/proper_benchmark_evaluator.py \\
        --model claude-opus-4-1-20250805 \\
        --dataset datasets/truthfulqa_complete.json \\
        --output-dir results/2026_03/proper/ \\
        --samples 250

Output:
    results/2026_03/proper/{benchmark}_{model}_n3_proper_results.json
    checkpoints/proper_checkpoint_{model}_{hash}.json
"""

import json
import time
import argparse
import hashlib
import random
import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bilateral_truth.model_router import ModelRouter
from bilateral_truth.assertions import Assertion
from bilateral_truth.truth_values import (
    GeneralizedTruthValue,
    TruthValueComponent,
    EpistemicPolicy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("proper_benchmark_evaluator")

try:
    from dotenv import load_dotenv
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info("Loaded environment variables from %s", env_path)
except ImportError:
    logger.warning("python-dotenv not installed; relying on shell environment")


# ──────────────────────────────────────────────────────────────────────────────
# Scoring helpers
# ──────────────────────────────────────────────────────────────────────────────

def _tv_to_str(tv: GeneralizedTruthValue) -> str:
    """Compact string representation e.g. '<t,f>'."""
    c = {
        TruthValueComponent.TRUE: "t",
        TruthValueComponent.FALSE: "f",
        TruthValueComponent.UNDEFINED: "e",
    }
    return f"<{c[tv.u]},{c[tv.v]}>"


def _component_label(c: TruthValueComponent) -> Optional[bool]:
    """Map TruthValueComponent to bool prediction (None = abstain)."""
    if c == TruthValueComponent.TRUE:
        return True
    elif c == TruthValueComponent.FALSE:
        return False
    return None  # UNDEFINED → abstain


def compute_metrics(
    predictions: List[Optional[bool]],
    ground_truth: List[bool],
) -> Dict[str, float]:
    """Compute coverage, accuracy-on-covered, and macro-F1-on-covered.

    Args:
        predictions: List of predicted values (True/False/None); None = abstain.
        ground_truth: List of boolean ground-truth labels.

    Returns:
        Dict with keys: coverage, accuracy, f1_macro, tp, tn, fp, fn, abstained.
    """
    assert len(predictions) == len(ground_truth)
    tp = tn = fp = fn = abstained = 0
    for pred, gt in zip(predictions, ground_truth):
        if pred is None:
            abstained += 1
        elif pred and gt:
            tp += 1
        elif not pred and not gt:
            tn += 1
        elif pred and not gt:
            fp += 1
        else:
            fn += 1

    total = len(predictions)
    covered = total - abstained
    coverage = covered / total if total > 0 else 0.0
    accuracy = (tp + tn) / covered if covered > 0 else 0.0

    # Macro-F1 on covered examples
    precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_pos = (
        2 * precision_pos * recall_pos / (precision_pos + recall_pos)
        if (precision_pos + recall_pos) > 0 else 0.0
    )
    precision_neg = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_neg = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_neg = (
        2 * precision_neg * recall_neg / (precision_neg + recall_neg)
        if (precision_neg + recall_neg) > 0 else 0.0
    )
    f1_macro = (f1_pos + f1_neg) / 2

    return {
        "coverage": round(coverage, 6),
        "accuracy": round(accuracy, 6),
        "f1_macro": round(f1_macro, 6),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "abstained": abstained,
        "covered": covered,
        "total": total,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluator class
# ──────────────────────────────────────────────────────────────────────────────

class ProperBenchmarkEvaluator:
    """Evaluates all four epistemic approaches in a single pass per assertion.

    The s₀ situation context for each assertion is source_question from the
    benchmark dataset — this is the base situation in the BBL paper formalism.

    API calls per assertion:
      - bilateral_samples × (verification + refutation)  → majority vote → 3 policies
      - forced_unilateral call
      - ternary call
      - confidence call

    bilateral_samples=3 matches the paper's methodology.
    """

    CHECKPOINT_INTERVAL = 25  # assertions

    def __init__(
        self,
        model_name: str,
        dataset_path: str,
        checkpoint_dir: str = "checkpoints",
        output_dir: str = "results",
        system_prompt: Optional[str] = None,
        bilateral_samples: int = 3,
    ):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.bilateral_samples = bilateral_samples
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT

        self.evaluator = ModelRouter.create_evaluator(model_name)
        self.dataset = self._load_dataset()

        benchmark = self.dataset["metadata"]["benchmark"]
        self._benchmark = benchmark

        # Per-assertion raw results — accumulated during the run
        self._detailed: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Dataset / sampling
    # ------------------------------------------------------------------

    def _load_dataset(self) -> Dict[str, Any]:
        p = Path(self.dataset_path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        with open(p) as f:
            d = json.load(f)
        if "metadata" not in d or "assertions" not in d:
            raise ValueError("Invalid dataset format")
        logger.info(
            "Loaded %s: %d assertions",
            d["metadata"]["benchmark"],
            d["metadata"]["total_assertions"],
        )
        return d

    @staticmethod
    def _sample_assertions(
        all_assertions: List[Dict], max_samples: Optional[int], seed: int = 42
    ) -> List[Dict]:
        """Balanced sampling matching the other evaluators (seed=42)."""
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
        logger.info(
            "Sampled %d assertions (%d positive, %d negative)",
            len(sampled), pos_n, neg_n,
        )
        return sampled

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def _checkpoint_path(self) -> Path:
        model_safe = self.model_name.replace("/", "_").replace(":", "_")
        h = hashlib.md5(
            f"{self.dataset_path}{self.system_prompt}{self.bilateral_samples}".encode()
        ).hexdigest()[:8]
        return self.checkpoint_dir / f"proper_checkpoint_{model_safe}_{h}.json"

    def _save_checkpoint(self, current_idx: int, start_time: float) -> None:
        data = {
            "model_name": self.model_name,
            "dataset_path": self.dataset_path,
            "system_prompt": self.system_prompt,
            "bilateral_samples": self.bilateral_samples,
            "current_assertion_idx": current_idx,
            "total_completed": len(self._detailed),
            "checkpoint_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": time.time() - start_time,
            "detailed_results": self._detailed,
        }
        path = self._checkpoint_path()
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.replace(path)
        logger.info(
            "Checkpoint saved: %d evaluations at index %d → %s",
            len(self._detailed), current_idx, path,
        )

    def _load_checkpoint(self) -> Optional[Dict]:
        path = self._checkpoint_path()
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            if (data.get("model_name") != self.model_name
                    or data.get("dataset_path") != self.dataset_path
                    or data.get("system_prompt") != self.system_prompt
                    or data.get("bilateral_samples") != self.bilateral_samples):
                logger.warning("Checkpoint found but config mismatch — starting fresh")
                return None
            logger.info(
                "Checkpoint found: %d evaluations completed (%s)",
                data["total_completed"], data["checkpoint_timestamp"],
            )
            return data
        except Exception as e:
            logger.warning("Cannot load checkpoint (%s) — starting fresh", e)
            return None

    def _delete_checkpoint(self) -> None:
        path = self._checkpoint_path()
        if path.exists():
            path.unlink()
            logger.info("Checkpoint deleted: %s", path)

    # ------------------------------------------------------------------
    # Single-assertion evaluation (5 API calls)
    # ------------------------------------------------------------------

    def _evaluate_one(
        self,
        assertion_text: str,
        source_question: Optional[str],
        assertion_id: str,
        expected_label: str,
    ) -> Dict[str, Any]:
        """Run all five approaches for one assertion.

        Uses source_question as the s₀ situation context for all calls.
        Returns a dict with all raw and projected results.
        """
        context = source_question if source_question else None
        assertion = Assertion(assertion_text)
        ground_truth = expected_label == "correct"

        result: Dict[str, Any] = {
            "assertion_id": assertion_id,
            "assertion_text": assertion_text,
            "source_question": source_question or "",
            "expected_label": expected_label,
            "ground_truth": ground_truth,
        }

        # ── 1 & 2. Bilateral (majority vote over bilateral_samples) ──────
        try:
            tv = self.evaluator.evaluate_bilateral(
                assertion, samples=self.bilateral_samples,
                system_prompt=self.system_prompt,
                context=context,
            )
        except Exception as e:
            logger.error(
                "Bilateral evaluation failed for %s: %s", assertion_id, e
            )
            tv = GeneralizedTruthValue(
                TruthValueComponent.UNDEFINED, TruthValueComponent.UNDEFINED
            )

        tv_str = _tv_to_str(tv)
        result["bilateral_tv"] = tv_str

        for policy in (
            EpistemicPolicy.CLASSICAL,
            EpistemicPolicy.PARACOMPLETE,
            EpistemicPolicy.PARACONSISTENT,
        ):
            projected = tv.project(policy)
            prediction = _component_label(projected)
            key = f"bilateral_{policy.value}"
            result[f"{key}_projected"] = (
                "TRUE" if prediction is True
                else "FALSE" if prediction is False
                else "ABSTAIN"
            )
            result[f"{key}_correct"] = (
                None if prediction is None else (prediction == ground_truth)
            )

        # ── 3. Forced unilateral ──────────────────────────────────────
        fu_component = self.evaluator.evaluate_forced_unilateral(
            assertion, context=context
        )
        fu_prediction = _component_label(fu_component)
        result["forced_unilateral_prediction"] = (
            "TRUE" if fu_prediction else "FALSE"
        )
        result["forced_unilateral_correct"] = (fu_prediction == ground_truth)

        # ── 4. Ternary ────────────────────────────────────────────────
        tern_component = self.evaluator.evaluate_ternary(
            assertion, context=context
        )
        tern_prediction = _component_label(tern_component)
        result["ternary_prediction"] = (
            "TRUE" if tern_prediction is True
            else "FALSE" if tern_prediction is False
            else "ABSTAIN"
        )
        result["ternary_correct"] = (
            None if tern_prediction is None
            else (tern_prediction == ground_truth)
        )

        # ── 5. Confidence ─────────────────────────────────────────────
        confidence = self.evaluator.evaluate_confidence(
            assertion, context=context
        )
        conf_prediction = confidence >= 0.5
        result["confidence_score"] = round(confidence, 4)
        result["confidence_05_prediction"] = "TRUE" if conf_prediction else "FALSE"
        result["confidence_05_correct"] = (conf_prediction == ground_truth)

        return result

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(
        self,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Run the full evaluation and return the result dict."""
        all_assertions = self.dataset["assertions"]
        sampled = self._sample_assertions(all_assertions, max_samples, seed=seed)
        total = len(sampled)

        # Try to resume from checkpoint
        checkpoint_data = self._load_checkpoint()
        if checkpoint_data:
            self._detailed = checkpoint_data["detailed_results"]
        completed_ids = {r["assertion_id"] for r in self._detailed}

        start_time = time.time()
        new_count = 0

        for i, assertion_data in enumerate(sampled):
            assertion_id = assertion_data["assertion_id"]
            if assertion_id in completed_ids:
                continue

            assertion_text = assertion_data["assertion_text"]
            source_question = assertion_data.get("context", {}).get("source_question", "")
            expected_label = assertion_data["expected_label"]

            elapsed = time.time() - start_time
            done_so_far = new_count + len(completed_ids)
            if done_so_far > 0:
                eta = elapsed / done_so_far * (total - done_so_far - len(completed_ids))
            else:
                eta = 0.0

            logger.info(
                "[%d/%d] %s | ETA %.0fs | %.60s",
                i + 1, total, assertion_id, eta, assertion_text,
            )

            row = self._evaluate_one(
                assertion_text=assertion_text,
                source_question=source_question or None,
                assertion_id=assertion_id,
                expected_label=expected_label,
            )
            self._detailed.append(row)
            new_count += 1

            logger.info(
                "  bilateral=%s | fu=%s | tern=%s | conf=%.2f",
                row["bilateral_tv"],
                row["forced_unilateral_prediction"],
                row["ternary_prediction"],
                row["confidence_score"],
            )

            if new_count % self.CHECKPOINT_INTERVAL == 0:
                self._save_checkpoint(i, start_time)

        total_time = time.time() - start_time
        logger.info(
            "Done: %d evaluated (%d new) in %.1fs",
            len(self._detailed), new_count, total_time,
        )

        # Build final result dict
        output = self._build_results(total_time)

        # Write output
        output_path = self._output_path()
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        logger.info("Results saved: %s", output_path)

        self._delete_checkpoint()
        return output

    # ------------------------------------------------------------------
    # Result aggregation
    # ------------------------------------------------------------------

    def _build_results(self, total_time: float) -> Dict[str, Any]:
        detailed = self._detailed
        n = len(detailed)

        gt = [r["ground_truth"] for r in detailed]

        # Bilateral distribution
        bilateral_dist: Dict[str, int] = defaultdict(int)
        for r in detailed:
            bilateral_dist[r["bilateral_tv"]] += 1

        # Metrics per approach
        approaches: Dict[str, Dict] = {}

        for policy_name in ("classical", "paracomplete", "paraconsistent"):
            key = f"bilateral_{policy_name}"
            preds = []
            for r in detailed:
                val = r[f"{key}_projected"]
                preds.append(
                    True if val == "TRUE" else False if val == "FALSE" else None
                )
            approaches[key] = compute_metrics(preds, gt)

        # Forced unilateral (no abstention)
        fu_preds = [r["forced_unilateral_prediction"] == "TRUE" for r in detailed]
        approaches["forced_unilateral"] = compute_metrics(fu_preds, gt)

        # Ternary
        tern_preds = []
        for r in detailed:
            val = r["ternary_prediction"]
            tern_preds.append(
                True if val == "TRUE" else False if val == "FALSE" else None
            )
        approaches["ternary"] = compute_metrics(tern_preds, gt)

        # Confidence @0.5 (no abstention)
        conf_preds = [r["confidence_05_prediction"] == "TRUE" for r in detailed]
        approaches["confidence_05"] = compute_metrics(conf_preds, gt)

        return {
            "model": self.model_name,
            "dataset": self._benchmark,
            "dataset_path": self.dataset_path,
            "bilateral_samples": self.bilateral_samples,
            "total_samples": n,
            "total_time_seconds": round(total_time, 1),
            "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "bilateral_distribution": dict(bilateral_dist),
            "approaches": approaches,
            "detailed_results": detailed,
        }

    def _output_path(self) -> Path:
        model_safe = self.model_name.replace("/", "_").replace(":", "_")
        return self.output_dir / f"{self._benchmark}_{model_safe}_n{self.bilateral_samples}_proper_results.json"


# ──────────────────────────────────────────────────────────────────────────────
# Default system prompt (same as generic_evaluator / modal_evaluator)
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_SYSTEM_PROMPT = """You are an expert evaluating factual statements for accuracy across diverse domains. Your task is to determine whether claims can be verified or refuted based on established knowledge.

Focus on:
- Factual accuracy based on authoritative sources
- Current and historical knowledge across all domains
- Precise details including names, dates, numbers, and places
- Distinguishing between correct facts and plausible misinformation

Be especially careful with:
- Statements that sound plausible but contain factual errors
- Claims mixing accurate and inaccurate information
- Details that may be close to correct but are precisely wrong
- Domain-specific expertise requirements"""


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run BBL proper benchmark evaluation (all 5 epistemic approaches)"
    )
    parser.add_argument(
        "--model", required=True,
        help="Model name (see ModelRouter.MODEL_PROVIDERS)",
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path to standard-format dataset JSON",
    )
    parser.add_argument(
        "--samples", type=int, default=None,
        help="Max assertions to evaluate (default: all); uses balanced sampling",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--bilateral-samples", type=int, default=3,
        help="Majority-vote samples for bilateral evaluation (default: 3, matching paper)",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Directory for output files (default: results)",
    )
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints",
        help="Directory for checkpoint files (default: checkpoints)",
    )
    args = parser.parse_args()

    evaluator = ProperBenchmarkEvaluator(
        model_name=args.model,
        dataset_path=args.dataset,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        bilateral_samples=args.bilateral_samples,
    )
    results = evaluator.run(max_samples=args.samples, seed=args.seed)

    # Print summary table to stdout
    print(f"\n{'='*70}")
    print(f"Model: {results['model']}  |  Dataset: {results['dataset']}")
    print(f"Samples: {results['total_samples']}  |  Time: {results['total_time_seconds']}s")
    print(f"{'='*70}")
    header = f"{'Approach':<28}  {'Coverage':>8}  {'Accuracy':>8}  {'F1-macro':>8}"
    print(header)
    print("-" * len(header))
    for approach, m in results["approaches"].items():
        print(
            f"{approach:<28}  {m['coverage']:>8.3f}  {m['accuracy']:>8.3f}  {m['f1_macro']:>8.3f}"
        )
    print(f"{'='*70}")
    print(f"\nBilateral distribution:")
    for tv, count in sorted(results["bilateral_distribution"].items()):
        pct = count / results["total_samples"] * 100
        print(f"  {tv}: {count:4d}  ({pct:.1f}%)")
    print()


if __name__ == "__main__":
    main()
