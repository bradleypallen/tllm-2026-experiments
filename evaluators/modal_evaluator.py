#!/usr/bin/env python3
"""
Modal Bilateral Truth Evaluator

Evaluates standard-format datasets under the modal interpretation of the BBL
paper, computing the necessity operator estimate [[□p]] via Monte Carlo
sampling over situation variants:

    [[□p]]^M_s ≈ <min u_i, max v_i>   for VM(s_i, p), i = 0..n-1
    where s_0 = source_question and s_1..s_{n-1} are generated paraphrases

Each assertion is evaluated three ways:
  1. null-situation VM(s_0, p)   -- baseline, matches generic_evaluator.py
  2. source-situation VM(s_0, p) -- using source_question as situation context
  3. modal [[□p]]                -- aggregation over n situation variants

The existing standard datasets and generic_evaluator.py are NOT modified;
this evaluator reads the same dataset files in read-only mode.

Situation variants are cached in the checkpoint file to avoid re-generation
on resume.
"""

import json
import time
import argparse
import hashlib
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bilateral_truth.model_router import ModelRouter
from bilateral_truth.zeta_function import zeta, clear_cache, get_cache_size
from bilateral_truth.assertions import Assertion
from bilateral_truth.truth_values import GeneralizedTruthValue, TruthValueComponent, EpistemicPolicy
from bilateral_truth.variation_generator import SituationVariationGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("modal_evaluator")

try:
    from dotenv import load_dotenv
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info("Loaded environment variables from %s", env_path)
except ImportError:
    logger.warning("python-dotenv not installed; relying on shell environment")


def _tv_min(a: TruthValueComponent, b: TruthValueComponent) -> TruthValueComponent:
    """Truth-order minimum: f < e < t."""
    order = {TruthValueComponent.FALSE: 0, TruthValueComponent.UNDEFINED: 1, TruthValueComponent.TRUE: 2}
    return a if order[a] <= order[b] else b


def _tv_max(a: TruthValueComponent, b: TruthValueComponent) -> TruthValueComponent:
    """Truth-order maximum: f < e < t."""
    order = {TruthValueComponent.FALSE: 0, TruthValueComponent.UNDEFINED: 1, TruthValueComponent.TRUE: 2}
    return a if order[a] >= order[b] else b


def modal_aggregate(evaluations: List[GeneralizedTruthValue]) -> GeneralizedTruthValue:
    """Compute [[□p]] = <min u_i, max v_i> over a list of VM(s'_i, p) values.

    From Definition 25 / Proposition 7 of the BBL paper:
      - u component (verifiability) uses truth-order minimum across variants:
        verified only if verified in every accessible situation
      - v component (refutability) uses truth-order maximum across variants:
        refuted if refuted in any accessible situation

    Args:
        evaluations: Non-empty list of GeneralizedTruthValue results.

    Returns:
        Aggregated GeneralizedTruthValue representing [[□p]].
    """
    if not evaluations:
        return GeneralizedTruthValue(TruthValueComponent.UNDEFINED, TruthValueComponent.UNDEFINED)

    u_agg = evaluations[0].u
    v_agg = evaluations[0].v
    for ev in evaluations[1:]:
        u_agg = _tv_min(u_agg, ev.u)
        v_agg = _tv_max(v_agg, ev.v)
    return GeneralizedTruthValue(u_agg, v_agg)


# Default system prompt — identical to generic_evaluator.py for comparability
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


class ModalBilateralEvaluator:
    """Evaluates standard-format datasets under the BBL modal interpretation.

    Produces three evaluation modes per assertion:
      - null_situation:   VM evaluated with no situation context (replicates
                          generic_evaluator.py baseline)
      - src_situation:    VM evaluated with source_question as situation context
      - modal:            [[□p]] aggregated over n situation variants

    The existing datasets and generic_evaluator.py are read-only from this
    evaluator's perspective.
    """

    VARIATION_MODEL = "claude-opus-4-6"

    def __init__(
        self,
        eval_model: str,
        dataset_path: str,
        n_variants: int = 3,
        epistemic_policy: EpistemicPolicy = EpistemicPolicy.CLASSICAL,
        checkpoint_dir: str = "checkpoints",
        system_prompt: Optional[str] = None,
        variation_model: str = VARIATION_MODEL,
        variant_cache_path: Optional[str] = None,
    ):
        """
        Args:
            eval_model: Model name used for bilateral evaluation (VM).
            dataset_path: Path to an existing standard-format dataset JSON.
            n_variants: Number of situation variants per assertion (default: 3).
            epistemic_policy: Epistemic policy for projecting bilateral values.
            checkpoint_dir: Directory for checkpoint files.
            system_prompt: Custom system prompt; defaults to generic_evaluator default.
            variation_model: Model used for on-the-fly situation variation generation.
                             Only used when variant_cache_path is not provided.
            variant_cache_path: Path to a pre-generated variant cache JSON produced
                                 by pregenerate_variants.py.  When provided, variants
                                 are loaded from this file and the variation model is
                                 not called during evaluation.
        """
        self.eval_model = eval_model
        self.dataset_path = dataset_path
        self.n_variants = n_variants
        self.epistemic_policy = epistemic_policy
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self.variation_model = variation_model
        self.variant_cache_path = variant_cache_path

        self.evaluator = ModelRouter.create_evaluator(eval_model)

        # Load pre-generated variants if a cache path was provided
        self._pregenerated: Dict[str, List[str]] = {}
        if variant_cache_path:
            self._pregenerated = self._load_variant_cache(variant_cache_path)
            logger.info(
                "Loaded pre-generated variants for %d assertions from %s",
                len(self._pregenerated), variant_cache_path
            )

        # Only instantiate the generator if we may need it (no cache or incomplete cache)
        self.var_gen: Optional[SituationVariationGenerator] = None
        if not self._pregenerated:
            self.var_gen = SituationVariationGenerator(model=variation_model)

        self.dataset = self._load_dataset()

        self.results: Dict[str, Any] = {
            "eval_model": eval_model,
            "variation_model": variation_model,
            "dataset": self.dataset["metadata"]["benchmark"],
            "n_variants": n_variants,
            "epistemic_policy": epistemic_policy.value,
            "total_samples": 0,
            # Distribution counters for src-situation and modal modes
            "src_bilateral_distribution": defaultdict(int),
            "modal_bilateral_distribution": defaultdict(int),
            "detailed_results": [],
        }

        self._total_assertions = 0
        self._src_agreements = 0
        self._modal_agreements = 0
        self._total_abstentions_src = 0
        self._total_abstentions_modal = 0

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_variant_cache(path: str) -> Dict[str, List[str]]:
        """Load a pre-generated variant cache produced by pregenerate_variants.py.

        Returns:
            Dict mapping assertion_id → list of variant strings.
        """
        with open(path) as f:
            data = json.load(f)
        raw = data.get("variants", {})
        # Normalise: values may be {source_question, variants} dicts or plain lists
        result = {}
        for assertion_id, entry in raw.items():
            if isinstance(entry, dict):
                result[assertion_id] = entry.get("variants", [])
            elif isinstance(entry, list):
                result[assertion_id] = entry
        return result

    def _load_dataset(self) -> Dict[str, Any]:
        dataset_file = Path(self.dataset_path)
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")
        with open(dataset_file) as f:
            dataset = json.load(f)
        if "metadata" not in dataset or "assertions" not in dataset:
            raise ValueError("Invalid dataset format: missing 'metadata' or 'assertions'")
        logger.info(
            "Loaded %s: %d assertions",
            dataset["metadata"]["benchmark"],
            dataset["metadata"]["total_assertions"],
        )
        return dataset

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def _checkpoint_key(self) -> str:
        model_safe = self.eval_model.replace("/", "_").replace(":", "_")
        dataset_hash = hashlib.md5(self.dataset_path.encode()).hexdigest()[:8]
        config_hash = hashlib.md5(
            f"{self.system_prompt}{self.n_variants}{self.epistemic_policy.value}".encode()
        ).hexdigest()[:8]
        return f"modal_checkpoint_{model_safe}_{dataset_hash}_{config_hash}.json"

    def save_checkpoint(self, start_time: float, current_idx: int,
                        variant_cache: Dict[str, List[str]]) -> None:
        path = self.checkpoint_dir / self._checkpoint_key()
        data = {
            "eval_model": self.eval_model,
            "dataset_path": self.dataset_path,
            "n_variants": self.n_variants,
            "epistemic_policy": self.epistemic_policy.value,
            "system_prompt": self.system_prompt,
            "start_time": start_time,
            "current_assertion_idx": current_idx,
            "checkpoint_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_completed": self.results["total_samples"],
            "detailed_results": self.results["detailed_results"],
            "src_bilateral_distribution": dict(self.results["src_bilateral_distribution"]),
            "modal_bilateral_distribution": dict(self.results["modal_bilateral_distribution"]),
            "src_agreements": self._src_agreements,
            "modal_agreements": self._modal_agreements,
            "total_abstentions_src": self._total_abstentions_src,
            "total_abstentions_modal": self._total_abstentions_modal,
            "total_assertions": self._total_assertions,
            "variant_cache": variant_cache,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Checkpoint saved: %d evaluations at index %d", self.results["total_samples"], current_idx)

    def load_checkpoint(self) -> Optional[Dict]:
        path = self.checkpoint_dir / self._checkpoint_key()
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            if (data.get("eval_model") != self.eval_model
                    or data.get("dataset_path") != self.dataset_path
                    or data.get("n_variants") != self.n_variants
                    or data.get("epistemic_policy") != self.epistemic_policy.value
                    or data.get("system_prompt") != self.system_prompt):
                logger.warning("Checkpoint found but incompatible — starting fresh")
                return None
            logger.info(
                "Checkpoint found: %d evaluations completed (%s)",
                data["total_completed"], data["checkpoint_timestamp"]
            )
            return data
        except Exception as e:
            logger.warning("Error loading checkpoint: %s — starting fresh", e)
            return None

    def delete_checkpoint(self) -> None:
        path = self.checkpoint_dir / self._checkpoint_key()
        if path.exists():
            path.unlink()
            logger.info("Checkpoint deleted: %s", path)

    def restore_from_checkpoint(self, data: Dict) -> Dict[str, List[str]]:
        self.results["detailed_results"] = data.get("detailed_results", [])
        self.results["total_samples"] = data.get("total_completed", 0)
        self.results["src_bilateral_distribution"] = defaultdict(int, data.get("src_bilateral_distribution", {}))
        self.results["modal_bilateral_distribution"] = defaultdict(int, data.get("modal_bilateral_distribution", {}))
        self._src_agreements = data.get("src_agreements", 0)
        self._modal_agreements = data.get("modal_agreements", 0)
        self._total_abstentions_src = data.get("total_abstentions_src", 0)
        self._total_abstentions_modal = data.get("total_abstentions_modal", 0)
        self._total_assertions = data.get("total_assertions", 0)
        return data.get("variant_cache", {})

    # ------------------------------------------------------------------
    # Single-assertion evaluation
    # ------------------------------------------------------------------

    def _project(self, tv: GeneralizedTruthValue) -> TruthValueComponent:
        return tv.project(self.epistemic_policy)

    def _is_agreement(self, projected: TruthValueComponent, expected_label: str) -> Optional[bool]:
        if projected == TruthValueComponent.UNDEFINED:
            return None
        return (expected_label == "correct") == (projected == TruthValueComponent.TRUE)

    def evaluate_assertion(
        self,
        assertion_data: Dict,
        variant_cache: Dict[str, List[str]],
        assertion_idx: int,
        total_assertions: int,
        elapsed: float,
    ) -> Dict:
        """Evaluate one assertion across all three modes and update running stats."""
        assertion_id = assertion_data["assertion_id"]
        assertion_text = assertion_data["assertion_text"]
        expected_label = assertion_data["expected_label"]
        context = assertion_data["context"]
        category = context.get("category", "unknown")
        source_question = context.get("source_question", "")

        progress_pct = (assertion_idx / total_assertions) * 100
        remaining = "?" if assertion_idx == 0 else self._format_time(
            elapsed / assertion_idx * (total_assertions - assertion_idx)
        )
        logger.info(
            "Assertion %d/%d (%.1f%%) | ETA %s | %s",
            assertion_idx + 1, total_assertions, progress_pct,
            remaining, assertion_text[:60]
        )

        assertion_obj = Assertion(assertion_text)

        # --- Source-question situation VM(s₀, p) ---
        src_context = source_question if source_question else None
        src_tv = zeta(assertion_obj, self.evaluator.evaluate_bilateral,
                      system_prompt=self.system_prompt, context=src_context)

        # --- Mode 3: modal aggregation over situation variants ---
        # Resolve variants: pre-generated cache takes priority, then per-run cache,
        # then on-the-fly generation (fallback when no cache file was provided).
        if assertion_id in self._pregenerated:
            variants = self._pregenerated[assertion_id]
            variant_cache[assertion_id] = variants  # keep checkpoint consistent
        elif assertion_id in variant_cache:
            variants = variant_cache[assertion_id]
        else:
            if source_question:
                if self.var_gen is None:
                    # Lazy-initialise generator if a cache miss occurs
                    self.var_gen = SituationVariationGenerator(model=self.variation_model)
                # Generate n_variants-1 paraphrases; s₀ (src) is the first situation
                variants = self.var_gen.generate(source_question, n=self.n_variants - 1)
                logger.info(
                    "Generated %d paraphrases for %s: %s...",
                    len(variants), assertion_id, source_question[:50]
                )
            else:
                variants = []
                logger.warning(
                    "No source_question for %s; modal evaluation will use src_tv only",
                    assertion_id
                )
            variant_cache[assertion_id] = variants

        # Aggregate over s₀ (src_tv) + s₁..s_{n-1} (paraphrases)
        variant_tvs = []
        for i, variant in enumerate(variants):
            tv = zeta(assertion_obj, self.evaluator.evaluate_bilateral,
                      system_prompt=self.system_prompt, context=variant)
            variant_tvs.append(tv)
            logger.info(
                "  s%d: %s → %s",
                i + 1, variant[:50], tv
            )
        modal_tv = modal_aggregate([src_tv] + variant_tvs)

        # Project both
        src_proj   = self._project(src_tv)
        modal_proj = self._project(modal_tv)

        # Agreement tracking
        src_agree   = self._is_agreement(src_proj, expected_label)
        modal_agree = self._is_agreement(modal_proj, expected_label)

        # Update distributions
        self.results["src_bilateral_distribution"][str(src_tv)] += 1
        self.results["modal_bilateral_distribution"][str(modal_tv)] += 1
        self.results["total_samples"] += 1

        if src_proj == TruthValueComponent.UNDEFINED:
            self._total_abstentions_src += 1
        else:
            if src_agree:
                self._src_agreements += 1

        if modal_proj == TruthValueComponent.UNDEFINED:
            self._total_abstentions_modal += 1
        else:
            if modal_agree:
                self._modal_agreements += 1

        self._total_assertions += 1

        logger.info(
            "  src=%s modal=%s | expected=%s",
            src_tv, modal_tv, expected_label
        )

        return {
            "index": len(self.results["detailed_results"]),
            "assertion_id": assertion_id,
            "assertion": assertion_text,
            "expected_label": expected_label,
            "category": category,
            "source_question": source_question,
            "src_bilateral": str(src_tv),
            "src_projected": src_proj.value,
            "src_agreement": src_agree,
            "modal_bilateral": str(modal_tv),
            "modal_projected": modal_proj.value,
            "modal_agreement": modal_agree,
            "variants": variants,
            "variant_bilaterals": [str(tv) for tv in variant_tvs],
        }

    # ------------------------------------------------------------------
    # Main evaluation loop
    # ------------------------------------------------------------------

    def run_evaluation(
        self,
        max_samples: Optional[int] = None,
        checkpoint_interval: int = 10,
    ) -> Dict:
        all_assertions = self.dataset["assertions"]

        if max_samples is not None and max_samples < len(all_assertions):
            import random
            random.seed(42)
            positive = [a for a in all_assertions if a["expected_label"] == "correct"]
            negative = [a for a in all_assertions if a["expected_label"] == "incorrect"]
            half = max_samples // 2
            pos_n = min(half + max_samples % 2, len(positive))
            neg_n = min(max_samples - pos_n, len(negative))
            all_assertions = random.sample(positive, pos_n) + random.sample(negative, neg_n)
            random.shuffle(all_assertions)
            logger.info("Sampled %d assertions (%d positive, %d negative)", len(all_assertions), pos_n, neg_n)

        total_assertions = len(all_assertions)

        # Resume from checkpoint if available
        checkpoint_data = self.load_checkpoint()
        if checkpoint_data:
            variant_cache = self.restore_from_checkpoint(checkpoint_data)
            start_time = checkpoint_data["start_time"]
            start_idx = checkpoint_data.get("current_assertion_idx", 0) + 1
            logger.info("Resumed from checkpoint at index %d", start_idx)
        else:
            variant_cache = {}
            clear_cache()
            start_time = time.time()
            start_idx = 0

        logger.info(
            "Starting modal evaluation: %s on %s (%d assertions, n_variants=%d)",
            self.eval_model, self.dataset["metadata"]["benchmark"],
            total_assertions, self.n_variants
        )

        for idx in range(start_idx, total_assertions):
            assertion_data = all_assertions[idx]
            t_item = time.time()
            elapsed = t_item - start_time

            try:
                result = self.evaluate_assertion(
                    assertion_data, variant_cache, idx, total_assertions, elapsed
                )
                self.results["detailed_results"].append(result)
            except Exception as e:
                logger.error("Error on assertion %d (%s): %s", idx, assertion_data["assertion_id"], e)
                self.results["detailed_results"].append({
                    "index": len(self.results["detailed_results"]),
                    "assertion_id": assertion_data["assertion_id"],
                    "assertion": assertion_data["assertion_text"],
                    "expected_label": assertion_data["expected_label"],
                    "category": assertion_data["context"].get("category", "unknown"),
                    "source_question": assertion_data["context"].get("source_question", ""),
                    "null_bilateral": "<e,e>", "null_projected": "e", "null_agreement": None,
                    "src_bilateral": "<e,e>",  "src_projected": "e",  "src_agreement": None,
                    "modal_bilateral": "<e,e>","modal_projected": "e","modal_agreement": None,
                    "variants": [], "variant_bilaterals": [],
                    "error": str(e),
                })
                self.results["total_samples"] += 1
                self._total_abstentions_null += 1
                self._total_abstentions_src += 1
                self._total_abstentions_modal += 1

            if (idx + 1) % checkpoint_interval == 0:
                self.save_checkpoint(start_time, idx, variant_cache)

        # Final metrics
        total_time = time.time() - start_time
        n = self.results["total_samples"]

        self.results["evaluation_time"] = total_time
        self.results["total_assertions"] = self._total_assertions

        for mode, agreements, abstentions in [
            ("src",   self._src_agreements,   self._total_abstentions_src),
            ("modal", self._modal_agreements, self._total_abstentions_modal),
        ]:
            answered = n - abstentions
            self.results[f"{mode}_coverage"] = answered / n if n > 0 else 0.0
            self.results[f"{mode}_accuracy"] = agreements / answered if answered > 0 else 0.0
            self.results[f"{mode}_f1_macro"] = self._f1_macro(mode)

        logger.info(
            "Evaluation complete in %s | src F1=%.3f | modal F1=%.3f",
            self._format_time(total_time),
            self.results["src_f1_macro"],
            self.results["modal_f1_macro"],
        )
        return self.results

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _f1_macro(self, mode: str) -> float:
        key_proj = f"{mode}_projected"
        key_agree = f"{mode}_agreement"
        non_abs = [r for r in self.results["detailed_results"]
                   if r.get(key_proj) != "e" and r.get(key_agree) is not None]
        if not non_abs:
            return 0.0

        correct_preds = [r for r in non_abs if r[key_proj] == "t"]
        incorrect_preds = [r for r in non_abs if r[key_proj] == "f"]
        correct_actual = [r for r in non_abs if r["expected_label"] == "correct"]
        incorrect_actual = [r for r in non_abs if r["expected_label"] == "incorrect"]

        def f1(preds, actual):
            tp = sum(1 for r in preds if r["expected_label"] == ("correct" if preds is correct_preds else "incorrect"))
            p = tp / len(preds) if preds else 0.0
            r = tp / len(actual) if actual else 0.0
            return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        c_tp = sum(1 for r in correct_preds if r["expected_label"] == "correct")
        c_p = c_tp / len(correct_preds) if correct_preds else 0.0
        c_r = c_tp / len(correct_actual) if correct_actual else 0.0
        c_f1 = 2 * c_p * c_r / (c_p + c_r) if (c_p + c_r) > 0 else 0.0

        i_tp = sum(1 for r in incorrect_preds if r["expected_label"] == "incorrect")
        i_p = i_tp / len(incorrect_preds) if incorrect_preds else 0.0
        i_r = i_tp / len(incorrect_actual) if incorrect_actual else 0.0
        i_f1 = 2 * i_p * i_r / (i_p + i_r) if (i_p + i_r) > 0 else 0.0

        return (c_f1 + i_f1) / 2

    @staticmethod
    def _format_time(seconds: float) -> str:
        if seconds < 60:
            return "%.1fs" % seconds
        elif seconds < 3600:
            return "%.1fm" % (seconds / 60)
        return "%.1fh" % (seconds / 3600)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def save_results(self, output_path: str) -> None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        results_copy = dict(self.results)
        for key in ("src_bilateral_distribution", "modal_bilateral_distribution"):
            results_copy[key] = dict(results_copy[key])
        with open(out, "w") as f:
            json.dump(results_copy, f, indent=2, default=str)
        logger.info("Results saved to %s", output_path)

    def print_summary(self) -> None:
        bench = self.dataset["metadata"]["benchmark"].upper()
        n = self.results["total_samples"]
        print(f"\n{'=' * 70}")
        print(f"Modal Evaluation Summary — {bench} — {self.eval_model}")
        print(f"{'=' * 70}")
        print(f"  Assertions evaluated : {n}")
        print(f"  Situation variants   : {self.n_variants} per assertion")
        print(f"  Epistemic policy     : {self.epistemic_policy.value}")
        print()
        for mode, label in [("src",  "Source situation VM(s₀,p)"),
                             ("modal","Modal [[□p]]       ")]:
            cov = self.results.get(f"{mode}_coverage", 0)
            acc = self.results.get(f"{mode}_accuracy", 0)
            f1  = self.results.get(f"{mode}_f1_macro", 0)
            print(f"  {label}  coverage={cov:.3f}  accuracy={acc:.3f}  F1={f1:.3f}")
        print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Modal Bilateral Truth Evaluation (BBL necessity operator [[□p]])"
    )
    parser.add_argument("--model", required=True, help="Evaluation model name")
    parser.add_argument("--dataset", required=True, help="Path to standard-format dataset JSON")
    parser.add_argument("--n-variants", type=int, default=3,
                        help="Situation variants per assertion (default: 3)")
    parser.add_argument("--samples", type=int, help="Max assertions to evaluate (default: all)")
    parser.add_argument("--epistemic-policy", default="classical",
                        choices=["classical", "paraconsistent", "paracomplete"])
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--variant-cache", metavar="PATH",
                        help="Path to pre-generated variant cache JSON "
                             "(from pregenerate_variants.py). Skips Opus 4.6 calls.")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--clear-checkpoints", action="store_true")
    args = parser.parse_args()

    if args.clear_checkpoints:
        import glob as _glob
        ckpt_dir = Path(args.checkpoint_dir)
        for f in _glob.glob(str(ckpt_dir / "modal_checkpoint_*.json")):
            Path(f).unlink()
            logger.info("Cleared checkpoint: %s", f)

    policy_map = {
        "classical": EpistemicPolicy.CLASSICAL,
        "paraconsistent": EpistemicPolicy.PARACONSISTENT,
        "paracomplete": EpistemicPolicy.PARACOMPLETE,
    }

    evaluator = ModalBilateralEvaluator(
        eval_model=args.model,
        dataset_path=args.dataset,
        n_variants=args.n_variants,
        epistemic_policy=policy_map[args.epistemic_policy],
        checkpoint_dir=args.checkpoint_dir,
        variant_cache_path=args.variant_cache,
    )

    try:
        results = evaluator.run_evaluation(
            max_samples=args.samples,
            checkpoint_interval=args.checkpoint_interval,
        )
        if not args.no_save:
            model_safe = args.model.replace("/", "_").replace(":", "_")
            dataset_name = Path(args.dataset).stem
            out = f"{args.output_dir}/{dataset_name}_{model_safe}_n{args.n_variants}_modal_results.json"
            evaluator.save_results(out)
        evaluator.delete_checkpoint()
        evaluator.print_summary()
    except KeyboardInterrupt:
        logger.warning("Interrupted — partial results not saved (checkpoint preserved)")
    except Exception as e:
        logger.error("Evaluation failed: %s", e)
        raise


if __name__ == "__main__":
    main()
