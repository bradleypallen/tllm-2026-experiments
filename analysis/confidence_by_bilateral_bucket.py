"""
Descriptive analysis: confidence-self-report scores grouped by bilateral truth value.

For Reviewer Question 1: "is there already-collected signal (e.g. the confidence
scores) bearing on whether ⟨t,t⟩ items differ from ⟨t,f⟩/⟨f,t⟩ items?"

If the conditional-compliance reading is correct, ⟨t,t⟩ items should cluster at
moderate confidence (thin evidence on which leading-prompt framing produces
double commitments). If the doxastic reading is correct, ⟨t,t⟩ items should
look like other designated cells in terms of confidence.

Data source: results/2026_03/proper/, canonical six models × four benchmarks.
Each per-assertion record has both bilateral_tv and confidence_score.

Outputs:
  - results/derived/2026_03/confidence_by_bucket.json  (per-cell + pooled stats)
  - prints summary table to stdout, suitable for incorporation in §7.2.

Run:  python3 analysis/confidence_by_bilateral_bucket.py
"""
from __future__ import annotations
import json
import logging
import statistics
from pathlib import Path
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [conf-by-bucket] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "results/2026_03/proper"
OUT_JSON = REPO / "results/derived/2026_03/confidence_by_bucket.json"

CANONICAL_MODELS = [
    ("claude-opus-4-1-20250805", "Claude Opus 4.1"),
    ("meta-llama_llama-4-maverick", "Llama 4 Maverick"),
    ("meta-llama_llama-4-scout", "Llama 4 Scout"),
    ("google_gemini-2.5-flash", "Gemini 2.5 Flash"),
    ("deepseek_deepseek-chat", "DeepSeek-V3"),
    ("qwen_qwen-2.5-72b-instruct", "Qwen 2.5 72B"),
]
BENCHMARKS = ["truthfulqa", "simpleqa", "mmlu-pro", "factscore"]
# Bucket definitions: the four pure cells + middle for everything else
BUCKETS = ["<t,f>", "<f,t>", "<f,f>", "<t,t>", "middle"]


def cell_path(model_key: str, bench: str) -> Path:
    return DATA_DIR / f"{bench}_{model_key}_n3_proper_results.json"


def bucketize(tv: str) -> str:
    if tv in ("<t,f>", "<f,t>", "<f,f>", "<t,t>"):
        return tv
    return "middle"


def describe(values):
    if not values:
        return {"n": 0}
    s = sorted(values)
    n = len(s)
    return {
        "n": n,
        "mean": statistics.mean(s),
        "median": statistics.median(s),
        "stdev": statistics.stdev(s) if n > 1 else 0.0,
        "min": s[0],
        "max": s[-1],
        "q25": s[int(0.25 * n)],
        "q75": s[int(0.75 * n)],
        # histogram-style binning for thin-evidence question
        "frac_low": sum(1 for v in s if v < 0.3) / n,       # < 0.3
        "frac_mid": sum(1 for v in s if 0.3 <= v <= 0.7) / n,  # [0.3, 0.7]
        "frac_high": sum(1 for v in s if v > 0.7) / n,      # > 0.7
    }


def main():
    log.info("loading from %s", DATA_DIR)
    pooled = defaultdict(list)  # bucket -> list of confidence_scores
    per_cell = []  # one record per (model, benchmark) for debugging

    for model_key, model_disp in CANONICAL_MODELS:
        for bench in BENCHMARKS:
            p = cell_path(model_key, bench)
            if not p.exists():
                log.warning("missing %s", p.name)
                continue
            blob = json.loads(p.read_text())
            records = blob.get("detailed_results", [])
            cell_buckets = defaultdict(list)
            for r in records:
                tv = r.get("bilateral_tv")
                cs = r.get("confidence_score")
                if tv is None or cs is None:
                    continue
                try:
                    cs = float(cs)
                except (TypeError, ValueError):
                    continue
                b = bucketize(tv)
                cell_buckets[b].append(cs)
                pooled[b].append(cs)
            per_cell.append({
                "model": model_disp, "benchmark": bench,
                "buckets": {b: describe(v) for b, v in cell_buckets.items()},
            })

    # ---- summary --------------------------------------------------------
    log.info("")
    log.info("=========================================================")
    log.info("Pooled across canonical 6 × 4 benchmarks")
    log.info("=========================================================")
    log.info(f"{'bucket':<10} {'n':>5} {'mean':>7} {'median':>8} "
             f"{'stdev':>7} {'low':>6} {'mid':>6} {'high':>6}")
    log.info(f"{'':10} {'':>5} {'':>7} {'':>8} {'':>7} {'<0.3':>6} {'.3-.7':>6} {'>0.7':>6}")
    pooled_stats = {}
    for b in BUCKETS:
        d = describe(pooled[b])
        pooled_stats[b] = d
        if d["n"] == 0:
            continue
        log.info(f"{b:<10} {d['n']:>5d} {d['mean']:>7.3f} {d['median']:>8.3f} "
                 f"{d['stdev']:>7.3f} {100*d['frac_low']:>5.1f}% "
                 f"{100*d['frac_mid']:>5.1f}% {100*d['frac_high']:>5.1f}%")

    # ---- key comparison: ⟨t,t⟩ vs the designated cells ⟨t,f⟩ / ⟨f,t⟩ --
    log.info("")
    log.info("Key comparison for Reviewer Q1:")
    tt = pooled_stats["<t,t>"]
    tf = pooled_stats["<t,f>"]
    ft = pooled_stats["<f,t>"]
    log.info(f"  ⟨t,t⟩ mean confidence: {tt['mean']:.3f} (n={tt['n']})")
    log.info(f"  ⟨t,f⟩ mean confidence: {tf['mean']:.3f} (n={tf['n']})")
    log.info(f"  ⟨f,t⟩ mean confidence: {ft['mean']:.3f} (n={ft['n']})")
    log.info(f"  ⟨t,t⟩ frac at mid (0.3-0.7): {100*tt['frac_mid']:.1f}%")
    log.info(f"  ⟨t,f⟩ frac at mid (0.3-0.7): {100*tf['frac_mid']:.1f}%")
    log.info(f"  ⟨f,t⟩ frac at mid (0.3-0.7): {100*ft['frac_mid']:.1f}%")

    # write JSON audit
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "config": {
            "canonical_models": [m for m, _ in CANONICAL_MODELS],
            "benchmarks": BENCHMARKS,
            "buckets": BUCKETS,
            "data_source": str(DATA_DIR),
            "bucket_definitions": {
                "low": "confidence_score < 0.3",
                "mid": "0.3 <= confidence_score <= 0.7",
                "high": "confidence_score > 0.7",
            },
        },
        "pooled": pooled_stats,
        "per_cell": per_cell,
    }, indent=2))
    log.info(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
