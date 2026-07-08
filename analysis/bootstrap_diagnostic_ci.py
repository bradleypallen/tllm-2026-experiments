"""
Cluster bootstrap for the S7.2 diagnostic asymmetry (the 65/3 result).

Backs the paper sentence: "A cluster bootstrap over the 24 <model, benchmark>
cells (10,000 resamples) places 95% confidence intervals at [...] for the <f,f>
rate and [...] for the <t,t> rate, with the asymmetry (the difference of rates)
at [...] percentage points and 95% CI [...]".

Method: the unit of resampling is the (model, benchmark) CELL (cluster
bootstrap -- respects within-cell dependence). For each of B = 10,000 resamples,
draw 24 cells with replacement from the canonical 6x4 grid; recompute the POOLED
ternary-abstention rate on the <f,f>-valued and <t,t>-valued assertions
(sum of abstains / sum of totals over the drawn cells) and their difference;
report percentile 95% intervals (2.5th / 97.5th). Point estimates are the
full-sample pooled rates. Seeded and deterministic.

Data  : results/2026_03/proper/*_n3_proper_results.json (canonical six models)
Output: results/derived/2026_03/bootstrap_diagnostic_ci.json

Run:  python3 analysis/bootstrap_diagnostic_ci.py
"""
from __future__ import annotations
import json
import logging
import random
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [boot-ci] %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
PROPER = REPO / "results/2026_03/proper"
OUT = REPO / "results/derived/2026_03/bootstrap_diagnostic_ci.json"

SEED = 42
B = 10_000
MODELS = [
    "claude-opus-4-1-20250805", "meta-llama_llama-4-maverick",
    "meta-llama_llama-4-scout", "google_gemini-2.5-flash",
    "deepseek_deepseek-chat", "qwen_qwen-2.5-72b-instruct",
]
BENCHES = ["truthfulqa", "simpleqa", "mmlu-pro", "factscore"]
BUCKETS = ("<f,f>", "<t,t>")


def percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolation percentile (numpy default), q in [0, 100]."""
    n = len(sorted_vals)
    pos = (q / 100.0) * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def main():
    # per-cell (total, abstain) counts for each bucket
    cells = []  # list of {bucket: (tot, ab)}
    for mk in MODELS:
        for bench in BENCHES:
            p = PROPER / f"{bench}_{mk}_n3_proper_results.json"
            counts = {b: [0, 0] for b in BUCKETS}
            for r in json.loads(p.read_text())["detailed_results"]:
                tv, tern = r.get("bilateral_tv"), r.get("ternary_prediction")
                if tv in BUCKETS and tern is not None:
                    counts[tv][0] += 1
                    if tern == "ABSTAIN":
                        counts[tv][1] += 1
            cells.append({b: tuple(counts[b]) for b in BUCKETS})
    n_cells = len(cells)
    log.info("loaded %d cells", n_cells)

    def pooled(drawn, bucket):
        tot = sum(c[bucket][0] for c in drawn)
        ab = sum(c[bucket][1] for c in drawn)
        return 100.0 * ab / tot if tot else 0.0

    point = {b: pooled(cells, b) for b in BUCKETS}
    point_diff = point["<f,f>"] - point["<t,t>"]
    log.info("point estimates: <f,f>=%.1f%%  <t,t>=%.1f%%  diff=%.1f pp",
             point["<f,f>"], point["<t,t>"], point_diff)

    rng = random.Random(SEED)
    ff, tt, diff = [], [], []
    for _ in range(B):
        drawn = [cells[rng.randrange(n_cells)] for _ in range(n_cells)]
        f = pooled(drawn, "<f,f>")
        t = pooled(drawn, "<t,t>")
        ff.append(f)
        tt.append(t)
        diff.append(f - t)
    ff.sort(); tt.sort(); diff.sort()

    ci = {
        "<f,f>": [percentile(ff, 2.5), percentile(ff, 97.5)],
        "<t,t>": [percentile(tt, 2.5), percentile(tt, 97.5)],
        "diff": [percentile(diff, 2.5), percentile(diff, 97.5)],
    }
    log.info("95%% CIs: <f,f> [%.1f, %.1f]  <t,t> [%.1f, %.1f]  diff [%.1f, %.1f]",
             *ci["<f,f>"], *ci["<t,t>"], *ci["diff"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "method": "cluster bootstrap over (model, benchmark) cells; pooled "
                  "abstention rates recomputed per resample; percentile 95% CI",
        "seed": SEED, "resamples": B, "n_cells": n_cells,
        "point_estimates_pct": {**{k: round(v, 3) for k, v in point.items()},
                                "diff": round(point_diff, 3)},
        "ci95_pct": {k: [round(v[0], 3), round(v[1], 3)] for k, v in ci.items()},
    }, indent=2) + "\n")
    log.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
