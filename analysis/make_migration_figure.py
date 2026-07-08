"""
Generate the per-cell ⟨t,f⟩ → ⟨f,f⟩ migration summary figure.

For each (model, benchmark) cell, computes the fraction of assertions whose
base-situation valuation $V_M(s_0, p) = \langle t,f \rangle$ migrates under
modal aggregation to $\llbracket \Box p \rrbracket = \langle f,f \rangle$
(prompt-sensitivity collapsing to ignorance). Renders as a 4×6 heatmap.

Data source: results/2026_03/modal/, the canonical six-model run referenced by
the paper.

Outputs (results/derived/2026_03/):
  - migration_tf_to_ff.pdf         (figure; copy into the paper repo for LaTeX)
  - migration_tf_to_ff_audit.json  (per-cell counts, for trace)

Run:  python3 analysis/make_migration_figure.py
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [migration-fig] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "results/2026_03/modal"
OUT_PDF = REPO / "results/derived/2026_03/migration_tf_to_ff.pdf"
OUT_AUDIT = REPO / "results/derived/2026_03/migration_tf_to_ff_audit.json"

# Display names for the canonical six in row/column order
CANONICAL_MODELS = [
    ("claude-opus-4-1-20250805", "Opus 4.1"),
    ("meta-llama_llama-4-maverick", "Llama 4\nMav."),
    ("meta-llama_llama-4-scout", "Llama 4\nScout"),
    ("google_gemini-2.5-flash", "Gemini 2.5\nFlash"),
    ("deepseek_deepseek-chat", "DeepSeek-V3"),
    ("qwen_qwen-2.5-72b-instruct", "Qwen 2.5\n72B"),
]
BENCHMARKS = [
    ("truthfulqa", "TruthfulQA"),
    ("simpleqa", "SimpleQA"),
    ("mmlupro", "MMLU-Pro"),
    ("factscore", "FACTScore"),
]


def cell_path(model_key: str, bench_key: str) -> Path:
    return DATA_DIR / f"{bench_key}_complete_{model_key}_n3_modal_results.json"


def migration_rate(records):
    """Return (n_base_tf, n_migrated_to_ff, fraction)."""
    n_tf = 0
    n_to_ff = 0
    for r in records:
        if r.get("src_bilateral") == "<t,f>":
            n_tf += 1
            if r.get("modal_bilateral") == "<f,f>":
                n_to_ff += 1
    return n_tf, n_to_ff, (n_to_ff / n_tf if n_tf else 0.0)


def main():
    log.info("loading from %s", DATA_DIR)
    grid_pct = np.full((len(BENCHMARKS), len(CANONICAL_MODELS)), np.nan)
    grid_n = np.zeros_like(grid_pct, dtype=int)
    audit = {"per_cell": [], "config": {
        "canonical_models": [m for m, _ in CANONICAL_MODELS],
        "benchmarks": [b for b, _ in BENCHMARKS],
        "src_class": "<t,f>",
        "dest_class": "<f,f>",
    }}

    for i, (bench_key, bench_disp) in enumerate(BENCHMARKS):
        for j, (model_key, model_disp) in enumerate(CANONICAL_MODELS):
            p = cell_path(model_key, bench_key)
            if not p.exists():
                log.warning("missing %s", p.name)
                continue
            blob = json.loads(p.read_text())
            records = blob.get("detailed_results", [])
            n_tf, n_to_ff, frac = migration_rate(records)
            grid_pct[i, j] = 100 * frac
            grid_n[i, j] = n_tf
            audit["per_cell"].append({
                "model": model_disp.replace("\n", " "),
                "benchmark": bench_disp,
                "n_base_tf": n_tf,
                "n_migrated_to_ff": n_to_ff,
                "migration_pct": 100 * frac,
            })
            log.info("%-15s × %-15s n_base_tf=%3d  n_to_ff=%3d  rate=%5.1f%%",
                     model_disp.replace("\n", " "), bench_disp,
                     n_tf, n_to_ff, 100 * frac)

    # ---- render ----------------------------------------------------------
    # Style matched to analysis/visualize_tv_diagnostic.py (the 65/3 diagnostic,
    # Figure 3): default DejaVu Sans, bold title, literal-Unicode notation, dotted
    # chrome, dpi 150. Chart type stays a heatmap (24 cells, one value each).
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
    })

    n_b, n_m = len(BENCHMARKS), len(CANONICAL_MODELS)
    # Use the full data range so colour carries signal (max migration is ~25%);
    # the old vmax=60 washed every cell to pale yellow.
    vmax = float(np.ceil(np.nanmax(grid_pct) / 5.0) * 5.0)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    fig.patch.set_facecolor("white")

    im = ax.imshow(grid_pct, aspect="auto", cmap=plt.cm.YlOrRd, vmin=0, vmax=vmax)

    # Annotate each cell with the migration percentage and (n) below it.
    for i in range(n_b):
        for j in range(n_m):
            if np.isnan(grid_pct[i, j]):
                continue
            val = grid_pct[i, j]
            txt_color = "white" if val > 0.55 * vmax else "#2a2a2a"
            ax.text(j, i - 0.12, f"{val:.0f}%", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=txt_color)
            ax.text(j, i + 0.24, f"(n={grid_n[i, j]})", ha="center", va="center",
                    fontsize=7.5, color=txt_color, alpha=0.8)

    ax.set_xticks(range(n_m))
    ax.set_xticklabels([d for _, d in CANONICAL_MODELS], fontsize=8.5)
    ax.set_yticks(range(n_b))
    ax.set_yticklabels([d for _, d in BENCHMARKS], fontsize=9.5)
    ax.tick_params(length=0)

    # White gridlines separating the cells; hide the axes frame.
    ax.set_xticks(np.arange(n_m + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_b + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Slim colourbar, frameless, label/ticks sized to match Figure 3.
    cbar = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cbar.set_label("% of base ⟨t,f⟩ → ⟨f,f⟩ under □", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)

    # No in-figure title --- the description lives in the LaTeX caption.

    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight", dpi=150)
    log.info("wrote %s", OUT_PDF)

    OUT_AUDIT.write_text(json.dumps(audit, indent=2))
    log.info("wrote %s", OUT_AUDIT)


if __name__ == "__main__":
    main()
