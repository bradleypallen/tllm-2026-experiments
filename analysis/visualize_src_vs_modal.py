#!/usr/bin/env python3
"""
Side-by-side comparison: VM(s₀, p)  vs  [[□p]] bilateral TV distributions.

Layout: 4 benchmarks × 6 models, each cell contains two stacked bars —
left = source (VM), right = modal ([[□p]]).  Rows = benchmarks, cols = models.

Reads *_n3_modal_results.json (which contain both src_ and modal_ distributions).
"""

import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("visualize_src_vs_modal")

BASE = os.path.join(os.path.dirname(__file__), "results")

MODEL_ORDER = [
    ("claude-opus-4-1-20250805",    "Opus 4.1"),
    ("meta-llama/llama-4-maverick", "Llama 4 Mav."),
    ("meta-llama/llama-4-scout",    "Llama 4 Scout"),
    ("google/gemini-2.5-flash",     "Gemini 2.5"),
    ("deepseek/deepseek-chat",      "DeepSeek-V3"),
    ("qwen/qwen-2.5-72b-instruct",  "Qwen 2.5 72B"),
]
BENCHMARK_ORDER = ["truthfulqa", "simpleqa", "mmlu-pro", "factscore"]
BENCHMARK_LABELS = {
    "truthfulqa": "TruthfulQA",
    "simpleqa":   "SimpleQA",
    "mmlu-pro":   "MMLU-Pro",
    "factscore":  "FACTScore",
}

TV_ORDER = ["<t,f>", "<f,t>", "<f,f>", "<t,t>",
            "<t,e>", "<e,f>", "<e,t>", "<f,e>", "<e,e>"]
TV_LABELS = {tv: f"\u27e8{tv[1]},{tv[3]}\u27e9" for tv in TV_ORDER}
TV_COLORS = {
    "<t,f>": "#1565C0",
    "<f,t>": "#C62828",
    "<f,f>": "#9E9E9E",
    "<t,t>": "#FF8F00",
    "<t,e>": "#90CAF9",
    "<e,f>": "#FFCC80",
    "<e,t>": "#CE93D8",
    "<f,e>": "#EF9A9A",
    "<e,e>": "#CFD8DC",
}

# ── Load data ──────────────────────────────────────────────────────────────────
data = {}
for path in glob.glob(os.path.join(BASE, "*_n3_modal_results.json")):
    d = json.load(open(path))
    key = (d["eval_model"], d["dataset"])
    data[key] = d
    logger.info("Loaded %s × %s", d["eval_model"], d["dataset"])

if not data:
    print("No modal result files found in", BASE)
    raise SystemExit(1)

# ── Figure ──────────────────────────────────────────────────────────────────────
n_rows = len(BENCHMARK_ORDER)
n_cols = len(MODEL_ORDER)

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(22, 13),
    sharey="row",
)
fig.patch.set_facecolor("white")

X_SRC   = -0.22   # x-centre of source bar
X_MODAL =  0.22   # x-centre of modal bar
BAR_W   =  0.38


def stacked_bar(ax, dist, n, x, bar_w, label_threshold=7):
    """Draw a single stacked bar at position x; annotate segments ≥ label_threshold %."""
    present = [tv for tv in TV_ORDER if dist.get(tv, 0) > 0]
    bottom = 0.0
    for tv in present:
        pct = dist.get(tv, 0) / n * 100
        ax.bar(x, pct, bar_w, bottom=bottom,
               color=TV_COLORS[tv], edgecolor="white", linewidth=0.4)
        if pct >= label_threshold:
            ax.text(x, bottom + pct / 2,
                    f"{pct:.0f}%", ha="center", va="center",
                    fontsize=6, color="white", fontweight="bold")
        bottom += pct


for row, bench in enumerate(BENCHMARK_ORDER):
    for col, (mid, mlabel) in enumerate(MODEL_ORDER):
        ax = axes[row, col]
        key = (mid, bench)

        if key not in data:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="#aaa")
            ax.set_xticks([])
            continue

        d = data[key]
        n = d["total_samples"]
        src_dist   = d["src_bilateral_distribution"]
        modal_dist = d["modal_bilateral_distribution"]

        stacked_bar(ax, src_dist,   n, X_SRC,   BAR_W)
        stacked_bar(ax, modal_dist, n, X_MODAL, BAR_W)

        # Coverage annotations above each bar
        src_cov   = d.get("src_coverage",   float("nan"))
        modal_cov = d.get("modal_coverage", float("nan"))
        ax.text(X_SRC,   103, f"{src_cov:.2f}",   ha="center", va="bottom",
                fontsize=5.5, color="#555")
        ax.text(X_MODAL, 103, f"{modal_cov:.2f}", ha="center", va="bottom",
                fontsize=5.5, color="#333", fontweight="bold")

        # Separator line between bars
        ax.axvline(0, color="#ddd", linewidth=0.6, linestyle="--", zorder=0)

        ax.set_xlim(-0.55, 0.55)
        ax.set_ylim(0, 112)
        ax.set_xticks([X_SRC, X_MODAL])
        ax.set_xticklabels(["VM", "□p"], fontsize=6.5, color="#444")
        ax.tick_params(axis="x", length=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        if row == 0:
            ax.set_title(mlabel, fontsize=8.5, fontweight="bold", pad=4)

        if col == 0:
            ax.set_ylabel(BENCHMARK_LABELS[bench], fontsize=9,
                          fontweight="bold", labelpad=6)
            ax.yaxis.set_tick_params(labelsize=7)
        else:
            ax.set_yticklabels([])

# ── Shared legend ──────────────────────────────────────────────────────────────
present_globally = set()
for d in data.values():
    present_globally |= set(d["src_bilateral_distribution"].keys())
    present_globally |= set(d["modal_bilateral_distribution"].keys())

legend_patches = [
    mpatches.Patch(color=TV_COLORS[tv], label=TV_LABELS[tv])
    for tv in TV_ORDER if tv in present_globally
]
fig.legend(
    handles=legend_patches,
    loc="lower center",
    ncol=len(legend_patches),
    fontsize=9,
    title="Bilateral truth value",
    title_fontsize=9,
    frameon=True,
    bbox_to_anchor=(0.5, -0.03),
)

fig.suptitle(
    "VM(s\u2080, p)  vs  [[\u25a1p]]  Bilateral TV Distributions\n"
    "Left bar = source situation only;  Right bar = modal necessity over s\u2080 + 2 paraphrases.  "
    "Numbers above bars = classical coverage.",
    fontsize=11, fontweight="bold", y=1.01,
)

plt.tight_layout(rect=[0, 0.04, 1, 1])

out_pdf = os.path.join(BASE, "src_vs_modal_distributions.pdf")
out_png = out_pdf.replace(".pdf", ".png")
plt.savefig(out_pdf, bbox_inches="tight", dpi=150)
plt.savefig(out_png, bbox_inches="tight", dpi=150)
logger.info("Saved: %s", out_pdf)
logger.info("Saved: %s", out_png)
