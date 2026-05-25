#!/usr/bin/env python3
"""
Modal [[□p]] TV distribution visualization.
4 benchmarks × 6 models — stacked bar chart grid, parallel to visualize_tv_distributions.py.

Reads *_n3_modal_results.json from the results directory and shows
modal_bilateral_distribution (VM aggregated over 3 situation variants).
"""

import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = os.path.join(os.path.dirname(__file__), "results")

MODEL_ORDER = [
    ("claude-opus-4-1-20250805",     "Opus 4.1"),
    ("meta-llama/llama-4-maverick",  "Llama 4 Mav."),
    ("meta-llama/llama-4-scout",     "Llama 4 Scout"),
    ("google/gemini-2.5-flash",      "Gemini 2.5"),
    ("deepseek/deepseek-chat",       "DeepSeek-V3"),
    ("qwen/qwen-2.5-72b-instruct",   "Qwen 2.5 72B"),
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
    data[(d["eval_model"], d["dataset"])] = d

if not data:
    print("No modal result files found in", BASE)
    raise SystemExit(1)

# ── Figure: 4 rows (benchmarks) × 6 cols (models) ────────────────────────────
fig, axes = plt.subplots(
    len(BENCHMARK_ORDER), len(MODEL_ORDER),
    figsize=(20, 12),
    sharey="row",
)
fig.patch.set_facecolor("white")

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
        dist = d["modal_bilateral_distribution"]
        n = d["total_samples"]

        # Stacked bar
        present = [tv for tv in TV_ORDER if dist.get(tv, 0) > 0]
        bottom = 0.0
        for tv in present:
            pct = dist.get(tv, 0) / n * 100
            ax.bar(0, pct, 0.6, bottom=bottom,
                   color=TV_COLORS[tv], edgecolor="white", linewidth=0.4)
            if pct >= 5:
                ax.text(0, bottom + pct / 2,
                        f"{pct:.0f}%", ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")
            bottom += pct

        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(0, 110)
        ax.set_xticks([])
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
    title="Bilateral truth value  (rows = benchmark, cols = model)",
    title_fontsize=9,
    frameon=True,
    bbox_to_anchor=(0.5, -0.03),
)

plt.tight_layout(rect=[0, 0.04, 1, 1])

out_pdf = os.path.join(BASE, "modal_distributions.pdf")
out_png = out_pdf.replace(".pdf", ".png")
plt.savefig(out_pdf, bbox_inches="tight", dpi=150)
plt.savefig(out_png, bbox_inches="tight", dpi=150)
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
