#!/usr/bin/env python3
"""
Full comparison visualization: bilateral vs. unilateral approaches
across 6 models × 4 benchmarks (N=250, bilateral_samples=3).

Produces:
  results/full_comparison_visualization.pdf / .png
"""

import json
import glob
import os
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("visualize_full_comparison")

BASE = os.path.join(os.path.dirname(__file__), "results")

MODEL_ORDER = [
    ("claude-opus-4-1-20250805",     "Opus 4.1"),
    ("meta-llama/llama-4-maverick",  "Llama 4 Maverick"),
    ("meta-llama/llama-4-scout",     "Llama 4 Scout"),
    ("google/gemini-2.5-flash",      "Gemini 2.5 Flash"),
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
APPROACHES = [
    ("bilateral_classical",      "Bilateral\n(classical)",      "#1565C0"),
    ("bilateral_paraconsistent", "Bilateral\n(paraconsist.)",   "#42A5F5"),
    ("forced_unilateral",        "Forced\nunilateral",          "#E65100"),
    ("ternary",                  "Ternary\n(evidence)",         "#2E7D32"),
    ("confidence_05",            "Confidence\n@0.5",            "#F9A825"),
]

# ── Load all results ───────────────────────────────────────────────────────────
data = {}
for path in glob.glob(os.path.join(BASE, "*n3_proper_results.json")):
    d = json.load(open(path))
    data[(d["model"], d["dataset"])] = d
    logger.info("Loaded %s × %s", d["model"], d["dataset"])

n_models    = len(MODEL_ORDER)
n_benchmarks = len(BENCHMARK_ORDER)

# ── Helper ─────────────────────────────────────────────────────────────────────
def get(model_id, bench, approach, metric="f1_macro"):
    key = (model_id, bench)
    if key not in data:
        return np.nan
    return data[key]["approaches"][approach][metric]

# ── Figure layout ──────────────────────────────────────────────────────────────
# Row 0: grouped bar chart per benchmark (Bil / FU / Tern / Conf)
# Row 1: delta heatmaps  (Bil − FU) and (Bil − Tern)
# Row 2: average-by-model bar chart

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(3, 1, height_ratios=[2.5, 1.6, 1.8], hspace=0.52)

model_short = [lbl for _, lbl in MODEL_ORDER]
model_ids   = [mid for mid, _ in MODEL_ORDER]
bench_labels = [BENCHMARK_LABELS[b] for b in BENCHMARK_ORDER]

# Colour for each model
MODEL_COLORS = [
    "#1565C0", "#E65100", "#F9A825", "#6A1B9A", "#00695C", "#4E342E"
]

# ── Panel 0: grouped bars — F1 by benchmark and approach ──────────────────────
gs0 = gs[0].subgridspec(1, n_benchmarks, wspace=0.30)
axes_bench = [fig.add_subplot(gs0[0, j]) for j in range(n_benchmarks)]

approach_keys   = ["bilateral_classical", "forced_unilateral", "ternary", "confidence_05"]
approach_colors = ["#1565C0", "#E65100", "#2E7D32", "#F9A825"]
approach_short  = ["Bilateral", "Forced Uni.", "Ternary", "Confidence"]

x = np.arange(n_models)
n_ap = len(approach_keys)
width = 0.18
offsets = np.linspace(-(n_ap-1)/2, (n_ap-1)/2, n_ap) * width

for j, bench in enumerate(BENCHMARK_ORDER):
    ax = axes_bench[j]
    for ai, (ak, ac) in enumerate(zip(approach_keys, approach_colors)):
        vals = [get(mid, bench, ak) for mid in model_ids]
        ax.bar(x + offsets[ai], vals, width, color=ac, alpha=0.85,
               label=approach_short[ai] if j == 0 else "_nolegend_")
    ax.set_title(BENCHMARK_LABELS[bench], fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_short, rotation=30, ha="right", fontsize=7.5)
    ax.set_ylim(0.3, 1.05)
    ax.set_ylabel("F1-macro" if j == 0 else "")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.7)
    ax.yaxis.set_tick_params(labelsize=8)

# Legend for panel 0
legend_patches = [
    mpatches.Patch(color=ac, alpha=0.85, label=al)
    for ac, al in zip(approach_colors, approach_short)
]
axes_bench[0].legend(
    handles=legend_patches, loc="lower left", fontsize=8,
    ncol=2, frameon=True,
)

axes_bench[0].set_title(
    f"Panel A — F1-macro by Benchmark and Approach  "
    f"(N=250, bilateral N=3 majority vote, evidence-based ternary)\n{BENCHMARK_LABELS[BENCHMARK_ORDER[0]]}",
    fontsize=10, fontweight="bold",
)

# ── Panel 1: delta heatmaps ────────────────────────────────────────────────────
gs1 = gs[1].subgridspec(1, 2, wspace=0.35)
ax_h1 = fig.add_subplot(gs1[0, 0])
ax_h2 = fig.add_subplot(gs1[0, 1])

# Build matrices: rows = models, cols = benchmarks
delta_fu   = np.full((n_models, n_benchmarks), np.nan)
delta_tern = np.full((n_models, n_benchmarks), np.nan)

for i, mid in enumerate(model_ids):
    for j, bench in enumerate(BENCHMARK_ORDER):
        bil  = get(mid, bench, "bilateral_classical")
        fu   = get(mid, bench, "forced_unilateral")
        tern = get(mid, bench, "ternary")
        delta_fu[i, j]   = bil - fu
        delta_tern[i, j] = bil - tern

# Diverging colormap centred at 0
cmap = LinearSegmentedColormap.from_list(
    "rdgn", ["#C62828", "#FFEBEE", "#F5F5F5", "#E8F5E9", "#1B5E20"], N=256
)
vabs = max(abs(np.nanmax(delta_fu)), abs(np.nanmin(delta_fu)),
           abs(np.nanmax(delta_tern)), abs(np.nanmin(delta_tern)))

for ax, mat, title in [
    (ax_h1, delta_fu,   "Panel B — Bilateral \u2212 Forced Unilateral F1"),
    (ax_h2, delta_tern, "Panel C — Bilateral \u2212 Ternary F1"),
]:
    im = ax.imshow(mat, cmap=cmap, vmin=-vabs, vmax=vabs, aspect="auto")
    ax.set_xticks(range(n_benchmarks))
    ax.set_xticklabels(bench_labels, fontsize=8.5)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(model_short, fontsize=8.5)
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format="%.2f")
    # Annotate cells
    for i in range(n_models):
        for j in range(n_benchmarks):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=8, color="black" if abs(v) < vabs * 0.6 else "white",
                        fontweight="bold" if abs(v) > 0.05 else "normal")

# ── Panel 2: average-by-model bar chart ───────────────────────────────────────
ax_avg = fig.add_subplot(gs[2])

avg_bil  = []
avg_fu   = []
avg_tern = []
avg_conf = []

for mid in model_ids:
    vals_b = [get(mid, b, "bilateral_classical")    for b in BENCHMARK_ORDER]
    vals_f = [get(mid, b, "forced_unilateral")       for b in BENCHMARK_ORDER]
    vals_t = [get(mid, b, "ternary")                 for b in BENCHMARK_ORDER]
    vals_c = [get(mid, b, "confidence_05")           for b in BENCHMARK_ORDER]
    avg_bil.append(np.nanmean(vals_b))
    avg_fu.append(np.nanmean(vals_f))
    avg_tern.append(np.nanmean(vals_t))
    avg_conf.append(np.nanmean(vals_c))

x4 = np.arange(n_models)
w4 = 0.18
off4 = np.array([-1.5, -0.5, 0.5, 1.5]) * w4

for vals, color, label in [
    (avg_bil,  "#1565C0", "Bilateral (classical)"),
    (avg_fu,   "#E65100", "Forced unilateral"),
    (avg_tern, "#2E7D32", "Ternary (evidence)"),
    (avg_conf, "#F9A825", "Confidence @0.5"),
]:
    ax_avg.bar(x4 + off4[0], vals, w4, color=color, alpha=0.85, label=label)
    off4 = np.roll(off4, -1)

ax_avg.set_xticks(x4)
ax_avg.set_xticklabels(model_short, fontsize=9)
ax_avg.set_ylim(0.5, 1.0)
ax_avg.set_ylabel("Mean F1-macro (4 benchmarks)")
ax_avg.set_title(
    "Panel D — Mean F1-macro Averaged Across All Benchmarks",
    fontsize=10, fontweight="bold",
)
ax_avg.spines["top"].set_visible(False)
ax_avg.spines["right"].set_visible(False)
ax_avg.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.7)
ax_avg.legend(loc="lower right", fontsize=8.5, ncol=2, frameon=True)

# Overall mean annotation
overall_bil  = np.nanmean(avg_bil)
overall_fu   = np.nanmean(avg_fu)
overall_tern = np.nanmean(avg_tern)
ax_avg.axhline(overall_bil,  color="#1565C0", linewidth=1.2, linestyle="--", alpha=0.6)
ax_avg.axhline(overall_fu,   color="#E65100", linewidth=1.2, linestyle="--", alpha=0.6)
ax_avg.axhline(overall_tern, color="#2E7D32", linewidth=1.2, linestyle="--", alpha=0.6)
ax_avg.text(n_models - 0.05, overall_bil  + 0.004, f"Bil={overall_bil:.3f}",
            ha="right", va="bottom", fontsize=8, color="#1565C0")
ax_avg.text(n_models - 0.05, overall_fu   - 0.012, f"FU={overall_fu:.3f}",
            ha="right", va="bottom", fontsize=8, color="#E65100")
ax_avg.text(n_models - 0.05, overall_tern + 0.004, f"Tern={overall_tern:.3f}",
            ha="right", va="top", fontsize=8, color="#2E7D32")

# ── Supertitle ─────────────────────────────────────────────────────────────────
fig.suptitle(
    "BBL Full Comparison: Bilateral vs. Unilateral Epistemic Approaches\n"
    "6 Models \u00d7 4 Benchmarks, N=250, bilateral N=3 majority vote",
    fontsize=12, fontweight="bold", y=1.00,
)

out_pdf = os.path.join(BASE, "full_comparison_visualization.pdf")
out_png = out_pdf.replace(".pdf", ".png")
plt.savefig(out_pdf, bbox_inches="tight", dpi=150)
plt.savefig(out_png, bbox_inches="tight", dpi=150)
logger.info("Saved: %s", out_pdf)
logger.info("Saved: %s", out_png)
