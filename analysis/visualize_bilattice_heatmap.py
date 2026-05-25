#!/usr/bin/env python3
"""
Bilattice heatmap grid: bilateral TV distributions on the 3×3 NINE bilattice.

Layout: 4 benchmarks (rows) × 6 models (cols).
Each cell is a 3×3 heatmap showing the fraction of assertions at each ⟨u,v⟩
position in NINE = ⟨V₃×V₃, ≤_t, ≤_k⟩.

Grid axes:
  x = u (verifiability): f | e | t   (left → right)
  y = v (refutability):  t | e | f   (top → bottom)

Corner semantics:
  bottom-right ⟨t,f⟩ = TRUE          top-left  ⟨f,t⟩ = FALSE
  bottom-left  ⟨f,f⟩ = IGNORANCE     top-right ⟨t,t⟩ = CONTRADICTION
"""

import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("visualize_bilattice_heatmap")

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

# ── TV → grid position ─────────────────────────────────────────────────────────
# col = u (verifiability): f→0, e→1, t→2
# row = v (refutability):  t→0, e→1, f→2  (inverted so ⟨t,f⟩ = TRUE is bottom-right)
U_IDX = {'f': 0, 'e': 1, 't': 2}
V_IDX = {'t': 0, 'e': 1, 'f': 2}

def tv_to_rc(tv):
    return V_IDX[tv[3]], U_IDX[tv[1]]   # (row, col)

# Axis tick labels
X_TICKS = ["f", "e", "t"]           # u: verifiability
Y_TICKS = ["t", "e", "f"]           # v: refutability (top→bottom)

# Key corner annotations: (row, col, label, edge-colour)
CORNERS = [
    (2, 2, "TRUE\n⟨t,f⟩",   "#1565C0"),   # bottom-right
    (0, 0, "FALSE\n⟨f,t⟩",  "#C62828"),   # top-left
    (2, 0, "IGN\n⟨f,f⟩",    "#757575"),   # bottom-left
    (0, 2, "CONT\n⟨t,t⟩",   "#E65100"),   # top-right
]

# Colourmap: white → dark indigo (neutral — semantics come from position)
CMAP = LinearSegmentedColormap.from_list("wh_indigo",
    ["#FFFFFF", "#E8EAF6", "#3949AB", "#1A237E"], N=256)

# ── Load proper benchmark results ─────────────────────────────────────────────
data = {}
for path in glob.glob(os.path.join(BASE, "*_n3_proper_results.json")):
    d = json.load(open(path))
    data[(d["model"], d["dataset"])] = d
    logger.info("Loaded %s × %s", d["model"], d["dataset"])

# ── Build 3×3 matrix ──────────────────────────────────────────────────────────
def build_matrix(model_id, bench):
    key = (model_id, bench)
    if key not in data:
        return None
    d   = data[key]
    n   = d["total_samples"]
    mat = np.zeros((3, 3))
    for tv, cnt in d["bilateral_distribution"].items():
        r, c = tv_to_rc(tv)
        mat[r, c] = cnt / n * 100
    return mat

# ── Figure ─────────────────────────────────────────────────────────────────────
n_rows, n_cols = len(BENCHMARK_ORDER), len(MODEL_ORDER)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 13))
fig.patch.set_facecolor("white")

VMAX = 70   # saturate at 70 % so mid-range values are visible

for row, bench in enumerate(BENCHMARK_ORDER):
    for col, (mid, mlabel) in enumerate(MODEL_ORDER):
        ax   = axes[row, col]
        mat  = build_matrix(mid, bench)

        if mat is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, color="#aaa", fontsize=9)
            ax.axis("off")
            continue

        # Heatmap
        ax.imshow(mat, cmap=CMAP, vmin=0, vmax=VMAX,
                  aspect="equal", interpolation="nearest")

        # Percentage annotations
        for r in range(3):
            for c in range(3):
                pct = mat[r, c]
                if pct >= 1.0:
                    fg = "white" if pct > VMAX * 0.45 else "#222"
                    ax.text(c, r, f"{pct:.0f}%",
                            ha="center", va="center",
                            fontsize=7, color=fg, fontweight="bold")

        # Coloured borders on the four semantic corners
        for (kr, kc, _, kclr) in CORNERS:
            rect = plt.Rectangle((kc - 0.5, kr - 0.5), 1, 1,
                                  linewidth=2.2, edgecolor=kclr,
                                  facecolor="none", zorder=4)
            ax.add_patch(rect)

        # Axes
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(X_TICKS, fontsize=7.5)
        ax.set_yticklabels(Y_TICKS, fontsize=7.5)
        ax.tick_params(length=0, pad=2)
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)

        if row == 0:
            ax.set_title(mlabel, fontsize=8.5, fontweight="bold", pad=5)
        if col == 0:
            ax.set_ylabel(BENCHMARK_LABELS[bench], fontsize=9,
                          fontweight="bold", labelpad=6)
        if row == n_rows - 1:
            ax.set_xlabel("u  (verif.)", fontsize=6.5, labelpad=2)
        if col == 0:
            # secondary y-label: "v (refut.)"
            ax.annotate("v  (refut.)", xy=(0, 0.5),
                        xycoords="axes fraction",
                        xytext=(-42, 0), textcoords="offset points",
                        fontsize=6.5, ha="center", va="center",
                        rotation=90, color="#555")

# ── Corner legend ──────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(edgecolor=kclr, facecolor="none",
                   linewidth=2, label=lbl.replace("\n", "  "))
    for (_, _, lbl, kclr) in CORNERS
]
fig.legend(
    handles=legend_patches,
    loc="lower center",
    ncol=4,
    fontsize=9,
    title="Semantic corners of NINE  |  x-axis = u (verifiability)  ·  y-axis = v (refutability)",
    title_fontsize=8.5,
    frameon=True,
    bbox_to_anchor=(0.5, -0.03),
)

# ── Colour scale bar ───────────────────────────────────────────────────────────
sm = plt.cm.ScalarMappable(cmap=CMAP,
                            norm=plt.Normalize(vmin=0, vmax=VMAX))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation="vertical",
                    fraction=0.015, pad=0.02, shrink=0.6)
cbar.set_label("% of assertions", fontsize=8)
cbar.ax.tick_params(labelsize=7)

fig.suptitle(
    "Bilateral Truth-Value Distributions on the Bilattice NINE  —  VM(s₀, p)\n"
    "Each cell: 3×3 heatmap over ⟨u,v⟩ ∈ V₃×V₃.  "
    "Colour intensity = % of N=250 assertions at that bilattice position.",
    fontsize=11, fontweight="bold", y=1.01,
)

plt.tight_layout(rect=[0, 0.05, 0.97, 1])

out_pdf = os.path.join(BASE, "bilattice_heatmap.pdf")
out_png = out_pdf.replace(".pdf", ".png")
plt.savefig(out_pdf, bbox_inches="tight", dpi=150)
plt.savefig(out_png, bbox_inches="tight", dpi=150)
logger.info("Saved: %s", out_pdf)
logger.info("Saved: %s", out_png)
