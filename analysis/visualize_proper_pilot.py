#!/usr/bin/env python3
"""
Proper pilot visualization: comparison table + bilateral distribution bar chart.

Reads the three proper_results.json files (Opus 4.1, GPT-4.1, DeepSeek-V3)
and produces:
  - Table: Model × Approach showing Coverage / F1-macro
  - Stacked bar chart: bilateral truth-value distributions per model
"""

import json
import os
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("visualize_proper_pilot")

BASE = os.path.join(os.path.dirname(__file__), "results")

MODELS = [
    ("Claude Opus 4.1",
     "truthfulqa_claude-opus-4-1-20250805_n3_proper_results.json"),
    ("GPT-4.1",
     "truthfulqa_gpt-4.1-2025-04-14_n3_proper_results.json"),
    ("DeepSeek-V3",
     "truthfulqa_deepseek_deepseek-chat_n3_proper_results.json"),
]

# Approach display order and labels
APPROACHES = [
    ("bilateral_classical",      "Bilateral\n(classical)",      "#1565C0"),
    ("bilateral_paracomplete",   "Bilateral\n(paracomplete)",   "#1976D2"),
    ("bilateral_paraconsistent", "Bilateral\n(paraconsistent)", "#42A5F5"),
    ("forced_unilateral",        "Forced\nunilateral",          "#E65100"),
    ("ternary",                  "Ternary",                     "#F57C00"),
    ("confidence_05",            "Confidence\n(@0.5)",          "#FF8F00"),
]

TV_ORDER  = ["<t,f>", "<f,t>", "<f,f>", "<t,t>", "<e,t>", "<t,e>", "<e,f>", "<f,e>", "<e,e>"]
TV_LABELS = {
    "<t,f>": "\u27e8t,f\u27e9",
    "<f,t>": "\u27e8f,t\u27e9",
    "<f,f>": "\u27e8f,f\u27e9",
    "<t,t>": "\u27e8t,t\u27e9",
    "<e,t>": "\u27e8e,t\u27e9",
    "<t,e>": "\u27e8t,e\u27e9",
    "<e,f>": "\u27e8e,f\u27e9",
    "<f,e>": "\u27e8f,e\u27e9",
    "<e,e>": "\u27e8e,e\u27e9",
}
TV_COLORS = {
    "<t,f>": "#2196F3",
    "<f,t>": "#F44336",
    "<f,f>": "#9E9E9E",
    "<t,t>": "#FF9800",
    "<e,t>": "#CE93D8",
    "<t,e>": "#90CAF9",
    "<e,f>": "#FFCC80",
    "<f,e>": "#EF9A9A",
    "<e,e>": "#CFD8DC",
}

# ── Load data ─────────────────────────────────────────────────────────────────

data = {}
for label, fname in MODELS:
    path = os.path.join(BASE, fname)
    with open(path) as f:
        data[label] = json.load(f)
    logger.info("Loaded %s", fname)

N = data[list(data.keys())[0]]["total_samples"]
model_labels = [lbl for lbl, _ in MODELS]

# ── Figure layout ─────────────────────────────────────────────────────────────
# Row 0: performance table
# Row 1: grouped bar chart (Coverage / F1-macro by approach)
# Row 2: stacked bilateral distribution bars

fig = plt.figure(figsize=(16, 13))
fig.patch.set_facecolor("white")

gs = fig.add_gridspec(
    3, 1,
    height_ratios=[1.6, 2.0, 2.0],
    hspace=0.55,
)

# ── Panel 0: performance table ────────────────────────────────────────────────

ax_tbl = fig.add_subplot(gs[0])
ax_tbl.axis("off")

col_labels = ["Model"] + [short.replace("\n", " ") for _, short, _ in APPROACHES]
col_labels_display = ["Model"] + [short for _, short, _ in APPROACHES]

rows_cov = []
rows_f1  = []
for lbl in model_labels:
    appr = data[lbl]["approaches"]
    rows_cov.append([lbl] + [f"{appr[k]['coverage']:.3f}" for k, _, _ in APPROACHES])
    rows_f1.append( [lbl] + [f"{appr[k]['f1_macro']:.3f}"  for k, _, _ in APPROACHES])

# Interleave: Coverage row then F1 row per model
table_rows = []
row_meta   = []  # ("coverage"/"f1", model_idx)
for mi, lbl in enumerate(model_labels):
    cov_row = rows_cov[mi].copy()
    cov_row[0] = lbl + "\nCoverage"
    f1_row  = rows_f1[mi].copy()
    f1_row[0]  = lbl + "\nF1-macro"
    table_rows.append(cov_row)
    row_meta.append(("coverage", mi))
    table_rows.append(f1_row)
    row_meta.append(("f1", mi))

tbl = ax_tbl.table(
    cellText=table_rows,
    colLabels=col_labels_display,
    cellLoc="center",
    loc="center",
    bbox=[0, 0, 1, 1],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)

# Header row colours (approach-column tints matching bar colors)
header_bg = ["#1565C0"] + [c for _, _, c in APPROACHES]
for j, bg in enumerate(header_bg):
    cell = tbl[0, j]
    cell.set_facecolor(bg)
    cell.set_text_props(color="white", fontweight="bold")
    cell.set_height(0.26)

# Data rows
model_row_bgs = [["#E3F2FD", "#BBDEFB"], ["#FFF9C4", "#FFF176"], ["#F1F8E9", "#DCEDC8"]]
for ri, (metric_type, mi) in enumerate(row_meta):
    bg = model_row_bgs[mi][0 if metric_type == "coverage" else 1]
    for j in range(len(col_labels_display)):
        cell = tbl[ri + 1, j]
        cell.set_facecolor(bg)
        cell.set_height(0.19)
        if j == 0:
            cell.set_text_props(fontweight="bold", fontsize=8)

ax_tbl.set_title(
    f"Table 2. TruthfulQA Proper Pilot — BBL vs. Alternative Epistemic Approaches  "
    f"($N={N}$, $s_0$ = source question)",
    fontsize=10.5, fontweight="bold", pad=6, loc="left",
)

# ── Panel 1: grouped bar chart — Coverage + F1 side by side ──────────────────

ax_bars = fig.add_subplot(gs[1])

n_approaches = len(APPROACHES)
n_models = len(model_labels)
x = np.arange(n_approaches)
width = 0.22
offsets = np.array([-1, 0, 1]) * width

model_marker_colors = ["#1565C0", "#E65100", "#2E7D32"]

for mi, (lbl, mk_color) in enumerate(zip(model_labels, model_marker_colors)):
    appr = data[lbl]["approaches"]

    # Coverage bars (lighter shade)
    cov_vals = [appr[k]["coverage"] for k, _, _ in APPROACHES]
    f1_vals  = [appr[k]["f1_macro"]  for k, _, _ in APPROACHES]

    ax_bars.bar(
        x + offsets[mi] - width * 0.05,
        cov_vals, width * 0.45,
        color=mk_color, alpha=0.35,
        label=f"{lbl} Cov" if mi == 0 else "_nolegend_",
    )
    ax_bars.bar(
        x + offsets[mi] + width * 0.05,
        f1_vals, width * 0.45,
        color=mk_color, alpha=0.85,
        label=f"{lbl} F1" if mi == 0 else "_nolegend_",
    )

# Approach x-tick labels
ax_bars.set_xticks(x)
ax_bars.set_xticklabels(
    [short for _, short, _ in APPROACHES],
    fontsize=8.5, multialignment="center",
)
ax_bars.set_ylim(0, 1.12)
ax_bars.set_ylabel("Score")
ax_bars.set_title(
    "Coverage (light) and F1-macro (solid) by Approach and Model",
    fontsize=10, fontweight="bold",
)
ax_bars.axvline(2.5, color="#BDBDBD", linewidth=1.2, linestyle="--")
ax_bars.text(1.0, 1.07, "Bilateral", ha="center", fontsize=9,
             color="#1565C0", fontweight="bold")
ax_bars.text(4.0, 1.07, "Unilateral / Confidence", ha="center", fontsize=9,
             color="#E65100", fontweight="bold")
ax_bars.spines["top"].set_visible(False)
ax_bars.spines["right"].set_visible(False)
ax_bars.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.7)

# Custom legend
legend_patches = []
for lbl, mk_color in zip(model_labels, model_marker_colors):
    legend_patches.append(mpatches.Patch(color=mk_color, alpha=0.85, label=lbl))
legend_patches.append(mpatches.Patch(color="#999999", alpha=0.35, label="Coverage"))
legend_patches.append(mpatches.Patch(color="#999999", alpha=0.85, label="F1-macro"))
ax_bars.legend(
    handles=legend_patches, loc="lower right", fontsize=8,
    ncol=2, frameon=True,
)

# ── Panel 2: stacked bilateral distribution per model ────────────────────────

ax_dist = fig.add_subplot(gs[2])

present = set()
for lbl in model_labels:
    present |= set(data[lbl]["bilateral_distribution"].keys())
tv_cols = [tv for tv in TV_ORDER if tv in present]

xm = np.arange(n_models)
bottom = np.zeros(n_models)
dist_patches = []

for tv in tv_cols:
    counts = np.array([
        data[lbl]["bilateral_distribution"].get(tv, 0)
        for lbl in model_labels
    ], dtype=float)
    pcts = counts / N * 100
    ax_dist.bar(xm, pcts, 0.5, bottom=bottom,
                color=TV_COLORS[tv], edgecolor="white", linewidth=0.5)
    bottom += pcts
    dist_patches.append(mpatches.Patch(color=TV_COLORS[tv], label=TV_LABELS[tv]))

ax_dist.set_xticks(xm)
ax_dist.set_xticklabels(model_labels, rotation=15, ha="right", fontsize=9)
ax_dist.set_ylim(0, 115)
ax_dist.set_ylabel("% of assertions")
ax_dist.set_title(
    "Bilateral Truth-Value Distribution per Model  (VM(s\u2080, p))",
    fontsize=10, fontweight="bold",
)
ax_dist.spines["top"].set_visible(False)
ax_dist.spines["right"].set_visible(False)
ax_dist.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.7)

# Annotate totals
for i, lbl in enumerate(model_labels):
    ax_dist.text(i, bottom[i] + 2, f"N={N}", ha="center", fontsize=8, color="#555555")

ax_dist.legend(
    handles=dist_patches,
    loc="upper right",
    ncol=len(dist_patches),
    fontsize=8.5,
    title="Bilateral truth value",
    title_fontsize=8.5,
    frameon=True,
)

# ── Supertitle + save ─────────────────────────────────────────────────────────

fig.suptitle(
    f"BBL Proper Pilot: Bilateral vs. Unilateral Epistemic Approaches (TruthfulQA, N={N})",
    fontsize=12, fontweight="bold", y=0.99,
)

out_pdf = os.path.join(BASE, "proper_pilot_visualization.pdf")
out_png = out_pdf.replace(".pdf", ".png")
plt.savefig(out_pdf, bbox_inches="tight", dpi=150)
plt.savefig(out_png, bbox_inches="tight", dpi=150)
logger.info("Saved: %s", out_pdf)
logger.info("Saved: %s", out_png)
