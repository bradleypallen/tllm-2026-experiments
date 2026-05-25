#!/usr/bin/env python3
"""
Diagnostic visualization: what does each bilateral TV group actually mean?

Shows that ⟨f,f⟩ (ignorance) and ⟨t,t⟩ (contradiction) are diagnostically
distinct epistemic states — both fall outside D∪D* under any bilateral policy,
but their ground-truth profiles and behaviour under unilateral evaluation differ
completely.

Output: results/tv_diagnostic.pdf / .png
"""

import json
import glob
import os
import logging
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("visualize_tv_diagnostic")

BASE = os.path.join(os.path.dirname(__file__), "results")

MODEL_SAFE = {
    "claude-opus-4-1-20250805",
    "meta-llama_llama-4-maverick",
    "meta-llama_llama-4-scout",
    "google_gemini-2.5-flash",
    "deepseek_deepseek-chat",
    "qwen_qwen-2.5-72b-instruct",
}

# ── TV grouping ───────────────────────────────────────────────────────────────
TV_GROUPS   = ["<t,f>", "<f,t>", "<f,f>", "<t,t>", "middle"]
TV_LABELS   = {
    "<t,f>":  "⟨t,f⟩\nTrue",
    "<f,t>":  "⟨f,t⟩\nFalse",
    "<f,f>":  "⟨f,f⟩\nIgnorance",
    "<t,t>":  "⟨t,t⟩\nInconsistency",
    "middle": "⟨u,e⟩ or\n⟨e,v⟩",
}
TV_COLORS   = {
    "<t,f>":  "#1565C0",
    "<f,t>":  "#C62828",
    "<f,f>":  "#757575",
    "<t,t>":  "#E65100",
    "middle": "#5C35A8",
}

def tv_group(tv):
    return tv if tv in ("<t,f>", "<f,t>", "<f,f>", "<t,t>") else "middle"

# ── Collect stats ─────────────────────────────────────────────────────────────
stats = defaultdict(lambda: {"n": 0, "gt_pos": 0, "uni_correct": 0,
                             "tern_abstain": 0, "tern_correct": 0})

for path in glob.glob(os.path.join(BASE, "*_n3_proper_results.json")):
    d = json.load(open(path))
    if d["model"].replace("/", "_") not in MODEL_SAFE:
        continue
    for r in d["detailed_results"]:
        g = tv_group(r["bilateral_tv"])
        stats[g]["n"]           += 1
        stats[g]["gt_pos"]      += int(r["ground_truth"])
        stats[g]["uni_correct"] += int(r["forced_unilateral_correct"])
        if r["ternary_prediction"] == "ABSTAIN":
            stats[g]["tern_abstain"] += 1
        elif r["ternary_correct"]:
            stats[g]["tern_correct"] += 1

logger.info("Loaded %d assertions across TV groups", sum(s["n"] for s in stats.values()))

# ── Derived metrics ───────────────────────────────────────────────────────────
groups  = TV_GROUPS
ns      = [stats[g]["n"]    for g in groups]
gt_pct  = [stats[g]["gt_pos"] / stats[g]["n"] * 100 if stats[g]["n"] else 0
           for g in groups]
uni_acc = [stats[g]["uni_correct"] / stats[g]["n"] * 100 if stats[g]["n"] else 0
           for g in groups]
tern_ab = [stats[g]["tern_abstain"] / stats[g]["n"] * 100 if stats[g]["n"] else 0
           for g in groups]

x = np.arange(len(groups))
BAR_W = 0.26

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(7, 4.6))
axes = [ax]
fig.patch.set_facecolor("white")

colors = [TV_COLORS[g] for g in groups]
xlabels = [TV_LABELS[g] for g in groups]

def make_bar_ax(ax, values, title, ylabel, ref_line=None, ref_label=None,
                highlight_groups=None):
    bars = ax.bar(x, values, width=0.55, color=colors, edgecolor="white",
                  linewidth=0.6, zorder=3)

    # highlight boxes around ⟨f,f⟩ and ⟨t,t⟩
    for i, g in enumerate(groups):
        if highlight_groups and g in highlight_groups:
            ax.bar(i, values[i], width=0.55,
                   color=TV_COLORS[g], edgecolor="black",
                   linewidth=1.8, zorder=4)

    if ref_line is not None:
        ax.axhline(ref_line, color="#555", linewidth=1.1,
                   linestyle="--", zorder=2, label=ref_label)
        if ref_label:
            ax.text(len(groups) - 0.5, ref_line + 1.5, ref_label,
                    fontsize=7, color="#555", va="bottom", ha="right")

    # value labels
    for i, v in enumerate(values):
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", va="bottom",
                fontsize=8, fontweight="bold",
                color=TV_COLORS[groups[i]])

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8, linespacing=1.3, ha="center")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)


# Single panel: Ternary abstain rate
make_bar_ax(axes[0], tern_ab,
            title="Ternary abstention rate by bilateral truth value",
            ylabel="% of assertions ternary abstains on",
            ref_line=None,
            highlight_groups={"<f,f>", "<t,t>"})

# ── Annotation arrow / callout connecting ⟨f,f⟩ and ⟨t,t⟩ ──────────────────
for ax in axes:
    ff_idx = groups.index("<f,f>")
    tt_idx = groups.index("<t,t>")
    ax.axvspan(ff_idx - 0.38, tt_idx + 0.38, alpha=0.06,
               color="#333", zorder=1, label="_nolegend_")

plt.tight_layout()

out_pdf = os.path.join(BASE, "tv_diagnostic.pdf")
out_png = out_pdf.replace(".pdf", ".png")
plt.savefig(out_pdf, bbox_inches="tight", dpi=150)
plt.savefig(out_png, bbox_inches="tight", dpi=150)
logger.info("Saved %s", out_pdf)
logger.info("Saved %s", out_png)
