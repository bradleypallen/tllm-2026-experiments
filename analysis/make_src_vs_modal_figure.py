#!/usr/bin/env python3
"""
Per-cell VM(s0,p) vs [[box p]] bilateral truth-value distributions, with coverage
annotations under the paper's commit/abstain decision rule (the Venn rule,
decision_rule.py).

NOTE: this figure (formerly Figure 2 / fig:boxp-distribution) was cut from the paper;
the script is retained as provenance, documenting how the full nine-value distributions
and their Venn-rule coverage were produced.

Faithful port of tllm-2026-experiments/analysis/visualize_src_vs_modal.py with three
changes: (1) data source is the canonical results dir (the analysis/results glob the
original used is now empty); (2) coverage is recomputed from the 9-value distributions
under COMMIT_CELLS so the figure matches the paper's rule by construction, rather than
reading the stale singleton src_coverage/modal_coverage fields; (3) canonical-6 filter
(the data dir also holds non-canonical gpt-4.1 cells, which are excluded).

Output: results/derived/2026_03/src_vs_modal_distributions.pdf (+ .png).
Provenance only --- no longer embedded in the paper.

Run:  python3 analysis/make_src_vs_modal_figure.py
"""
import json
import glob
import os
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from decision_rule import COMMIT_CELLS  # single source of truth for the decision rule

logging.basicConfig(level=logging.INFO, format="%(asctime)s [fig2] %(message)s")
log = logging.getLogger("make_src_vs_modal")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "results", "2026_03", "modal")
OUT_DIR = os.path.join(REPO, "results", "derived", "2026_03")

# Canonical six (slash-form eval_model, matching the modal files), Table-1 order.
MODEL_ORDER = [
    ("claude-opus-4-1-20250805",    "Opus 4.1"),
    ("meta-llama/llama-4-maverick", "Llama 4 Mav."),
    ("meta-llama/llama-4-scout",    "Llama 4 Scout"),
    ("google/gemini-2.5-flash",     "Gemini 2.5"),
    ("deepseek/deepseek-chat",      "DeepSeek-V3"),
    ("qwen/qwen-2.5-72b-instruct",  "Qwen 2.5 72B"),
]
CANON = {k for k, _ in MODEL_ORDER}
BENCHMARK_ORDER = ["truthfulqa", "simpleqa", "mmlu-pro", "factscore"]
BENCHMARK_LABELS = {"truthfulqa": "TruthfulQA", "simpleqa": "SimpleQA",
                    "mmlu-pro": "MMLU-Pro", "factscore": "FACTScore"}

TV_ORDER = ["<t,f>", "<f,t>", "<f,f>", "<t,t>",
            "<t,e>", "<e,f>", "<e,t>", "<f,e>", "<e,e>"]
TV_LABELS = {tv: f"⟨{tv[1]},{tv[3]}⟩" for tv in TV_ORDER}
TV_COLORS = {
    "<t,f>": "#1565C0", "<f,t>": "#C62828", "<f,f>": "#9E9E9E", "<t,t>": "#FF8F00",
    "<t,e>": "#90CAF9", "<e,f>": "#FFCC80", "<e,t>": "#CE93D8", "<f,e>": "#EF9A9A",
    "<e,e>": "#CFD8DC",
}


def venn_coverage(dist, n):
    """Fraction not abstaining under the decision rule = mass on the commit cells."""
    return sum(c for k, c in dist.items() if k in COMMIT_CELLS) / n if n else float("nan")


# ── Load canonical-6 data ────────────────────────────────────────────────────────
data = {}
for path in glob.glob(os.path.join(DATA_DIR, "*_n3_modal_results.json")):
    d = json.load(open(path))
    if d["eval_model"] not in CANON:
        continue
    data[(d["eval_model"], d["dataset"])] = d
log.info("Loaded %d canonical cells from %s", len(data), DATA_DIR)
if len(data) != 24:
    log.warning("expected 24 canonical cells, got %d", len(data))

# ── Figure (layout identical to the original) ─────────────────────────────────────
n_rows, n_cols = len(BENCHMARK_ORDER), len(MODEL_ORDER)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 13), sharey="row")
fig.patch.set_facecolor("white")

X_SRC, X_MODAL, BAR_W = -0.22, 0.22, 0.38


def stacked_bar(ax, dist, n, x, bar_w, label_threshold=7):
    bottom = 0.0
    for tv in [t for t in TV_ORDER if dist.get(t, 0) > 0]:
        pct = dist.get(tv, 0) / n * 100
        ax.bar(x, pct, bar_w, bottom=bottom, color=TV_COLORS[tv],
               edgecolor="white", linewidth=0.4)
        if pct >= label_threshold:
            ax.text(x, bottom + pct / 2, f"{pct:.0f}%", ha="center", va="center",
                    fontsize=6, color="white", fontweight="bold")
        bottom += pct


for row, bench in enumerate(BENCHMARK_ORDER):
    for col, (mid, mlabel) in enumerate(MODEL_ORDER):
        ax = axes[row, col]
        d = data.get((mid, bench))
        if d is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="#aaa")
            ax.set_xticks([])
            continue
        n = d["total_samples"]
        src_dist, modal_dist = d["src_bilateral_distribution"], d["modal_bilateral_distribution"]
        stacked_bar(ax, src_dist, n, X_SRC, BAR_W)
        stacked_bar(ax, modal_dist, n, X_MODAL, BAR_W)

        # Coverage annotations: Venn decision rule, recomputed from the distributions.
        src_cov, modal_cov = venn_coverage(src_dist, n), venn_coverage(modal_dist, n)
        ax.text(X_SRC, 103, f"{src_cov:.2f}", ha="center", va="bottom",
                fontsize=5.5, color="#555")
        ax.text(X_MODAL, 103, f"{modal_cov:.2f}", ha="center", va="bottom",
                fontsize=5.5, color="#333", fontweight="bold")

        ax.axvline(0, color="#ddd", linewidth=0.6, linestyle="--", zorder=0)
        ax.set_xlim(-0.55, 0.55)
        ax.set_ylim(0, 112)
        ax.set_xticks([X_SRC, X_MODAL])
        ax.set_xticklabels(["VM", "□p"], fontsize=6.5, color="#444")
        ax.tick_params(axis="x", length=0)
        for sp in ("top", "right", "bottom"):
            ax.spines[sp].set_visible(False)
        if row == 0:
            ax.set_title(mlabel, fontsize=8.5, fontweight="bold", pad=4)
        if col == 0:
            ax.set_ylabel(BENCHMARK_LABELS[bench], fontsize=9, fontweight="bold", labelpad=6)
            ax.yaxis.set_tick_params(labelsize=7)
        else:
            ax.set_yticklabels([])

# ── Legend + title ───────────────────────────────────────────────────────────────
present = set()
for d in data.values():
    present |= set(d["src_bilateral_distribution"]) | set(d["modal_bilateral_distribution"])
legend_patches = [mpatches.Patch(color=TV_COLORS[tv], label=TV_LABELS[tv])
                  for tv in TV_ORDER if tv in present]
fig.legend(handles=legend_patches, loc="lower center", ncol=len(legend_patches),
           fontsize=9, title="Bilateral truth value", title_fontsize=9,
           frameon=True, bbox_to_anchor=(0.5, -0.03))
fig.suptitle(
    "VM(s₀, p)  vs  [[□p]]  Bilateral TV Distributions\n"
    "Left bar = source situation only;  Right bar = modal necessity over s₀ + 2 paraphrases.  "
    "Numbers above bars = commit/abstain decision-rule coverage.",
    fontsize=11, fontweight="bold", y=1.01,
)
plt.tight_layout(rect=[0, 0.04, 1, 1])

os.makedirs(OUT_DIR, exist_ok=True)
out_pdf = os.path.join(OUT_DIR, "src_vs_modal_distributions.pdf")
plt.savefig(out_pdf, bbox_inches="tight", dpi=150)
plt.savefig(out_pdf.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
log.info("Saved %s", out_pdf)
