"""
Generate the ternary-abstention-by-bilateral-value diagnostic (the 65/3 figure).

Self-contained, in-repo reproduction of the bar chart formerly produced by
tllm-2026-experiments/analysis/visualize_tv_diagnostic.py (whose BASE data dir is
now empty). Style is copied verbatim from that script -- semantic colours, the
black-edged highlight on the two carry cells, the shaded ignorance/inconsistency
band, dotted gridlines, DejaVu Sans, dpi 150 -- with ONE change: no in-figure
title (the description lives in the LaTeX caption).

For each bilateral truth-value bucket, the bar is the percentage of assertions
carrying that value on which the unilateral-ternary baseline answers ABSTAIN.
This figure is below the decision layer (it buckets raw <u,v> and counts ternary's
own ABSTAIN), so it is invariant under the commit/abstain decision rule.

Data: results/2026_03/proper/, canonical six models.
Output: results/derived/2026_03/tv_diagnostic.pdf (copy into the paper repo for LaTeX).

Run:  python3 analysis/make_tv_diagnostic_figure.py
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [tv-diagnostic] %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "results/2026_03/proper"
OUT_PDF = REPO / "results/derived/2026_03/tv_diagnostic.pdf"

MODELS = [
    "claude-opus-4-1-20250805", "meta-llama_llama-4-maverick",
    "meta-llama_llama-4-scout", "google_gemini-2.5-flash",
    "deepseek_deepseek-chat", "qwen_qwen-2.5-72b-instruct",
]
BENCHES = ["truthfulqa", "simpleqa", "mmlu-pro", "factscore"]

# ── style (verbatim from visualize_tv_diagnostic.py) ──────────────────────────
TV_GROUPS = ["<t,f>", "<f,t>", "<f,f>", "<t,t>", "middle"]
TV_LABELS = {
    "<t,f>":  "⟨t,f⟩\nTrue",
    "<f,t>":  "⟨f,t⟩\nFalse",
    "<f,f>":  "⟨f,f⟩\nIgnorance",
    "<t,t>":  "⟨t,t⟩\nInconsistency",
    "middle": "⟨u,e⟩ or\n⟨e,v⟩",
}
TV_COLORS = {
    "<t,f>":  "#1565C0", "<f,t>":  "#C62828", "<f,f>":  "#757575",
    "<t,t>":  "#E65100", "middle": "#5C35A8",
}


def tv_group(tv):
    return tv if tv in ("<t,f>", "<f,t>", "<f,f>", "<t,t>") else "middle"


def main():
    stats = defaultdict(lambda: {"n": 0, "abstain": 0})
    for mk in MODELS:
        for b in BENCHES:
            p = DATA_DIR / f"{b}_{mk}_n3_proper_results.json"
            if not p.exists():
                log.warning("missing %s", p.name)
                continue
            for r in json.loads(p.read_text()).get("detailed_results", []):
                tv, tern = r.get("bilateral_tv"), r.get("ternary_prediction")
                if tv is None or tern is None:
                    continue
                g = tv_group(tv)
                stats[g]["n"] += 1
                if tern == "ABSTAIN":
                    stats[g]["abstain"] += 1

    tern_ab = [100 * stats[g]["abstain"] / stats[g]["n"] if stats[g]["n"] else 0
               for g in TV_GROUPS]
    for g, v in zip(TV_GROUPS, tern_ab):
        log.info("%-8s ternary-abstain = %5.1f%%  (n=%d)", g, v, stats[g]["n"])

    # ── render ────────────────────────────────────────────────────────────────
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["DejaVu Sans"]})
    x = np.arange(len(TV_GROUPS))
    colors = [TV_COLORS[g] for g in TV_GROUPS]

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.3))
    fig.patch.set_facecolor("white")

    ax.bar(x, tern_ab, width=0.55, color=colors, edgecolor="white",
           linewidth=0.6, zorder=3)
    for i, g in enumerate(TV_GROUPS):           # black-edge highlight on carry cells
        if g in ("<f,f>", "<t,t>"):
            ax.bar(i, tern_ab[i], width=0.55, color=TV_COLORS[g],
                   edgecolor="black", linewidth=1.8, zorder=4)
    for i, v in enumerate(tern_ab):             # value labels, coloured + bold
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color=TV_COLORS[TV_GROUPS[i]])

    ax.set_xticks(x)
    ax.set_xticklabels([TV_LABELS[g] for g in TV_GROUPS], fontsize=8,
                       linespacing=1.3, ha="center")
    ax.set_ylabel("% of assertions ternary abstains on", fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.6, len(TV_GROUPS) - 0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    ff, tt = TV_GROUPS.index("<f,f>"), TV_GROUPS.index("<t,t>")
    ax.axvspan(ff - 0.38, tt + 0.38, alpha=0.06, color="#333", zorder=1)

    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight", dpi=150)
    log.info("wrote %s", OUT_PDF)


if __name__ == "__main__":
    main()
