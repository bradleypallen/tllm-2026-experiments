"""
Generate F-3': abstention under the four-option prompt, by original bilateral value.

The replacement for the paper's Figure 3 (fig:ternary-abstention) from the November 2026
revision data (E2 and E2-0; spec §7, table F-3'). For each bilateral value measured in the
2026-03 run, the bars are the share of valid E2 responses that are UNCERTAIN and the share
that are CONFLICTING (the option the ternary prompt lacked), and, for the three values the
E2-0 baseline subset covers, the share of same-day three-option (E2-0) responses that are
UNCERTAIN. Values are grouped as in Figure 3 (the four corner cells plus the pooled e-bearing
"middle" cells) so the two figures read side by side.

Computed from the raw E2 / E2-0 records with the frozen single-token parser
(evaluators/revision_2026_11_prompts.py, parse_single); PARSE_FAIL and API_ERROR responses
are excluded from denominators and their counts logged. The four named cells are checked
against results/derived/2026_11/T-E2_four_option.json (F3prime_by_original_value) so the
figure cannot drift from the table.

Style follows make_tv_diagnostic_figure.py (DejaVu Sans, dotted grid, the shaded
ignorance/inconsistency band, no in-figure title) with two series hues validated for
colour-vision deficiency; the same-day E2-0 bar is the UNCERTAIN hue with a hatch texture
rather than a third hue.

Data   : results/2026_11/e2/, results/2026_11/e2-0/ (five models; Opus 4.1 retired)
Outputs: results/derived/2026_11/f3prime_four_option.pdf (+ .png; copy into bbl_paper/bbl_revisions/)
         results/derived/2026_11/f3prime_four_option_audit.json (plotted numbers, for trace)

Run:  python3 analysis/make_f3prime_figure.py
"""
from __future__ import annotations
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "evaluators"))
import revision_2026_11_prompts as P  # noqa: E402  frozen parser

logging.basicConfig(level=logging.INFO, format="%(asctime)s [f3prime] %(levelname)s %(message)s")
log = logging.getLogger(__name__)

E2_DIR, E20_DIR = REPO / "results/2026_11/e2", REPO / "results/2026_11/e2-0"
TABLE = REPO / "results/derived/2026_11/T-E2_four_option.json"
OUT = REPO / "results/derived/2026_11/f3prime_four_option"

# ── style (groups and labels as in make_tv_diagnostic_figure.py) ─────────────
TV_GROUPS = ["<t,f>", "<f,t>", "<f,f>", "<t,t>", "middle"]
TV_LABELS = {
    "<t,f>": "⟨t,f⟩\nTrue", "<f,t>": "⟨f,t⟩\nFalse", "<f,f>": "⟨f,f⟩\nIgnorance",
    "<t,t>": "⟨t,t⟩\nInconsistency", "middle": "⟨u,e⟩ or\n⟨e,v⟩",
}
# Two categorical hues (validated: CVD ΔE and normal-vision floor pass); same-day = hatch texture.
C_UNCERTAIN, C_CONFLICTING, INK = "#1565C0", "#E65100", "#333333"
SERIES = [  # fixed order
    ("E2 UNCERTAIN", "uncertain"),
    ("E2 CONFLICTING", "conflicting"),
    ("E2-0 UNCERTAIN (same day, three-option)", "e2-0_uncertain"),
]


def tv_group(tv: str) -> str:
    return tv if tv in ("<t,f>", "<f,t>", "<f,f>", "<t,t>") else "middle"


def load(d: Path, mode: str) -> list[dict]:
    recs = []
    for p in sorted(d.glob(f"{mode}_*.jsonl")):
        recs += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return recs


def pred(rec: dict, legal: tuple[str, ...]) -> str:
    e = rec["elicitations"][0]
    return P.API_ERROR if e.get("error") else P.parse_single(e["raw"] or "", legal)


def main() -> None:
    e2, e20 = load(E2_DIR, "e2"), load(E20_DIR, "e2-0")
    e20_keys = {(r["model"], r["assertion_id"]) for r in e20}
    log.info("E2 records %d, E2-0 records %d", len(e2), len(e20))

    c = defaultdict(lambda: {"n": 0, "valid": 0, "uncertain": 0, "conflicting": 0, "fail": 0,
                             "e2-0_n": 0, "e2-0_valid": 0, "e2-0_uncertain": 0, "e2-0_fail": 0})
    per_value = defaultdict(lambda: {"valid": 0, "uncertain": 0, "conflicting": 0})
    for r in e2:
        g, v = tv_group(r["orig_bilateral_tv"]), r["orig_bilateral_tv"]
        q = pred(r, P.FOUR_OPTION_TOKENS)
        c[g]["n"] += 1
        if q in (P.PARSE_FAIL, P.API_ERROR):
            c[g]["fail"] += 1; continue
        c[g]["valid"] += 1; per_value[v]["valid"] += 1
        c[g]["uncertain"] += q == "UNCERTAIN"; per_value[v]["uncertain"] += q == "UNCERTAIN"
        c[g]["conflicting"] += q == "CONFLICTING"; per_value[v]["conflicting"] += q == "CONFLICTING"
    for r in e20:
        g = tv_group(r["orig_bilateral_tv"])
        q = pred(r, P.TERNARY_TOKENS)
        c[g]["e2-0_n"] += 1
        if q in (P.PARSE_FAIL, P.API_ERROR):
            c[g]["e2-0_fail"] += 1; continue
        c[g]["e2-0_valid"] += 1; c[g]["e2-0_uncertain"] += q == "UNCERTAIN"

    shares = {}
    for g in TV_GROUPS:
        s = c[g]
        shares[g] = {
            "uncertain": 100 * s["uncertain"] / s["valid"] if s["valid"] else None,
            "conflicting": 100 * s["conflicting"] / s["valid"] if s["valid"] else None,
            "e2-0_uncertain": 100 * s["e2-0_uncertain"] / s["e2-0_valid"] if s["e2-0_valid"] else None,
        }
        log.info("%-8s n=%4d valid=%4d fail=%3d  UNCERTAIN=%5.1f%%  CONFLICTING=%4.1f%%  | E2-0 valid=%4d UNCERTAIN=%s",
                 g, s["n"], s["valid"], s["fail"], shares[g]["uncertain"] or 0, shares[g]["conflicting"] or 0,
                 s["e2-0_valid"], f"{shares[g]['e2-0_uncertain']:.1f}%" if shares[g]["e2-0_uncertain"] is not None else "n/a")

    # ── consistency with the derived table (the four corner values, per-value shares) ──
    if TABLE.exists():
        t = json.loads(TABLE.read_text())["F3prime_by_original_value"]
        for v in ("<t,f>", "<f,t>", "<f,f>", "<t,t>"):
            pv = per_value[v]
            mine = (round(pv["uncertain"] / pv["valid"], 4), round(pv["conflicting"] / pv["valid"], 4), pv["valid"])
            theirs = (t[v]["uncertain"], t[v]["conflicting"], t[v]["n_valid"])
            if mine != theirs:
                raise SystemExit(f"figure/table mismatch for {v}: figure {mine} vs T-E2 {theirs}")
        log.info("consistent with T-E2_four_option.json on the four corner values")

    # ── render ────────────────────────────────────────────────────────────────
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"]})
    x = np.arange(len(TV_GROUPS))
    w = 0.26
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    fig.patch.set_facecolor("white")

    style = {"uncertain": dict(color=C_UNCERTAIN),
             "conflicting": dict(color=C_CONFLICTING),
             "e2-0_uncertain": dict(color="white", edgecolor=C_UNCERTAIN, hatch="////", linewidth=1.0)}
    for j, (label, key) in enumerate(SERIES):
        vals = [shares[g][key] for g in TV_GROUPS]
        xs = x + (j - 1) * w
        for xi, v, g in zip(xs, vals, TV_GROUPS):
            if v is None:
                if key == "e2-0_uncertain":
                    ax.text(xi, 1.0, "–", ha="center", va="bottom", fontsize=8, color="#888888")
                continue
            kw = dict(style[key])
            if key != "e2-0_uncertain":
                kw.update(edgecolor="white", linewidth=0.6)
            ax.bar(xi, v, width=w, zorder=3, **kw)
            ax.text(xi, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=7.5, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels([TV_LABELS[g] for g in TV_GROUPS], fontsize=8, linespacing=1.3)
    ax.set_ylabel("% of valid responses", fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.6, len(TV_GROUPS) - 0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ff, tt = TV_GROUPS.index("<f,f>"), TV_GROUPS.index("<t,t>")
    ax.axvspan(ff - 0.45, tt + 0.45, alpha=0.06, color="#333", zorder=1)

    handles = [Patch(facecolor=C_UNCERTAIN, label=SERIES[0][0]),
               Patch(facecolor=C_CONFLICTING, label=SERIES[1][0]),
               Patch(facecolor="white", edgecolor=C_UNCERTAIN, hatch="////", label=SERIES[2][0])]
    ax.legend(handles=handles, fontsize=7.5, frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", dpi=150)
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=150)
    log.info("wrote %s.pdf and .png", OUT)

    audit = {"description": __doc__.strip().split("\n")[0], "groups": TV_GROUPS,
             "counts": {g: dict(c[g]) for g in TV_GROUPS},
             "shares_pct": {g: {k: (round(v, 2) if v is not None else None) for k, v in shares[g].items()} for g in TV_GROUPS}}
    OUT.with_name(OUT.name + "_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    log.info("wrote %s_audit.json", OUT)


if __name__ == "__main__":
    main()
