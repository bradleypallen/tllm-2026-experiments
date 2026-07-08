"""
Recompute every decision-rule-dependent number in the paper from the single
canonical rule in `decision_rule.py`. Pure re-derivation from cached <u,v> values
-- no model queries, no resampling: free, deterministic, drift-immune.

Produces:
  - Table 1 (bilateral macro-F1 vs ternary) under the Venn rule, with the
    singleton rule alongside for a before/after audit + LaTeX-ready rows.
  - Figure 2 coverage annotations (src / modal) under the Venn rule.
  - Invariance check on the Figure 4 headline (ternary-abstain-by-bilateral-bucket,
    the 65/3 asymmetry) -- which is below the decision layer and must not move.
  - bbl_revisions/decision_rule_audit.json with all of the above.

Data (canonical six models x four benchmarks, cached majority-voted <u,v>):
  - Table 1 + headline : results/2026_03/proper/*_n3_proper_results.json
  - coverage           : results/2026_03/modal/*_n3_modal_results.json
Output: results/derived/2026_03/decision_rule_audit.json

Run:  python3 analysis/recompute_decision_rule.py
"""
from __future__ import annotations
import glob
import json
import logging
import statistics
from pathlib import Path

from decision_rule import commit_bool, COMMIT_CELLS, ALL_NINE  # canonical rule

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [recompute] %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
PROPER = REPO / "results/2026_03/proper"
MODAL = REPO / "results/2026_03/modal"
OUT = REPO / "results/derived/2026_03/decision_rule_audit.json"

# (model_key as used in proper filenames, display name). Order = Table 1 order.
MODELS = [
    ("claude-opus-4-1-20250805", "Claude Opus 4.1"),
    ("meta-llama_llama-4-maverick", "Llama 4 Maverick"),
    ("meta-llama_llama-4-scout", "Llama 4 Scout"),
    ("google_gemini-2.5-flash", "Gemini 2.5 Flash"),
    ("deepseek_deepseek-chat", "DeepSeek-V3"),
    ("qwen_qwen-2.5-72b-instruct", "Qwen 2.5 72B"),
]
BENCHES = ["truthfulqa", "simpleqa", "mmlu-pro", "factscore"]
BENCH_LABEL = {
    "truthfulqa": "TruthfulQA", "simpleqa": "SimpleQA",
    "mmlu-pro": "MMLU-Pro", "factscore": "FACTScore",
}


# --- macro-F1 on covered, replicating proper_benchmark_evaluator.py:90 verbatim ----
def compute_metrics(predictions, ground_truth):
    tp = tn = fp = fn = abstained = 0
    for pred, gt in zip(predictions, ground_truth):
        if pred is None:
            abstained += 1
        elif pred and gt:
            tp += 1
        elif not pred and not gt:
            tn += 1
        elif pred and not gt:
            fp += 1
        else:
            fn += 1
    total = len(predictions)
    covered = total - abstained
    coverage = covered / total if total else 0.0

    def f1(a, b, c):  # tp, fp, fn  ->  positive-class F1
        pr = a / (a + b) if (a + b) else 0.0
        rc = a / (a + c) if (a + c) else 0.0
        return 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0

    f1_macro = (f1(tp, fp, fn) + f1(tn, fn, fp)) / 2
    return {"coverage": coverage, "f1_macro": f1_macro,
            "covered": covered, "total": total}


def rule_singleton(tv):
    """The current (pre-fix) bilateral-classical rule -- strict singleton. Kept
    only to reproduce today's Table 1 and quantify the before/after delta."""
    return True if tv == "<t,f>" else False if tv == "<f,t>" else None


def rule_ternary(rec):
    t = rec.get("ternary_prediction")
    return True if t == "TRUE" else False if t == "FALSE" else None


def venn_coverage_from_dist(dist, n):
    """Figure-2 coverage under the Venn rule, read off a 9-value distribution."""
    return sum(c for k, c in dist.items() if k in COMMIT_CELLS) / n if n else 0.0


# ---------------------------------------------------------------------------------
def recompute_table1():
    log.info("=== Table 1: bilateral macro-F1 (Venn) vs ternary, from proper_results ===")
    rows, d_singleton, d_venn = [], [], []
    log.info("%-10s %-17s %7s %7s %7s  %7s %7s  %6s %6s",
             "benchmark", "model", "F1_old", "F1venn", "F1tern",
             "cov_old", "covVenn", "dOld", "dVenn")
    for bench in BENCHES:
        for mk, md in MODELS:
            p = PROPER / f"{bench}_{mk}_n3_proper_results.json"
            if not p.exists():
                log.warning("MISSING %s", p.name)
                continue
            recs = [r for r in json.loads(p.read_text())["detailed_results"]
                    if r.get("bilateral_tv") and r.get("ground_truth") is not None]
            gts = [bool(r["ground_truth"]) for r in recs]
            tvs = [r["bilateral_tv"] for r in recs]
            m_old = compute_metrics([rule_singleton(t) for t in tvs], gts)
            m_venn = compute_metrics([commit_bool(t) for t in tvs], gts)
            m_tern = compute_metrics([rule_ternary(r) for r in recs], gts)
            do = m_old["f1_macro"] - m_tern["f1_macro"]
            dv = m_venn["f1_macro"] - m_tern["f1_macro"]
            d_singleton.append(abs(do))
            d_venn.append(abs(dv))
            rows.append({
                "benchmark": bench, "benchmark_label": BENCH_LABEL[bench],
                "model": md, "n": m_venn["total"],
                "f1_singleton": round(m_old["f1_macro"], 3),
                "f1_venn": round(m_venn["f1_macro"], 3),
                "f1_ternary": round(m_tern["f1_macro"], 3),
                "delta_venn": round(dv, 3),
                "cov_singleton": round(m_old["coverage"], 3),
                "cov_venn": round(m_venn["coverage"], 3),
            })
            log.info("%-10s %-17s %7.3f %7.3f %7.3f  %7.3f %7.3f  %+6.3f %+6.3f",
                     bench, md, m_old["f1_macro"], m_venn["f1_macro"],
                     m_tern["f1_macro"], m_old["coverage"], m_venn["coverage"], do, dv)
    summary = {
        "singleton": {"median": statistics.median(d_singleton),
                      "mean": statistics.mean(d_singleton), "max": max(d_singleton)},
        "venn": {"median": statistics.median(d_venn),
                 "mean": statistics.mean(d_venn), "max": max(d_venn)},
    }
    log.info("")
    log.info("|dF1 vs ternary|  singleton: median=%.3f mean=%.3f max=%.3f",
             *[summary["singleton"][k] for k in ("median", "mean", "max")])
    log.info("|dF1 vs ternary|  VENN:      median=%.3f mean=%.3f max=%.3f",
             *[summary["venn"][k] for k in ("median", "mean", "max")])
    return rows, summary


def latex_table1(rows):
    """Emit the body of tab:f1-percell (Bilateral=Venn, Ternary unchanged, Delta)."""
    log.info("")
    log.info("=== LaTeX rows for tab:f1-percell (paste into paper.tex) ===")
    by_b = {b: [r for r in rows if r["benchmark"] == b] for b in BENCHES}
    lines = []
    for bi, b in enumerate(BENCHES):
        for j, r in enumerate(by_b[b]):
            head = r["benchmark_label"] if j == 0 else ""
            d = r["delta_venn"]
            lines.append(f" {head} & {r['model']} & {r['f1_venn']:.3f} & "
                         f"{r['f1_ternary']:.3f} & {d:+.3f} \\\\")
        if bi < len(BENCHES) - 1:
            lines.append("\\midrule")
    for ln in lines:
        log.info(ln)
    return lines


def recompute_coverage():
    log.info("")
    log.info("=== Figure 2 coverage (Venn) from modal_results ===")
    canon_keys = {mk for mk, _ in MODELS}
    by_key = {}
    for path in glob.glob(str(MODAL / "*_n3_modal_results.json")):
        d = json.load(open(path))
        em = d["eval_model"]
        if not any(c in path or c == em.replace("/", "_") for c in canon_keys):
            continue
        ds = d["dataset"].replace("mmlupro", "mmlu-pro")
        by_key[(em.replace("/", "_"), ds)] = d
    out = []
    log.info("%-10s %-17s %8s %8s   %8s %8s", "benchmark", "model",
             "srcOld", "srcVenn", "modOld", "modVenn")
    for bench in BENCHES:
        for mk, md in MODELS:
            d = by_key.get((mk, bench))
            if d is None:
                log.warning("MISSING modal cell %s x %s", md, bench)
                continue
            n = d["total_samples"]
            sv = venn_coverage_from_dist(d["src_bilateral_distribution"], n)
            mv = venn_coverage_from_dist(d["modal_bilateral_distribution"], n)
            out.append({"benchmark": bench, "model": md,
                        "src_cov_singleton": d.get("src_coverage"),
                        "src_cov_venn": round(sv, 3),
                        "modal_cov_singleton": d.get("modal_coverage"),
                        "modal_cov_venn": round(mv, 3)})
            log.info("%-10s %-17s %8.3f %8.3f   %8.3f %8.3f", bench, md,
                     d.get("src_coverage", float("nan")), sv,
                     d.get("modal_coverage", float("nan")), mv)
    return out


def invariance_headline():
    """Figure 4 (65/3) lives below the decision layer: it buckets raw <u,v> and
    counts the ternary baseline's own ABSTAIN. Recompute it to show it is invariant
    under the decision-rule change."""
    log.info("")
    log.info("=== Invariance check: Figure 4 headline (ternary abstain by bucket) ===")

    def bucket(tv):
        return tv if tv in ("<t,f>", "<f,t>", "<f,f>", "<t,t>") else "middle"

    tot = {b: 0 for b in ("<t,f>", "<f,t>", "<f,f>", "<t,t>", "middle")}
    ab = dict(tot)
    for bench in BENCHES:
        for mk, _ in MODELS:
            p = PROPER / f"{bench}_{mk}_n3_proper_results.json"
            if not p.exists():
                continue
            for r in json.loads(p.read_text())["detailed_results"]:
                tv, tp = r.get("bilateral_tv"), r.get("ternary_prediction")
                if tv is None or tp is None:
                    continue
                b = bucket(tv)
                tot[b] += 1
                if tp == "ABSTAIN":
                    ab[b] += 1
    pct = {b: (100 * ab[b] / tot[b] if tot[b] else 0.0) for b in tot}
    for b in ("<t,f>", "<f,t>", "<f,f>", "<t,t>", "middle"):
        log.info("  %-8s ternary-abstain = %5.1f%%  (n=%d)", b, pct[b], tot[b])
    log.info("  -> headline asymmetry: <f,f>=%.1f%%  vs  <t,t>=%.1f%%  (rule-independent)",
             pct["<f,f>"], pct["<t,t>"])
    return pct


def main():
    rows, summary = recompute_table1()
    latex = latex_table1(rows)
    coverage = recompute_coverage()
    headline = invariance_headline()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "rule": "Venn: TRUE={<t,f>,<t,e>}, FALSE={<f,t>,<e,t>}, ABSTAIN=I({<t,t>})+neither",
        "commit_cells": sorted(COMMIT_CELLS),
        "all_nine": ALL_NINE,
        "table1_rows": rows,
        "table1_summary_abs_delta_vs_ternary": summary,
        "figure2_coverage": coverage,
        "figure4_headline_invariant_ternary_abstain_pct": headline,
    }, indent=2))
    log.info("")
    log.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
